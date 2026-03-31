"""
Stage 1/2 optical flow training script for FullVideoClassifier.

Stage 1 (--stage 1, default):
  Trains FlowEncoder + classification head on flow maps only.
  Success: >65% -> proceed to Stage 2, 60-65% -> investigate, <60% -> debug Farneback.

Stage 2 (--stage 2):
  Full FlowVideoClassifier: backbone + FlowEncoder fine-tune.
  Load backbone from --pretrained-checkpoint (best_model.pt).
  Use --freeze-backbone to freeze LatentEncoder + PatchEncoder.
"""

import os
import gc
import glob
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from full_scale_classifier import (
    FullLatentEncoder, FullPatchEncoder, FullClassifier,
    FlowEncoder, FlowVideoClassifier, FlowStageOneModel
)
from dataset import FlowVideoDataset, ManifestFlowDataset, flow_collate_fn


def find_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pt')
    torch.save(state, path)
    # Keep only last 3
    old = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))[:-3]
    for f in old:
        os.remove(f)


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_train, stage):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch[0] is None:
                continue
            if stage == 1:
                _, flow_maps, labels, _ = batch
                flow_maps = flow_maps.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(flow_maps)
                    loss = criterion(outputs, labels)
            else:
                videos, flow_maps, labels, _ = batch
                videos = videos.to(device, non_blocking=True)
                flow_maps = flow_maps.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(videos, flow_maps)
                    loss = criterion(outputs, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            preds = torch.argmax(outputs, dim=1)
            scores = torch.softmax(outputs, dim=1)[:, 1]
            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores.detach().cpu().tolist())

            del outputs, loss, preds, scores
            gc.collect()

    n_batches = max(len(loader), 1)
    metrics = {
        'loss': total_loss / n_batches,
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    if len(set(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        metrics['auc'] = auc(fpr, tpr)
    return metrics, all_preds, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Optical flow branch training')
    parser.add_argument('--training', type=str, default='./data')
    parser.add_argument('--model-dir', type=str, default='./flow_model_output')
    parser.add_argument('--checkpoint-dir', type=str, default='./flow_checkpoints')
    parser.add_argument('--output-data-dir', type=str, default='./flow_output')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--pretrained-checkpoint', type=str, default=None,
                        help='Path to best_model.pt from sm_train_v2.py (Stage 2 backbone init)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze LatentEncoder + PatchEncoder (Stage 2)')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--target-size', type=int, default=512)
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument('--flow-h', type=int, default=64)
    parser.add_argument('--flow-w', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Cap dataset size (balanced AI/Real). None = use all.')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to flow_manifest.csv (overrides --training)')
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nStage {args.stage} training")

    if args.manifest:
        # ── Manifest-based dataset (pre-built CSV with train/val splits) ──
        print(f"Using manifest: {args.manifest}")
        train_dataset = ManifestFlowDataset(
            args.manifest, split='train',
            target_size=args.target_size, max_frames=args.max_frames,
            flow_h=args.flow_h, flow_w=args.flow_w,
        )
        test_dataset = ManifestFlowDataset(
            args.manifest, split='val',
            target_size=args.target_size, max_frames=args.max_frames,
            flow_h=args.flow_h, flow_w=args.flow_w,
        )
        print(f"Manifest splits: {len(train_dataset)} train / {len(test_dataset)} val")
    else:
        # ── Directory-based dataset (original behaviour) ──
        print(f"Scanning data at: {args.training}")

        full_dataset = FlowVideoDataset(
            args.training,
            target_size=args.target_size,
            max_frames=args.max_frames,
            flow_h=args.flow_h,
            flow_w=args.flow_w,
        )

        if len(full_dataset) == 0:
            raise RuntimeError(f"No labeled videos found in {args.training}")

        # Optional balanced subsample (--max-videos N -> N/2 AI + N/2 Real)
        if args.max_videos is not None and args.max_videos < len(full_dataset):
            ai_vids = [(p, l) for p, l in full_dataset.videos if l == 0]
            real_vids = [(p, l) for p, l in full_dataset.videos if l == 1]
            random.shuffle(ai_vids); random.shuffle(real_vids)
            half = args.max_videos // 2
            subset_paths = [p for p, _ in ai_vids[:half]] + [p for p, _ in real_vids[:half]]
            full_dataset = FlowVideoDataset(
                args.training,
                target_size=args.target_size,
                max_frames=args.max_frames,
                flow_h=args.flow_h,
                flow_w=args.flow_w,
                video_paths=subset_paths,
            )
            print(f"Subsampled to {len(full_dataset)} videos ({half} AI + {half} Real)")

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        split_path = os.path.join(args.model_dir, 'train_test_split.json')
        with open(split_path, 'w') as f:
            json.dump({
                'seed': args.seed,
                'total': len(full_dataset),
                'train_indices': train_dataset.indices,
                'test_indices': test_dataset.indices,
            }, f)
        print(f"Split saved: {split_path}  ({train_size} train / {test_size} test)")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=flow_collate_fn, num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=flow_collate_fn, num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    if args.stage == 1:
        model = FlowStageOneModel().to(device)
        print(f"Stage 1: FlowEncoder ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    else:
        flow_encoder = FlowEncoder()
        model = FlowVideoClassifier(
            FullLatentEncoder(), FullPatchEncoder(), flow_encoder, FullClassifier()
        ).to(device)

        if args.pretrained_checkpoint:
            print(f"Loading backbone from: {args.pretrained_checkpoint}")
            ckpt = torch.load(args.pretrained_checkpoint, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            # Load only backbone keys (latent_encoder, patch_encoder, classifier)
            model_state = model.state_dict()
            loaded = {k: v for k, v in state.items()
                      if k in model_state and not k.startswith('flow_encoder')}
            model_state.update(loaded)
            model.load_state_dict(model_state)
            print(f"Loaded {len(loaded)} backbone params")

        if args.freeze_backbone:
            for p in model.latent_encoder.parameters():
                p.requires_grad = False
            for p in model.patch_encoder.parameters():
                p.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Stage 2: backbone frozen, trainable params: {trainable/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, eps=1e-8, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    # -----------------------------------------------------------------------
    # Checkpoint recovery
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_accuracy = 0.0
    latest_ckpt = find_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best: {best_accuracy:.2f}%")
    else:
        print("No checkpoint -- training from scratch")

    # -----------------------------------------------------------------------
    # TensorBoard
    # -----------------------------------------------------------------------
    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join(args.output_data_dir, f'tb_{run_name}'))

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    early_stop_patience = 4
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  LR={optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, is_train=True, stage=args.stage
        )
        val_metrics, val_preds, val_labels = run_epoch(
            model, test_loader, criterion, optimizer, scaler, scheduler,
            device, is_train=False, stage=args.stage
        )

        elapsed = (time.time() - epoch_start) / 60
        print(f"Train -- loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.2f}%  f1: {train_metrics['f1']:.4f}")
        print(f"Val   -- loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.2f}%  f1: {val_metrics['f1']:.4f}  auc: {val_metrics.get('auc', 0):.4f}")
        print(f"Epoch time: {elapsed:.1f} min")

        for k, v in train_metrics.items():
            writer.add_scalar(f'Train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if len(set(val_labels)) == 2:
            cm = confusion_matrix(val_labels, val_preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['AI', 'Real'], yticklabels=['AI', 'Real'])
            ax.set_title(f'Epoch {epoch+1}')
            writer.add_figure('Confusion_Matrix', fig, epoch)
            plt.close(fig)

        writer.flush()

        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_accuracy': best_accuracy,
            'val_accuracy': val_metrics['accuracy'],
            'stage': args.stage,
        }
        save_checkpoint(ckpt_state, args.checkpoint_dir, epoch)

        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            epochs_no_improve = 0
            best_path = os.path.join(args.model_dir, 'best_flow_model.pt')
            torch.save(ckpt_state, best_path)
            print(f"New best: {best_accuracy:.2f}% -> {best_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if val_metrics['accuracy'] < 55.0 and epoch >= 2:
            print(f"WARNING: Accuracy {val_metrics['accuracy']:.1f}% near random -- check for collapse")

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_accuracy:.2f}%")

    # Stage 1 verdict
    if args.stage == 1:
        print("\n--- Stage 1 Verdict ---")
        if best_accuracy > 65.0:
            print(f"PASS ({best_accuracy:.1f}%): Flow features discriminative -> proceed to Stage 2")
        elif best_accuracy >= 60.0:
            print(f"MARGINAL ({best_accuracy:.1f}%): Investigate Farneback params before Stage 2")
        else:
            print(f"FAIL ({best_accuracy:.1f}%): Check flow computation -- NaN/Inf, resolution, Farneback params")


if __name__ == '__main__':
    main()
