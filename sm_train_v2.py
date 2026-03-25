"""
SageMaker training entry point for FullVideoClassifier v2.
- Tubelet-corrected FullPatchEncoder (concatenation, not averaging)
- Vectorized FullLatentEncoder (no per-frame synchronize)
- Spot-instance safe: loads latest checkpoint from /opt/ml/checkpoints on startup
- Saves split indices so the exact train/test split is reproducible
- File mode: reads from SM_CHANNEL_TRAINING (S3 copied to local EBS before job starts)
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from full_scale_classifier import (
    FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoDataset(Dataset):
    """
    Scans a directory tree for .mp4 files and infers labels from folder structure.
    Supports: AI-Generated/<generator>/*.mp4 → 0, Real/<source>/*.mp4 → 1
    Falls back to filename prefix (real_*, ai_*, fake_*) if folder names are ambiguous.
    """

    def __init__(self, data_dir, target_size=512, max_frames=24, video_paths=None):
        self.target_size = target_size
        self.max_frames = max_frames
        self.videos = []

        if video_paths is not None:
            for path, label in video_paths:
                self.videos.append((str(path), label))
        else:
            self._scan_directory(data_dir)

        random.shuffle(self.videos)
        real_count = sum(1 for _, l in self.videos if l == 1)
        ai_count = sum(1 for _, l in self.videos if l == 0)
        print(f"Dataset: {len(self.videos)} videos — {real_count} real, {ai_count} AI")

    def _scan_directory(self, data_dir):
        data_path = Path(data_dir)
        skipped = 0
        for video_path in data_path.rglob('*.mp4'):
            label = self._infer_label(video_path)
            if label is not None:
                self.videos.append((str(video_path), label))
            else:
                skipped += 1
        if skipped:
            print(f"Warning: skipped {skipped} videos with ambiguous labels")

    @staticmethod
    def _infer_label(path):
        """Infer 0=AI, 1=Real from folder structure or filename prefix."""
        parts = [p.lower() for p in Path(path).parts]
        for part in parts:
            if part in ('real', 'real_videos', 'authentic', 'genuine'):
                return 1
            if part in ('ai-generated', 'ai_generated', 'fake', 'synthetic', 'generated'):
                return 0
        name = Path(path).name.lower()
        if name.startswith(('real_',)):
            return 1
        if name.startswith(('ai_', 'fake_', 'gen_', 'synthetic_')):
            return 0
        return None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        try:
            frames = self._load_video(video_path)
            if frames is None:
                return None, None, video_path
            frames_tensor = torch.FloatTensor(frames)
            return frames_tensor, label, video_path
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None, None, video_path

    def _load_video(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        frames = []
        while len(frames) < self.max_frames * 2:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.target_size, self.target_size))
            frames.append(frame)
        cap.release()
        if not frames:
            return None
        frames = self._sample_frames(np.array(frames, dtype=np.float32))
        frames = frames / 255.0
        return frames

    def _sample_frames(self, frames):
        n = len(frames)
        if n <= self.max_frames:
            return frames
        start = random.randint(0, n - self.max_frames)
        return frames[start:start + self.max_frames]


def custom_collate_fn(batch):
    valid = [(v, l, p) for v, l, p in batch if v is not None and isinstance(v, torch.Tensor)]
    if not valid:
        raise RuntimeError("No valid videos in batch")
    videos, labels, paths = zip(*valid)
    max_frames = max(v.shape[0] for v in videos)
    h = videos[0].shape[1]
    w = videos[0].shape[2]
    padded = torch.zeros(len(videos), max_frames, h, w, 3)
    for i, v in enumerate(videos):
        n = min(v.shape[0], max_frames)
        padded[i, :n] = v[:n]
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, labels_t, list(paths)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir):
    """Return path to highest-epoch checkpoint, or None if none exist."""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pt')
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (videos, labels, _) in enumerate(loader):
            if videos is None:
                continue
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(videos)
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

            del outputs, loss, preds, scores, videos, labels
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker environment
    parser.add_argument('--training', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './model_output'))
    parser.add_argument('--checkpoint-dir', type=str,
                        default=os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'))
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--target-size', type=int, default=512)
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=314159)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
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

    # -----------------------------------------------------------------------
    # Dataset & split
    # -----------------------------------------------------------------------
    print(f"\nScanning data at: {args.training}")
    full_dataset = VideoDataset(
        args.training,
        target_size=args.target_size,
        max_frames=args.max_frames
    )

    if len(full_dataset) == 0:
        raise RuntimeError(f"No labeled videos found in {args.training}")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    # Persist split indices so this run is exactly reproducible
    split_path = os.path.join(args.model_dir, 'train_test_split.json')
    split_data = {
        'seed': args.seed,
        'total': len(full_dataset),
        'train_indices': train_dataset.indices,
        'test_indices': test_dataset.indices,
        'train_paths': [full_dataset.videos[i][0] for i in train_dataset.indices],
        'test_paths': [full_dataset.videos[i][0] for i in test_dataset.indices],
    }
    with open(split_path, 'w') as f:
        json.dump(split_data, f)
    print(f"Split saved: {split_path}  ({train_size} train / {test_size} test)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = FullVideoClassifier(
        FullLatentEncoder(),
        FullPatchEncoder(),
        FullClassifier()
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps]
    )

    # -----------------------------------------------------------------------
    # Checkpoint recovery (spot resumption)
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
        print(f"Resumed from epoch {ckpt['epoch']}, best accuracy: {best_accuracy:.2f}%")
    else:
        print("No checkpoint found — training from scratch")

    # -----------------------------------------------------------------------
    # TensorBoard
    # -----------------------------------------------------------------------
    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_dir = os.path.join(args.output_data_dir, f'tb_{run_name}')
    writer = SummaryWriter(tb_dir)

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
            model, train_loader, criterion, optimizer, scaler, scheduler, device, is_train=True
        )
        val_metrics, val_preds, val_labels = run_epoch(
            model, test_loader, criterion, optimizer, scaler, scheduler, device, is_train=False
        )

        elapsed = (time.time() - epoch_start) / 60
        print(f"Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.2f}%  f1: {train_metrics['f1']:.4f}")
        print(f"Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.2f}%  f1: {val_metrics['f1']:.4f}  auc: {val_metrics.get('auc', 0):.4f}")
        print(f"Epoch time: {elapsed:.1f} min")

        # TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f'Train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Confusion matrix plot (epoch end only)
        if len(set(val_labels)) == 2:
            cm = confusion_matrix(val_labels, val_preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['AI', 'Real'], yticklabels=['AI', 'Real'])
            ax.set_title(f'Epoch {epoch+1}')
            writer.add_figure('Confusion_Matrix', fig, epoch)
            plt.close(fig)

        writer.flush()

        # Save checkpoint every epoch to /opt/ml/checkpoints (auto-synced to S3 by SageMaker)
        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_accuracy': best_accuracy,
            'val_accuracy': val_metrics['accuracy'],
        }
        save_checkpoint(ckpt_state, args.checkpoint_dir, epoch)

        # Save best model to model_dir (gets uploaded to S3 output at job end)
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            epochs_no_improve = 0
            best_path = os.path.join(args.model_dir, 'best_model.pt')
            torch.save(ckpt_state, best_path)
            print(f"New best: {best_accuracy:.2f}% → {best_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Mode collapse guard
        if val_metrics['accuracy'] < 55.0 and epoch >= 2:
            print(f"WARNING: Accuracy {val_metrics['accuracy']:.1f}% near random — check for collapse")

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {args.model_dir}")


if __name__ == '__main__':
    main()
