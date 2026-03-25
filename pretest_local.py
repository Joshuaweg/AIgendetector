"""
6-stage local pretest for sm_train_v2.py
Run this entirely before launching SageMaker. All 6 stages must pass.

Usage:
    python pretest_local.py                          # stages 1-4 (no video files needed)
    python pretest_local.py --data-dir /path/to/vids # all 6 stages
"""

import argparse
import os
import sys
import json
import tempfile
import traceback


PASS = "[PASS]"
FAIL = "[FAIL]"


def stage(n, name):
    print(f"\n{'='*60}")
    print(f"Stage {n}: {name}")
    print('='*60)


def run_stage(n, name, fn):
    stage(n, name)
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except Exception as e:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Stage 1: Imports
# ---------------------------------------------------------------------------

def test_imports():
    import torch
    import torch.nn as nn
    import cv2
    import numpy as np
    import sklearn
    import seaborn
    import matplotlib
    import captum
    from full_scale_classifier import (
        FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
    )
    from sm_train_v2 import VideoDataset, custom_collate_fn, save_checkpoint, find_latest_checkpoint
    print(f"  torch={torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  cv2={cv2.__version__}")
    print(f"  captum={captum.__version__}")
    print("  All imports OK")


# ---------------------------------------------------------------------------
# Stage 2: Model forward pass with dummy data
# ---------------------------------------------------------------------------

def test_model_forward():
    import torch
    from full_scale_classifier import (
        FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FullClassifier()
    ).to(device)
    model.eval()

    # Shape: [batch=2, frames=24, H=512, W=512, C=3]
    dummy = torch.zeros(2, 24, 512, 512, 3, device=device)
    print(f"  Input shape: {dummy.shape}")

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(dummy)

    assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"  Output shape: {out.shape}  ✓")
    print(f"  No NaN/Inf  ✓")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")


# ---------------------------------------------------------------------------
# Stage 3: Single training step
# ---------------------------------------------------------------------------

def test_train_step():
    import torch
    import torch.nn as nn
    from full_scale_classifier import (
        FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FullClassifier()
    ).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    # Minimal batch
    dummy = torch.zeros(2, 24, 512, 512, 3, device=device)
    labels = torch.tensor([0, 1], device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        out = model(dummy)
        loss = criterion(out, labels)

    assert not torch.isnan(loss), f"NaN loss: {loss.item()}"
    assert not torch.isinf(loss), f"Inf loss: {loss.item()}"
    print(f"  Forward loss: {loss.item():.4f}  ✓")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()

    # Check gradients flowed
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    print(f"  Backward pass OK, {len(grads)} param groups have gradients  ✓")


# ---------------------------------------------------------------------------
# Stage 4: Checkpoint save and load
# ---------------------------------------------------------------------------

def test_checkpoint():
    import torch
    from full_scale_classifier import (
        FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
    )
    from sm_train_v2 import save_checkpoint, find_latest_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FullClassifier()
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    # Record weights before save
    w_before = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        state = {
            'epoch': 3,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': {},
            'scaler_state_dict': scaler.state_dict(),
            'best_accuracy': 72.5,
            'val_accuracy': 74.1,
        }
        save_checkpoint(state, tmpdir, epoch=3)

        ckpt_path = find_latest_checkpoint(tmpdir)
        assert ckpt_path is not None, "find_latest_checkpoint returned None"
        print(f"  Checkpoint found: {os.path.basename(ckpt_path)}  ✓")

        # Load into fresh model
        model2 = FullVideoClassifier(
            FullLatentEncoder(), FullPatchEncoder(), FullClassifier()
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model2.load_state_dict(ckpt['model_state_dict'])
        assert ckpt['epoch'] == 3
        assert abs(ckpt['best_accuracy'] - 72.5) < 1e-5

        # Verify weights match
        for k, v in model2.state_dict().items():
            assert torch.allclose(w_before[k].float(), v.float()), f"Weight mismatch: {k}"

    print("  Checkpoint save/load/verify  ✓")


# ---------------------------------------------------------------------------
# Stage 5: DataLoader with real video files
# ---------------------------------------------------------------------------

def infer_label_local(path):
    """
    Label detection for local test videos using uuid__source.mp4 convention.
    Also handles folder structure and filename prefixes as fallback.
    """
    import re
    from pathlib import Path
    AI_SOURCES   = ('sora', 'veo', 'runway', 'pika', 'kling', 'zeroscope',
                    'stable', 'modelscope', 'gen2', 'gen3', 'lumiere')
    REAL_SOURCES = ('msrvtt', 'kinetics', 'webvid', 'hdvila', 'intern')

    parts = [p.lower() for p in Path(path).parts]
    for part in parts:
        if part in ('real', 'real_videos', 'authentic'):
            return 1
        if part in ('ai-generated', 'ai_generated', 'fake', 'synthetic'):
            return 0

    name = Path(path).stem.lower()
    if name.startswith('real_'):
        return 1
    if name.startswith(('ai_', 'fake_')):
        return 0

    if '__' in name:
        source = name.split('__')[-1]
        for ai in AI_SOURCES:
            if source.startswith(ai):
                return 0
        for real in REAL_SOURCES:
            if source.startswith(real):
                return 1
        # YouTube-style clip IDs (e.g. 00dF_UdPwj0_000001_000011)
        if re.match(r'^[A-Za-z0-9_\-]{6,20}(_\d{6}_\d{6})?$', source):
            return 1

    return None


def build_local_video_list(data_dir):
    """Scan data_dir, infer labels with local convention, return (path, label) list."""
    from pathlib import Path
    labeled, skipped = [], 0
    for p in Path(data_dir).rglob('*.mp4'):
        label = infer_label_local(p)
        if label is not None:
            labeled.append((str(p), label))
        else:
            skipped += 1
    if skipped:
        print(f"  Warning: {skipped} videos with ambiguous labels skipped")
    return labeled


def test_dataloader(data_dir):
    from sm_train_v2 import VideoDataset, custom_collate_fn
    from torch.utils.data import DataLoader

    video_list = build_local_video_list(data_dir)
    assert len(video_list) > 0, f"No labeled videos found in {data_dir}"

    dataset = VideoDataset(data_dir=None, target_size=512, max_frames=24,
                           video_paths=video_list)
    assert len(dataset) > 0, f"No labeled videos found in {data_dir}"
    print(f"  Found {len(dataset)} labeled videos  ✓")

    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        collate_fn=custom_collate_fn, num_workers=0)
    videos, labels, paths = next(iter(loader))

    assert videos is not None, "DataLoader returned None batch"
    assert videos.dim() == 5, f"Expected 5D tensor, got {videos.dim()}D"
    assert labels.shape[0] == videos.shape[0]
    print(f"  Batch shape: {videos.shape}  ✓")
    print(f"  Labels: {labels.tolist()}  ✓")
    print(f"  Value range: [{videos.min():.3f}, {videos.max():.3f}]  ✓")


# ---------------------------------------------------------------------------
# Stage 6: Mini-epoch simulation (10 batches)
# ---------------------------------------------------------------------------

def test_mini_epoch(data_dir):
    import torch
    import torch.nn as nn
    from full_scale_classifier import (
        FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
    )
    from sm_train_v2 import VideoDataset, custom_collate_fn, run_epoch
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FullClassifier()
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    # Dummy scheduler that does nothing
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    video_list = build_local_video_list(data_dir)
    dataset = VideoDataset(data_dir=None, target_size=512, max_frames=24,
                           video_paths=video_list)

    # Cap at 10 samples for speed
    from torch.utils.data import Subset
    indices = list(range(min(10, len(dataset))))
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=2, shuffle=True,
                        collate_fn=custom_collate_fn, num_workers=0)

    metrics, preds, labels = run_epoch(
        model, loader, criterion, optimizer, scaler, scheduler, device, is_train=True
    )

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert not (metrics['loss'] != metrics['loss'])  # NaN check
    print(f"  Loss: {metrics['loss']:.4f}  Accuracy: {metrics['accuracy']:.1f}%  ✓")
    print("  Mini-epoch completed successfully  ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to labeled video directory (enables stages 5-6)')
    args = parser.parse_args()

    results = {}
    results[1] = run_stage(1, 'Imports', test_imports)
    results[2] = run_stage(2, 'Model forward pass', test_model_forward)
    results[3] = run_stage(3, 'Single train step', test_train_step)
    results[4] = run_stage(4, 'Checkpoint save/load', test_checkpoint)

    if args.data_dir:
        results[5] = run_stage(5, 'DataLoader with real videos',
                               lambda: test_dataloader(args.data_dir))
        if results[5]:
            results[6] = run_stage(6, 'Mini-epoch simulation (10 samples)',
                                   lambda: test_mini_epoch(args.data_dir))
    else:
        print("\n[SKIP] Stages 5-6 require --data-dir (no video files needed for 1-4)")

    print(f"\n{'='*60}")
    print("PRETEST SUMMARY")
    print('='*60)
    all_pass = True
    for n, passed in sorted(results.items()):
        status = PASS if passed else FAIL
        print(f"  Stage {n}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll stages passed — safe to launch SageMaker job.")
        sys.exit(0)
    else:
        print("\nOne or more stages failed — DO NOT launch SageMaker until fixed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
