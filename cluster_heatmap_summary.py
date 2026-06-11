"""
Cluster attribution heatmap summarizer.

For each cluster in concept_candidates.yaml, computes the mean absolute
attribution map across all candidate videos and saves:
  concepts/cluster_{id}/mean_attribution.png   — spatial heatmap
  concepts/cluster_{id}/mean_attribution.txt   — peak region summary

Uses only 50 IG steps (sufficient for averaging — noise washes out).
Model is loaded once and reused.

Usage:
  python cluster_heatmap_summary.py
  python cluster_heatmap_summary.py --clusters 0 1 2 --steps 50
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_flow_maps
from full_scale_classifier import (
    FullClassifier, FullLatentEncoder, FullPatchEncoder,
    FlowEncoder, FlowVideoClassifier,
)
from captum.attr import IntegratedGradients


CHECKPOINT = 'flow_stage2_checkpoints/checkpoint_epoch_0004.pt'
YAML_PATH  = '_meta/concepts/concept_candidates.yaml'
OUTPUT_DIR = 'concepts'
N_FRAMES   = 24
FRAME_SIZE = 512


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> FlowVideoClassifier:
    model = FlowVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FlowEncoder(), FullClassifier()
    ).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


# ── video loading ──────────────────────────────────────────────────────────────

def load_video(video_path: str):
    """Returns (frames_tensor [1,T,H,W,3], raw_frames list[ndarray])."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(
        int(round(i * (total - 1) / max(N_FRAMES - 1, 1))) for i in range(N_FRAMES)
    ) if total >= N_FRAMES else None

    frames, fi = [], 0
    while cap.isOpened() and len(frames) < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if indices is None or fi in indices:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        fi += 1
    cap.release()

    while len(frames) < N_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
    frames = frames[:N_FRAMES]

    mean_n = np.array([0.485, 0.456, 0.406], np.float32)
    std_n  = np.array([0.229, 0.224, 0.225], np.float32)
    frames_np   = np.stack(frames).astype(np.float32) / 255.0
    frames_norm = (frames_np - mean_n) / std_n
    tensor = torch.from_numpy(frames_norm).unsqueeze(0)  # [1,T,H,W,3]
    return tensor, frames


# ── attribution (returns spatial mean, no video saved) ────────────────────────

def compute_spatial_mean(model: FlowVideoClassifier,
                         video_tensor: torch.Tensor,
                         raw_frames: list,
                         device: torch.device,
                         n_steps: int) -> np.ndarray | None:
    """
    Returns mean absolute attribution map [H, W] normalised to [0,1],
    or None on failure.
    """
    video = video_tensor.to(device)
    frames_np = np.stack(raw_frames).astype(np.float32) / 255.0
    flow_maps = compute_flow_maps(frames_np, 64, 64).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            output = model(video, flow_maps)
            pred = torch.argmax(output, dim=1).item()

        ig = IntegratedGradients(model)
        attr_video, _ = ig.attribute(
            inputs=(video, flow_maps),
            baselines=(torch.zeros_like(video), torch.zeros_like(flow_maps)),
            target=pred,
            return_convergence_delta=False,
            n_steps=n_steps,
            internal_batch_size=1,
        )
        # [1,T,H,W,3] → abs → mean over T and C → [H,W]
        spatial = attr_video.squeeze(0).abs().detach().cpu().numpy()  # [T,H,W,3]
        spatial = spatial.mean(axis=(0, 3))                            # [H,W]
        denom   = spatial.max()
        return spatial / denom if denom > 1e-8 else spatial
    except Exception as e:
        print(f'    attribution failed: {e}')
        return None


# ── region analysis ────────────────────────────────────────────────────────────

def describe_peak_region(heatmap: np.ndarray, threshold: float = 0.6) -> str:
    """
    Returns a human-readable description of where the top activations are.
    Divides the map into a 3×3 grid and reports the hottest cells.
    """
    H, W = heatmap.shape
    cells = {}
    row_names = ['top', 'mid', 'bottom']
    col_names = ['left', 'center', 'right']
    for ri in range(3):
        for ci in range(3):
            r0, r1 = ri * H // 3, (ri + 1) * H // 3
            c0, c1 = ci * W // 3, (ci + 1) * W // 3
            cells[(ri, ci)] = heatmap[r0:r1, c0:c1].mean()

    max_val = max(cells.values())
    hot = [(row_names[r], col_names[c])
           for (r, c), v in cells.items()
           if v >= threshold * max_val]
    hot_str = ', '.join(f'{r}-{c}' for r, c in hot)

    # also report peak pixel location as % of frame
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    px_pct = int(100 * peak_x / W)
    py_pct = int(100 * peak_y / H)

    return f'Hot cells: {hot_str}\nPeak pixel: x={px_pct}% from left, y={py_pct}% from top'


# ── visualisation ──────────────────────────────────────────────────────────────

def save_cluster_heatmap(mean_map: np.ndarray,
                         cluster_id: int,
                         n_videos: int,
                         region_desc: str,
                         out_path: str):
    """Save annotated heatmap as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Cluster {cluster_id} — mean attribution ({n_videos} videos)',
                 fontsize=13, fontweight='bold')

    # left: raw heatmap
    im = axes[0].imshow(mean_map, cmap='inferno', vmin=0, vmax=1)
    axes[0].set_title('Mean |attribution| (normalised)')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # right: 3×3 grid overlay
    H, W  = mean_map.shape
    grid  = np.zeros((3, 3))
    for ri in range(3):
        for ci in range(3):
            r0, r1 = ri * H // 3, (ri + 1) * H // 3
            c0, c1 = ci * W // 3, (ci + 1) * W // 3
            grid[ri, ci] = mean_map[r0:r1, c0:c1].mean()
    grid /= grid.max() + 1e-8

    row_labels = ['top', 'mid', 'bottom']
    col_labels = ['left', 'center', 'right']
    im2 = axes[1].imshow(grid, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_xticks([0, 1, 2]); axes[1].set_xticklabels(col_labels)
    axes[1].set_yticks([0, 1, 2]); axes[1].set_yticklabels(row_labels)
    axes[1].set_title('3×3 region summary')
    for ri in range(3):
        for ci in range(3):
            axes[1].text(ci, ri, f'{grid[ri, ci]:.2f}',
                         ha='center', va='center',
                         color='white' if grid[ri, ci] < 0.6 else 'black',
                         fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.text(0.5, 0.01, region_desc, ha='center', fontsize=9, style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yaml',       default=YAML_PATH)
    p.add_argument('--output',     default=OUTPUT_DIR)
    p.add_argument('--checkpoint', default=CHECKPOINT)
    p.add_argument('--steps',      type=int, default=50,
                   help='IG steps per video (50 is fine for averaged maps)')
    p.add_argument('--clusters',   nargs='+', type=int, default=None,
                   help='Only process these cluster IDs (default: all)')
    p.add_argument('--force',      action='store_true',
                   help='Recompute even if mean_attribution.png already exists')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading model...')
    model = load_model(args.checkpoint, device)
    print('Model ready.\n')

    with open(args.yaml) as f:
        data = yaml.safe_load(f)
    clusters = data['concept_candidates']
    if args.clusters:
        clusters = [c for c in clusters if c['cluster_id'] in args.clusters]

    for cluster in clusters:
        cid         = cluster['cluster_id']
        candidates  = cluster['top_20_videos']
        cluster_dir = Path(args.output) / f'cluster_{cid}'
        cluster_dir.mkdir(parents=True, exist_ok=True)

        out_png = cluster_dir / 'mean_attribution.png'
        out_txt = cluster_dir / 'mean_attribution.txt'

        if out_png.exists() and not args.force:
            print(f'[skip] cluster {cid} — already done (use --force to recompute)')
            continue

        print(f'\n── Cluster {cid}: {cluster["ai_count"]} AI / {cluster["real_count"]} real '
              f'({len(candidates)} candidates) ──')

        accumulated = None
        count       = 0

        for entry in tqdm(candidates, desc=f'cluster {cid}', leave=False):
            src = entry['path']
            if not os.path.exists(src):
                continue
            try:
                video_tensor, raw_frames = load_video(src)
                spatial = compute_spatial_mean(model, video_tensor, raw_frames,
                                               device, args.steps)
                if spatial is None:
                    continue
                accumulated = spatial if accumulated is None else accumulated + spatial
                count += 1
            except Exception as e:
                print(f'  [SKIP] {Path(src).name}: {e}')

        if accumulated is None or count == 0:
            print(f'  No successful attributions for cluster {cid}')
            continue

        mean_map     = accumulated / count
        mean_map    /= mean_map.max() + 1e-8
        region_desc  = describe_peak_region(mean_map)

        save_cluster_heatmap(mean_map, cid, count, region_desc, str(out_png))

        summary = (f'Cluster {cid}\n'
                   f'Videos averaged: {count}/{len(candidates)}\n'
                   f'AI: {cluster["ai_count"]}  Real: {cluster["real_count"]}\n'
                   f'{region_desc}\n')
        out_txt.write_text(summary)

        print(f'  {region_desc.replace(chr(10), "  |  ")}')
        print(f'  → {out_png}')

    print('\nDone.')


if __name__ == '__main__':
    main()
