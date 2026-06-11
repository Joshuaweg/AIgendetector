"""
Batch concept attribution generator.

For every cluster in concept_candidates.yaml, creates:
  concepts/cluster_{id}/
    candidate_0.mp4
    candidate_0_attribution.mp4
    candidate_1.mp4
    candidate_1_attribution.mp4
    ...

The model is loaded once and reused across all videos.
Already-completed candidates are skipped on re-run.

Usage:
  python generate_concept_attributions.py
  python generate_concept_attributions.py --yaml _meta/concepts/concept_candidates.yaml \
      --output concepts/ --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_flow_maps
from full_scale_classifier import (
    FullClassifier, FullLatentEncoder, FullPatchEncoder,
    FlowEncoder, FlowVideoClassifier,
)


CHECKPOINT = 'flow_stage2_checkpoints/checkpoint_epoch_0004.pt'
YAML_PATH  = '_meta/concepts/concept_candidates.yaml'
OUTPUT_DIR = 'concepts'
N_STEPS    = 300
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
    """Returns (frames_tensor [1,T,H,W,3], raw_frames list[ndarray], label int)."""
    cap = cv2.VideoCapture(video_path)
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

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_norm = (frames_np - mean) / std
    tensor = torch.from_numpy(frames_norm).unsqueeze(0)  # [1,T,H,W,3]
    label = 0 if 'real' not in os.path.basename(video_path) else 1
    return tensor, frames, label


# ── attribution ────────────────────────────────────────────────────────────────

def run_attribution(model: FlowVideoClassifier,
                    video_tensor: torch.Tensor,
                    raw_frames: list,
                    device: torch.device) -> tuple[torch.Tensor, str, float]:
    """
    Returns (attributions [1,T,H,W,3], predicted_class str, confidence float).
    Raises RuntimeError if model forward fails.
    """
    classes = ['AI-Generated', 'Real']
    video = video_tensor.to(device)

    frames_np = np.stack(raw_frames).astype(np.float32) / 255.0
    flow_maps = compute_flow_maps(frames_np, 64, 64).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(video, flow_maps)
        pred       = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    ig = IntegratedGradients(model)
    attr_video, _ = ig.attribute(
        inputs=(video, flow_maps),
        baselines=(torch.zeros_like(video), torch.zeros_like(flow_maps)),
        target=pred,
        return_convergence_delta=False,
        n_steps=N_STEPS,
        internal_batch_size=1,
    )
    return attr_video, classes[pred], confidence


# ── visualization ──────────────────────────────────────────────────────────────

def save_attribution_video(attributions: torch.Tensor,
                           raw_frames: list,
                           label: str,
                           output_path: str):
    """Render attribution overlay frames → H.264 mp4 at output_path."""
    attr_np  = attributions.squeeze(0).permute(0, 3, 1, 2).detach().cpu().numpy()
    # unnormalize frames for display
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        for i, (a_frame, raw) in enumerate(zip(attr_np, raw_frames)):
            a_frame = a_frame.transpose(1, 2, 0)  # H,W,C
            a_smooth = np.stack(
                [gaussian_filter(a_frame[..., c], sigma=2) for c in range(3)], axis=-1
            )
            if np.max(np.abs(a_smooth)) == 0:
                a_smooth += 1e-10

            v_frame = raw.astype(np.uint8)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = viz.visualize_image_attr(
                a_smooth, v_frame,
                sign='absolute_value',
                method='blended_heat_map',
                cmap='inferno',
                alpha_overlay=0.65,
                fig_size=(12, 9),
                show_colorbar=True,
                use_pyplot=False,
                title=label,
            )
            fig.savefig(os.path.join(tmp, f'frame_{i:03d}.png'), bbox_inches='tight')
            plt.close(fig)

        # assemble video
        images = sorted(f for f in os.listdir(tmp) if f.endswith('.png'))
        if not images:
            return
        first = cv2.imread(os.path.join(tmp, images[0]))
        h, w  = first.shape[:2]

        tmp_mp4 = output_path + '.tmp.mp4'
        writer  = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, h))
        for img_name in images:
            writer.write(cv2.imread(os.path.join(tmp, img_name)))
        writer.release()

        # re-encode to H.264
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = shutil.which('ffmpeg')

        if ffmpeg:
            subprocess.run(
                [ffmpeg, '-y', '-i', tmp_mp4,
                 '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', output_path],
                check=True, capture_output=True,
            )
            os.remove(tmp_mp4)
        else:
            os.rename(tmp_mp4, output_path)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yaml',       default=YAML_PATH)
    p.add_argument('--output',     default=OUTPUT_DIR)
    p.add_argument('--checkpoint', default=CHECKPOINT)
    p.add_argument('--clusters',   nargs='+', type=int, default=None,
                   help='Only process these cluster IDs (default: all)')
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

    total_candidates = sum(len(c['top_20_videos']) for c in clusters)
    print(f'{len(clusters)} clusters, {total_candidates} candidates total\n')

    overall = tqdm(total=total_candidates, desc='Total', unit='vid')

    for cluster in clusters:
        cid       = cluster['cluster_id']
        cluster_dir = Path(args.output) / f'cluster_{cid}'
        cluster_dir.mkdir(parents=True, exist_ok=True)

        candidates = cluster['top_20_videos']
        print(f'\n── Cluster {cid} ({len(candidates)} candidates) ──')

        for n, entry in enumerate(candidates):
            src_path   = entry['path']
            orig_dest  = cluster_dir / f'candidate_{n}.mp4'
            attr_dest  = cluster_dir / f'candidate_{n}_attribution.mp4'

            # resume: skip if both outputs exist
            if orig_dest.exists() and attr_dest.exists():
                overall.update(1)
                continue

            # copy original video
            if not orig_dest.exists():
                try:
                    shutil.copy2(src_path, orig_dest)
                except Exception as e:
                    print(f'  [SKIP copy] candidate_{n}: {e}')
                    overall.update(1)
                    continue

            # run attribution
            if not attr_dest.exists():
                try:
                    video_tensor, raw_frames, _ = load_video(src_path)
                    attrs, pred_class, conf = run_attribution(
                        model, video_tensor, raw_frames, device
                    )
                    label_str = f'Cluster {cid} · candidate {n} · {pred_class} ({conf:.3f})'
                    save_attribution_video(attrs, raw_frames, label_str, str(attr_dest))
                    print(f'  [OK] candidate_{n}: {pred_class} {conf:.3f}  ← {Path(src_path).name}')
                except Exception as e:
                    print(f'  [FAIL] candidate_{n}: {e}')

            overall.update(1)

    overall.close()
    print(f'\nDone. Results in {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
