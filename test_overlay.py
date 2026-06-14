"""
Quick smoke test for tcav_overlay.py.
Uses lightweight IG (10 steps, no NoiseTunnel) — fast but visually approximate.
Run: python test_overlay.py [video_path]
"""
import sys
import os
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients

sys.path.insert(0, str(Path(__file__).parent))
from full_scale_classifier import (
    FullLatentEncoder, FullPatchEncoder, FullClassifier,
    FlowEncoder, FlowVideoClassifier,
)
from dataset import compute_flow_maps
from interpret import load_video
from tcav_overlay import load_cavs, run_overlay

CHECKPOINT = 'flow_stage2_checkpoints/checkpoint_epoch_0004.pt'
TCAV_DIR   = Path('_meta/tcav')
OUTPUT     = Path('_meta/tcav/test_overlay.html')
VIDEO = (sys.argv[1] if len(sys.argv) > 1
         else '/media/joshua/WD_BLACK/Gen-Video/dataset/'
              'ai_dynamiccrafter_DynamicCrafter_021.mp4')

if not os.path.isfile(VIDEO):
    print(f"ERROR: video not found: {VIDEO}")
    print("Usage: python test_overlay.py /path/to/video.mp4")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model…")
model = FlowVideoClassifier(
    FullLatentEncoder(), FullPatchEncoder(), FlowEncoder(), FullClassifier()
).to(device)
try:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
except Exception:
    import numpy.core.multiarray
    with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f"  checkpoint epoch {ckpt.get('epoch')}  acc {ckpt.get('best_accuracy'):.2f}%")

# ── Load CAVs ──────────────────────────────────────────────────────────────────
print("Loading CAVs…")
cavs = load_cavs(TCAV_DIR)
if not cavs:
    print("ERROR: no cavs_layer7.npy found. Run tcav_interpret.py --per-token-only first.")
    sys.exit(1)
print(f"  concepts: {list(cavs.keys())}")

# ── Load video ─────────────────────────────────────────────────────────────────
print(f"Loading video: {VIDEO}")
frames_tensor, raw_frames, label, _ = load_video(VIDEO)
frames_tensor = frames_tensor.unsqueeze(0).to(device)   # [1, 24, 512, 512, 3]

frames_np  = np.array(raw_frames).astype(np.float32) / 255.0
flow_maps  = compute_flow_maps(frames_np, 64, 64).unsqueeze(0).to(device)  # [1,23,6,64,64]

# ── Quick prediction ───────────────────────────────────────────────────────────
with torch.no_grad():
    logits = model(frames_tensor, flow_maps)
    pred   = int(torch.argmax(logits, dim=1).item())
    probs  = torch.softmax(logits, dim=1)[0]
    conf   = float(probs[pred])

classes    = ['AI-Generated', 'Real']
prediction = classes[pred]
print(f"  Prediction: {prediction}  Confidence: {conf*100:.1f}%")

# ── Lightweight IG (10 steps — fast approximation for testing) ─────────────────
print("Running IG (50 steps)…")
frames_tensor.requires_grad_(True)
baseline      = torch.zeros_like(frames_tensor)
frozen_flow   = flow_maps.detach()

def _forward(videos):
    logits = model(videos, frozen_flow.expand(videos.shape[0], *frozen_flow.shape[1:]))
    return torch.log_softmax(logits, dim=1)

ig           = IntegratedGradients(_forward)
attributions = ig.attribute(
    frames_tensor,
    baselines=baseline,
    target=pred,
    n_steps=50,
    internal_batch_size=1,
)
print(f"  Attribution shape: {attributions.shape}  max: {attributions.abs().max():.4f}")

# ── Generate overlay ───────────────────────────────────────────────────────────
print("Generating TCAV overlay…")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out = run_overlay(
    model          = model,
    frames_tensor  = frames_tensor.detach(),
    flow_maps      = flow_maps,
    ig_attributions= attributions.detach(),
    raw_frames     = raw_frames,
    cavs           = cavs,
    output_path    = str(OUTPUT),
    video_id       = 'test',
    prediction     = prediction,
    confidence     = conf,
    device         = device,
)

if out:
    print(f"\nOverlay saved → {out}")
    print("Opening in browser…")
    os.system(f'xdg-open "{out}" 2>/dev/null &')
else:
    print("Overlay generation returned None — check logs above.")
