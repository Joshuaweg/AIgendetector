# Optical Flow Branch — Architecture & Training Plan

**Status:** Ready to implement (local environment)
**Date:** March 2026

---

## Decision: Token-Append Fusion

Tubelet encoding = local 2-frame appearance change.
Optical flow = motion physics violations across full sequence.
Flow residuals generalize 2.5x better across generators, survive 84% post-compression.
Expected gain: +3–7% accuracy. Source: `ai_video_detection_research.md` §7B, arXiv 2508.00397.

## Architecture

```
frames → FullLatentEncoder → FullPatchEncoder → [B, 768, 768] tubelet tokens ─┐
                                                                                ├─ cat(dim=1) → [B, 791, 768] → FullClassifier (unchanged)
frames → FlowEncoder → [B, 23, 768] flow tokens ───────────────────────────────┘
```

**FlowEncoder (~0.6M params):**
- Input: [B, T-1, 6, H_f, W_f] — 6-channel flow maps (u, v, mag, angle, Δu, Δv)
- Conv2d(6→32→64→128→256) + GroupNorm + ReLU
- AdaptiveAvgPool2d(1,1) + flatten
- Linear(256, 768) + LayerNorm
- Output: [B, T-1, 768] — one 768-dim token per frame pair (23 tokens for 24 frames)

**Flow features (6 channels per frame pair, computed via cv2.calcOpticalFlowFarneback):**
- u, v — motion vectors
- magnitude = sqrt(u²+v²), angle = atan2(v,u)
- Δu = u[t+1]-u[t], Δv = v[t+1]-v[t] — 2nd-order residuals (most discriminative)

## Why Token-Append (not linear fusion)

Token-append preserves existing transformer attention weights over tubelet tokens.
Transformer just learns to additionally attend to flow tokens — no full retrain.
Linear fusion would shift the entire input distribution → full retrain required.

## Files to Create / Modify

| File | Action |
|---|---|
| `full_scale_classifier.py` | Add `FlowEncoder`; update `FullVideoClassifier` to accept `(videos, flow_maps)`, append flow tokens before transformer |
| `dataset.py` | Add `FlowVideoDataset(VideoDataset)` subclass + `flow_collate_fn` — **existing VideoDataset untouched** |
| `sm_flow_train.py` | New script based on `sm_train_v2.py`; add `--freeze-backbone` and `--pretrained-checkpoint` flags |

Existing scripts unaffected: `full_train.py`, `sm_train_v2.py`, `main.py`, `validation_matrix.py`

## Staged Training

| Stage | Frozen | Training | Objective |
|---|---|---|---|
| 1 (local) | LatentEncoder, PatchEncoder, Classifier | FlowEncoder + temp Linear(768,2) head | CrossEntropyLoss real/fake on flow only |
| 2 (SageMaker) | LatentEncoder, PatchEncoder | FlowEncoder + Transformer fine-tune | CrossEntropyLoss real/fake, load Stage 1 FlowEncoder weights |

Load `best_model.pt` (92% checkpoint) as init for Stage 2. 92% is the starting point.

**Stage 1 success criteria:**
- Accuracy > 65%: flow features are discriminative, proceed to Stage 2
- Accuracy 60–65%: marginal signal, investigate flow computation before Stage 2
- Accuracy < 60%: flow not learning — check Farneback params, flow resolution, NaN/Inf

## Training Time Estimates

Baseline confirmed: batch_size=8 (sm_train_v2.py default), 6 hrs / 5 epochs = 1.2 hrs/epoch.

| Run | Estimate |
|---|---|
| Stage 1 local (5–8 epochs, 0.6M params) | 30–60 min |
| Stage 2 SageMaker (15 epochs, patience=4, early stop ~epoch 10–12) | 10–16 hrs |
| Total | 12–19 hrs |

## Notes from sm_train_v2.py Review

- Bug: `run_epoch` for validation receives optimizer/scaler/scheduler unnecessarily — harmless but clean up in new script
- Copy directly: argparse pattern, spot-safe checkpoint recovery, SequentialLR schedule, TensorBoard + confusion matrix logging, early stopping + mode collapse guard
- New args needed: `--freeze-backbone` (bool), `--pretrained-checkpoint` (str path)
