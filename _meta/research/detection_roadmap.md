# AI Video Detection — Full Architecture Roadmap

**Updated:** 2026-03-29
**Constraint:** Every component must be end-to-end IG-traceable (raw pixel → prediction). No frozen pretrained backbones.

---

## Related

**Summary:** [[RESEARCH_SUMMARY]] · [[ai_video_detection_research]] · [[frontier-model-detection-research]] · [[mvad-dataset-research]]

**Experiments:** [[EXPERIMENTS_SUMMARY]] · [[EXPERIMENTS_GUIDE]]

**Architecture:** [[flow_branch_plan]] · [[vivit-tubelet-embeddings]] · [[integrated-gradients-baselines-research]]

**Project:** [[MANIFEST]] · [[STATE]]

---

## Architecture Philosophy

All models trained from scratch. No pretrained feature extractors (DINOv2, CLIP, etc.) — they break IG traceability because gradients flow through weights optimized for a different task, not this one. Every weight in the system must be optimized for AI video detection so that IG attributions are meaningful.

---

## Phase 1 — Two Independent Classifiers (in progress)

Train two models completely independently from zero. Ensemble at inference.

### 1A: Flow Classifier (training now)
```
frames → Farneback optical flow → FlowEncoder (0.6M) → mean pool → Linear(768, 2)
```
- Dataset: `data/flow_manifest.csv` (30,790 videos, 19 generators, 1K cap)
- max_frames=8, epochs=8, batch_size=4
- IG traces: prediction ← Linear ← FlowEncoder ← flow maps ← frames ✅

### 1B: Spatial Classifier (next)
```
frames → FullLatentEncoder → FullPatchEncoder → FullClassifier
```
- Same manifest, same train/val/test split
- Trained independently, no flow input
- IG traces: prediction ← Classifier ← PatchEncoder ← LatentEncoder ← frames ✅

### Phase 1 Ensemble
```
P(fake) = 0.5 * flow_softmax + 0.5 * spatial_softmax
```
Weighting tunable on held-out val set after both models trained.

---

## Phase 2 — Additional Interpretable Signal Branches

All signals are differentiable operations on raw frames → fully IG-traceable.

### 2A: Trajectory Curvature Module (replaces ReStraV/DINOv2)

**Why not DINOv2:** Frozen pretrained backbone. IG traces through generic ImageNet features, not task-specific signals. Breaks interpretability.

**Our approach:** Compute trajectory curvature on features from our own trained PatchEncoder.

```
frames → LatentEncoder → PatchEncoder → [B, T, N_patches, 768]
                                              ↓
                               mean pool patches → [B, T, 768] per-frame feature
                                              ↓
                               consecutive differences: d[t] = f[t+1] - f[t]
                                              ↓
                               curvature[t] = ||d[t+1] - d[t]|| / ||d[t]||
                                              ↓
                               [mean_curvature, std_curvature, max_curvature,
                                total_path_length, straightness_ratio]  → [5-dim]
                                              ↓
                               MLP(5 → 64 → 2)  →  prediction
```

- Real video: low curvature (consistent motion in latent space)
- AI video: high curvature (frame-to-frame inconsistency in learned space)
- Since PatchEncoder is task-trained, curvature measures *detection-relevant* inconsistency
- IG traces all the way to pixels ✅
- Same geometric insight as ReStraV, task-specific features

### 2B: Temporal FFT Branch

```
frames [B, T, H, W, C] → per-pixel time-series [B, H*W*C, T]
                        → 1D FFT along T axis
                        → magnitude spectrum [B, H*W*C, T//2]
                        → spatial pooling + stats (mean energy, spectral entropy,
                          high-freq ratio, phase coherence)
                        → MLP → prediction
```

- Catches temporal flicker and jitter invisible per-frame
- Fully differentiable PyTorch FFT ops ✅
- ~100ms per video at 224×224

### 2C: Camera Physics Branch

All implemented as differentiable PyTorch ops on raw frames.

| Signal | What it detects | Implementation |
|---|---|---|
| Sensor noise / FPN | Real cameras have fixed spatial noise pattern; AI noise is random | Subtract temporal mean, analyze residual spatial consistency |
| Temporal noise autocorrelation | Real sensor noise is frame-independent; AI noise can be correlated | Per-pixel noise autocorrelation at lag-1 |
| Motion blur alignment | Real optical blur aligns with flow direction; AI post-process blur doesn't | Cross-correlate blur kernel direction with optical flow vectors |
| Rolling shutter skew | CMOS sensors skew fast objects in a specific physical way; AI ignores this | Measure skew of fast-moving vertical edges vs flow magnitude |
| Chromatic aberration | Real lenses fringe high-contrast edges; AI often omits or incorrectly simulates | Detect RGB channel misalignment at high-contrast edges |

Bundled as a single `CameraPhysicsEncoder` producing a compact feature vector → MLP head.

### 2D: NSG-VD Physics Branch (extends optical flow)

Builds on already-computed optical flow. Measures probability flow conservation:

```
optical flow [u, v] → spatial divergence ∇·[u,v]
                    → temporal gradient ∂ρ/∂t  (density change between frames)
                    → NSG ratio: ||spatial_grad|| / ||temporal_change||
                    → aggregated stats → MLP → prediction
```

- Real video obeys conservation laws; AI video violates them
- Fully differentiable ✅
- Near-zero additional compute (reuses flow already computed for Flow Classifier)

---

## Phase 3 — Late Fusion Ensemble

All branches produce a 2-class logit. Combine with a learned fusion head:

```
[spatial_logit, flow_logit, curvature_logit, fft_logit,
 camera_physics_logit, nsg_logit]  →  FusionMLP(12 → 64 → 2)
```

Fusion MLP trained on held-out val set with all branch models frozen.
IG on FusionMLP traces which *branch* drove the prediction → second level of interpretability.

---

## IG Interpretability Chain

Full trace for any branch:

```
Spatial:   pixel → LatentEncoder → PatchEncoder → Classifier → prediction
Flow:      pixel → Farneback → FlowEncoder → head → prediction
Curvature: pixel → LatentEncoder → PatchEncoder → curvature stats → MLP → prediction
Temp FFT:  pixel → FFT → spectral stats → MLP → prediction
Camera:    pixel → physics ops → CameraPhysicsEncoder → MLP → prediction
NSG:       pixel → Farneback → divergence → NSG stats → MLP → prediction
Ensemble:  all logits → FusionMLP → prediction  (IG shows branch weights)
```

Every path is end-to-end differentiable. No frozen pretrained backbone anywhere. ✅

---

## What Was Explicitly Rejected

| Approach | Reason |
|---|---|
| ReStraV with DINOv2 | Frozen pretrained backbone — IG traces through ImageNet features, not task-specific |
| CLIP features | Same reason |
| Any frozen ViT/ResNet | Breaks IG traceability — gradients through generic weights are not meaningful for detection |
| Stage 2 attach (flow onto trained spatial) | User clarified: train both from zero independently |

---

## Current Status

- [x] Flow Classifier training (Phase 1A) — in progress
- [ ] Spatial Classifier training (Phase 1B)
- [ ] Phase 1 ensemble evaluation
- [ ] Phase 2A: Trajectory Curvature Module
- [ ] Phase 2B: Temporal FFT Branch
- [ ] Phase 2C: Camera Physics Branch
- [ ] Phase 2D: NSG-VD Physics Branch
- [ ] Phase 3: Late Fusion Ensemble
