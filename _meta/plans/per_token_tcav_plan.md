# Plan: Per-Token TCAV with Concept Heatmaps

## Related

**Handoff:** [[per_token_tcav_2026_06_13]] · [[tcav-2026-06-13]]

**Session:** [[session_20260613_per_token_tcav]]

**Research:** [[sae-vision-transformer-research]] · [[integrated-gradients-baselines-research]] · [[INDEX_SAE_RESEARCH]]

**Project:** [[MANIFEST]] · [[STATE]]

---

## Context

Current TCAV mean-pools the token sequence before CAV training and gradient computation, discarding all spatial/temporal localization. We're adding per-token TCAV (Option A from the research report) that produces a **concept heatmap over the 8×8×12 patch grid**, telling you which spatial regions and temporal segments carry each concept — not just whether the concept influences the video globally.

The user asked about Gaussian smoothing: IG applies `sigma=12` in pixel space on 512×512 maps. The patch grid is only 8×8 per temporal segment, so applying sigma=12 directly in patch space would average everything to a uniform blob. **The plan matches IG exactly by upsampling the 8×8 heatmap to 512×512 first, then applying gaussian_filter(sigma=12)** — producing overlays in the same visual scale as existing IG frames.

---

## Architecture Numbers (confirmed)

| Stage | Output Shape |
|-------|-------------|
| FullLatentEncoder | `[B, 24, 128, 64, 64]` |
| extractPatches | 12 temporal segments × 8×8 spatial = **768 frame tokens** |
| FlowEncoder | **23 flow tokens** |
| Transformer input | `[B, 791, 768]` |

---

## File to Modify

**`AIgendetector/tcav_interpret.py`** — add 4 new functions + 1 CLI flag. All additions are append-only; existing functions (`extract_activations`, `compute_sign_count`, `validate_concept`, `main`) are unchanged.

---

## Implementation

### 1. `extract_per_token_activations()` — ~30 LOC

Returns `[N_examples, 791, 768]` (no pooling). Mirrors `_extract_transformer_acts` but drops the `.mean(dim=1)` call.

```
acts.append(tokens.squeeze(0).cpu().numpy())  # [791, 768] per example
→ np.stack(acts)  # [N, 791, 768]
```

Only implemented for transformer layers (same as current); hook-based pathway layers stay pooled.

### 2. `train_per_token_cavs()` — ~25 LOC

Trains one L2 logistic regression per **frame token position** (positions 0–767 only; flow tokens excluded from spatial CAV).

```python
cavs = np.zeros((768, 768), dtype=np.float32)
accs = np.zeros(768, dtype=np.float32)
for p in range(768):
    cav, acc = train_cav(pos_acts[:, p, :], neg_acts[:, p, :])
    if cav is not None:
        cavs[p] = cav
        accs[p] = acc
return cavs, accs  # [768, 768], [768]
```

Degenerate positions (all-zero activations, or fewer than 4 unique examples) set to zero CAV and acc=0 — handled by existing `train_cav` returning `(None, 0.0)`.

### 3. `compute_per_token_sign_counts()` — ~35 LOC

Runs one backward pass per test example; slices `act.grad[0, p, :]` per position instead of mean-pooling the gradient.

```python
# grad: [1, 791, 768] — already computed in backward pass
for p in range(768):  # frame tokens only
    directional = np.dot(act.grad[0, p, :].cpu().numpy(), cavs[p])
    positive[p] += 1 if directional > 0 else 0
return positive / total  # [768]
```

One backward pass per video — same cost as current implementation.

### 4. `build_concept_heatmap()` — ~20 LOC

Converts `[768]` sign counts → visual overlay matching IG resolution:

```
sign_counts [768]
  → reshape [12, 8, 8]  (12 temporal segments, 8×8 spatial grid)
  → for each segment: bilinear upsample [8, 8] → [512, 512]
  → gaussian_filter(sigma=12)  ← matches IG exactly
  → normalize to [0, 1]
  → save as PNG per segment (same naming convention as IG frames)
```

Output directory: `_meta/tcav/{concept}/heatmaps/segment_{i:02d}.png`

### 5. `validate_concept_per_token()` — ~40 LOC

Wrapper that calls steps 1–4. Returns the standard result dict (same schema as `validate_concept`) plus two new keys:

```python
{
    ...existing keys...,
    'per_token_sign_counts': counts.tolist(),   # [768] — raw spatial data
    'mean_cav_accuracy_spatial': float,          # mean of accs[768] where acc > 0
}
```

### 6. CLI flag — `--per-token` (default: off)

In `main()`, after existing concept loop:

```python
if args.per_token and result['validated']:
    counts = validate_concept_per_token(...)
    build_concept_heatmap(counts, args.output / concept)
```

Only runs on validated concepts (no wasted compute on failed probes).

---

## Smoothing Decision

| Space | IG | TCAV heatmap |
|-------|-----|--------------|
| Resolution | 512×512 pixels | 8×8 patches → upsample to 512×512 |
| sigma | 12 | **12** (applied after upsampling) |
| Library | `scipy.ndimage.gaussian_filter` | same |

Visual output: TCAV concept heatmaps will have the same blob-size as IG attributions when overlaid.

---

## Verification

```bash
cd AIgendetector
python tcav_interpret.py \
    --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt \
    --probes _meta/probes/ \
    --test-manifest data/flow_manifest.csv \
    --output _meta/tcav/ \
    --per-token

# Check outputs:
ls _meta/tcav/temporal_motion_inconsistency/heatmaps/
# → segment_00.png … segment_11.png (12 temporal segment maps)

# Sanity check: do heatmap hot-spots correlate with IG frame attributions?
# - Load one IG frame and corresponding TCAV heatmap segment
# - Compute Spearman rank correlation of pixel intensities
# - Expect r > 0.3 for validated concepts
```
