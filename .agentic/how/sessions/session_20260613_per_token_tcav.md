---
type: session
status: closed
created: 2026-06-13
updated: 2026-06-13
last_edited_by: claude-sonnet-4-6
tags: [session, tcav, mechanistic-interpretability, per-token]
---

# Session: Per-Token TCAV + Mechanistic Interpretability Research

## Goal

Investigate the TCAV implementation's spatial limitations and establish a mechanistic interpretability roadmap. User's question: why are highly-attributed pixels important, not just where they are.

## Work Done

### Research

Produced an 8-section practitioner research report covering:
1. **Diagnosis of current TCAV** — mean-pool bug: `tokens.mean(dim=1)` in `_extract_transformer_acts` and `act.grad[0].mean(dim=0)` in `compute_sign_count` discard all spatial localization
2. **Per-token TCAV** — algorithm for [N, 791, 768] activations; per-patch CAV training; spatial heatmaps
3. **Pixel ↔ Concept bridging** — Concept Integrated Gradients (CIG), CRAFT/NMF, attention-weighted attribution
4. **Mechanistic interp toolkit** — attention head analysis, activation patching/causal tracing, residual stream analysis
5. **SAEs** — TopK SAE for patch token decomposition; separate spatial vs motion SAEs
6. **Optical flow pathway** — per-layer convergence analysis; flow channel importance (u, v, divergence, curl)
7. **Automated concept discovery** — ACE (SLIC segmentation + K-means + TCAV scoring), Network Dissection
8. **Implementation roadmap** — 4-phase plan with effort estimates

Key papers surfaced: arXiv:2411.05698 (Visual-TCAV), arXiv:2604.14477 (Seeing Through Circuits), arXiv:2509.00749 (Causal SAE features / ERF), arXiv:1902.03129 (ACE).

### Implementation

Added to `tcav_interpret.py` (all existing functions untouched):

| Function | Purpose |
|---|---|
| `extract_per_token_activations()` | Returns `[N, 791, 768]` — no mean-pool |
| `train_per_token_cavs()` | 768 logistic regression CAVs, one per frame token |
| `compute_per_token_sign_counts()` | Slices `act.grad[0, p, :]` per token; vectorised dot |
| `build_concept_heatmap()` | `[768]` → `[12, 8, 8]` → upsample 512×512 → sigma=12 → 12 PNGs |
| `validate_concept_per_token()` | Wrapper; saves `per_token_results.json` |
| `--per-token` flag | Opt-in in main(); runs only on validated concepts |

Smoothing: `gaussian_filter(sigma=12)` applied in pixel space after bilinear upsampling, matching IG pipeline exactly.

Architecture confirmed: FullLatentEncoder → `[B, 24, 128, 64, 64]`; extractPatches (8×8, 2 frames/patch) → 768 frame tokens (12 temporal × 8×8 spatial); FlowEncoder → 23 flow tokens; transformer input `[B, 791, 768]`.

## Key Decision

Previous handoff described a "sc computation bug" (sc frozen at 0.480 ± 0.000). On re-examination: TCAV correctly uses a fixed test set for sc across all splits — if CAV converges to a stable direction (high CAV accuracy), sc will indeed be constant. This is not a code bug. The ≈0.48 sc means the 5 hand-defined concepts are likely not aligned with model decisions. Per-token heatmaps will confirm this.

## SITREP

- **Delivered**: per-token TCAV implementation ready to run
- **Open**: need to run with `--per-token`, inspect heatmaps, cross-reference with IG
- **If concepts misalign**: pivot to CRAFT/NMF automated discovery
- **Next phase**: Concept Integrated Gradients links pixel maps to concept directions

## Run Command

```bash
cd AIgendetector
python tcav_interpret.py \
    --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt \
    --probes _meta/probes/ \
    --test-manifest data/flow_manifest.csv \
    --output _meta/tcav/ \
    --per-token
```

---

## Related

**Handoff:** [[per_token_tcav_2026_06_13]]

**Plan:** [[per_token_tcav_plan]]

**Research:** [[sae-vision-transformer-research]] · [[INDEX_SAE_RESEARCH]] · [[integrated-gradients-baselines-research]] · [[vivit-tubelet-embeddings]]

**Project:** [[MANIFEST]] · [[STATE]]
