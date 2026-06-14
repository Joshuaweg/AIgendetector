# CONTEXT HANDOFF
Generated: 2026-06-13
Supersedes: `tcav-2026-06-13.md`

**Summary**: Per-token TCAV implemented. Spatial concept heatmaps ready to run. Previous "sc bug" re-diagnosed as genuine concept misalignment.

---

## State Coming In

Previous session (`tcav-2026-06-13.md`) ran global TCAV on 4 concepts; all failed validation with sc ≈ 0.480 ± 0.000. The previous handoff attributed this to a code bug in the sc computation loop. That diagnosis was incorrect — TCAV correctly uses a fixed test set for sc. The ≈0.48 sc plateau indicates the 5 hand-defined concepts do not align with the model's internal decision directions.

CAV accuracy was high (0.73–0.94), meaning the model separates concept probe examples well in activation space — but the concept directions are orthogonal to the class-decision gradient. The model learned to detect AI-generated video via different features than the hand-labelled concepts.

---

## Work Done This Session

### Research
Produced a comprehensive research report on:
- TCAV for ViTs (per-token extension)
- Concept ↔ pixel bridging (CIG, CRAFT/NMF)
- Mechanistic interpretability (attention heads, activation patching, residual stream)
- Sparse Autoencoders for patch token decomposition
- Optical flow pathway interpretation
- Automated concept discovery (ACE, Network Dissection)

**4-phase implementation roadmap:**

| Phase | Work | Effort |
|---|---|---|
| 1 (done) | Per-token TCAV | 6 hrs |
| 2 | Concept Integrated Gradients + CRAFT/NMF | 2-3 days |
| 3 | Attention head analysis + activation patching | 1-2 weeks |
| 4 | Sparse Autoencoders on patch tokens | 3-4 weeks |

### Code Changes

**`tcav_interpret.py`** — added 5 functions + `--per-token` CLI flag:

| Function | Role |
|---|---|
| `extract_per_token_activations()` | `[N, 791, 768]` — full token sequence, no pooling |
| `train_per_token_cavs()` | 768 CAVs × 768-dim; one per frame patch position |
| `compute_per_token_sign_counts()` | Vectorised per-token directional derivative |
| `build_concept_heatmap()` | `[768]` → reshape → upsample 512×512 → sigma=12 → PNG |
| `validate_concept_per_token()` | Wrapper; saves `per_token_results.json` |

All existing functions (global TCAV) untouched. Per-token runs on validated concepts only.

**Architecture numbers confirmed:**
- FullLatentEncoder output: `[B, 24, 128, 64, 64]`
- extractPatches: 2 frames/patch, 8×8 patch size → **768 frame tokens** (12 temporal × 64 spatial)
- FlowEncoder: **23 flow tokens** (T−1 for T=24 frames)
- Transformer sequence: `[B, 791, 768]`

---

## Open Threads

- **[IMMEDIATE]** Run per-token TCAV and inspect heatmaps: do concept hot-spots correlate with IG attributions?
- **[IF HEATMAPS SHOW STRUCTURE]** Concepts are localized — proceed to Phase 2 (CIG)
- **[IF HEATMAPS ARE FLAT]** Model uses different features → pivot to automated discovery:
  - NMF/CRAFT on `[N_patches × 768]` activation matrix from `patch_encoder` output
  - ACE: segment latent frames, cluster activations, score via TCAV
- **[TODO]** Attribution caching in `api_server.py` — deferred two sessions
- **[TODO]** Rebalance `temporal_motion_inconsistency` probes (98+/39−) if re-running global TCAV

---

## Run Command

```bash
cd AIgendetector
python tcav_interpret.py \
    --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt \
    --probes _meta/probes/ \
    --test-manifest data/flow_manifest.csv \
    --output _meta/tcav/ \
    --per-token
# Heatmaps land in: _meta/tcav/{concept}/heatmaps/segment_00.png … segment_11.png
```

---

## Key Papers for Next Phase (CIG + circuits)

- arXiv:2411.05698 — Visual-TCAV: TCAV → saliency heatmaps
- arXiv:2604.14477 — "Seeing Through Circuits": ViT mechanistic interpretability
- arXiv:2509.00749 — Causal SAE feature interpretation (Effective Receptive Field)
- arXiv:1902.03129 — ACE: automatic concept-based explanations

**Continuation prompt:**
> Continue mechanistic interpretability work on AIgendetector. Per-token TCAV is implemented (`--per-token` flag in `tcav_interpret.py`). Run it, inspect heatmaps in `_meta/tcav/{concept}/heatmaps/`, cross-reference with IG frames. If heatmaps are spatially flat, pivot to CRAFT/NMF automated concept discovery. Read [[per_token_tcav_2026_06_13]] for full context.

---

## Related

**Supersedes:** [[tcav-2026-06-13]]

**Session log:** [[session_20260613_per_token_tcav]]

**Plan:** [[per_token_tcav_plan]]

**Research context:** [[sae-vision-transformer-research]] · [[INDEX_SAE_RESEARCH]] · [[integrated-gradients-baselines-research]] · [[vivit-tubelet-embeddings]]

**Project:** [[MANIFEST]] · [[STATE]]
