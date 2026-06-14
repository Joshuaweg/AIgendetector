---
type: state
created: 2026-06-13
updated: 2026-06-13
last_edited_by: claude-sonnet-4-6
tags: [state, governance]
---

# Operational State

## Current Phase

**Active development — Mechanistic Interpretability / Per-Token TCAV**

Moving from ad-hoc attribution (Integrated Gradients) toward mechanistic understanding. Per-token TCAV is now implemented and ready to run. The key open question: do the 5 validated concept probes genuinely misalign with model decisions (sc ≈ 0.48), or is this a probe quality issue?

## Recent Work

- **2026-06-13**: Deep research session — TCAV + Mechanistic Interpretability for ViTs. Produced 8-section research report covering per-token TCAV, CIG, CRAFT/NMF, attention head analysis, activation patching, SAEs, optical flow pathway interpretation, and ACE concept discovery.
- **2026-06-13**: Implemented per-token TCAV in `tcav_interpret.py` — 5 new functions, `--per-token` flag, spatial concept heatmaps (8×8×12 patch grid → 512×512 Gaussian-smoothed PNGs matching IG sigma=12). Previous TCAV functions untouched.
- **2026-06-10**: Large session commit (39 files) — TCAV probing pipeline, concept discovery, `tcav_probes.py`
- **2026-03 to 2026-04**: Optical flow branch (FlowVideoClassifier, Stage 1 + Stage 2), async API refactor
- Model: `checkpoint_epoch_0004.pt` — current active weights (92.14% train, 85.12% test)

## Next Steps

- [ ] **Run per-token TCAV** — `python tcav_interpret.py --per-token ...` — inspect `_meta/tcav/{concept}/heatmaps/` outputs
- [ ] **Cross-reference heatmaps with IG** — do per-token concept hot-spots overlap with IG pixel attributions? If yes, concepts are real. If no, model uses different features.
- [ ] **Diagnose sc ≈ 0.48** — if heatmaps show no spatial structure, concepts genuinely don't align with model decisions → need probe redesign or automated concept discovery (ACE/NMF)
- [ ] **If concepts misalign**: run CRAFT (NMF on patch activations) to discover what the model actually uses, without manual probes
- [ ] **Review attribution caching in `api_server.py`** — deferred from previous session

## Blockers

Previous handoff noted a "sc computation bug" — on re-examination this is likely not a code bug (TCAV correctly uses a fixed test set for sc). The sc ≈ 0.48 plateau is more likely genuine concept misalignment. Per-token heatmaps will resolve this.

## Environment

- Model checkpoint: `model/checkpoint_epoch_0004.pt` (active, 1.1GB)
- API: Flask on port 5000 (behind Nginx in production)
- GPU: CUDA 12.1 required for training; inference can run CPU-only
- Flow cache: `flow_output/`, `flow_checkpoints/`
- Per-token TCAV output: `_meta/tcav/{concept}/heatmaps/segment_00.png` … `segment_11.png`

## aDNA Layer

Added 2026-06-13. Embedded triad at `.agentic/`. Starter conformance level.

---

## Related

**Project:** [[MANIFEST]] · [[CLAUDE]] · [[README]]

**Current work:** [[per_token_tcav_2026_06_13]] · [[per_token_tcav_plan]] · [[session_20260613_per_token_tcav]]

**History:** [[tcav-2026-06-13]] · [[unknown-1623532-snapshot]]

**Research:** [[detection_roadmap]] · [[INDEX_SAE_RESEARCH]] · [[integrated-gradients-baselines-research]]
