---
type: manifest
created: 2026-06-13
updated: 2026-06-13
last_edited_by: agent_init
tags: [manifest, governance]
---

# AIgendetector

## What This Is

A deep learning system for detecting AI-generated video content, achieving 85.12% test accuracy via Vision Transformers with optical flow analysis. Ships as a production Flask API with Docker deployment, AWS SageMaker training support, and interpretability tooling (TCAV, Integrated Gradients, Sparse Autoencoders).

## Architecture

```
Video input → OpenCV frame extraction → FullVideoClassifier (ViT + optical flow)
           → Flask API (api_server.py) → JSON response (real/fake + confidence)
           → Attribution pipeline (async) → Integrated Gradients heatmaps
```

Training: local GPU or AWS SageMaker (`sm_train_v3.py`).
Deployment: Docker Compose (API + Nginx) or systemd service.

## Entry Points

| Audience | Start here |
|----------|-----------|
| Agents | `CLAUDE.md` |
| Developers | `README.md` → `QUICKSTART.md` |
| Deployment | `DEPLOYMENT.md` |
| Research | `EXPERIMENTS_SUMMARY.md` → `EXPERIMENTS_GUIDE.md` |

## Key Components

| Component | Files | Status |
|-----------|-------|--------|
| Core classifier | `full_scale_classifier.py`, `dataset.py` | Production |
| Optical flow branch | `flow_train.py`, `FlowVideoClassifier` | Active (Stage 2) |
| Flask API | `api_server.py` | Production |
| Interpretability | `interpret.py`, `tcav_interpret.py`, `train_sae.py` | Active (per-token TCAV + SAE research) |
| Forensic analysis | `camera_forensics.py`, `spectral_analysis.py` | Complete |
| Deployment | `docker-compose.yml`, `Dockerfile`, `deploy/` | Production |
| SageMaker training | `sm_train_v3.py` | Available |

## Active Builds

- **Per-token TCAV** — June 2026, spatial concept heatmaps over 8×8×12 patch grid; `--per-token` flag in `tcav_interpret.py`. See `_meta/plans/per_token_tcav_plan.md`.
- **Mechanistic Interpretability roadmap** — research complete; 4-phase plan: per-token TCAV → CIG/CRAFT → attention circuits → SAEs. See `_meta/handoffs/per_token_tcav_2026_06_13.md`.
- **Optical flow Stage 2** — FlowVideoClassifier complete; dual-pathway TCAV analysis wired in.

## Performance

| Metric | Value |
|--------|-------|
| Test accuracy | 85.12% |
| Training accuracy | 92.14% (epoch 4) |
| Active checkpoint | `model/checkpoint_epoch_0004.pt` (1.1GB) |

---

## Related

**Entry points:** [[CLAUDE]] · [[README]] · [[QUICKSTART]] · [[DEPLOYMENT]]

**Interpretability:** [[per_token_tcav_2026_06_13]] · [[tcav-2026-06-13]] · [[INDEX_SAE_RESEARCH]] · [[sae-vision-transformer-research]] · [[integrated-gradients-baselines-research]] · [[SAE_INTERPRETABILITY]] · [[SAE_QUICKSTART]]

**Research:** [[detection_roadmap]] · [[RESEARCH_SUMMARY]] · [[ai_video_detection_research]] · [[frontier-model-detection-research]] · [[mvad-dataset-research]] · [[flow_branch_plan]] · [[vivit-tubelet-embeddings]] · [[EXPERIMENTS_SUMMARY]] · [[EXPERIMENTS_GUIDE]]

**Operations:** [[STATE]] · [[per_token_tcav_plan]] · [[session_20260613_per_token_tcav]] · [[inference_serving_reconnaissance]] · [[aws-cost-optimization-research]]

**Forensics:** [[SPECTRAL_ANALYSIS_README]] · [[APP_SUMMARY]]
