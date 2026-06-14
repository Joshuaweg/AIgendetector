# CLAUDE.md — AIgendetector

## Identity

This is **AIgendetector**: a production-grade deep learning system for detecting AI-generated video content. It achieves 85.12% test accuracy using Vision Transformers with optical flow analysis, ships a Flask REST API for frontend integration, and includes interpretability tooling (Integrated Gradients, TCAV, Sparse Autoencoders) for understanding model decisions.

Purpose: forensic analysis and authenticity verification of video content. Active research + deployed production system.

---

## Structure

```
AIgendetector/
├── CLAUDE.md           ← you are here (agent entry point)
├── MANIFEST.md         ← project overview, architecture, active builds
├── STATE.md            ← current operational state, next steps
├── .agentic/           ← aDNA governance layer
│   ├── who/            ← people, teams, collaborators
│   ├── what/           ← knowledge, experiments, decisions
│   └── how/            ← sessions, missions, pipelines
├── _meta/              ← KERNEL metadata (do not modify)
├── model/              ← trained checkpoints (do not delete)
├── api_server.py       ← Flask REST API (987 lines)
├── full_scale_classifier.py  ← model architecture
├── dataset.py          ← data loading + optical flow
└── ...                 ← all other source files
```

---

## Agent Protocol

1. Read `CLAUDE.md` (auto-loaded)
2. Read `STATE.md` for current operational context
3. Read `AGENTS.md` in the directory you're working in
4. For architecture questions: read `MANIFEST.md`

---

## Safety Rules

- **Read before write** — always read a file before modifying it
- **Never touch `_meta/`** — this is KERNEL's directory
- **Never delete `model/`** — checkpoints are irreplaceable without retraining
- **Set `updated` and `last_edited_by`** on any aDNA triad file you edit
- **Don't modify deployment configs** (`docker-compose.yml`, `nginx.conf`, `deploy/`) without explicit instruction

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core model | PyTorch 2.1 + CUDA 12.1, Vision Transformer (12-layer, 12-head attention) |
| Feature extraction | LatentEncoder, PatchEncoder, optical flow (Farneback) |
| Interpretability | Captum (Integrated Gradients), TCAV, Sparse Autoencoders |
| API | Flask 3.0, Gunicorn, async attribution pipeline |
| Video processing | OpenCV, imageio + ffmpeg |
| Deployment | Docker + Docker Compose, Nginx, systemd, AWS SageMaker |
| Frontend | Next.js (TypeScript), TailwindCSS, Axios |

---

## Key Files

| File | Purpose |
|------|---------|
| `full_scale_classifier.py` | Main classifier architecture (FullVideoClassifier, FlowVideoClassifier) |
| `api_server.py` | Flask REST API with async attribution pipeline (987 lines) |
| `dataset.py` | Data loading, optical flow computation (883 lines) |
| `feature_extractor.py` | Feature analysis tools (720 lines) |
| `interpret.py` | Integrated Gradients visualization (467 lines) |
| `tcav_interpret.py` | TCAV probing (614 lines) |
| `camera_forensics.py` | Forensic analysis tools (1548 lines) |
| `model/checkpoint_epoch_0004.pt` | Active model weights (1.1GB, 92.14% train accuracy) |

---

## Domain Context

- **Classification task**: Real vs. AI-Generated video (binary)
- **Optical flow**: Temporal motion patterns computed via Farneback algorithm; key signal for detecting generation artifacts
- **TCAV**: Testing with Concept Activation Vectors — probes what semantic concepts the model uses
- **Sparse Autoencoders (SAE)**: Decompose model internals into interpretable features
- **Integrated Gradients**: Attribution method showing which input regions drive predictions

---

## aDNA Layer

This project uses the **embedded triad** form. aDNA governance lives in `.agentic/` alongside the codebase. For sessions, missions, and knowledge tracking, work inside `.agentic/how/`, `.agentic/what/`, and `.agentic/who/` respectively.
