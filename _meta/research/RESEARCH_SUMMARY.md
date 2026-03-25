# AI Video Detection Research — Executive Summary

**Document:** `/home/ubuntu/AIgendetector/_meta/research/ai_video_detection_research.md` (1082 lines)
**Date:** March 2026
**Status:** Complete; Ready for Implementation Planning

---

## 8 Research Domains Covered

### 1. Current SOTA Models (2023-2025)
- **DIVID** (Columbia): CNN+LSTM for diffusion videos, 93.7% in-domain
- **AltFreezing** (CVPR 2023): Spatial/temporal kernel freezing for generalization
- **LSDA** (CVPR 2024): Latent space augmentation for wider decision boundaries
- **Critical Finding:** 50% accuracy drop when detectors trained on one generator tested on unseen generator

### 2. Fingerprints: Spatial, Temporal, Spectral
- **Spatial:** Generator-specific (poor cross-gen); GAN checkerboard ≠ diffusion noise
- **Temporal:** 2.5x better generalization; optical flow residuals > raw flow
  - Key: Flicker, jitter, inter-frame coherence metrics work across Sora/Runway/Pika
- **Spectral:** DCT, wavelets, FFT; 99%+ accuracy if properly extracted
  - Compression survival: Temporal flicker (90%), optical flow (84%), DCT (73%), checkerboard (36%)

### 3. Fine-Tuning Without Catastrophic Forgetting
- **LoRA Adapters:** <2% parameters, 95% of full-tuning accuracy, <5% forgetting
- **Elastic Weight Consolidation:** Fisher information tracking; 10-15% improvement
- **Few-Shot:** LoRA Recycle meta-learning enables 1-shot adaptation (+9% vs. baseline)
- **Data Augmentation:** 30% jitter + 30% compression + 20% flow noise + 20% interpolation

### 4. Knowledge Distillation (Explainability-Preserving)
- **Standard KD (logit-only):** 60% IntegratedGradients correlation with teacher
- **Feature-Level KD:** 85% IG correlation; gradient-weighted importance matching
- **Attention Distillation:** Match attention maps; preserves saliency map quality
- **SpectralKD:** Layer-wise frequency analysis; reduces 12→8 layers with <2% accuracy loss

### 5. Neurosymbolic Integration
- **Vision:** Neural learning + explicit rules (optical flow physics, face anatomy)
- **Framework:** Logic Tensor Networks (LTN) for fuzzy first-order logic
- **Hybrid Architecture:** Neural path (your model) → Symbolic rules → Decision fusion
- **Example Rules:** 
  - Flow magnitude variance should follow natural distributions
  - Faces must blink at 0.1-0.4 Hz
  - Optical flow magnitude penalties for physics violations
- **Expected Gain:** +3-7% accuracy + full explainability

### 6. Advanced Explainability Beyond IntegratedGradients
| Method | Speed | Fidelity | Interpretability | Difficulty |
|---|---|---|---|---|
| **GradCAM** | 10x faster | Medium | Patch-level | Low |
| **DINO Attn** | Fast (inherent) | High | Semantic patch | Medium |
| **Concept Bottleneck** | Fast (inference) | High | Named concepts | High |
| **TCAV** | Medium | Medium | Post-hoc concepts | Medium |

- **Concept Bottleneck Recommendation:** 15-20 forensic concepts (boundary blur, temporal jitter, etc.)
- **Phased Rollout:** Phase 1 (keep IG + caching), Phase 2 (GradCAM alternative), Phase 3 (Concept model)

### 7. Architecture Improvements
**Priority 1 (2-3 weeks, +3-7%):** Optical Flow Branch
- FlowNet-style CNN → 2-channel optical flow at FullLatentEncoder scale
- Extract magnitude + angle + residual (2nd-order temporal derivative)
- Fusion with spatial branch before Transformer

**Priority 2 (4-6 weeks, +2-5%):** Spectral Branch
- DCT path: Block-wise DCT → CNN (384-dim)
- Wavelet path: Multi-level DWT → CNN (384-dim)
- Fusion: Concat (768-dim) + linear projection → Late fusion with spatial
- Transformer input grows to 1536-dim

**Priority 3 (Prototype):** DINOv2 Backbone
- Replace FullLatentEncoder with pre-trained DINOv2-Base
- Gain: Better generalization, inherent interpretability
- Cost: 1.5-2x slower inference
- ROI: Validate first

**Priority 4 (Lower):** Multi-Scale Patches
- 4×4 (details) + 8×8 (current) + 16×16 (global)
- Gain: +1-3% accuracy
- Cost: Transformer must handle 2-3x more tokens

### 8. Datasets & Continual Learning
**Minimum for Production:**
- 10K real videos (diverse: faces, objects, landscapes, text)
- 10K synthetic from each: Sora, Veo 3, Kling, Runway, Pika, OpenSora
- Compression variants: original, YouTube H.264, TikTok

**Latest Benchmarks:**
- **FaceForensics++:** Standard baseline (1000 originals + 4 face methods)
- **GenVidBench:** 6M clips, all new generators, cross-compression testing
- **Deepfake-Eval-2024:** Real-world 2024 deepfakes; watermarks, social-media re-encoding
- **VideoDiffusion:** 10K+ videos from 5 diffusion models; generalization focus

**When New Generator Released (Sora 3, Veo 4, etc.):**
1. **Day 1-2:** Collect 100-200 samples; assess accuracy (<50% likely)
2. **Week 1-2:** Train LoRA adapter (200-500 samples sufficient)
3. **Week 3:** A/B test + deploy with versioning
4. **Month 2-3:** Merge into main model via periodic full training

---

## Key Findings Summary

### What Actually Works
1. ✅ **Temporal metrics** generalize 2.5x better across generators
2. ✅ **Spectral features (DCT, wavelet)** are generator-agnostic if explicitly extracted
3. ✅ **Optical flow residuals** (2nd-order) more robust than raw optical flow
4. ✅ **LoRA adapters** effective for new generators without catastrophic forgetting
5. ✅ **Feature-level + attention KD** preserves explainability vs. logit-only KD

### What Doesn't Work (or Fails)
1. ❌ **Spatial artifacts alone** — fail on unseen generators (GAN checkerboard ≠ diffusion noise)
2. ❌ **Logit-only knowledge distillation** — IG correlation drops to 60%
3. ❌ **Pure deepfake datasets** — poor transfer to AI-generated video
4. ❌ **Zero-shot transfer** — even foundation models (CLIP) need domain adaptation
5. ❌ **Frame-level detection** — must use sequence/temporal context

### Critical Gaps (Open Research)
1. 🔴 **Multi-modal (audio+video) detection** — synchronized dialogue in newer generators
2. 🔴 **Physics constraint learning** — how to formally encode rigid body dynamics?
3. 🔴 **Watermark artifacts** — do watermark removal methods introduce detectable traces?
4. 🔴 **Real-time inference** — can we detect at 60fps on mobile/edge?

---

## Recommended 6-Month Roadmap

| Phase | Timeline | Items | Expected Gain |
|---|---|---|---|
| **Month 1** | Week 1-4 | Optical flow branch + LoRA infrastructure | +3-7% |
| **Month 2-3** | Week 5-12 | Spectral branch + continual learning | +2-5% |
| **Month 4-6** | Week 13-24 | DINOv2 eval + Concept Bottleneck + Neurosymbolic | +3-7% |

**Total Expected Improvement:**
- **Current:** FF++ 92%, Sora (unseen) 45%, Cross-Gen Avg 65%
- **After 6 Months:** FF++ 96%, Sora (unseen) 85%, Cross-Gen Avg 87%

---

## Critical Papers & Implementations

### Must Read (2024-2025)
1. **Turns Out I'm Not Real** (Columbia) — DIVID, Sora detection baseline
2. **DeepfakeBench** (NeurIPS 2023) — Unified benchmark
3. **What Matters in Detecting AI-Generated Videos** — Artifact analysis for Sora
4. **Video Forgery Detection with Optical Flow Residuals** — Temporal consistency focus
5. **AltFreezing** (CVPR 2023) — Spatial/temporal architecture design

### Implementation Resources
- **DeepfakeBench GitHub:** https://github.com/SCLBD/DeepfakeBench
- **LoRA Library:** https://github.com/microsoft/LoRA
- **Logic Tensor Networks:** https://github.com/logictensornetwork/logictensornetwork
- **DINO/DINOv2:** https://github.com/facebookresearch/dino

---

## File Location
**Full Document:** `/home/ubuntu/AIgendetector/_meta/research/ai_video_detection_research.md`

Contains 28 detailed citations, 8 architecture diagrams (text), 6 comparison tables, and 3-month action roadmap.

