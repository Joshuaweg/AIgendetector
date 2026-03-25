# AI Video Detection Research Collection
**Complete: March 25, 2026**

This directory contains comprehensive research on AI-generated video detection across 8 strategic domains, covering state-of-the-art models, architectural improvements, and a 6-month implementation roadmap.

---

## Files in This Collection

### 1. **ai_video_detection_research.md** (Main Document — 1082 lines, 48 KB)
**The definitive research document.** Read this for comprehensive understanding.

**Contents:**
- Section 1: Current SOTA models (DIVID, AltFreezing, LSDA, etc.) with generalization analysis
- Section 2: Spatial/temporal/spectral fingerprints and compression robustness
- Section 3: Fine-tuning strategies (LoRA, EWC, few-shot adaptation)
- Section 4: Knowledge distillation preserving explainability
- Section 5: Neurosymbolic integration (Logic Tensor Networks for rule-based detection)
- Section 6: Advanced explainability (GradCAM, DINO, Concept Bottleneck, TCAV)
- Section 7: Architecture improvements with priority roadmap
- Section 8: Datasets and continual learning protocols
- Appendix: Quick reference tables

**Key Finding:** Temporal features generalize 2.5x better than spatial; explicit spectral features required.

**28 citations** from 2023-2026; includes architecture diagrams, comparison tables, hybrid frameworks.

---

### 2. **RESEARCH_SUMMARY.md** (Executive Overview — 7.5 KB)
**Start here if you have 10 minutes.** Executive summary of all 8 domains.

**Contents:**
- 1-paragraph summary of each research domain
- Key findings (what works, what doesn't, open gaps)
- 6-month implementation roadmap with expected gains
- Critical papers to read
- Implementation resources and GitHub links

**Use Case:** Share with stakeholders; technical leadership alignment.

---

### 3. **QUICK_DECISIONS.md** (Decision Tree — 8.8 KB)
**Use when you need to make NOW.** 10 common architectural/strategy decisions.

**Decisions Covered:**
1. How to handle new generators (LoRA vs. retraining)
2. Explainability method (GradCAM vs. Concept Bottleneck)
3. Optical flow vs. spectral branch first
4. Vision Foundation Models (DINOv2 backbone)
5. Single model vs. ensemble
6. Knowledge distillation target
7. Dataset collection priority
8. Neurosymbolic timing
9. Continuous benchmarking strategy
10. Success criteria for Phase 1-3

Each decision shows: options, pros/cons, recommendation, timing.

**Use Case:** Engineering team sprint planning; technical decisions in meetings.

---

## How to Use This Research

### For Leadership/Product
1. Read **RESEARCH_SUMMARY.md** (10 min)
2. Focus on: Key findings, 6-month roadmap, expected accuracy gains
3. Decide: Budget allocation across Phase 1-3

### For Engineering Team
1. Read **QUICK_DECISIONS.md** first (15 min) to understand priorities
2. Deep-dive **ai_video_detection_research.md** Section 7 (Architecture Improvements)
3. Refer to **RESEARCH_SUMMARY.md** Section on datasets for training protocol

### For Research/Grad Students
1. Start with **ai_video_detection_research.md** (full read, 1.5 hours)
2. Follow citations; access full papers via arXiv/conferences
3. Focus on Sections 2, 5, 6 for novel research directions

### For Implementation Sprint
1. Check **QUICK_DECISIONS.md** Decision #3 (optical flow vs. spectral)
2. Implementation guide in **ai_video_detection_research.md** Section 7
3. Training protocol in **ai_video_detection_research.md** Section 8

---

## Key Findings at a Glance

### What We Know Works
✅ Temporal metrics (flicker, jitter, optical flow residuals) — 2.5x better generalization  
✅ Spectral features (DCT, wavelets) — 99%+ accuracy if extracted properly  
✅ LoRA adapters — <2% parameters, 95% of full fine-tuning accuracy  
✅ Feature-level + attention KD — 85% explainability preservation (vs. 60% logit-only)  
✅ Optical flow explicit modeling — +3-7% accuracy gain expected  

### What Doesn't Work
❌ Spatial artifacts alone — 50% accuracy drop on unseen generators  
❌ Logit-only knowledge distillation — explainability collapses  
❌ Pure deepfake training → AI video transfer — poor generalization  
❌ Zero-shot transfer — even CLIP needs domain adaptation  

### Critical Gaps (Open Research)
🔴 Multimodal (audio+video) detection for synchronized dialogue  
🔴 Physics constraint learning (rigid body dynamics)  
🔴 Watermark removal artifact exploitation  
🔴 Real-time 60fps edge deployment  

---

## 6-Month Implementation Roadmap

| Phase | Timeline | Work Items | Expected Gain | Success Metric |
|---|---|---|---|---|
| **1** | Month 1 | Optical flow branch + LoRA infrastructure | +3-7% | Sora 45%→65% |
| **2** | Months 2-3 | Spectral branch + continual learning + concepts | +2-5% | Cross-gen 65%→80% |
| **3** | Months 4-6 | DINOv2 eval + Concept Bottleneck + neurosymbolic | +3-7% | Cross-gen 80%→87% |

**Current Baseline:** FF++ 92% | Sora (unseen) 45% | Cross-Gen Avg 65%  
**After Phase 3:** FF++ 96% | Sora (unseen) 85% | Cross-Gen Avg 87%

---

## Critical Papers (Must Read)

| Priority | Paper | Why | Minutes |
|---|---|---|---|
| 🔴 Must | "Turns Out I'm Not Real" (CVPR 2024) | DIVID baseline for Sora | 20 |
| 🔴 Must | "Video Forgery with Optical Flow" (2024) | Justifies optical flow choice | 15 |
| 🟠 Should | "AltFreezing" (CVPR 2023) | Architecture inspiration | 15 |
| 🟠 Should | "LoRA Recycle" (CVPR 2025) | Few-shot adaptation | 15 |
| 🟡 Nice | "KD in ViTs" (2023) | Distillation approaches | 20 |

**Total time to be proficient: 90 minutes**

---

## Integration with Your Current Model

Your model: **FullLatentEncoder** (3-layer CNN, 8x reduction) → **FullPatchEncoder** (2-frame 8×8 patches) → **FullClassifier** (12-layer Transformer)

### Recommended Phase 1 Addition
```
Input (T frames)
  ↓
├─ Spatial Branch (Your Current Model) → 768-dim
│  
└─ Temporal Branch (NEW)
   ├─ Optical flow estimation (FlowNet-style)
   └─ Feature extraction (magnitude + angle + residuals)
   → 384-dim
  
Fusion (1536-dim) → Transformer (12-layer, update d_model)
```

**Complexity:** Moderate | **Effort:** 2-3 weeks | **ROI:** High (+3-7% accuracy)

---

## Continuous Learning Protocol

**When new generator released (Sora 3, Veo 4, etc.):**

```
Day 1-2:   Collect 100-200 samples → Evaluate (<50% likely)
Week 1-2:  Train LoRA adapter on 200-500 samples
Week 3:    A/B test + Deploy (versioned as "v12-Sora3-LoRA")
Month 2-3: Merge knowledge into main model via periodic full training
```

**Target:** Full adaptation in 2 weeks; <5% forgetting of old generators.

---

## Data Requirements

**Minimum for Production:**
- 10K real videos (diverse: faces, objects, landscapes, text, animation)
- 10K synthetic per generator (Sora, Veo 3, Kling, Runway, Pika, OpenSora)
- Compression variants: original + YouTube H.264 + TikTok (if possible)

**Recommended Benchmarks:**
- FaceForensics++ (baseline)
- GenVidBench (6M clips, latest generators)
- Deepfake-Eval-2024 (real-world 2024 deepfakes)

---

## Explainability Roadmap

| Phase | Method | Speed | Quality | Implementation Complexity |
|---|---|---|---|---|
| Current | IntegratedGradients | 5-10s | High | Low (already deployed) |
| Phase 1 | GradCAM (parallel) | <1s | Medium | Low (1 week) |
| Phase 2 | Concept Bottleneck | <100ms | High | High (4 weeks) |
| Phase 3 | Neurosymbolic Rules | Real-time | Excellent | High (6 weeks) |

**Phased Rollout:** Keep IG for detailed reports; add GradCAM for real-time UI; migrate to concepts when ready.

---

## Open Questions for Further Research

1. **Multimodal Detection:** How to effectively combine audio-video signals for newer generators (Veo 3, Seedance)?
2. **Physics Constraints:** Can we formally encode rigid body dynamics as soft constraints in the loss function?
3. **Watermark Artifacts:** Do watermark removal techniques leave exploitable traces?
4. **Edge Deployment:** Can we achieve 60fps on mobile/embedded hardware?
5. **Adversarial Robustness:** How to defend against adaptive attacks targeting your specific detector?

---

## Files Location

```
/home/ubuntu/AIgendetector/_meta/research/
├── README.md (this file)
├── ai_video_detection_research.md (full document, 1082 lines)
├── RESEARCH_SUMMARY.md (executive summary, 7.5 KB)
└── QUICK_DECISIONS.md (decision reference, 8.8 KB)
```

All files are Markdown; open with any text editor or GitHub viewer.

---

## Research Methodology

This research was conducted systematically across 8 domains:

1. **Phase 1:** Anti-patterns first (what fails, why it fails)
2. **Phase 2:** State-of-the-art models and papers (2023-2026)
3. **Phase 3:** Technical deep-dives (fingerprints, architecture, training)
4. **Phase 4:** Synthesis into actionable roadmap

**Data Sources:** 28+ peer-reviewed papers, GitHub benchmarks, arXiv preprints, conference proceedings (NeurIPS, CVPR, ICCV, ECCV, WACV 2023-2026)

**Coverage:** Video generation (Sora, Veo, Kling, Runway, Pika, Seedance, OpenSora), deepfake detection (FF++, DFDC, WildDeepfake), forensic techniques (spectral, temporal, multimodal)

---

## Questions?

- **Architectural questions:** See Section 7 of main document
- **Implementation questions:** See QUICK_DECISIONS.md
- **Theoretical questions:** See Sections 1-2 and citations in main document
- **Roadmap questions:** See RESEARCH_SUMMARY.md or QUICK_DECISIONS.md

---

**Research Date:** March 25, 2026  
**Status:** Complete and ready for implementation planning  
**Next Review:** After Phase 1 completion (Month 1)

