# Quick Decision Reference — AI Video Detection Architecture

Use this when deciding between competing approaches.

---

## Decision 1: How to Handle New Generators (Sora 3, Veo 4, etc.)?

**Scenario:** New generator released; detection accuracy <50%

**Option A: Full Retraining** ❌
- Retrain entire 12-layer model from scratch
- Cost: 2-3 weeks
- Risk: Catastrophic forgetting of old generators
- Use only if: Fundamental change in generator architecture

**Option B: LoRA Adapter (Recommended)** ✅
- Train 2% additional parameters in attention layers
- Cost: 2-3 days
- Risk: Minimal forgetting
- Maintenance: Each generator gets small adapter file (2-3 MB)
- Decision: Choose this unless new generator is completely novel

**Option C: Fine-tune with EWC** 🟡
- Fine-tune last 3 layers; track Fisher information
- Cost: 1 week
- Risk: Some forgetting; reduced by Fisher penalty
- Use if: LoRA not sufficient (validate first)

**Recommendation:** Start with LoRA (Option B). If accuracy doesn't recover to >80%, then use EWC (Option C).

---

## Decision 2: Explainability Method

**Scenario:** Need to explain to stakeholder "why is this fake?"

**Decision: Integrated Gradients (IG) exclusively** ✅

Classification and IG evidence generation run on **separate async streams**, so IG latency is not a bottleneck. The frontend receives the classification result first, then the IG heatmap arrives independently. This removes the original motivation for GradCAM (real-time feedback) while retaining IG's theoretical guarantees (completeness, sensitivity).

**Performance headroom:** IG steps can be parallelized later by increasing `internal_batch_size` in `ig.attribute()` — currently set to 1 in most places, can be tuned up based on available VRAM.

**GradCAM: not used.** Violated completeness axiom; less meaningful for fine-grained texture artifact detection in AI-generated content.

**Option C: Concept Bottleneck** 🟢 (more feasible than previously estimated)
- Pro: Can say "detected boundary blur + optical flow anomaly"; highest user trust
- **Revised Con:** Previously estimated 15-20 human-labeled concepts as blocking effort.
  This is no longer true — the project already has discrete forensic tools
  (`spectral_analysis.py`, `camera_forensics.py`, `diffusion_fingerprints.py`) whose
  scalar outputs directly map to concept nodes. No labeling effort required.
- **Gap:** Dense optical flow and face anatomy not yet implemented. Everything else covered.
- Decision: Wire forensic tools as a parallel concept stream alongside IG in Phase 2.

**Recommendation:**
1. **UI Shows:** IG heatmap via async stream (no GradCAM fallback needed)
2. **Optimization Path:** Increase `internal_batch_size` when VRAM allows
3. **Phase 2:** Add concept stream — forensic tool outputs as named scalar scores alongside IG

---

## Decision 3: Spectral vs. Temporal Branch First?

**Scenario:** Budget for ONE architectural change this quarter

**Option A: Optical Flow (Temporal) Branch First** ✅
- Gain: +3-7% accuracy
- Effort: 2-3 weeks
- Complexity: Moderate (requires flow model)
- Generalization: Excellent (works across generators)
- Recommendation: **Choose this first**

**Option B: Spectral (DCT+Wavelet) Branch First** 🟡
- Gain: +2-5% accuracy
- Effort: 4-6 weeks
- Complexity: High (two parallel paths)
- Generalization: Good (but fewer cross-gen studies)
- Recommendation: Do this in Phase 2

**Recommendation:** Optical flow first. It's faster, better-studied, and higher ROI. Add spectral in next iteration.

---

## Decision 4: Vision Foundation Model Backbone?

**Scenario:** Should we replace FullLatentEncoder with DINOv2?

**Yes (✅) if:**
- Generalization to new generators is critical
- Inference latency <2s acceptable for your product
- You have budget to experiment (2 week prototype)

**No (❌) if:**
- Real-time inference required (<500ms per video)
- Your current FullLatentEncoder + patches already working well
- Unknown new issues from switching

**Recommendation:** Prototype in parallel (~2 weeks). If top-1 accuracy beats current model AND cross-generator accuracy improves >5%, migrate. Otherwise stay with current.

---

## Decision 5: Single Model vs. Ensemble?

**Scenario:** Should we detect Sora differently than Veo?

**Option A: Single Model + LoRA Adapters** ✅
- One base model; different LoRA for each generator
- Pro: Easy to deploy; A/B testable
- Con: Still ~500MB base model
- Recommendation: **Choose this** for production

**Option B: Generator-Specific Ensemble** 🟡
- Different models optimized for different generators
- Pro: Marginally higher accuracy per generator
- Con: Complex infrastructure; 5-10x model size; slow inference
- Use only if: Single model <70% accuracy per generator

**Recommendation:** Start with single model + LoRA. Migrate to ensemble only if accuracy insufficient.

---

## Decision 6: Knowledge Distillation Target?

**Scenario:** Need faster inference for mobile/edge

**Option A: Keep Current 12-Layer Model** ✅
- Current inference: ~800ms (1 GPU frame @ 1080p)
- Recommendation: **Keep for flagship product**

**Option B: Distill to 8 Layers** 🟡
- Target inference: ~500ms
- Accuracy loss: <2% (with feature-level + attention KD)
- Explainability loss: <15% (vs. logit-only KD)
- Use if: Mobile app required

**Option C: Distill to 4 Layers** ❌
- Target inference: ~250ms
- Accuracy loss: >5%
- Explainability: Degraded significantly
- Recommendation: Not recommended; skip this

**Recommendation:** Keep 12-layer for server; optionally distill to 8-layer for mobile app (if demand exists).

---

## Decision 7: New Dataset Priority?

**Scenario:** What data to collect first for improved training?

**Top Priority (Collect First):** ✅
- Sora (1000s of videos): Rapidly becoming most common generator
- Veo 3 (500+): Google; strong market position
- Kling (500+): Chinese market leader; temporal consistency breakthrough

**Secondary Priority:** 🟡
- Runway Gen-2 (200+): Established; good benchmark
- Pika (200+): Motion-focused; useful for optical flow testing

**Lower Priority:** 🔵
- Old GAN deepfakes (you have FF++): Returns diminishing
- Synthesia/AI presenters: Niche use case

**Compression Variants (Always Add):** ✅
- Original + YouTube H.264 (most common path)
- TikTok compression if possible (harder to obtain)

**Recommendation:** Prioritize Sora + Veo 3 + Kling. Spend 70% budget on new generators; 30% on compression variants.

---

## Decision 8: Neurosymbolic Now or Later?

**Scenario:** How urgently to implement symbolic rules?

**Option A: Defer (Continue Pure Neural)** ✅ for now
- Current model works; explainability via IG acceptable
- Risk: "Black box" criticism grows as model ages
- Timeline: Revisit after Phase 2 improvements

**Option B: Parallel Research Track** 🟡
- Spend 10% team bandwidth researching LTN for video
- Design rules for optical flow physics, face anatomy
- Timeline: 2-3 month research; production in Phase 3

**Option C: Implement Neurosymbolic Now** ❌
- Would delay optical flow + spectral branch work
- High risk; unproven for video detection
- Recommendation: Not recommended in current timeline

**Recommendation:** Option B (parallel research track). Get Phase 1-2 improvements out first; neurosymbolic becomes Phase 3 differentiator.

---

## Decision 9: Continuous Benchmarking Against What?

**Scenario:** How to track progress across improvements?

**Benchmark Set (Recommended):** ✅
1. **In-Domain:** FF++, DFDC (training generators)
2. **Cross-Generator:** Sora, Veo, Kling (unseen during training)
3. **Compression:** Original + YouTube H.264 variants
4. **Real-World:** Deepfake-Eval-2024 (actual deepfakes in the wild)

**Metrics to Track:**
- Accuracy per generator (avoid averaging; per-generator is real metric)
- Generalization gap (in-domain accuracy - cross-gen accuracy); target <10%
- Compression robustness (accuracy drop on H.264; target <15%)
- Inference time (on target hardware; track as architecture grows)

**Reporting:** Weekly dashboard showing 4-8 metrics across these sets.

---

## Decision 10: When to Declare Victory?

**Scenario:** How to know Phase 1-2-3 improvements are successful?

**Phase 1 (Optical Flow) Success Criteria:** ✅
- FF++ accuracy: ≥94% (from 92%)
- Sora accuracy (unseen): ≥65% (from 45%)
- Inference time: ≤900ms (from 800ms; acceptable overhead)
- Explainability: IG correlation ≥85% with original model

**Phase 2 (Spectral + Continual Learning) Success Criteria:** ✅
- Cross-generator average: ≥80% (from 65%)
- New generator adaptation time: ≤2 days with LoRA
- Concept bottleneck interpretability: >90% user trust (measure via user study)

**Phase 3 (Neurosymbolic + Advanced Methods) Success Criteria:** ✅
- Cross-generator average: ≥87% (from initial 65%)
- Generalization gap: <8% (in-domain - cross-gen)
- Explainability: Stakeholders can trace decisions to named concepts + rules

**Recommendation:** Stop when Phase 2 criteria met. Phase 3 is "nice to have" if time permits.

---

## Quick Reference: Which Papers to Read First?

| Paper | Reason | Read Time |
|---|---|---|
| "Turns Out I'm Not Real" (CVPR 2024) | Sora detection baseline; SOTA | 20 min |
| "Video Forgery with Optical Flow Residuals" (2024) | Justifies optical flow priority | 15 min |
| "AltFreezing" (CVPR 2023) | Architecture inspiration | 15 min |
| "LoRA Recycle" (CVPR 2025) | Few-shot adaptation | 15 min |
| "Knowledge Distillation in ViTs" (2023) | Distillation approaches | 20 min |

**Total time to be conversant: 90 minutes**

---

**Last Updated:** March 25, 2026
**For Questions:** Refer to full research document at `/home/ubuntu/AIgendetector/_meta/research/ai_video_detection_research.md`

