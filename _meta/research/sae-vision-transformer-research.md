# Sparse Autoencoders for Vision Transformers: Comprehensive Research Guide

**Research Date:** March 2026  
**Focus:** SAE Training, Feature Interpretation, and Application to Video Classification  
**Status:** Active Research Area with Mature Methodology (2024-2026)

---

## PITFALLS & COMMON FAILURES (START HERE)

### 1. Max-Activating Examples Alone Are Misleading
**Problem:** Simply showing the image patches where SAE features activate most strongly often fails to reveal what actually *causes* the activation.

**Why It Fails:** Self-attention mixes information across the entire image during processing. An activated patch often co-occurs with—but does not causally drive—feature activation.

**Solution:** Use **Effective Receptive Field (ERF)** with input attribution methods:
- Apply gradient-based attribution (e.g., integrated gradients) to identify which input patches have highest *causal contribution*
- Compare ERF-ranked patches vs. activation-ranked patches
- Recent work shows ERF-ranked patches recover activations more effectively, especially for non-localized features
- Implementation: See "Causal Interpretation of Sparse Autoencoder Features in Vision" (arXiv:2509.00749)

---

### 2. Dead Latents Without Proper Loss Scheduling
**Problem:** Training collapses to using only a subset of learned features; most latents remain inactive.

**Why It Fails:** ReLU SAE requires balancing reconstruction loss against L1 sparsity penalty, but fixed L1 coefficient leads to either:
- Too high L1 → most latents dead, high sparsity
- Too low L1 → poor sparsity, reconstruction-focused, features become entangled

**Solution:** Use L1 Warm-up + TopK as alternatives:
- **L1 Warm-up:** Start with zero L1 penalty, linearly increase over first 5% of training steps
- **TopK SAE:** Replaces L1 + ReLU with explicit top-k selection (e.g., keep k=32 active per sample)
  - Directly controls sparsity without tuning
  - Better feature quality than ReLU (higher probe scores)
  - More seed-dependent, requires careful initialization
- **Initialization:** Always use geometric_median or mean for decoder bias (mean faster but suboptimal)
- **Auxiliary Loss:** OpenAI's auxiliary loss (reconstruct residual of main reconstruction) prevents dead latents robustly

See "Scaling and Evaluating Sparse Autoencoders" (arXiv:2406.04093) for benchmarks; TopK SAEs shown ~15-20% better reconstruction-sparsity frontier.

---

### 3. Wrong Hook Point Gives Low-Quality Features
**Problem:** Training SAE on wrong layer activations yields features that don't compose well, are hard to interpret, or don't causally affect output.

**Why It Fails:** Different layers process information differently:
- **Attention output (Z):** Low effective rank (~2x lower than residual stream); most info already collapsed
- **MLP output:** Sparse, harder to decompose
- **Residual stream:** Sweet spot for capturing composed features, directly tracks information flow

**Solution:** For vision transformers:
- **Primary choice:** Train on residual stream (post-attention, post-MLP) of intermediate layers
- **For CLIP-style vision transformers:** 
  - Extract from class token ([CLS]) residual stream for global semantics
  - Extract from patch token residuals for patch-level concepts
  - Middle layers (8-10 for ViT-B) show best disentanglement (CVPR 2025 mechanistic interpretability work)
- **Why middle layers:** Earlier layers = low-level features; later layers = entangled task-specific; middle = best interpretable compositionality
- Attention outputs work but require SAE to be 2-3x larger to match quality (lower effective rank compresses feature space)

---

### 4. Insufficient Data/Activation Coverage During Training
**Problem:** SAE trained on small dataset or single distribution learns brittle features; fails on new images/videos.

**Why It Fails:** Features learn to be highly specific to training distribution; activation patterns don't generalize.

**Solution:**
- **Training batch size:** 4096 tokens standard; batch_size x context_window = total activations per step
- **Training tokens:** Aspire for 40B+ tokens (Anthropic uses for GPT-4 SAEs)
- **Data diversity:** If training on domain-specific data (e.g., video classification), include diverse backgrounds, lighting, object poses
- **For video:** Include frames from multiple generators, compression levels, temporal ranges

---

### 5. Confusing Sparsity Metrics (L0 vs. L1 coefficient)
**Problem:** ReLU SAE with L1 coefficient λ=0.001 produces wildly different L0 sparsity across models; hyperparameter transfer fails.

**Why It Fails:** Relationship between L1 coefficient and actual sparsity is nonlinear and model-dependent; no universal tuning.

**Solution:**
- Use **TopK SAE** if you need predictable sparsity (directly set k, then L0 = k)
- If using ReLU: employ **autotuning controller**—dynamically adjust λ multiplier to hit target L0
- Benchmark: L1 coefficient ~10 for Gemma-2-2b with L0~50; use as starting point only
- Monitor L0 during training; adjust λ every N steps to stay near target

---

## RECOMMENDED APPROACH: SAE Architecture & Training

### Best Overall Architecture: **TopK SAE** (for vision)

**Why TopK over ReLU:**
- Directly controllable sparsity (just set k)
- ~10-15% better feature quality (higher probe scores in SAEBench)
- Cleaner scaling laws w.r.t. model/SAE size
- No dead latent penalty tuning needed

**Architecture:**
```
Input (residual stream) → Linear Encoder → TopK Selection → Linear Decoder → Reconstruction
                          (d_model → d_latent)  (keep top k)  (d_latent → d_model)
```

**Why TopK vs. Gated SAE:**
- Gated SAE (separate gate/mag networks) adds complexity; TopK simpler
- Gated SAE better for highly entangled features; vision features moderate entanglement
- TopK proven on CLIP, ViT-B, other vision models (2024-2025 work)

---

### Where to Hook (Vision Transformers)

**Recommended Multi-Layer Strategy:**
1. **For global concepts (classification):** Extract from [CLS] token residual stream of layer 10-11 (in 12-layer ViT-B)
2. **For spatial concepts (segmentation/localization):** Extract from all patch tokens of layer 9-10
3. **For temporal video features:** Extract from all frame tokens of middle layers (avoid early + late)

**Single-Layer vs. Multi-Layer SAEs:**
- **Single-layer SAE per layer** (standard): Easy to train, fast inference
- **Multi-layer SAE** (emerging): Train one SAE on all layers simultaneously
  - Reveals features active across multiple layers
  - Captures information flow better
  - Slower training, larger SAE needed
  - Use if studying concept composition; skip for production

**For CLIP specifically:**
- ViT-B-32: Extract from layer 9-11 residual stream
- ViT-L-14: Extract from layer 20-23 residual stream
- Spatial tokens ~10x more active than [CLS] token (different sparsity patterns vision vs. language!)

---

## FEATURE LABELING & INTERPRETATION

### Methodology: Max-Activating Examples + Causal Attribution

**Step 1: Gather Activations**
```
For each SAE latent i:
  - Run forward pass on training dataset (or larger validation set)
  - Record activation values (magnitude of latent code)
  - Identify top-k images/patches where latent activates highest
```

**Step 2: Extract Max-Activating Patches**
```
For top-k activations:
  - Extract the patch (or receptive field) around maximum activation
  - Visualize as image gallery
  - Collect ~10-20 exemplars per feature
```

**Step 3: Causal Attribution (ERF Method)**
- Apply gradient-based input attribution to each activated latent
- For each high-activation example:
  - Compute ∂latent_i / ∂input (via backprop)
  - Mask gradients below threshold to find critical patches
  - Record which patches *causally* drive activation (not just spatially present)
- **Why:** A feature might activate on "blue feathers" but only if eyes visible; ERF reveals this dependency

**Step 4: Automated Labeling (CLIP-Dissect approach)**
```
Option A: CLIP-Dissect (2023 ICLR):
  - For each latent's top-k images, generate CLIP embeddings
  - Match against concept vocabulary (e.g., "fur", "metal", "blur")
  - Return top-k matching concepts
  - Computationally efficient (~4 min for ResNet-50 5 layers)

Option B: LLM Description (recent):
  - Feed max-activating images + top patches to CLIP-text encoder
  - Generate natural language description via LLM
  - Have simulator model verify description (RAVEL metric)

Option C: Manual Inspection (small SAEs):
  - For <1000 features, manual labeling still practical
  - Use interactive visualization (Neuronpedia-style)
```

**Implementation Recommendation:** Combine max-activating examples (quick) + ERF (thorough) + CLIP-Dissect (automated). Total time: <1 hr per 1000 features.

---

## TRAINING RECIPE FOR VISION TRANSFORMERS

### Hyperparameters (Vision-Optimized)

```
Architecture: TopK SAE (or Gated SAE for extra caution)

d_model: [768, 1024]       # ViT hidden dimension
d_latent: [4*d_model, 16*d_model]  # Typical: 3K-16K latents
k_topk: 32-64              # Sparsity; vision = lower than language
                            # (vision tokens ~2-5x sparser than language)

Optimizer: AdamW
Learning rate: 4e-4        # Can start here; monitor loss
LR scheduler: Cosine annealing + 5% linear warmup + 5% linear decay

Batch size: 4096 tokens    # Standard; no clear benefit to varying
L1 coefficient: Start 0, warm up to 2e-3 over first 5% training
Auxiliary loss: Yes        # Prevents dead latents
Decoder bias init: geometric_median (or mean if speed critical)

Training steps: 40B tokens / context_length (e.g., 10K steps for 4M tokens/step)
                = ~10-50 hours on single GPU for vision SAE
```

### Training Procedure

1. **Freeze vision transformer weights** (don't fine-tune backbone)
2. **Extract activations** (residual stream from chosen layer)
3. **Train SAE with auxiliary loss + L1 warm-up:**
   - Gradient projection technique to stabilize training dynamics
   - Monitor L0 sparsity every 100 steps
   - If L0 > target + 10%: increase λ by 10%
   - If L0 < target - 10%: decrease λ by 5%
4. **Validate** on held-out images (not used during training)
5. **Evaluate:** KL divergence, reconstruction loss, probe accuracy

### Expected Performance (Benchmarks)

- **Reconstruction KL divergence:** <200 nats (TopK SAE)
- **L0 sparsity:** 20-50 features per sample (vision is sparser than language)
- **Monosemanticity:** ~60-70% of features have clean semantic interpretation
- **Training time:** 24-72 hours on single A100 for 1M+ latent SAE

---

## APPLICATION TO VIDEO CLASSIFICATION

### Key Differences from Static Images

1. **Temporal Dimension:** Video transformers process frame sequences
   - Extract activations from all frame tokens (don't just use [CLS])
   - Temporal features are lower-sparsity than spatial (tokens co-activate across frames)
   - Hook from layers 9-12 for mid-level temporal patterns

2. **Optical Flow Artifacts:** SAE features may learn:
   - Motion boundaries (edges with temporal gradient)
   - Flow consistency violations (temporal incoherence artifact)
   - Subject tracking (object identity across frames)
   - Useful for AI-generated video detection (your use case!)

3. **Computational Cost:** Video SAE = larger than image SAE
   - Patch count grows O(num_frames × spatial_patches)
   - Use batching carefully; consider gradient checkpointing
   - Multi-layer extraction feasible only if <10 frames per sample

### Training Data for Video Classification

- **Synthetic videos:** Generated by Sora, Runway, Pika, Kling, etc.
- **Real videos:** High-quality authentic footage
- **Distribution:** Balance real/fake; include compression artifacts (H.264, VP9)
- **Recommendation:** Start with CLIP's [CLS] token features (frame-level); graduate to temporal-token features if needed

### Interpretation Strategy for Video SAEs

1. **Spatial interpretation:** Standard (max-activating patches per frame)
2. **Temporal interpretation:**
   - Visualize activation over time; show frame sequences where feature fires
   - Check temporal coherence: does feature activate on impossible motion?
   - For detection: flag features that activate on unnatural flow patterns
3. **Causal probing:**
   - Insert/remove feature via decoder manipulation
   - Measure impact on classification logits
   - For synthetic detection: probe which features matter for distinguishing real vs. AI-generated

---

## PAPERS & RESOURCES

### Core Papers (2024-2026)

**Foundational SAE Work:**
- [Scaling and Evaluating Sparse Autoencoders (2024)](https://arxiv.org/abs/2406.04093) — OpenAI/Anthropic benchmark; TopK vs ReLU comparison; 16M latent SAE on GPT-4
- [Towards Monosemanticity: Decomposing Language Models with Dictionary Learning (2023)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Anthropic; foundational SAE theory
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Large-scale SAE application

**Vision-Specific SAE Papers:**
- [Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models (2025)](https://arxiv.org/abs/2502.06755) — SAEV package; real-image exemplars + causal edits; multi-task evaluation
- [Sparse autoencoders reveal selective remapping of visual concepts during adaptation (2024)](https://arxiv.org/abs/2412.05276) — PatchSAE on CLIP; concept adaptation analysis; patch-level spatial attribution
- [Causal Interpretation of Sparse Autoencoder Features in Vision (2025)](https://arxiv.org/html/2509.00749v1) — Effective Receptive Field (ERF); input attribution for causal interpretation
- [Steering CLIP's vision transformer with sparse autoencoders (2025, CVPR workshop)](https://arxiv.org/abs/2504.08729) — Steerability metrics; vision vs. language sparsity patterns; 10-15% features steerable; middle layers optimal for disentanglement

**Vision Interpretability (Related):**
- [CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks (2023, ICLR)](https://arxiv.org/abs/2204.10965) — Automated neuron labeling via CLIP; scales to large networks
- [Describe-and-Dissect: Interpreting Neurons in Vision Networks with Language Models (2024)](https://arxiv.org/abs/2403.13771) — Extends CLIP-Dissect with natural language generation

**Benchmarking & Evaluation:**
- [SAEBench: Comprehensive Benchmark for Sparse Autoencoders (2025)](https://arxiv.org/abs/2503.09532) — 200+ SAEs evaluated; 8 metrics; reproducible benchmarks
- [Residual Stream Analysis with Multi-Layer SAEs (2024)](https://arxiv.org/abs/2409.04185) — Multi-layer extraction; cross-layer information flow
- [Measuring Sparse Autoencoder Feature Sensitivity (2024)](https://arxiv.org/abs/2509.23717) — Sensitivity metrics for feature robustness

**Architectural Variants:**
- [Improving Dictionary Learning with Gated Sparse Autoencoders (2024)](https://arxiv.org/abs/2404.16014) — Gated SAE; separate gate/magnitude networks
- [Archetypal SAEs (Kempner Institute, 2025)](https://kempnerinstitute.harvard.edu/research/deeper-learning/archetypal-saes-adaptive-and-stable-dictionary-learning-for-concept-extraction-in-large-vision-models/) — Constrained to convex hull; improved stability

### Tools & Implementations

- **SAEV** (OSU-NLP): https://github.com/OSU-NLP-Group/saev — PyTorch SAE training for ViTs; interactive demos
- **SAE Lens**: https://github.com/decoderesearch/SAELens — General-purpose SAE training framework; comprehensive docs
- **MATS SAE Training**: https://github.com/AlignmentResearch/mats_sae_training — Reference implementation
- **Neuronpedia**: https://neuronpedia.org — Visualization & exploration of published SAEs

---

## BIG 5 GUIDANCE FOR VIDEO CLASSIFICATION SAE

### 1. Input Validation
**How to validate with SAE:**
- Check activation distributions across video dataset
- Ensure no extreme outliers (>3σ) in activation magnitude
- Verify batch norm statistics don't shift across train/val sets
- For video: monitor temporal coherence of activations (shouldn't jump discontinuously frame-to-frame)

### 2. Edge Cases
**What SAEs don't handle well:**
- **Very dark/bright frames:** Activations may collapse; use histogram equalization or normalization
- **Motion blur:** Temporal tokens struggle; may overactivate on boundary features
- **Occlusions:** Spatial receptive field may be insufficient; consider patch-pooling SAE variants
- **Compression artifacts:** High-frequency features may be spurious; validate on lightly-compressed video
- **Generator-specific artifacts:** SAE trained on Sora may not transfer to Kling; recommend multi-generator training

### 3. Error Handling
**Common SAE failures and recovery:**
- **Dead latents (no activation):** Check L1 coefficient warm-up; if still present, increase auxiliary loss weight
- **Reconstruction divergence:** Exploding gradient; use gradient clipping (norm 1.0)
- **OOM during extraction:** Reduce batch size; use activation checkpointing
- **Features don't change after intervention:** Feature may be redundant; merge latents via clustering

### 4. Duplication
**How SAEs prevent redundancy:**
- Sparsity constraint directly prevents latent duplication (only k active per sample)
- Monitor max cosine similarity between latent directions; if >0.95, indicates redundancy
- Use probing task (linear classifier on latents) to detect feature overlap
- For video: temporal and spatial features may partially overlap; use mutual information to measure redundancy

### 5. Complexity
**Acceptable complexity for video SAEs:**
- **Single-layer extraction + TopK SAE:** 10-20 lines of hook code; manageable
- **Multi-layer extraction:** 50-100 lines; requires careful book-keeping
- **Causal intervention (steering):** 30-50 lines per intervention experiment
- **Automated labeling (CLIP-Dissect):** 50-100 lines with PyTorch; more if using LLM descriptions
- **Total system:** ~300-500 LOC for end-to-end interpretation pipeline; acceptable complexity

**Recommendation:** Start with single-layer TopK SAE + max-activating examples. Graduate to ERF + causal probing only if interpretability insufficient for your video classification task.

---

## SUMMARY: APPLY SAE TO YOUR VIDEO CLASSIFICATION ViT

### 3-Step Implementation

**Phase 1: Extract Activations (1-2 hours)**
- Load frozen video classification ViT
- Hook residual stream at layer 10 (mid-level features)
- Forward pass on 10K video samples (balanced real/synthetic)
- Collect 1B+ activation vectors

**Phase 2: Train SAE (24-48 hours)**
- TopK SAE, d_latent = 4×d_model
- k_topk = 40 (vision sparsity)
- 40B token training with L1 warm-up + auxiliary loss
- Validate on held-out 1K videos

**Phase 3: Interpret Features (4-8 hours)**
- Extract max-activating patches for each latent (parallelizable)
- Compute ERF via gradient attribution
- Annotate with CLIP-Dissect or manual inspection
- Focus on features that distinguish real from AI-generated

### Specific Hypotheses to Test

1. **Temporal incoherence features:** SAE should learn features activating on impossible optical flow (diagnostic for synthetic video)
2. **Frame boundary artifacts:** Expect features firing on synthetic frame transitions
3. **Object identity violation:** Features detecting subject teleportation or impossible morphs
4. **Compression artifact patterns:** Features specific to H.264 vs. AV1 codec artifacts

**Expected outcome:** 5-15 interpretable features strongly predictive of synthetic video generation; use for explainable detection pipeline.

---

## SOURCES

- [Scaling and Evaluating Sparse Autoencoders (2024)](https://arxiv.org/abs/2406.04093)
- [Interpretable and Testable Vision Features via Sparse Autoencoders (2025)](https://arxiv.org/abs/2502.06755)
- [Causal Interpretation of Sparse Autoencoder Features in Vision (2025)](https://arxiv.org/html/2509.00749v1)
- [Sparse autoencoders reveal selective remapping of visual concepts during adaptation (2024)](https://arxiv.org/abs/2412.05276)
- [Steering CLIP's vision transformer with sparse autoencoders (2025)](https://arxiv.org/abs/2504.08729)
- [Towards Monosemanticity (2023)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [CLIP-Dissect (2023, ICLR)](https://arxiv.org/abs/2204.10965)
- [SAEBench: Comprehensive Benchmark (2025)](https://arxiv.org/abs/2503.09532)
- [Training SAEs - SAE Lens Documentation](https://decoderesearch.github.io/SAELens/latest/training_saes/)
- [SAEV: Sparse Autoencoders for Vision](https://github.com/OSU-NLP-Group/saev)
- [Residual Stream Analysis with Multi-Layer SAEs (2024)](https://arxiv.org/abs/2409.04185)
- [Improving Dictionary Learning with Gated Sparse Autoencoders (2024)](https://arxiv.org/abs/2404.16014)
- [Towards Multimodal Interpretability (2024)](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2)
- [CE-Bench: Contrastive Evaluation Benchmark (2025)](https://arxiv.org/abs/2509.00691)

