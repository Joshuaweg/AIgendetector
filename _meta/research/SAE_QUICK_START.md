# SAE for Video Classification ViT: Quick Start

## The 5-Minute Overview

**What:** Train a Sparse Autoencoder on your video ViT's middle layers to extract interpretable features that explain what your model attends to when classifying real vs. AI-generated video.

**Why:** SAEs decompose entangled neural activations into sparse, interpretable directions. For video forensics, this reveals which features your model uses to detect synthetic artifacts (temporal incoherence, impossible motion, etc.).

**How long:** 
- Extract activations: 1-2 hours
- Train SAE: 24-48 hours
- Interpret: 4-8 hours
- **Total: ~3-5 days**

---

## Critical Success Factors (Don't Skip These)

1. **Use TopK SAE, not ReLU SAE**
   - Directly set sparsity (k=32-64 for vision)
   - Better feature quality than ReLU
   - Simpler to tune

2. **Hook from middle layer (10-11 in ViT-B)**
   - Extract residual stream (not attention output)
   - Middle layers have best interpretable features
   - Avoid early (too low-level) and late (too entangled)

3. **Don't trust max-activating examples alone**
   - Use Effective Receptive Field (ERF) with gradient attribution
   - Tells you what *causes* activation, not just where it peaks
   - 30-50 lines of code to implement

4. **Use auxiliary loss + L1 warm-up during training**
   - Prevents dead latents naturally
   - Start L1 coefficient at 0, increase over first 5% of training

5. **Validate on held-out videos**
   - Training video distribution != production distribution
   - Test interpretability on unseen generators (Sora, Kling, etc.)

---

## The 3-Phase Implementation

### Phase 1: Activation Extraction (2 hours)

```python
import torch
from video_vit import load_vit  # your model

model = load_vit()
model.eval()

activations = []
with torch.no_grad():
    for video_batch in dataloader:  # 10K videos, real + synthetic
        # Hook into layer 10 residual stream
        x = model.forward_with_hook(video_batch, layer=10)
        activations.append(x)  # shape: [batch, num_tokens, d_model]

# Save activations (1B+ vectors for training)
torch.save(torch.cat(activations), 'activations.pt')
```

### Phase 2: SAE Training (48 hours)

Use **SAE Lens** or **SAEV**:
```
d_latent = 4 × d_model  (e.g., 3K latents for ViT-B)
k_topk = 40-64
learning_rate = 4e-4
auxiliary_loss = True
l1_warmup_steps = training_steps * 0.05
```

Expect: KL divergence ~150-200 nats, L0 ~40 features/sample.

### Phase 3: Feature Interpretation (4 hours)

For each of ~3K features:
1. Find top-k images where it activates (parallel scan)
2. Extract surrounding patches
3. Compute ERF (gradient attribution) to find causal patches
4. Label with CLIP-Dissect or manual inspection

Focus on features that differ between real/AI-generated videos.

---

## Gotchas & Fixes

| Problem | Fix |
|---------|-----|
| Dead latents | L1 warm-up + auxiliary loss |
| Low-quality features | Wrong layer hook; use layer 10-11 |
| Misleading interpretations | Use ERF, not just max-activating examples |
| Won't generalize to new generators | Train on multi-generator dataset |
| Entangled features | Use larger d_latent (16× instead of 4×) |
| OOM during training | Reduce batch size; use gradient checkpointing |

---

## Papers to Read (In Order)

1. **[Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)** — TopK vs ReLU benchmark
2. **[Interpretable and Testable Vision Features](https://arxiv.org/abs/2502.06755)** — Vision SAE methodology (your baseline)
3. **[Causal Interpretation of SAE Features](https://arxiv.org/abs/2509.00749)** — ERF + causal probing
4. **[Steering CLIP's ViT with SAEs](https://arxiv.org/abs/2504.08729)** — Vision-specific insights

---

## Tools

- **SAEV** (recommended): https://github.com/OSU-NLP-Group/saev
- **SAE Lens**: https://github.com/decoderesearch/SAELens
- Full research: `/home/ubuntu/AIgendetector/_meta/research/sae-vision-transformer-research.md`

