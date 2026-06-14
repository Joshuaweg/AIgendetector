# ViViT Tubelet Embeddings - Complete Technical Analysis

## Related

**Architecture:** [[detection_roadmap]] · [[flow_branch_plan]] · [[MANIFEST]]

**Interpretability:** [[per_token_tcav_2026_06_13]] · [[integrated-gradients-baselines-research]]

---

**Paper:** ViViT: A Video Vision Transformer  
**Authors:** Arnab et al., 2021  
**arXiv:** 2103.15691  
**Section:** 3.2 - Embedding video clips

---

## 1. Definition and Core Concept

Tubelet embedding is the method by which ViViT converts video input into tokens for transformer processing. Rather than treating each frame independently (as in "Uniform frame sampling"), tubelet embedding extracts **non-overlapping, spatio-temporal "tubes" from the input volume, and linearly projects them to ℝ^d**.

**Key insight:** This is an extension of Vision Transformer's (ViT) 2D embedding approach to 3D, and **corresponds to a 3D convolution** operation.

---

## 2. Mathematical Formulation

### 2.1 Input and Output Notation

```
Video input:  V ∈ ℝ^(T×H×W×C)
Token output: z̃ ∈ ℝ^(n_t×n_h×n_w×d)
```

Where:
- **T** = number of temporal frames
- **H** = height of input video
- **W** = width of input video
- **C** = number of channels (typically 3 for RGB)
- **d** = embedding dimension (transformer hidden dimension)

### 2.2 Token Count Formula

For a tubelet of dimensions **t×h×w** (temporal × height × width):

```
n_t = ⌊T/t⌋  (number of temporal tokens)
n_h = ⌊H/h⌋  (number of height tokens)
n_w = ⌊W/w⌋  (number of width tokens)
```

**Total tokens:** N = n_t × n_h × n_w

### 2.3 Linear Projection

Each non-overlapping tubelet (a 3D volume of size t×h×w×C) is linearly projected to a d-dimensional embedding vector through a learned weight matrix. This projection operation is mathematically equivalent to applying a 3D convolutional filter with kernel size t×h×w and stride equal to the kernel size (no overlapping).

---

## 3. Typical Tubelet Dimensions from Experiments

The notation used in the paper is **spatial×temporal** (h×w×t):

| Model | Tubelet Size | Interpretation |
|-------|--------------|-----------------|
| ViViT-B/16x2 | 16×16×2 | 16×16 spatial patches, 2 frames temporal |
| ViViT-L/16x2 | 16×16×2 | Same as ViViT-B |
| ViViT-H/14x2 | 14×14×2 | 14×14 spatial patches, 2 frames temporal |

These represent the **temporal window width** (how many consecutive frames are fused into a single token).

---

## 4. Temporal Information Encoding

### How Tubelets Capture Temporal Information

The paper explicitly contrasts two approaches:

**Uniform Frame Sampling (Frame-level approach):**
- Each frame is embedded independently using ViT's 2D embedding
- Produces n_t × n_h × n_w total tokens (frame count × spatial tokens per frame)
- Temporal fusion happens **at the transformer layer level** through self-attention
- No explicit spatial-temporal structure in tokenization

**Tubelet Embedding (Proposed approach):**
- "Tubelets fuse spatio-temporal information during tokenisation, in contrast to uniform frame sampling where temporal information from different frames is fused by the transformer."
- Temporal information is captured **at tokenization time**, not left for transformer layers to discover
- t consecutive frames are grouped with their spatial patch and embedded together
- Creates a **3D receptive field** at the token level

### Why This Is Better

The paper's analysis shows that embedding at the token level allows:
1. **Early temporal fusion:** The model starts with spatially and temporally structured tokens
2. **Efficiency:** Fewer total tokens (n_t × n_h × n_w vs. n_t × n_h × n_w for each frame independently)
3. **Built-in temporal context:** Each token contains multiple frames' information from the start

---

## 5. Implementation Details

### 5.1 Method: Learned Linear Projection (3D Convolution)

```
For each non-overlapping tubelet: z_patch = W · x_tubelet + b
```

Where:
- **x_tubelet** ∈ ℝ^(t×h×w×C) - the flattened 3D tubelet volume
- **W** ∈ ℝ^(d×t×h×w×C) - learned embedding matrix
- **z_patch** ∈ ℝ^d - resulting embedding vector
- **b** ∈ ℝ^d - bias term

This is mathematically identical to a 3D convolution with:
- Kernel size: t×h×w
- Stride: t×h×w (non-overlapping)
- Output channels: d

### 5.2 No Averaging or Concatenation

Critically, the paper uses **learned linear projection**, not:
- Averaging pooling over the tubelet volume
- Concatenation of frame embeddings
- Differencing or arithmetic operations on frames

The learning of the projection parameters allows the model to discover the optimal way to aggregate temporal information.

---

## 6. Filter Initialization Strategies (Equation 8 and 9)

The paper demonstrates that initializing the 3D filters from pretrained 2D image models is crucial:

### 6.1 Filter Inflation (Equation 8) - Not Recommended

```
E = 1/t [E_image, …, E_image]
```

Takes the 2D embedding filter and replicates it across all t temporal positions, then averages.

**Result:** 77.6% accuracy on Kinetics-400

### 6.2 Central Frame Initialization (Equation 9) - Recommended

```
E = [0, …, E_image, …, 0]
```

Places the pretrained 2D filter at the **center temporal position** ⌊t/2⌋ and zeros elsewhere.

**Why this works:** 
- At initialization, the model effectively functions like "Uniform frame sampling" (only the center frame's information flows)
- During training, the model learns to incorporate temporal context from neighboring frames
- Leverages pretrained image model knowledge while allowing temporal learning

**Performance Results (Kinetics-400):**
- Central frame: **79.2%** accuracy
- Uniform frame sampling: 78.5% accuracy  
- Filter inflation: 77.6% accuracy
- Random initialization: 73.2% accuracy

**Central frame outperformed filter inflation by 1.6 percentage points.**

---

## 7. Advantage Over Frame-Level Approaches

### Theoretical Advantage

Tubelet embedding captures spatiotemporal structure **before** the transformer processes it, whereas frame-level approaches rely on the transformer's self-attention to discover temporal correlations.

### From the Paper

"Tubelets fuse spatio-temporal information during tokenisation, in contrast to uniform frame sampling where temporal information from different frames is fused by the transformer."

### Practical Implications

1. **Token efficiency:** Fewer tokens (n_t × n_h × n_w vs. T × n_h × n_w for full frame sampling)
2. **Inductive bias:** The tokenization itself encodes the prior that neighboring frames are related
3. **Better initialization:** Using pretrained image filters at the center position gives the model a strong starting point
4. **Reduced transformer burden:** Temporal fusion happens at tokenization, not requiring attention layers to redundantly learn it

---

## 8. Architecture Integration

Within the full ViViT model, after tubelet embedding:

1. Tokens z̃ ∈ ℝ^(n_t×n_h×n_w×d) are flattened to z ∈ ℝ^(N×d) where N = n_t × n_h × n_w
2. A learnable class token is prepended: z_0 = [CLS]
3. Optional positional embeddings are added
4. Tokens pass through L transformer blocks with:
   - Multi-head self-attention: MSA(LN(z_ℓ)) + z_ℓ
   - Feed-forward MLP: MLP(LN(y_ℓ)) + y_ℓ
   - Residual connections maintaining dimension d throughout
5. Final class token z_0^L is used for video classification

---

## 9. Model Variants

ViViT proposes four factorization strategies to handle the computational burden of long token sequences. Tubelet embedding is used in all variants:

1. **Spatio-temporal attention:** Joint attention over all spatial-temporal tokens
2. **Factorised encoder:** Separate spatial attention followed by temporal attention
3. **Factorised self-attention:** Alternating spatial-temporal attention blocks
4. **Factorised dot-product attention:** Decomposed attention computation

---

## Key Takeaways

| Aspect | Detail |
|--------|--------|
| **Method** | Non-overlapping 3D tubes linearly projected via learned weights (3D convolution) |
| **Temporal aggregation** | Via learned linear projection, not averaging/pooling |
| **Typical dimensions** | 16×16 spatial, 2 temporal frames per tubelet |
| **Token count** | N = ⌊T/t⌋ × ⌊H/h⌋ × ⌊W/w⌋ (much fewer than frame-level) |
| **Initialization** | Central frame (Eq. 9) outperforms filter inflation (Eq. 8) by 1.6% |
| **Advantage over frames** | Fuses spatiotemporal info at tokenization, not relying on transformer attention alone |
| **Key insight** | Combines 2D pretrained image model knowledge with learned 3D temporal aggregation |

---

## Sources

- [ViViT: A Video Vision Transformer (HTML)](https://ar5iv.labs.arxiv.org/html/2103.15691)
- [ViViT: A Video Vision Transformer (arXiv PDF)](https://arxiv.org/pdf/2103.15691)
- [ViViT arXiv Abstract](https://arxiv.org/abs/2103.15691)
