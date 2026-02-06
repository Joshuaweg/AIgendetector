# Sparse Autoencoder (SAE) for Model Interpretability

This directory contains a complete implementation of Sparse Autoencoders (SAEs) for interpreting the learned features of the AI-generated video classifier.

## Overview

Sparse Autoencoders help us understand what features the transformer model has learned by decomposing neural network activations into interpretable, sparse features. This is based on the "Towards Monosemanticity" approach from Anthropic's interpretability research.

## Architecture

### SparseAutoencoder (`sparse_autoencoder.py`)

The SAE learns to:
1. **Encode** transformer activations into a larger sparse feature space
2. **Decode** those sparse features back to reconstruct the original activations
3. **Enforce sparsity** through L1 regularization, encouraging each feature to activate only for specific patterns

**Key Features:**
- Configurable input/hidden dimensions
- Tied weights option (decoder = encoder transpose)
- Automatic dead feature tracking
- Feature activation statistics

**Architecture:**
```
Input (768-dim) -> Pre-bias -> Encoder -> ReLU -> Sparse Features (4096-dim)
                                                          ↓
                                                       Decoder
                                                          ↓
                                              Reconstructed Input (768-dim)
```

### MultiLayerSAE

Allows training separate SAEs for different transformer layers to analyze features at different depths.

## Files

### Core Implementation
- **`sparse_autoencoder.py`**: Main SAE model implementation
  - `SparseAutoencoder`: Single-layer SAE
  - `MultiLayerSAE`: Multi-layer SAE wrapper

### Training
- **`train_sae.py`**: Training script for SAEs
  - Collects activations from the pretrained classifier
  - Trains SAE to reconstruct these activations
  - Monitors feature statistics (dead features, activation rates)
  - Saves checkpoints and TensorBoard logs

### Visualization
- **`visualize_sae_features.py`**: Visualization utilities
  - `SAEFeatureVisualizer`: Main visualization class
  - Feature direction visualization
  - Top activating samples
  - Feature dashboards
  - Class-discriminative feature comparison

### Integration
- **`interpret_with_sae.py`**: Integration with existing interpretation pipeline
  - Combines SAE analysis with Integrated Gradients
  - Per-video feature analysis
  - Temporal feature evolution
  - Fake vs. real feature comparison

## Usage

### 1. Train the SAE

First, ensure you have a trained video classifier at `/media/joshua/WD_BLACK/Gen-Video/model/full_classifier_best.pt`

```bash
python train_sae.py
```

This will:
- Load the pretrained classifier
- Collect activations from the last transformer layer
- Train the SAE to reconstruct these activations
- Save checkpoints to `model/sae/`

**Configuration** (in `train_sae.py`):
```python
SAE_CONFIG = {
    'input_dim': 768,           # Transformer dimension
    'hidden_dim': 4096,         # Sparse feature space (larger = more features)
    'sparsity_coefficient': 1e-3,  # L1 penalty strength
    'batch_size': 8,
    'learning_rate': 3e-4,
    'epochs': 20,
}
```

### 2. Visualize Features

After training, explore what features were learned:

```bash
python visualize_sae_features.py
```

This generates:
- Feature activation distributions
- Top feature dashboards
- Feature direction visualizations

### 3. Interpret Videos with SAE

Analyze specific videos using both Integrated Gradients and SAE features:

```bash
python interpret_with_sae.py
```

This provides:
- Which features activate for a given video
- How features evolve over time
- Which features discriminate fake vs. real videos
- Comparison with ground truth labels

## Key Metrics

### During Training

- **Reconstruction Loss**: MSE between original and reconstructed activations (lower is better)
- **Sparsity Loss**: L1 norm of feature activations (encourages sparsity)
- **L0 Norm**: Average number of active features per sample (target: 10-100)
- **Feature Density**: Percentage of non-zero features (target: 1-5%)
- **Dead Features**: Features that rarely activate (< 0.1% of time)

### During Interpretation

- **Active Features**: Which of the 4096 features fire for this video
- **Top Features**: Strongest activating features
- **Temporal Patterns**: How features change over video frames
- **Discriminative Features**: Features that differ between fake/real

## Example Output

```
Prediction: AI-Generated (confidence: 87.3%)
True label: AI-Generated

SAE Metrics:
  Reconstruction Loss: 0.0234
  L0 Norm (active features): 47.3
  Feature Density: 1.15%

Top 10 Features:
  1. Feature 1847: max=3.421, mean=1.234
  2. Feature 892: max=2.987, mean=1.089
  3. Feature 3201: max=2.654, mean=0.923
  ...
```

## Interpretation Guidelines

### Understanding Features

1. **High activation frequency** (>5%): General features used across many videos
2. **Low activation frequency** (<1%): Specific features for rare patterns
3. **Class-discriminative features**: Features with large difference in mean activation between fake/real

### What to Look For

- **Temporal patterns**: Do certain features activate at specific points?
- **Consistent discriminators**: Do the same features always help classify fake videos?
- **Dead features**: High number suggests SAE needs retraining with adjusted hyperparameters

## Troubleshooting

### High Reconstruction Loss
- Increase `hidden_dim` to give more capacity
- Decrease `sparsity_coefficient` to allow more features
- Train for more epochs

### Too Many Dead Features (>50%)
- Decrease `hidden_dim`
- Decrease `sparsity_coefficient`
- Increase learning rate
- Collect more diverse activations

### All Features Active (Low Sparsity)
- Increase `sparsity_coefficient`
- Decrease learning rate
- Check that ReLU is applied in encoder

## Advanced Usage

### Training on Specific Layers

Modify `train_sae.py` to hook different layers:

```python
# Hook earlier layer
target_layer = classifier.classifier.transformer_encoder.layers[0]

# Hook patch encoder output
target_layer = classifier.patch_encoder
```

### Multi-Layer Analysis

Use `MultiLayerSAE` to analyze multiple layers:

```python
layer_dims = {
    'layer_0': 768,
    'layer_6': 768,
    'layer_11': 768,
}

multi_sae = MultiLayerSAE(layer_dims, hidden_dim=4096)
```

### Feature Steering

Test how individual features affect predictions:

```python
# Get reconstruction for a specific feature
feature_direction = sae.reconstruct_from_feature(feature_idx=1847, activation=1.0)

# Add this to actual activations and see how prediction changes
modified_activations = original_activations + 0.5 * feature_direction
```

## References

- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

## Citation

If you use this SAE implementation in your research, please cite:

```bibtex
@software{sae_video_classifier,
  title={Sparse Autoencoder for Video Classifier Interpretability},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/yourrepo}
}
```

## Future Improvements

- [ ] Automated feature labeling using activation maximization
- [ ] Feature visualization on actual video frames
- [ ] Cross-video feature consistency analysis
- [ ] Online SAE training during classifier training
- [ ] Feature importance for specific predictions
- [ ] Interactive feature explorer web interface
