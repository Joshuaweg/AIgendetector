# SAE Interpretability - Quick Start Guide

Get started with Sparse Autoencoder (SAE) interpretability in 3 simple steps!

## Prerequisites

- A trained video classifier model at `/media/joshua/WD_BLACK/Gen-Video/model/`
- Video dataset at `/media/joshua/WD_BLACK/Gen-Video/dataset/`
- CUDA-capable GPU (recommended, but CPU works too)

## Step 1: Train the SAE (20-30 minutes)

```bash
python train_sae.py
```

**What happens:**
- Loads your pretrained video classifier
- Collects activations from 5,000 video samples
- Trains an SAE to learn 4,096 interpretable features
- Saves checkpoints every epoch to `model/sae/`

**Expected output:**
```
Epoch 20 Summary:
  Total Loss: 0.0234
  Reconstruction Loss: 0.0231
  Sparsity Loss: 0.0003
  L0 Norm: 47.3
  Feature Density: 1.15%
  Dead Features: 423/4096
  Alive Features: 3673/4096
```

**Good results:**
- Reconstruction loss < 0.05
- L0 norm between 20-100
- Feature density 1-5%
- Dead features < 1000

## Step 2: Visualize Features (2-3 minutes)

```bash
python visualize_sae_features.py
```

**What happens:**
- Loads the trained SAE
- Analyzes feature activation patterns
- Creates visualizations of top features
- Saves to `model/sae/visualizations/`

**Output files:**
- `feature_{N}_direction.png` - What each feature represents
- `feature_{N}_distribution.png` - How often it activates
- `feature_{N}_stats.txt` - Detailed statistics

## Step 3: Interpret Videos (1-2 minutes per video)

```bash
python interpret_with_sae.py
```

**What happens:**
- Selects a random video from your test set
- Runs classifier and captures activations
- Analyzes which SAE features activate
- Shows temporal evolution of features
- Saves comprehensive analysis to `sae_interpretation_results/`

**Output:**
```
Prediction: AI-Generated (confidence: 87.3%)
True label: AI-Generated

Top 20 active features:
  1. Feature 1847: max=3.421, mean=1.234
  2. Feature 892: max=2.987, mean=1.089
  3. Feature 3201: max=2.654, mean=0.923
  ...
```

## Understanding the Results

### Feature Activation Heatmap
Shows which features fire at which points in the video:
- **Bright spots**: Strong feature activation
- **Vertical lines**: Features that persist over time
- **Scattered activations**: Transient features

### Temporal Feature Evolution
Line plots showing how top features change:
- **Smooth curves**: Gradually changing patterns
- **Spikes**: Sudden changes in the video
- **Plateaus**: Stable patterns

### Fake vs Real Comparison
Bar charts showing discriminative features:
- **Large differences**: Features that help classify
- **Similar heights**: Features common to both classes

## Quick Configuration

### Change SAE Size (in `train_sae.py`)

```python
SAE_CONFIG = {
    'hidden_dim': 4096,  # More features (try 2048, 8192)
    'sparsity_coefficient': 1e-3,  # Higher = sparser (try 5e-4, 2e-3)
}
```

### Analyze Different Layer (in `train_sae.py`)

```python
# Change from last layer to first layer
target_layer = classifier.classifier.transformer_encoder.layers[0]  # Instead of layers[-1]
```

### Analyze Specific Video (in `interpret_with_sae.py`)

```python
# Instead of select_random_video()
video_path = "/path/to/your/video.mp4"
```

## Common Issues

### "No trained SAE found"
**Solution:** Run `python train_sae.py` first

### "CUDA out of memory"
**Solutions:**
- Reduce `batch_size` in `SAE_CONFIG`
- Reduce `max_samples` when collecting activations
- Use fewer workers: `num_workers: 0`

### High reconstruction loss (>0.1)
**Solutions:**
- Train for more epochs
- Increase `hidden_dim` (more capacity)
- Decrease `sparsity_coefficient` (less sparsity penalty)

### Too many dead features (>1500)
**Solutions:**
- Decrease `hidden_dim` (fewer features to learn)
- Decrease `sparsity_coefficient` (encourage more activations)
- Collect activations from more diverse videos

## Next Steps

1. **Compare multiple videos**: Modify `interpret_with_sae.py` to batch process
2. **Track specific features**: Monitor how a feature behaves across videos
3. **Feature steering**: Test how modifying features changes predictions
4. **Multi-layer analysis**: Train SAEs for multiple transformer layers

## Example Workflow

```bash
# 1. Train SAE overnight
python train_sae.py

# 2. Check TensorBoard while training
tensorboard --logdir=/media/joshua/WD_BLACK/Gen-Video/runs

# 3. Visualize learned features
python visualize_sae_features.py

# 4. Interpret specific videos
python interpret_with_sae.py

# 5. Compare fake vs real
# Edit interpret_with_sae.py to add:
fake_path = "dataset/ai_video_1.mp4"
real_path = "dataset/real_video_1.mp4"
compare_fake_vs_real_features(model, sae, fake_path, real_path, device, save_dir)
```

## Performance Tips

- **GPU Memory**: SAE training uses ~4-6GB VRAM
- **Training Time**: ~20-30 minutes for 20 epochs on RTX 3090
- **Disk Space**: Checkpoints are ~200MB each
- **CPU Mode**: Works but 5-10x slower

## Getting Help

If something doesn't work:
1. Check the full documentation: `SAE_INTERPRETABILITY.md`
2. Verify your paths in the config sections
3. Check TensorBoard logs for training issues
4. Ensure your classifier is properly trained

## What's Next?

After getting familiar with basic SAE analysis:
- Try different sparsity levels to find optimal features
- Analyze which features correlate with model errors
- Build a feature library for common patterns
- Create automated reports for video analysis

Happy interpreting! 🔍
