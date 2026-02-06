# Experiments Guide: Spectral Analysis & Diffusion Fingerprints

Quick reference for running spectral analysis experiments on your AI-Generated Video Detector.

## 🚀 Quick Start

### Run Spectral Analysis on a Single Video

```bash
python spectral_analysis.py
```

This will:
1. Load your trained model
2. Analyze the default test video
3. Generate comprehensive spectral visualizations
4. Save results to `spectral_analysis/` directory

### Run Diffusion Fingerprint Detection

```bash
python diffusion_fingerprints.py
```

This will:
1. Detect checkerboard artifacts
2. Analyze noise residuals
3. Find temporal inconsistencies
4. Compute overall diffusion likelihood
5. Save results to `diffusion_analysis/` directory

## 📊 Key Experiments

### Experiment 1: Understand Model's Frequency Focus

**Question**: Does the model attend to high-frequency artifacts or low-frequency structures?

```python
from spectral_analysis import SpectralAnalyzer

analyzer = SpectralAnalyzer()
results = analyzer.correlate_spectrum_with_model('test_video.mp4')

# Check frequency attention ratio
freq_ratio = results['correlation']['freq_attention_ratio']

if freq_ratio > 1.0:
    print("Model focuses on HIGH frequencies (textures, artifacts)")
else:
    print("Model focuses on LOW frequencies (shapes, structures)")
```

**Expected Results**:
- AI videos: Model likely attends to high-frequency artifacts
- Real videos: Model may attend to structural features

### Experiment 2: Detect Diffusion-Specific Patterns

**Question**: Can we identify videos from diffusion models?

```python
from diffusion_fingerprints import DiffusionFingerprintDetector

detector = DiffusionFingerprintDetector()
results = detector.comprehensive_analysis('video.mp4')

print(f"Diffusion Score: {results['diffusion_score']:.2f}")
print("Detected fingerprints:")
for finding in results['interpretation']:
    print(f"  - {finding}")
```

**Expected Results**:
- Diffusion videos: Score > 0.7, multiple fingerprints
- GAN videos: Score < 0.4, few fingerprints
- Real videos: Score < 0.3, minimal fingerprints

### Experiment 3: Compare Real vs AI Spectra

**Question**: How do spectral characteristics differ?

```python
from spectral_analysis import compare_real_vs_ai_spectra

real_results, ai_results = compare_real_vs_ai_spectra(
    'path/to/real_video.mp4',
    'path/to/ai_video.mp4',
    save_dir='comparison_results'
)

# Compare diffusion likelihoods
real_dl = real_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']
ai_dl = ai_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']

print(f"Real video diffusion likelihood: {real_dl:.2f}")
print(f"AI video diffusion likelihood: {ai_dl:.2f}")
```

**Expected Results**:
- Real videos: Lower diffusion likelihood, 1/f power decay
- AI videos: Higher diffusion likelihood, artifacts at specific frequencies

### Experiment 4: Batch Analysis Across Dataset

**Question**: What are the overall patterns in our dataset?

```python
from spectral_analysis import SpectralAnalyzer
from pathlib import Path
import pandas as pd

analyzer = SpectralAnalyzer()

# Analyze subset of dataset
real_videos = list(Path('dataset/real').glob('*.mp4'))[:10]
ai_videos = list(Path('dataset/ai').glob('*.mp4'))[:10]

results = []

for video in real_videos + ai_videos:
    try:
        result = analyzer.correlate_spectrum_with_model(str(video))
        results.append({
            'video': video.name,
            'true_label': 'real' if 'real' in str(video) else 'ai',
            'prediction': result['prediction']['class'],
            'confidence': result['prediction']['confidence'],
            'diffusion_likelihood': result['spectral_features']['diffusion_fingerprint']['diffusion_likelihood'],
            'freq_attention_ratio': result['correlation']['freq_attention_ratio']
        })
    except Exception as e:
        print(f"Error processing {video}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv('batch_spectral_analysis.csv', index=False)

print(df.groupby('true_label').mean())
```

### Experiment 5: Identify Key Discriminative Frequencies

**Question**: Which frequency bands are most important for classification?

```python
from spectral_analysis import SpectralAnalyzer
import numpy as np

analyzer = SpectralAnalyzer()

# Analyze multiple videos
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']

freq_importances = []

for video_path in videos:
    results = analyzer.correlate_spectrum_with_model(video_path)

    # Extract attribution spectrum
    attr_spectrum = results['attribution_spectrum']

    # Compute radial profile of attribution
    h, w = attr_spectrum.shape
    center_h, center_w = h // 2, w // 2

    y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
    r = np.sqrt(x*x + y*y).astype(int)

    max_r = min(center_h, center_w)
    radial_attr = []

    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_attr.append(attr_spectrum[mask].mean())

    freq_importances.append(radial_attr)

# Average across videos
avg_importance = np.mean(freq_importances, axis=0)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(avg_importance)
plt.xlabel('Frequency (radius from center)')
plt.ylabel('Average Attribution')
plt.title('Frequency Importance for Classification')
plt.grid(True, alpha=0.3)
plt.savefig('frequency_importance.png', dpi=150)
print("Saved frequency importance plot")
```

## 🔬 Advanced Experiments

### Experiment 6: Test Checkerboard Artifact Hypothesis

**Hypothesis**: Diffusion videos have more checkerboard artifacts than GAN or real videos

```python
from diffusion_fingerprints import DiffusionFingerprintDetector
import matplotlib.pyplot as plt

detector = DiffusionFingerprintDetector()

# Test on different types
video_types = {
    'diffusion': ['diffusion1.mp4', 'diffusion2.mp4'],
    'gan': ['gan1.mp4', 'gan2.mp4'],
    'real': ['real1.mp4', 'real2.mp4']
}

checkerboard_scores = {vtype: [] for vtype in video_types}

for vtype, videos in video_types.items():
    for video in videos:
        _, frames, _, _ = load_video(video)
        result = detector.detect_checkerboard_artifacts(frames)
        checkerboard_scores[vtype].append(result['mean_score'])

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
for vtype, scores in checkerboard_scores.items():
    ax.scatter([vtype] * len(scores), scores, alpha=0.6, s=100)

ax.set_ylabel('Checkerboard Score')
ax.set_title('Checkerboard Artifacts by Video Type')
ax.axhline(y=1.5, color='r', linestyle='--', label='Detection Threshold')
ax.legend()
plt.savefig('checkerboard_comparison.png', dpi=150)
```

### Experiment 7: Temporal Chunk Boundary Detection

**Hypothesis**: Video diffusion models show discontinuities at regular intervals

```python
from diffusion_fingerprints import DiffusionFingerprintDetector
from interpret import load_video
import numpy as np

detector = DiffusionFingerprintDetector()

# Load video
_, frames, _, _ = load_video('diffusion_video.mp4')

# Detect temporal inconsistencies
temporal = detector.detect_temporal_inconsistencies(frames)

if temporal['has_chunk_artifacts']:
    print("✓ Chunk artifacts detected!")
    print(f"Discontinuities at frames: {temporal['discontinuity_frames']}")

    # Check if periodic
    if len(temporal['discontinuity_frames']) > 1:
        intervals = np.diff(temporal['discontinuity_frames'])
        print(f"Chunk intervals: {intervals}")
        print(f"Mean interval: {np.mean(intervals):.1f} frames")
        print(f"Likely chunk size: {np.median(intervals):.0f} frames")
else:
    print("✗ No chunk artifacts detected")
```

### Experiment 8: Correlation with Model Confidence

**Question**: Are spectral features correlated with model confidence?

```python
from spectral_analysis import SpectralAnalyzer
import matplotlib.pyplot as plt
import numpy as np

analyzer = SpectralAnalyzer()

# Analyze multiple videos
videos = [...]  # Your video list

confidences = []
diffusion_likelihoods = []
freq_ratios = []

for video in videos:
    results = analyzer.correlate_spectrum_with_model(video)

    confidences.append(results['prediction']['confidence'])
    diffusion_likelihoods.append(
        results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']
    )
    freq_ratios.append(results['correlation']['freq_attention_ratio'])

# Plot correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confidence vs Diffusion Likelihood
axes[0].scatter(confidences, diffusion_likelihoods, alpha=0.6)
axes[0].set_xlabel('Model Confidence')
axes[0].set_ylabel('Diffusion Likelihood')
axes[0].set_title('Confidence vs Diffusion Fingerprint')
axes[0].grid(True, alpha=0.3)

# Confidence vs Frequency Ratio
axes[1].scatter(confidences, freq_ratios, alpha=0.6)
axes[1].set_xlabel('Model Confidence')
axes[1].set_ylabel('Frequency Attention Ratio')
axes[1].set_title('Confidence vs Frequency Focus')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=150)

# Compute correlations
corr1 = np.corrcoef(confidences, diffusion_likelihoods)[0, 1]
corr2 = np.corrcoef(confidences, freq_ratios)[0, 1]

print(f"Confidence-Diffusion correlation: {corr1:.3f}")
print(f"Confidence-FreqRatio correlation: {corr2:.3f}")
```

## 📈 Expected Results Summary

| Experiment | Real Videos | AI (Diffusion) | AI (GAN) |
|------------|-------------|----------------|----------|
| Diffusion Likelihood | < 0.3 | > 0.7 | 0.3 - 0.5 |
| Checkerboard Score | < 1.5 | > 2.0 | 1.0 - 1.8 |
| Freq Attention Ratio | Variable | > 1.5 | > 1.2 |
| Temporal Chunks | No | Yes | No |
| Gaussian Noise | No | Yes | Variable |

## 🎯 Interpretation Tips

### High Diffusion Likelihood (> 0.7)
- ✅ Strong evidence of diffusion generation
- Multiple fingerprints detected (checkerboard, noise, temporal)
- Model likely uses these cues for classification

### High Frequency Attention (ratio > 2.0)
- ✅ Model focuses on textures and fine details
- Suggests detection of high-frequency artifacts
- Important for identifying generation artifacts

### Temporal Discontinuities
- ✅ Video processed in chunks
- Characteristic of video diffusion models
- Look for periodic patterns (8, 16, or 24 frame intervals)

### Gaussian Noise Patterns
- ✅ Residual from denoising process
- Check mean ≈ 0, skewness ≈ 0, kurtosis ≈ 3
- More uniform than natural video noise

## 🐛 Troubleshooting

### Low Correlation Values
```python
# Increase IG steps for better attributions
attributions = ig.attribute(video, baseline, target, n_steps=100)
```

### Memory Errors
```python
# Analyze fewer frames
results = analyzer.extract_fft_features(frames[::2])  # Every other frame
```

### Slow Processing
```python
# Use smaller resolution or sample frames
# Disable explanations for faster analysis
results = analyzer.correlate_spectrum_with_model(video, generate_attributions=False)
```

## 📚 Further Reading

- See `SPECTRAL_ANALYSIS_README.md` for detailed documentation
- Review visualizations in output directories
- Check example outputs in `examples/` folder

## 🔗 Integration with Main Model

These experiments complement the main classifier by:
1. **Explaining decisions**: Why did the model classify this way?
2. **Detecting fingerprints**: What artifacts are present?
3. **Improving robustness**: Identify potential weaknesses
4. **Dataset analysis**: Understand dataset characteristics

---

**Start experimenting! 🧪🔬**

Run `python spectral_analysis.py` or `python diffusion_fingerprints.py` to begin!
