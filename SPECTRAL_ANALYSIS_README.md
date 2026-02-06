# Spectral Analysis & Diffusion Fingerprint Detection

Advanced experiments to improve explainability of the AI-Generated Video Detector by analyzing frequency domain patterns and diffusion model fingerprints.

## Overview

This module provides two complementary approaches to understanding model decisions:

1. **Spectral Analysis** (`spectral_analysis.py`) - Analyzes frequency domain characteristics and correlates them with model attention
2. **Diffusion Fingerprints** (`diffusion_fingerprints.py`) - Detects specific artifacts left by diffusion models

## Research Motivation

### Why Spectral Analysis?

AI-generated videos often exhibit different frequency characteristics than real videos:

- **Real videos**: Natural 1/f power law decay in frequency domain
- **AI videos**: Artifacts at specific frequencies from generation process
- **Diffusion models**: Gaussian noise residuals, checkerboard patterns from U-Net upsampling

### Key Questions Addressed

1. **Does the model attend to frequency-domain artifacts?**
   - Correlation between model attention and spectral features
   - Which frequency bands drive classification decisions?

2. **Can we detect diffusion-specific fingerprints?**
   - Checkerboard artifacts from transposed convolutions
   - Gaussian noise patterns from iterative denoising
   - Temporal inconsistencies at chunk boundaries

3. **How do spectral patterns differ between real and AI videos?**
   - Power spectrum analysis
   - DCT (Discrete Cosine Transform) features
   - Wavelet multi-scale analysis

## Installation

```bash
# Install required dependencies
pip install scipy>=1.11.0 PyWavelets>=1.4.0

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### 1. Spectral Analysis

Analyze frequency domain patterns and correlate with model decisions:

```python
from spectral_analysis import SpectralAnalyzer

# Initialize analyzer
analyzer = SpectralAnalyzer()

# Analyze single video
results = analyzer.visualize_spectral_analysis('path/to/video.mp4')

# Results include:
# - FFT power spectrum
# - DCT frequency band analysis
# - Diffusion fingerprint likelihood
# - Model attribution in frequency domain
# - Correlation between spectral features and model attention
```

### 2. Compare Real vs AI

```python
from spectral_analysis import compare_real_vs_ai_spectra

# Compare spectral characteristics
real_results, ai_results = compare_real_vs_ai_spectra(
    'path/to/real_video.mp4',
    'path/to/ai_video.mp4'
)
```

### 3. Diffusion Fingerprint Detection

Detect specific artifacts from diffusion models:

```python
from diffusion_fingerprints import DiffusionFingerprintDetector

# Initialize detector
detector = DiffusionFingerprintDetector()

# Comprehensive analysis
results = detector.visualize_fingerprints('path/to/video.mp4')

# Results include:
# - Checkerboard artifact detection
# - Gaussian noise analysis
# - Temporal chunk boundaries
# - Wavelet multi-scale features
# - Overall diffusion likelihood score
```

## Features

### Spectral Analysis Module

#### 1. FFT (Fast Fourier Transform) Analysis
```python
fft_features = analyzer.extract_fft_features(frames)
```
- **2D FFT** of each frame
- **Power spectrum** analysis
- **Radial power profile** (1/f decay check)
- Detects periodic patterns and artifacts

#### 2. DCT (Discrete Cosine Transform) Analysis
```python
dct_features = analyzer.extract_dct_features(frames)
```
- Frequency band energy (low, mid, high)
- Compression artifact detection (JPEG-like patterns)
- Ratio analysis between frequency bands

#### 3. Diffusion Fingerprint Detection
```python
fingerprint = analyzer.detect_diffusion_fingerprint(frames)
```
- **Checkerboard score**: Detects U-Net upsampling artifacts
- **Noise uniformity**: Measures Gaussian noise patterns
- **Radial profile**: Checks for power law violations
- **Temporal variance**: Frame-to-frame spectral consistency

#### 4. Model-Spectrum Correlation
```python
correlation = analyzer.correlate_spectrum_with_model(video_path)
```
- Computes **Integrated Gradients** attributions
- Transforms attributions to frequency domain
- **Correlates** attribution with spectral power
- Analyzes which frequency bands model attends to

### Diffusion Fingerprints Module

#### 1. Checkerboard Artifact Detection
```python
checkerboard = detector.detect_checkerboard_artifacts(frames)
```
- **FFT peak detection** at Nyquist frequency
- **Kernel convolution** with checkerboard pattern
- Identifies U-Net transposed convolution artifacts

**Why it works**: Diffusion models use U-Nets with upsampling layers. Transposed convolutions create characteristic checkerboard patterns at specific frequencies.

#### 2. Noise Residual Analysis
```python
noise = detector.analyze_noise_residuals(frames)
```
- High-pass filtering to extract noise
- Statistical analysis (mean, std, skewness, kurtosis)
- **Gaussian noise detection**
- Deviation from natural image statistics

**Why it works**: Diffusion models iteratively denoise images, often leaving Gaussian noise residuals. Real videos have different noise characteristics.

#### 3. Temporal Inconsistency Detection
```python
temporal = detector.detect_temporal_inconsistencies(frames)
```
- Frame-to-frame difference analysis
- **Discontinuity detection** at chunk boundaries
- Periodic pattern analysis
- Identifies video diffusion artifacts

**Why it works**: Video diffusion models process frames in chunks (e.g., 8 or 16 frames). Chunk boundaries often show discontinuities.

#### 4. Wavelet Multi-Scale Analysis
```python
wavelet = detector.wavelet_analysis(frames)
```
- Multi-level 2D wavelet decomposition
- Energy distribution across scales
- Mid-level detail detection
- U-Net skip connection artifacts

**Why it works**: U-Net architectures have skip connections that combine features at multiple scales, creating characteristic energy distributions in wavelet domain.

#### 5. Color Distribution Analysis
```python
color = detector.detect_color_distribution_anomalies(frames)
```
- Per-channel histogram analysis
- Entropy computation
- Statistical moments
- Color space uniformity

**Why it works**: Diffusion models can produce slightly different color distributions than natural videos due to the denoising process.

## Output & Visualizations

### Spectral Analysis Outputs

The `visualize_spectral_analysis()` function creates a comprehensive 9-panel figure:

1. **Sample Frame** - Original video frame
2. **Power Spectrum** - FFT magnitude (log scale)
3. **Attribution Spectrum** - Model attention in frequency domain
4. **Radial Profile** - Power vs frequency (log-log plot)
5. **DCT Bands** - Energy in low/mid/high frequencies
6. **Correlation Metrics** - Spectrum-attribution correlation
7. **Diffusion Indicators** - Fingerprint detection scores
8. **Prediction Info** - Classification results and interpretation
9. **Temporal Variance** - Frame-to-frame spectral consistency

### Diffusion Fingerprint Outputs

The `visualize_fingerprints()` function creates a 9-panel analysis:

1. **Sample Frame** - Original video frame
2. **Checkerboard Scores** - Per-frame detection over time
3. **Noise Residual** - High-frequency noise pattern
4. **Temporal Diffs** - Frame-to-frame changes with discontinuities
5. **Wavelet Energy** - Multi-scale energy distribution
6. **Color Entropy** - RGB channel entropy
7. **Fingerprint Summary** - All detection indicators
8. **Overall Analysis** - Diffusion likelihood score and findings
9. **FFT Magnitude** - Frequency domain visualization

## Interpretation Guide

### Spectral Correlation Interpretation

**High Frequency Attention > Low Frequency Attention**
- Model focuses on textures, noise, and fine details
- Suggests detection of high-frequency artifacts

**Low Frequency Attention > High Frequency Attention**
- Model focuses on overall structure and shapes
- Suggests detection of structural anomalies

**Positive Correlation (> 0.5)**
- Model attention aligns with spectral power
- Decisions based on frequency domain patterns

**Negative Correlation (< -0.3)**
- Model attends to regions with unusual spectral characteristics
- Inverse relationship with power distribution

### Diffusion Fingerprint Scores

**Diffusion Likelihood: 0.7 - 1.0 (High)**
- Strong evidence of diffusion model generation
- Multiple fingerprints detected
- High confidence in AI generation

**Diffusion Likelihood: 0.4 - 0.7 (Moderate)**
- Some diffusion characteristics present
- May be diffusion-generated or different AI method
- Further analysis recommended

**Diffusion Likelihood: 0.0 - 0.4 (Low)**
- Few diffusion fingerprints detected
- Likely real video or non-diffusion AI method
- Other generation methods possible (GAN, autoregressive, etc.)

## Scientific Background

### Frequency Domain Analysis

**Why Frequency Domain?**
- Many visual artifacts are more apparent in frequency domain
- Compression, generation, and processing leave spectral signatures
- Natural images follow 1/f power law (scale-invariance)

**Mathematical Foundation**:
```
Power Spectrum: P(f) = |F(x,y)|²
where F(x,y) is 2D Fourier Transform

Natural images: P(f) ∝ 1/f^α where α ≈ 2
```

### Diffusion Model Artifacts

**U-Net Architecture Issues**:
1. **Transposed Convolutions**: Create checkerboard patterns
   - Uneven overlap in upsampling
   - Visible at 2× upsampling rates

2. **Skip Connections**: Alter multi-scale energy distribution
   - Combine features from encoder and decoder
   - Create characteristic wavelet signatures

**Iterative Denoising**:
- Starts with Gaussian noise
- Iteratively removes noise over T steps
- Residual noise patterns remain

**Video-Specific Artifacts**:
- Chunk-based processing (8-16 frames)
- Discontinuities at chunk boundaries
- Temporal inconsistencies

## Research Applications

### 1. Model Interpretability
```python
# Understand what frequency features drive decisions
results = analyzer.correlate_spectrum_with_model(video_path)
interpretation = results['interpretation']
```

### 2. Adversarial Analysis
```python
# Identify frequency bands to target for adversarial attacks
correlation = results['correlation']
vulnerable_frequencies = correlation['freq_attention_ratio']
```

### 3. Dataset Analysis
```python
# Analyze entire dataset for spectral characteristics
from spectral_analysis import SpectralAnalyzer

analyzer = SpectralAnalyzer()
dataset_stats = []

for video_path in dataset:
    results = analyzer.correlate_spectrum_with_model(video_path)
    dataset_stats.append(results['spectral_features'])

# Compare real vs AI distributions
```

### 4. Model Improvement
```python
# Use findings to augment training data
# - Add frequency domain augmentations
# - Target specific artifacts model focuses on
# - Balance frequency spectrum across classes
```

## Example Workflow

### Complete Analysis Pipeline

```python
from spectral_analysis import SpectralAnalyzer, compare_real_vs_ai_spectra
from diffusion_fingerprints import DiffusionFingerprintDetector

# 1. Initialize
spectral_analyzer = SpectralAnalyzer()
fingerprint_detector = DiffusionFingerprintDetector()

# 2. Single video analysis
video_path = 'test_video.mp4'

# Spectral analysis
spectral_results = spectral_analyzer.visualize_spectral_analysis(video_path)

# Diffusion fingerprints
fingerprint_results = fingerprint_detector.visualize_fingerprints(video_path)

# 3. Compare real vs AI
real_video = 'real_video.mp4'
ai_video = 'ai_video.mp4'

comparison = compare_real_vs_ai_spectra(real_video, ai_video)

# 4. Extract insights
print(f"Model Prediction: {spectral_results['prediction']['class']}")
print(f"Confidence: {spectral_results['prediction']['confidence']:.2%}")
print(f"Diffusion Likelihood: {fingerprint_results['diffusion_score']:.2f}")
print(f"Freq Attention Ratio: {spectral_results['correlation']['freq_attention_ratio']:.2f}")

# 5. Interpretation
for finding in fingerprint_results['interpretation']:
    print(f"  {finding}")
```

## Performance Considerations

- **FFT Computation**: O(N log N) per frame
- **Wavelet Transform**: O(N) per frame
- **IG Attribution**: Slow (~30-60 seconds per video)
- **Memory Usage**: ~2-4GB for spectral analysis

**Optimization Tips**:
- Sample frames instead of analyzing all
- Use lower resolution for initial analysis
- Cache attribution results
- Parallelize batch processing

## Limitations

1. **Heuristic Thresholds**: Detection thresholds are empirically determined
2. **Generator-Specific**: Some fingerprints are specific to certain diffusion models
3. **Resolution Dependent**: Some artifacts only visible at specific resolutions
4. **Post-Processing**: Videos may be compressed/filtered, hiding artifacts

## Future Work

1. **Learned Fingerprints**: Train classifier on spectral features
2. **Model-Agnostic Detection**: Extend beyond diffusion models
3. **Temporal Spectral Analysis**: 3D FFT for spatiotemporal patterns
4. **Adversarial Robustness**: Test against anti-forensic techniques
5. **Real-Time Analysis**: Optimize for streaming applications

## References

### Diffusion Model Artifacts
- "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
- "Understanding Artifacts in Diffusion Models" (various papers)

### Frequency Domain Forensics
- "Image Forgery Detection Using Frequency Domain Analysis"
- "Natural Image Statistics in Frequency Domain"

### Video Analysis
- "Temporal Consistency in Video Generation Models"
- "Detecting AI-Generated Videos via Temporal Artifacts"

## Citation

If you use this work in your research:

```bibtex
@misc{spectral_analysis_aigen2024,
  title={Spectral Analysis and Diffusion Fingerprint Detection for AI-Generated Videos},
  author={Joshua Weg},
  year={2024},
  howpublished={https://github.com/Joshuaweg/AIgendetector}
}
```

## Support

For questions about spectral analysis:
- Check visualizations in `spectral_analysis/` directory
- Review example outputs
- Open GitHub issue with specific questions

---

**Happy analyzing! 🔬📊**
