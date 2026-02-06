"""
Detailed comparison between specific real and AI videos
Helps understand why both might have similar diffusion likelihoods
"""

from spectral_analysis import SpectralAnalyzer
from diffusion_fingerprints import DiffusionFingerprintDetector
from interpret import load_video
import matplotlib.pyplot as plt
import numpy as np

def deep_compare(real_video_path, ai_video_path):
    """
    Perform detailed comparison to understand similarities/differences

    Args:
        real_video_path: path to real video
        ai_video_path: path to AI video
    """
    analyzer = SpectralAnalyzer()
    detector = DiffusionFingerprintDetector()

    print("="*80)
    print("DETAILED COMPARISON: Real vs AI Video")
    print("="*80)

    # Analyze both videos
    print("\n1. Analyzing REAL video...")
    real_results = analyzer.correlate_spectrum_with_model(real_video_path)
    _, real_frames, _, _ = load_video(real_video_path)
    real_fingerprints = detector.comprehensive_analysis(real_video_path)

    print("\n2. Analyzing AI video...")
    ai_results = analyzer.correlate_spectrum_with_model(ai_video_path)
    _, ai_frames, _, _ = load_video(ai_video_path)
    ai_fingerprints = detector.comprehensive_analysis(ai_video_path)

    # Compare predictions
    print("\n" + "="*80)
    print("MODEL PREDICTIONS")
    print("="*80)
    print(f"\nREAL Video:")
    print(f"  Prediction: {real_results['prediction']['class']}")
    print(f"  Confidence: {real_results['prediction']['confidence']:.2%}")
    print(f"\nAI Video:")
    print(f"  Prediction: {ai_results['prediction']['class']}")
    print(f"  Confidence: {ai_results['prediction']['confidence']:.2%}")

    # Compare spectral features
    print("\n" + "="*80)
    print("SPECTRAL FEATURES COMPARISON")
    print("="*80)

    real_sf = real_results['spectral_features']
    ai_sf = ai_results['spectral_features']

    print(f"\nDiffusion Likelihood:")
    print(f"  Real: {real_sf['diffusion_fingerprint']['diffusion_likelihood']:.3f}")
    print(f"  AI:   {ai_sf['diffusion_fingerprint']['diffusion_likelihood']:.3f}")
    print(f"  Difference: {abs(real_sf['diffusion_fingerprint']['diffusion_likelihood'] - ai_sf['diffusion_fingerprint']['diffusion_likelihood']):.3f}")

    print(f"\nCheckerboard Score:")
    print(f"  Real: {real_sf['diffusion_fingerprint']['checkerboard_score']:.3f}")
    print(f"  AI:   {ai_sf['diffusion_fingerprint']['checkerboard_score']:.3f}")

    print(f"\nNoise Uniformity:")
    print(f"  Real: {real_sf['diffusion_fingerprint']['noise_uniformity']:.3f}")
    print(f"  AI:   {ai_sf['diffusion_fingerprint']['noise_uniformity']:.3f}")

    print(f"\nFrequency Attention Ratio:")
    print(f"  Real: {real_results['correlation']['freq_attention_ratio']:.3f}")
    print(f"  AI:   {ai_results['correlation']['freq_attention_ratio']:.3f}")

    print(f"\nSpectrum Correlation:")
    print(f"  Real: {real_results['correlation']['attr_spectrum_corr']:.3f}")
    print(f"  AI:   {ai_results['correlation']['attr_spectrum_corr']:.3f}")

    # Compare fingerprints in detail
    print("\n" + "="*80)
    print("DIFFUSION FINGERPRINTS DETAILED")
    print("="*80)

    real_fp = real_fingerprints['fingerprints']
    ai_fp = ai_fingerprints['fingerprints']

    print(f"\nCheckerboard Detection:")
    print(f"  Real: {real_fp['checkerboard']['detection']} (score: {real_fp['checkerboard']['mean_score']:.2f})")
    print(f"  AI:   {ai_fp['checkerboard']['detection']} (score: {ai_fp['checkerboard']['mean_score']:.2f})")

    print(f"\nGaussian Noise:")
    print(f"  Real: {real_fp['noise_residuals']['is_gaussian_noise']} (likelihood: {real_fp['noise_residuals']['diffusion_likelihood']:.2f})")
    print(f"  AI:   {ai_fp['noise_residuals']['is_gaussian_noise']} (likelihood: {ai_fp['noise_residuals']['diffusion_likelihood']:.2f})")

    print(f"\nTemporal Chunk Artifacts:")
    print(f"  Real: {real_fp['temporal']['has_chunk_artifacts']} ({real_fp['temporal']['num_discontinuities']} discontinuities)")
    print(f"  AI:   {ai_fp['temporal']['has_chunk_artifacts']} ({ai_fp['temporal']['num_discontinuities']} discontinuities)")

    print(f"\nWavelet Mid-Level Indicator:")
    print(f"  Real: {real_fp['wavelet']['diffusion_indicator']} (ratio: {real_fp['wavelet']['mid_level_ratio']:.2f})")
    print(f"  AI:   {ai_fp['wavelet']['diffusion_indicator']} (ratio: {ai_fp['wavelet']['mid_level_ratio']:.2f})")

    # Overall scores
    print("\n" + "="*80)
    print("OVERALL DIFFUSION SCORES")
    print("="*80)
    print(f"\nReal video: {real_fingerprints['diffusion_score']:.3f}")
    print(f"AI video:   {ai_fingerprints['diffusion_score']:.3f}")
    print(f"Difference: {abs(real_fingerprints['diffusion_score'] - ai_fingerprints['diffusion_score']):.3f}")

    # Key differences
    print("\n" + "="*80)
    print("KEY DISCRIMINATING FEATURES")
    print("="*80)

    discriminators = []

    # Check each metric
    if abs(real_fp['checkerboard']['mean_score'] - ai_fp['checkerboard']['mean_score']) > 0.5:
        discriminators.append({
            'metric': 'Checkerboard Score',
            'real': real_fp['checkerboard']['mean_score'],
            'ai': ai_fp['checkerboard']['mean_score'],
            'diff': abs(real_fp['checkerboard']['mean_score'] - ai_fp['checkerboard']['mean_score'])
        })

    if real_fp['temporal']['has_chunk_artifacts'] != ai_fp['temporal']['has_chunk_artifacts']:
        discriminators.append({
            'metric': 'Temporal Chunks',
            'real': real_fp['temporal']['num_discontinuities'],
            'ai': ai_fp['temporal']['num_discontinuities'],
            'diff': abs(real_fp['temporal']['num_discontinuities'] - ai_fp['temporal']['num_discontinuities'])
        })

    if abs(real_results['correlation']['freq_attention_ratio'] - ai_results['correlation']['freq_attention_ratio']) > 0.5:
        discriminators.append({
            'metric': 'Freq Attention Ratio',
            'real': real_results['correlation']['freq_attention_ratio'],
            'ai': ai_results['correlation']['freq_attention_ratio'],
            'diff': abs(real_results['correlation']['freq_attention_ratio'] - ai_results['correlation']['freq_attention_ratio'])
        })

    if discriminators:
        print("\nMost discriminating features:")
        for d in sorted(discriminators, key=lambda x: x['diff'], reverse=True):
            print(f"\n  {d['metric']}:")
            print(f"    Real: {d['real']:.3f}")
            print(f"    AI:   {d['ai']:.3f}")
            print(f"    Difference: {d['diff']:.3f}")
    else:
        print("\n⚠️  WARNING: Videos have very similar spectral characteristics!")
        print("   This could indicate:")
        print("   1. Real video has heavy post-processing (compression, upscaling)")
        print("   2. AI video is very high quality / well-trained model")
        print("   3. Threshold calibration needed")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if real_fingerprints['diffusion_score'] > 0.6 and ai_fingerprints['diffusion_score'] > 0.6:
        print("\n⚠️  Both videos show high diffusion likelihood!")
        print("\nPossible explanations:")
        print("  1. Real video may have undergone heavy processing:")
        print("     - H.264/H.265 compression artifacts")
        print("     - Upscaling (creates checkerboard patterns)")
        print("     - Denoising filters (Gaussian-like residuals)")
        print("     - Multiple re-encodings")
        print("\n  2. Consider using COMPLEMENTARY features:")
        print("     - Model confidence (check if model is more confident on one)")
        print("     - Temporal consistency patterns")
        print("     - Multiple fingerprint voting (2/3 rule)")
        print("\n  3. Run threshold calibration:")
        print("     python calibrate_thresholds.py")

    # Model confidence comparison
    conf_diff = abs(real_results['prediction']['confidence'] - ai_results['prediction']['confidence'])
    if conf_diff > 0.15:
        print(f"\n✓ Model shows clear confidence difference: {conf_diff:.2%}")
        print(f"  More confident on: {'Real' if real_results['prediction']['confidence'] > ai_results['prediction']['confidence'] else 'AI'} video")
        print("  → Model may be relying on features beyond spectral analysis")

    # Create side-by-side visualization
    create_comparison_plot(real_results, ai_results, real_fingerprints, ai_fingerprints)

    return real_results, ai_results, real_fingerprints, ai_fingerprints

def create_comparison_plot(real_results, ai_results, real_fp, ai_fp):
    """Create side-by-side comparison visualization"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Metrics to compare
    metrics = [
        ('Diffusion\nLikelihood',
         real_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood'],
         ai_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']),

        ('Checkerboard\nScore',
         real_fp['fingerprints']['checkerboard']['mean_score'],
         ai_fp['fingerprints']['checkerboard']['mean_score']),

        ('Freq Attention\nRatio',
         real_results['correlation']['freq_attention_ratio'],
         ai_results['correlation']['freq_attention_ratio']),

        ('Spectrum\nCorrelation',
         real_results['correlation']['attr_spectrum_corr'],
         ai_results['correlation']['attr_spectrum_corr']),

        ('Noise\nUniformity',
         real_results['spectral_features']['diffusion_fingerprint']['noise_uniformity'],
         ai_results['spectral_features']['diffusion_fingerprint']['noise_uniformity']),

        ('Temporal\nDiscontinuities',
         real_fp['fingerprints']['temporal']['num_discontinuities'],
         ai_fp['fingerprints']['temporal']['num_discontinuities']),

        ('Wavelet\nRatio',
         real_fp['fingerprints']['wavelet']['mid_level_ratio'],
         ai_fp['fingerprints']['wavelet']['mid_level_ratio']),

        ('Model\nConfidence',
         real_results['prediction']['confidence'],
         ai_results['prediction']['confidence'])
    ]

    for idx, (label, real_val, ai_val) in enumerate(metrics):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        # Bar plot
        bars = ax.bar(['Real', 'AI'], [real_val, ai_val], color=['green', 'red'], alpha=0.7, edgecolor='black')

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Value')
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Highlight if different
        if abs(real_val - ai_val) > 0.3:
            ax.set_facecolor('#ffffcc')

    plt.suptitle('Detailed Comparison: Real vs AI Video', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = 'detailed_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Update these paths
    real_video = "/media/joshua/WD_BLACK/Gen-Video/dataset/real_yplug_pre_train_0838697_55_10.mp4"
    ai_video = "/media/joshua/WD_BLACK/Gen-Video/dataset/ai_dynamiccrafter_DynamicCrafter_43162.mp4"

    deep_compare(real_video, ai_video)

    print("\n" + "="*80)
    print("Analysis complete! Check 'detailed_comparison.png' for visual comparison.")
    print("="*80)
