"""
Calibrate detection thresholds for spectral analysis
Analyzes multiple real and AI videos to find optimal thresholds
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spectral_analysis import SpectralAnalyzer
from diffusion_fingerprints import DiffusionFingerprintDetector
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_dataset(real_dir, ai_dir, num_samples=20):
    """
    Analyze real and AI videos to find distinguishing thresholds

    Args:
        real_dir: directory with real videos
        ai_dir: directory with AI videos
        num_samples: number of videos to analyze per class
    """
    analyzer = SpectralAnalyzer()
    detector = DiffusionFingerprintDetector()

    results = []

    # Analyze real videos
    print("Analyzing REAL videos...")
    real_videos = list(Path(real_dir).glob('*.mp4'))[:num_samples]

    for video_path in tqdm(real_videos, desc="Real videos"):
        try:
            # Spectral analysis
            spectral = analyzer.correlate_spectrum_with_model(str(video_path))

            # Diffusion fingerprints
            from interpret import load_video
            _, frames, _, _ = load_video(str(video_path))

            checkerboard = detector.detect_checkerboard_artifacts(frames)
            noise = detector.analyze_noise_residuals(frames)
            temporal = detector.detect_temporal_inconsistencies(frames)

            results.append({
                'video': video_path.name,
                'true_label': 'real',
                'prediction': spectral['prediction']['class'],
                'confidence': spectral['prediction']['confidence'],
                'diffusion_likelihood': spectral['spectral_features']['diffusion_fingerprint']['diffusion_likelihood'],
                'freq_attention_ratio': spectral['correlation']['freq_attention_ratio'],
                'spectrum_correlation': spectral['correlation']['attr_spectrum_corr'],
                'checkerboard_score': checkerboard['mean_score'],
                'gaussian_noise': noise['is_gaussian_noise'],
                'noise_uniformity': noise['mean_gaussian_score'],
                'temporal_chunks': temporal['has_chunk_artifacts'],
                'num_discontinuities': temporal['num_discontinuities']
            })
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    # Analyze AI videos
    print("\nAnalyzing AI videos...")
    ai_videos = list(Path(ai_dir).glob('*.mp4'))[:num_samples]

    for video_path in tqdm(ai_videos, desc="AI videos"):
        try:
            # Spectral analysis
            spectral = analyzer.correlate_spectrum_with_model(str(video_path))

            # Diffusion fingerprints
            from interpret import load_video
            _, frames, _, _ = load_video(str(video_path))

            checkerboard = detector.detect_checkerboard_artifacts(frames)
            noise = detector.analyze_noise_residuals(frames)
            temporal = detector.detect_temporal_inconsistencies(frames)

            results.append({
                'video': video_path.name,
                'true_label': 'ai',
                'prediction': spectral['prediction']['class'],
                'confidence': spectral['prediction']['confidence'],
                'diffusion_likelihood': spectral['spectral_features']['diffusion_fingerprint']['diffusion_likelihood'],
                'freq_attention_ratio': spectral['correlation']['freq_attention_ratio'],
                'spectrum_correlation': spectral['correlation']['attr_spectrum_corr'],
                'checkerboard_score': checkerboard['mean_score'],
                'gaussian_noise': noise['is_gaussian_noise'],
                'noise_uniformity': noise['mean_gaussian_score'],
                'temporal_chunks': temporal['has_chunk_artifacts'],
                'num_discontinuities': temporal['num_discontinuities']
            })
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    return pd.DataFrame(results)

def find_optimal_thresholds(df):
    """Find thresholds that best separate real from AI"""

    real_df = df[df['true_label'] == 'real']
    ai_df = df[df['true_label'] == 'ai']

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Compare distributions
    metrics = [
        'diffusion_likelihood',
        'freq_attention_ratio',
        'spectrum_correlation',
        'checkerboard_score',
        'noise_uniformity',
        'num_discontinuities'
    ]

    optimal_thresholds = {}

    for metric in metrics:
        real_vals = real_df[metric].values
        ai_vals = ai_df[metric].values

        print(f"\n{metric.upper()}:")
        print(f"  Real: mean={np.mean(real_vals):.3f}, std={np.std(real_vals):.3f}")
        print(f"  AI:   mean={np.mean(ai_vals):.3f}, std={np.std(ai_vals):.3f}")

        # Find threshold that maximizes separation
        all_vals = np.concatenate([real_vals, ai_vals])
        all_vals_sorted = np.sort(all_vals)

        best_threshold = None
        best_accuracy = 0

        for threshold in all_vals_sorted:
            # Classify based on threshold
            real_correct = np.sum(real_vals < threshold)
            ai_correct = np.sum(ai_vals >= threshold)
            accuracy = (real_correct + ai_correct) / len(df)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        optimal_thresholds[metric] = {
            'threshold': best_threshold,
            'accuracy': best_accuracy
        }

        print(f"  Optimal threshold: {best_threshold:.3f} (accuracy: {best_accuracy:.2%})")

    return optimal_thresholds

def visualize_distributions(df, save_path='threshold_analysis.png'):
    """Create visualization of metric distributions"""

    real_df = df[df['true_label'] == 'real']
    ai_df = df[df['true_label'] == 'ai']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        ('diffusion_likelihood', 'Diffusion Likelihood'),
        ('freq_attention_ratio', 'Frequency Attention Ratio'),
        ('spectrum_correlation', 'Spectrum Correlation'),
        ('checkerboard_score', 'Checkerboard Score'),
        ('noise_uniformity', 'Noise Uniformity'),
        ('num_discontinuities', 'Temporal Discontinuities')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]

        real_vals = real_df[metric].values
        ai_vals = ai_df[metric].values

        # Histogram
        ax.hist(real_vals, bins=15, alpha=0.6, label='Real', color='green', edgecolor='black')
        ax.hist(ai_vals, bins=15, alpha=0.6, label='AI', color='red', edgecolor='black')

        # Mean lines
        ax.axvline(np.mean(real_vals), color='darkgreen', linestyle='--', linewidth=2, label=f'Real Mean: {np.mean(real_vals):.2f}')
        ax.axvline(np.mean(ai_vals), color='darkred', linestyle='--', linewidth=2, label=f'AI Mean: {np.mean(ai_vals):.2f}')

        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution Comparison: Real vs AI Videos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution visualization saved to: {save_path}")
    plt.close()

def generate_recommendations(optimal_thresholds, df):
    """Generate calibrated recommendations"""

    print("\n" + "="*60)
    print("CALIBRATED THRESHOLDS & RECOMMENDATIONS")
    print("="*60)

    print("\nRecommended threshold updates for spectral_analysis.py:\n")

    print("In _compute_diffusion_likelihood() method:")
    print("```python")
    print("# Current vs Recommended thresholds:")

    dl_threshold = optimal_thresholds['diffusion_likelihood']['threshold']
    cb_threshold = optimal_thresholds['checkerboard_score']['threshold']
    nu_threshold = optimal_thresholds['noise_uniformity']['threshold']

    print(f"# Checkerboard: 2.0 → {cb_threshold:.2f}")
    print(f"if checkerboard_score > {cb_threshold:.2f}:")
    print(f"    score += 0.3")
    print()
    print(f"# Noise uniformity: 0.5 → {nu_threshold:.2f}")
    print(f"if noise_uniformity > {nu_threshold:.2f}:")
    print(f"    score += 0.3")
    print("```")

    print("\n" + "-"*60)
    print("MULTI-METRIC CLASSIFIER")
    print("-"*60)

    # Combined classifier using multiple metrics
    real_df = df[df['true_label'] == 'real']
    ai_df = df[df['true_label'] == 'ai']

    # Test combined approach
    correct_real = 0
    correct_ai = 0

    for _, row in real_df.iterrows():
        score = 0
        if row['checkerboard_score'] > cb_threshold:
            score += 1
        if row['temporal_chunks']:
            score += 1
        if row['diffusion_likelihood'] > dl_threshold:
            score += 1

        if score < 2:  # Predict real
            correct_real += 1

    for _, row in ai_df.iterrows():
        score = 0
        if row['checkerboard_score'] > cb_threshold:
            score += 1
        if row['temporal_chunks']:
            score += 1
        if row['diffusion_likelihood'] > dl_threshold:
            score += 1

        if score >= 2:  # Predict AI
            correct_ai += 1

    total_accuracy = (correct_real + correct_ai) / len(df)

    print(f"\nMulti-metric classifier (2/3 votes):")
    print(f"  Real accuracy: {correct_real}/{len(real_df)} = {correct_real/len(real_df):.2%}")
    print(f"  AI accuracy: {correct_ai}/{len(ai_df)} = {correct_ai/len(ai_df):.2%}")
    print(f"  Overall accuracy: {total_accuracy:.2%}")

if __name__ == "__main__":
    base_dir = '/media/joshua/WD_BLACK/Gen-Video/dataset'

    # Update these paths to your dataset structure
    real_dir = base_dir  # Directory with real videos
    ai_dir = base_dir    # Directory with AI videos

    print("="*60)
    print("Spectral Analysis Threshold Calibration")
    print("="*60)
    print(f"\nAnalyzing videos from: {base_dir}")
    print("This will analyze 20 real and 20 AI videos to find optimal thresholds.\n")

    # Analyze dataset
    df = analyze_dataset(real_dir, ai_dir, num_samples=10)  # Start with 10 each

    # Save results
    df.to_csv('threshold_calibration_results.csv', index=False)
    print("\n✓ Results saved to: threshold_calibration_results.csv")

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(df)

    # Visualize
    visualize_distributions(df)

    # Generate recommendations
    generate_recommendations(optimal_thresholds, df)

    print("\n" + "="*60)
    print("Calibration complete!")
    print("="*60)
