"""
Debug diffusion likelihood calculation
Shows exactly which conditions are triggered for each video
"""

import numpy as np
import scipy.fft as fft
import cv2
from interpret import load_video

def debug_diffusion_likelihood(video_path, label="Video"):
    """
    Step-by-step breakdown of diffusion likelihood calculation

    Args:
        video_path: path to video
        label: label for output (e.g., "Real" or "AI")
    """
    print(f"\n{'='*80}")
    print(f"DEBUGGING DIFFUSION LIKELIHOOD: {label}")
    print(f"{'='*80}")

    # Load video
    _, frames, _, _ = load_video(video_path)
    print(f"Loaded {len(frames)} frames")

    # Extract FFT features
    print("\n1. Computing FFT features...")
    fft_magnitudes = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        f_transform = fft.fft2(gray)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        fft_magnitudes.append(magnitude_spectrum)

    fft_magnitudes = np.array(fft_magnitudes)
    power_spectrum = np.mean(fft_magnitudes ** 2, axis=0)

    # Compute radial profile
    h, w = power_spectrum.shape
    center_h, center_w = h // 2, w // 2

    y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
    r = np.sqrt(x*x + y*y).astype(int)

    max_r = min(center_h, center_w)
    radial_profile = []

    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile.append(power_spectrum[mask].mean())

    radial_profile = np.array(radial_profile)
    print(f"   Radial profile length: {len(radial_profile)}")

    # Compute checkerboard score
    print("\n2. Computing checkerboard score...")
    high_freq_region = power_spectrum[center_h+center_h//2:, center_w+center_w//2:]
    checkerboard_score = np.std(high_freq_region) / (np.mean(high_freq_region) + 1e-8)
    print(f"   Checkerboard score: {checkerboard_score:.4f}")
    print(f"   Threshold: 2.0")
    print(f"   Triggered: {checkerboard_score > 2.0}")

    # Compute noise uniformity
    print("\n3. Computing noise uniformity...")
    if len(radial_profile) > 10:
        high_freq_start = len(radial_profile) * 2 // 3
        high_freq_power = radial_profile[high_freq_start:]
        expected_decay = radial_profile[high_freq_start] * np.exp(-0.1 * np.arange(len(high_freq_power)))
        noise_uniformity = np.std(high_freq_power - expected_decay)
    else:
        noise_uniformity = 0

    print(f"   Noise uniformity: {noise_uniformity:.4f}")
    print(f"   Threshold: 0.5")
    print(f"   Triggered: {noise_uniformity > 0.5}")

    # Compute power law deviation
    print("\n4. Computing power law deviation...")
    score = 0.0

    if len(radial_profile) > 10:
        frequencies = np.arange(1, len(radial_profile) + 1)
        log_freq = np.log(frequencies[1:])
        log_power = np.log(radial_profile[1:] + 1e-8)

        coeffs = np.polyfit(log_freq, log_power, 1)
        slope = coeffs[0]
        expected_slope = -1.0
        slope_deviation = abs(slope - expected_slope)

        print(f"   Observed slope: {slope:.4f}")
        print(f"   Expected slope: {expected_slope:.4f}")
        print(f"   Deviation: {slope_deviation:.4f}")
        print(f"   Threshold: 0.5")
        print(f"   Triggered: {slope_deviation > 0.5}")

    # Calculate final score step by step
    print(f"\n{'='*80}")
    print("SCORE CALCULATION")
    print(f"{'='*80}")

    score = 0.0
    print(f"Starting score: {score}")

    if checkerboard_score > 2.0:
        score += 0.3
        print(f"+ 0.3 (checkerboard) → {score}")
    else:
        print(f"+ 0.0 (checkerboard below threshold) → {score}")

    if noise_uniformity > 0.5:
        score += 0.3
        print(f"+ 0.3 (noise uniformity) → {score}")
    else:
        print(f"+ 0.0 (noise uniformity below threshold) → {score}")

    if len(radial_profile) > 10:
        if slope_deviation > 0.5:
            score += 0.4
            print(f"+ 0.4 (power law violation) → {score}")
        else:
            print(f"+ 0.0 (power law OK) → {score}")

    final_score = min(score, 1.0)
    print(f"\nFinal score (capped at 1.0): {final_score}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Checkerboard: {checkerboard_score:.3f} {'✓' if checkerboard_score > 2.0 else '✗'}")
    print(f"Noise Uniformity: {noise_uniformity:.3f} {'✓' if noise_uniformity > 0.5 else '✗'}")
    print(f"Power Law Deviation: {slope_deviation:.3f} {'✓' if slope_deviation > 0.5 else '✗'}")
    print(f"Total Score: {final_score:.3f}")

    return {
        'checkerboard_score': checkerboard_score,
        'noise_uniformity': noise_uniformity,
        'slope': slope,
        'slope_deviation': slope_deviation,
        'final_score': final_score,
        'radial_profile': radial_profile
    }

def compare_scores(real_path, ai_path):
    """Compare scores side by side"""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)

    real_debug = debug_diffusion_likelihood(real_path, "REAL VIDEO")
    ai_debug = debug_diffusion_likelihood(ai_path, "AI VIDEO")

    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Real':<15} {'AI':<15} {'Difference':<15}")
    print(f"{'-'*70}")

    print(f"{'Checkerboard Score':<25} {real_debug['checkerboard_score']:<15.3f} {ai_debug['checkerboard_score']:<15.3f} {abs(real_debug['checkerboard_score'] - ai_debug['checkerboard_score']):<15.3f}")
    print(f"{'Noise Uniformity':<25} {real_debug['noise_uniformity']:<15.3f} {ai_debug['noise_uniformity']:<15.3f} {abs(real_debug['noise_uniformity'] - ai_debug['noise_uniformity']):<15.3f}")
    print(f"{'Power Law Slope':<25} {real_debug['slope']:<15.3f} {ai_debug['slope']:<15.3f} {abs(real_debug['slope'] - ai_debug['slope']):<15.3f}")
    print(f"{'Slope Deviation':<25} {real_debug['slope_deviation']:<15.3f} {ai_debug['slope_deviation']:<15.3f} {abs(real_debug['slope_deviation'] - ai_debug['slope_deviation']):<15.3f}")
    print(f"{'-'*70}")
    print(f"{'FINAL SCORE':<25} {real_debug['final_score']:<15.3f} {ai_debug['final_score']:<15.3f} {abs(real_debug['final_score'] - ai_debug['final_score']):<15.3f}")

    # Analyze why scores are similar
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    if abs(real_debug['final_score'] - ai_debug['final_score']) < 0.1:
        print("\n⚠️  SCORES ARE VERY SIMILAR!")
        print("\nReasons:")

        if real_debug['slope_deviation'] > 0.5 and ai_debug['slope_deviation'] > 0.5:
            print(f"  ✓ BOTH violate power law (contribute +0.4 each)")
            print(f"    Real slope: {real_debug['slope']:.3f} (deviation: {real_debug['slope_deviation']:.3f})")
            print(f"    AI slope:   {ai_debug['slope']:.3f} (deviation: {ai_debug['slope_deviation']:.3f})")
            print(f"    → This is likely the main issue!")

        if (real_debug['checkerboard_score'] > 2.0) == (ai_debug['checkerboard_score'] > 2.0):
            print(f"  ✓ Both have same checkerboard status")

        if (real_debug['noise_uniformity'] > 0.5) == (ai_debug['noise_uniformity'] > 0.5):
            print(f"  ✓ Both have same noise uniformity status")

        print("\n💡 RECOMMENDATION:")
        print("   The power law check may be too sensitive or not appropriate for")
        print("   compressed/processed videos. Consider:")
        print("   1. Adjusting slope deviation threshold (current: 0.5)")
        print("   2. Using different slope for processed videos")
        print("   3. Weighting checkerboard/noise more heavily")
        print("   4. Using multiple metrics with voting system")

    # Visualize radial profiles
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot radial profiles
    ax1 = axes[0]
    freq_real = np.arange(len(real_debug['radial_profile']))
    freq_ai = np.arange(len(ai_debug['radial_profile']))

    ax1.loglog(freq_real[1:], real_debug['radial_profile'][1:], 'b-', linewidth=2, label=f"Real (slope: {real_debug['slope']:.2f})")
    ax1.loglog(freq_ai[1:], ai_debug['radial_profile'][1:], 'r-', linewidth=2, label=f"AI (slope: {ai_debug['slope']:.2f})")

    # Expected 1/f
    expected = real_debug['radial_profile'][1] * (freq_real[1:] / freq_real[1]) ** -1.0
    ax1.loglog(freq_real[1:], expected, 'k--', linewidth=2, label='Expected (1/f)')

    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Power')
    ax1.set_title('Radial Power Spectrum Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot score breakdown
    ax2 = axes[1]
    metrics = ['Checkerboard\n(0.3 max)', 'Noise\n(0.3 max)', 'Power Law\n(0.4 max)']

    real_components = [
        0.3 if real_debug['checkerboard_score'] > 2.0 else 0,
        0.3 if real_debug['noise_uniformity'] > 0.5 else 0,
        0.4 if real_debug['slope_deviation'] > 0.5 else 0
    ]

    ai_components = [
        0.3 if ai_debug['checkerboard_score'] > 2.0 else 0,
        0.3 if ai_debug['noise_uniformity'] > 0.5 else 0,
        0.4 if ai_debug['slope_deviation'] > 0.5 else 0
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax2.bar(x - width/2, real_components, width, label='Real', color='blue', alpha=0.7)
    ax2.bar(x + width/2, ai_components, width, label='AI', color='red', alpha=0.7)

    ax2.set_ylabel('Score Contribution')
    ax2.set_title('Diffusion Score Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim([0, 0.5])
    ax2.grid(axis='y', alpha=0.3)

    # Add total scores as text
    ax2.text(0.02, 0.98, f"Real Total: {real_debug['final_score']:.2f}",
            transform=ax2.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(0.02, 0.88, f"AI Total: {ai_debug['final_score']:.2f}",
            transform=ax2.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig('diffusion_score_debug.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Debug visualization saved to: diffusion_score_debug.png")
    plt.close()

    return real_debug, ai_debug

if __name__ == "__main__":
    real_path = "/media/joshua/WD_BLACK/Gen-Video/dataset/real_yplug_pre_train_0838697_55_10.mp4"
    ai_path = "/media/joshua/WD_BLACK/Gen-Video/dataset/ai_dynamiccrafter_DynamicCrafter_43162.mp4"

    compare_scores(real_path, ai_path)

    print("\n" + "="*80)
    print("Debug complete! Check 'diffusion_score_debug.png' for visualization.")
    print("="*80)
