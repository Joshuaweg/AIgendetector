"""
Spectral Analysis for AI-Generated Video Detection
Analyzes frequency domain patterns and diffusion fingerprints in videos

This module provides tools to:
1. Extract frequency domain features (FFT, DCT)
2. Detect diffusion model fingerprints
3. Compare model attention with spectral patterns
4. Visualize frequency-based artifacts
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.fft as fft
from scipy.signal import stft, welch
from pathlib import Path
import os

from full_scale_classifier import FullVideoClassifier, FullLatentEncoder, FullPatchEncoder, FullClassifier
from interpret import load_video, load_model_correctly
from captum.attr import IntegratedGradients


class SpectralAnalyzer:
    """
    Analyzes frequency domain characteristics of videos
    to detect AI generation patterns
    """

    def __init__(self, model_path=None):
        """Initialize analyzer with model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            base_dir = '/media/joshua/WD_BLACK/Gen-Video'
            model_path = os.path.join(base_dir, 'model', 'full_classifier_1_85.pt')

        self.model = load_model_correctly(model_path, self.device)
        self.model.eval()

    def extract_fft_features(self, frames):
        """
        Extract 2D FFT features from video frames

        Args:
            frames: numpy array of shape (num_frames, H, W, C)

        Returns:
            dict with FFT magnitude spectra and statistics
        """
        fft_magnitudes = []

        for frame in frames:
            # Convert to grayscale for FFT analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Apply 2D FFT
            f_transform = fft.fft2(gray)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            fft_magnitudes.append(magnitude_spectrum)

        fft_magnitudes = np.array(fft_magnitudes)

        # Compute statistics
        return {
            'magnitudes': fft_magnitudes,
            'mean_spectrum': np.mean(fft_magnitudes, axis=0),
            'std_spectrum': np.std(fft_magnitudes, axis=0),
            'power_spectrum': np.mean(fft_magnitudes ** 2, axis=0)
        }

    def extract_dct_features(self, frames):
        """
        Extract DCT (Discrete Cosine Transform) features
        DCT is commonly used in image compression (JPEG)
        AI models may leave artifacts in DCT domain

        Args:
            frames: numpy array of shape (num_frames, H, W, C)

        Returns:
            dict with DCT coefficients and analysis
        """
        dct_coeffs = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = frame.astype(np.float32)

            # Apply 2D DCT
            dct = cv2.dct(gray)
            dct_coeffs.append(dct)

        dct_coeffs = np.array(dct_coeffs)

        # Analyze low vs high frequency components
        h, w = dct_coeffs.shape[1], dct_coeffs.shape[2]

        # Divide into frequency bands
        low_freq = dct_coeffs[:, :h//4, :w//4]  # Low frequencies (top-left)
        mid_freq = dct_coeffs[:, h//4:h//2, w//4:w//2]  # Mid frequencies
        high_freq = dct_coeffs[:, h//2:, w//2:]  # High frequencies (bottom-right)

        return {
            'coefficients': dct_coeffs,
            'mean_dct': np.mean(dct_coeffs, axis=0),
            'low_freq_energy': np.mean(np.abs(low_freq)),
            'mid_freq_energy': np.mean(np.abs(mid_freq)),
            'high_freq_energy': np.mean(np.abs(high_freq)),
            'freq_ratio': np.mean(np.abs(high_freq)) / (np.mean(np.abs(low_freq)) + 1e-8)
        }

    def detect_diffusion_fingerprint(self, frames):
        """
        Detect spectral fingerprints characteristic of diffusion models

        Diffusion models often leave specific patterns:
        1. Gaussian noise residuals in high frequencies
        2. Periodic patterns from iterative denoising
        3. Checkerboard artifacts in certain frequency bands

        Args:
            frames: numpy array of shape (num_frames, H, W, C)

        Returns:
            dict with diffusion fingerprint analysis
        """
        # Extract FFT features
        fft_features = self.extract_fft_features(frames)
        power_spectrum = fft_features['power_spectrum']

        h, w = power_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # 1. Analyze radial power spectrum
        y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
        r = np.sqrt(x*x + y*y).astype(int)

        max_r = min(center_h, center_w)
        radial_profile = []

        for radius in range(max_r):
            mask = (r == radius)
            if mask.sum() > 0:
                radial_profile.append(power_spectrum[mask].mean())

        radial_profile = np.array(radial_profile)

        # 2. Detect checkerboard artifacts (alternating pattern in high freq)
        # Check for periodic patterns at specific frequencies
        high_freq_region = power_spectrum[center_h+center_h//2:, center_w+center_w//2:]
        checkerboard_score = np.std(high_freq_region) / (np.mean(high_freq_region) + 1e-8)

        # 3. Gaussian noise detection in high frequencies
        # Real videos: power law decay (1/f noise)
        # AI videos: more uniform noise distribution
        if len(radial_profile) > 10:
            high_freq_start = len(radial_profile) * 2 // 3
            high_freq_power = radial_profile[high_freq_start:]

            # Measure deviation from expected power law
            expected_decay = radial_profile[high_freq_start] * np.exp(-0.1 * np.arange(len(high_freq_power)))
            noise_uniformity = np.std(high_freq_power - expected_decay)
        else:
            noise_uniformity = 0

        # 4. Temporal consistency in spectral domain
        temporal_spectra = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            f_transform = fft.fft2(gray)
            f_shift = fft.fftshift(f_transform)
            temporal_spectra.append(np.abs(f_shift))

        temporal_spectra = np.array(temporal_spectra)
        temporal_variance = np.var(temporal_spectra, axis=0)

        return {
            'radial_profile': radial_profile,
            'checkerboard_score': float(checkerboard_score),
            'noise_uniformity': float(noise_uniformity),
            'temporal_variance': temporal_variance,
            'diffusion_likelihood': self._compute_diffusion_likelihood(
                checkerboard_score, noise_uniformity, radial_profile
            )
        }

    def _compute_diffusion_likelihood(self, checkerboard_score, noise_uniformity, radial_profile):
        """
        Compute likelihood that video is from diffusion model
        based on spectral fingerprints
        """
        # Heuristic scoring (can be refined with training data)
        score = 0.0

        # High checkerboard artifacts suggest diffusion
        if checkerboard_score > 2.0:
            score += 0.3

        # Uniform high-frequency noise suggests diffusion
        if noise_uniformity > 0.5:
            score += 0.3

        # Check for power law violations
        if len(radial_profile) > 10:
            # Real videos follow power law decay (~1/f)
            # Fit power law and check residuals
            frequencies = np.arange(1, len(radial_profile) + 1)
            log_freq = np.log(frequencies[1:])
            log_power = np.log(radial_profile[1:] + 1e-8)

            # Linear fit in log-log space
            coeffs = np.polyfit(log_freq, log_power, 1)
            expected_slope = -1.0  # Power law exponent for natural images

            # Deviation from expected slope
            slope_deviation = abs(coeffs[0] - expected_slope)
            if slope_deviation > 0.5:
                score += 0.4

        return min(score, 1.0)

    def correlate_spectrum_with_model(self, video_path):
        """
        Correlate spectral features with model predictions and attention

        This reveals whether the model is attending to:
        - Low frequency features (overall structure, shapes)
        - High frequency features (textures, noise, artifacts)
        - Specific frequency bands characteristic of AI generation

        Args:
            video_path: path to video file

        Returns:
            dict with correlation analysis
        """
        # Load video
        video_tensor, frames, label, _ = load_video(video_path)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        video_tensor.requires_grad = True

        # Get model prediction
        with torch.no_grad():
            output = self.model(video_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()

        # Get attributions using Integrated Gradients
        baseline = torch.zeros_like(video_tensor)
        ig = IntegratedGradients(self.model)

        attributions = ig.attribute(
            video_tensor,
            baseline,
            target=pred,
            n_steps=50,
            internal_batch_size=1
        )

        # Move to numpy
        attributions_np = attributions.squeeze(0).detach().cpu().numpy()  # (frames, H, W, C)

        # Extract spectral features
        fft_features = self.extract_fft_features(frames)
        dct_features = self.extract_dct_features(frames)
        diffusion_fp = self.detect_diffusion_fingerprint(frames)

        # Compute attribution in frequency domain
        attribution_fft = []
        for frame_attr in attributions_np:
            # Average across channels
            attr_gray = np.mean(np.abs(frame_attr), axis=2)

            # FFT of attribution
            f_transform = fft.fft2(attr_gray)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            attribution_fft.append(magnitude)

        attribution_fft = np.array(attribution_fft)
        mean_attr_spectrum = np.mean(attribution_fft, axis=0)

        # Compute correlation between attribution and spectral features
        # Flatten for correlation
        attr_flat = mean_attr_spectrum.flatten()
        power_flat = fft_features['power_spectrum'].flatten()

        correlation = np.corrcoef(attr_flat, power_flat)[0, 1]

        # Analyze which frequency bands model attends to
        h, w = mean_attr_spectrum.shape
        center_h, center_w = h // 2, w // 2

        low_freq_attr = mean_attr_spectrum[center_h-h//8:center_h+h//8,
                                          center_w-w//8:center_w+w//8]
        high_freq_attr = np.concatenate([
            mean_attr_spectrum[:h//4, :].flatten(),
            mean_attr_spectrum[-h//4:, :].flatten(),
            mean_attr_spectrum[:, :w//4].flatten(),
            mean_attr_spectrum[:, -w//4:].flatten()
        ])

        return {
            'prediction': {
                'class': ['AI-Generated', 'Real'][pred],
                'confidence': confidence
            },
            'spectral_features': {
                'fft': fft_features,
                'dct': dct_features,
                'diffusion_fingerprint': diffusion_fp
            },
            'attribution_spectrum': mean_attr_spectrum,
            'correlation': {
                'attr_spectrum_corr': float(correlation),
                'low_freq_attention': float(np.mean(low_freq_attr)),
                'high_freq_attention': float(np.mean(high_freq_attr)),
                'freq_attention_ratio': float(np.mean(high_freq_attr) / (np.mean(low_freq_attr) + 1e-8))
            },
            'interpretation': self._interpret_spectral_correlation(
                correlation, np.mean(low_freq_attr), np.mean(high_freq_attr),
                diffusion_fp['diffusion_likelihood']
            )
        }

    def _interpret_spectral_correlation(self, correlation, low_freq_att, high_freq_att, diffusion_likelihood):
        """Generate human-readable interpretation"""
        interpretation = []

        if high_freq_att > low_freq_att:
            interpretation.append(
                f"Model primarily attends to HIGH frequency features "
                f"(ratio: {high_freq_att/low_freq_att:.2f}x). "
                f"This suggests focus on textures, noise patterns, and fine details."
            )
        else:
            interpretation.append(
                f"Model primarily attends to LOW frequency features "
                f"(ratio: {low_freq_att/high_freq_att:.2f}x). "
                f"This suggests focus on overall structure and shapes."
            )

        if correlation > 0.5:
            interpretation.append(
                f"Strong correlation ({correlation:.2f}) between model attention "
                f"and spectral power. Model decisions align with frequency domain patterns."
            )
        elif correlation < -0.3:
            interpretation.append(
                f"Negative correlation ({correlation:.2f}) suggests model attends to "
                f"regions with unusual spectral characteristics."
            )

        if diffusion_likelihood > 0.6:
            interpretation.append(
                f"High diffusion fingerprint likelihood ({diffusion_likelihood:.2f}). "
                f"Spectral analysis suggests diffusion model generation patterns."
            )

        return interpretation

    def visualize_spectral_analysis(self, video_path, save_dir='spectral_analysis'):
        """
        Create comprehensive visualization of spectral analysis

        Args:
            video_path: path to video file
            save_dir: directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)

        # Perform full analysis
        results = self.correlate_spectrum_with_model(video_path)

        # Load frames for visualization
        _, frames, _, _ = load_video(video_path)

        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Sample frame
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(frames[len(frames)//2])
        ax1.set_title('Sample Frame')
        ax1.axis('off')

        # 2. Power spectrum
        ax2 = plt.subplot(3, 3, 2)
        power = results['spectral_features']['fft']['power_spectrum']
        ax2.imshow(np.log(power + 1), cmap='hot')
        ax2.set_title('Power Spectrum (Log Scale)')
        ax2.axis('off')

        # 3. Attribution spectrum
        ax3 = plt.subplot(3, 3, 3)
        attr_spectrum = results['attribution_spectrum']
        ax3.imshow(np.log(attr_spectrum + 1), cmap='viridis')
        ax3.set_title('Model Attribution Spectrum')
        ax3.axis('off')

        # 4. Radial profile
        ax4 = plt.subplot(3, 3, 4)
        radial = results['spectral_features']['diffusion_fingerprint']['radial_profile']
        frequencies = np.arange(len(radial))
        ax4.loglog(frequencies[1:], radial[1:], 'b-', linewidth=2, label='Observed')

        # Plot expected 1/f decay
        if len(radial) > 1:
            expected = radial[1] * (frequencies[1:] / frequencies[1]) ** -1
            ax4.loglog(frequencies[1:], expected, 'r--', linewidth=2, label='Expected (1/f)')

        ax4.set_xlabel('Frequency (pixels)')
        ax4.set_ylabel('Power')
        ax4.set_title('Radial Power Spectrum')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. DCT frequency bands
        ax5 = plt.subplot(3, 3, 5)
        dct_features = results['spectral_features']['dct']
        freq_energies = [
            dct_features['low_freq_energy'],
            dct_features['mid_freq_energy'],
            dct_features['high_freq_energy']
        ]
        ax5.bar(['Low', 'Mid', 'High'], freq_energies, color=['blue', 'orange', 'red'])
        ax5.set_ylabel('Energy')
        ax5.set_title('DCT Frequency Band Energy')
        ax5.grid(axis='y', alpha=0.3)

        # 6. Correlation metrics
        ax6 = plt.subplot(3, 3, 6)
        corr = results['correlation']
        metrics = {
            'Spectrum\nCorrelation': corr['attr_spectrum_corr'],
            'Low Freq\nAttention': corr['low_freq_attention'],
            'High Freq\nAttention': corr['high_freq_attention']
        }
        colors = ['green' if v > 0 else 'red' for v in metrics.values()]
        ax6.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.7)
        ax6.set_xlabel('Value')
        ax6.set_title('Spectral Correlation Metrics')
        ax6.grid(axis='x', alpha=0.3)

        # 7. Diffusion fingerprint indicators
        ax7 = plt.subplot(3, 3, 7)
        diff_fp = results['spectral_features']['diffusion_fingerprint']
        indicators = {
            'Checkerboard\nScore': diff_fp['checkerboard_score'] / 5.0,  # Normalize
            'Noise\nUniformity': diff_fp['noise_uniformity'],
            'Diffusion\nLikelihood': diff_fp['diffusion_likelihood']
        }
        bars = ax7.bar(list(indicators.keys()), list(indicators.values()),
                       color=['purple', 'orange', 'red'], alpha=0.7)
        ax7.set_ylabel('Score')
        ax7.set_title('Diffusion Fingerprint Indicators')
        ax7.set_ylim([0, 1])
        ax7.grid(axis='y', alpha=0.3)

        # Add threshold line
        ax7.axhline(y=0.6, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax7.legend()

        # 8. Prediction info
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        pred_text = (
            f"Prediction: {results['prediction']['class']}\n"
            f"Confidence: {results['prediction']['confidence']:.2%}\n\n"
            f"Spectral Analysis:\n"
            f"- Diffusion Likelihood: {diff_fp['diffusion_likelihood']:.2f}\n"
            f"- Freq Attention Ratio: {corr['freq_attention_ratio']:.2f}\n"
            f"- Spectrum Correlation: {corr['attr_spectrum_corr']:.2f}\n\n"
            f"Interpretation:\n"
        )
        for i, interp in enumerate(results['interpretation'], 1):
            pred_text += f"{i}. {interp[:60]}...\n"

        ax8.text(0.1, 0.5, pred_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 9. Temporal variance in spectrum
        ax9 = plt.subplot(3, 3, 9)
        temporal_var = diff_fp['temporal_variance']
        ax9.imshow(np.log(temporal_var + 1), cmap='plasma')
        ax9.set_title('Temporal Spectral Variance')
        ax9.axis('off')

        plt.suptitle(f'Spectral Analysis: {Path(video_path).name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(save_dir, f'spectral_analysis_{Path(video_path).stem}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")

        plt.close()

        return results


def compare_real_vs_ai_spectra(real_video_path, ai_video_path, save_dir='spectral_comparison'):
    """
    Compare spectral characteristics between real and AI-generated videos

    Args:
        real_video_path: path to real video
        ai_video_path: path to AI-generated video
        save_dir: directory to save comparison
    """
    analyzer = SpectralAnalyzer()
    os.makedirs(save_dir, exist_ok=True)

    print("Analyzing real video...")
    real_results = analyzer.correlate_spectrum_with_model(real_video_path)

    print("Analyzing AI-generated video...")
    ai_results = analyzer.correlate_spectrum_with_model(ai_video_path)

    # Create comparison figure
    fig = plt.figure(figsize=(20, 10))

    # Real video analysis
    ax1 = plt.subplot(2, 4, 1)
    real_power = real_results['spectral_features']['fft']['power_spectrum']
    ax1.imshow(np.log(real_power + 1), cmap='hot')
    ax1.set_title('Real: Power Spectrum')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    real_radial = real_results['spectral_features']['diffusion_fingerprint']['radial_profile']
    freq = np.arange(len(real_radial))
    ax2.loglog(freq[1:], real_radial[1:], 'b-', linewidth=2, label='Real')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    ax2.set_title('Real: Radial Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 4, 3)
    real_attr = real_results['attribution_spectrum']
    ax3.imshow(np.log(real_attr + 1), cmap='viridis')
    ax3.set_title('Real: Attribution Spectrum')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    real_text = (
        f"REAL VIDEO\n\n"
        f"Prediction: {real_results['prediction']['class']}\n"
        f"Confidence: {real_results['prediction']['confidence']:.2%}\n\n"
        f"Diffusion Likelihood: {real_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']:.2f}\n"
        f"Freq Attention Ratio: {real_results['correlation']['freq_attention_ratio']:.2f}\n"
    )
    ax4.text(0.1, 0.5, real_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # AI video analysis
    ax5 = plt.subplot(2, 4, 5)
    ai_power = ai_results['spectral_features']['fft']['power_spectrum']
    ax5.imshow(np.log(ai_power + 1), cmap='hot')
    ax5.set_title('AI: Power Spectrum')
    ax5.axis('off')

    ax6 = plt.subplot(2, 4, 6)
    ai_radial = ai_results['spectral_features']['diffusion_fingerprint']['radial_profile']
    freq_ai = np.arange(len(ai_radial))
    ax6.loglog(freq_ai[1:], ai_radial[1:], 'r-', linewidth=2, label='AI')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Power')
    ax6.set_title('AI: Radial Profile')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(2, 4, 7)
    ai_attr = ai_results['attribution_spectrum']
    ax7.imshow(np.log(ai_attr + 1), cmap='viridis')
    ax7.set_title('AI: Attribution Spectrum')
    ax7.axis('off')

    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    ai_text = (
        f"AI-GENERATED VIDEO\n\n"
        f"Prediction: {ai_results['prediction']['class']}\n"
        f"Confidence: {ai_results['prediction']['confidence']:.2%}\n\n"
        f"Diffusion Likelihood: {ai_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']:.2f}\n"
        f"Freq Attention Ratio: {ai_results['correlation']['freq_attention_ratio']:.2f}\n"
    )
    ax8.text(0.1, 0.5, ai_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.suptitle('Spectral Comparison: Real vs AI-Generated', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(save_dir, 'real_vs_ai_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")

    plt.close()

    return real_results, ai_results


if __name__ == "__main__":
    # Example usage
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'

    # Analyze a single video
    video_path = os.path.join(base_dir, "dataset", "ai_dynamiccrafter_DynamicCrafter_43162.mp4")

    real_path = os.path.join(base_dir, "dataset", "real_yplug_pre_train_0838697_55_10.mp4")
    if os.path.exists(video_path):
        print("=" * 60)
        print("Spectral Analysis of AI-Generated Video vs Real Video Detection")
        print("=" * 60)

        analyzer = SpectralAnalyzer()
        results = analyzer.visualize_spectral_analysis(video_path)
        real_results = analyzer.visualize_spectral_analysis(real_path)

        print("\nAnalysis complete!")
        print(f"Prediction: {results['prediction']['class']}")
        print(f"Confidence: {results['prediction']['confidence']:.2%}")
        print(f"Diffusion Likelihood: {results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']:.2f}")
        print("\nInterpretation:")
        for i, interp in enumerate(results['interpretation'], 1):
            print(f"{i}. {interp}")
        print(f"Real Video Prediction: {real_results['prediction']['class']}")
        print(f"Real Video Confidence: {real_results['prediction']['confidence']:.2%}")
        print(f"Real Video Diffusion Likelihood: {real_results['spectral_features']['diffusion_fingerprint']['diffusion_likelihood']:.2f}")
        print("\nInterpretation:")
        for i, interp in enumerate(real_results['interpretation'], 1):
            print(f"{i}. {interp}")
    else:
        print(f"Video not found: {video_path}")
        print("Please update the video path in the script.")
