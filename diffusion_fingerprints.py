"""
Advanced Diffusion Model Fingerprint Detection

Implements specialized techniques to detect and analyze fingerprints
left by diffusion models (Stable Diffusion, DynamicCrafter, etc.)

Based on research showing diffusion models leave characteristic patterns:
1. Gaussian noise residuals in denoising process
2. U-Net upsampling artifacts (checkerboard patterns)
3. Temporal inconsistencies in video diffusion
4. Frequency domain signatures
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt  # Wavelet transform
from pathlib import Path
import os

from interpret import load_video


class DiffusionFingerprintDetector:
    """
    Detects specific fingerprints left by diffusion models
    """

    def __init__(self):
        pass

    def detect_checkerboard_artifacts(self, frames):
        """
        Detect checkerboard patterns from U-Net upsampling

        Diffusion models often use transposed convolutions which
        create characteristic checkerboard artifacts

        Args:
            frames: numpy array (num_frames, H, W, C)

        Returns:
            dict with checkerboard detection results
        """
        checkerboard_scores = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Method 1: FFT analysis for periodic patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # Look for peaks at specific frequencies (checkerboard pattern)
            # Checkerboard creates peaks at Nyquist frequency
            nyquist_region = magnitude[center_h-5:center_h+5, center_w-5:center_w+5]
            corners = np.concatenate([
                magnitude[:10, :10].flatten(),
                magnitude[:10, -10:].flatten(),
                magnitude[-10:, :10].flatten(),
                magnitude[-10:, -10:].flatten()
            ])

            peak_ratio = np.max(corners) / (np.median(magnitude) + 1e-8)
            checkerboard_scores.append(peak_ratio)

            # Method 2: Direct checkerboard kernel convolution
            checkerboard_kernel = np.array([
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
                [1, -1, 1, -1],
                [-1, 1, -1, 1]
            ], dtype=np.float32)

            response = cv2.filter2D(gray.astype(np.float32), -1, checkerboard_kernel)
            checkerboard_response = np.abs(response).mean()

        return {
            'scores': checkerboard_scores,
            'mean_score': np.mean(checkerboard_scores),
            'max_score': np.max(checkerboard_scores),
            'std_score': np.std(checkerboard_scores),
            'detection': np.mean(checkerboard_scores) > 1.5  # Threshold
        }

    def analyze_noise_residuals(self, frames):
        """
        Analyze high-frequency noise patterns

        Diffusion models denoise images iteratively, often leaving
        characteristic Gaussian noise patterns in residuals

        Args:
            frames: numpy array (num_frames, H, W, C)

        Returns:
            dict with noise analysis
        """
        noise_patterns = []

        for i, frame in enumerate(frames):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = frame.astype(np.float32)

            # Extract high-frequency noise using high-pass filter
            # Apply Gaussian blur and subtract to get high-freq components
            blurred = cv2.GaussianBlur(gray, (5, 5), 2)
            noise = gray - blurred

            # Analyze noise statistics
            noise_flat = noise.flatten()

            stats = {
                'mean': np.mean(noise_flat),
                'std': np.std(noise_flat),
                'skewness': skew(noise_flat),
                'kurtosis': kurtosis(noise_flat)
            }

            # Gaussian noise has:
            # - Mean ≈ 0
            # - Skewness ≈ 0
            # - Kurtosis ≈ 3 (excess kurtosis ≈ 0)

            # Deviation from Gaussian
            gaussian_score = (
                abs(stats['mean']) +
                abs(stats['skewness']) +
                abs(stats['kurtosis'] - 3)
            )

            noise_patterns.append({
                'stats': stats,
                'gaussian_score': gaussian_score
            })

        # Aggregate
        mean_gaussian_score = np.mean([p['gaussian_score'] for p in noise_patterns])

        # Lower score = more Gaussian = more likely diffusion
        is_gaussian = mean_gaussian_score < 1.0

        return {
            'patterns': noise_patterns,
            'mean_gaussian_score': mean_gaussian_score,
            'is_gaussian_noise': is_gaussian,
            'diffusion_likelihood': 1.0 - min(mean_gaussian_score / 2.0, 1.0)
        }

    def detect_temporal_inconsistencies(self, frames):
        """
        Detect temporal inconsistencies in video diffusion models

        Video diffusion models process frames in chunks, which can
        create temporal artifacts at chunk boundaries

        Args:
            frames: numpy array (num_frames, H, W, C)

        Returns:
            dict with temporal analysis
        """
        num_frames = len(frames)
        temporal_diffs = []

        # Compute frame-to-frame differences
        for i in range(num_frames - 1):
            frame1 = frames[i].astype(np.float32)
            frame2 = frames[i + 1].astype(np.float32)

            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2

            diff = np.abs(gray2 - gray1)
            temporal_diffs.append(np.mean(diff))

        temporal_diffs = np.array(temporal_diffs)

        # Look for discontinuities (sudden spikes in difference)
        # These occur at chunk boundaries in video diffusion
        if len(temporal_diffs) > 0:
            mean_diff = np.mean(temporal_diffs)
            std_diff = np.std(temporal_diffs)

            # Find anomalous transitions
            threshold = mean_diff + 2 * std_diff
            discontinuities = np.where(temporal_diffs > threshold)[0]

            # Check for periodic discontinuities (common chunk size: 8, 16 frames)
            if len(discontinuities) > 1:
                intervals = np.diff(discontinuities)
                periodic_score = np.std(intervals) / (np.mean(intervals) + 1e-8)
            else:
                periodic_score = 1.0
        else:
            discontinuities = []
            periodic_score = 1.0

        return {
            'temporal_diffs': temporal_diffs.tolist(),
            'mean_diff': float(np.mean(temporal_diffs)) if len(temporal_diffs) > 0 else 0,
            'std_diff': float(np.std(temporal_diffs)) if len(temporal_diffs) > 0 else 0,
            'discontinuity_frames': discontinuities.tolist(),
            'num_discontinuities': len(discontinuities),
            'periodic_score': float(periodic_score),
            'has_chunk_artifacts': len(discontinuities) > 2 and periodic_score < 0.3
        }

    def wavelet_analysis(self, frames):
        """
        Wavelet-based analysis for multi-scale artifacts

        Diffusion models can leave artifacts at multiple scales
        due to the U-Net architecture with skip connections

        Args:
            frames: numpy array (num_frames, H, W, C)

        Returns:
            dict with wavelet analysis
        """
        wavelet_features = []

        for frame in frames[:5]:  # Sample first 5 frames for speed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Multi-level 2D wavelet decomposition
            coeffs = pywt.wavedec2(gray, 'db4', level=3)

            # Analyze energy in different subbands
            cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

            # Energy in each subband
            energy = {
                'approx': np.sum(cA3 ** 2),
                'detail_level3': np.sum(cH3 ** 2) + np.sum(cV3 ** 2) + np.sum(cD3 ** 2),
                'detail_level2': np.sum(cH2 ** 2) + np.sum(cV2 ** 2) + np.sum(cD2 ** 2),
                'detail_level1': np.sum(cH1 ** 2) + np.sum(cV1 ** 2) + np.sum(cD1 ** 2)
            }

            # Normalize
            total_energy = sum(energy.values())
            energy_dist = {k: v / total_energy for k, v in energy.items()}

            wavelet_features.append(energy_dist)

        # Average across frames
        avg_energy = {}
        for key in wavelet_features[0].keys():
            avg_energy[key] = np.mean([f[key] for f in wavelet_features])

        # Diffusion models tend to have more energy in mid-level details
        # due to U-Net skip connections
        mid_level_ratio = avg_energy['detail_level2'] / (avg_energy['detail_level1'] + 1e-8)

        return {
            'energy_distribution': avg_energy,
            'mid_level_ratio': float(mid_level_ratio),
            'diffusion_indicator': mid_level_ratio > 1.2  # Heuristic threshold
        }

    def detect_color_distribution_anomalies(self, frames):
        """
        Analyze color distribution for diffusion artifacts

        Diffusion models can produce slightly different color
        distributions than natural videos

        Args:
            frames: numpy array (num_frames, H, W, C)

        Returns:
            dict with color analysis
        """
        color_stats = []

        for frame in frames:
            if len(frame.shape) != 3:
                continue

            # Analyze each channel
            for c in range(3):
                channel = frame[:, :, c].flatten()

                hist, _ = np.histogram(channel, bins=256, range=(0, 256))
                hist = hist / hist.sum()  # Normalize

                # Compute entropy (higher = more uniform distribution)
                entropy = -np.sum(hist * np.log(hist + 1e-10))

                # Compute moments
                mean = np.mean(channel)
                std = np.std(channel)

                color_stats.append({
                    'channel': c,
                    'entropy': entropy,
                    'mean': mean,
                    'std': std
                })

        # Aggregate by channel
        channel_stats = {c: [] for c in range(3)}
        for stat in color_stats:
            channel_stats[stat['channel']].append(stat)

        avg_channel_stats = {}
        for c in range(3):
            if channel_stats[c]:
                avg_channel_stats[c] = {
                    'mean_entropy': np.mean([s['entropy'] for s in channel_stats[c]]),
                    'mean_value': np.mean([s['mean'] for s in channel_stats[c]]),
                    'mean_std': np.mean([s['std'] for s in channel_stats[c]])
                }

        return {
            'channel_statistics': avg_channel_stats,
            'color_uniformity': np.mean([avg_channel_stats[c]['mean_entropy'] for c in range(3)])
        }

    def comprehensive_analysis(self, video_path):
        """
        Run all fingerprint detection methods

        Args:
            video_path: path to video file

        Returns:
            dict with comprehensive analysis
        """
        print(f"\nAnalyzing: {Path(video_path).name}")

        # Load video
        _, frames, label, _ = load_video(video_path)

        print("  Running checkerboard detection...")
        checkerboard = self.detect_checkerboard_artifacts(frames)

        print("  Analyzing noise residuals...")
        noise = self.analyze_noise_residuals(frames)

        print("  Detecting temporal inconsistencies...")
        temporal = self.detect_temporal_inconsistencies(frames)

        print("  Performing wavelet analysis...")
        wavelet = self.wavelet_analysis(frames)

        print("  Analyzing color distribution...")
        color = self.detect_color_distribution_anomalies(frames)

        # Compute overall diffusion likelihood
        diffusion_score = self._compute_overall_score(
            checkerboard, noise, temporal, wavelet
        )

        return {
            'video_path': str(video_path),
            'true_label': ['AI-Generated', 'Real'][label.item()],
            'fingerprints': {
                'checkerboard': checkerboard,
                'noise_residuals': noise,
                'temporal': temporal,
                'wavelet': wavelet,
                'color': color
            },
            'diffusion_score': diffusion_score,
            'interpretation': self._interpret_results(
                checkerboard, noise, temporal, wavelet, diffusion_score
            )
        }

    def _compute_overall_score(self, checkerboard, noise, temporal, wavelet):
        """
        Compute overall diffusion likelihood from all fingerprints

        Args:
            checkerboard, noise, temporal, wavelet: analysis results

        Returns:
            float: overall diffusion likelihood score (0-1)
        """
        score = 0.0
        weights = []

        # Checkerboard artifacts (strong indicator)
        if checkerboard['detection']:
            score += 0.3
            weights.append(0.3)

        # Gaussian noise (moderate indicator)
        if noise['is_gaussian_noise']:
            score += 0.25 * noise['diffusion_likelihood']
            weights.append(0.25)

        # Temporal chunk artifacts (strong indicator for video diffusion)
        if temporal['has_chunk_artifacts']:
            score += 0.3
            weights.append(0.3)

        # Wavelet mid-level energy (weak indicator)
        if wavelet['diffusion_indicator']:
            score += 0.15
            weights.append(0.15)

        # Normalize
        if weights:
            score = score / sum(weights) if sum(weights) > 0 else 0

        return float(min(score, 1.0))

    def _interpret_results(self, checkerboard, noise, temporal, wavelet, overall_score):
        """Generate human-readable interpretation"""
        findings = []

        if checkerboard['detection']:
            findings.append(
                f"✓ Checkerboard artifacts detected (score: {checkerboard['mean_score']:.2f}). "
                f"Likely from U-Net upsampling in diffusion models."
            )

        if noise['is_gaussian_noise']:
            findings.append(
                f"✓ Gaussian noise pattern detected (likelihood: {noise['diffusion_likelihood']:.2f}). "
                f"Consistent with iterative denoising process."
            )

        if temporal['has_chunk_artifacts']:
            findings.append(
                f"✓ Temporal chunk artifacts found at {len(temporal['discontinuity_frames'])} locations. "
                f"Characteristic of video diffusion models."
            )

        if wavelet['diffusion_indicator']:
            findings.append(
                f"✓ Wavelet analysis shows mid-level energy concentration (ratio: {wavelet['mid_level_ratio']:.2f}). "
                f"Suggests U-Net skip connection artifacts."
            )

        if overall_score > 0.7:
            findings.append(
                f"\n🔴 HIGH confidence ({overall_score:.2f}) that this is diffusion-generated content."
            )
        elif overall_score > 0.4:
            findings.append(
                f"\n🟡 MODERATE confidence ({overall_score:.2f}) of diffusion generation."
            )
        else:
            findings.append(
                f"\n🟢 LOW confidence ({overall_score:.2f}) of diffusion generation. "
                f"Likely real or non-diffusion AI method."
            )

        return findings

    def visualize_fingerprints(self, video_path, save_dir='diffusion_analysis'):
        """
        Create visualization of all detected fingerprints

        Args:
            video_path: path to video file
            save_dir: directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)

        # Run analysis
        results = self.comprehensive_analysis(video_path)

        # Load frames
        _, frames, _, _ = load_video(video_path)

        # Create figure
        fig = plt.figure(figsize=(18, 12))

        # 1. Sample frame
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(frames[len(frames)//2])
        ax1.set_title(f'Sample Frame\nLabel: {results["true_label"]}')
        ax1.axis('off')

        # 2. Checkerboard detection
        ax2 = plt.subplot(3, 3, 2)
        checkerboard = results['fingerprints']['checkerboard']
        ax2.plot(checkerboard['scores'], marker='o')
        ax2.axhline(y=1.5, color='r', linestyle='--', label='Threshold')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Checkerboard Score')
        ax2.set_title(f'Checkerboard Artifacts\nDetected: {checkerboard["detection"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Noise analysis
        ax3 = plt.subplot(3, 3, 3)
        noise = results['fingerprints']['noise_residuals']
        frame_mid = len(frames) // 2
        if len(frames[frame_mid].shape) == 3:
            gray = cv2.cvtColor(frames[frame_mid], cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = frames[frame_mid].astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 2)
        noise_img = gray - blurred
        ax3.imshow(noise_img, cmap='gray')
        ax3.set_title(f'High-Freq Noise Residual\nGaussian: {noise["is_gaussian_noise"]}')
        ax3.axis('off')

        # 4. Temporal consistency
        ax4 = plt.subplot(3, 3, 4)
        temporal = results['fingerprints']['temporal']
        if temporal['temporal_diffs']:
            ax4.plot(temporal['temporal_diffs'], marker='o')
            for disc in temporal['discontinuity_frames']:
                ax4.axvline(x=disc, color='r', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Frame Difference')
            ax4.set_title(f'Temporal Consistency\nChunk Artifacts: {temporal["has_chunk_artifacts"]}')
            ax4.grid(True, alpha=0.3)

        # 5. Wavelet energy distribution
        ax5 = plt.subplot(3, 3, 5)
        wavelet = results['fingerprints']['wavelet']
        energy_dist = wavelet['energy_distribution']
        labels = list(energy_dist.keys())
        values = list(energy_dist.values())
        ax5.bar(range(len(labels)), values, color=['blue', 'orange', 'green', 'red'])
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(['Approx', 'L3', 'L2', 'L1'], rotation=45)
        ax5.set_ylabel('Energy Proportion')
        ax5.set_title(f'Wavelet Energy Distribution\nDiffusion Indicator: {wavelet["diffusion_indicator"]}')
        ax5.grid(axis='y', alpha=0.3)

        # 6. Color distribution
        ax6 = plt.subplot(3, 3, 6)
        color = results['fingerprints']['color']
        if color['channel_statistics']:
            entropies = [color['channel_statistics'][c]['mean_entropy'] for c in range(3)]
            ax6.bar(['R', 'G', 'B'], entropies, color=['red', 'green', 'blue'], alpha=0.7)
            ax6.set_ylabel('Mean Entropy')
            ax6.set_title('Color Channel Entropy')
            ax6.grid(axis='y', alpha=0.3)

        # 7. Fingerprint summary
        ax7 = plt.subplot(3, 3, 7)
        indicators = {
            'Checkerboard': 1.0 if checkerboard['detection'] else 0.0,
            'Gaussian\nNoise': noise['diffusion_likelihood'],
            'Temporal\nChunks': 1.0 if temporal['has_chunk_artifacts'] else 0.0,
            'Wavelet\nArtifacts': 1.0 if wavelet['diffusion_indicator'] else 0.0
        }
        colors = ['red' if v > 0.5 else 'green' for v in indicators.values()]
        ax7.barh(list(indicators.keys()), list(indicators.values()), color=colors, alpha=0.7)
        ax7.set_xlabel('Detection Score')
        ax7.set_title('Fingerprint Indicators')
        ax7.set_xlim([0, 1])
        ax7.grid(axis='x', alpha=0.3)

        # 8. Overall score
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        score = results['diffusion_score']
        color_bg = 'lightcoral' if score > 0.7 else 'lightyellow' if score > 0.4 else 'lightgreen'

        summary_text = (
            f"DIFFUSION FINGERPRINT ANALYSIS\n\n"
            f"Overall Score: {score:.2f}\n"
            f"True Label: {results['true_label']}\n\n"
            f"Findings:\n"
        )
        for finding in results['interpretation'][:3]:
            summary_text += f"\n{finding[:80]}..."

        ax8.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color_bg, alpha=0.8))

        # 9. FFT checkerboard pattern
        ax9 = plt.subplot(3, 3, 9)
        frame_mid = frames[len(frames)//2]
        if len(frame_mid.shape) == 3:
            gray = cv2.cvtColor(frame_mid, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_mid
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        ax9.imshow(np.log(magnitude + 1), cmap='hot')
        ax9.set_title('FFT Magnitude (Log)\nLook for corner peaks')
        ax9.axis('off')

        plt.suptitle(f'Diffusion Fingerprint Analysis: {Path(video_path).name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_path = os.path.join(save_dir, f'diffusion_fingerprints_{Path(video_path).stem}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")

        plt.close()

        return results


if __name__ == "__main__":
    # Example usage
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'

    detector = DiffusionFingerprintDetector()

    # Test on AI-generated video
    ai_video = os.path.join(base_dir, "dataset", "ai_dynamiccrafter_DynamicCrafter_43162.mp4")

    if os.path.exists(ai_video):
        print("=" * 60)
        print("Diffusion Fingerprint Detection")
        print("=" * 60)

        results = detector.visualize_fingerprints(ai_video)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Overall Diffusion Score: {results['diffusion_score']:.2f}")
        print("\nFindings:")
        for finding in results['interpretation']:
            print(f"  {finding}")
    else:
        print(f"Video not found: {ai_video}")
