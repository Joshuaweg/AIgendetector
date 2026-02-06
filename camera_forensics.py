"""
Camera Forensics Module for AI-Generated Video Detection

This module implements forensic metrics based on physical camera characteristics
that distinguish real camera footage from AI-generated video.

Categories:
1. Optical Detail - Lens & sensor resolution characteristics
2. Sensor Noise - Physical sensor fingerprints
3. Sampling Structure - Bayer filter & ADC artifacts
4. Surface/Material - Texture authenticity
5. Lens Optics - Chromatic aberration, vignetting, distortion
6. Temporal Consistency - Frame-to-frame physics
7. Compression Forensics - Codec artifacts
8. Color/Radiometry - Color response characteristics
9. Statistical Measures - Information-theoretic metrics

Usage:
    from camera_forensics import CameraForensics

    forensics = CameraForensics()
    results = forensics.analyze_video(video_path)
    # or
    results = forensics.analyze_frame(frame)
"""

import numpy as np
import cv2
from scipy import ndimage, signal, fft, stats
from scipy.ndimage import sobel, laplace, gaussian_filter
from scipy.fftpack import dct, fft2, fftshift
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class CameraForensics:
    """
    Comprehensive camera forensics analyzer for AI video detection.
    """

    def __init__(self, config=None):
        """
        Initialize the forensics analyzer.

        Args:
            config: Optional dict to enable/disable specific metric categories
        """
        self.config = config or {
            'optical_detail': True,
            'sensor_noise': True,
            'sampling_structure': True,
            'surface_material': True,
            'lens_optics': True,
            'temporal': True,
            'compression': True,
            'color_radiometry': True,
            'statistical': True,
        }

        # Cache for temporal analysis
        self._frame_cache = []
        self._noise_residual_cache = []
        self._max_cache_size = 30

    def clear_cache(self):
        """Clear temporal analysis cache."""
        self._frame_cache = []
        self._noise_residual_cache = []

    # =========================================================================
    # MAIN ANALYSIS FUNCTIONS
    # =========================================================================

    def analyze_frame(self, frame, include_all=True):
        """
        Analyze a single frame for forensic metrics.

        Args:
            frame: numpy array (H, W, 3) in RGB format, values 0-255 or 0-1
            include_all: If True, compute all metrics. If False, use config.

        Returns:
            dict with all computed metrics organized by category
        """
        # Normalize frame
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        frame = frame.astype(np.float32)

        results = {}

        # Optical Detail
        if include_all or self.config.get('optical_detail'):
            results['optical_detail'] = self._compute_optical_detail(frame)

        # Sensor Noise
        if include_all or self.config.get('sensor_noise'):
            results['sensor_noise'] = self._compute_sensor_noise(frame)

        # Sampling Structure
        if include_all or self.config.get('sampling_structure'):
            results['sampling_structure'] = self._compute_sampling_structure(frame)

        # Surface/Material
        if include_all or self.config.get('surface_material'):
            results['surface_material'] = self._compute_surface_material(frame)

        # Lens Optics
        if include_all or self.config.get('lens_optics'):
            results['lens_optics'] = self._compute_lens_optics(frame)

        # Compression
        if include_all or self.config.get('compression'):
            results['compression'] = self._compute_compression_artifacts(frame)

        # Color/Radiometry
        if include_all or self.config.get('color_radiometry'):
            results['color_radiometry'] = self._compute_color_radiometry(frame)

        # Statistical
        if include_all or self.config.get('statistical'):
            results['statistical'] = self._compute_statistical_measures(frame)

        # Update cache for temporal analysis
        self._update_cache(frame)

        return results

    def analyze_video(self, video_path, max_frames=24, sample_rate=1):
        """
        Analyze a video file for forensic metrics.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze
            sample_rate: Sample every Nth frame

        Returns:
            dict with aggregated metrics and temporal analysis
        """
        self.clear_cache()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly
        if total_frames <= max_frames:
            frame_indices = list(range(0, total_frames, sample_rate))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        all_frame_results = []
        frames_analyzed = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_results = self.analyze_frame(frame_rgb)
            all_frame_results.append(frame_results)
            frames_analyzed += 1

        cap.release()

        # Aggregate results across frames
        aggregated = self._aggregate_frame_results(all_frame_results)

        # Add temporal analysis
        if self.config.get('temporal') and len(self._noise_residual_cache) > 1:
            aggregated['temporal'] = self._compute_temporal_consistency()

        aggregated['_meta'] = {
            'frames_analyzed': frames_analyzed,
            'video_path': video_path,
        }

        return aggregated

    def get_feature_vector(self, results):
        """
        Flatten results dict into a feature vector for classification.

        Args:
            results: Output from analyze_frame or analyze_video

        Returns:
            numpy array of features, list of feature names
        """
        features = []
        names = []

        for category, metrics in results.items():
            if category.startswith('_'):
                continue
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        features.append(float(value))
                        names.append(f"{category}.{metric_name}")

        return np.array(features), names

    # =========================================================================
    # OPTICAL DETAIL METRICS
    # =========================================================================

    def _compute_optical_detail(self, frame):
        """Compute optical detail metrics."""
        gray = self._to_grayscale(frame)

        return {
            'microcontrast_strength': self._microcontrast_strength(gray),
            'edge_density': self._edge_density(gray),
            'gradient_entropy': self._gradient_entropy(gray),
            'high_frequency_energy': self._high_frequency_energy(gray),
            'local_contrast_variance': self._local_contrast_variance(gray),
            'sharpness_score': self._sharpness_score(gray),
            'detail_preservation': self._detail_preservation(gray),
        }

    def _microcontrast_strength(self, gray):
        """
        Measure fine detail contrast.
        Real lenses produce characteristic microcontrast patterns.
        """
        # High-pass filter to isolate fine details
        blur = gaussian_filter(gray, sigma=2)
        high_pass = gray - blur

        # Measure local contrast in high-frequency component
        local_std = ndimage.generic_filter(high_pass, np.std, size=5)

        # Weight by edge presence (microcontrast matters more at edges)
        edges = np.abs(sobel(gray))
        weighted = local_std * (edges / (edges.max() + 1e-8))

        return float(np.mean(weighted))

    def _edge_density(self, gray):
        """
        Compute edge density in the image.
        Real scenes have natural edge distributions.
        """
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        return float(np.mean(edges > 0))

    def _gradient_entropy(self, gray):
        """
        Compute entropy of gradient directions.
        Real images have stochastic gradients; AI may be more structured.
        """
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)

        # Gradient angles
        angles = np.arctan2(gy, gx)

        # Histogram of angles
        hist, _ = np.histogram(angles.flatten(), bins=36, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-8)

        # Entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return float(entropy)

    def _high_frequency_energy(self, gray):
        """
        Measure energy in high-frequency bands.
        Real cameras have characteristic HF rolloff from lens blur.
        """
        # 2D FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Create radial frequency mask
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = min(cx, cy)

        # High frequency = outer 30% of frequency space
        hf_mask = r > 0.7 * r_max

        total_energy = np.sum(magnitude**2)
        hf_energy = np.sum((magnitude * hf_mask)**2)

        return float(hf_energy / (total_energy + 1e-8))

    def _local_contrast_variance(self, gray):
        """Variance of local contrast across the image."""
        local_mean = ndimage.uniform_filter(gray, size=11)
        local_sqr_mean = ndimage.uniform_filter(gray**2, size=11)
        local_var = local_sqr_mean - local_mean**2
        return float(np.var(local_var))

    def _sharpness_score(self, gray):
        """Laplacian-based sharpness measure."""
        lap = laplace(gray)
        return float(np.var(lap))

    def _detail_preservation(self, gray):
        """Ratio of detail preserved across scales."""
        blur1 = gaussian_filter(gray, sigma=1)
        blur2 = gaussian_filter(gray, sigma=2)

        detail1 = np.std(gray - blur1)
        detail2 = np.std(blur1 - blur2)

        return float(detail1 / (detail2 + 1e-8))

    # =========================================================================
    # SENSOR NOISE METRICS
    # =========================================================================

    def _compute_sensor_noise(self, frame):
        """Compute sensor noise metrics."""
        gray = self._to_grayscale(frame)
        noise_residual = self._extract_noise_residual(gray)

        return {
            'noise_residual_strength': self._noise_residual_strength(noise_residual),
            'noise_spatial_variability': self._noise_spatial_variability(noise_residual),
            'rgb_noise_independence': self._rgb_noise_independence(frame),
            'noise_frequency_profile': self._noise_frequency_profile(noise_residual),
            'shot_noise_conformance': self._shot_noise_conformance(frame),
            'fixed_pattern_noise_score': self._fixed_pattern_noise_score(noise_residual),
            'read_noise_estimate': self._read_noise_estimate(gray),
        }

    def _extract_noise_residual(self, gray):
        """Extract noise residual using denoising."""
        denoised = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, 10, 7, 21)
        residual = gray - denoised.astype(np.float32)
        return residual

    def _noise_residual_strength(self, residual):
        """Strength of noise residual."""
        return float(np.std(residual))

    def _noise_spatial_variability(self, residual):
        """
        How noise varies spatially.
        Real sensors have fixed-pattern noise; AI noise is often uniform.
        """
        # Divide into blocks and measure variance of noise level
        h, w = residual.shape
        block_size = 32
        block_stds = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = residual[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))

        return float(np.std(block_stds))

    def _rgb_noise_independence(self, frame):
        """
        Test if RGB channel noise is independent.
        Real Bayer sensors have independent channel noise.
        """
        r_noise = self._extract_noise_residual(frame[:, :, 0])
        g_noise = self._extract_noise_residual(frame[:, :, 1])
        b_noise = self._extract_noise_residual(frame[:, :, 2])

        # Flatten and compute correlations
        r_flat = r_noise.flatten()
        g_flat = g_noise.flatten()
        b_flat = b_noise.flatten()

        # Correlation coefficients
        rg_corr = np.abs(np.corrcoef(r_flat, g_flat)[0, 1])
        rb_corr = np.abs(np.corrcoef(r_flat, b_flat)[0, 1])
        gb_corr = np.abs(np.corrcoef(g_flat, b_flat)[0, 1])

        # Lower correlation = more independent = more likely real
        mean_corr = (rg_corr + rb_corr + gb_corr) / 3

        # Return independence score (1 - correlation)
        return float(1 - mean_corr)

    def _noise_frequency_profile(self, residual):
        """
        Analyze frequency distribution of noise.
        Real sensor noise has characteristic spectrum.
        """
        f_transform = fft2(residual)
        magnitude = np.abs(fftshift(f_transform))

        # Radial average
        h, w = residual.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

        # Average magnitude at each radius
        radial_profile = ndimage.mean(magnitude, r, index=np.arange(0, r.max()))

        # Measure flatness (white noise = flat spectrum)
        if len(radial_profile) > 10:
            slope = np.polyfit(np.arange(len(radial_profile)), np.log(radial_profile + 1e-8), 1)[0]
            return float(-slope)  # Negative slope = more white-noise-like
        return 0.0

    def _shot_noise_conformance(self, frame):
        """
        Test if noise follows Poisson statistics (shot noise).
        Real sensors have √intensity noise scaling.
        """
        gray = self._to_grayscale(frame)
        noise = self._extract_noise_residual(gray)

        # Bin pixels by intensity and measure noise variance
        bins = np.linspace(0, 255, 11)
        bin_indices = np.digitize(gray.flatten(), bins)

        intensities = []
        variances = []

        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 100:
                intensities.append(bins[i-1] + (bins[i] - bins[i-1])/2)
                variances.append(np.var(noise.flatten()[mask]))

        if len(intensities) < 3:
            return 0.5

        # Fit variance ~ intensity (Poisson: var = mean)
        intensities = np.array(intensities)
        variances = np.array(variances)

        # Linear fit
        slope, intercept = np.polyfit(intensities, variances, 1)

        # R² of fit
        predicted = slope * intensities + intercept
        ss_res = np.sum((variances - predicted)**2)
        ss_tot = np.sum((variances - np.mean(variances))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))

        return float(max(0, r_squared))

    def _fixed_pattern_noise_score(self, residual):
        """
        Detect fixed-pattern noise (FPN).
        Real sensors have column/row noise patterns.
        """
        # Check for row patterns
        row_means = np.mean(residual, axis=1)
        row_pattern = np.std(row_means)

        # Check for column patterns
        col_means = np.mean(residual, axis=0)
        col_pattern = np.std(col_means)

        return float(row_pattern + col_pattern)

    def _read_noise_estimate(self, gray):
        """Estimate read noise from dark regions."""
        dark_mask = gray < 30
        if dark_mask.sum() < 100:
            return 0.0
        dark_region = gray[dark_mask]
        return float(np.std(dark_region))

    # =========================================================================
    # SAMPLING STRUCTURE METRICS
    # =========================================================================

    def _compute_sampling_structure(self, frame):
        """Compute sampling structure metrics related to Bayer patterns."""
        gray = self._to_grayscale(frame)

        return {
            'axis_frequency_bias': self._axis_frequency_bias(gray),
            'moire_energy': self._moire_energy(gray),
            'directional_anisotropy': self._directional_anisotropy(gray),
            'recapture_grid_score': self._recapture_grid_score(frame),
            'recapture_peak_ratio': self._recapture_peak_ratio(frame),
            'rgb_subpixel_score': self._rgb_subpixel_score(frame),
            'bayer_pattern_residual': self._bayer_pattern_residual(frame),
            'demosaic_artifact_score': self._demosaic_artifact_score(frame),
        }

    def _axis_frequency_bias(self, gray):
        """
        Measure bias toward horizontal/vertical frequencies.
        Bayer demosaicing creates characteristic H/V artifacts.
        """
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Horizontal frequencies (vertical line through center)
        h_energy = np.sum(magnitude[cy-5:cy+5, :])

        # Vertical frequencies (horizontal line through center)
        v_energy = np.sum(magnitude[:, cx-5:cx+5])

        # Bias ratio
        total = h_energy + v_energy
        return float(abs(h_energy - v_energy) / (total + 1e-8))

    def _moire_energy(self, gray):
        """
        Detect moiré patterns from sensor sampling.
        """
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Look for peaks in mid-frequency range
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Mid-frequency band
        mid_band = (r > 0.2 * min(cx, cy)) & (r < 0.6 * min(cx, cy))

        mid_magnitude = magnitude * mid_band

        # Peak to mean ratio in mid-band
        peak = np.max(mid_magnitude)
        mean = np.mean(mid_magnitude[mid_band])

        return float(peak / (mean + 1e-8))

    def _directional_anisotropy(self, gray):
        """
        Measure directional bias in frequency domain.
        """
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Compute energy in different angular sectors
        y, x = np.ogrid[:h, :w]
        angles = np.arctan2(y - cy, x - cx)

        # 8 angular sectors
        sector_energies = []
        for i in range(8):
            angle_min = -np.pi + i * np.pi / 4
            angle_max = -np.pi + (i + 1) * np.pi / 4
            mask = (angles >= angle_min) & (angles < angle_max)
            sector_energies.append(np.sum(magnitude[mask]))

        sector_energies = np.array(sector_energies)
        # Anisotropy = variance in sector energies
        return float(np.std(sector_energies) / (np.mean(sector_energies) + 1e-8))

    def _recapture_grid_score(self, frame):
        """
        Detect if video was captured from a screen.
        Screen recapture shows LCD/OLED subpixel patterns.
        """
        # Check for periodic patterns at display frequencies
        gray = self._to_grayscale(frame)
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Look for peaks at typical screen pixel densities
        # Common patterns: 3-pixel RGB stripe period
        scores = []
        for period in [3, 4, 6]:  # Common subpixel periods
            if w > period * 10:
                freq_x = cx + w // period
                freq_x2 = cx - w // period
                local_peak = max(magnitude[cy, freq_x], magnitude[cy, freq_x2])
                local_mean = np.mean(magnitude[cy, :])
                scores.append(local_peak / (local_mean + 1e-8))

        return float(max(scores) if scores else 0)

    def _recapture_peak_ratio(self, frame):
        """
        Ratio of periodic peaks to background in FFT.
        """
        gray = self._to_grayscale(frame)
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        # Find peaks
        threshold = np.mean(magnitude) + 3 * np.std(magnitude)
        peaks = magnitude > threshold

        peak_energy = np.sum(magnitude[peaks])
        total_energy = np.sum(magnitude)

        return float(peak_energy / (total_energy + 1e-8))

    def _rgb_subpixel_score(self, frame):
        """
        Analyze RGB channel alignment patterns.
        Real Bayer sensors have specific R/G/B spatial offsets.
        """
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        # Cross-correlation between channels at small offsets
        def channel_correlation(c1, c2, offset):
            if offset > 0:
                return np.corrcoef(c1[:, offset:].flatten(), c2[:, :-offset].flatten())[0, 1]
            elif offset < 0:
                return np.corrcoef(c1[:, :offset].flatten(), c2[:, -offset:].flatten())[0, 1]
            return np.corrcoef(c1.flatten(), c2.flatten())[0, 1]

        # Check correlations at different offsets
        offsets = [-2, -1, 0, 1, 2]
        rg_corrs = [channel_correlation(r, g, o) for o in offsets]

        # Real cameras often have peak correlation at non-zero offset due to Bayer
        peak_offset = offsets[np.argmax(rg_corrs)]

        return float(abs(peak_offset))

    def _bayer_pattern_residual(self, frame):
        """
        Detect residual Bayer pattern from incomplete demosaicing.
        """
        # Check for 2x2 periodic patterns in green channel
        g = frame[:, :, 1]

        # Extract checkerboard components
        even_even = g[0::2, 0::2]
        even_odd = g[0::2, 1::2]
        odd_even = g[1::2, 0::2]
        odd_odd = g[1::2, 1::2]

        # In raw Bayer, G appears in checkerboard pattern
        # Variance difference indicates residual pattern
        min_shape = (min(even_even.shape[0], odd_odd.shape[0]),
                     min(even_even.shape[1], odd_odd.shape[1]))

        ee = even_even[:min_shape[0], :min_shape[1]]
        oo = odd_odd[:min_shape[0], :min_shape[1]]
        eo = even_odd[:min_shape[0], :min_shape[1]]
        oe = odd_even[:min_shape[0], :min_shape[1]]

        # Checkerboard residual
        checker1 = np.mean(np.abs(ee - oo))
        checker2 = np.mean(np.abs(eo - oe))

        return float(checker1 + checker2)

    def _demosaic_artifact_score(self, frame):
        """
        Detect demosaicing artifacts (zipper, maze patterns).
        """
        # High-frequency cross-channel correlation
        r_hf = laplace(frame[:, :, 0].astype(np.float32))
        g_hf = laplace(frame[:, :, 1].astype(np.float32))
        b_hf = laplace(frame[:, :, 2].astype(np.float32))

        # Demosaic artifacts show up as correlated HF patterns
        rg_corr = np.corrcoef(r_hf.flatten(), g_hf.flatten())[0, 1]
        gb_corr = np.corrcoef(g_hf.flatten(), b_hf.flatten())[0, 1]

        return float((rg_corr + gb_corr) / 2)

    # =========================================================================
    # SURFACE/MATERIAL METRICS
    # =========================================================================

    def _compute_surface_material(self, frame):
        """Compute surface and material texture metrics."""
        return {
            'chroma_microstructure_energy': self._chroma_microstructure_energy(frame),
            'texture_regularity': self._texture_regularity(frame),
            'surface_normal_consistency': self._surface_normal_consistency(frame),
            'material_spectral_diversity': self._material_spectral_diversity(frame),
            'micro_texture_variance': self._micro_texture_variance(frame),
        }

    def _chroma_microstructure_energy(self, frame):
        """
        Fine color texture patterns in surfaces.
        Real materials have characteristic chroma microstructure.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        # Extract chroma channels
        a, b = lab[:, :, 1], lab[:, :, 2]

        # High-pass filter chroma
        a_hp = a - gaussian_filter(a, sigma=3)
        b_hp = b - gaussian_filter(b, sigma=3)

        # Energy in chroma microstructure
        energy = np.mean(a_hp**2 + b_hp**2)

        return float(energy)

    def _texture_regularity(self, frame):
        """
        Measure how regular/repetitive textures are.
        AI often produces too-regular textures.
        """
        gray = self._to_grayscale(frame)

        # Autocorrelation
        f_transform = fft2(gray)
        power_spectrum = np.abs(f_transform)**2
        autocorr = np.abs(fftshift(np.fft.ifft2(power_spectrum)))

        # Normalize
        autocorr = autocorr / autocorr.max()

        h, w = autocorr.shape
        cy, cx = h // 2, w // 2

        # Look for secondary peaks (indicates repetition)
        # Exclude center region
        autocorr[cy-10:cy+10, cx-10:cx+10] = 0

        secondary_peak = np.max(autocorr)

        return float(secondary_peak)

    def _surface_normal_consistency(self, frame):
        """
        Check if surface normals (from shading) are consistent.
        """
        gray = self._to_grayscale(frame)

        # Estimate normals from gradient
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)

        # Local smoothness of normal field
        gx_smooth = gaussian_filter(gx, sigma=5)
        gy_smooth = gaussian_filter(gy, sigma=5)

        normal_consistency = np.mean(np.abs(gx - gx_smooth) + np.abs(gy - gy_smooth))

        return float(normal_consistency)

    def _material_spectral_diversity(self, frame):
        """
        Diversity of spectral (color) signatures in the image.
        """
        # Reshape to pixel list
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Sample for efficiency
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # Compute covariance
        cov = np.cov(pixels.T)

        # Eigenvalues represent spectral diversity
        eigenvalues = np.linalg.eigvalsh(cov)

        return float(np.sum(eigenvalues))

    def _micro_texture_variance(self, frame):
        """
        Variance in micro-texture across the image.
        """
        gray = self._to_grayscale(frame)

        # Local texture energy using Laws' texture energy
        # Simple approximation: local variance
        local_var = ndimage.generic_filter(gray, np.var, size=7)

        return float(np.var(local_var))

    # =========================================================================
    # LENS OPTICS METRICS
    # =========================================================================

    def _compute_lens_optics(self, frame):
        """Compute lens-related optical metrics."""
        return {
            'chromatic_aberration': self._chromatic_aberration(frame),
            'lateral_ca_profile': self._lateral_ca_profile(frame),
            'vignetting_score': self._vignetting_score(frame),
            'radial_distortion': self._radial_distortion_score(frame),
            'lens_blur_profile': self._lens_blur_profile(frame),
        }

    def _chromatic_aberration(self, frame):
        """
        Measure chromatic aberration (color fringing at edges).
        Real lenses separate RGB wavelengths differently.
        """
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        # Edge detection on each channel
        r_edges = np.abs(sobel(r))
        g_edges = np.abs(sobel(g))
        b_edges = np.abs(sobel(b))

        # Misalignment at edges
        # Compare edge positions between channels
        edge_threshold = 30
        edge_mask = g_edges > edge_threshold

        if edge_mask.sum() < 100:
            return 0.0

        # Measure RGB edge offset
        rg_diff = np.mean(np.abs(r_edges[edge_mask] - g_edges[edge_mask]))
        gb_diff = np.mean(np.abs(g_edges[edge_mask] - b_edges[edge_mask]))

        return float((rg_diff + gb_diff) / 2)

    def _lateral_ca_profile(self, frame):
        """
        How chromatic aberration varies from center to edge.
        Real lenses have increasing CA toward edges.
        """
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2

        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        # Distance from center
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)

        # Measure CA in radial bins
        n_bins = 5
        ca_by_radius = []

        for i in range(n_bins):
            r_min = i * max_dist / n_bins
            r_max = (i + 1) * max_dist / n_bins
            mask = (dist >= r_min) & (dist < r_max)

            if mask.sum() > 100:
                rg_diff = np.mean(np.abs(r[mask].astype(float) - g[mask].astype(float)))
                ca_by_radius.append(rg_diff)

        if len(ca_by_radius) < 2:
            return 0.0

        # Slope of CA vs radius (should increase for real lenses)
        slope = np.polyfit(range(len(ca_by_radius)), ca_by_radius, 1)[0]

        return float(slope)

    def _vignetting_score(self, frame):
        """
        Detect vignetting (corner darkening).
        Real lenses have smooth radial falloff.
        """
        gray = self._to_grayscale(frame)
        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Center region brightness
        center_region = gray[cy-h//8:cy+h//8, cx-w//8:cx+w//8]
        center_brightness = np.mean(center_region)

        # Corner regions
        corner_size = min(h, w) // 8
        corners = [
            gray[:corner_size, :corner_size],  # Top-left
            gray[:corner_size, -corner_size:],  # Top-right
            gray[-corner_size:, :corner_size],  # Bottom-left
            gray[-corner_size:, -corner_size:],  # Bottom-right
        ]
        corner_brightness = np.mean([np.mean(c) for c in corners])

        # Vignetting = center-to-corner ratio
        vignetting = (center_brightness - corner_brightness) / (center_brightness + 1e-8)

        return float(vignetting)

    def _radial_distortion_score(self, frame):
        """
        Estimate lens distortion from straight lines.
        """
        gray = self._to_grayscale(frame)

        # Detect lines using Hough transform
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) < 5:
            return 0.0

        # Measure deviation from straightness
        # For a proper implementation, would fit lines and measure curvature
        # Simplified: count lines vs total edge pixels
        line_pixels = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_pixels += np.sqrt((x2-x1)**2 + (y2-y1)**2)

        total_edge_pixels = np.sum(edges > 0)

        return float(line_pixels / (total_edge_pixels + 1e-8))

    def _lens_blur_profile(self, frame):
        """
        Analyze blur characteristics across the frame.
        Real lenses have characteristic blur profiles.
        """
        gray = self._to_grayscale(frame)

        # Measure local sharpness in grid
        h, w = gray.shape
        grid_size = 4
        sharpness_map = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                region = gray[y1:y2, x1:x2]
                sharpness_map[i, j] = np.var(laplace(region))

        # Center vs edge sharpness ratio
        center = sharpness_map[1:3, 1:3].mean()
        edge = (sharpness_map[0, :].mean() + sharpness_map[-1, :].mean() +
                sharpness_map[:, 0].mean() + sharpness_map[:, -1].mean()) / 4

        return float(center / (edge + 1e-8))

    # =========================================================================
    # COMPRESSION FORENSICS
    # =========================================================================

    def _compute_compression_artifacts(self, frame):
        """Compute compression-related forensic metrics."""
        gray = self._to_grayscale(frame)

        return {
            'jpeg_block_artifact': self._jpeg_block_artifact(gray),
            'dct_coefficient_stats': self._dct_coefficient_stats(gray),
            'quantization_noise': self._quantization_noise(gray),
            'double_compression_score': self._double_compression_score(gray),
            'blocking_artifact_strength': self._blocking_artifact_strength(gray),
        }

    def _jpeg_block_artifact(self, gray):
        """
        Detect 8x8 block artifacts from JPEG/video compression.
        """
        h, w = gray.shape

        if h < 16 or w < 16:
            return 0.0

        # Look at pixel differences at 8-pixel boundaries
        # Ensure same shape by using min length
        h_bound_a = gray[:, 7:-8:8]
        h_bound_b = gray[:, 8::8]
        min_w = min(h_bound_a.shape[1], h_bound_b.shape[1])
        h_boundaries = h_bound_a[:, :min_w] - h_bound_b[:, :min_w]

        v_bound_a = gray[7:-8:8, :]
        v_bound_b = gray[8::8, :]
        min_h = min(v_bound_a.shape[0], v_bound_b.shape[0])
        v_boundaries = v_bound_a[:min_h, :] - v_bound_b[:min_h, :]

        # Non-boundary differences
        h_nb_a = gray[:, 3:-8:8]
        h_nb_b = gray[:, 4::8]
        min_w = min(h_nb_a.shape[1], h_nb_b.shape[1])
        h_non_boundary = h_nb_a[:, :min_w] - h_nb_b[:, :min_w]

        v_nb_a = gray[3:-8:8, :]
        v_nb_b = gray[4::8, :]
        min_h = min(v_nb_a.shape[0], v_nb_b.shape[0])
        v_non_boundary = v_nb_a[:min_h, :] - v_nb_b[:min_h, :]

        # Ratio of boundary to non-boundary discontinuity
        boundary_strength = np.mean(np.abs(h_boundaries)) + np.mean(np.abs(v_boundaries))
        non_boundary_strength = np.mean(np.abs(h_non_boundary)) + np.mean(np.abs(v_non_boundary))

        return float(boundary_strength / (non_boundary_strength + 1e-8))

    def _dct_coefficient_stats(self, gray):
        """
        Analyze DCT coefficient statistics.
        Compressed images have specific DCT patterns.
        """
        h, w = gray.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        gray_cropped = gray[:h8, :w8]

        # Compute DCT for 8x8 blocks
        dct_coeffs = []
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = gray_cropped[i:i+8, j:j+8]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten())

        dct_coeffs = np.array(dct_coeffs)

        # First-digit distribution (Benford's law)
        abs_coeffs = np.abs(dct_coeffs[dct_coeffs != 0])
        if len(abs_coeffs) < 100:
            return 0.5

        first_digits = (abs_coeffs / 10**np.floor(np.log10(abs_coeffs + 1e-10))).astype(int) % 10
        first_digits = first_digits[first_digits > 0]

        hist, _ = np.histogram(first_digits, bins=9, range=(1, 10))
        hist = hist / (hist.sum() + 1e-8)

        # Benford's law expected distribution
        benford = np.log10(1 + 1/np.arange(1, 10))

        # KL divergence from Benford
        kl_div = np.sum(hist * np.log((hist + 1e-8) / (benford + 1e-8)))

        return float(kl_div)

    def _quantization_noise(self, gray):
        """
        Estimate quantization noise level.
        """
        # Histogram analysis
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))

        # Look for periodic gaps (quantization)
        hist_diff = np.diff(hist.astype(float))

        return float(np.std(hist_diff))

    def _double_compression_score(self, gray):
        """
        Detect double compression artifacts.
        """
        # Double compression creates ghost artifacts in DCT domain
        h, w = gray.shape

        if h < 24 or w < 24:
            return 0.0

        h8 = (h // 8) * 8
        w8 = (w // 8) * 8

        # Shifted DCT analysis
        scores = []
        for shift in range(1, 8):
            gray_shifted = gray[shift:shift+h8, shift:shift+w8]
            if gray_shifted.shape[0] >= 16 and gray_shifted.shape[1] >= 16:
                h_s, w_s = gray_shifted.shape
                h_s = (h_s // 8) * 8
                w_s = (w_s // 8) * 8
                gray_shifted = gray_shifted[:h_s, :w_s]

                # Compute blocking at shifted positions - handle shape carefully
                if w_s > 16:
                    h_a = gray_shifted[:, 7:-8:8]
                    h_b = gray_shifted[:, 8::8]
                    min_w = min(h_a.shape[1], h_b.shape[1])
                    if min_w > 0:
                        h_bound = h_a[:, :min_w] - h_b[:, :min_w]
                        scores.append(np.mean(np.abs(h_bound)))

        if not scores:
            return 0.0

        # Double compression shows different blocking at different shifts
        return float(np.std(scores))

    def _blocking_artifact_strength(self, gray):
        """
        Overall strength of blocking artifacts.
        """
        # Gradient at block boundaries
        gx = np.diff(gray.astype(float), axis=1)
        gy = np.diff(gray.astype(float), axis=0)

        # Block boundary gradients (every 8th pixel)
        block_gx = np.abs(gx[:, 7::8]).mean() if gx.shape[1] > 8 else 0
        block_gy = np.abs(gy[7::8, :]).mean() if gy.shape[0] > 8 else 0

        # Non-boundary gradients
        non_block_gx = np.abs(gx[:, 3::8]).mean() if gx.shape[1] > 8 else 1
        non_block_gy = np.abs(gy[3::8, :]).mean() if gy.shape[0] > 8 else 1

        ratio = (block_gx + block_gy) / (non_block_gx + non_block_gy + 1e-8)

        return float(ratio)

    # =========================================================================
    # COLOR / RADIOMETRY
    # =========================================================================

    def _compute_color_radiometry(self, frame):
        """Compute color and radiometry metrics."""
        return {
            'color_response_linearity': self._color_response_linearity(frame),
            'white_balance_consistency': self._white_balance_consistency(frame),
            'saturation_distribution': self._saturation_distribution(frame),
            'highlight_rolloff': self._highlight_rolloff(frame),
            'shadow_noise_color': self._shadow_noise_color(frame),
            'color_channel_correlation': self._color_channel_correlation(frame),
        }

    def _color_response_linearity(self, frame):
        """
        Test linearity of color response.
        """
        r, g, b = frame[:, :, 0].flatten(), frame[:, :, 1].flatten(), frame[:, :, 2].flatten()

        # Sample for efficiency
        n = min(10000, len(r))
        indices = np.random.choice(len(r), n, replace=False)
        r, g, b = r[indices], g[indices], b[indices]

        # Check if channels maintain linear relationship
        rg_corr = np.corrcoef(r, g)[0, 1]
        gb_corr = np.corrcoef(g, b)[0, 1]
        rb_corr = np.corrcoef(r, b)[0, 1]

        return float((rg_corr + gb_corr + rb_corr) / 3)

    def _white_balance_consistency(self, frame):
        """
        Check white balance consistency across frame.
        """
        h, w = frame.shape[:2]

        # Divide into regions
        n_regions = 4
        ratios = []

        for i in range(n_regions):
            for j in range(n_regions):
                y1, y2 = i * h // n_regions, (i + 1) * h // n_regions
                x1, x2 = j * w // n_regions, (j + 1) * w // n_regions
                region = frame[y1:y2, x1:x2]

                r_mean = np.mean(region[:, :, 0])
                g_mean = np.mean(region[:, :, 1])
                b_mean = np.mean(region[:, :, 2])

                if g_mean > 10:  # Avoid dark regions
                    ratios.append((r_mean / g_mean, b_mean / g_mean))

        if len(ratios) < 2:
            return 1.0

        ratios = np.array(ratios)
        # Variance in ratios = inconsistency
        consistency = 1 / (1 + np.var(ratios[:, 0]) + np.var(ratios[:, 1]))

        return float(consistency)

    def _saturation_distribution(self, frame):
        """
        Analyze saturation distribution.
        """
        hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].flatten()

        # Entropy of saturation distribution
        hist, _ = np.histogram(saturation, bins=32, range=(0, 255))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log2(hist + 1e-8))

        return float(entropy)

    def _highlight_rolloff(self, frame):
        """
        Analyze how highlights roll off to white.
        """
        gray = self._to_grayscale(frame)

        # Look at near-white pixels
        bright_mask = gray > 230
        if bright_mask.sum() < 100:
            return 0.0

        bright_values = gray[bright_mask]

        # Distribution shape near clipping
        return float(np.std(bright_values))

    def _shadow_noise_color(self, frame):
        """
        Analyze color of noise in shadow regions.
        Real sensors often have color-biased shadow noise.
        """
        gray = self._to_grayscale(frame)
        dark_mask = gray < 30

        if dark_mask.sum() < 100:
            return 0.0

        r_shadow = frame[:, :, 0][dark_mask]
        g_shadow = frame[:, :, 1][dark_mask]
        b_shadow = frame[:, :, 2][dark_mask]

        # Color bias in shadows
        r_bias = np.mean(r_shadow) - np.mean(g_shadow)
        b_bias = np.mean(b_shadow) - np.mean(g_shadow)

        return float(np.sqrt(r_bias**2 + b_bias**2))

    def _color_channel_correlation(self, frame):
        """
        Correlation structure between color channels.
        """
        r = frame[:, :, 0].flatten().astype(float)
        g = frame[:, :, 1].flatten().astype(float)
        b = frame[:, :, 2].flatten().astype(float)

        # Partial correlations
        rg = np.corrcoef(r, g)[0, 1]
        gb = np.corrcoef(g, b)[0, 1]
        rb = np.corrcoef(r, b)[0, 1]

        return float(np.mean([rg, gb, rb]))

    # =========================================================================
    # STATISTICAL MEASURES
    # =========================================================================

    def _compute_statistical_measures(self, frame):
        """Compute statistical/information-theoretic metrics."""
        gray = self._to_grayscale(frame)

        return {
            'local_entropy_mean': self._local_entropy_mean(gray),
            'local_entropy_variance': self._local_entropy_variance(gray),
            'gradient_magnitude_kurtosis': self._gradient_magnitude_kurtosis(gray),
            'natural_image_score': self._natural_image_score(gray),
            'complexity_measure': self._complexity_measure(gray),
            'edge_coherence': self._edge_coherence(gray),
        }

    def _local_entropy_mean(self, gray):
        """Mean of local entropy."""
        from skimage.filters.rank import entropy
        from skimage.morphology import disk

        gray_uint8 = gray.astype(np.uint8)
        local_entropy = entropy(gray_uint8, disk(5))

        return float(np.mean(local_entropy))

    def _local_entropy_variance(self, gray):
        """Variance of local entropy."""
        from skimage.filters.rank import entropy
        from skimage.morphology import disk

        gray_uint8 = gray.astype(np.uint8)
        local_entropy = entropy(gray_uint8, disk(5))

        return float(np.var(local_entropy))

    def _gradient_magnitude_kurtosis(self, gray):
        """
        Kurtosis of gradient magnitude distribution.
        Natural images have characteristic kurtosis.
        """
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)

        return float(stats.kurtosis(magnitude.flatten()))

    def _natural_image_score(self, gray):
        """
        Score based on natural image statistics (1/f spectrum).
        """
        f_transform = fft2(gray)
        magnitude = np.abs(fftshift(f_transform))

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Radial average of power spectrum
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

        power = magnitude**2
        radial_power = ndimage.mean(power, r, index=np.arange(1, r.max()))

        # Natural images follow 1/f^2 power law
        if len(radial_power) < 10:
            return 0.0

        freqs = np.arange(1, len(radial_power) + 1)
        log_freqs = np.log(freqs)
        log_power = np.log(radial_power + 1e-8)

        # Linear fit in log-log space
        slope, intercept = np.polyfit(log_freqs, log_power, 1)

        # Natural images have slope around -2
        natural_slope = -2
        deviation = abs(slope - natural_slope)

        return float(1 / (1 + deviation))

    def _complexity_measure(self, gray):
        """
        Image complexity based on compression ratio estimate.
        """
        # Use gradient as proxy for compressibility
        gx = np.diff(gray.astype(float), axis=1)
        gy = np.diff(gray.astype(float), axis=0)

        # Non-zero gradients indicate complexity
        complexity = (np.sum(np.abs(gx) > 1) + np.sum(np.abs(gy) > 1))
        complexity = complexity / (gray.size)

        return float(complexity)

    def _edge_coherence(self, gray):
        """
        Measure coherence of edge directions.
        """
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)

        # Angle field
        angles = np.arctan2(gy, gx)

        # Local coherence
        angle_cos = np.cos(angles)
        angle_sin = np.sin(angles)

        local_cos = ndimage.uniform_filter(angle_cos, size=5)
        local_sin = ndimage.uniform_filter(angle_sin, size=5)

        coherence = np.sqrt(local_cos**2 + local_sin**2)

        return float(np.mean(coherence))

    # =========================================================================
    # TEMPORAL ANALYSIS
    # =========================================================================

    def _update_cache(self, frame):
        """Update frame cache for temporal analysis."""
        gray = self._to_grayscale(frame)
        noise = self._extract_noise_residual(gray)

        self._frame_cache.append(gray)
        self._noise_residual_cache.append(noise)

        # Limit cache size
        if len(self._frame_cache) > self._max_cache_size:
            self._frame_cache.pop(0)
            self._noise_residual_cache.pop(0)

    def _compute_temporal_consistency(self):
        """Compute temporal consistency metrics from cached frames."""
        if len(self._noise_residual_cache) < 2:
            return {}

        noise_stack = np.stack(self._noise_residual_cache, axis=0)

        return {
            'temporal_noise_consistency': self._temporal_noise_consistency(noise_stack),
            'prnu_stability': self._prnu_stability(noise_stack),
            'noise_temporal_correlation': self._noise_temporal_correlation(noise_stack),
            'frame_to_frame_noise_variance': self._frame_to_frame_noise_variance(noise_stack),
        }

    def _temporal_noise_consistency(self, noise_stack):
        """
        PRNU should be identical frame-to-frame for real cameras.
        """
        # Average noise pattern
        mean_noise = np.mean(noise_stack, axis=0)

        # Correlation of each frame's noise with mean
        correlations = []
        for i in range(noise_stack.shape[0]):
            corr = np.corrcoef(noise_stack[i].flatten(), mean_noise.flatten())[0, 1]
            correlations.append(corr)

        return float(np.mean(correlations))

    def _prnu_stability(self, noise_stack):
        """
        Stability of PRNU pattern across frames.
        """
        # Variance of noise at each pixel location across time
        temporal_var = np.var(noise_stack, axis=0)

        # Should be low for real cameras (fixed pattern)
        return float(1 / (1 + np.mean(temporal_var)))

    def _noise_temporal_correlation(self, noise_stack):
        """
        Correlation between consecutive frame noise patterns.
        """
        correlations = []
        for i in range(len(noise_stack) - 1):
            corr = np.corrcoef(
                noise_stack[i].flatten(),
                noise_stack[i + 1].flatten()
            )[0, 1]
            correlations.append(corr)

        return float(np.mean(correlations))

    def _frame_to_frame_noise_variance(self, noise_stack):
        """
        Variance of noise difference between frames.
        """
        diffs = []
        for i in range(len(noise_stack) - 1):
            diff = noise_stack[i + 1] - noise_stack[i]
            diffs.append(np.var(diff))

        return float(np.mean(diffs))

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _to_grayscale(self, frame):
        """Convert RGB frame to grayscale."""
        if len(frame.shape) == 2:
            return frame
        return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

    def _aggregate_frame_results(self, all_results):
        """Aggregate results from multiple frames."""
        if not all_results:
            return {}

        aggregated = {}

        # Get all categories
        categories = all_results[0].keys()

        for category in categories:
            if category.startswith('_'):
                continue

            aggregated[category] = {}
            metrics = all_results[0][category].keys()

            for metric in metrics:
                values = [r[category][metric] for r in all_results if metric in r.get(category, {})]
                if values:
                    aggregated[category][metric] = float(np.mean(values))
                    # Only add _std features when multiple frames are available
                    if len(all_results) > 1:
                        aggregated[category][f"{metric}_std"] = float(np.std(values))

        return aggregated


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_video_forensics(video_path, max_frames=24):
    """
    Convenience function to analyze a video.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to analyze

    Returns:
        dict with forensic metrics
    """
    forensics = CameraForensics()
    return forensics.analyze_video(video_path, max_frames=max_frames)


def analyze_frame_forensics(frame):
    """
    Convenience function to analyze a single frame.

    Args:
        frame: numpy array (H, W, 3) RGB

    Returns:
        dict with forensic metrics
    """
    forensics = CameraForensics()
    return forensics.analyze_frame(frame)


def get_forensic_feature_vector(video_path, max_frames=24):
    """
    Get a flat feature vector for classification.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to analyze

    Returns:
        features: numpy array
        names: list of feature names
    """
    forensics = CameraForensics()
    results = forensics.analyze_video(video_path, max_frames=max_frames)
    return forensics.get_feature_vector(results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python camera_forensics.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Analyzing: {video_path}")

    results = analyze_video_forensics(video_path)

    print("\n" + "=" * 60)
    print("CAMERA FORENSICS ANALYSIS")
    print("=" * 60)

    for category, metrics in results.items():
        if category.startswith('_'):
            continue
        print(f"\n{category.upper()}")
        print("-" * 40)
        for metric, value in metrics.items():
            if not metric.endswith('_std'):
                std_key = f"{metric}_std"
                std = metrics.get(std_key, 0)
                print(f"  {metric}: {value:.4f} (±{std:.4f})")
