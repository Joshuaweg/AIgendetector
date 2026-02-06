"""
Visualization utilities for Sparse Autoencoder features.
Helps interpret what features the SAE has learned.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import cv2
from tqdm import tqdm

from sparse_autoencoder import SparseAutoencoder
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from dataset import VideoDataset, custom_collate_fn
from torch.utils.data import DataLoader


class SAEFeatureVisualizer:
    """Visualizer for SAE features."""

    def __init__(self, sae: SparseAutoencoder, device: str = 'cuda'):
        """
        Args:
            sae: Trained SparseAutoencoder
            device: Device to run on
        """
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device

    def get_top_activating_samples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find samples that maximally activate a specific feature.

        Args:
            feature_idx: Index of the feature to analyze
            activations: All activations (n_samples, seq_len, input_dim)
            k: Number of top samples to return

        Returns:
            Tuple of (top_activation_values, top_sample_indices)
        """
        with torch.no_grad():
            activations = activations.to(self.device)

            # Encode to get feature activations
            # Handle both 2D and 3D inputs
            if activations.dim() == 3:
                batch, seq, dim = activations.shape
                activations_flat = activations.reshape(-1, dim)
                features = self.sae.encode(activations_flat)
                features = features.reshape(batch, seq, -1)
            else:
                features = self.sae.encode(activations)

            # Get activations for this specific feature
            feature_activations = features[..., feature_idx]

            # Flatten to get max activation per sample
            if feature_activations.dim() == 2:
                # (batch, seq) -> max over sequence
                max_activations, _ = feature_activations.max(dim=1)
            else:
                max_activations = feature_activations

            # Get top k
            top_values, top_indices = torch.topk(max_activations, min(k, len(max_activations)))

            return top_values.cpu(), top_indices.cpu()

    def visualize_feature_direction(
        self,
        feature_idx: int,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Visualize what a feature represents by decoding it in isolation.

        Args:
            feature_idx: Index of the feature
            save_path: Optional path to save the visualization

        Returns:
            The decoded feature direction
        """
        with torch.no_grad():
            # Get the decoder direction for this feature
            feature_direction = self.sae.reconstruct_from_feature(feature_idx, activation=1.0)

            # Convert to numpy for visualization
            feature_np = feature_direction.cpu().numpy()

            # Plot
            fig, ax = plt.subplots(figsize=(12, 4))

            # Plot as heatmap if dimension is reasonable
            if len(feature_np) <= 1024:
                # Reshape for better visualization
                n = len(feature_np)
                ncols = min(64, n)
                nrows = int(np.ceil(n / ncols))

                # Pad to make it rectangular
                padded = np.zeros(nrows * ncols)
                padded[:n] = feature_np
                feature_2d = padded.reshape(nrows, ncols)

                sns.heatmap(feature_2d, cmap='RdBu_r', center=0, ax=ax, cbar=True)
                ax.set_title(f'Feature {feature_idx} Decoder Direction')
            else:
                # Plot as line plot for high dimensions
                ax.plot(feature_np)
                ax.set_title(f'Feature {feature_idx} Decoder Direction')
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Value')
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved feature visualization to {save_path}")

            plt.close()

            return feature_direction

    def analyze_feature_interpretability(
        self,
        activations: torch.Tensor,
        top_k: int = 20
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze interpretability metrics for all features.

        Args:
            activations: Sample activations
            top_k: Number of top features to analyze in detail

        Returns:
            Dictionary mapping feature indices to their interpretability metrics
        """
        with torch.no_grad():
            activations = activations.to(self.device)

            # Encode
            if activations.dim() == 3:
                batch, seq, dim = activations.shape
                activations_flat = activations.reshape(-1, dim)
            else:
                activations_flat = activations

            features = self.sae.encode(activations_flat)

            # Calculate metrics for each feature
            feature_metrics = {}

            # Activation frequency
            activation_freq = (features > 0).float().mean(dim=0)

            # Mean activation (when active)
            mean_activation = torch.zeros(features.shape[1], device=self.device)
            for i in range(features.shape[1]):
                active_mask = features[:, i] > 0
                if active_mask.sum() > 0:
                    mean_activation[i] = features[active_mask, i].mean()

            # Max activation
            max_activation, _ = features.max(dim=0)

            # Sparsity (lower is sparser)
            sparsity = (features > 0).float().mean(dim=0)

            # Get statistics
            stats = self.sae.get_feature_statistics()
            activation_rates = stats['feature_activation_rates']

            # Find top features by activation frequency
            top_features = torch.argsort(activation_freq, descending=True)[:top_k]

            for idx in top_features.cpu().tolist():
                feature_metrics[idx] = {
                    'activation_frequency': activation_freq[idx].item(),
                    'mean_activation': mean_activation[idx].item(),
                    'max_activation': max_activation[idx].item(),
                    'sparsity': sparsity[idx].item(),
                    'activation_rate': activation_rates[idx].item()
                }

            return feature_metrics

    def create_feature_dashboard(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        save_dir: str,
        samples_data: Optional[List[Tuple[torch.Tensor, str]]] = None
    ):
        """
        Create a comprehensive dashboard for a single feature.

        Args:
            activations: Activations to analyze
            feature_idx: Feature to visualize
            save_dir: Directory to save visualizations
            samples_data: Optional list of (sample, label) tuples for context
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"Creating dashboard for feature {feature_idx}...")

        # 1. Feature direction
        feature_dir_path = os.path.join(save_dir, f'feature_{feature_idx}_direction.png')
        self.visualize_feature_direction(feature_idx, feature_dir_path)

        # 2. Activation distribution
        with torch.no_grad():
            activations = activations.to(self.device)

            if activations.dim() == 3:
                batch, seq, dim = activations.shape
                activations_flat = activations.reshape(-1, dim)
            else:
                activations_flat = activations

            features = self.sae.encode(activations_flat)
            feature_activations = features[:, feature_idx].cpu().numpy()

            # Plot activation distribution
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Histogram of all activations
            axes[0].hist(feature_activations, bins=50, edgecolor='black')
            axes[0].set_xlabel('Activation Value')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'Feature {feature_idx} Activation Distribution')
            axes[0].axvline(0, color='r', linestyle='--', label='Zero')
            axes[0].legend()

            # Histogram of non-zero activations only
            non_zero = feature_activations[feature_activations > 0]
            if len(non_zero) > 0:
                axes[1].hist(non_zero, bins=50, edgecolor='black')
                axes[1].set_xlabel('Activation Value')
                axes[1].set_ylabel('Count')
                axes[1].set_title(f'Feature {feature_idx} Non-Zero Activations')

            plt.tight_layout()
            dist_path = os.path.join(save_dir, f'feature_{feature_idx}_distribution.png')
            plt.savefig(dist_path, dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Statistics
        stats = self.sae.get_feature_statistics()
        activation_rate = stats['feature_activation_rates'][feature_idx].item()

        stats_path = os.path.join(save_dir, f'feature_{feature_idx}_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Feature {feature_idx} Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Activation rate: {activation_rate:.4%}\n")
            f.write(f"Mean activation (non-zero): {non_zero.mean() if len(non_zero) > 0 else 0:.4f}\n")
            f.write(f"Max activation: {feature_activations.max():.4f}\n")
            f.write(f"Std activation: {feature_activations.std():.4f}\n")
            f.write(f"Sparsity: {(feature_activations == 0).mean():.4%}\n")

        print(f"Dashboard saved to {save_dir}")


def compare_features_across_classes(
    sae: SparseAutoencoder,
    activations_fake: torch.Tensor,
    activations_real: torch.Tensor,
    top_k: int = 20,
    save_path: Optional[str] = None
):
    """
    Compare which features activate differently for fake vs real videos.

    Args:
        sae: Trained SAE
        activations_fake: Activations from fake videos
        activations_real: Activations from real videos
        top_k: Number of top discriminative features to return
        save_path: Optional path to save visualization
    """
    device = next(sae.parameters()).device

    with torch.no_grad():
        # Encode both
        if activations_fake.dim() == 3:
            batch_f, seq_f, dim = activations_fake.shape
            activations_fake_flat = activations_fake.reshape(-1, dim)
        else:
            activations_fake_flat = activations_fake

        if activations_real.dim() == 3:
            batch_r, seq_r, dim = activations_real.shape
            activations_real_flat = activations_real.reshape(-1, dim)
        else:
            activations_real_flat = activations_real

        features_fake = sae.encode(activations_fake_flat.to(device))
        features_real = sae.encode(activations_real_flat.to(device))

        # Mean activation per feature for each class
        mean_fake = features_fake.mean(dim=0)
        mean_real = features_real.mean(dim=0)

        # Difference
        difference = (mean_fake - mean_real).abs()

        # Top discriminative features
        top_features = torch.argsort(difference, descending=True)[:top_k]

        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Mean activations comparison
        x = np.arange(top_k)
        width = 0.35

        fake_vals = mean_fake[top_features].cpu().numpy()
        real_vals = mean_real[top_features].cpu().numpy()

        axes[0].bar(x - width/2, fake_vals, width, label='AI-Generated', alpha=0.8)
        axes[0].bar(x + width/2, real_vals, width, label='Real', alpha=0.8)
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Mean Activation')
        axes[0].set_title('Top Discriminative Features: Mean Activation Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'{i.item()}' for i in top_features])
        axes[0].legend()

        # Plot 2: Difference magnitude
        diff_vals = difference[top_features].cpu().numpy()
        axes[1].bar(x, diff_vals, color='coral', alpha=0.8)
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('|Difference|')
        axes[1].set_title('Feature Discrimination Power')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{i.item()}' for i in top_features])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")

        plt.close()

        return top_features.cpu(), difference[top_features].cpu()


def main():
    """Example usage of visualization utilities."""
    import sys

    BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
    SAE_DIR = os.path.join(BASE_DIR, 'model', 'sae')
    VIZ_DIR = os.path.join(SAE_DIR, 'visualizations')
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load trained SAE
    sae_path = os.path.join(SAE_DIR, 'sae_best.pt')
    if not os.path.exists(sae_path):
        print(f"No trained SAE found at {sae_path}")
        print("Please run train_sae.py first")
        sys.exit(1)

    print(f"Loading SAE from {sae_path}")
    checkpoint = torch.load(sae_path, map_location='cuda', weights_only=False)

    config = checkpoint['config']
    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        sparsity_coefficient=config['sparsity_coefficient'],
        tie_weights=config['tie_weights']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualizer = SAEFeatureVisualizer(sae, device)

    print("SAE loaded successfully")
    print(f"Feature statistics: {checkpoint.get('feature_stats', 'N/A')}")

    # For demonstration, create some random activations
    # In practice, you'd load real activations from your model
    print("\nGenerating example visualizations...")

    dummy_activations = torch.randn(100, 50, config['input_dim'])

    # Analyze top features
    print("Analyzing feature interpretability...")
    feature_metrics = visualizer.analyze_feature_interpretability(dummy_activations, top_k=10)

    print("\nTop 10 Features by Activation Frequency:")
    for idx, metrics in sorted(feature_metrics.items(), key=lambda x: x[1]['activation_frequency'], reverse=True):
        print(f"Feature {idx}:")
        print(f"  Activation frequency: {metrics['activation_frequency']:.4%}")
        print(f"  Mean activation: {metrics['mean_activation']:.4f}")
        print(f"  Max activation: {metrics['max_activation']:.4f}")

    # Create dashboard for top feature
    top_feature = max(feature_metrics.items(), key=lambda x: x[1]['activation_frequency'])[0]
    dashboard_dir = os.path.join(VIZ_DIR, f'feature_{top_feature}')

    print(f"\nCreating dashboard for feature {top_feature}...")
    visualizer.create_feature_dashboard(
        dummy_activations,
        top_feature,
        dashboard_dir
    )

    print(f"\nVisualizations saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
