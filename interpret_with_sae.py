"""
Integration of Sparse Autoencoder analysis with the existing interpretation pipeline.
Combines Integrated Gradients with SAE feature analysis for comprehensive interpretability.
"""

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm

from sparse_autoencoder import SparseAutoencoder
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from interpret import load_model_correctly, load_video, select_random_video
from visualize_sae_features import SAEFeatureVisualizer


class ActivationCapture:
    """Captures activations from specific layers during forward pass."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def register_hook(self, module, name):
        """Register forward hook on a module."""
        def hook_fn(module, input, output):
            self.activations[name] = output.detach().clone()

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def clear(self):
        """Clear stored activations."""
        self.activations.clear()

    def remove_all_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def load_sae(sae_path: str, device: str) -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    print(f"Loading SAE from {sae_path}")

    try:
        checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading SAE: {e}")
        raise

    config = checkpoint['config']

    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        sparsity_coefficient=config.get('sparsity_coefficient', 1e-3),
        tie_weights=config.get('tie_weights', True)
    )

    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    print(f"SAE loaded successfully")
    if 'feature_stats' in checkpoint:
        stats = checkpoint['feature_stats']
        print(f"  Dead features: {stats.get('dead_features', 'N/A')}")
        print(f"  Alive features: {stats.get('alive_features', 'N/A')}")

    return sae


def analyze_video_with_sae(
    video_path: str,
    model: nn.Module,
    sae: SparseAutoencoder,
    device: str,
    save_dir: str
) -> Dict:
    """
    Analyze a video using both the classifier and SAE.

    Args:
        video_path: Path to video file
        model: Video classifier model
        sae: Trained SAE
        device: Device to run on
        save_dir: Directory to save results

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load video
    print(f"\nLoading video: {video_path}")
    video, frames, label, _ = load_video(video_path)
    video = video.unsqueeze(0).to(device)  # Add batch dimension

    # Setup activation capture
    activation_capture = ActivationCapture()

    # Hook into the last transformer layer
    target_layer = model.classifier.transformer_encoder.layers[-1]
    activation_capture.register_hook(target_layer, 'transformer_output')

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(video)
        pred = torch.argmax(output, dim=1)
        confidence = torch.softmax(output, dim=1)[0][pred.item()].item()

    # Get captured activations
    activations = activation_capture.activations['transformer_output']
    print(f"Captured activations shape: {activations.shape}")

    # Analyze with SAE
    print("Analyzing activations with SAE...")
    with torch.no_grad():
        reconstructed, features, loss_dict = sae(activations)

    # Remove hooks
    activation_capture.remove_all_hooks()

    # Get prediction info
    classes = ['AI-Generated', 'Real']
    predicted_class = classes[pred.item()]
    true_class = classes[label.item()]

    print(f"\nPrediction: {predicted_class} (confidence: {confidence:.2%})")
    print(f"True label: {true_class}")
    print(f"Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"Sparsity (L0): {loss_dict['l0_norm'].item():.2f} active features")
    print(f"Feature density: {loss_dict['feature_density'].item():.2%}")

    # Analyze which features are active
    features_np = features.cpu().numpy().squeeze(0)  # Remove batch dim

    # Get top features across all time steps
    features_max_per_feature = features_np.max(axis=0)  # Max activation per feature
    top_k = 20
    top_feature_indices = np.argsort(features_max_per_feature)[-top_k:][::-1]

    print(f"\nTop {top_k} active features:")
    for i, feat_idx in enumerate(top_feature_indices):
        max_act = features_max_per_feature[feat_idx]
        mean_act = features_np[:, feat_idx].mean()
        print(f"  {i+1}. Feature {feat_idx}: max={max_act:.3f}, mean={mean_act:.3f}")

    # Visualizations
    visualize_sae_analysis(
        features_np,
        top_feature_indices,
        predicted_class,
        confidence,
        save_dir
    )

    # Analyze temporal evolution of top features
    visualize_temporal_features(
        features_np,
        top_feature_indices,
        predicted_class,
        save_dir
    )

    results = {
        'video_path': video_path,
        'predicted_class': predicted_class,
        'true_class': true_class,
        'confidence': confidence,
        'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
        'l0_norm': loss_dict['l0_norm'].item(),
        'feature_density': loss_dict['feature_density'].item(),
        'top_features': top_feature_indices.tolist(),
        'top_feature_activations': features_max_per_feature[top_feature_indices].tolist()
    }

    # Save results
    results_path = os.path.join(save_dir, 'analysis_results.txt')
    with open(results_path, 'w') as f:
        f.write("SAE Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Predicted: {predicted_class} (confidence: {confidence:.2%})\n")
        f.write(f"True label: {true_class}\n\n")
        f.write(f"SAE Metrics:\n")
        f.write(f"  Reconstruction Loss: {results['reconstruction_loss']:.4f}\n")
        f.write(f"  L0 Norm (active features): {results['l0_norm']:.2f}\n")
        f.write(f"  Feature Density: {results['feature_density']:.2%}\n\n")
        f.write(f"Top {top_k} Features:\n")
        for i, (feat_idx, act_val) in enumerate(zip(top_feature_indices, results['top_feature_activations'])):
            f.write(f"  {i+1}. Feature {feat_idx}: {act_val:.3f}\n")

    print(f"\nResults saved to {save_dir}")

    return results


def visualize_sae_analysis(
    features: np.ndarray,
    top_feature_indices: np.ndarray,
    predicted_class: str,
    confidence: float,
    save_dir: str
):
    """Create visualizations of SAE analysis."""

    # 1. Heatmap of top features over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Top features heatmap
    top_features_data = features[:, top_feature_indices].T  # (features, time)

    sns.heatmap(
        top_features_data,
        cmap='YlOrRd',
        ax=axes[0],
        cbar_kws={'label': 'Activation'},
        yticklabels=[f'F{i}' for i in top_feature_indices]
    )
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Feature')
    axes[0].set_title(f'Top Feature Activations Over Time\nPrediction: {predicted_class} ({confidence:.1%})')

    # Feature activation distribution
    feature_means = features.mean(axis=0)
    axes[1].hist(feature_means[feature_means > 0], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Mean Activation')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title('Distribution of Mean Feature Activations (Active Features Only)')
    axes[1].set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'sae_feature_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature analysis to {save_path}")


def visualize_temporal_features(
    features: np.ndarray,
    top_feature_indices: np.ndarray,
    predicted_class: str,
    save_dir: str
):
    """Visualize how features evolve over time."""

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_feature_indices[:20]):
        ax = axes[i]
        feature_timeline = features[:, feat_idx]

        ax.plot(feature_timeline, linewidth=2)
        ax.fill_between(range(len(feature_timeline)), feature_timeline, alpha=0.3)
        ax.set_title(f'Feature {feat_idx}', fontsize=10)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Activation', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Temporal Evolution of Top 20 Features\nPrediction: {predicted_class}', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'temporal_features.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved temporal analysis to {save_path}")


def compare_fake_vs_real_features(
    model: nn.Module,
    sae: SparseAutoencoder,
    fake_video_path: str,
    real_video_path: str,
    device: str,
    save_dir: str
):
    """Compare SAE features between fake and real videos."""

    os.makedirs(save_dir, exist_ok=True)

    def get_features(video_path):
        video, frames, label, _ = load_video(video_path)
        video = video.unsqueeze(0).to(device)

        activation_capture = ActivationCapture()
        target_layer = model.classifier.transformer_encoder.layers[-1]
        activation_capture.register_hook(target_layer, 'transformer_output')

        with torch.no_grad():
            output = model(video)
            activations = activation_capture.activations['transformer_output']
            _, features, _ = sae(activations)

        activation_capture.remove_all_hooks()
        return features.cpu().numpy().squeeze(0)

    print("Analyzing fake video...")
    features_fake = get_features(fake_video_path)

    print("Analyzing real video...")
    features_real = get_features(real_video_path)

    # Compare mean activations
    mean_fake = features_fake.mean(axis=0)
    mean_real = features_real.mean(axis=0)

    difference = np.abs(mean_fake - mean_real)
    top_discriminative = np.argsort(difference)[-20:][::-1]

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Bar chart comparison
    x = np.arange(len(top_discriminative))
    width = 0.35

    axes[0].bar(x - width/2, mean_fake[top_discriminative], width, label='AI-Generated', alpha=0.8)
    axes[0].bar(x + width/2, mean_real[top_discriminative], width, label='Real', alpha=0.8)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Mean Activation')
    axes[0].set_title('Top Discriminative Features: Fake vs Real')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{i}' for i in top_discriminative])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Difference magnitude
    axes[1].bar(x, difference[top_discriminative], color='coral', alpha=0.8)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('|Difference|')
    axes[1].set_title('Discrimination Power of Features')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{i}' for i in top_discriminative])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fake_vs_real_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTop discriminative features:")
    for i, feat_idx in enumerate(top_discriminative[:10]):
        print(f"  {i+1}. Feature {feat_idx}: fake={mean_fake[feat_idx]:.3f}, real={mean_real[feat_idx]:.3f}, diff={difference[feat_idx]:.3f}")

    print(f"\nComparison saved to {save_path}")


def main():
    """Main function for SAE-enhanced interpretation."""

    BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    SAE_DIR = os.path.join(MODEL_DIR, 'sae')
    RESULTS_DIR = os.path.join(BASE_DIR, 'sae_interpretation_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = os.path.join(MODEL_DIR, 'full_classifier_best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'full_classifier_1_85.pt')

    print(f"Loading classifier from {model_path}")
    model = load_model_correctly(model_path, device)

    # Load SAE
    sae_path = os.path.join(SAE_DIR, 'sae_best.pt')
    if not os.path.exists(sae_path):
        print(f"Error: No trained SAE found at {sae_path}")
        print("Please run train_sae.py first")
        return

    sae = load_sae(sae_path, device)

    # Analyze a random video
    print("\n" + "="*80)
    print("Analyzing random video with SAE")
    print("="*80)

    video_path = select_random_video()
    video_save_dir = os.path.join(RESULTS_DIR, 'random_video_analysis')

    results = analyze_video_with_sae(
        video_path,
        model,
        sae,
        device,
        video_save_dir
    )

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
