"""
Training script for Sparse Autoencoder (SAE) on video classifier activations.
This script trains SAEs to identify learned features in the transformer layers.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_autoencoder import SparseAutoencoder, MultiLayerSAE
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from dataset import VideoDataset, custom_collate_fn

# Configuration
BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SAE_DIR = os.path.join(MODEL_DIR, 'sae')
os.makedirs(SAE_DIR, exist_ok=True)

# SAE Training Configuration
SAE_CONFIG = {
    'input_dim': 768,  # Transformer embedding dimension
    'hidden_dim': 4096,  # Reduced for better feature utilization
    'sparsity_coefficient': 5e-4,  # Reduced L1 penalty (was 1e-3, caused 56% dead features)
    'tie_weights': True,  # Tie encoder/decoder weights
    'batch_size': 8,
    'learning_rate': 3e-4,
    'epochs': 20,
    'layer_to_hook': 'transformer_layer_6',  # Which transformer layer to analyze
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Early stopping based on L0 norm
    'l0_early_stopping_patience': 3,  # Stop if L0 rises for N consecutive epochs
    'min_epochs_before_stopping': 5,  # Don't stop before this many epochs
}


class ActivationHook:
    """Hook to capture activations from specific layers."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def register_hook(self, module, name):
        """Register a forward hook on a module."""
        def hook_fn(module, input, output):
            # Store the output activation
            self.activations[name] = output.detach()

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def clear(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_pretrained_classifier(model_path, device):
    """Load the pretrained video classifier."""
    print(f"Loading pretrained classifier from {model_path}")

    # Initialize model components
    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)

    # Create full model
    model = FullVideoClassifier(latentEncoder, patchEncoder, classifier).to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load with weights_only=False: {e}")
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle DataParallel format
    state_dict = checkpoint['model_state_dict']
    is_data_parallel = any(k.startswith('module.') for k in state_dict.keys())
    if is_data_parallel:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded classifier with accuracy: {checkpoint.get('best_accuracy', checkpoint.get('accuracy', 'N/A'))}")

    return model


def collect_activations(model, data_loader, hook, config, max_samples=10000):
    """
    Collect activations from the model on the dataset.

    Flattens variable-length sequences to (total_tokens, hidden_dim).

    Args:
        model: The video classifier model
        data_loader: DataLoader for the dataset
        hook: ActivationHook instance
        config: Configuration dictionary
        max_samples: Maximum number of video samples to collect

    Returns:
        Tensor of shape (total_tokens, hidden_dim)
    """
    device = config['device']
    activations_list = []
    samples_collected = 0

    print(f"Collecting activations from model (max {max_samples} samples)...")

    with torch.no_grad():
        for batch_idx, (data, labels, paths) in enumerate(tqdm(data_loader, desc='Collecting activations')):
            data = data.to(device)

            # Forward pass to trigger hooks
            _ = model(data)

            # Get activations from the hook
            for name, activation in hook.activations.items():
                # activation shape: (batch, seq_len, hidden_dim)
                # Flatten to (batch * seq_len, hidden_dim) to handle variable seq_len
                flat_activation = activation.reshape(-1, activation.shape[-1])
                activations_list.append(flat_activation.cpu())

            hook.clear()

            # Clear CUDA cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Track samples
            samples_collected += data.shape[0]
            if samples_collected >= max_samples:
                break

    # Concatenate all flattened activations
    all_activations = torch.cat(activations_list, dim=0)
    print(f"Collected activations shape: {all_activations.shape}")
    print(f"Total tokens: {all_activations.shape[0]}, Hidden dim: {all_activations.shape[1]}")

    return all_activations


def train_sae(sae, activations, config, writer):
    """
    Train the Sparse Autoencoder on collected activations.

    Args:
        sae: SparseAutoencoder model
        activations: Tensor of shape (total_tokens, hidden_dim)
        config: Configuration dictionary
        writer: TensorBoard writer
    """
    device = config['device']
    sae = sae.to(device)
    sae.train()

    # Create optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=config['learning_rate'])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    # Activations are already flattened to (total_tokens, hidden_dim)
    dataset = torch.utils.data.TensorDataset(activations)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'] * 64,  # Larger batches since tokens are small
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print(f"\nTraining SAE for {config['epochs']} epochs...")
    print(f"Dataset size: {len(dataset)} tokens")

    best_loss = float('inf')
    best_l0 = float('inf')  # Track best (lowest) L0 norm
    l0_rising_count = 0  # Count consecutive epochs where L0 rises
    prev_l0 = float('inf')
    early_stopped = False

    for epoch in range(config['epochs']):
        epoch_metrics = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'sparsity_loss': 0.0,
            'feature_density': 0.0,
            'l0_norm': 0.0
        }

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')

        for batch_idx, (batch_activations,) in enumerate(progress_bar):
            batch_activations = batch_activations.to(device)

            # Forward pass
            reconstructed, features, loss_dict = sae(batch_activations)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics.keys():
                epoch_metrics[key] += loss_dict[key].item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'recon': loss_dict['reconstruction_loss'].item(),
                'l0': loss_dict['l0_norm'].item()
            })

            # Log batch metrics
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('SAE/Batch_Loss', loss_dict['total_loss'].item(), global_step)

        # Calculate epoch averages
        num_batches = len(data_loader)
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches

        # Log epoch metrics
        writer.add_scalar('SAE/Epoch_Loss', epoch_metrics['total_loss'], epoch)
        writer.add_scalar('SAE/Reconstruction_Loss', epoch_metrics['reconstruction_loss'], epoch)
        writer.add_scalar('SAE/Sparsity_Loss', epoch_metrics['sparsity_loss'], epoch)
        writer.add_scalar('SAE/Feature_Density', epoch_metrics['feature_density'], epoch)
        writer.add_scalar('SAE/L0_Norm', epoch_metrics['l0_norm'], epoch)
        writer.add_scalar('SAE/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Get feature statistics
        stats = sae.get_feature_statistics()
        writer.add_scalar('SAE/Dead_Features', stats['dead_features'].item(), epoch)
        writer.add_scalar('SAE/Alive_Features', stats['alive_features'].item(), epoch)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_metrics['total_loss']:.6f}")
        print(f"  Reconstruction Loss: {epoch_metrics['reconstruction_loss']:.6f}")
        print(f"  Sparsity Loss: {epoch_metrics['sparsity_loss']:.6f}")
        print(f"  L0 Norm: {epoch_metrics['l0_norm']:.2f}")
        print(f"  Feature Density: {epoch_metrics['feature_density']:.2%}")
        dead_pct = 100 * stats['dead_features'].item() / config['hidden_dim']
        alive_pct = 100 * stats['alive_features'].item() / config['hidden_dim']
        print(f"  Dead Features: {stats['dead_features'].item()}/{config['hidden_dim']} ({dead_pct:.1f}%)")
        print(f"  Alive Features: {stats['alive_features'].item()}/{config['hidden_dim']} ({alive_pct:.1f}%)")

        # Save best model (lowest total loss)
        if epoch_metrics['total_loss'] < best_loss:
            best_loss = epoch_metrics['total_loss']
            save_path = os.path.join(SAE_DIR, 'sae_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': best_loss,
                'l0_norm': epoch_metrics['l0_norm'],
                'feature_stats': stats
            }, save_path)
            print(f"  Saved best loss model to {save_path}")

        # Save sparsest model (lowest L0 norm) - best for interpretability
        current_l0 = epoch_metrics['l0_norm']
        if current_l0 < best_l0:
            best_l0 = current_l0
            save_path = os.path.join(SAE_DIR, 'sae_sparse_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': epoch_metrics['total_loss'],
                'l0_norm': current_l0,
                'feature_stats': stats
            }, save_path)
            print(f"  Saved sparsest model (L0={current_l0:.2f}) to {save_path}")
            l0_rising_count = 0  # Reset counter when we find a new best
        else:
            # L0 didn't improve
            if current_l0 > prev_l0:
                l0_rising_count += 1
                print(f"  Warning: L0 rising for {l0_rising_count} epoch(s) (current: {current_l0:.2f}, best: {best_l0:.2f})")
            else:
                l0_rising_count = 0  # Reset if L0 decreased but didn't beat best

        prev_l0 = current_l0

        # Early stopping based on L0 norm rising
        min_epochs = config.get('min_epochs_before_stopping', 5)
        patience = config.get('l0_early_stopping_patience', 3)
        if epoch >= min_epochs and l0_rising_count >= patience:
            print(f"\n  Early stopping triggered: L0 has been rising for {patience} consecutive epochs")
            print(f"  Best L0 was {best_l0:.2f} at an earlier epoch")
            print(f"  Use sae_sparse_best.pt for the sparsest model")
            early_stopped = True
            break

        # Step scheduler
        scheduler.step()

        # Visualize feature activation distribution every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_feature_statistics(sae, epoch, writer)

    # Save final model
    final_epoch = epoch + 1 if not early_stopped else epoch + 1
    save_path = os.path.join(SAE_DIR, 'sae_final.pt')
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': epoch_metrics['total_loss'],
        'l0_norm': epoch_metrics['l0_norm'],
        'feature_stats': sae.get_feature_statistics(),
        'early_stopped': early_stopped
    }, save_path)
    print(f"\nSaved final model to {save_path}")

    # Print summary of saved models
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"  Epochs completed: {final_epoch}/{config['epochs']}")
    if early_stopped:
        print(f"  Early stopped: Yes (L0 rose for {patience} consecutive epochs)")
    print(f"  Best total loss: {best_loss:.6f}")
    print(f"  Best L0 norm: {best_l0:.2f}")
    print(f"\nSaved models:")
    print(f"  sae_best.pt        - Lowest total loss (best reconstruction)")
    print(f"  sae_sparse_best.pt - Lowest L0 norm (best for interpretability)")
    print(f"  sae_final.pt       - Final epoch state")
    print(f"{'='*60}")


def visualize_feature_statistics(sae, epoch, writer):
    """Visualize SAE feature statistics."""
    stats = sae.get_feature_statistics()
    activation_rates = stats['feature_activation_rates'].cpu().numpy()

    # Create histogram of activation rates
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(activation_rates, bins=50, edgecolor='black')
    axes[0].set_xlabel('Activation Rate')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title(f'Feature Activation Rate Distribution (Epoch {epoch+1})')
    axes[0].axvline(1e-3, color='r', linestyle='--', label='Dead Feature Threshold')
    axes[0].legend()

    # Log scale histogram
    axes[1].hist(activation_rates[activation_rates > 0], bins=50, edgecolor='black')
    axes[1].set_xlabel('Activation Rate')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title(f'Feature Activation Rate (Log Scale, Epoch {epoch+1})')
    axes[1].set_yscale('log')

    plt.tight_layout()
    writer.add_figure('SAE/Feature_Activation_Distribution', fig, epoch)
    plt.close()


def main():
    """Main training function."""
    print("="*80)
    print("Training Sparse Autoencoder for Video Classifier Interpretability")
    print("="*80)

    # Setup
    device = torch.device(SAE_CONFIG['device'])
    print(f"Using device: {device}")

    # Load pretrained classifier
    model_path = os.path.join(MODEL_DIR, 'full_classifier_best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'full_classifier_1_85.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No pretrained model found at {MODEL_DIR}")

    classifier = load_pretrained_classifier(model_path, device)

    # Setup activation hook on transformer layer
    hook = ActivationHook()

    # Hook into the transformer encoder output
    # The transformer output is what we want to analyze
    target_layer = classifier.classifier.transformer_encoder.layers[-1]  # Last transformer layer
    hook.register_hook(target_layer, 'transformer_output')

    print(f"Registered hook on transformer layer")

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}")
    dataset = VideoDataset(DATASET_PATH, target_size=512, max_frames=24)
    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=SAE_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=SAE_CONFIG['num_workers'],
        pin_memory=True
    )

    # Collect activations (streams to disk to avoid OOM)
    # Increase max_samples if you have more RAM available
    activations = collect_activations(
        classifier,
        data_loader,
        hook,
        SAE_CONFIG,
        max_samples=3000  # Increased for better feature coverage
    )

    # Remove hooks
    hook.remove_hooks()

    # Initialize SAE
    print(f"\nInitializing Sparse Autoencoder...")
    print(f"  Input dim: {SAE_CONFIG['input_dim']}")
    print(f"  Hidden dim: {SAE_CONFIG['hidden_dim']}")
    print(f"  Sparsity coefficient: {SAE_CONFIG['sparsity_coefficient']}")

    sae = SparseAutoencoder(
        input_dim=SAE_CONFIG['input_dim'],
        hidden_dim=SAE_CONFIG['hidden_dim'],
        sparsity_coefficient=SAE_CONFIG['sparsity_coefficient'],
        tie_weights=SAE_CONFIG['tie_weights']
    )

    # Setup TensorBoard
    tensorboard_dir = os.path.join(BASE_DIR, 'runs')
    run_name = f"sae_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f'{tensorboard_dir}/{run_name}')

    # Train SAE
    train_sae(sae, activations, SAE_CONFIG, writer)

    # Close writer
    writer.close()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"SAE models saved to: {SAE_DIR}")
    print(f"TensorBoard logs: {tensorboard_dir}/{run_name}")
    print("="*80)


if __name__ == "__main__":
    main()
