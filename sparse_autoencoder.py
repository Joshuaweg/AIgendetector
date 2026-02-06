"""
Sparse Autoencoder for interpretability of the video classifier.
Implements a sparse autoencoder to identify learned features in the transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for feature extraction and interpretability.

    The SAE learns to reconstruct activations from a layer while maintaining sparsity,
    which helps identify monosemantic features learned by the model.

    Args:
        input_dim: Dimension of the input activations (e.g., 768 for transformer embeddings)
        hidden_dim: Dimension of the sparse feature space (typically larger than input_dim)
        sparsity_coefficient: Weight for L1 sparsity penalty
        tie_weights: Whether to tie encoder and decoder weights (transposed)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 4096,
        sparsity_coefficient: float = 1e-3,
        tie_weights: bool = True,
        use_bias: bool = True
    ):
        super(SparseAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coefficient = sparsity_coefficient
        self.tie_weights = tie_weights

        # Pre-encoder bias (learned offset before encoding)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim)) if use_bias else None

        # Encoder: maps from input_dim to hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=use_bias)

        # Decoder: maps from hidden_dim back to input_dim
        if tie_weights:
            # Don't create separate decoder weights, use transposed encoder weights
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim)) if use_bias else None
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=use_bias)

        # Initialize weights using techniques from "Towards Monosemanticity"
        self._init_weights()

        # Statistics for monitoring
        self.register_buffer('feature_activation_counts', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0))

    def _init_weights(self):
        """Initialize weights using proper scaling for SAE training."""
        # Encoder initialization - normalized columns (important for SAE)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Normalize encoder weight columns to unit norm
        with torch.no_grad():
            self.encoder.weight.data = F.normalize(self.encoder.weight.data, dim=0)

        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        # Decoder initialization
        if not self.tie_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)
        else:
            if self.decoder_bias is not None:
                nn.init.zeros_(self.decoder_bias)

        if self.pre_bias is not None:
            nn.init.zeros_(self.pre_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Sparse feature activations of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        """
        # Subtract learned pre-bias
        if self.pre_bias is not None:
            x = x - self.pre_bias

        # Linear projection
        h = self.encoder(x)

        # ReLU activation for sparsity
        features = F.relu(h)

        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to original activation space.

        Args:
            features: Sparse features of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            Reconstructed activations of shape (batch, seq_len, input_dim) or (batch, input_dim)
        """
        if self.tie_weights:
            # Use transposed encoder weights
            x_reconstructed = F.linear(features, self.encoder.weight.t(), self.decoder_bias)
        else:
            x_reconstructed = self.decoder(features)

        # Add back the pre-bias
        if self.pre_bias is not None:
            x_reconstructed = x_reconstructed + self.pre_bias

        return x_reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations

        Returns:
            Tuple of (reconstructed, features, loss_dict)
            - reconstructed: Reconstructed input
            - features: Sparse feature activations
            - loss_dict: Dictionary containing loss components
        """
        # Encode to sparse features
        features = self.encode(x)

        # Decode back to original space
        reconstructed = self.decode(features)

        # Calculate losses
        loss_dict = self.calculate_loss(x, reconstructed, features)

        # Update statistics (only during training)
        if self.training:
            self._update_statistics(features)

        return reconstructed, features, loss_dict

    def calculate_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate SAE loss components.

        Args:
            original: Original input activations
            reconstructed: Reconstructed activations
            features: Sparse features from encoder

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original)

        # L1 sparsity loss on features
        sparsity_loss = torch.mean(torch.abs(features))

        # Total loss
        total_loss = reconstruction_loss + self.sparsity_coefficient * sparsity_loss

        # Additional metrics
        with torch.no_grad():
            # Feature density (percentage of non-zero features)
            feature_density = (features > 0).float().mean()

            # Average L0 norm (number of active features per sample)
            l0_norm = (features > 0).float().sum(dim=-1).mean()

        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'feature_density': feature_density,
            'l0_norm': l0_norm
        }

        return loss_dict

    def _update_statistics(self, features: torch.Tensor):
        """Update feature activation statistics for monitoring dead features."""
        with torch.no_grad():
            # Flatten to (total_samples, hidden_dim)
            if features.dim() == 3:
                features_flat = features.reshape(-1, self.hidden_dim)
            else:
                features_flat = features

            # Count activations (feature > 0)
            active = (features_flat > 0).float().sum(dim=0)
            self.feature_activation_counts += active
            self.total_samples += features_flat.shape[0]

    def get_feature_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get statistics about feature usage.

        Returns:
            Dictionary with feature statistics
        """
        if self.total_samples == 0:
            return {
                'dead_features': torch.tensor(0),
                'feature_activation_rates': torch.zeros(self.hidden_dim),
                'alive_features': torch.tensor(self.hidden_dim)
            }

        activation_rates = self.feature_activation_counts / self.total_samples.float()

        # Consider feature "dead" if it activates less than 0.1% of the time
        dead_threshold = 1e-3
        dead_features = (activation_rates < dead_threshold).sum()
        alive_features = (activation_rates >= dead_threshold).sum()

        return {
            'dead_features': dead_features,
            'alive_features': alive_features,
            'feature_activation_rates': activation_rates,
            'mean_activation_rate': activation_rates.mean(),
            'median_activation_rate': activation_rates.median()
        }

    def reset_statistics(self):
        """Reset feature statistics."""
        self.feature_activation_counts.zero_()
        self.total_samples.zero_()

    def get_top_activating_features(
        self,
        features: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k activating features for each sample.

        Args:
            features: Feature activations of shape (batch, hidden_dim) or (batch, seq, hidden_dim)
            k: Number of top features to return

        Returns:
            Tuple of (top_values, top_indices)
        """
        # Handle both 2D and 3D inputs
        if features.dim() == 3:
            batch, seq, hidden = features.shape
            features_flat = features.reshape(batch * seq, hidden)
            top_values, top_indices = torch.topk(features_flat, k, dim=-1)
            top_values = top_values.reshape(batch, seq, k)
            top_indices = top_indices.reshape(batch, seq, k)
        else:
            top_values, top_indices = torch.topk(features, k, dim=-1)

        return top_values, top_indices

    def reconstruct_from_feature(self, feature_idx: int, activation: float = 1.0) -> torch.Tensor:
        """
        Reconstruct what a single feature represents by decoding it in isolation.

        Args:
            feature_idx: Index of the feature to reconstruct
            activation: Activation level for the feature

        Returns:
            Reconstructed activation pattern for this feature
        """
        with torch.no_grad():
            # Create zero vector with single feature activated
            feature_vec = torch.zeros(1, self.hidden_dim, device=self.encoder.weight.device)
            feature_vec[0, feature_idx] = activation

            # Decode to see what this feature represents
            reconstruction = self.decode(feature_vec)

            return reconstruction.squeeze(0)


class MultiLayerSAE(nn.Module):
    """
    Multiple SAEs for different layers of the transformer.
    Allows analyzing features at different depths.
    """

    def __init__(
        self,
        layer_dims: Dict[str, int],
        hidden_dim: int = 4096,
        sparsity_coefficient: float = 1e-3,
        tie_weights: bool = True
    ):
        """
        Args:
            layer_dims: Dictionary mapping layer names to their dimensions
            hidden_dim: Hidden dimension for all SAEs
            sparsity_coefficient: Sparsity coefficient for all SAEs
            tie_weights: Whether to tie weights in SAEs
        """
        super(MultiLayerSAE, self).__init__()

        self.layer_names = list(layer_dims.keys())
        self.saes = nn.ModuleDict({
            name: SparseAutoencoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                sparsity_coefficient=sparsity_coefficient,
                tie_weights=tie_weights
            )
            for name, dim in layer_dims.items()
        })

    def forward(
        self,
        activations: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process activations from multiple layers.

        Args:
            activations: Dictionary mapping layer names to their activations

        Returns:
            Dictionary mapping layer names to (reconstructed, features, loss_dict) tuples
        """
        results = {}
        for name, act in activations.items():
            if name in self.saes:
                results[name] = self.saes[name](act)
        return results

    def get_all_statistics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get feature statistics for all layers."""
        return {
            name: sae.get_feature_statistics()
            for name, sae in self.saes.items()
        }

    def reset_all_statistics(self):
        """Reset statistics for all SAEs."""
        for sae in self.saes.values():
            sae.reset_statistics()
