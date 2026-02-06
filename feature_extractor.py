import torch
import torch.nn as nn
from full_scale_classifier import FullVideoClassifier, FullLatentEncoder, FullPatchEncoder, FullClassifier
from interpret import load_video
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

class FeatureExtractor:
    def __init__(self, model_path, device=None, enable_visualizations=True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Store visualization flag
        self.enable_visualizations = enable_visualizations
        
        # Initialize model components
        self.latent_encoder = FullLatentEncoder().to(self.device)
        self.patch_encoder = FullPatchEncoder().to(self.device)
        self.classifier = FullClassifier().to(self.device)
        
        # Create full model
        self.model = FullVideoClassifier(
            self.latent_encoder,
            self.patch_encoder,
            self.classifier
        ).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel state dict
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Initialize hooks and storage
        self.hooks = []
        self.features = {}
        
        # Create output directory for visualizations only if enabled
        if self.enable_visualizations:
            os.makedirs('feature_visualizations', exist_ok=True)
    
    def visualize_patch_vectors(self, features, save_dir=None):
        """
        Visualize each patch vector as a separate figure.
        
        Args:
            features: Dictionary containing extracted features
            save_dir: Directory to save visualizations. If None, will use default path.
        """
        patch_features = features['patch_features']
        patches = patch_features.squeeze(0)  # Shape: (768, feature_dim)
        
        # Create save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_patches'
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate number of features per patch
        feature_dim = patches.shape[1]
        grid_size = int(np.ceil(np.sqrt(feature_dim)))
        
        # Create figure for each patch
        for i in tqdm(range(768), desc="Saving patch visualizations"):
            # Create new figure for this patch
            plt.figure(figsize=(8, 8))
            
            # Get feature vector for this patch
            patch_vector = patches[i]
            
            # Reshape feature vector to a square grid
            patch_grid = np.zeros((grid_size, grid_size))
            patch_grid.flat[:feature_dim] = patch_vector
            
            # Create heatmap for this patch
            sns.heatmap(patch_grid, 
                       cmap='viridis',
                       xticklabels=False,
                       yticklabels=False)
            
            plt.title(f'Patch {i+1} Feature Vector')
            
            # Save individual patch visualization
            save_path = os.path.join(save_dir, f'patch_{i+1:03d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nSaved {768} patch visualizations to {save_dir}")
        return save_dir
    
    def visualize_latent_layers(self, features, save_dir=None):
        """
        Visualize latent encoder layers for a random frame.
        Creates visualizations for each layer's feature maps:
        - Layer 1: 32 plots (256x256)
        - Layer 2: 64 plots (128x128)
        - Layer 3: 128 plots (64x64)
        """
        # Get a random frame index
        num_frames = self.features['latent_layer1'].shape[0]
        random_frame = np.random.randint(0, num_frames)
        
        # Create base save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_latent'
        
        # Create directories for each layer
        layer_dirs = {
            'layer1': os.path.join(save_dir, 'layer1_32ch_256x256'),
            'layer2': os.path.join(save_dir, 'layer2_64ch_128x128'),
            'layer3': os.path.join(save_dir, 'layer3_128ch_64x64')
        }
        for dir_path in layer_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Layer configurations
        layer_configs = [
            ('layer1', 'latent_layer1', 32, 256),
            ('layer2', 'latent_layer2', 64, 128),
            ('layer3', 'latent_layer3', 128, 64)
        ]
        
        # Process each layer
        for layer_name, feature_key, num_channels, target_size in layer_configs:
            # Get features for this layer
            layer_features = self.features[feature_key].cpu().numpy()
            layer_features = layer_features[random_frame]  # Get random frame
            
            print(f"\nProcessing {layer_name} with shape {layer_features.shape}")
            
            # Create visualization for each channel
            for channel in tqdm(range(num_channels), desc=f"Saving {layer_name} visualizations"):
                plt.figure(figsize=(8, 8))
                feature_map = layer_features[channel]
                
                # Resize if needed
                if feature_map.shape[0] != target_size:
                    feature_map = cv2.resize(feature_map, (target_size, target_size))
                
                # Normalize feature map
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
                # Create heatmap
                sns.heatmap(feature_map, 
                           cmap='viridis',
                           xticklabels=False,
                           yticklabels=False)
                
                plt.title(f'{layer_name} Channel {channel+1}')
                
                # Save visualization
                save_path = os.path.join(layer_dirs[layer_name], f'channel_{channel+1:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Saved {num_channels} visualizations for {layer_name}")
        
        print(f"\nSaved all latent layer visualizations to {save_dir}")
        return save_dir
    
    def _get_activation(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        # Clear existing hooks
        self.remove_hooks()
        
        # Register hooks for each layer of the latent encoder
        self.hooks.extend([
            self.model.latent_encoder.conv1.register_forward_hook(
                self._get_activation('latent_layer1')
            ),
            self.model.latent_encoder.conv2.register_forward_hook(
                self._get_activation('latent_layer2')
            ),
            self.model.latent_encoder.conv3.register_forward_hook(
                self._get_activation('latent_layer3')
            )
        ])
        
        # Register hook for raw patches before patch encoder
        def get_raw_patches(module, input):
            # input[0] contains the patches with shape (batch, segments, num_patches, frames_per_patch, channels, height, width)
            self.features['raw_patches'] = input[0].detach()
            return input  # Return input unchanged
        
        self.hooks.append(
            self.model.patch_encoder.register_forward_pre_hook(get_raw_patches)
        )
        
        # Register hooks for patch encoder layers
        self.hooks.extend([
            self.model.patch_encoder.conv1.register_forward_hook(
                self._get_activation('patch_layer1')
            ),
            self.model.patch_encoder.conv2.register_forward_hook(
                self._get_activation('patch_layer2')
            ),
            self.model.patch_encoder.conv3.register_forward_hook(
                self._get_activation('patch_layer3')
            )
        ])
        
        # Register hooks for transformer layers
        for i in range(12):  # 12 transformer layers
            self.hooks.append(
                self.model.classifier.transformer_encoder.layers[i].register_forward_hook(
                    self._get_activation(f'transformer_layer_{i}')
                )
            )
        
        # Register hooks for other features
        self.hooks.extend([
            self.model.patch_encoder.register_forward_hook(
                self._get_activation('patch_features')
            ),
            self.model.classifier.transformer_encoder.register_forward_hook(
                self._get_activation('transformer_features')
            )
        ])
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_patch_encoder_layers(self, features, save_dir=None):
        """
        Visualize patch encoder layers for a random patch.
        Creates visualizations for each layer's feature maps:
        - Layer 1: 192 channels
        - Layer 2: 384 channels
        - Layer 3: 768 channels
        """
        print("\nExamining Patch Encoder Layer Shapes:")
        print("=====================================")
        
        # Print shapes of all patch encoder features
        for key in ['patch_layer1', 'patch_layer2', 'patch_layer3']:
            if key in self.features:
                feature = self.features[key]
                print(f"\n{key}:")
                print(f"  Full tensor shape: {feature.shape}")
                if len(feature.shape) >= 4:
                    print(f"  Batch size: {feature.shape[0]}")
                    print(f"  Number of patches: {feature.shape[1]}")
                    print(f"  Spatial dimensions: {feature.shape[2:] if len(feature.shape) > 3 else 'N/A'}")
                    print(f"  Number of channels: {feature.shape[-1] if len(feature.shape) > 3 else feature.shape[-1]}")
        
        # Create base save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_patch_encoder'
        
        # Create directories for each layer
        layer_dirs = {
            'layer1': os.path.join(save_dir, 'patch_layer1_192ch'),
            'layer2': os.path.join(save_dir, 'patch_layer2_384ch'),
            'layer3': os.path.join(save_dir, 'patch_layer3_768ch')
        }
        for dir_path in layer_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Layer configurations
        layer_configs = [
            ('layer1', 'patch_layer1', 192),
            ('layer2', 'patch_layer2', 384),
            ('layer3', 'patch_layer3', 768)
        ]
        
        # Select a random patch to visualize
        batch_size = self.features['patch_layer1'].shape[0]
        num_patches = self.features['patch_layer1'].shape[1]
        random_batch = np.random.randint(0, batch_size)
        random_patch = np.random.randint(0, num_patches)
        
        print(f"\nVisualizing patch encoder layers for batch {random_batch}, patch {random_patch}")
        
        # Process each layer
        for layer_name, feature_key, num_channels in layer_configs:
            # Get features for this layer
            layer_features = self.features[feature_key].cpu().numpy()
            
            # Print debug information
            print(f"\nProcessing {layer_name}")
            print(f"Layer features shape: {layer_features.shape}")
            
            # Extract features for the selected patch
            patch_features = layer_features[random_batch, random_patch]
            print(f"Patch features shape: {patch_features.shape}")
            
            # Create visualization for each channel
            for channel in tqdm(range(num_channels), desc=f"Saving {layer_name} visualizations"):
                fig = plt.figure(figsize=(15, 8))
                
                # Debug print for feature shapes
                print(f"\nDebug - {layer_name} Channel {channel+1}:")
                print(f"Patch features shape before extraction: {patch_features.shape}")
                
                # Get the feature map for this channel
                if len(patch_features.shape) == 3:
                    print(f"Using 3D tensor extraction for channel {channel}")
                    feature_map = patch_features[:, :, channel]
                    print(f"Extracted feature map shape: {feature_map.shape}")
                    print(f"Channel {channel} value range: [{feature_map.min():.3f}, {feature_map.max():.3f}]")
                elif len(patch_features.shape) == 2:
                    print(f"Using 2D tensor extraction")
                    # For 2D tensors (4x4), we only visualize once since it's a single feature map
                    if channel > 0:  # Skip after first visualization
                        continue
                    feature_map = patch_features
                    print(f"Extracted feature map shape: {feature_map.shape}")
                    print(f"Value range: [{feature_map.min():.3f}, {feature_map.max():.3f}]")
                else:
                    print(f"Using 1D tensor reshaping for channel {channel}")
                    if channel < len(patch_features):
                        feature_map = patch_features[channel]
                        print(f"Single channel value: {feature_map}")
                    else:
                        print(f"Warning: Channel {channel} exceeds feature dimension {len(patch_features)}")
                        continue
                
                # Ensure feature_map is 2D (but don't reshape if it's already 4x4)
                if len(feature_map.shape) == 1:
                    print("Reshaping 1D feature map to 2D grid")
                    size = int(np.ceil(np.sqrt(len(feature_map))))
                    feature_map = feature_map.reshape(size, size)
                elif len(feature_map.shape) == 3:
                    print("Averaging 3D feature map across last dimension")
                    feature_map = feature_map.mean(axis=-1)
                
                print(f"Final feature map shape: {feature_map.shape}")
                print(f"Final value range: [{feature_map.min():.3f}, {feature_map.max():.3f}]")
                
                # Create two subplots
                plt.subplot(1, 2, 1)
                # Plot raw values without normalization
                sns.heatmap(feature_map, 
                           cmap='viridis',
                           xticklabels=False,
                           yticklabels=False,
                           annot=True,  # Show actual values
                           fmt='.2f',   # Format to 2 decimal places
                           annot_kws={'size': 6})  # Adjust annotation size
                plt.title(f'{layer_name} Channel {channel+1} (Raw Values)')
                
                # Add statistics
                stats_text = f"Mean: {feature_map.mean():.3f}\n"
                stats_text += f"Std: {feature_map.std():.3f}\n"
                stats_text += f"Min: {feature_map.min():.3f}\n"
                stats_text += f"Max: {feature_map.max():.3f}"
                
                plt.subplot(1, 2, 2)
                # Plot normalized values
                normalized_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                sns.heatmap(normalized_map, 
                           cmap='viridis',
                           xticklabels=False,
                           yticklabels=False)
                plt.title(f'{layer_name} Channel {channel+1} (Normalized)')
                
                # Add stats text box
                plt.text(1.5, 0.5, stats_text, transform=plt.gca().transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                
                # Save visualization
                save_path = os.path.join(layer_dirs[layer_name], f'channel_{channel+1:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Print separator for readability
                print("-" * 50)
            
            print(f"Saved {num_channels} visualizations for {layer_name}")
        
        print(f"\nSaved all patch encoder layer visualizations to {save_dir}")
        return save_dir
    
    def visualize_patch_transformation(self, features, save_dir=None):
        """
        Visualize how a single patch is transformed through the classifier's transformer layers.
        Creates visualizations showing the evolution of the patch's values through each layer.
        """
        # Create base save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_patch_transformation'
        os.makedirs(save_dir, exist_ok=True)
        
        # Select a random patch
        batch_size = self.features['transformer_layer_0'].shape[0]
        num_patches = self.features['transformer_layer_0'].shape[1]
        random_batch = np.random.randint(0, batch_size)
        random_patch = np.random.randint(0, num_patches)
        
        print(f"\nVisualizing transformation of patch {random_patch} through transformer layers")
        
        # Get initial patch values (before transformer)
        initial_values = self.features['patch_features'][random_batch, random_patch].cpu().numpy()
        
        # Create visualization for initial patch values
        plt.figure(figsize=(15, 8))
        
        # Plot initial values as 24x32 grid
        plt.subplot(1, 2, 1)
        feature_grid = initial_values.reshape(24, 32)  # Reshape to match patch grid
        sns.heatmap(feature_grid, cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title('Initial Patch Values (24x32 Grid)')
        
        # Plot value distribution
        plt.subplot(1, 2, 2)
        plt.hist(initial_values, bins=50)
        plt.title('Initial Value Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'initial_patch.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Process each transformer layer
        for i in range(12):
            layer_values = self.features[f'transformer_layer_{i}'][random_batch, random_patch].cpu().numpy()
            
            plt.figure(figsize=(15, 8))
            
            # Plot transformed values as 24x32 grid
            plt.subplot(1, 2, 1)
            feature_grid = layer_values.reshape(24, 32)  # Reshape to match patch grid
            sns.heatmap(feature_grid, cmap='viridis', xticklabels=False, yticklabels=False)
            plt.title(f'Layer {i+1} Transformed Values (24x32 Grid)')
            
            # Plot value distribution
            plt.subplot(1, 2, 2)
            plt.hist(layer_values, bins=50)
            plt.title(f'Layer {i+1} Value Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'layer_{i+1:02d}_transformation.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print statistics
            print(f"\nLayer {i+1} Statistics:")
            print(f"Mean: {layer_values.mean():.4f}")
            print(f"Std: {layer_values.std():.4f}")
            print(f"Min: {layer_values.min():.4f}")
            print(f"Max: {layer_values.max():.4f}")
        
        print(f"\nSaved patch transformation visualizations to {save_dir}")
        return save_dir
    
    def visualize_all_patch_relationships(self, features, save_dir=None):
        """
        Visualize how all patches relate to each other through the transformer layers.
        Creates 12 heatmaps (768x768) showing patch relationships at each layer.
        """
        # Create base save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_patch_relationships'
        os.makedirs(save_dir, exist_ok=True)
        
        # Get initial patch values (before transformer)
        initial_values = self.features['patch_features'][0].cpu().numpy()  # Shape: (768, 768)
        
        # Create correlation matrix for initial patches
        plt.figure(figsize=(20, 20))
        correlation_matrix = np.corrcoef(initial_values)
        sns.heatmap(correlation_matrix, 
                   cmap='RdBu_r',  # Red-Blue diverging colormap
                   center=0,       # Center the colormap at 0
                   vmin=-1,        # Set minimum value to -1
                   vmax=1,         # Set maximum value to 1
                   xticklabels=False,
                   yticklabels=False)
        plt.title('Initial Patch Relationships')
        plt.savefig(os.path.join(save_dir, 'initial_relationships.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Process each transformer layer
        for i in range(12):
            # Get layer values for all patches
            layer_values = self.features[f'transformer_layer_{i}'][0].cpu().numpy()  # Shape: (768, 768)
            
            # Create correlation matrix
            plt.figure(figsize=(20, 20))
            correlation_matrix = np.corrcoef(layer_values)
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       cmap='RdBu_r',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       xticklabels=False,
                       yticklabels=False)
            
            plt.title(f'Layer {i+1} Patch Relationships')
            
            # Add grid lines to show 24x32 patch structure
            for x in range(0, 768, 32):
                plt.axvline(x=x, color='black', linewidth=0.5, alpha=0.3)
                plt.axhline(y=x, color='black', linewidth=0.5, alpha=0.3)
            
            # Save visualization
            plt.savefig(os.path.join(save_dir, f'layer_{i+1:02d}_relationships.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print statistics about relationships
            print(f"\nLayer {i+1} Relationship Statistics:")
            print(f"Mean correlation: {correlation_matrix.mean():.4f}")
            print(f"Std correlation: {correlation_matrix.std():.4f}")
            print(f"Min correlation: {correlation_matrix.min():.4f}")
            print(f"Max correlation: {correlation_matrix.max():.4f}")
        
        print(f"\nSaved patch relationship visualizations to {save_dir}")
        return save_dir
    
    def visualize_raw_patches(self, features, save_dir=None):
        """
        Visualize the raw patches before they go through the patch encoder.
        Shape: (batch, frames, channels, height, width) = (1, 24, 128, 64, 64)
        We'll select two random frames and show one 8x8 segment across all channels
        """
        if 'raw_patches' not in self.features:
            print("No raw patches found in features")
            return
            
        # Create save directory
        if save_dir is None:
            video_name = os.path.basename(features.get('metadata', {}).get('video_path', 'unknown'))
            base_name = os.path.splitext(video_name)[0]
            save_dir = f'feature_visualizations/{base_name}_raw_patches'
        os.makedirs(save_dir, exist_ok=True)
        
        # Get raw patches
        patches = self.features['raw_patches'].cpu().numpy()
        print(f"\nRaw patches shape: {patches.shape}")
        
        # Select two consecutive frames (first frame must be even)
        frame1_idx = 2 * np.random.randint(0, patches.shape[1] // 2 - 1)  # Ensure even index and room for next frame
        frame2_idx = frame1_idx + 1  # Get the next consecutive frame
        print(f"Selected consecutive frames: {frame1_idx} (even) and {frame2_idx} (odd)")
        
        # Select random 8x8 segment position
        start_h = np.random.randint(0, patches.shape[3] - 8)
        start_w = np.random.randint(0, patches.shape[4] - 8)
        print(f"Selected segment position: ({start_h}:{start_h+8}, {start_w}:{start_w+8})")
        
        # Extract 8x8 segments for both frames across all channels
        segment1 = patches[0, frame1_idx, :, start_h:start_h+8, start_w:start_w+8]  # Shape: (128, 8, 8)
        segment2 = patches[0, frame2_idx, :, start_h:start_h+8, start_w:start_w+8]  # Shape: (128, 8, 8)
        
        print(f"Segment shapes: {segment1.shape}")
        
        # Create visualizations for each channel
        for channel in tqdm(range(128), desc="Visualizing channel segments"):
            fig = plt.figure(figsize=(15, 8))
            
            # Plot first frame's segment
            plt.subplot(1, 2, 1)
            sns.heatmap(segment1[channel], 
                       cmap='viridis',
                       xticklabels=False,
                       yticklabels=False,
                       annot=True,  # Show actual values
                       fmt='.2f')   # Format to 2 decimal places
            plt.title(f'Frame {frame1_idx} Channel {channel} (8x8 Segment)')
            
            # Plot second frame's segment
            plt.subplot(1, 2, 2)
            sns.heatmap(segment2[channel], 
                       cmap='viridis',
                       xticklabels=False,
                       yticklabels=False,
                       annot=True,  # Show actual values
                       fmt='.2f')   # Format to 2 decimal places
            plt.title(f'Frame {frame2_idx} Channel {channel} (8x8 Segment)')
            
            # Add statistics
            stats_text1 = f"Frame {frame1_idx} Stats:\n"
            stats_text1 += f"Mean: {segment1[channel].mean():.3f}\n"
            stats_text1 += f"Std: {segment1[channel].std():.3f}\n"
            stats_text1 += f"Min: {segment1[channel].min():.3f}\n"
            stats_text1 += f"Max: {segment1[channel].max():.3f}"
            
            stats_text2 = f"Frame {frame2_idx} Stats:\n"
            stats_text2 += f"Mean: {segment2[channel].mean():.3f}\n"
            stats_text2 += f"Std: {segment2[channel].std():.3f}\n"
            stats_text2 += f"Min: {segment2[channel].min():.3f}\n"
            stats_text2 += f"Max: {segment2[channel].max():.3f}"
            
            plt.text(-0.6, -0.3, stats_text1, transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
            plt.text(1.1, -0.3, stats_text2, transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'channel_{channel:03d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nSaved channel segment visualizations to {save_dir}")
        return save_dir
    
    def extract_features(self, video_path):
        """Extract features from a single video."""
        try:
            # Load and preprocess video
            frames_tensor, frames, label, _ = load_video(video_path)
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            
            # Register hooks
            self._register_hooks()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Collect features
            features = {
                'latent_features': self.features['latent_layer3'].cpu().numpy(),
                'raw_patches': self.features['raw_patches'].cpu().numpy(),
                'patch_features': self.features['patch_features'].cpu().numpy(),
                'transformer_features': self.features['transformer_features'].cpu().numpy(),
                'final_logits': outputs.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'prediction': prediction,
                'confidence': confidence,
                'true_label': label.item(),
                'metadata': {'video_path': video_path}
            }
            
            # Generate visualizations only if enabled
            if self.enable_visualizations:
                #self.visualize_patch_vectors(features)
                #self.visualize_latent_layers(features)
                self.visualize_raw_patches(features)  # Add raw patch visualization
                self.visualize_patch_encoder_layers(features)
                #self.visualize_patch_transformation(features)
                #self.visualize_all_patch_relationships(features)
            
            return features
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None
        finally:
            self.remove_hooks()
            
    def batch_extract_features(self, video_paths, output_dir='features'):
        """Extract features from a batch of videos and save them."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for video_path in tqdm(video_paths, desc="Extracting features"):
            features = self.extract_features(video_path)
            if features is not None:
                # Save features
                video_name = os.path.basename(video_path)
                save_path = os.path.join(output_dir, f"{video_name}_features.npz")
                np.savez(
                    save_path,
                    latent_features=features['latent_features'],
                    raw_patches=features['raw_patches'],
                    patch_features=features['patch_features'],
                    transformer_features=features['transformer_features'],
                    final_logits=features['final_logits'],
                    probabilities=features['probabilities'],
                    metadata={
                        'prediction': features['prediction'],
                        'confidence': features['confidence'],
                        'true_label': features['true_label'],
                        'video_path': video_path
                    }
                )
                results.append({
                    'video_path': video_path,
                    'prediction': features['prediction'],
                    'confidence': features['confidence'],
                    'true_label': features['true_label']
                })
        
        return results

def main():
    # Example usage
    model_path = 'model/full_classifier_1_85.pt'
    extractor = FeatureExtractor(model_path)
    
    # Test with a single video to examine layer shapes
    video_path = "F:/Gen-Video/dataset/ai_dynamiccrafter_DynamicCrafter_43162.mp4"
    print(f"\nExamining feature shapes for video: {os.path.basename(video_path)}")
    features = extractor.extract_features(video_path)
    
if __name__ == "__main__":
    main() 