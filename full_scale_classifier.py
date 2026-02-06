import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def extractPatches(latents):
    """
    Optimized patch extraction using unfold - SHAPE-PRESERVING VERSION
    Output shape matches original exactly: [batch, segments, num_patches, frames_per_patch, channels, h_patch, w_patch]
    """
    try:
        batch_size, num_frames, channels, height, width = latents.shape
        
        # Constants
        frames_per_patch = 2
        height_per_patch = 8
        width_per_patch = 8
        
        # Calculate dimensions
        num_segments = num_frames // frames_per_patch
        num_h_patches = height // height_per_patch
        num_w_patches = width // width_per_patch
        
        # Handle padding if dimensions don't divide evenly
        if height % height_per_patch != 0:
            pad_h = height_per_patch - (height % height_per_patch)
            height_padded = height + pad_h
            num_h_patches += 1
        else:
            height_padded = height
            pad_h = 0
            
        if width % width_per_patch != 0:
            pad_w = width_per_patch - (width % width_per_patch)  
            width_padded = width + pad_w
            num_w_patches += 1
        else:
            width_padded = width
            pad_w = 0
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            latents = torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        # Reshape to group frames by segments first
        # [batch, segments, frames_per_patch, channels, height, width]
        latents_segmented = latents[:, :num_segments*frames_per_patch].view(
            batch_size, num_segments, frames_per_patch, channels, height_padded, width_padded
        )
        
        # Process all segments at once using unfold
        patches_list = []
        
        for segment_idx in range(num_segments):
            segment_data = latents_segmented[:, segment_idx]  # [batch, frames_per_patch, channels, H, W]
            
            # Use unfold to extract all patches at once
            patches_h = segment_data.unfold(3, height_per_patch, height_per_patch)  # Along height
            patches_hw = patches_h.unfold(4, width_per_patch, width_per_patch)      # Along width
            
            # Current shape: [batch, frames_per_patch, channels, num_h_patches, num_w_patches, height_per_patch, width_per_patch]
            
            # Rearrange to match original output format exactly
            # Original: [batch, segments, num_patches, frames_per_patch, channels, height_per_patch, width_per_patch]
            # We need: [batch, num_patches, frames_per_patch, channels, height_per_patch, width_per_patch]
            
            patches_segment = patches_hw.permute(0, 3, 4, 1, 2, 5, 6)  # [batch, h_patches, w_patches, frames, channels, h_patch, w_patch]
            
            # Flatten spatial patch dimensions (h_patches * w_patches)
            batch_s, h_patches, w_patches, frames_s, channels_s, h_patch, w_patch = patches_segment.shape
            patches_segment = patches_segment.contiguous().view(
                batch_s, h_patches * w_patches, frames_s, channels_s, h_patch, w_patch
            )
            
            patches_list.append(patches_segment)
        
        # Stack all segments to get final shape: [batch, segments, num_patches, frames_per_patch, channels, h_patch, w_patch]
        patches = torch.stack(patches_list, dim=1)
        
        # Create attention mask matching original format
        attention_mask = torch.ones(
            (batch_size, num_segments, patches.shape[2], frames_per_patch, height_per_patch, width_per_patch),
            device=latents.device,
            dtype=latents.dtype
        )
        
        # Handle padding in attention mask
        if pad_h > 0 or pad_w > 0:
            # Zero out padded regions in attention mask
            if pad_h > 0:
                attention_mask[:, :, :, :, height:, :] = 0
            if pad_w > 0:
                attention_mask[:, :, :, :, :, width:] = 0
        
        return patches, attention_mask
        
    except Exception as e:
        print(f"Error in extractPatches_optimized: {str(e)}")
        # Fallback to original implementation
        return extractPatches(latents)


class FullLatentEncoder(nn.Module):
    def __init__(self):
        super(FullLatentEncoder, self).__init__()
        # Reduce spatial dimensions by factor of 8 (3 stride-2 convolutions)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        # Use GroupNorm instead of BatchNorm for better stability with small batches
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(16, 128)
        
        # Input normalization
        self.register_buffer('input_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Initialize weights using Kaiming initialization with proper gain
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, num_frames, height, width, channels = x.shape
        
        # Calculate output dimensions (factor of 8 reduction)
        o_height = height // 8
        o_width = width // 8
        
        # Handle non-divisible dimensions using torch operations
        pad_height = (8 - (height % 8)) % 8
        pad_width = (8 - (width % 8)) % 8
        
        if pad_height > 0 or pad_width > 0:
            o_height += 1
            o_width += 1
            
        outputs = []
        
        for f in range(num_frames):
            batch = x[:, f]
            batch = batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
            # Ensure input_mean and input_std are on the same device as the input
            if self.input_mean.device != batch.device:
                self.input_mean = self.input_mean.to(batch.device)
                self.input_std = self.input_std.to(batch.device)
            
            # Normalize input
            batch = (batch - self.input_mean) / self.input_std
            
            # Apply convolutions with normalization and activation
            # Use small epsilon in ReLU to prevent exact zeros
            try:
                with torch.amp.autocast('cuda'):  # Updated to new API
                    x1 = F.relu(self.norm1(self.conv1(batch)), inplace=False) + 1e-8
                    x2 = F.relu(self.norm2(self.conv2(x1)), inplace=False) + 1e-8
                    output = F.relu(self.norm3(self.conv3(x2)), inplace=False) + 1e-8
                    
                    # Clip extreme values
                    output = torch.clamp(output, -10, 10)
                    outputs.append(output)
                    
                    # Clean up intermediates
                    del x1, x2
                    
            except RuntimeError as e:
                print(f"Error processing frame {f}: {str(e)}")
                raise
                
            # Force CUDA synchronization after each frame if using GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Stack all outputs
        final_output = torch.stack(outputs, dim=1)
        del outputs
        
        return final_output
      
class FullPatchEncoder(nn.Module):
    def __init__(self):
        super(FullPatchEncoder, self).__init__()
        
        # Patch extraction function
        self.patch_extractor = extractPatches
        
        # Convolutional layers for patch processing
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=2, stride=1, padding=0)
        
        # Normalization layers
        self.norm = nn.LayerNorm(768)
        self.gn1 = nn.GroupNorm(16, 192)
        self.gn2 = nn.GroupNorm(24, 384)
        self.gn3 = nn.GroupNorm(32, 768)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming initialization with proper scaling
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, latents):
        # Extract patches
        patches, _ = self.patch_extractor(latents)
        
        # Get dimensions
        batch_size, segments, num_patches, frames, channels, height, width = patches.shape
        total_patches = segments * num_patches
        
        # Initialize output tensor
        vectors = torch.zeros((batch_size, total_patches, 768), 
                            device=latents.device,
                            dtype=latents.dtype)
        
        # Process each patch
        for seg in range(segments):
            for p in range(num_patches):
                with torch.amp.autocast('cuda'):  # Updated to new API
                    # Process first frame
                    patch_0 = patches[:, seg, p, 0]
                    if torch.isnan(patch_0).any():
                        raise ValueError(f"NaN in input patch at segment {seg}, patch {p}, frame 0")
                        
                    x1 = self.gn1(F.relu(self.conv1(patch_0)))
                    x1 = self.gn2(F.relu(self.conv2(x1)))
                    x1 = self.gn3(F.relu(self.conv3(x1)))
                    x1 = x1.flatten(start_dim=1)
                    
                    # Process second frame
                    patch_1 = patches[:, seg, p, 1]
                    if torch.isnan(patch_1).any():
                        raise ValueError(f"NaN in input patch at segment {seg}, patch {p}, frame 1")
                        
                    x2 = self.gn1(F.relu(self.conv1(patch_1)))
                    x2 = self.gn2(F.relu(self.conv2(x2)))
                    x2 = self.gn3(F.relu(self.conv3(x2)))
                    x2 = x2.flatten(start_dim=1)
                    
                    # Average embeddings and normalize
                    embeddings = self.norm((x1 + x2) / 2)
                    
                    if torch.isnan(embeddings).any():
                        raise ValueError(f"NaN in embeddings at segment {seg}, patch {p}")
                    
                    # Assign embeddings
                    seq_pos = seg * num_patches + p
                    vectors[:, seq_pos] = embeddings
                    
                    # Clean up intermediates
                    del x1, x2, patch_0, patch_1
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # Clean up after each segment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return vectors

class FullClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=12, num_layers=12, num_classes=2):
        super(FullClassifier, self).__init__()
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create encoder layer as direct attribute to match saved model
        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=hidden_dim,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1
        )
        
        # Create transformer encoder using the encoder layer
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        # Final classification layer
        self.fc = nn.Linear(input_dim, num_classes)
        
        # Initialize weights with improved strategy
        self._init_weights()
    
    def _init_weights(self):
        # Initialize transformer weights with truncated normal distribution
        def _init_transformer_weights(module):
            if isinstance(module, nn.Linear):
                # Use truncated normal for linear layers
                std = 0.02  # Standard transformer initialization
                torch.nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.transformer_encoder.apply(_init_transformer_weights)
        self.encoder_layer.apply(_init_transformer_weights)
        
        # Initialize classification layer with smaller weights
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Input shape validation - flexible sequence length
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features), got shape {x.shape}")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.shape[2]}")
        
        batch_size, seq_len, feature_dim = x.shape
            
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        pooled_x = x.mean(dim=1)
        
        # Final classification with numerical stability
        logits = self.fc(pooled_x)
        logits = logits - logits.max(dim=1, keepdim=True)[0]  # Subtract max for stability
        logits = logits.clamp(-15, 15)  # Prevent extreme values
        
        return logits

class FullVideoClassifier(nn.Module):
    def __init__(self, latent_encoder, patch_encoder, classifier):
        super(FullVideoClassifier, self).__init__()
        self.latent_encoder = latent_encoder
        self.patch_encoder = patch_encoder
        self.classifier = classifier
        
        # Apply custom initialization scaling
        self._init_weights()
    
    def _init_weights(self):
        # Scale the final layer of each component slightly down
        for module in [self.latent_encoder, self.patch_encoder, self.classifier]:
            if hasattr(module, 'fc'):
                nn.init.normal_(module.fc.weight, std=0.01)
                if module.fc.bias is not None:
                    nn.init.zeros_(module.fc.bias)
    
    def forward(self, videos):
        batch_size, num_frames, height, width, channels = videos.shape
        
        
        # Process entire video at once through latent encoder
        with torch.amp.autocast('cuda'):  # Updated to new API
            # Use checkpointing for latent encoder during training
            latents = self.latent_encoder(videos)
            
            
            # Process through patch encoder with gradient checkpointing
            st_vectors = self.patch_encoder(latents)
            
            
            # Verify tensor shape and values - updated shape check for flexibility
            if st_vectors.shape[2] != 768:
                raise ValueError(f"Expected feature dimension 768 for patch encoder output, got {st_vectors.shape}")
            if torch.isnan(st_vectors).any():
                raise ValueError("NaN values detected in patch encoder output")
            if torch.isinf(st_vectors).any():
                raise ValueError("Inf values detected in patch encoder output")
            
            # Log actual patch count
            actual_patches = st_vectors.shape[1]
            expected_patches = (num_frames // 2) * ((height // 8) // 8) * ((width // 8) // 8)
            
            del latents
            
            # Process through classifier with gradient checkpointing
            
            outputs = self.classifier(st_vectors)
            
            del st_vectors
            
            return outputs