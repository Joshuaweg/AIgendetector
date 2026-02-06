import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def extractPatches(latents):
    # Input shape: (num_frames=24, height=120, width=120)
    batch_size = latents.shape[0]
    num_frames = latents.shape[1]
    height = latents.shape[2]
    width = latents.shape[3]
    
    # Constants
    frames_per_patch = 4
    height_per_patch = 24
    width_per_patch = 24
    
    # Calculate number of patches in height and width
    num_h_patches = height // height_per_patch  # 120//24 = 5
    num_w_patches = width // width_per_patch    # 120//24 = 5
    num_segments = num_frames // frames_per_patch  # 24//4 = 6
    
    # Initialize output tensor with correct shape
    # (batch_size=1,num_segments=6, spatial_patches=25, frames_per_patch=4, patch_height=24, patch_width=24)
    patches = torch.zeros((batch_size, num_segments, num_h_patches * num_w_patches, frames_per_patch,
                          height_per_patch, width_per_patch), device=latents.device)
    
    # Extract patches
    for batch in range(batch_size):
        for segment in range(num_segments):
            patch_idx = 0
            for h in range(0, height, height_per_patch):
                for w in range(0, width, width_per_patch):
                    for f_idx in range(frames_per_patch):
                        frame_idx = segment * frames_per_patch + f_idx
                        patch = latents[batch, frame_idx, h:h+height_per_patch, w:w+width_per_patch]
                        patches[batch, segment, patch_idx, f_idx] = patch
                    patch_idx += 1
    
    return patches
class LatentEncoder(nn.Module):
    def __init__(self):
        super(LatentEncoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((15, 15))
        
        # Fully connected layers
        self.fc1 = nn.Linear(8 * 15 * 15, 14400)
        
        # Add batch normalization layers
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(8)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: [batch_size, num_frames, height, width, channels]
        batch_size, num_frames, height, width, channels = x.shape
        latents = torch.zeros(batch_size, num_frames, 120, 120, device=x.device)
        
        # Process each frame in the batch separately
        for b in range(batch_size):
            for f in range(num_frames):
                # Permute the frame to get channels first
                frame = x[b, f].permute(2, 0, 1)  # [channels, height, width]
                frame = frame.unsqueeze(0)  # Add batch dimension [1, channels, height, width]
                
                # Convolutional layers with batch norm and ReLU
                cl1 = F.leaky_relu(self.bn1(self.conv1(frame)))
                cl2 = F.leaky_relu(self.bn2(self.conv2(cl1)))
                cl3 = F.leaky_relu(self.bn3(self.conv3(cl2)))
                cl4 = F.leaky_relu(self.bn4(self.conv4(cl3)))
                
                # Pooling and flatten
                pooled = self.pool(cl4)
                flat = pooled.view(-1, 8 * 15 * 15)
                
                # Fully connected layers with dropout
                fc1_out = F.relu(self.fc1(flat))
                fc1_out = self.dropout(fc1_out)
                output = fc1_out
                
                # Reshape output to 120x120 and store in latents tensor
                latents[b, f] = output.view(120, 120)
        
        return latents
class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        
        # Convolutional layers with batch norm
        self.patch_extractor = extractPatches
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        
        # Layer normalization for the final output
        self.layer_norm = nn.LayerNorm(8 * 24 * 24)
        
        # Fully connected layer with proper initialization
        self.fc = nn.Linear(8 * 24 * 24, 100)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def check_nan(self, x, name):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf detected in {name}")
            print(f"Range: [{x.min()}, {x.max()}]")
            return True
        return False
    
    def forward(self, latents):
        if self.check_nan(latents, "input latents"):
            raise ValueError("NaN in input")
            
        x = self.patch_extractor(latents)
        batch_size, num_segments, patches, frames, height, width = x.shape
        vectors = torch.zeros(batch_size, num_segments*patches, 100, device=latents.device)
        
        for b in range(batch_size):
            for segment in range(num_segments):
                for patch in range(patches):
                    # Get patch
                    pat = x[b, segment, patch]
                    pat = pat.unsqueeze(0)
                    
                    # Convolutional layers with batch norm, ReLU, and dropout
                    cl1 = self.dropout(F.leaky_relu(self.bn1(self.conv1(pat))))
                    if self.check_nan(cl1, "conv1"): continue
                    
                    cl2 = self.dropout(F.leaky_relu(self.bn2(self.conv2(cl1))))
                    if self.check_nan(cl2, "conv2"): continue
                    
                    cl3 = self.dropout(F.leaky_relu(self.bn3(self.conv3(cl2))))
                    if self.check_nan(cl3, "conv3"): continue
                    
                    # Flatten with proper reshaping
                    flat = cl3.reshape(1, -1)
                    
                    # Apply layer normalization
                    flat = self.layer_norm(flat)
                    
                    # Apply FC layer with gradient clipping
                    output = self.fc(flat)
                    output = torch.clamp(output, -100, 100)  # Prevent extreme values
                    
                    vectors[b, segment*patches + patch] = output.squeeze(0)
                    
                    # Clean up
        
        # Final check before returning
        if self.check_nan(vectors, "output vectors"):
            # Return zeros instead of NaN
            return torch.zeros_like(vectors)
        return vectors


class Classifier(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=4, num_layers=1, num_classes=2):
        super(Classifier, self).__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Create encoder layer with gradient tracking
        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=hidden_dim,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        
        # Create transformer encoder
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.fc = nn.Linear(input_dim, num_classes)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        # Initialize transformer weights with more conservative scaling
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)
        
        # Initialize FC layer with more conservative scaling
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)
        
        # Explicitly enable gradients
        for p in self.parameters():
            p.requires_grad = True
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Add small epsilon to prevent exact zeros
        x = x + 1e-8
        
        # Input normalization
        x = self.input_norm(x)
        
        # Ensure input requires gradients
        if not x.requires_grad:
            x.requires_grad = True
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Clip extreme values
        encoded = torch.clamp(encoded, -10, 10)
        
        # Mean pooling with stable computation
        # Use sum and divide instead of mean for better numerical stability
        pooled = encoded.sum(dim=1) / encoded.size(1)
        
        # Additional normalization
        pooled = F.normalize(pooled, p=2, dim=-1)
        
        # Final classification with normalization
        output = self.fc(pooled)
        output = self.output_norm(output)
        
        return output
class VideoClassifier(nn.Module):
    def __init__(self, latent_encoder, patch_encoder, classifier):
        super(VideoClassifier, self).__init__()
        self.latent_encoder = latent_encoder
        self.patch_encoder = patch_encoder
        self.classifier = classifier
        
    def forward(self, videos):
        # Add value checking at each step
        latents = self.latent_encoder(videos)
        if torch.isnan(latents).any():
            print("NaN detected in latents")
            return torch.zeros(videos.size(0), 2, device=videos.device)
            
        st_vectors = self.patch_encoder(latents)
        if torch.isnan(st_vectors).any():
            print("NaN detected in st_vectors")
            return torch.zeros(videos.size(0), 2, device=videos.device)
            
        outputs = self.classifier(st_vectors)
        if torch.isnan(outputs).any():
            print("NaN detected in outputs")
            return torch.zeros(videos.size(0), 2, device=videos.device)
            
        return outputs