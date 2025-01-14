import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def extractPatches(latents):
    # Input shape: (batch_size, num_frames, channels, height, width)
    batch_size = latents.shape[0]
    num_frames = latents.shape[1]
    channels = latents.shape[2]
    height = latents.shape[3]
    width = latents.shape[4]
    
    # Constants
    frames_per_patch = 2
    height_per_patch = 8
    width_per_patch = 8
    
    # Calculate number of patches using torch operations
    num_h_patches = (height + height_per_patch - 1) // height_per_patch
    num_w_patches = (width + width_per_patch - 1) // width_per_patch
    num_segments = num_frames // frames_per_patch
    
    # Initialize output tensor with correct shape
    patches = torch.zeros((batch_size, num_segments, num_h_patches * num_w_patches, frames_per_patch,
                          channels, height_per_patch, width_per_patch), device=latents.device)
    attention_mask = torch.zeros((batch_size, num_segments, num_h_patches * num_w_patches, 
                            frames_per_patch, height_per_patch, width_per_patch),
                           device=latents.device)
    
    # Extract patches using torch operations
    for batch in range(batch_size):
        for segment in range(num_segments):
            patch_idx = 0
            for h in range(0, height, height_per_patch):
                curr_h = torch.minimum(torch.tensor(height_per_patch), torch.tensor(height-h))
                for w in range(0, width, width_per_patch):
                    curr_w = torch.minimum(torch.tensor(width_per_patch), torch.tensor(width-w))
                    for f_idx in range(frames_per_patch):
                        frame_idx = segment * frames_per_patch + f_idx
                        # Get patch
                        patch = latents[batch, frame_idx, :, h:h+height_per_patch, w:w+width_per_patch]
                        if curr_h < height_per_patch or curr_w < width_per_patch:
                            temp_patch = torch.zeros((channels, height_per_patch, width_per_patch),
                                                   device=latents.device)
                            temp_patch[:, :curr_h, :curr_w] = patch[:, :curr_h, :curr_w]
                            patch = temp_patch
                        patches[batch, segment, patch_idx, f_idx] = patch
                        attention_mask[batch, segment, patch_idx, f_idx, :curr_h, :curr_w] = 1
                    patch_idx += 1
    
    return patches, attention_mask


class FullLatentEncoder(nn.Module):
    def __init__(self):
        super(FullLatentEncoder, self).__init__()
        # Reduce spatial dimensions by factor of 8 (3 stride-2 convolutions)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        # Configure batch norm for small batches
        self.norm1 = nn.BatchNorm2d(32, momentum=0.1, eps=1e-5, track_running_stats=True)
        self.norm2 = nn.BatchNorm2d(64, momentum=0.1, eps=1e-5, track_running_stats=True)
        self.norm3 = nn.BatchNorm2d(128, momentum=0.1, eps=1e-5, track_running_stats=True)
        
        # Input normalization
        self.register_buffer('input_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
            
        outputs = torch.zeros((batch_size, num_frames, 128, o_height, o_width), device=x.device)
        
        for f in range(num_frames):
            batch = x[:, f]
            batch = batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
            # Normalize input
            batch = (batch - self.input_mean) / self.input_std
            
            # Apply convolutions with normalization and activation
            # Use small epsilon in ReLU to prevent exact zeros
            x1 = F.relu(self.norm1(self.conv1(batch)), inplace=False) + 1e-8
            x2 = F.relu(self.norm2(self.conv2(x1)), inplace=False) + 1e-8
            output = F.relu(self.norm3(self.conv3(x2)), inplace=False) + 1e-8
            
            # Clip extreme values
            output = torch.clamp(output, -10, 10)
            
            outputs[:, f] = output
            del x1, x2, output
            
        return outputs
      
class FullPatchEncoder(nn.Module):
    def __init__(self):
        super(FullPatchEncoder, self).__init__()
        
        # Patch extraction function
        self.patch_extractor = extractPatches
        
        # More efficient convolutional layers
        # Input is 128 channels from latent encoder
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=2, padding=1)  # 4x4 -> 2x2
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=896, kernel_size=2, stride=1, padding=0)  # 2x2 -> 1x1
        
        # Fully connected layer for embedding
        self.fc = nn.Linear(896, 768)  # 896 features to 768 embedding
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(768)
        
        # Configure batch norms for small batches
        self.bn1 = nn.BatchNorm2d(192, momentum=0.1, eps=1e-5, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(384, momentum=0.1, eps=1e-5, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(896, momentum=0.1, eps=1e-5, track_running_stats=True)
        
    def forward(self, latents):
        # Extract patches
        patches, attention_mask = self.patch_extractor(latents)
        
        # Get dimensions
        batch_size, segments, num_patches, frames, channels, height, width = patches.shape
        vectors = torch.zeros((batch_size, segments*num_patches, 768), device=latents.device)
        
        # If batch size is 1, switch batch norm to eval mode temporarily
        if batch_size == 1:
            was_training = self.training
            self.eval()
        
        # Process each patch
        for seg in range(segments):
            for p in range(num_patches):
                # Process first frame
                x1 = self.bn1(F.relu(self.conv1(patches[:, seg, p, 0])))
                x1 = self.bn2(F.relu(self.conv2(x1)))
                x1 = self.bn3(F.relu(self.conv3(x1)))
                x1 = x1.flatten(start_dim=1)
                emb1 = self.fc(x1)
                
                # Process second frame
                x2 = self.bn1(F.relu(self.conv1(patches[:, seg, p, 1])))
                x2 = self.bn2(F.relu(self.conv2(x2)))
                x2 = self.bn3(F.relu(self.conv3(x2)))
                x2 = x2.flatten(start_dim=1)
                emb2 = self.fc(x2)
                
                # Average embeddings and normalize
                embeddings = self.norm((emb1 + emb2) / 2)
                vectors[:, (num_patches*seg)+p] = embeddings
                
                del x1, x2, emb1, emb2
        
        # Restore training mode if it was changed
        if batch_size == 1 and was_training:
            self.train()
        
        return vectors

class FullClassifier(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=12, num_layers=12, num_classes=2):
        super(FullClassifier, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=hidden_dim,batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        
        # Initialize the classifier with proper scaling
       
        
        # Initialize weights with better scaling
        
    
    def forward(self, x):
        # Apply transformer encoder
        x = x
        x = self.transformer_encoder(x)
        pooled_x = x.mean(dim=1)
        logits = self.fc(pooled_x)
        
        # Add numerical stability fixes
        logits = logits - logits.max(dim=1, keepdim=True)[0]  # Subtract max for stability
        logits = logits.clamp(-15, 15)  # Prevent extreme values
        return logits
class FullVideoClassifier(nn.Module):
    def __init__(self, latent_encoder, patch_encoder, classifier):
        super(FullVideoClassifier, self).__init__()
        self.latent_encoder = latent_encoder
        self.patch_encoder = patch_encoder
        self.classifier = classifier
        self.chunk_size = 8  # Process 8 frames at a time
        
    @torch.cuda.amp.autocast()
    def process_chunk(self, chunk):
        return self.latent_encoder(chunk)
        
    def forward(self, videos):
        batch_size, num_frames = videos.shape[:2]
        latents_list = []
        
        # Process video in chunks
        for i in range(0, num_frames, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_frames)
            chunk = videos[:, i:end_idx]
            
            # Process chunk with autocast
            with torch.cuda.amp.autocast():
                latent = self.latent_encoder(chunk)
                latents_list.append(latent)
            
            # Optional: Force CUDA synchronization after each chunk
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Concatenate chunks
        with torch.cuda.amp.autocast():
            latents = torch.cat(latents_list, dim=1)
            del latents_list
            
            # Process through patch encoder
            if self.training:
                st_vectors = torch.utils.checkpoint.checkpoint(self.patch_encoder, latents)
            else:
                st_vectors = self.patch_encoder(latents)
            del latents
            
            # Process through classifier
            if self.training:
                outputs = torch.utils.checkpoint.checkpoint(self.classifier, st_vectors)
            else:
                outputs = self.classifier(st_vectors)
            del st_vectors
            
            return outputs