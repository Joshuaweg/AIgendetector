import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def extractPatches(latents):
    # Input shape: (num_frames=24, height=120, width=120)
    batch_size = latents.shape[0]
    num_frames = latents.shape[1]
    channels = latents.shape[2]
    height = latents.shape[3]
    width = latents.shape[4]
    
    # Constants
    frames_per_patch = 2
    height_per_patch = 4
    width_per_patch = 4
    
    # Calculate number of patches in height and width
    num_h_patches = (height+height_per_patch-1) // height_per_patch  
    num_w_patches = (width+width_per_patch-1) // width_per_patch    
    num_segments = num_frames // frames_per_patch  
    
    # Initialize output tensor with correct shape
    # (batch_size=1,num_segments=6, spatial_patches=25, frames_per_patch=4, patch_height=24, patch_width=24)
    patches = torch.zeros((batch_size, num_segments, num_h_patches * num_w_patches, frames_per_patch,
                          channels,height_per_patch, width_per_patch), device=latents.device)
    attention_mask = torch.zeros((batch_size, num_segments, num_h_patches * num_w_patches, 
                            frames_per_patch, height_per_patch, width_per_patch),
                           device=latents.device)
    
    # Extract patches
    for batch in range(batch_size):
        for segment in range(num_segments):
            patch_idx = 0
            for h in range(0, height, height_per_patch):
                curr_h=min(height_per_patch,height-h)
                for w in range(0, width, width_per_patch):
                    curr_w=min(width_per_patch,width-w)
                    for f_idx in range(frames_per_patch):
                        frame_idx = segment * frames_per_patch + f_idx
                        #get patch
                        patch = latents[batch, frame_idx,:, h:h+height_per_patch, w:w+width_per_patch]
                        if curr_h<height_per_patch or curr_w < width_per_patch:
                            temp_patch = torch.zeros((channels, height_per_patch, width_per_patch),
                                                   device=latents.device)
                            temp_patch[:, :curr_h, :curr_w] = patch[:, :curr_h, :curr_w]
                            patch = temp_patch
                        patches[batch, segment, patch_idx, f_idx] = patch
                        for i in range(curr_h):
                            for j in range(curr_w):
                                attention_mask[batch, segment, patch_idx, f_idx, i, j] = 1
                    patch_idx += 1
    
    return patches, attention_mask


class FullLatentEncoder(nn.Module):
    def __init__(self):
        super(FullLatentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)  # 3 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # 32 -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 64 -> 128 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 128 -> 256 channels
        # Adjust based on the output size of the conv layers
    

    def forward(self, x):
        batch_size, num_frames, height, width, channels = x.shape
        if(height%16>0 or width%16>0):
            o_height=height//16
            o_width =width//16
            if height%16>0:
                o_height+=1
            if width%16>0:
                o_width+=1
            
            outputs =torch.zeros((batch_size,num_frames,256,o_height,o_width),device=x.device)
        else:
            outputs =torch.zeros((batch_size,num_frames,256,(height//16),(width//16)),device=x.device)
        for f in range(num_frames):
            batch = x[:,f]
            batch = batch.view(batch_size, channels, height, width)
            cl1 = F.relu(self.conv1(batch))  # Output shape: (batch, 32, height/2, width/2)
            cl2 = F.relu(self.conv2(cl1))  # Output shape: (batch, 64, height/4, width/4)
            cl3 = F.relu(self.conv3(cl2))  # Output shape: (batch, 128, height/8, width/8)
            output = F.relu(self.conv4(cl3))  # Output shape: (batch, 256, height/16, width/16)
            outputs[:,f] = output
            del cl1,cl2,cl3,output
        return outputs
      
class FullPatchEncoder(nn.Module):
    def __init__(self):
        super(FullPatchEncoder, self).__init__()
        
        # Convolutional layers to extract spatial features
        self.patch_extractor = extractPatches
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        
        # Fully connected layer to reduce to 100-dimensional vector
        self.fc = nn.Linear(1024, 768)
        
    def forward(self, latents):
        # Extract patches
        patches, attention_mask = self.patch_extractor(latents)
        #print("latent dims: ", patches.shape)
        
        # Get dimensions
        batch_size, segments, num_patches, frames, channels, height, width = patches.shape
        vectors = torch.zeros((batch_size, segments*num_patches, 768), device=latents.device)
        
        # Process each patch
        for seg in range(segments):
            for p in range(num_patches):
                # Concatenate frames within the patch
                if frames == 2:
                    current_patch = torch.cat([
                        patches[:, seg, p, 0],
                        patches[:, seg, p, 1]
                    ], dim=1)
                else:
                    current_patch = patches[:, seg, p, 0]
                
                # Convert to precision
                #print("patch dims: ",current_patch.shape)
                # Skip if the patch is empty or malformed
                if len(current_patch.shape) != 4:
                    print(f"Skipping malformed patch at seg={seg}, p={p}, shape={current_patch.shape}")
                    continue
                
                # Process through convolution layers
                cn1 = F.leaky_relu(self.conv1(current_patch))
                cn2 = F.leaky_relu(self.conv2(cn1))
                cn3 = F.leaky_relu(self.conv3(cn2))
                
                # Flatten and get embeddings
                fc1 = cn3.flatten(start_dim=1)
                embeddings = F.leaky_relu(self.fc(fc1))
                
                # Store the embeddings
                vectors[:, (num_patches*seg)+p] = embeddings
                
                # Clean up
                del cn1, cn2, cn3, fc1
        
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
    def __init__(self,latent_encoder, patch_encoder, classifier):
        super(FullVideoClassifier, self).__init__()
        self.latent_encoder = latent_encoder
        self.patch_encoder = patch_encoder
        self.classifier = classifier
    def forward(self, videos):
        with torch.amp.autocast("cuda"):
            latents = self.latent_encoder(videos)
            del videos
            st_vectors = self.patch_encoder(latents)
            del latents
            outputs = self.classifier(st_vectors)
            del st_vectors
            torch.cuda.empty_cache()
            return outputs