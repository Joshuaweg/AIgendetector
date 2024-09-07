import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def extractPatches(latents):
    num_videos = latents.shape[0]
    num_frames = latents.shape[1]
    height = latents.shape[2]
    width = latents.shape[3]
    frames_per_patch = 4
    height_per_patch = 24
    width_per_patch = 24
    batch_patches = []
    for video in range(num_videos):
        patches = []
        for segment in range(int(num_frames/frames_per_patch)):
            segment_patches = []
            for h in range(0, height, height_per_patch):
                for w in range(0, width, width_per_patch):
                    for frame in range(segment*frames_per_patch, (segment+1)*frames_per_patch):
                        patch = latents[video, frame, h:h+height_per_patch, w:w+width_per_patch]
                        segment_patches.append(patch)
            segment_patches = torch.stack(segment_patches)
            patches.append(segment_patches)
        patches = torch.stack(patches)
        batch_patches.append(patches)
    batch_patches=torch.stack(batch_patches)
    batch_patches = batch_patches.view(-1,int(num_frames/frames_per_patch)*100, 24, 24)
    #print("batch_patches shape: ",batch_patches.shape)
    return batch_patches
class LatentEncoder(nn.Module):
    def __init__(self):
        super(LatentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)  # 3 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)  # 32 -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)  # 64 -> 128 channels
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)  # 128 -> 256 channels
        
        self.pool = nn.AdaptiveAvgPool2d((15,15))

        # Fully connected layers
        self.fc1 = nn.Linear(8*15*15, 14400)  # Adjust based on the output size of the conv layers
        self.fc2 = nn.Linear(14400, 120 * 120)  # Map to 120x120 latent space
    

    def forward(self, x):
        batch_size, num_frames, height, width, channels = x.shape

        # Reshape input to [batch_size * num_frames, channels, height, width] for Conv2d
        x = x.view(batch_size * num_frames, channels, height, width)  # [B * F, C, H, W]
        cl1 = F.relu(self.conv1(x))  # Output shape: (batch, 32, height/2, width/2)
        cl2 = F.relu(self.conv2(cl1))  # Output shape: (batch, 64, height/4, width/4)
        cl3 = F.relu(self.conv3(cl2))  # Output shape: (batch, 128, height/8, width/8)
        cl4 = F.relu(self.conv4(cl3))  # Output shape: (batch, 256, height/16, width/16)
        # Pool the output to reduce the spatial dimensions
        pooled = self.pool(cl4)
        # Flatten the output of the convolutional layers
        flat = pooled.view(pooled.size(0), -1)  # Flatten to (batch_size, 256 * 15 * 15)
        #print("pooled flattened shape: ",flat.shape)
        # Pass through fully connected layers
        r_flat= F.relu(self.fc1(flat))  # First fully connected layer
        #print("first transformation shape: ",r_flat.shape)
        output = self.fc2(r_flat)  # Second fully connected layer
        #print("output transformation shape: ",output.shape)
        
        # Reshape to (batch_size, 120, 120)
        output = output.view(batch_size, num_frames, 120, 120)
        return output
class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        
         # Convolutional layers to extract spatial features
        self.patch_extractor = extractPatches
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer to reduce to 100-dimensional vector
        self.fc = nn.Linear(8 * 24 * 24, 100)
        
    def forward(self, latents):
        #print("latents shape: ",latents.shape)
        x = self.patch_extractor(latents)
        #print("patches shape: ",x.shape)
        batch_size, patches, height, width = x.shape
        # Apply convolutional layers with ReLU activation
         # Reshape input from (segments, patches, height, width) -> (segments * patches, 1, height, width)
        x = x.view(-1, 1, height,width)  # Combine segments and patches into one batch dimension
        
        # Apply convolutional layers with ReLU activation
        cl1= torch.relu(self.conv1(x))
        cl2 = torch.relu(self.conv2(cl1))
        cl3 = torch.relu(self.conv3(cl2))
        
        # Flatten the output, keeping the batch dimension
        flat = cl3.view(cl3.size(0), -1)  # Flatten to (batch_size, 64*24*24)
        #print("patch encoder shape: ",flat.shape)
        # Apply the fully connected layer
        output = self.fc(flat)  # Now (batch_size, 100)
        output = output.view(batch_size, patches, -1)  # Reshape to (batch_size, segments, patches, 100)
        return output

class Classifier(nn.Module):
    def __init__(self,input_dim=100,hidden_dim=4, num_layers=1, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=hidden_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        pooled_x = x.mean(dim=1)
        output= self.fc(pooled_x)
        return output
class VideoClassifier(nn.Module):
    def __init__(self,latent_encoder, patch_encoder, classifier):
        super(VideoClassifier, self).__init__()
        self.latent_encoder = latent_encoder
        self.patch_encoder = patch_encoder
        self.classifier = classifier
    def forward(self, videos):
        latents = self.latent_encoder(videos)
        st_vectors = self.patch_encoder(latents)
        outputs = self.classifier(st_vectors)
        return outputs