"""This file will be used to process the images and extract the features from them. 
The features will be used to train the model. The features will be extracted using the following methods:
We will take a video and extract the frames from it. We will then use the frames to extract the features.
"""
import cv2
import os
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import classifier
import sys
from captum.attr import IntegratedGradients

class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        
         # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer to reduce to 100-dimensional vector
        self.fc = nn.Linear(64 * 24 * 24, 100)
        
    def forward(self, x):
       
        batch_size, segments, patches, height, width = x.shape
        # Apply convolutional layers with ReLU activation
         # Reshape input from (segments, patches, height, width) -> (segments * patches, 1, height, width)
        x = x.view(-1, 1, height,width)  # Combine segments and patches into one batch dimension
        
        # Apply convolutional layers with ReLU activation
        cl1= torch.relu(self.conv1(x))
        cl2 = torch.relu(self.conv2(cl1))
        cl3 = torch.relu(self.conv3(cl2))
        
        # Flatten the output, keeping the batch dimension
        flat = cl3.view(cl3.size(0), -1)  # Flatten to (batch_size, 64*24*24)
        
        # Apply the fully connected layer
        output = self.fc(flat)  # Now (batch_size, 100)
        output = output.view(batch_size, segments*patches, -1)  # Reshape to (batch_size, segments, patches, 100)
        return output
class CustomResNetEncoder(nn.Module):
    def __init__(self):
        super(CustomResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)  # 3 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # 32 -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 64 -> 128 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 128 -> 256 channels
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 15 * 48, 14400)  # Adjust based on the output size of the conv layers
        self.fc2 = nn.Linear(14400, 120 * 120)  # Map to 120x120 latent space
    

    def forward(self, x):
        cl1 = F.relu(self.conv1(x))  # Output shape: (batch, 32, height/2, width/2)
        cl2 = F.relu(self.conv2(cl1))  # Output shape: (batch, 64, height/4, width/4)
        cl3 = F.relu(self.conv3(cl2))  # Output shape: (batch, 128, height/8, width/8)
        cl4 = F.relu(self.conv4(cl3))  # Output shape: (batch, 256, height/16, width/16)

        # Flatten the output of the convolutional layers
        flat = cl4.view(cl4.size(0), -1)  # Flatten to (batch_size, 256 * 15 * 15)
        
        # Pass through fully connected layers
        r_flat= F.relu(self.fc1(flat))  # First fully connected layer
        output = self.fc2(r_flat)  # Second fully connected layer
        
        # Reshape to (batch_size, 120, 120)
        output = output.view(x.size(0), 120, 120)
        return output
#videoPath = "data/one/VideoCrafter_067.mp4
def extractFrames(videoPath):
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    frames =[]
    while success:
        frames.append(image)
        success,image = vidcap.read()
        count += 1

    vidcap.release()
    return frames
#after we get the frames, we will extract features from the frames in the form of space-time patches
#patches will be of size 4x32x32x3 (4 frames, 32x32 pixels, 3 channels)
#the dimens of a single frame is 320 x 576. each frame can be cut into 180 32x32 pieces, so 4 frames will have 180 patches
#again, we will take 4 frames at a time and extract 180 patches from them
def extractPatches(latents):
    
    num_frames = latents.shape[0]
    height = latents.shape[1]
    width = latents.shape[2]
    frames_per_patch = 4
    height_per_patch = 24
    width_per_patch = 24
    patches = []
    for segment in range(int(num_frames/frames_per_patch)):
        segment_patches = []
        for h in range(0, height, height_per_patch):
            for w in range(0, width, width_per_patch):
                for frame in range(segment*frames_per_patch, (segment+1)*frames_per_patch):
                    patch = latents[frame, h:h+height_per_patch, w:w+width_per_patch]
                    segment_patches.append(patch)
        segment_patches = torch.stack(segment_patches)
        patches.append(segment_patches)
    return torch.stack(patches)
#now what we will do is use a Variational Encoder to compress the frames into a lower dimension latent space (120x120)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def compressFrames(frames):
    latent_encoder = CustomResNetEncoder().to(device)
    latent_frames = []
    for frame in frames:
        frame = transform(frame)
        frame = frame.unsqueeze(0)
        latent_frame = latent_encoder(frame)
        latent_frames.append(latent_frame)
    #print(len(latent_frames))
    return torch.stack(latent_frames)
def vectorizePatches(patches):
    patch_encoder = PatchEncoder().to(device)
    vectors = patch_encoder(patches)
    return vectors
def create_patch_vector_dictionary(patches, vectors):
    patch_dict = {}
    vector_dict ={}
    for segment in range(6):
        segment_vectors = []
        for patch in range(25):
            if patches[segment,:,patch,:,:] not in patch_dict:
                patch_dict[patches[segment,:,patch,:,:]] = vectors[segment,patch,0,:]
                vector_dict[vectors[segment,patch,0,:]] = patches[segment,:,patch,:,:]
    return patch_dict, vector_dict
def get_spacetime (patches):
    spacetime_patches = []
    for segment in range(6):
        for patch in range(25):
            sp_patch = patches[segment,:,patch,:,:]
            sp_patch = sp_patch.unsqueeze(0)
            spacetime_patches.append(sp_patch)
    return torch.stack(spacetime_patches)
if __name__ == "__main__":
    device = torch.device("cpu")
# Initialize models
    latent_encoder = CustomResNetEncoder().to(device)
    patch_encoder = PatchEncoder().to(device)
    clf = classifier.Classifier(100, 10, 1, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)

# Extract frames from the video and keep them in the graph
    frames = extractFrames(r"data/one/VideoCrafter_067.mp4")
    frames = [torch.from_numpy(frame).float() for frame in frames]
    frames = torch.stack(frames)  # Preprocess and move to device
    frames = frames.transpose(1,3).to(device)
    print(frames.shape)
# Forward pass through LatentEncoder (keep computation graph intact)
    latents = latent_encoder(frames)
    print(latents.shape)
# Extract patches and spacetime vectors (do not detach)
    patches = extractPatches(latents)  # Ensure this returns tensors connected to the graph
    print(patches.shape)

# Vectorize patches
    vectors = vectorizePatches(patches.unsqueeze(0)).float().to(device)
    print(vectors.shape)
# Forward pass through Classifier
    outputs = clf(vectors)
    print(outputs)
# Calculate loss
    label = torch.tensor([1], dtype=torch.long).to(device)  # Fake label for example
    print(label)
    loss = criterion(outputs, label)

# Backward pass (triggers gradients for Integrated Gradients)
    loss.backward()

# Perform Integrated Gradients (target = final output)
    ig = IntegratedGradients(clf)

# Now perform integrated gradients with respect to the original frames
    attributions = ig.attribute(vectors, target=label, n_steps=1)
    print(attributions.shape)
    
