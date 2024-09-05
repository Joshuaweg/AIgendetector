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

resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-2]  # delete the last fc layer.
resnet = nn.Sequential(*modules)

resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False
class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        
        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer to reduce to 100-dimensional vector
        self.fc = nn.Linear(64*24*24, 100)
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten the output
        x = x.flatten()
        
        # Apply the fully connected layer
        x = self.fc(x)
        return x
class CustomResNetEncoder(nn.Module):
    def __init__(self, original_model):
        super(CustomResNetEncoder, self).__init__()
        self.features =original_model
        self.fc1 = nn.Linear(368640, 14400)  # Assuming final feature map size is (batch, 2048, 1, 1)
        self.fc2 = nn.Linear(14400, 120*120)  # Map to 120x120

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 120, 120)  # Reshape to (batch, 120, 120)
        return x
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
def extractPatches(frames):
    spacetime_patches = []
    for i in range(0,len(frames),4):
        patches = []
        for j in range(4):
            frame = frames[i+j][0]
            frame_patches = []
            for k in range(0,120,24):
                for l in range(0,120,24):
                    patch = frame[k:k+24,l:l+24]
                    if patch.shape == (24,24):
                        #print(patch.shape)
                        frame_patches.append(patch)
            frame_patches = torch.stack(frame_patches)
            patches.append(frame_patches)
        patches = torch.stack(patches)
        print(patches.shape)
        spacetime_patches.append(patches)
    return torch.stack(spacetime_patches)
#now what we will do is use a Variational Encoder to compress the frames into a lower dimension latent space (120x120)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def compressFrames(frames):
    latent_encoder = CustomResNetEncoder(resnet)
    latent_frames = []
    for frame in frames:
        frame = transform(frame)
        frame = frame.unsqueeze(0)
        latent_frame = latent_encoder(frame)
        latent_frames.append(latent_frame)
    #print(len(latent_frames))
    return torch.stack(latent_frames)
def vectorizePatches(patches):
    encoder = PatchEncoder()
    vectorized_patches = []
    for p in range(150):
        patch  = patches[p]
        vectorized_patch = encoder(patch)
        vectorized_patches.append(vectorized_patch)
    return torch.stack(vectorized_patches)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)       
    frames = extractFrames(r"data/one/VideoCrafter_067.mp4")
    label = "fake"
    categories = ["real", "fake"] 
    # Convert label to tensor only once outside the loop
    label_index = categories.index(label)
    label_tensor = torch.tensor([label_index]).long().to(device)
    label_list = [label_tensor.clone().detach() for _ in range(1000)]

    latents = compressFrames(frames)
    patches = extractPatches(latents)
    sp_patches = get_spacetime(patches)
    sp_patches = sp_patches.squeeze(1)

    # Convert sp_patches to device only once
    sp_vectors = vectorizePatches(sp_patches).float().to(device)
    sp_vector_list = [sp_vectors.clone().detach() for _ in range(1000)]
    clf = classifier.Classifier(100, 10, 1, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)

    for sp_vectors, label_tensor in zip(sp_vector_list, label_list):
        optimizer.zero_grad()  # Reset gradients
        outputs = clf(sp_vectors)  # Forward pass
        print(outputs)
        #print(label_tensor)
        loss = criterion(outputs, label_tensor)  # Calculate loss
        print("Loss: ",loss.item())  # Print loss value
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    
