""" This file contains the functions that interpret the output of the videoClassifier using captums Integrated Gradients and Layer Integrated Gradients. """
import torch
import sys
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from classifier import *
from dataset import VideoDataset, custom_collate_fn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


#first we need to select a video from saved test paths at random:
def select_random_video():
    with open('data\\test_paths.txt', 'r') as f:
        lines = f.readlines()
    video_path = random.choice(lines)
    return video_path

#next we need to load the video and preprocess it using dataloaders custom_collate_fn
def load_video(video_path, transform = None):
    if 'real' in video_path:
        label = torch.tensor([1]).long()
    else:
        label = torch.tensor([0]).long()
    video_capture = cv2.VideoCapture(video_path)
        
    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
            # Convert the frame from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply any optional transformations (e.g., resizing, normalization)
        if transform:
            frame = transform(frame)

            # Collect the frames
        frames.append(frame)
        
        # Release the video capture object
    video_capture.release()
    frames_np = np.array(frames) 
     # Convert list of frames to numpy array
    frames_tensor = torch.from_numpy(frames_np).float()  # Convert numpy array to a tensor
    return frames_tensor, label, video_path
def forward_and_interpret(video):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    latentEncoder = LatentEncoder().to(device)
    patchEncoder = PatchEncoder().to(device)
    classifier = Classifier().to(device)
    vclf = VideoClassifier(latentEncoder,patchEncoder,classifier).to(device)
    vclf.load_state_dict(torch.load('model\\videoClassifier.pth',weights_only=True))
    vclf.eval()
    #we will use Integrated Gradients to generate the attributions
    torch.cuda.empty_cache()
    ig = IntegratedGradients(vclf)
    video = video.to(device)
    #we will use the first frame of the video as the baseline
    baseline = torch.zeros_like(video)
    #conduct a forward pass
    output = vclf(video)
    pred = torch.argmax(output, dim=1)
    #generate the attributions
    attributions, delta = ig.attribute(video, baseline, target=pred, return_convergence_delta=True)
    return attributions, output
#finally we will visualize the attributions
def visualize(attributions, video):
    # Squeeze out the batch dimension
    video = video.squeeze(0)

    # Permute the video to PyTorch format [frames, channels, height, width]
    video = video.permute(0, 3, 1, 2)

    # Extract the first frame and convert it to NumPy format
    first_frame = video[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    print("first_frame: ", first_frame.shape)

    # Concatenate the frames horizontally (if desired) for visualization
    # Convert to [frames, height, width, channels] and stack frames horizontally
    video_np = video.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    video_concat = np.concatenate(video_np, axis=1)  # Horizontal concatenation
    print("video_concat: ", video_concat.shape)

    # Process attributions in the same way
    attributions = attributions.squeeze(0)
    attributions = attributions.permute(0, 3, 1, 2)
    
    first_attribution = attributions[0].detach().cpu().numpy().transpose(1, 2, 0)
    print("first_attribution: ", first_attribution.shape)

    # Concatenate the attribution frames horizontally
    attributions_np = attributions.permute(0, 2, 3, 1).detach().cpu().numpy()
    attributions_concat = np.concatenate(attributions_np, axis=1)  # Horizontal concatenation
    plt.imshow(attributions_concat, cmap='viridis')
    plt.axis('off')
    plt.show()
    sys.exit()
    print("attributions_concat: ", attributions_concat.shape)

    # Visualize using Captum
    viz.visualize_image_attr_multiple(
        attributions_concat, 
        video_concat, 
        signs=["all", "positive"], 
        methods=["original_image", "heat_map"], 
        cmap='viridis', 
        show_colorbar=True
    )

def save_attributions_video(attributions, video_path, frames_per_second):
    # Process attributions for saving
    attr_video = attributions.squeeze(0).detach().cpu().numpy()
    print(attr_video.shape)
    attr_video_combined = np.mean(attr_video, axis=-1)
    attr_video_combined = np.clip(attr_video_combined, 0, 1)

    # Get the colormap from matplotlib
    cmap = plt.get_cmap('viridis')

    # Prepare the video frames
    frames, height, width = attr_video_combined.shape
    colored_video = np.zeros((frames, height, width, 3), dtype=np.uint8)

    for i in range(frames):
        frame = attr_video_combined[i]
        colored_frame = cmap(frame)[:, :, :3]  # Ignore alpha channel
        colored_frame = (colored_frame * 255).astype(np.uint8)
        colored_video[i] = colored_frame

    # Write the video to disk
    save_path = 'attributions.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, frames_per_second, (width, height))

    for i in range(frames):
        out.write(colored_video[i])

    out.release()
if __name__=="__main__":
    video_path = "data\\many\\fake\\VideoCrafter_117891.mp4"
    if "VideoCraft" in video_path:
        frames_per_second = 8
    else:
        frames_per_second = 30
    print(video_path)
    video,label,path = load_video(video_path)
    video = video.unsqueeze(0)
    attributions, output = forward_and_interpret(video)
    visualize(attributions, video)
    save_attributions_video(attributions, "attributions.mp4", frames_per_second)