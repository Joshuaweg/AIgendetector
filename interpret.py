""" This file contains the functions that interpret the output of the videoClassifier using captums Integrated Gradients and Layer Integrated Gradients. """
import torch
import os
import sys
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LRP
from captum.attr import visualization as viz
from classifier import *
from dataset import VideoDataset, custom_collate_fn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

def delete_all_files_in_folder(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Loop through all files and directories in the specified folder
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # If it's a file, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # If it's a directory, delete it and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory {directory} does not exist.")


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
def load_model_for_inference(checkpoint_path, device):
    # Initialize model
    latentEncoder = LatentEncoder()
    patchEncoder = PatchEncoder()
    classifier = Classifier()
    model = VideoClassifier(latentEncoder, patchEncoder, classifier).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load only model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model
def forward_and_interpret(video):
    classes = ['AI-Generated', 'real']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    #device = torch.device("cpu")
    torch.cuda.empty_cache()
    vclf = load_model_for_inference('model\\videoClassifier_epoch_11.pth', device)
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
    print("predicted result: ", classes[pred.item()])
    progress_bar = tqdm(total=100, desc='Calculating Gradients', position=pred.item(), leave=True)
    def hook_fn(module, inputs):
        progress_bar.update(1)
    hook = vclf.register_forward_pre_hook(hook_fn)
    attributions, delta = ig.attribute(video, baseline, target=pred.item(), return_convergence_delta=True, n_steps=100,internal_batch_size=1)
    progress_bar.close()
    hook.remove()
    return attributions, output, classes[pred.item()]
def forward_and_interpret_LRP(video):
    classes = ['AI-Generated', 'real']
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    torch.cuda.empty_cache()
    latentEncoder = LatentEncoder().to(device)
    patchEncoder = PatchEncoder().to(device)
    classifier = Classifier().to(device)
    vclf = VideoClassifier(latentEncoder,patchEncoder,classifier).to(device)
    vclf.load_state_dict(torch.load('model\\videoClassifier_epoch_8.pth', map_location=device))
    vclf.eval()
    #we will use Integrated Gradients to generate the attributions
    torch.cuda.empty_cache()
    lrp = LRP(vclf)
    video = video.to(device)
    #we will use the first frame of the video as the baseline
    baseline = torch.zeros_like(video)
    #conduct a forward pass
    output = vclf(video)
    pred = torch.argmax(output, dim=1)
    #generate the attributions
    print("predicted result: ", classes[pred.item()])
    progress_bar = tqdm(total=50, desc='Calculating Gradients', position=0, leave=True)
    def hook_fn(module, inputs):
        progress_bar.update(1)
    hook = vclf.register_forward_pre_hook(hook_fn)
    attributions, delta = lrp.attribute(video,target=pred, return_convergence_delta=True,verbose=True)
    attributions = attributions+1e-8
    progress_bar.close()
    hook.remove()
    return attributions, output, classes[pred.item()]
#finally we will visualize the attributions
def visualize(attributions, video, save_path='plots', label=""):
    # Squeeze out the batch dimension
    video = video.squeeze(0)
    attributions = attributions.squeeze(0)
    video = video.permute(0, 3, 1, 2).detach().cpu().numpy()
    attributions = attributions.permute(0, 3, 1, 2).detach().cpu().numpy()
    delete_all_files_in_folder(save_path)
    for i, (v_frame, a_frame) in enumerate(zip(video, attributions)):
        v_frame = v_frame.transpose(1, 2, 0).astype(np.uint8)
        a_frame = a_frame.transpose(1, 2, 0)
        fig,ax=viz.visualize_image_attr_multiple(
            a_frame,
            v_frame,
            signs=["all", "positive"], 
            methods=["original_image", "heat_map"], 
            cmap='coolwarm', 
            fig_size=(12, 9),
            show_colorbar=True,
            use_pyplot=False,
            titles=["Original Video -"+label, "Pixel-Wise Attribution of Video - "+label],
        )
        
        # Save the plot as an image
        save_frame_path = f"{save_path}\\attributions_frame_{i+1:03}.png"
        fig.savefig(save_frame_path, bbox_inches='tight')
        
        # Close the figure to avoid displaying it and free up memory
        plt.close(fig)
def visualize_overlay(attributions, video, save_path='plots', label=""):
    # Squeeze out the batch dimension
    print("attributions from 1 frame: ",attributions[0])
    video = video.squeeze(0)
    attributions = attributions.squeeze(0)
    video = video.permute(0, 3, 1, 2).detach().cpu().numpy()
    attributions = attributions.permute(0, 3, 1, 2).detach().cpu().numpy()
    delete_all_files_in_folder(save_path)
    for i, (v_frame, a_frame) in enumerate(zip(video, attributions)):
        v_frame = v_frame.transpose(1, 2, 0).astype(np.uint8)
        a_frame = a_frame.transpose(1, 2, 0)
        fig, ax=viz.visualize_image_attr(
            a_frame,
            v_frame,
            method="heat_map", 
            cmap='seismic',
            fig_size=(20, 15),
            show_colorbar=True,
            use_pyplot=False,
            title="Pixel-Wise Attribution of Video - "+label,
        )
        
        # Save the plot as an image
        save_frame_path = f"{save_path}\\attributions_frame_{i+1:03}.png"
        fig.savefig(save_frame_path, bbox_inches='tight')
        
        # Close the figure to avoid displaying it and free up memory
        plt.close(fig)

def save_attributions_video(image_folder='plots', output_video='output_video_real_Overlay_test_e8_2.mp4', fps=8):

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print(images)
    images.sort()  # Optional: Sort to ensure correct frame order

# Read the first image to get dimensions (assuming all images have the same size)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or other codecs
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Loop through all images and write them into the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

# Release the video writer
    video.release()

    print(f"Video saved as {output_video}")
if __name__=="__main__":
    video_path = "data\\many\\fake\\VideoCrafter_117891.mp4" #fake
    #video_path = "data\\many\\fake\VideoCrafter_100838.mp4"
    #video_path = "data\\many\\real\\5lwcgRvkUAI_000011_000021.mp4" #real
    if "VideoCraft" in video_path:
        frames_per_second = 8
    else:
        frames_per_second = 30
    print(video_path)
    video,label,path = load_video(video_path)
    video = video.unsqueeze(0)
    attributions, output, label = forward_and_interpret(video)
    print("generating attribution visualizations")
    visualize_overlay(attributions, video, label=label)
    save_attributions_video(output_video='output_video_real_Overlay_test_e11_1_as.mp4')