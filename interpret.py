""" This file contains the functions that interpret the output of the videoClassifier using captums Integrated Gradients and Layer Integrated Gradients. """
import torch
import os
import sys
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LRP
from captum.attr import visualization as viz
from classifier import *
from full_scale_classifier import *
from dataset import VideoDataset, custom_collate_fn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import math

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
    # Base directory for the project
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'
    test_paths_file = os.path.join(base_dir, 'data', 'test_paths.txt')
    with open(test_paths_file, 'r') as f:
        lines = f.readlines()
    video_path = random.choice(lines).strip()  # Added strip() to remove newline characters
    return video_path

#next we need to load the video and preprocess it using dataloaders custom_collate_fn
def load_video(video_path, transform = None):
    if 'real' in video_path:
        label = torch.tensor([1]).long()
    else:
        label = torch.tensor([0]).long()
    video_capture = cv2.VideoCapture(video_path)
        
    frames = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling rate to get 24 frames
    if total_frames < 24:
        # If video has less than 24 frames, duplicate frames
        sampling_rate = 1
        duplicate_factor = math.ceil(24 / total_frames)
    else:
        # If video has more than 24 frames, sample frames evenly
        indices = torch.linspace(0, total_frames - 1, 24).long().tolist()
        
    frame_count = 0
    frame_idx = 0
    
    while frame_count < 24 and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Only process frames at calculated indices
        if total_frames >= 24:
            if frame_idx in indices:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1
        else:
            # For short videos, use the duplicate strategy
            if frame_idx % sampling_rate == 0:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for _ in range(duplicate_factor):
                    if frame_count < 24:
                        frames.append(frame)
                        frame_count += 1
                
        frame_idx += 1
    
    video_capture.release()
    
    # If we still don't have enough frames, duplicate the last frame
    while len(frames) < 24:
        frames.append(frames[-1])
    
    # If we have too many frames, truncate
    frames = frames[:24]
    
    # Convert frames list to tensor
    frames_np = np.array(frames)
    frames_tensor = torch.from_numpy(frames_np).float()  # Convert numpy array to a tensor
    
    # Normalize using same values as training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    frames_tensor = frames_tensor / 255.0  # Scale to [0,1]
    frames_tensor = (frames_tensor - mean) / std
    
    # Verify shape
    assert frames_tensor.shape == (24, 512, 512, 3), f"Incorrect shape: {frames_tensor.shape}, expected (24, 512, 512, 3)"
    
    return frames_tensor,frames, label, video_path

def verify_model_state(model, checkpoint_path):
    # Load original checkpoint
    device = next(model.parameters()).device  # Get the device of the model
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint for verification with weights_only=False: {e}")
        # Fallback: try with safe globals
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = checkpoint['model_state_dict']
    
    # Compare weights
    model_state = model.state_dict()
    for key in state_dict.keys():
        if key in model_state:
            checkpoint_weight = state_dict[key].to(device)  # Move checkpoint weight to same device
            model_weight = model_state[key]
            if not torch.equal(checkpoint_weight, model_weight):
                print(f"Mismatch in {key}")
                print(f"Checkpoint: mean={checkpoint_weight.mean()}, std={checkpoint_weight.std()}")
                print(f"Model: mean={model_weight.mean()}, std={model_weight.std()}")

def load_model_correctly(model_path, device):
    # Initialize model components
    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)
    
    # Create the full model and move to device
    model = FullVideoClassifier(latentEncoder, patchEncoder, classifier).to(device)
    
    # Load the checkpoint with weights_only=False for compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load with weights_only=False: {e}")
        # Fallback: try with safe globals
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    state_dict = checkpoint['model_state_dict']
    
    # Check if the state dict has DataParallel format
    is_data_parallel = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_data_parallel:
        # Remove the 'module.' prefix if loading to a single GPU/CPU
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the state dict
    model.load_state_dict(state_dict)
    
    # Ensure all parameters are on the correct device
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    # Verify the model state after loading
    print("\nVerifying model state after loading...")
    verify_model_state(model, model_path)
    
    return model

def forward_and_interpret(video, frames):
    classes = ['AI-Generated', 'Real']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    torch.cuda.empty_cache()
    
    # Base directory and model path
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'
    model_path = os.path.join(base_dir, 'model', 'full_classifier_1_85.pt')
    vclf = load_model_correctly(model_path, device)
    vclf.eval()
    
    # Store original unnormalized video for visualization
    video_unnorm = video.clone()
    
    # Normalize video for model input
   
    
    # Enable gradients for the input
    video = video.to(device)
    video.requires_grad = True
    
    # Create baseline (black frames)
    baseline = torch.zeros_like(video, device=device)
    
    # Initialize IG with the model
    ig = IntegratedGradients(vclf)
    
    # Forward pass to get prediction
    with torch.no_grad():
        output = vclf(video)
        pred = torch.argmax(output, dim=1)
        confidence = torch.softmax(output, dim=1)[0][pred.item()].item()
        print("\nPrediction Results:")
        print(f"Predicted class: {classes[pred.item()]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: AI-Gen: {torch.softmax(output, dim=1)[0][0]:.4f}, Real: {torch.softmax(output, dim=1)[0][1]:.4f}")
    
    # Calculate attributions with progress bar
    print("\nCalculating frame attributions...")
    progress_bar = tqdm(total=300, desc='Processing', position=0, leave=True)
    
    def hook_fn(module, inputs):
        progress_bar.update(1)
    
    hook = vclf.register_forward_pre_hook(hook_fn)
    
    try:
        attributions, delta = ig.attribute(
            video,
            baseline,
            target=pred.item(),
            return_convergence_delta=True,
            n_steps=300,
            internal_batch_size=1
        )
        
        # Check if attributions are meaningful
        if torch.all(attributions == 0) or torch.isnan(attributions).any():
            print("\nWarning: Initial attributions are zero or NaN. Trying alternative approach...")
            noise = torch.randn_like(video) * 0.1
            baseline = torch.zeros_like(video) + noise
            video_perturbed = video + torch.randn_like(video) * 1e-7
            
            attributions, delta = ig.attribute(
                video_perturbed,
                baseline,
                target=pred.item(),
                return_convergence_delta=True,
                n_steps=5,
                internal_batch_size=2
            )
    
    finally:
        progress_bar.close()
        hook.remove()
    
    print("\nGenerating visualization...")
    
    # Process attributions for visualization
    attr_per_frame = attributions.squeeze(0)  # Remove batch dimension
    video_frames = video.squeeze(0)  # Remove batch dimension
    frames = torch.tensor(frames)    
    
    # Create plots directory relative to base directory
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    visualize(attributions, frames, save_path=plots_dir, label="Video Classified: "+classes[pred.item()]+" with Integrated Gradients")
    
    # Create video from the frames
    output_video_path = os.path.join(base_dir, 'attributions_v3o.mp4')
    save_attributions_video(plots_dir, output_video_path)
    print(f"\nAttribution video saved as '{output_video_path}'")
    
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
    
    # Use absolute path with base directory
    base_dir = '/media/joshua/WD_BLACK/Gen-Video'
    model_path = os.path.join(base_dir, 'model', 'full_classifier_best.pth')
    try:
        vclf.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except Exception as e:
        print(f"Failed to load model with weights_only=False: {e}")
        # Fallback: try with safe globals
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            vclf.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
        fig,ax=viz.visualize_image_attr(
            a_frame,
            v_frame,
            sign="positive", 
            method="blended_heat_map", 
            cmap='seismic', 
            fig_size=(12, 9),
            show_colorbar=True,
            use_pyplot=False,
            title=label,
        )
        
        # Save the plot as an image using cross-platform path
        save_frame_path = os.path.join(save_path, f"attributions_frame_{i+1:03}.png")
        fig.savefig(save_frame_path, bbox_inches='tight')
        
        # Close the figure to avoid displaying it and free up memory
        plt.close(fig)

def visualize_overlay(frame, attr, importance, save_path='plots', label=""):
    # Convert tensors to numpy arrays and move to CPU
    frame = frame.detach().cpu().numpy()
    attr = attr.detach().cpu().numpy()
    
    # Unnormalize the frame using the same values from training
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Unnormalize: pixel = (normalized * std) + mean
    frame = (frame * std + mean)
    
    # Clip values to [0, 1] range and convert to uint8
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    
    # Create heatmap from attributions
    attr_norm = np.abs(attr)
    attr_norm = attr_norm / (attr_norm.max() + 1e-8)  # Add small epsilon to prevent division by zero
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Display the original frame
    ax.imshow(frame)
    
    # Overlay heatmap
    heatmap = ax.imshow(attr_norm.mean(axis=2), cmap='seismic', alpha=0.5)
    
    # Add colorbar
    plt.colorbar(heatmap)
    
    # Add title with importance score
    plt.title(f"Frame Attribution (Importance: {importance:.3f})")
    
    # Save the figure using cross-platform path
    temp_path = os.path.join(save_path, "frame_temp.png")
    plt.savefig(temp_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Read the saved image and convert for video
    overlay_img = cv2.imread(temp_path)
    os.remove(temp_path)  # Clean up temporary file
    
    return overlay_img

def save_attributions_video(image_folder, output_video, fps=30):

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

if __name__ == "__main__":
    # Process a single video with attributions
    # Updated to use your specific directory structure
    base_dir = '/home/joshua/'
    video_path = os.path.join(base_dir, "Downloads", "v3o_1.mp4")
    print(f"\nProcessing video: {video_path}")
    
    # Load and preprocess the video
    video, frames, label, _ = load_video(video_path)
    video = video.unsqueeze(0)  # Add batch dimension
    
    # Ensure video is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video = video.to(device)
    
    # Run attribution analysis
    attributions, output, prediction = forward_and_interpret(video, frames)
    
    print("\nAttribution analysis complete!")
    print(f"Check '{os.path.join(base_dir, 'plots')}' directory for frame visualizations")
    print(f"Check '{os.path.join(base_dir, 'attributions_5.mp4')}' for the complete visualization video")