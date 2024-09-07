import cv2
import os
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import classifier
import sys
import shutil
import pandas as pd

realVideoPath =r"C:/Users/joshu/OneDrive/Desktop/GenVideo/kinetics/k400/train"
fakeVideoPath =r"C:/Users/joshu/OneDrive/Desktop/GenVideo/zeroScope/ZeroScope/train_ZeroScope"

"""This file will be responsible for extracting dataset from video locations. first step is to
grab 1000 fake videos and 1000 real videos and copy them to location data/many/fake and data/many/real respectively
videos should be selected randomly from the paths provided above. So we need to get the list of all the videos in the path and then randomly select 1000 from each path
"""
def check_and_clear_directory(directory):
    # Check if directory exists
    if os.path.exists(directory):
        # Check if the directory is empty
        if not os.listdir(directory):
            print(f"{directory} is already empty.")
        else:
            # If not empty, delete all contents
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory and its contents
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Cleared contents of {directory}.")
    else:
        print(f"{directory} does not exist. Creating it...")
        os.makedirs(directory)

def get_video_list(path):
    video_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                video_list.append(os.path.join(root, file))
    return video_list

"""Next we need to define a dataloader for preparing the data for training.
we will use OpenCV to extract the frames from the videos"""

class VideoDataset (Dataset):
    def __init__(self,real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
       
        self.real_videos = [os.path.join(real_dir, file) for file in os.listdir(real_dir) if file.endswith('.mp4')]
        self.fake_videos = [os.path.join(fake_dir, file) for file in os.listdir(fake_dir) if file.endswith('.mp4')]
        
        # Create labels for real (1) and fake (0) videos
        self.video_paths = self.real_videos + self.fake_videos
        self.labels = [1] * len(self.real_videos) + [0] * len(self.fake_videos)
    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        # Get the video file path and corresponding label
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load the video using OpenCV
        video_capture = cv2.VideoCapture(video_path)
        
        frames = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the frame from BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply any optional transformations (e.g., resizing, normalization)
            if self.transform:
                frame = self.transform(frame)

            # Collect the frames
            frames.append(frame)
        
        # Release the video capture object
        video_capture.release()

        # Convert list of frames to a single numpy array, then to a tensor
        frames_np = np.array(frames)  # Convert list of frames to numpy array
        frames_tensor = torch.from_numpy(frames_np).float()  # Convert numpy array to a tensor

        # Return the video frames and label
        return frames_tensor, torch.tensor(label, dtype=torch.long), video_path
class PreprocessedVideoDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_files = [os.path.join(real_dir, file) for file in os.listdir(real_dir) if file.endswith('.npy')]
        self.fake_files = [os.path.join(fake_dir, file) for file in os.listdir(fake_dir) if file.endswith('.npy')]
        self.transform = transform
        
        self.video_paths = self.real_files + self.fake_files
        self.labels = [1] * len(self.real_files) + [0] * len(self.fake_files)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_file = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load pre-processed frames from .npy file
        frames_np = np.load(video_file)
        
        frames_tensor = torch.from_numpy(frames_np).float()

        return frames_tensor, torch.tensor(label, dtype=torch.long)
def pad_videos(videos, max_frames, max_height, max_width):
    padded_videos = []
    for video in videos:
        num_frames, height, width, channels = video.shape
        pad_frames = max_frames - num_frames
        pad_height = max_height - height
        pad_width = max_width - width
        
        # Pad the video with zeros (assuming RGB format with 3 channels)
        padded_video = np.pad(
            video,
            ((0, pad_frames), (0, pad_height), (0, pad_width), (0, 0)),  # Only pad along time, height, and width
            mode='constant',
            constant_values=0
        )
        padded_videos.append(padded_video)
    
    # Convert list of padded videos to a single numpy array before converting to a tensor
    padded_videos_np = np.array(padded_videos)
    
    return torch.from_numpy(padded_videos_np).float()

def custom_collate_fn(batch):
    videos, labels, paths = zip(*batch)
    
    for i, (video, path) in enumerate(zip(videos,paths)):
        if len(video.shape) != 4:
            print(f"Error: Video at index {i}, path {path} has an invalid shape: {video.shape}")
            continue  # Skip this video or handle the error
    # Find the max dimensions in the batch
    max_frames = max([video.shape[0] for video in videos])
    max_height = max([video.shape[1] for video in videos])
    max_width = max([video.shape[2] for video in videos])
    
    # Pad all videos to the max dimensions
    padded_videos = pad_videos(videos, max_frames, max_height, max_width)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_videos, labels, paths
def preprocess_videos(video_paths, output_dir, transform=None):
    for idx, video_path in enumerate(video_paths):
        # Load the video
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if transform:
                frame = transform(frame)

            frames.append(frame)

        video_capture.release()
        
        # Save frames as numpy array
        frames_np = np.array(frames)
        np.save(os.path.join(output_dir, f"video_{idx}.npy"), frames_np)

if __name__ == '__main__':
    real_videos = r'data/many/real'
    fake_videos = r'data/many/fake'
    dataset = VideoDataset(real_videos, fake_videos)
    print(f"Dataset length: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False, collate_fn=custom_collate_fn)

    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} - Data shape: {data.shape}, Labels shape: {labels.shape}")
        
