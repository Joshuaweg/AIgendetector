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
import random

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
    def __init__(self, real_dir, fake_dir, target_size=512, max_frames=24):
        self.real_videos = [(os.path.join(real_dir, f), 1) for f in os.listdir(real_dir) if f.endswith('.mp4')]
        self.fake_videos = [(os.path.join(fake_dir, f), 0) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
        self.videos = self.real_videos + self.fake_videos
        random.shuffle(self.videos)
        self.target_size = target_size
        self.max_frames = max_frames
        
    def resize_frame(self, frame):
        """Resize frame to target_size x target_size"""
        return cv2.resize(frame, (self.target_size, self.target_size))
        
    def sample_frames(self, frames):
        """Sample frames according to the specified strategy"""
        num_frames = len(frames)
        
        if num_frames <= self.max_frames:
            # If we have fewer frames than max_frames, use all frames
            return frames
        else:
            # Calculate random interval start
            diff = num_frames - self.max_frames
            interval_start = random.randint(0, diff)
            # Select consecutive frames starting from interval_start
            return frames[interval_start:interval_start + self.max_frames]
        
    def __len__(self):
        return len(self.videos)
        
    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Read all frames first
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_frame(frame)
            frames.append(frame)
        cap.release()
        
        # Sample frames according to our strategy
        frames = self.sample_frames(frames)
        
        # Convert to numpy array and normalize
        frames = np.array(frames) / 255.0
        
        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames)
        return frames_tensor, label, video_path
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

class SizeBatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        print(f"SizeBatchSampler initialized with batch_size: {batch_size}")  # Debug print
        self.drop_last = drop_last
        self.video_sizes = []
        
        # Get sizes of all videos and group them by size ranges
        print("Analyzing video sizes for efficient batching...")
        size_groups = {}  # Group videos by size ranges
        
        for idx in range(len(dataset)):
            # Handle both Dataset and Subset objects
            if hasattr(dataset, 'dataset'):  # If it's a Subset
                video_path = dataset.dataset.videos[dataset.indices[idx]][0]
            else:  # If it's the original Dataset
                video_path = dataset.videos[idx][0]
                
            cap = cv2.VideoCapture(video_path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            # Calculate size score
            size_score = frames * height * width
            
            # Group into size buckets (using log scale to create reasonable groups)
            size_bucket = int(np.log2(size_score))
            if size_bucket not in size_groups:
                size_groups[size_bucket] = []
            size_groups[size_bucket].append((idx, size_score))
        
        # Shuffle within each size group and create final list
        self.video_sizes = []
        for bucket in sorted(size_groups.keys()):
            group = size_groups[bucket]
            random.shuffle(group)  # Randomize videos within same size group
            self.video_sizes.extend(group)
        
    def __iter__(self):
        # Create batches while maintaining size similarity
        current_batch = []
        current_size = 0
        
        # Create a copy of video_sizes to avoid modifying original
        available_videos = self.video_sizes.copy()
        print(f"Starting new iteration with {len(available_videos)} videos")  # Debug print
        
        while available_videos:
            if current_batch:
                # Find videos within 300% size difference (0.33x to 3x)
                valid_indices = [i for i, (_, size) in enumerate(available_videos)
                               if 0.33 <= size/current_size <= 3.0]
                print(f"Current batch size: {len(current_batch)}, Target: {self.batch_size}, Valid videos: {len(valid_indices)}")  # Debug print
                
                if valid_indices:
                    idx = random.choice(valid_indices)
                    video_idx, size = available_videos.pop(idx)
                else:
                    print(f"No similar sized videos found, yielding batch of size {len(current_batch)}")  # Debug print
                    yield current_batch
                    current_batch = []
                    current_size = 0
                    continue
            else:
                video_idx, size = available_videos.pop(0)
            
            current_batch.append(video_idx)
            current_size = size if current_size == 0 else (current_size + size) / len(current_batch)
            
            if len(current_batch) == self.batch_size:
                print(f"Yielding full batch of size {len(current_batch)}")  # Debug print
                yield current_batch
                current_batch = []
                current_size = 0
        
        # Handle last batch
        if len(current_batch) > 0 and not self.drop_last:
            yield current_batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def custom_collate_fn(batch):
    videos, labels, paths = zip(*batch)
    
    # Find the max dimensions in the batch
    max_frames = max([video.shape[0] for video in videos])
    max_height = max([video.shape[1] for video in videos])
    max_width = max([video.shape[2] for video in videos])
    
    # Create padded batch with memory tracking
    batch_size = len(videos)
    estimated_memory = batch_size * max_frames * max_height * max_width * 3 * 4  # 4 bytes per float
    
    # If batch would use too much memory, reduce max dimensions
    MAX_BATCH_MEMORY = 4 * 1024 * 1024 * 1024  # 4GB
    if estimated_memory > MAX_BATCH_MEMORY:
        scale_factor = (MAX_BATCH_MEMORY / estimated_memory) ** (1/3)
        max_frames = int(max_frames * scale_factor)
        max_height = int(max_height * scale_factor)
        max_width = int(max_width * scale_factor)
    
    # Initialize tensor for batch
    padded_videos = torch.zeros(batch_size, max_frames, max_height, max_width, 3)
    
    for i, video in enumerate(videos):
        # First handle temporal dimension
        if video.shape[0] > max_frames:
            # Temporal downsampling
            indices = torch.linspace(0, video.shape[0]-1, max_frames).long()
            video = video[indices]
        
        # Then handle spatial dimensions
        if video.shape[1] != max_height or video.shape[2] != max_width:
            # Spatial resizing for all frames
            resized_frames = []
            for f in range(video.shape[0]):
                frame = video[f].numpy()
                frame = cv2.resize(frame, (max_width, max_height))
                resized_frames.append(torch.from_numpy(frame))
            video = torch.stack(resized_frames)
        
        # Now we can safely copy the video into the padded tensor
        n_frames = min(video.shape[0], max_frames)
        padded_videos[i, :n_frames] = video[:n_frames]
    
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
        
