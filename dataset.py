import cv2
import os
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import shutil
import pandas as pd
import random
from pathlib import Path
import tempfile
import io

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

class VideoDataset(Dataset):
    def __init__(self, data_dir, target_size=512, max_frames=24, video_paths=None):
        """
        Initialize the dataset from S3, local directory, or explicit path list.

        Args:
            data_dir (str): S3 path or local directory containing videos (ignored if video_paths provided)
            target_size (int): Size to resize frames to
            max_frames (int): Maximum number of frames to use per video
            video_paths (list, optional): Explicit list of video paths to use
        """
        self.target_size = target_size
        self.max_frames = max_frames

        # If explicit paths provided, use those
        if video_paths is not None:
            print(f"Initializing dataset from {len(video_paths):,} provided paths")
            self.videos = []
            for path in video_paths:
                # Determine label from filename
                filename = os.path.basename(path).lower()
                if filename.startswith(('real_',)):
                    label = 1
                else:
                    label = 0
                self.videos.append((path, label))

            # Shuffle
            random.shuffle(self.videos)

            # Stats
            real_count = sum(1 for _, label in self.videos if label == 1)
            ai_count = sum(1 for _, label in self.videos if label == 0)
            print(f"\nDataset Statistics:")
            print(f"Total videos: {len(self.videos)}")
            print(f"Real videos: {real_count}")
            print(f"AI-generated videos: {ai_count}")
            return

        if str(data_dir).startswith('s3://'):
            print(f"Initializing dataset from S3: {data_dir}")
            import boto3  # Import boto3 only when needed
            self.s3_client = boto3.client('s3')
            bucket = data_dir.split('/')[2]
            prefix = '/'.join(data_dir.split('/')[3:])
            self.bucket = bucket
            
            # List all videos in the S3 bucket
            print(f"Scanning S3 bucket: {bucket} with prefix: {prefix}")
            paginator = self.s3_client.get_paginator('list_objects_v2')
            self.videos = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.mp4'):
                        path = obj['Key']
                        # Determine label based on filename prefix
                        label = 1 if 'real_' in path.lower() else 0
                        self.videos.append((path, label))
                        
            print(f"Found {len(self.videos)} videos in S3")
        else:
            print(f"Initializing dataset from local path: {data_dir}")
            self.data_dir = Path(data_dir)
            self.videos = []
            for video in self.data_dir.glob('**/*.mp4'):
                parts = video.parts
                if 'Real' in parts:
                    self.videos.append((str(video), 1))
                elif 'AI-Generated' in parts:
                    self.videos.append((str(video), 0))
        
        # Shuffle the videos
        random.shuffle(self.videos)
        
        # Print dataset statistics
        real_count = sum(1 for _, label in self.videos if label == 1)
        ai_count = sum(1 for _, label in self.videos if label == 0)
        print(f"\nDataset Statistics:")
        print(f"Total videos: {len(self.videos)}")
        print(f"Real videos: {real_count}")
        print(f"AI-generated videos: {ai_count}")

    @classmethod
    def from_path_file(cls, path_file, target_size=512, max_frames=24):
        """
        Create dataset from a file containing video paths (one per line).

        Args:
            path_file (str): Path to text file with video paths
            target_size (int): Size to resize frames to
            max_frames (int): Maximum number of frames per video

        Returns:
            VideoDataset instance
        """
        print(f"Loading paths from {path_file}")
        with open(path_file, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(paths):,} video paths")
        return cls(data_dir=None, target_size=target_size, max_frames=max_frames, video_paths=paths)

    def load_video_from_s3(self, video_path):
        """Load video from S3 bucket"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
                self.s3_client.download_file(self.bucket, video_path, temp_file.name)
                frames = []
                cap = cv2.VideoCapture(temp_file.name)
                
                # Check if video opened successfully
                if not cap.isOpened():
                    print(f"Failed to open video file: {video_path}")
                    return None
                
                # Get video properties with validation
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0 or total_frames > 1000:  # Add reasonable upper limit
                    print(f"Invalid frame count ({total_frames}) for video: {video_path}")
                    return None
                
                # Validate frame dimensions
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width <= 0 or height <= 0:
                    print(f"Invalid dimensions (width={width}, height={height}) for video: {video_path}")
                    return None
                
                # Read frames with additional validation
                frame_count = 0
                while frame_count < self.max_frames * 2:  # Limit maximum frames to prevent memory issues
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self.resize_frame(frame)
                        
                        # Validate frame after processing
                        if frame.shape[:2] != (self.target_size, self.target_size):
                            print(f"Incorrect frame size {frame.shape} for video {video_path}")
                            continue
                        
                        frames.append(frame)
                        frame_count += 1
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count} from {video_path}: {str(e)}")
                        continue
                    
                cap.release()
                
                if not frames:
                    print(f"No valid frames extracted from video: {video_path}")
                    return None
                
                # Convert to numpy array and validate
                frames = np.array(frames, dtype=np.float32)
                if np.isnan(frames).any() or np.isinf(frames).any():
                    print(f"Invalid values (NaN/Inf) detected in frames from {video_path}")
                    return None
                    
                return frames
                
        except Exception as e:
            print(f"Error loading video from S3 ({video_path}): {str(e)}")
            return None
        
    def resize_frame(self, frame):
        """Resize frame to target_size x target_size"""
        return cv2.resize(frame, (self.target_size, self.target_size))
        
    def sample_frames(self, frames):
        """Sample frames according to the specified strategy"""
        num_frames = len(frames)
        
        if num_frames <= self.max_frames:
            return frames
        else:
            diff = num_frames - self.max_frames
            interval_start = random.randint(0, diff)
            return frames[interval_start:interval_start + self.max_frames]
        
    def __len__(self):
        return len(self.videos)
        
    def __getitem__(self, idx):
        try:
            video_path, label = self.videos[idx]
            
            if hasattr(self, 's3_client'):
                frames = self.load_video_from_s3(video_path)
                if len(frames) == 0:
                    print(f"Failed to load video: {video_path}")
                    return None, None, str(video_path)
            else:
                try:
                    frames = []
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        return None, None, str(video_path)
                        
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self.resize_frame(frame)
                        frames.append(frame)
                    cap.release()
                    
                    if len(frames) == 0:
                        print(f"No frames extracted from video: {video_path}")
                        return None, None, str(video_path)
                except Exception as e:
                    print(f"Error loading video {video_path}: {str(e)}")
                    return None, None, str(video_path)
            
            # Sample frames according to our strategy
            frames = self.sample_frames(frames)
            
            # Convert to numpy array and normalize
            frames = np.array(frames) / 255.0
            
            # Convert to tensor
            frames_tensor = torch.FloatTensor(frames)
            return frames_tensor, label, str(video_path)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None, None, str(video_path)

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
    """
    Custom collate function to handle variable-length videos.
    Filters out any None values from failed video loads.
    """
    # Filter out None values from failed video loads
    valid_batch = [(video, label, path) for video, label, path in batch if video is not None and isinstance(video, torch.Tensor)]
    
    if len(valid_batch) == 0:
        raise RuntimeError("No valid videos in batch")
    
    videos, labels, paths = zip(*valid_batch)
    
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

# ---------------------------------------------------------------------------
# Optical Flow Dataset
# ---------------------------------------------------------------------------

def evenly_spaced_indices(total_frames, n=24):
    """Return n evenly-spaced frame indices covering [0, total_frames-1]."""
    if total_frames <= n:
        return list(range(total_frames))
    return [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]


def compute_flow_maps(frames_np, flow_h=64, flow_w=64):
    """
    Compute 6-channel optical flow maps between consecutive frame pairs.

    Args:
        frames_np: numpy array [T, H, W, 3] float32 in [0,1]
        flow_h, flow_w: spatial resolution for flow maps

    Returns:
        flow_tensor: torch.FloatTensor [T-1, 6, flow_h, flow_w]
    """
    T = len(frames_np)
    flow_maps = []

    prev_gray = None
    prev_u, prev_v = None, None

    for i in range(T):
        frame_uint8 = (frames_np[i] * 255).astype(np.uint8)
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (flow_w, flow_h))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray_resized,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            u, v = flow[..., 0], flow[..., 1]
            mag = np.sqrt(u**2 + v**2)
            angle = np.arctan2(v, u)

            if prev_u is not None:
                delta_u = u - prev_u
                delta_v = v - prev_v
            else:
                delta_u = np.zeros_like(u)
                delta_v = np.zeros_like(v)

            # Stack 6 channels: [u, v, mag, angle, delta_u, delta_v]
            flow_6ch = np.stack([u, v, mag, angle, delta_u, delta_v], axis=0)  # [6, H_f, W_f]

            # Normalize each channel to [-1, 1] using percentile clipping
            for c in range(6):
                p99 = np.percentile(np.abs(flow_6ch[c]), 99) + 1e-6
                flow_6ch[c] = np.clip(flow_6ch[c] / p99, -1.0, 1.0)

            flow_maps.append(flow_6ch)
            prev_u, prev_v = u, v

        prev_gray = gray_resized

    if len(flow_maps) == 0:
        # Single-frame video — return zeros
        return torch.zeros(1, 6, flow_h, flow_w, dtype=torch.float32)

    flow_array = np.stack(flow_maps, axis=0)  # [T-1, 6, H_f, W_f]
    return torch.from_numpy(flow_array).float()


class FlowVideoDataset(Dataset):
    """
    Wraps VideoDataset to also return optical flow maps.
    Returns: (frames_tensor, flow_maps, label, path)
      - frames_tensor: [T, H, W, C] float32
      - flow_maps:     [T-1, 6, flow_h, flow_w] float32
      - label:         int (0=AI, 1=Real)
      - path:          str
    """
    def __init__(self, data_dir, target_size=512, max_frames=24,
                 video_paths=None, flow_h=64, flow_w=64):
        self.inner = VideoDataset(
            data_dir, target_size=target_size, max_frames=max_frames,
            video_paths=video_paths
        )
        self.flow_h = flow_h
        self.flow_w = flow_w

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        frames, label, path = self.inner[idx]
        if frames is None:
            return None, None, None, path

        frames_np = frames.numpy()  # [T, H, W, C] float32 in [0,1]
        try:
            flow_maps = compute_flow_maps(frames_np, self.flow_h, self.flow_w)
        except Exception as e:
            print(f"Flow computation failed for {path}: {e}")
            T = frames_np.shape[0]
            flow_maps = torch.zeros(T - 1, 6, self.flow_h, self.flow_w)

        return frames, flow_maps, label, path

    @property
    def videos(self):
        return self.inner.videos


class ManifestFlowDataset(Dataset):
    """
    Loads videos from a manifest CSV (path, label, generator, split).
    Returns: (frames_tensor, flow_maps, label, path)
    """
    def __init__(self, manifest_path, split='train', target_size=512,
                 max_frames=24, flow_h=64, flow_w=64):
        df = pd.read_csv(manifest_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.target_size = target_size
        self.max_frames = max_frames
        self.flow_h = flow_h
        self.flow_w = flow_w

        # Build (path, label) list for VideoDataset
        video_paths = self.df['path'].tolist()
        labels = self.df['label'].tolist()

        # Create inner VideoDataset with explicit paths
        self._inner = VideoDataset(
            data_dir=None,
            target_size=target_size,
            max_frames=max_frames,
            video_paths=video_paths,
        )
        # Override labels from manifest (VideoDataset infers from filename — ignore its stats print)
        self._inner.videos = list(zip(video_paths, labels))

        real_count = sum(1 for l in labels if l == 1)
        ai_count = sum(1 for l in labels if l == 0)
        print(f"ManifestFlowDataset [{split}]: {len(self._inner)} videos ({real_count} real / {ai_count} AI)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frames, label, path = self._inner[idx]
        if frames is None:
            return None, None, None, path
        frames_np = frames.numpy()
        try:
            flow_maps = compute_flow_maps(frames_np, self.flow_h, self.flow_w)
        except Exception as e:
            T = frames_np.shape[0]
            flow_maps = torch.zeros(T - 1, 6, self.flow_h, self.flow_w)
        return frames, flow_maps, label, path

    @property
    def videos(self):
        return self._inner.videos


def _load_cached_item(video_path, label, npy_path, target_size, n_frames, flow_h, flow_w):
    """
    Shared loader for cached flow datasets.
    Loads precomputed flow from npy_path and the corresponding n_frames
    evenly-spaced RGB frames from video_path for spatial input.

    Returns: (frames_tensor [T, H, W, C], flow_maps [T-1, 6, flow_h, flow_w], label, path)
    """
    # Load cached flow
    if not os.path.exists(npy_path):
        return None, None, label, str(video_path)
    flow_array = np.load(npy_path)                            # [T-1, 6, H_f, W_f]
    flow_maps  = torch.from_numpy(flow_array).float()

    # Load evenly-spaced RGB frames from video.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, label, str(video_path)

    frames = []
    while len(frames) < n_frames * 2:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
    cap.release()

    if not frames:
        return None, None, label, str(video_path)

    sample_idx = evenly_spaced_indices(len(frames), n_frames)
    frames = [frames[i] for i in sample_idx]

    if not frames:
        return None, None, label, str(video_path)

    frames_np     = np.array(frames, dtype=np.float32) / 255.0   # [T, H, W, C]
    frames_tensor = torch.from_numpy(frames_np)
    return frames_tensor, flow_maps, label, str(video_path)


class CachedFlowDataset(Dataset):
    """
    Directory-scanning dataset that loads precomputed flow from a cache root
    and reads the matching evenly-spaced RGB frames from the source video.

    Directory layout expected (mirrored between src and cache):
        src_root/AI-Generated/<gen>/<name>.mp4
        flow_cache_root/AI-Generated/<gen>/<name>.npy

    Returns: (frames_tensor, flow_maps, label, path)
      - frames_tensor: [n_frames, H, W, C] float32 in [0, 1]
      - flow_maps:     [n_frames-1, 6, flow_h, flow_w] float32
    """

    def __init__(self, src_root, flow_cache_root, target_size=512,
                 n_frames=24, flow_h=64, flow_w=64):
        self.src_root        = Path(src_root)
        self.flow_cache_root = Path(flow_cache_root)
        self.target_size     = target_size
        self.n_frames        = n_frames
        self.flow_h          = flow_h
        self.flow_w          = flow_w

        self.videos = []
        for video in self.src_root.rglob('*.mp4'):
            parts = video.parts
            if 'Real' in parts:
                self.videos.append((str(video), 1))
            elif 'AI-Generated' in parts:
                self.videos.append((str(video), 0))

        random.shuffle(self.videos)
        real_count = sum(1 for _, l in self.videos if l == 1)
        print(f"CachedFlowDataset: {len(self.videos)} videos  "
              f"({real_count} real / {len(self.videos) - real_count} AI)")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        rel      = Path(video_path).relative_to(self.src_root)
        npy_path = (self.flow_cache_root / rel).with_suffix('.npy')
        return _load_cached_item(
            video_path, label, str(npy_path),
            self.target_size, self.n_frames, self.flow_h, self.flow_w,
        )


class CachedManifestFlowDataset(Dataset):
    """
    Manifest-based dataset that loads precomputed flow from a cache root
    and reads matching evenly-spaced RGB frames from source videos.

    Args:
        manifest_path:   path to flow_manifest.csv (columns: path, label, split)
        src_root:        root that video paths in the manifest are relative to
                         (used to compute relative paths for cache lookup)
        flow_cache_root: root of precomputed .npy files
        split:           'train' or 'val'

    Returns: (frames_tensor, flow_maps, label, path)
    """

    def __init__(self, manifest_path, src_root, flow_cache_root,
                 split='train', target_size=512, n_frames=24,
                 flow_h=64, flow_w=64):
        df = pd.read_csv(manifest_path)
        self.df              = df[df['split'] == split].reset_index(drop=True)
        self.src_root        = Path(src_root)
        self.flow_cache_root = Path(flow_cache_root)
        self.target_size     = target_size
        self.n_frames        = n_frames
        self.flow_h          = flow_h
        self.flow_w          = flow_w

        self.videos = list(zip(self.df['path'].tolist(), self.df['label'].tolist()))
        real_count  = sum(1 for _, l in self.videos if l == 1)
        print(f"CachedManifestFlowDataset [{split}]: {len(self.videos)} videos  "
              f"({real_count} real / {len(self.videos) - real_count} AI)")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        try:
            rel = Path(video_path).relative_to(self.src_root)
        except ValueError:
            print(f"CachedManifestFlowDataset: path not under src_root.\n"
                  f"  path:     {video_path}\n"
                  f"  src_root: {self.src_root}\n"
                  f"  Pass --src-root matching the manifest path prefix.")
            return None, None, label, str(video_path)
        npy_path = (self.flow_cache_root / rel).with_suffix('.npy')
        return _load_cached_item(
            video_path, label, str(npy_path),
            self.target_size, self.n_frames, self.flow_h, self.flow_w,
        )


def flow_collate_fn(batch):
    """Collate function for FlowVideoDataset — filters None entries."""
    batch = [(f, fl, l, p) for f, fl, l, p in batch if f is not None]
    if not batch:
        return None, None, None, []

    frames_list, flow_list, labels_list, paths = zip(*batch)

    # Pad frames to same shape
    max_T = max(f.shape[0] for f in frames_list)
    max_H = max(f.shape[1] for f in frames_list)
    max_W = max(f.shape[2] for f in frames_list)

    padded_frames = []
    for f in frames_list:
        T, H, W, C = f.shape
        pad = torch.zeros(max_T - T, H, W, C)
        padded_frames.append(torch.cat([f, pad], dim=0))

    # Flow maps: [T-1, 6, H_f, W_f] — T-1 consistent if max_frames fixed
    max_T1 = max(fl.shape[0] for fl in flow_list)
    _, flow_C, flow_H, flow_W = flow_list[0].shape
    padded_flows = []
    for fl in flow_list:
        T1 = fl.shape[0]
        if T1 < max_T1:
            pad = torch.zeros(max_T1 - T1, flow_C, flow_H, flow_W)
            fl = torch.cat([fl, pad], dim=0)
        padded_flows.append(fl)

    return (
        torch.stack(padded_frames),
        torch.stack(padded_flows),
        torch.tensor(labels_list, dtype=torch.long),
        list(paths)
    )


if __name__ == '__main__':
    # Example usage
    dataset = VideoDataset('F:/Gen-Video/dataset')
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # Test the data loader
    for batch_idx, (data, labels, paths) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample paths: {paths[:2]}")
        break
        
