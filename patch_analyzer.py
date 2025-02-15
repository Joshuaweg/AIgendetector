import torch
import numpy as np
from feature_extractor import FeatureExtractor
from collections import Counter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import struct
from pathlib import Path
import random
import sys

class PatchAnalyzer:
    def __init__(self, model_path, feature_cache_dir='patch_features_cache'):
        # Initialize feature extractor with visualizations disabled
        self.extractor = FeatureExtractor(model_path, enable_visualizations=False)
        self.feature_cache_dir = feature_cache_dir
        os.makedirs(feature_cache_dir, exist_ok=True)
        
        # Initialize storage for analysis
        self.unique_patches = set()  # Store hashes instead of vectors
        self.patch_occurrences = {
            'real_counts': Counter(),
            'ai_counts': Counter()
        }
    
    def _hash_patch(self, patch_vector):
        """
        Create a hash for a patch vector that's stable across runs.
        Using first 12 bytes of SHA-256 (96 bits) gives us collision probability of ~1 in 2^48 
        for 15.3M patches, which is far better than 1 in 20M requirement.
        """
        # Round to 5 decimal places for consistent hashing
        rounded = np.round(patch_vector, decimals=5)
        # Convert to bytes in a consistent way
        vector_bytes = rounded.tobytes()
        # Get SHA-256 hash
        hash_obj = hashlib.sha256(vector_bytes)
        # Return first 12 bytes as hex string
        return hash_obj.hexdigest()[:24]  # 24 hex chars = 12 bytes = 96 bits

    def collect_all_patches(self, dataset_path, cache=True):
        """Collect patch hashes from all videos in the dataset."""
        print("\nCollecting patches from all videos...")
        
        # Check if cached data exists
        cache_file = os.path.join(self.feature_cache_dir, 'patch_stats.npz')
        
        if cache and os.path.exists(cache_file):
            print("Loading cached data...")
            cached_data = np.load(cache_file, allow_pickle=True)
            self.unique_patches = set(cached_data['unique_patches'])
            self.patch_occurrences = {
                'real_counts': Counter(cached_data['real_counts'].item()),
                'ai_counts': Counter(cached_data['ai_counts'].item())
            }
            return len(self.unique_patches)
        
        # Get all video paths
        real_videos = []
        ai_videos = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    if 'real_' in file.lower():
                        real_videos.append(full_path)
                    else:
                        ai_videos.append(full_path)
        
        print(f"Found {len(real_videos)} real videos and {len(ai_videos)} AI videos")
        print(f"Theoretical maximum patches: {(len(real_videos) + len(ai_videos)) * 768:,}")
        
        # Process videos
        total_patches = 0
        for video_path in tqdm(real_videos + ai_videos, desc="Processing videos"):
            try:
                features = self.extractor.extract_features(video_path)
                if features is not None:
                    patches = features['patch_features'].squeeze(0)
                    is_real = 'real_' in os.path.basename(video_path).lower()
                    
                    # Process each patch
                    for patch in patches:
                        patch_hash = self._hash_patch(patch)
                        self.unique_patches.add(patch_hash)
                        
                        if is_real:
                            self.patch_occurrences['real_counts'][patch_hash] += 1
                        else:
                            self.patch_occurrences['ai_counts'][patch_hash] += 1
                    
                    total_patches += len(patches)
                    
                    # Print progress periodically
                    if total_patches % 100000 == 0:
                        print(f"\nProcessed {total_patches:,} patches")
                        print(f"Unique patches so far: {len(self.unique_patches):,}")
                        print(f"Memory usage: {sys.getsizeof(self.unique_patches) / (1024*1024):.2f} MB")
            
            except Exception as e:
                print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
                continue
        
        # Save cache if requested
        if cache:
            print("\nCaching data...")
            np.savez(cache_file,
                    unique_patches=list(self.unique_patches),
                    real_counts=dict(self.patch_occurrences['real_counts']),
                    ai_counts=dict(self.patch_occurrences['ai_counts']))
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total patches processed: {total_patches:,}")
        print(f"Unique patches found: {len(self.unique_patches):,}")
        print(f"Compression ratio: {len(self.unique_patches)/total_patches*100:.2f}%")
        print("\nPatch Distribution:")
        real_only = sum(1 for h in self.unique_patches 
                       if self.patch_occurrences['real_counts'][h] > 0 
                       and self.patch_occurrences['ai_counts'][h] == 0)
        ai_only = sum(1 for h in self.unique_patches 
                     if self.patch_occurrences['ai_counts'][h] > 0 
                     and self.patch_occurrences['real_counts'][h] == 0)
        shared = sum(1 for h in self.unique_patches 
                    if self.patch_occurrences['real_counts'][h] > 0 
                    and self.patch_occurrences['ai_counts'][h] > 0)
        
        print(f"Patches only in real videos: {real_only:,}")
        print(f"Patches only in AI videos: {ai_only:,}")
        print(f"Patches shared between both: {shared:,}")
        
        return len(self.unique_patches)

def main():
    # Initialize analyzer
    model_path = 'model/full_classifier_1_85.pt'
    analyzer = PatchAnalyzer(model_path)
    
    # Collect and analyze patches
    dataset_path = "F:/Gen-Video/dataset"
    unique_count = analyzer.collect_all_patches(dataset_path)
    
    print(f"\nAnalysis complete!")
    print(f"Found {unique_count:,} unique patches")

if __name__ == "__main__":
    main() 