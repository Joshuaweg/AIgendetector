import os
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures

# Source paths configuration
SOURCE_PATHS = {
    'DynamicCrafter': r'F:\Gen-Video\DynamicCrafter\train_DynamicCrafter',
    'latte': r'F:\Gen-Video\latte\train_Latte',
    'OpenSora': r'F:\Gen-Video\OpenSora\train_OpenSora',
    'pika': r'F:\Gen-Video\pika.tar\train_pika',
    'sd': r'F:\Gen-Video\sd\train_SD',
    'seine': r'F:\Gen-Video\seine\train_SEINE',
    'svd': r'F:\Gen-Video\svd\train_SVD',
    'ZeroScope': r'F:\Gen-Video\train_ZeroScope',
    'VideoCrafter': r'F:\Gen-Video\VideoCrafter\train_VideoCrafter'
}

REAL_VIDEO_BASE = r'F:\Gen-Video\Youku_1M_10s'
DATASET_PATH = r'F:\Gen-Video\dataset'

class DatasetPreparer:
    def __init__(self, output_path=DATASET_PATH):
        """Initialize dataset preparer with output path."""
        self.output_path = Path(output_path)
        self.videos_dir = self.output_path
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def list_videos_in_directory(self, directory, pattern='*.mp4'):
        """List all videos in a directory and its subdirectories."""
        directory = Path(directory)
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            return []
        return list(directory.rglob(pattern))

    def list_real_videos(self):
        """List videos from all subdirectories in Youku dataset."""
        all_videos = []
        base_path = Path(REAL_VIDEO_BASE)
        
        # List all subdirectories (100 folders)
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        
        # Collect videos from each subdirectory
        for subdir in tqdm(subdirs, desc="Scanning Youku folders"):
            videos = self.list_videos_in_directory(subdir)
            all_videos.extend(videos)
        
        return all_videos

    def copy_video(self, source_path, prefix):
        """Copy a single video with appropriate prefix."""
        try:
            new_name = f"{prefix}_{source_path.name}"
            target_path = self.videos_dir / new_name
            shutil.copy2(source_path, target_path)
            return True
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
            return False

    def process_source(self, source_dir, source_name, num_videos):
        """Process videos from a single source directory."""
        videos = self.list_videos_in_directory(source_dir)
        
        if len(videos) < num_videos:
            print(f"Warning: Only {len(videos)} videos available in {source_dir}")
            selected_videos = videos
        else:
            selected_videos = random.sample(videos, num_videos)
        
        print(f"Processing {len(selected_videos)} videos from {source_name}")
        prefix = f"ai_{source_name.lower()}"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for video in selected_videos:
                futures.append(executor.submit(self.copy_video, video, prefix))
            
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc=source_name):
                future.result()

    def process_real_videos(self, num_videos):
        """Process real videos from Youku dataset."""
        print("\nScanning real videos...")
        all_videos = self.list_real_videos()
        
        if len(all_videos) < num_videos:
            print(f"Warning: Only {len(all_videos)} real videos available")
            selected_videos = all_videos
        else:
            selected_videos = random.sample(all_videos, num_videos)
        
        print(f"Processing {len(selected_videos)} real videos")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for video in selected_videos:
                futures.append(executor.submit(self.copy_video, video, "real"))
            
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc="Real videos"):
                future.result()

    def prepare_dataset(self):
        """Prepare the complete dataset."""
        print(f"Starting dataset preparation in {self.output_path}...")
        
        # Process AI videos (1112 from each source)
        for source_name, source_dir in SOURCE_PATHS.items():
            print(f"\nProcessing {source_name} videos from {source_dir}...")
            self.process_source(source_dir, source_name, 1112)
        
        # Process Real videos (10000 total)
        print("\nProcessing real videos...")
        self.process_real_videos(10000)
        
        # Verify dataset
        self.verify_dataset()

    def verify_dataset(self):
        """Verify the dataset structure and counts."""
        all_videos = list(self.videos_dir.glob('*.mp4'))
        ai_videos = [v for v in all_videos if v.name.startswith('ai_')]
        real_videos = [v for v in all_videos if v.name.startswith('real_')]
        
        print("\nDataset Statistics:")
        print(f"Dataset location: {self.videos_dir}")
        print(f"AI Videos: {len(ai_videos)}")
        print(f"Real Videos: {len(real_videos)}")
        print(f"Total Videos: {len(all_videos)}")
        
        # Print per-source statistics
        for source in SOURCE_PATHS.keys():
            source_videos = [v for v in ai_videos if f"ai_{source.lower()}_" in v.name.lower()]
            print(f"{source}: {len(source_videos)} videos")

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.prepare_dataset()