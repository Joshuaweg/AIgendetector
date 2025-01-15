import os
import boto3
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures

# S3 Configuration
S3_BUCKET = 'genvideo-dataset-complete'
AI_SOURCES = [
    'DynamicCrafter',
    'latte',
    'OpenSora',
    'pika',
    'sd',
    'svd',
    'seine',
    'ZeroScope',
    'VideoCrafter'
]

class DatasetPreparer:
    def __init__(self, local_base_path='data'):
        self.s3_client = boto3.client('s3')
        self.local_base_path = Path(local_base_path)
        
        # Create necessary directories
        self.local_base_path.mkdir(parents=True, exist_ok=True)
        (self.local_base_path / 'real').mkdir(exist_ok=True)
        (self.local_base_path / 'ai').mkdir(exist_ok=True)

    def list_s3_videos(self, prefix):
        """List all videos in a given S3 prefix."""
        videos = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                videos.extend([obj['Key'] for obj in page['Contents'] 
                             if obj['Key'].lower().endswith(('.mp4', '.avi', '.mov'))])
        return videos

    def download_video(self, s3_key, local_path):
        """Download a single video from S3."""
        try:
            self.s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
            return True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False

    def process_source(self, source, num_videos, is_real=False):
        """Process videos from a single source."""
        prefix = f"{'Real Videos/Youku_10_s' if is_real else f'AI Videos/{source}'}"
        videos = self.list_s3_videos(prefix)
        
        if len(videos) < num_videos:
            print(f"Warning: Only {len(videos)} videos available for {source}")
            selected_videos = videos
        else:
            selected_videos = random.sample(videos, num_videos)
        
        # Process videos
        print(f"Processing videos for {source}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for video in selected_videos:
                target_dir = 'real' if is_real else 'ai'
                local_path = self.local_base_path / target_dir / Path(video).name
                futures.append(executor.submit(self.download_video, video, local_path))
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc=f"{source}"):
                future.result()

    def prepare_dataset(self):
        """Prepare the complete dataset."""
        print("Starting dataset preparation...")
        
        # Process AI videos (1112 from each source)
        for source in AI_SOURCES:
            print(f"\nProcessing {source} videos...")
            self.process_source(source, 1112, is_real=False)
        
        # Process Real videos (10000 total)
        print("\nProcessing real videos...")
        self.process_source('Youku', 10000, is_real=True)
        
        # Verify dataset
        self.verify_dataset()

    def verify_dataset(self):
        """Verify the dataset structure and counts."""
        ai_count = len(list(Path(self.local_base_path / 'ai').glob('*')))
        real_count = len(list(Path(self.local_base_path / 'real').glob('*')))
        
        print("\nDataset Statistics:")
        print(f"AI Videos: {ai_count}")
        print(f"Real Videos: {real_count}")
        print(f"Total Videos: {ai_count + real_count}")

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.prepare_dataset() 