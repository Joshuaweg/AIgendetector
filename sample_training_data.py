"""
Training Data Sampling Script

Combines stratified sampling by AI model + hard example mining.
All embeddings and results stored on disk, not in RAM.
"""

import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import pickle
import h5py  # For memory-mapped storage
from tqdm import tqdm
from datetime import datetime
import random

# CLIP
import open_clip

# Local imports
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from dataset import VideoDataset, custom_collate_fn
from torch.utils.data import DataLoader, Subset

# Configuration
BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sampling_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,

    # Stratified sampling config
    'samples_per_ai_model': 10000,  # Per AI model type
    'real_samples': 90000,  # Total real samples to match AI count

    # Hard example mining config
    'hard_example_confidence_low': 0.3,  # Uncertain range lower bound
    'hard_example_confidence_high': 0.7,  # Uncertain range upper bound
    'hard_examples_per_model': 2000,  # Additional hard examples per model

    # CLIP config
    'clip_model': 'ViT-B-32',
    'clip_pretrained': 'laion2b_s34b_b79k',
    'clip_batch_size': 32,

    # Output
    'output_sample_list': os.path.join(OUTPUT_DIR, 'training_sample_list.json'),
}


def scan_dataset_by_model(dataset_path):
    """
    Scan dataset and organize videos by source model.

    Returns:
        model_videos: dict mapping model_name -> list of video paths
    """
    print("Scanning dataset for model types...")

    model_videos = defaultdict(list)
    real_videos = []

    # Walk through dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(('.mp4', '.avi', '.mov', '.webm')):
                continue

            filepath = os.path.join(root, file)
            filename = file.lower()

            if 'real' in filename:
                real_videos.append(filepath)
            else:
                # Extract model name from filename
                # Format: ai_modelname_ModelName_12345.mp4
                if filename.startswith('ai_'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        model_name = parts[1]  # e.g., 'pika', 'svd', 'sd'
                        model_videos[model_name].append(filepath)
                else:
                    # Unknown AI format, put in 'other'
                    model_videos['other'].append(filepath)

    model_videos['real'] = real_videos

    # Print summary
    print("\nDataset Summary:")
    print("-" * 40)
    total = 0
    for model, videos in sorted(model_videos.items()):
        print(f"  {model}: {len(videos):,} videos")
        total += len(videos)
    print("-" * 40)
    print(f"  Total: {total:,} videos")

    return dict(model_videos)


def load_classifier(config):
    """Load the pretrained video classifier for hard example mining."""
    device = config['device']
    model_path = os.path.join(MODEL_DIR, 'full_classifier_1_85.pt')

    print(f"Loading classifier from {model_path}...")

    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)
    model = FullVideoClassifier(latentEncoder, patchEncoder, classifier).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_clip_model(config):
    """Load CLIP model for diversity sampling."""
    print("Loading CLIP model...")
    device = config['device']

    model, _, preprocess = open_clip.create_model_and_transforms(
        config['clip_model'],
        pretrained=config['clip_pretrained']
    )
    model = model.to(device)
    model.eval()

    return model, preprocess


def get_classifier_predictions_streaming(model, video_paths, config, output_path):
    """
    Get classifier predictions for videos, streaming results to disk.

    Saves predictions to HDF5 file for memory-efficient access.

    Returns:
        output_path: Path to HDF5 file with predictions
    """
    import cv2

    device = config['device']
    batch_size = config['batch_size']
    target_size = 512
    max_frames = 24

    print(f"Getting classifier predictions for {len(video_paths):,} videos...")
    print(f"Streaming results to {output_path}")

    # Create HDF5 file for streaming storage
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        f.create_dataset('predictions', shape=(len(video_paths),), dtype='float32')
        f.create_dataset('confidences', shape=(len(video_paths),), dtype='float32')

        # Store paths as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('paths', shape=(len(video_paths),), dtype=dt)

        # Process videos one batch at a time
        for batch_start in tqdm(range(0, len(video_paths), batch_size), desc="Classifying"):
            batch_end = min(batch_start + batch_size, len(video_paths))
            batch_paths = video_paths[batch_start:batch_end]

            # Load videos in batch
            batch_frames_list = []
            valid_indices = []

            for i, path in enumerate(batch_paths):
                try:
                    cap = cv2.VideoCapture(path)
                    frames = []

                    while len(frames) < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (target_size, target_size))
                        frames.append(frame)
                    cap.release()

                    if len(frames) == 0:
                        # Store invalid prediction
                        f['predictions'][batch_start + i] = 0.5
                        f['confidences'][batch_start + i] = 0.0
                        f['paths'][batch_start + i] = path
                        continue

                    # Format: (num_frames, H, W, C) - matches original dataset
                    frames = np.array(frames, dtype=np.float32) / 255.0
                    batch_frames_list.append(torch.FloatTensor(frames))
                    valid_indices.append(batch_start + i)

                except Exception as e:
                    # Store invalid prediction
                    f['predictions'][batch_start + i] = 0.5
                    f['confidences'][batch_start + i] = 0.0
                    f['paths'][batch_start + i] = path

            if not batch_frames_list:
                continue

            # Pad to same number of frames
            max_frame_count = max(fr.shape[0] for fr in batch_frames_list)
            padded = []
            for fr in batch_frames_list:
                if fr.shape[0] < max_frame_count:
                    # Pad: (num_frames, H, W, C)
                    padding = torch.zeros(max_frame_count - fr.shape[0], target_size, target_size, 3)
                    fr = torch.cat([fr, padding], dim=0)
                padded.append(fr)

            batch_tensor = torch.stack(padded).to(device)

            # Run classifier
            with torch.no_grad():
                try:
                    outputs = model(batch_tensor)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                    for j, idx in enumerate(valid_indices):
                        f['predictions'][idx] = probs[j]
                        f['confidences'][idx] = abs(probs[j] - 0.5) * 2
                        f['paths'][idx] = video_paths[idx]

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    for idx in valid_indices:
                        f['predictions'][idx] = 0.5
                        f['confidences'][idx] = 0.0
                        f['paths'][idx] = video_paths[idx]

            # Flush periodically
            if batch_start % 500 == 0:
                f.flush()
                torch.cuda.empty_cache()

    print(f"Predictions saved to {output_path}")
    return output_path


def compute_clip_embeddings_streaming(clip_model, preprocess, video_paths, config, output_path):
    """
    Compute CLIP embeddings for video frames, streaming to disk.

    Stores one embedding per video (average of frame embeddings).
    """
    import cv2

    device = config['device']
    batch_size = config['clip_batch_size']

    print(f"Computing CLIP embeddings for {len(video_paths):,} videos...")
    print(f"Streaming to {output_path}")

    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        dummy_emb = clip_model.encode_image(dummy)
        emb_dim = dummy_emb.shape[1]

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', shape=(len(video_paths), emb_dim), dtype='float32')

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('paths', shape=(len(video_paths),), dtype=dt)

        for idx, video_path in enumerate(tqdm(video_paths, desc="CLIP embeddings")):
            try:
                # Load a few frames from video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Sample 4 frames evenly
                frame_indices = np.linspace(0, max(0, total_frames - 1), 4, dtype=int)

                frames = []
                for fi in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        from PIL import Image
                        pil_frame = Image.fromarray(frame)
                        frames.append(preprocess(pil_frame))
                cap.release()

                if frames:
                    # Get embeddings
                    frames_tensor = torch.stack(frames).to(device)
                    with torch.no_grad():
                        embeddings = clip_model.encode_image(frames_tensor)
                        # Average across frames
                        avg_embedding = embeddings.mean(dim=0).cpu().numpy()
                else:
                    avg_embedding = np.zeros(emb_dim, dtype='float32')

                f['embeddings'][idx] = avg_embedding
                f['paths'][idx] = video_path

                # Flush periodically
                if idx % 500 == 0:
                    f.flush()

            except Exception as e:
                f['embeddings'][idx] = np.zeros(emb_dim, dtype='float32')
                f['paths'][idx] = video_path

    print(f"CLIP embeddings saved to {output_path}")
    return output_path


def stratified_sample(model_videos, config):
    """
    Perform stratified sampling across AI models.

    Returns:
        selected_videos: dict mapping model -> list of selected video paths
    """
    print("\nPerforming stratified sampling...")

    selected = {}
    samples_per_model = config['samples_per_ai_model']

    for model_name, videos in model_videos.items():
        if model_name == 'real':
            # Sample real videos
            n_samples = min(config['real_samples'], len(videos))
            selected[model_name] = random.sample(videos, n_samples)
        else:
            # Sample AI videos
            n_samples = min(samples_per_model, len(videos))
            selected[model_name] = random.sample(videos, n_samples)

        print(f"  {model_name}: selected {len(selected[model_name]):,} / {len(videos):,}")

    return selected


def hard_example_mining(predictions_path, model_videos, config):
    """
    Find hard examples (uncertain predictions) from each model.

    Reads predictions from disk, doesn't load all into RAM.

    Returns:
        hard_examples: dict mapping model -> list of hard example paths
    """
    print("\nMining hard examples...")

    low = config['hard_example_confidence_low']
    high = config['hard_example_confidence_high']
    max_per_model = config['hard_examples_per_model']

    # Build path -> model mapping
    path_to_model = {}
    for model_name, videos in model_videos.items():
        for v in videos:
            path_to_model[v] = model_name

    # Read predictions from disk and find hard examples
    hard_examples = defaultdict(list)

    with h5py.File(predictions_path, 'r') as f:
        predictions = f['predictions']
        paths = f['paths']

        for idx in tqdm(range(len(predictions)), desc="Finding hard examples"):
            pred = predictions[idx]
            path = paths[idx]

            if isinstance(path, bytes):
                path = path.decode('utf-8')

            # Check if uncertain
            if low <= pred <= high:
                model_name = path_to_model.get(path, 'unknown')
                hard_examples[model_name].append({
                    'path': path,
                    'prediction': float(pred),
                    'uncertainty': float(1 - abs(pred - 0.5) * 2)
                })

    # Sort by uncertainty and take top N per model
    selected_hard = {}
    for model_name, examples in hard_examples.items():
        examples.sort(key=lambda x: x['uncertainty'], reverse=True)
        selected_hard[model_name] = [e['path'] for e in examples[:max_per_model]]
        print(f"  {model_name}: {len(selected_hard[model_name]):,} hard examples")

    return selected_hard


def diversity_sample_from_clusters(embeddings_path, video_paths, n_samples, n_clusters=50):
    """
    Select diverse samples using CLIP embeddings stored on disk.

    Uses clustering to ensure diversity.
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"\nDiversity sampling {n_samples:,} from {len(video_paths):,} videos...")

    # Load embeddings for specified paths
    path_to_idx = {}

    with h5py.File(embeddings_path, 'r') as f:
        all_paths = f['paths'][:]
        for idx, p in enumerate(all_paths):
            if isinstance(p, bytes):
                p = p.decode('utf-8')
            path_to_idx[p] = idx

        # Get embeddings for our subset - filter to valid paths
        valid_paths = [p for p in video_paths if p in path_to_idx]
        indices = [path_to_idx[p] for p in valid_paths]

        if len(indices) == 0:
            return random.sample(video_paths, min(n_samples, len(video_paths)))

        # Sort indices for HDF5 (required) and track original positions
        sorted_order = np.argsort(indices)
        sorted_indices = np.array(indices)[sorted_order]

        # Load in chunks to avoid memory issues
        chunk_size = 10000
        embeddings_list = []

        for i in range(0, len(sorted_indices), chunk_size):
            chunk_indices = sorted_indices[i:i + chunk_size].tolist()
            chunk_embeddings = f['embeddings'][chunk_indices]
            embeddings_list.append(chunk_embeddings)

        sorted_embeddings = np.vstack(embeddings_list)

        # Restore original order
        inverse_order = np.argsort(sorted_order)
        embeddings = sorted_embeddings[inverse_order]

    # Cluster
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters < 2:
        return valid_paths[:n_samples]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Sample equally from each cluster
    samples_per_cluster = max(1, n_samples // n_clusters)
    selected_indices = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        n_select = min(samples_per_cluster, len(cluster_indices))
        if n_select > 0:
            selected = np.random.choice(cluster_indices, n_select, replace=False)
            selected_indices.extend(selected)

    # Limit to n_samples
    if len(selected_indices) > n_samples:
        selected_indices = random.sample(selected_indices, n_samples)

    # Map back to paths
    selected_paths = [valid_paths[i] for i in selected_indices]

    print(f"  Selected {len(selected_paths):,} diverse samples")

    return selected_paths


def combine_samples(stratified, hard_examples, config):
    """
    Combine stratified samples with hard examples, removing duplicates.
    """
    print("\nCombining samples...")

    combined = defaultdict(set)

    # Add stratified samples
    for model, paths in stratified.items():
        combined[model].update(paths)

    # Add hard examples
    for model, paths in hard_examples.items():
        combined[model].update(paths)

    # Convert to lists
    final = {model: list(paths) for model, paths in combined.items()}

    # Summary
    total = 0
    print("\nFinal sample distribution:")
    print("-" * 40)
    for model, paths in sorted(final.items()):
        print(f"  {model}: {len(paths):,}")
        total += len(paths)
    print("-" * 40)
    print(f"  Total: {total:,}")

    return final


def save_sample_list(samples, config):
    """Save the final sample list to JSON."""
    output_path = config['output_sample_list']

    # Convert to serializable format
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
        'samples_by_model': samples,
        'total_samples': sum(len(v) for v in samples.values()),
        'all_paths': [p for paths in samples.values() for p in paths],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSample list saved to {output_path}")

    # Also save just the paths for easy loading
    paths_only = output_path.replace('.json', '_paths.txt')
    with open(paths_only, 'w') as f:
        for path in output['all_paths']:
            f.write(path + '\n')

    print(f"Path list saved to {paths_only}")

    return output_path


def main():
    """Main sampling pipeline."""
    print("=" * 80)
    print("Training Data Sampling Pipeline")
    print("Stratified + Hard Examples + CLIP Diversity")
    print("=" * 80)

    device = CONFIG['device']
    print(f"Using device: {device}")

    # Step 1: Scan dataset
    model_videos = scan_dataset_by_model(DATASET_PATH)

    # Step 2: Stratified sampling
    stratified_samples = stratified_sample(model_videos, CONFIG)

    # Step 3: Get classifier predictions for hard example mining
    # Collect all videos for prediction
    all_videos = []
    for videos in model_videos.values():
        all_videos.extend(videos)

    predictions_path = os.path.join(OUTPUT_DIR, 'classifier_predictions.h5')

    if not os.path.exists(predictions_path):
        classifier = load_classifier(CONFIG)
        get_classifier_predictions_streaming(
            classifier, all_videos, CONFIG, predictions_path
        )
        del classifier
        torch.cuda.empty_cache()
    else:
        print(f"Using existing predictions from {predictions_path}")

    # Step 4: Hard example mining
    hard_examples = hard_example_mining(predictions_path, model_videos, CONFIG)

    # Step 5: CLIP diversity sampling
    # Compute embeddings for all videos, then select diverse samples per model
    clip_embeddings_path = os.path.join(OUTPUT_DIR, 'clip_embeddings.h5')

    if not os.path.exists(clip_embeddings_path):
        clip_model, preprocess = load_clip_model(CONFIG)
        compute_clip_embeddings_streaming(
            clip_model, preprocess, all_videos, CONFIG, clip_embeddings_path
        )
        del clip_model
        torch.cuda.empty_cache()
    else:
        print(f"Using existing CLIP embeddings from {clip_embeddings_path}")

    # Step 6: Apply diversity sampling to stratified samples
    print("\nApplying diversity sampling within each model category...")
    diverse_samples = {}
    for model_name, videos in stratified_samples.items():
        if len(videos) > 100:  # Only cluster if enough samples
            diverse_samples[model_name] = diversity_sample_from_clusters(
                clip_embeddings_path,
                videos,
                n_samples=len(videos),  # Keep same count but diverse selection
                n_clusters=min(50, len(videos) // 20)
            )
        else:
            diverse_samples[model_name] = videos

    # Replace stratified with diverse
    stratified_samples = diverse_samples

    # Step 7: Combine samples
    final_samples = combine_samples(stratified_samples, hard_examples, CONFIG)

    # Step 8: Save
    save_sample_list(final_samples, CONFIG)

    print("\n" + "=" * 80)
    print("Sampling complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
