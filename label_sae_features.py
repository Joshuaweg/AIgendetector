"""
SAE Feature Labeling Script

Uses OpenCLIP for embeddings and moondream2 for captioning to automatically
interpret and label SAE features based on their top-activating video segments.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Vision-language models
import open_clip
from transformers import AutoModelForCausalLM

# Local imports
from sparse_autoencoder import SparseAutoencoder
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from dataset import VideoDataset, custom_collate_fn
from torch.utils.data import DataLoader

# Configuration
BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SAE_DIR = os.path.join(MODEL_DIR, 'sae')
OUTPUT_DIR = os.path.join(BASE_DIR, 'feature_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 4,
    'top_k_per_feature': 20,  # Top activating samples per feature
    'num_features_to_analyze': 100,  # Analyze top N most active features
    'clip_model': 'ViT-B-32',
    'clip_pretrained': 'laion2b_s34b_b79k',
    'use_moondream': True,  # Set False to skip captioning (faster)
    'num_workers': 2,
}


class ActivationCollector:
    """Collects activations and maps them back to source frames."""

    def __init__(self):
        self.activations = None
        self.hook = None

    def register_hook(self, module):
        """Register forward hook on module."""
        def hook_fn(module, input, output):
            self.activations = output.detach()
        self.hook = module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook:
            self.hook.remove()


def load_models(config):
    """Load all required models."""
    device = config['device']

    # Load video classifier
    print("Loading video classifier...")
    model_path = os.path.join(MODEL_DIR, 'full_classifier_best.pt')

    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)
    video_model = FullVideoClassifier(latentEncoder, patchEncoder, classifier).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    video_model.load_state_dict(state_dict)
    video_model.eval()

    # Load SAE
    print("Loading SAE...")
    sae_path = os.path.join(SAE_DIR, 'sae_sparse_best.pt')
    sae_checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    sae_config = sae_checkpoint['config']

    sae = SparseAutoencoder(
        input_dim=sae_config['input_dim'],
        hidden_dim=sae_config['hidden_dim'],
        sparsity_coefficient=sae_config['sparsity_coefficient'],
        tie_weights=sae_config.get('tie_weights', True)
    ).to(device)
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae.eval()

    # Load OpenCLIP
    print("Loading OpenCLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        config['clip_model'],
        pretrained=config['clip_pretrained']
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(config['clip_model'])

    # Load moondream2 (optional)
    caption_model = None
    if config['use_moondream']:
        print("Loading moondream2 for captioning...")
        try:
            caption_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",  # Use older stable revision
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            )
            # Workaround for missing attribute in older moondream versions
            if not hasattr(caption_model, '_tied_weights_keys'):
                caption_model._tied_weights_keys = []
            caption_model = caption_model.to(device)
            caption_model.eval()
        except Exception as e:
            print(f"Warning: Could not load moondream2: {e}")
            print("Continuing without captioning...")
            config['use_moondream'] = False

    return {
        'video_model': video_model,
        'sae': sae,
        'sae_config': sae_config,
        'clip_model': clip_model,
        'clip_preprocess': clip_preprocess,
        'clip_tokenizer': clip_tokenizer,
        'caption_model': caption_model,
    }


def collect_feature_activations(models, dataset, config):
    """
    Collect SAE feature activations across the dataset.
    Uses bounded heaps to limit memory usage - only keeps top-k per feature.
    Periodically saves checkpoints to disk.

    Returns:
        feature_activations: dict mapping feature_idx -> list of top activations
    """
    import heapq
    import pickle

    device = config['device']
    video_model = models['video_model']
    sae = models['sae']
    top_k = config['top_k_per_feature']

    # Setup activation hook
    collector = ActivationCollector()
    target_layer = video_model.classifier.transformer_encoder.layers[-1]
    collector.register_hook(target_layer)

    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Use min-heaps to keep only top-k activations per feature (memory bounded)
    # Heap contains tuples: (activation_value, counter, data_dict)
    # Counter is tie-breaker to avoid comparing dicts when activations are equal
    # Min-heap so we can efficiently remove smallest when full
    feature_heaps = defaultdict(list)
    heap_counter = 0  # Unique counter for tie-breaking

    checkpoint_path = os.path.join(OUTPUT_DIR, 'activations_checkpoint.pkl')

    print("Collecting SAE feature activations (memory-bounded)...")
    print(f"Keeping top {top_k} activations per feature")
    video_idx_offset = 0

    with torch.no_grad():
        for batch_idx, (data, labels, paths) in enumerate(tqdm(data_loader, desc="Processing videos")):
            data = data.to(device)
            batch_size = data.shape[0]

            # Forward through video model to get transformer activations
            _ = video_model(data)
            transformer_out = collector.activations  # (batch, seq_len, hidden_dim)

            # Forward through SAE to get feature activations
            _, features, _ = sae(transformer_out.reshape(-1, transformer_out.shape[-1]))
            # features shape: (batch * seq_len, hidden_dim_sae)

            # Reshape back to (batch, seq_len, hidden_dim_sae)
            seq_len = transformer_out.shape[1]
            features = features.reshape(batch_size, seq_len, -1)

            # Record activations per feature using bounded heaps
            for b in range(batch_size):
                video_idx = video_idx_offset + b
                label = labels[b].item()
                video_path = paths[b]

                for t in range(seq_len):
                    feature_vec = features[b, t]  # (hidden_dim_sae,)

                    # Get top activated features for this token
                    top_vals, top_idxs = torch.topk(feature_vec, k=50)

                    for val, feat_idx in zip(top_vals, top_idxs):
                        feat_idx = feat_idx.item()
                        activation = val.item()

                        item = {
                            'activation': activation,
                            'video_idx': video_idx,
                            'token_idx': t,
                            'label': label,
                            'video_path': video_path,
                        }

                        heap = feature_heaps[feat_idx]

                        if len(heap) < top_k:
                            # Heap not full, just add
                            heapq.heappush(heap, (activation, heap_counter, item))
                            heap_counter += 1
                        elif activation > heap[0][0]:
                            # New activation is larger than smallest in heap, replace
                            heapq.heapreplace(heap, (activation, heap_counter, item))
                            heap_counter += 1
                        # else: activation too small, skip

            video_idx_offset += batch_size

            # Clear CUDA cache periodically
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

            # Save checkpoint every 500 batches
            if batch_idx > 0 and batch_idx % 500 == 0:
                print(f"\nSaving checkpoint at batch {batch_idx}...")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'feature_heaps': dict(feature_heaps),
                        'video_idx_offset': video_idx_offset,
                        'batch_idx': batch_idx,
                    }, f)

    collector.remove_hook()

    # Convert heaps to sorted lists (highest activation first)
    print("Converting heaps to sorted lists...")
    feature_activations = {}
    for feat_idx, heap in feature_heaps.items():
        # Extract items and sort by activation descending
        # Tuple format: (activation, counter, item)
        items = [item for (_, _, item) in heap]
        items.sort(key=lambda x: x['activation'], reverse=True)
        feature_activations[feat_idx] = items

    # Save final activations
    final_path = os.path.join(OUTPUT_DIR, 'feature_activations.pkl')
    print(f"Saving activations to {final_path}...")
    with open(final_path, 'wb') as f:
        pickle.dump(feature_activations, f)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return feature_activations


def extract_frames_for_feature(feature_data, dataset, top_k=10):
    """
    Extract actual video frames for a feature's top activations.

    Returns list of PIL Images.
    """
    frames = []

    for item in feature_data[:top_k]:
        video_path = item['video_path']
        token_idx = item['token_idx']

        try:
            # Load video and get the relevant frame
            # token_idx roughly corresponds to frame position (depends on your tokenization)
            import cv2
            cap = cv2.VideoCapture(video_path)

            # Estimate frame number from token index
            # This depends on how your model tokenizes videos
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = min(token_idx, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append({
                    'image': pil_image,
                    'activation': item['activation'],
                    'label': item['label'],
                    'path': video_path,
                })
        except Exception as e:
            print(f"Warning: Could not extract frame from {video_path}: {e}")
            continue

    return frames


def get_clip_embeddings(frames, models, config):
    """Get CLIP embeddings for a list of frames."""
    if not frames:
        return None

    device = config['device']
    clip_model = models['clip_model']
    clip_preprocess = models['clip_preprocess']

    # Preprocess images
    images = torch.stack([clip_preprocess(f['image']) for f in frames]).to(device)

    with torch.no_grad():
        embeddings = clip_model.encode_image(images)
        embeddings = F.normalize(embeddings, dim=-1)

    return embeddings.cpu().numpy()


def get_clip_text_similarity(embeddings, text_queries, models, config):
    """
    Compare CLIP image embeddings to text queries.

    Returns dict mapping query -> mean similarity score.
    """
    if embeddings is None:
        return {}

    device = config['device']
    clip_model = models['clip_model']
    clip_tokenizer = models['clip_tokenizer']

    # Encode text queries
    text_tokens = clip_tokenizer(text_queries).to(device)

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute similarities
    embeddings_tensor = torch.tensor(embeddings).to(device)
    similarities = embeddings_tensor @ text_embeddings.T  # (n_images, n_queries)

    # Mean similarity per query
    mean_sims = similarities.mean(dim=0).cpu().numpy()

    return {query: float(sim) for query, sim in zip(text_queries, mean_sims)}


def generate_captions(frames, models, config, max_frames=5):
    """Generate captions for frames using moondream2."""
    if not config['use_moondream'] or models['caption_model'] is None:
        return []

    caption_model = models['caption_model']
    captions = []

    for frame_data in frames[:max_frames]:
        try:
            image = frame_data['image']
            # moondream2 transformers API
            enc_image = caption_model.encode_image(image)
            caption = caption_model.answer_question(
                enc_image,
                "Describe this video frame in detail. Focus on visual qualities, artifacts, and any unusual patterns.",
                tokenizer=None
            )
            captions.append({
                'caption': caption,
                'label': frame_data['label'],
                'activation': frame_data['activation'],
            })
        except Exception as e:
            print(f"Warning: Captioning failed: {e}")
            continue

    return captions


# Content-focused concept queries for organizing videos by likeness
# These are concepts CLIP can reliably recognize
# Artifact analysis will be done separately after grouping by content
CONTENT_CONCEPTS = [
    # === PEOPLE ===
    "human face close up",
    "person portrait",
    "full body person",
    "multiple people",
    "crowd of people",
    "person walking",
    "person talking",
    "person sitting",
    "hands",
    "eyes",

    # === SCENES - OUTDOOR ===
    "outdoor scene",
    "nature landscape",
    "forest trees",
    "mountains",
    "beach ocean",
    "river stream",
    "sky clouds",
    "sunset sunrise",
    "field grass",
    "garden flowers",

    # === SCENES - URBAN ===
    "city street",
    "buildings architecture",
    "road highway",
    "bridge",
    "night city lights",

    # === SCENES - INDOOR ===
    "indoor room",
    "living room",
    "kitchen",
    "office workspace",
    "bedroom",

    # === ANIMALS ===
    "dog",
    "cat",
    "bird",
    "horse",
    "wildlife animal",
    "fish underwater",

    # === OBJECTS ===
    "car vehicle",
    "food meal",
    "furniture",
    "electronics device",
    "book",
    "plant houseplant",
    "clothing fashion",

    # === VISUAL QUALITIES (CLIP-friendly) ===
    "close up shot",
    "wide angle shot",
    "blurry",
    "sharp detailed",
    "bright lighting",
    "dark shadows",
    "colorful vibrant",
    "black and white",
    "shallow depth of field",

    # === ACTIONS/MOTION ===
    "movement motion",
    "still static",
    "flying",
    "swimming",
    "running",
    "dancing",

    # === STYLE ===
    "photograph realistic",
    "cinematic film",
    "amateur home video",
    "professional quality",
]

# Alias for backward compatibility
AI_DETECTION_CONCEPTS = CONTENT_CONCEPTS


def analyze_feature(feat_idx, feature_data, dataset, models, config):
    """
    Analyze a single SAE feature.

    Returns analysis dict with:
    - concept_scores: CLIP similarity to predefined concepts
    - captions: moondream2 descriptions
    - class_distribution: real vs AI label distribution
    - mean_activation: average activation strength
    """
    # Extract frames
    frames = extract_frames_for_feature(feature_data, dataset)

    if not frames:
        return None

    # Get CLIP embeddings
    embeddings = get_clip_embeddings(frames, models, config)

    # Compare to predefined concepts
    concept_scores = get_clip_text_similarity(
        embeddings,
        AI_DETECTION_CONCEPTS,
        models,
        config
    )

    # Generate captions (expensive, limit frames)
    captions = generate_captions(frames, models, config, max_frames=5)

    # Class distribution
    # Label 0 = AI/Fake, Label 1 = Real
    labels = [f['label'] for f in frames]
    real_count = sum(1 for l in labels if l == 1)
    ai_count = sum(1 for l in labels if l == 0)

    # Mean activation
    mean_activation = np.mean([f['activation'] for f in frames])

    # Top concepts
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        'feature_idx': feat_idx,
        'mean_activation': mean_activation,
        'class_distribution': {
            'real': real_count,
            'ai': ai_count,
            'ai_ratio': ai_count / len(labels) if labels else 0,
        },
        'top_concepts': sorted_concepts[:10],
        'all_concept_scores': concept_scores,
        'captions': captions,
        'num_samples': len(frames),
    }


def cluster_features(feature_analyses, n_clusters=10):
    """
    Cluster features based on their concept score profiles.
    """
    # Build feature matrix from concept scores
    feature_ids = []
    feature_vectors = []

    for analysis in feature_analyses:
        if analysis is None:
            continue
        feature_ids.append(analysis['feature_idx'])
        # Create vector from concept scores
        vec = [analysis['all_concept_scores'].get(c, 0) for c in CONTENT_CONCEPTS]
        feature_vectors.append(vec)

    if len(feature_vectors) < n_clusters:
        return None

    feature_matrix = np.array(feature_vectors)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # PCA for visualization
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(feature_matrix)

    return {
        'feature_ids': feature_ids,
        'cluster_labels': cluster_labels.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'pca_coords': coords_2d.tolist(),
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
    }


def generate_feature_label(analysis):
    """
    Generate a human-readable label for a feature based on its analysis.
    """
    if analysis is None:
        return "Unknown"

    # Get top concepts
    top_concepts = analysis['top_concepts'][:3]
    concept_str = ", ".join([c[0] for c in top_concepts])

    # Class bias
    ai_ratio = analysis['class_distribution']['ai_ratio']
    if ai_ratio > 0.7:
        bias = "AI-biased"
    elif ai_ratio < 0.3:
        bias = "Real-biased"
    else:
        bias = "Neutral"

    return f"{bias}: {concept_str}"


def visualize_feature_grid(feat_idx, frames, analysis, output_dir):
    """Create visualization grid for a feature."""
    if not frames or len(frames) < 4:
        return

    n_show = min(8, len(frames))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_show:
            frame = frames[i]
            ax.imshow(frame['image'])
            label_str = "Real" if frame['label'] == 1 else "AI"
            ax.set_title(f"Act: {frame['activation']:.3f}\n{label_str}", fontsize=10)
        ax.axis('off')

    # Add feature info
    label = generate_feature_label(analysis)
    fig.suptitle(f"Feature {feat_idx}: {label}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_{feat_idx:04d}.png'), dpi=150)
    plt.close()


def main():
    """Main feature labeling pipeline."""
    import pickle

    print("="*80)
    print("SAE Feature Labeling Pipeline")
    print("="*80)

    device = CONFIG['device']
    print(f"Using device: {device}")

    # Load all models
    models = load_models(CONFIG)
    sae_config = models['sae_config']

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}")
    dataset = VideoDataset(DATASET_PATH, target_size=512, max_frames=24)
    print(f"Dataset size: {len(dataset)}")

    # Check if activations already exist
    activations_path = os.path.join(OUTPUT_DIR, 'feature_activations.pkl')
    if os.path.exists(activations_path):
        print(f"\nLoading existing activations from {activations_path}")
        with open(activations_path, 'rb') as f:
            feature_activations = pickle.load(f)
        print(f"Loaded activations for {len(feature_activations)} features")
    else:
        # Collect feature activations (streams to disk, memory-bounded)
        feature_activations = collect_feature_activations(models, dataset, CONFIG)
        print(f"Collected activations for {len(feature_activations)} features")

    # Identify most active features
    feature_activity = {
        feat_idx: sum(item['activation'] for item in items)
        for feat_idx, items in feature_activations.items()
    }
    top_features = sorted(feature_activity.items(), key=lambda x: x[1], reverse=True)
    top_features = top_features[:CONFIG['num_features_to_analyze']]

    print(f"\nAnalyzing top {len(top_features)} most active features...")

    # Setup directories and streaming output
    vis_dir = os.path.join(OUTPUT_DIR, 'feature_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    analyses_path = os.path.join(OUTPUT_DIR, 'feature_analyses.jsonl')
    labels_path = os.path.join(OUTPUT_DIR, 'feature_labels.json')

    # Stream analysis results to disk (JSON Lines format)
    feature_labels = {}
    feature_analyses = []  # Keep minimal for clustering

    with open(analyses_path, 'w') as f_analyses:
        for feat_idx, _ in tqdm(top_features, desc="Analyzing features"):
            feature_data = feature_activations[feat_idx]

            # Analyze
            analysis = analyze_feature(feat_idx, feature_data, dataset, models, CONFIG)

            if analysis:
                # Generate label
                label = generate_feature_label(analysis)
                feature_labels[feat_idx] = label
                analysis['label'] = label

                # Write to disk immediately (JSON Lines)
                f_analyses.write(json.dumps(analysis) + '\n')
                f_analyses.flush()

                # Keep minimal data for clustering
                feature_analyses.append({
                    'feature_idx': analysis['feature_idx'],
                    'all_concept_scores': analysis['all_concept_scores'],
                    'class_distribution': analysis['class_distribution'],
                })

                # Visualize
                frames = extract_frames_for_feature(feature_data, dataset, top_k=8)
                visualize_feature_grid(feat_idx, frames, analysis, vis_dir)

                print(f"  Feature {feat_idx}: {label}")

            # Free memory from feature_activations as we go
            del feature_activations[feat_idx]

    print(f"\nAnalyses streamed to {analyses_path}")

    # Save labels
    with open(labels_path, 'w') as f:
        json.dump(feature_labels, f, indent=2)
    print(f"Labels saved to {labels_path}")

    # Cluster features (uses minimal data kept in memory)
    print("\nClustering features...")
    clustering = cluster_features(feature_analyses)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    # Save summary results (full analyses already streamed to jsonl)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'sae_config': sae_config,
        'num_features_analyzed': len(feature_analyses),
        'feature_labels': feature_labels,
        'clustering': convert_numpy(clustering),
        'concepts_used': CONTENT_CONCEPTS,
        'output_files': {
            'analyses': analyses_path,
            'labels': labels_path,
            'visualizations': vis_dir,
        }
    }

    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print(f"Full analyses: {analyses_path}")
    print(f"Visualizations: {vis_dir}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Class bias distribution
    ai_biased = sum(1 for a in feature_analyses if a['class_distribution']['ai_ratio'] > 0.7)
    real_biased = sum(1 for a in feature_analyses if a['class_distribution']['ai_ratio'] < 0.3)
    neutral = len(feature_analyses) - ai_biased - real_biased

    print(f"Feature class bias:")
    print(f"  AI-biased features: {ai_biased}")
    print(f"  Real-biased features: {real_biased}")
    print(f"  Neutral features: {neutral}")

    # Top concepts overall
    concept_totals = defaultdict(float)
    for analysis in feature_analyses:
        for concept, score in analysis['all_concept_scores'].items():
            concept_totals[concept] += score

    print(f"\nTop concepts across all features:")
    sorted_concepts = sorted(concept_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    for concept, score in sorted_concepts:
        print(f"  {concept}: {score:.3f}")

    print("\n" + "="*80)
    print("Feature labeling complete!")
    print("="*80)


if __name__ == "__main__":
    main()
