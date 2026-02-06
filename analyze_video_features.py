"""
Analyze which SAE features activate for a given video.

Usage:
    python analyze_video_features.py /path/to/video.mp4
    python analyze_video_features.py /path/to/video.mp4 --top_k 20
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import cv2

# Local imports
from sparse_autoencoder import SparseAutoencoder
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from dataset import VideoDataset

# Configuration
BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SAE_DIR = os.path.join(MODEL_DIR, 'sae')
FEATURE_ANALYSIS_DIR = os.path.join(BASE_DIR, 'feature_analysis')


def load_models(device='cuda'):
    """Load the video classifier and SAE."""
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

    # Load feature labels if available
    labels_path = os.path.join(FEATURE_ANALYSIS_DIR, 'feature_labels.json')
    feature_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            feature_labels = json.load(f)
        # Convert string keys to int
        feature_labels = {int(k): v for k, v in feature_labels.items()}

    return video_model, sae, feature_labels


def load_video(video_path, target_size=512, max_frames=24):
    """Load and preprocess a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            frame_np = frame_resized.astype(np.float32) / 255.0
            frames.append(frame_np)

    cap.release()

    if not frames:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Stack frames: shape (T, H, W, C) - matches dataset format
    video_tensor = np.stack(frames, axis=0)
    video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

    # Add batch dimension: (B, T, H, W, C)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor


class ActivationCollector:
    """Collects activations from a specific layer."""

    def __init__(self):
        self.activations = None
        self.hook = None

    def register_hook(self, module):
        def hook_fn(module, input, output):
            self.activations = output.detach()
        self.hook = module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook:
            self.hook.remove()


def analyze_video(video_path, video_model, sae, feature_labels, device='cuda', top_k=10):
    """
    Analyze a video and return its top activated SAE features.

    Returns:
        dict with:
        - prediction: model's prediction (0=real, 1=AI)
        - confidence: prediction confidence
        - top_features: list of (feature_idx, activation, label) tuples
        - per_frame_features: per-frame breakdown
    """
    # Load video
    print(f"Loading video: {video_path}")
    video_tensor = load_video(video_path)
    video_tensor = video_tensor.to(device)

    # Setup activation hook
    collector = ActivationCollector()
    target_layer = video_model.classifier.transformer_encoder.layers[-1]
    collector.register_hook(target_layer)

    # Forward pass
    with torch.no_grad():
        output = video_model(video_tensor)
        # Handle both single logit and 2-class output
        if output.numel() == 1:
            prob = torch.sigmoid(output).item()
        else:
            # 2-class output: use softmax and take Real class probability
            # Class 0 = AI/Fake, Class 1 = Real
            probs = torch.softmax(output, dim=-1)
            prob = probs[0, 1].item()  # Class 1 = Real
        prediction = 1 if prob > 0.5 else 0

        # Get transformer activations
        transformer_out = collector.activations  # (1, seq_len, hidden_dim)

        # Forward through SAE
        _, features, _ = sae(transformer_out.reshape(-1, transformer_out.shape[-1]))
        # features: (seq_len, sae_hidden_dim)

        seq_len = transformer_out.shape[1]
        features = features.reshape(1, seq_len, -1)  # (1, seq_len, sae_hidden_dim)

    collector.remove_hook()

    # Analyze feature activations
    features_np = features[0].cpu().numpy()  # (seq_len, sae_hidden_dim)

    # Mean activation per feature across all frames
    mean_activations = features_np.mean(axis=0)

    # Max activation per feature (peak activation)
    max_activations = features_np.max(axis=0)

    # Get top-k features by mean activation
    top_indices = np.argsort(mean_activations)[::-1][:top_k]

    top_features = []
    for idx in top_indices:
        feat_idx = int(idx)
        mean_act = float(mean_activations[idx])
        max_act = float(max_activations[idx])
        label = feature_labels.get(feat_idx, "Unknown")
        top_features.append({
            'feature_idx': feat_idx,
            'mean_activation': mean_act,
            'max_activation': max_act,
            'label': label,
        })

    # Per-frame analysis (which features are most active per frame)
    per_frame = []
    for t in range(seq_len):
        frame_features = features_np[t]
        top_frame_indices = np.argsort(frame_features)[::-1][:5]
        frame_top = []
        for idx in top_frame_indices:
            feat_idx = int(idx)
            act = float(frame_features[idx])
            label = feature_labels.get(feat_idx, "Unknown")
            frame_top.append({
                'feature_idx': feat_idx,
                'activation': act,
                'label': label,
            })
        per_frame.append(frame_top)

    return {
        'video_path': video_path,
        'prediction': 'Real' if prediction == 1 else 'AI-generated',
        'confidence': prob if prediction == 1 else 1 - prob,
        'raw_score': prob,
        'top_features': top_features,
        'per_frame_features': per_frame,
        'num_frames_analyzed': seq_len,
    }


def print_analysis(result):
    """Pretty print the analysis results."""
    print("\n" + "=" * 70)
    print("VIDEO ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nVideo: {result['video_path']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Raw score (Real probability): {result['raw_score']:.4f}")
    print(f"Frames analyzed: {result['num_frames_analyzed']}")

    print("\n" + "-" * 70)
    print("TOP ACTIVATED SAE FEATURES")
    print("-" * 70)
    print(f"{'Rank':<6}{'Feature':<10}{'Mean Act':<12}{'Max Act':<12}{'Label'}")
    print("-" * 70)

    for i, feat in enumerate(result['top_features'], 1):
        print(f"{i:<6}{feat['feature_idx']:<10}{feat['mean_activation']:<12.4f}{feat['max_activation']:<12.4f}{feat['label']}")

    print("\n" + "-" * 70)
    print("PER-FRAME TOP FEATURES (first 5 frames)")
    print("-" * 70)

    for t, frame_feats in enumerate(result['per_frame_features'][:5]):
        print(f"\nFrame {t}:")
        for feat in frame_feats[:3]:
            print(f"  Feature {feat['feature_idx']}: {feat['activation']:.4f} - {feat['label']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze SAE features for a video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top features to show')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    video_model, sae, feature_labels = load_models(device)
    print(f"Loaded {len(feature_labels)} feature labels")

    # Analyze video
    result = analyze_video(
        args.video_path,
        video_model,
        sae,
        feature_labels,
        device=device,
        top_k=args.top_k
    )

    if args.json:
        # JSON output (exclude per_frame for brevity)
        output = {k: v for k, v in result.items() if k != 'per_frame_features'}
        print(json.dumps(output, indent=2))
    else:
        print_analysis(result)


if __name__ == "__main__":
    main()
