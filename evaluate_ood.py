"""
Out-of-Distribution (OOD) Evaluation Script

Evaluates the classifier on a held-out dataset and generates:
- Confusion matrix
- Probability score distributions stratified by class
- Classification metrics (accuracy, precision, recall, F1)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import cv2

from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier

# Configuration
MODEL_DIR = '/media/joshua/WD_BLACK/Gen-Video/model'


def load_model(device='cuda'):
    """Load the trained video classifier."""
    print("Loading video classifier...")
    model_path = os.path.join(MODEL_DIR, 'full_classifier_best.pt')

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


def load_video(video_path, target_size=512, max_frames=24):
    """Load and preprocess a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

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
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            frame_np = frame_resized.astype(np.float32) / 255.0
            frames.append(frame_np)

    cap.release()

    if not frames:
        return None

    # Stack frames: shape (T, H, W, C) - matches dataset format
    video_tensor = np.stack(frames, axis=0)
    video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

    # Add batch dimension: (B, T, H, W, C)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor


def collect_videos(data_dir):
    """
    Collect all video paths and their labels from directory.

    Supports two structures:
    1. Folder-based: Real/ and Fake/ subdirectories
    2. Filename-based: real_*.mp4 and ai_*.mp4/fake_*.mp4
    """
    data_path = Path(data_dir)
    videos = []
    generator_counts = {}

    # Check for folder-based structure (Real/ and Fake/ subdirectories)
    real_dir = data_path / 'Real'
    fake_dir = data_path / 'Fake'

    if real_dir.exists() or fake_dir.exists():
        print("Detected folder-based structure (Real/, Fake/)")

        # Collect Real videos
        if real_dir.exists():
            for video_file in real_dir.glob('*.mp4'):
                videos.append((str(video_file), 1, 'Real'))
            generator_counts['Real'] = len(list(real_dir.glob('*.mp4')))

        # Collect Fake videos (may have subdirectories per generator)
        if fake_dir.exists():
            # Check if Fake has direct videos or subdirectories
            direct_fakes = list(fake_dir.glob('*.mp4'))
            if direct_fakes:
                for video_file in direct_fakes:
                    videos.append((str(video_file), 0, 'Fake'))
                generator_counts['Fake'] = len(direct_fakes)

            # Check subdirectories (e.g., Sora/, Gen2/, etc.)
            for subdir in fake_dir.iterdir():
                if subdir.is_dir():
                    generator_name = subdir.name
                    count = 0
                    for video_file in subdir.glob('*.mp4'):
                        videos.append((str(video_file), 0, generator_name))
                        count += 1
                    if count > 0:
                        generator_counts[generator_name] = count

    else:
        # Fallback to filename-based detection
        print("Using filename-based detection (real_*, ai_*, fake_*)")
        for video_file in data_path.glob('**/*.mp4'):
            filename_lower = video_file.name.lower()
            if filename_lower.startswith('real_'):
                videos.append((str(video_file), 1, 'Real'))
            elif filename_lower.startswith(('ai_', 'fake_')):
                videos.append((str(video_file), 0, 'Fake'))

    # Print generator breakdown
    if generator_counts:
        print("\nVideos by source:")
        for gen, count in sorted(generator_counts.items()):
            print(f"  {gen}: {count}")

    return videos


def evaluate_dataset(model, videos, device='cuda'):
    """
    Evaluate model on all videos.

    Returns:
        results dict with predictions, labels, probabilities, paths, generators
    """
    all_labels = []
    all_predictions = []
    all_probs = []  # P(Real)
    all_paths = []
    all_generators = []
    failed_videos = []

    print(f"\nEvaluating {len(videos)} videos...")

    for video_path, label, generator in tqdm(videos, desc="Processing"):
        try:
            video_tensor = load_video(video_path)
            if video_tensor is None:
                failed_videos.append(video_path)
                continue

            video_tensor = video_tensor.to(device)

            with torch.no_grad():
                output = model(video_tensor)
                # 2-class output: Class 0 = AI, Class 1 = Real
                probs = torch.softmax(output, dim=-1)
                prob_real = probs[0, 1].item()
                prediction = 1 if prob_real > 0.5 else 0

            all_labels.append(label)
            all_predictions.append(prediction)
            all_probs.append(prob_real)
            all_paths.append(video_path)
            all_generators.append(generator)

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            failed_videos.append(video_path)
            continue

    if failed_videos:
        print(f"\nFailed to process {len(failed_videos)} videos")

    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probs': np.array(all_probs),
        'paths': all_paths,
        'generators': all_generators,
        'failed': failed_videos,
    }


def plot_results(results, output_dir, generator_stats=None):
    """Generate and save visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    labels = results['labels']
    predictions = results['predictions']
    probs = results['probs']

    # Separate probabilities by true class
    real_probs = probs[labels == 1]
    ai_probs = probs[labels == 0]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['AI/Fake', 'Real'], yticklabels=['AI/Fake', 'Real'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    axes[0, 0].set_title('Confusion Matrix')

    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0, 1],
                xticklabels=['AI/Fake', 'Real'], yticklabels=['AI/Fake', 'Real'])
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('True')
    axes[0, 1].set_title('Normalized Confusion Matrix')

    # 3. Probability Distribution by Class
    axes[1, 0].hist(real_probs, bins=30, alpha=0.7, label='Real Videos', color='green', edgecolor='black')
    axes[1, 0].hist(ai_probs, bins=30, alpha=0.7, label='AI Videos', color='red', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[1, 0].set_xlabel('P(Real) Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Probability Distribution by True Class')
    axes[1, 0].legend()

    # 4. ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_evaluation_results.png'), dpi=150)
    plt.close()

    # Additional: Detailed probability histograms
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Real videos distribution
    axes2[0].hist(real_probs, bins=50, alpha=0.8, color='green', edgecolor='black')
    axes2[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    axes2[0].set_xlabel('P(Real) Score')
    axes2[0].set_ylabel('Count')
    axes2[0].set_title(f'Real Videos (n={len(real_probs)})\nMean: {real_probs.mean():.3f}, Median: {np.median(real_probs):.3f}')

    # AI videos distribution
    axes2[1].hist(ai_probs, bins=50, alpha=0.8, color='red', edgecolor='black')
    axes2[1].axvline(x=0.5, color='green', linestyle='--', linewidth=2)
    axes2[1].set_xlabel('P(Real) Score')
    axes2[1].set_ylabel('Count')
    axes2[1].set_title(f'AI Videos (n={len(ai_probs)})\nMean: {ai_probs.mean():.3f}, Median: {np.median(ai_probs):.3f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_probability_distributions.png'), dpi=150)
    plt.close()

    # Per-generator accuracy bar chart
    if generator_stats and len(generator_stats) > 1:
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # Separate real and AI generators
        real_stats = [s for s in generator_stats if s['is_real']]
        ai_stats = sorted([s for s in generator_stats if not s['is_real']], key=lambda x: x['accuracy'], reverse=True)

        all_stats = real_stats + ai_stats
        names = [s['name'] for s in all_stats]
        accuracies = [s['accuracy'] * 100 for s in all_stats]
        colors = ['green' if s['is_real'] else 'red' for s in all_stats]

        bars = ax3.bar(names, accuracies, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_xlabel('Video Source')
        ax3.set_title('Classification Accuracy by Video Source')
        ax3.set_ylim(0, 105)
        plt.xticks(rotation=45, ha='right')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Real'),
                         Patch(facecolor='red', alpha=0.7, label='AI Generated')]
        ax3.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ood_per_generator_accuracy.png'), dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def print_metrics(results):
    """Print classification metrics."""
    labels = results['labels']
    predictions = results['predictions']
    probs = results['probs']
    generators = results['generators']

    print("\n" + "=" * 70)
    print("OOD EVALUATION RESULTS")
    print("=" * 70)

    # Basic counts
    n_real = (labels == 1).sum()
    n_ai = (labels == 0).sum()
    print(f"\nDataset composition:")
    print(f"  Real videos: {n_real}")
    print(f"  AI videos: {n_ai}")
    print(f"  Total: {len(labels)}")

    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    # Per-class metrics
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['AI/Fake', 'Real']))

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(f"                Predicted AI  Predicted Real")
    print(f"  Actual AI     {cm[0, 0]:>10}  {cm[0, 1]:>14}")
    print(f"  Actual Real   {cm[1, 0]:>10}  {cm[1, 1]:>14}")

    # Probability statistics
    real_probs = probs[labels == 1]
    ai_probs = probs[labels == 0]

    print("\nProbability Statistics:")
    print(f"  Real videos - Mean P(Real): {real_probs.mean():.3f}, Std: {real_probs.std():.3f}")
    print(f"  AI videos   - Mean P(Real): {ai_probs.mean():.3f}, Std: {ai_probs.std():.3f}")

    # ROC AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.3f}")

    # Error analysis
    false_positives = ((predictions == 1) & (labels == 0)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()
    print(f"\nError Analysis:")
    print(f"  False Positives (AI classified as Real): {false_positives}")
    print(f"  False Negatives (Real classified as AI): {false_negatives}")

    # Per-generator accuracy breakdown
    print("\n" + "-" * 70)
    print("PER-GENERATOR ACCURACY")
    print("-" * 70)

    generators_arr = np.array(generators)
    unique_generators = sorted(set(generators))

    generator_stats = []
    for gen in unique_generators:
        mask = generators_arr == gen
        gen_labels = labels[mask]
        gen_preds = predictions[mask]
        gen_probs = probs[mask]

        if len(gen_labels) > 0:
            gen_acc = accuracy_score(gen_labels, gen_preds)
            gen_mean_prob = gen_probs.mean()
            is_real = gen_labels[0] == 1  # All same class per generator

            generator_stats.append({
                'name': gen,
                'count': len(gen_labels),
                'accuracy': gen_acc,
                'mean_prob': gen_mean_prob,
                'is_real': is_real,
            })

    # Print Real first, then AI generators sorted by accuracy
    real_stats = [s for s in generator_stats if s['is_real']]
    ai_stats = sorted([s for s in generator_stats if not s['is_real']], key=lambda x: x['accuracy'])

    print(f"\n{'Generator':<20} {'Count':>8} {'Accuracy':>10} {'Mean P(Real)':>14}")
    print("-" * 55)

    for stats in real_stats:
        print(f"{stats['name']:<20} {stats['count']:>8} {stats['accuracy']:>10.1%} {stats['mean_prob']:>14.3f}")

    if ai_stats:
        print("-" * 55)
        for stats in ai_stats:
            print(f"{stats['name']:<20} {stats['count']:>8} {stats['accuracy']:>10.1%} {stats['mean_prob']:>14.3f}")

    print("\n" + "=" * 70)

    return generator_stats


def main():
    parser = argparse.ArgumentParser(description='OOD Evaluation for Video Classifier')
    parser.add_argument('--data_dir', type=str,
                        default='/media/joshua/WD_BLACK/Gen-Video/GenVideo-Val/GenVideo-Val',
                        help='Directory containing videos to evaluate')
    parser.add_argument('--output_dir', type=str,
                        default='/media/joshua/WD_BLACK/Gen-Video/ood_evaluation',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Collect videos
    print(f"\nScanning directory: {args.data_dir}")
    videos = collect_videos(args.data_dir)
    print(f"Found {len(videos)} labeled videos")

    if len(videos) == 0:
        print("No videos found! Check that filenames start with 'real_' or 'ai_'/'fake_'")
        sys.exit(1)

    # Count by class
    n_real = sum(1 for _, label, _ in videos if label == 1)
    n_ai = sum(1 for _, label, _ in videos if label == 0)
    print(f"  Real: {n_real}, AI/Fake: {n_ai}")

    # Evaluate
    results = evaluate_dataset(model, videos, device)

    # Print metrics
    generator_stats = print_metrics(results)

    # Plot results
    plot_results(results, args.output_dir, generator_stats)

    # Save detailed results
    import json
    results_file = os.path.join(args.output_dir, 'ood_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'accuracy': float(accuracy_score(results['labels'], results['predictions'])),
            'n_samples': len(results['labels']),
            'n_real': int((results['labels'] == 1).sum()),
            'n_ai': int((results['labels'] == 0).sum()),
            'mean_prob_real': float(results['probs'][results['labels'] == 1].mean()),
            'mean_prob_ai': float(results['probs'][results['labels'] == 0].mean()),
            'failed_videos': results['failed'],
        }, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
