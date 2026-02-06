"""
Large-scale forensic classifier training script.

Samples videos from multiple generators and trains a Bayesian forensic classifier.
"""

import os
import sys
import random
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_forensics import CameraForensics
from forensic_classifier import ForensicClassifier, print_evaluation_report


# Configuration - can be overridden via command line
TRAIN_DIR = '/media/joshua/WD_BLACK/Gen-Video/GenVideo-Train'
OUTPUT_DIR = '/media/joshua/WD_BLACK/Gen-Video/model'
SAMPLES_PER_GENERATOR = 10000
MAX_FRAMES = 1
N_WORKERS = 14  # Parallel feature extraction


def collect_video_paths(base_dir, samples_per_generator=10000):
    """
    Collect video paths from training directory.

    Returns:
        ai_videos: list of (path, generator_name) tuples
        real_videos: list of paths
    """
    base_path = Path(base_dir)

    ai_videos = []
    real_videos = []

    # AI-Generated videos
    ai_dir = base_path / 'AI-Generated'
    if ai_dir.exists():
        print("\nCollecting AI-generated videos...")
        for generator_dir in sorted(ai_dir.iterdir()):
            if generator_dir.is_dir():
                generator_name = generator_dir.name
                videos = list(generator_dir.glob('**/*.mp4'))

                if len(videos) == 0:
                    print(f"  {generator_name}: 0 videos (skipping)")
                    continue

                # Sample randomly
                if len(videos) > samples_per_generator:
                    sampled = random.sample(videos, samples_per_generator)
                else:
                    sampled = videos

                print(f"  {generator_name}: {len(videos)} total, sampled {len(sampled)}")

                for v in sampled:
                    ai_videos.append((str(v), generator_name))

    # Real videos
    real_dir = base_path / 'Real'
    if real_dir.exists():
        print("\nCollecting real videos...")
        all_real = []
        for source_dir in sorted(real_dir.iterdir()):
            if source_dir.is_dir():
                source_name = source_dir.name
                videos = list(source_dir.glob('**/*.mp4'))
                print(f"  {source_name}: {len(videos)} videos")
                all_real.extend(videos)

        # Match AI count
        target_count = len(ai_videos)
        if len(all_real) > target_count:
            real_videos = random.sample(all_real, target_count)
        else:
            real_videos = all_real

        print(f"  Total real sampled: {len(real_videos)} (to match {len(ai_videos)} AI)")

    return ai_videos, [str(v) for v in real_videos]


def extract_features_single(args):
    """Extract features from a single video (for parallel processing)."""
    video_path, label, generator = args

    try:
        forensics = CameraForensics()
        results = forensics.analyze_video(video_path, max_frames=MAX_FRAMES)
        features, names = forensics.get_feature_vector(results)
        return features, names, label, generator, None
    except Exception as e:
        return None, None, label, generator, str(e)


def extract_features_batch(video_list, n_workers=4):
    """
    Extract features from videos in parallel.

    Args:
        video_list: list of (path, label, generator) tuples
        n_workers: number of parallel workers

    Returns:
        X: feature matrix
        y: labels
        generators: generator names
        feature_names: list of feature names
    """
    all_features = []
    all_labels = []
    all_generators = []
    feature_names = None
    failed = 0

    print(f"\nExtracting features from {len(video_list)} videos using {n_workers} workers...")

    # Process in chunks to avoid memory issues
    chunk_size = 100

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(video_list), chunk_size):
            chunk = video_list[chunk_start:chunk_start + chunk_size]

            futures = {executor.submit(extract_features_single, item): item for item in chunk}

            for future in tqdm(as_completed(futures), total=len(chunk),
                              desc=f"Chunk {chunk_start//chunk_size + 1}"):
                features, names, label, generator, error = future.result()

                if features is not None:
                    if feature_names is None:
                        feature_names = names
                    all_features.append(features)
                    all_labels.append(label)
                    all_generators.append(generator)
                else:
                    failed += 1

    print(f"\nExtracted features from {len(all_features)} videos ({failed} failed)")

    X = np.array(all_features)
    y = np.array(all_labels)

    return X, y, all_generators, feature_names


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("FORENSIC CLASSIFIER TRAINING")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect videos
    print(f"\nData directory: {TRAIN_DIR}")
    print(f"Samples per generator: {SAMPLES_PER_GENERATOR}")

    ai_videos, real_videos = collect_video_paths(TRAIN_DIR, SAMPLES_PER_GENERATOR)

    print(f"\nTotal videos to process:")
    print(f"  AI-generated: {len(ai_videos)}")
    print(f"  Real: {len(real_videos)}")
    print(f"  Total: {len(ai_videos) + len(real_videos)}")

    # Create unified video list with labels
    # Label: 0 = AI, 1 = Real
    video_list = []
    for path, generator in ai_videos:
        video_list.append((path, 0, generator))
    for path in real_videos:
        video_list.append((path, 1, 'Real'))

    # Shuffle
    random.shuffle(video_list)

    # Extract features
    X, y, generators, feature_names = extract_features_batch(video_list, n_workers=N_WORKERS)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels: {(y == 1).sum()} Real, {(y == 0).sum()} AI")

    # Save features for later use
    features_path = os.path.join(OUTPUT_DIR, 'forensic_features.pkl')
    print(f"\nSaving features to {features_path}")
    with open(features_path, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'generators': generators,
            'feature_names': feature_names,
        }, f)

    # Split into train/test
    n_samples = len(y)
    n_test = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    generators_test = [generators[i] for i in test_idx]

    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # Train classifier
    print("\n" + "=" * 70)
    print("Training XGBoost Bayesian Ensemble...")
    print("=" * 70)

    clf = ForensicClassifier(
        model_type='xgb_ensemble',
        n_ensemble=30,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        confidence_threshold=0.6,
        abstain_on_uncertain=True
    )

    clf.fit(X_train, y_train, feature_names=feature_names)

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    metrics = clf.evaluate(X_test, y_test)
    print_evaluation_report(metrics, "Test Set Results")

    # Per-generator accuracy
    print("\n" + "-" * 70)
    print("PER-GENERATOR ACCURACY")
    print("-" * 70)

    results = clf.predict_with_uncertainty(X_test)
    predictions = np.array([r.prediction for r in results])

    generator_set = sorted(set(generators_test))
    print(f"\n{'Generator':<20} {'Count':>8} {'Accuracy':>10} {'Abstained':>10}")
    print("-" * 55)

    for gen in generator_set:
        mask = np.array([g == gen for g in generators_test])
        gen_y = y_test[mask]
        gen_pred = predictions[mask]

        # Filter out abstentions
        valid = gen_pred != -1
        if valid.sum() > 0:
            acc = accuracy_score(gen_y[valid], gen_pred[valid])
            abstained = (~valid).sum()
            print(f"{gen:<20} {mask.sum():>8} {acc:>10.1%} {abstained:>10}")

    # Feature importance
    print("\n" + "-" * 70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("-" * 70)

    importance = clf.get_feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:20]):
        print(f"  {i+1:2}. {name:<45} {imp:.4f}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'forensic_classifier.pkl')
    clf.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, 'forensic_training_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'feature_importance': importance,
            'feature_names': feature_names,
            'n_train': len(y_train),
            'n_test': len(y_test),
        }, f)

    print(f"Results saved to {results_path}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train forensic classifier on large dataset')
    parser.add_argument('--samples_per_gen', type=int, default=10000,
                       help='Samples per generator (default: 10000)')
    parser.add_argument('--max_frames', type=int, default=1,
                       help='Max frames per video (default: 1)')
    parser.add_argument('--workers', type=int, default=14,
                       help='Parallel workers (default: 14)')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR,
                       help='Training data directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for model')

    args = parser.parse_args()

    # Update globals
    SAMPLES_PER_GENERATOR = args.samples_per_gen
    MAX_FRAMES = args.max_frames
    N_WORKERS = args.workers
    TRAIN_DIR = args.train_dir
    OUTPUT_DIR = args.output_dir

    # Import here to avoid issues with multiprocessing
    from sklearn.metrics import accuracy_score
    main()
