"""
Confusion matrix evaluation for FullVideoClassifier.

Usage:
    python confusion_matrix_eval.py
    python confusion_matrix_eval.py --checkpoint model/checkpoint_epoch_0004.pt
    python confusion_matrix_eval.py --dataset F:/Gen-Video/dataset --batch-size 4

Note: train_test_split.json was not recovered from the training run, so this
evaluates on the full dataset. Metrics will be slightly optimistic for training
videos but gives a solid overall picture of model performance.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import VideoDataset, custom_collate_fn
from full_scale_classifier import FullVideoClassifier


def load_model(checkpoint_path: str, device: torch.device) -> FullVideoClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullVideoClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", "?")
    print(f"Loaded checkpoint: epoch {epoch}, val_accuracy={val_acc}")
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels = []
    all_preds = []
    all_probs = []
    failed = 0

    for batch_idx, (videos, labels, paths) in enumerate(loader):
        videos = videos.to(device)
        labels = labels.to(device)

        logits = model(videos)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class 1 (Real)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * loader.batch_size} videos...")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, output_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    class_names = ["AI-Generated", "Real"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix (counts)")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp_norm.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix (normalized)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {output_path}")
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/checkpoint_epoch_0004.pt")
    parser.add_argument("--dataset", default="F:/Gen-Video/dataset")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"\nScanning dataset at {args.dataset}")
    dataset = VideoDataset(args.dataset, target_size=512, max_frames=24)

    if len(dataset) == 0:
        raise RuntimeError(f"No labeled videos found in {args.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
    )

    print(f"\nRunning inference on {len(dataset)} videos...")
    labels, preds, probs = run_inference(model, loader, device)

    # Metrics
    cm = plot_confusion_matrix(labels, preds, args.output)
    accuracy = (labels == preds).mean() * 100
    auc = roc_auc_score(labels, probs)

    print(f"\n{'='*50}")
    print(f"Overall Accuracy : {accuracy:.2f}%")
    print(f"ROC AUC          : {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=["AI-Generated", "Real"]))
    print(f"Confusion Matrix (raw):")
    print(f"  {'':15s}  Pred AI  Pred Real")
    print(f"  {'True AI':15s}  {cm[0,0]:7d}  {cm[0,1]:9d}")
    print(f"  {'True Real':15s}  {cm[1,0]:7d}  {cm[1,1]:9d}")

    # Save results JSON
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "total_videos": int(len(labels)),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "class_names": ["AI-Generated", "Real"],
    }
    results_path = args.output.replace(".png", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
