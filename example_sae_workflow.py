"""
Complete example workflow for SAE interpretability analysis.
This script demonstrates the full pipeline from training to interpretation.
"""

import os
import sys
import torch
from datetime import datetime

# Configuration
BASE_DIR = '/media/joshua/WD_BLACK/Gen-Video'
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SAE_DIR = os.path.join(MODEL_DIR, 'sae')
RESULTS_DIR = os.path.join(BASE_DIR, 'sae_example_results')

# Create directories
os.makedirs(SAE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def check_prerequisites():
    """Check if all required files and directories exist."""
    print("Checking prerequisites...")

    issues = []

    # Check for trained classifier
    classifier_paths = [
        os.path.join(MODEL_DIR, 'full_classifier_best.pt'),
        os.path.join(MODEL_DIR, 'full_classifier_1_85.pt')
    ]

    classifier_exists = any(os.path.exists(p) for p in classifier_paths)
    if not classifier_exists:
        issues.append("❌ No trained classifier found in model directory")
    else:
        print("✓ Trained classifier found")

    # Check for dataset
    dataset_path = os.path.join(BASE_DIR, 'dataset')
    if not os.path.exists(dataset_path):
        issues.append(f"❌ Dataset directory not found: {dataset_path}")
    else:
        print("✓ Dataset directory found")

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, will use CPU (slower)")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("\n✓ All prerequisites met!\n")
    return True


def step1_train_sae():
    """Step 1: Train the Sparse Autoencoder."""
    print("="*80)
    print("STEP 1: Training Sparse Autoencoder")
    print("="*80)
    print("\nThis will take approximately 20-30 minutes...")
    print("The SAE will learn interpretable features from the classifier's activations.\n")

    # Check if SAE already exists
    sae_path = os.path.join(SAE_DIR, 'sae_best.pt')
    if os.path.exists(sae_path):
        response = input("SAE already exists. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training, using existing SAE.\n")
            return True

    try:
        print("Starting SAE training...")
        import train_sae
        train_sae.main()
        print("\n✓ SAE training completed successfully!\n")
        return True
    except Exception as e:
        print(f"\n❌ Error during SAE training: {e}")
        print("Check the error message and try again.\n")
        return False


def step2_visualize_features():
    """Step 2: Visualize learned features."""
    print("="*80)
    print("STEP 2: Visualizing Learned Features")
    print("="*80)
    print("\nGenerating feature visualizations...\n")

    sae_path = os.path.join(SAE_DIR, 'sae_best.pt')
    if not os.path.exists(sae_path):
        print("❌ No trained SAE found. Please run Step 1 first.\n")
        return False

    try:
        from visualize_sae_features import SAEFeatureVisualizer, SparseAutoencoder
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load SAE
        print("Loading SAE...")
        checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
        config = checkpoint['config']

        sae = SparseAutoencoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            sparsity_coefficient=config['sparsity_coefficient'],
            tie_weights=config['tie_weights']
        )
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(device)

        visualizer = SAEFeatureVisualizer(sae, device)

        # Generate sample visualizations
        print("Creating sample feature visualizations...")

        # Create dummy activations for demonstration
        dummy_activations = torch.randn(100, 50, config['input_dim'])

        # Analyze features
        feature_metrics = visualizer.analyze_feature_interpretability(dummy_activations, top_k=5)

        print("\n✓ Top 5 Features:")
        for idx, metrics in sorted(feature_metrics.items(),
                                   key=lambda x: x[1]['activation_frequency'],
                                   reverse=True)[:5]:
            print(f"\nFeature {idx}:")
            print(f"  Activation frequency: {metrics['activation_frequency']:.2%}")
            print(f"  Mean activation: {metrics['mean_activation']:.4f}")
            print(f"  Max activation: {metrics['max_activation']:.4f}")

            # Create dashboard for this feature
            dashboard_dir = os.path.join(RESULTS_DIR, f'feature_{idx}')
            visualizer.create_feature_dashboard(
                dummy_activations,
                idx,
                dashboard_dir
            )

        print(f"\n✓ Feature visualizations saved to: {RESULTS_DIR}\n")
        return True

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_interpret_video():
    """Step 3: Interpret a video using SAE."""
    print("="*80)
    print("STEP 3: Interpreting Video with SAE")
    print("="*80)
    print("\nAnalyzing a random video from the dataset...\n")

    sae_path = os.path.join(SAE_DIR, 'sae_best.pt')
    if not os.path.exists(sae_path):
        print("❌ No trained SAE found. Please run Step 1 first.\n")
        return False

    try:
        from interpret_with_sae import (
            load_model_correctly, load_sae,
            analyze_video_with_sae
        )
        from interpret import select_random_video

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load classifier
        print("Loading classifier...")
        model_path = os.path.join(MODEL_DIR, 'full_classifier_best.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, 'full_classifier_1_85.pt')

        model = load_model_correctly(model_path, device)

        # Load SAE
        print("Loading SAE...")
        sae = load_sae(sae_path, device)

        # Select and analyze video
        print("\nSelecting random video...")
        video_path = select_random_video()

        analysis_dir = os.path.join(RESULTS_DIR, 'video_analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        print(f"Analyzing: {os.path.basename(video_path)}")

        results = analyze_video_with_sae(
            video_path,
            model,
            sae,
            device,
            analysis_dir
        )

        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print(f"\nVideo: {os.path.basename(results['video_path'])}")
        print(f"Predicted: {results['predicted_class']} ({results['confidence']:.1%} confidence)")
        print(f"True Label: {results['true_class']}")
        print(f"\nSAE Metrics:")
        print(f"  Active features (L0): {results['l0_norm']:.1f}")
        print(f"  Feature density: {results['feature_density']:.1%}")
        print(f"  Reconstruction loss: {results['reconstruction_loss']:.4f}")

        print(f"\nTop 5 Active Features:")
        for i, (feat_idx, activation) in enumerate(zip(
            results['top_features'][:5],
            results['top_feature_activations'][:5]
        )):
            print(f"  {i+1}. Feature {feat_idx}: {activation:.3f}")

        print(f"\n✓ Full analysis saved to: {analysis_dir}\n")
        return True

    except Exception as e:
        print(f"\n❌ Error during interpretation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete SAE workflow."""
    print("\n" + "="*80)
    print("SAE INTERPRETABILITY - COMPLETE WORKFLOW")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    # Check prerequisites
    if not check_prerequisites():
        print("Please resolve the issues above before continuing.")
        sys.exit(1)

    # Menu
    print("What would you like to do?")
    print("  1. Complete workflow (all steps)")
    print("  2. Train SAE only (Step 1)")
    print("  3. Visualize features only (Step 2)")
    print("  4. Interpret video only (Step 3)")
    print("  5. Steps 2 & 3 (skip training)")

    choice = input("\nEnter choice (1-5): ").strip()

    success = True

    if choice == '1':
        # Complete workflow
        success = step1_train_sae()
        if success:
            success = step2_visualize_features()
        if success:
            success = step3_interpret_video()

    elif choice == '2':
        success = step1_train_sae()

    elif choice == '3':
        success = step2_visualize_features()

    elif choice == '4':
        success = step3_interpret_video()

    elif choice == '5':
        success = step2_visualize_features()
        if success:
            success = step3_interpret_video()

    else:
        print("Invalid choice.")
        sys.exit(1)

    # Summary
    print("\n" + "="*80)
    if success:
        print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nAll results saved to: {RESULTS_DIR}")
        print("\nNext steps:")
        print("  - Review visualizations in the results directory")
        print("  - Analyze more videos by running Step 3 again")
        print("  - Compare fake vs. real features")
        print("  - Read SAE_INTERPRETABILITY.md for advanced usage")
    else:
        print("❌ WORKFLOW INCOMPLETE")
        print("="*80)
        print("\nSome steps failed. Please check the error messages above.")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
