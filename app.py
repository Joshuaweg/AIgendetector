"""
Full-featured web application for AI-Generated Video Detection
Provides upload, prediction, and XAI visualization capabilities
"""

import gradio as gr
import torch
import cv2
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image

# Import model components
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from interpret import load_video, load_model_correctly, visualize, save_attributions_video

# Global model instance
model = None
device = None

def initialize_model(model_path=None):
    """Initialize the model once at startup"""
    global model, device

    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on device: {device}")

        # Use provided path or default
        if model_path is None:
            base_dir = '/media/joshua/WD_BLACK/Gen-Video'
            model_path = os.path.join(base_dir, 'model', 'full_classifier_best.pt')

        # Check if model exists locally
        if not os.path.exists(model_path):
            model_path = 'model/full_classifier_best.pt'  # Try relative path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please place your trained model at this location."
            )

        model = load_model_correctly(model_path, device)
        model.eval()
        print("Model loaded successfully!")

    return model, device

def preprocess_video_from_upload(video_file):
    """Process uploaded video file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    try:
        # Load video using existing function
        video_tensor, frames, _, _ = load_video(tmp_path)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        return video_tensor, frames, tmp_path
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")
    finally:
        # Cleanup will happen after processing
        pass

def predict_video(video_file, show_explanations=True):
    """
    Main prediction function with optional XAI

    Args:
        video_file: Uploaded video file
        show_explanations: Whether to generate attribution visualizations

    Returns:
        Tuple of (prediction_text, confidence_plot, attribution_video or None)
    """
    global model, device

    if model is None:
        initialize_model()

    try:
        # Process video
        video_tensor, frames, tmp_path = preprocess_video_from_upload(video_file)
        video_tensor = video_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[pred].item()

        classes = ['AI-Generated', 'Real']
        prediction_text = (
            f"### Prediction: **{classes[pred]}**\n\n"
            f"**Confidence:** {confidence:.2%}\n\n"
            f"**Probabilities:**\n"
            f"- AI-Generated: {probs[0]:.2%}\n"
            f"- Real: {probs[1]:.2%}"
        )

        # Create confidence plot
        fig = create_confidence_plot(probs.cpu().numpy(), classes)

        # Generate explanations if requested
        attribution_video = None
        if show_explanations:
            attribution_video = generate_attributions(
                video_tensor, frames, pred, model
            )

        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

        return prediction_text, fig, attribution_video

    except Exception as e:
        error_msg = f"### Error during prediction\n\n{str(e)}"
        return error_msg, None, None

def create_confidence_plot(probabilities, class_names):
    """Create a bar chart showing class probabilities"""
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    colors = ['#FF6B6B' if probabilities[i] > 0.5 else '#4ECDC4'
              for i in range(len(probabilities))]

    bars = ax.bar(class_names, probabilities, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}',
                ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    return fig

def generate_attributions(video_tensor, frames, pred_class, model):
    """Generate attribution visualizations using Integrated Gradients"""
    from captum.attr import IntegratedGradients

    # Create temporary directory for plots
    temp_dir = tempfile.mkdtemp()

    try:
        # Enable gradients
        video_tensor.requires_grad = True

        # Create baseline
        baseline = torch.zeros_like(video_tensor)

        # Initialize IG
        ig = IntegratedGradients(model)

        # Calculate attributions
        attributions, delta = ig.attribute(
            video_tensor,
            baseline,
            target=pred_class,
            return_convergence_delta=True,
            n_steps=50,
            internal_batch_size=1
        )

        # Visualize attributions
        visualize(
            attributions,
            torch.tensor(frames),
            save_path=temp_dir,
            label=f"Attribution Analysis (Class: {['AI-Generated', 'Real'][pred_class]})"
        )

        # Create video from frames
        output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        save_attributions_video(temp_dir, output_video, fps=8)

        return output_video

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def analyze_frame_importance(video_file):
    """Analyze temporal importance of frames"""
    global model, device

    if model is None:
        initialize_model()

    try:
        from captum.attr import IntegratedGradients

        # Process video
        video_tensor, frames, tmp_path = preprocess_video_from_upload(video_file)
        video_tensor = video_tensor.to(device)
        video_tensor.requires_grad = True

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor)
            pred = torch.argmax(output, dim=1).item()

        # Calculate attributions
        baseline = torch.zeros_like(video_tensor)
        ig = IntegratedGradients(model)

        attributions, _ = ig.attribute(
            video_tensor,
            baseline,
            target=pred,
            n_steps=50
        )

        # Compute frame importance (aggregate over spatial dimensions)
        frame_importance = attributions.abs().mean(dim=[0, 2, 3, 4]).cpu().numpy()

        # Normalize
        frame_importance = (frame_importance - frame_importance.min()) / \
                          (frame_importance.max() - frame_importance.min() + 1e-8)

        # Create plot
        fig = Figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        frames_range = list(range(len(frame_importance)))
        ax.plot(frames_range, frame_importance, linewidth=2, marker='o', markersize=4)
        ax.fill_between(frames_range, frame_importance, alpha=0.3)

        # Mark key frames
        threshold = frame_importance.mean() + frame_importance.std()
        key_frames = np.where(frame_importance > threshold)[0]
        if len(key_frames) > 0:
            ax.scatter(key_frames, frame_importance[key_frames],
                      color='red', s=100, zorder=5, label='Key Frames')

        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title('Temporal Frame Importance Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

        # Create summary text
        summary = (
            f"### Frame Importance Analysis\n\n"
            f"**Key Frames:** {', '.join(map(str, key_frames.tolist()))}\n\n"
            f"**Most Important Frame:** {frame_importance.argmax()}\n\n"
            f"**Average Importance:** {frame_importance.mean():.3f}\n\n"
            f"**Temporal Variance:** {frame_importance.std():.3f}"
        )

        return fig, summary

    except Exception as e:
        error_msg = f"### Error during analysis\n\n{str(e)}"
        return None, error_msg

def create_app():
    """Create the Gradio interface"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    .gr-box {
        border-radius: 10px;
    }
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="AI Video Detector") as app:

        gr.Markdown(
            """
            # 🎥 AI-Generated Video Detector
            ### Deep Learning System for Detecting AI-Generated Videos

            Upload a video to analyze whether it's AI-generated or real. The system uses a hybrid CNN-Transformer
            architecture with 85.12% accuracy and provides explainable AI visualizations.
            """
        )

        with gr.Tabs():
            # Tab 1: Quick Prediction
            with gr.TabItem("🔍 Quick Prediction"):
                gr.Markdown("### Upload a video to get instant predictions")

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_quick = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )

                        show_explanations = gr.Checkbox(
                            label="Generate Attribution Visualizations (slower)",
                            value=False,
                            info="Creates frame-by-frame attribution analysis"
                        )

                        predict_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        prediction_output = gr.Markdown(label="Prediction Results")
                        confidence_plot = gr.Plot(label="Confidence Scores")

                with gr.Row():
                    attribution_output = gr.Video(
                        label="Attribution Visualization",
                        visible=True
                    )

                predict_btn.click(
                    fn=predict_video,
                    inputs=[video_input_quick, show_explanations],
                    outputs=[prediction_output, confidence_plot, attribution_output]
                )

            # Tab 2: Detailed Analysis
            with gr.TabItem("📊 Detailed Analysis"):
                gr.Markdown("### In-depth temporal analysis of video frames")

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_detailed = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )

                        analyze_btn = gr.Button("📈 Analyze Frame Importance", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        frame_analysis_summary = gr.Markdown(label="Analysis Summary")
                        frame_importance_plot = gr.Plot(label="Frame Importance Over Time")

                analyze_btn.click(
                    fn=analyze_frame_importance,
                    inputs=[video_input_detailed],
                    outputs=[frame_importance_plot, frame_analysis_summary]
                )

            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                    ## About This System

                    ### Model Architecture

                    This detector uses a sophisticated three-stage architecture:

                    1. **Latent Encoder**: Processes raw video frames through 3 convolutional layers
                       - Reduces spatial dimensions by factor of 8
                       - Output channels: 32 → 64 → 128

                    2. **Patch Encoder**: Extracts 8x8 patches from latent representations
                       - Maps features to 768-dimensional embeddings
                       - Efficient spatiotemporal processing

                    3. **Transformer Classifier**: 12-layer transformer with 12-head attention
                       - Models complex temporal relationships
                       - Global context understanding

                    ### Performance Metrics

                    - **Accuracy**: 85.12%
                    - **F1 Score**: 86.72%
                    - **Validation Videos**: 3,981
                    - **Balanced Performance**: Equal detection rates for both classes

                    ### Explainable AI Features

                    - **Integrated Gradients**: Frame-by-frame attribution analysis
                    - **Temporal Importance**: Identifies critical moments in videos
                    - **Visual Heatmaps**: Shows where the model focuses attention
                    - **Confidence Scores**: Transparent probability distributions

                    ### Hardware Requirements

                    - **Minimum**: 6GB GPU VRAM (inference)
                    - **Recommended**: 8GB+ GPU VRAM (with XAI)
                    - **CPU Mode**: Supported but slower

                    ### Citation

                    ```bibtex
                    @misc{aigenvideodetection2024,
                      title={AI-Generated Video Detection Using Deep Learning},
                      author={Joshua Weg},
                      year={2024},
                      publisher={GitHub},
                      howpublished={\\url{https://github.com/Joshuaweg/AIgendetector}}
                    }
                    ```

                    ---

                    **Note**: This is a research prototype. Results should be validated by human experts
                    for critical applications.
                    """
                )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666;">
                <p><strong>AI-Generated Video Detector</strong> | Built with PyTorch & Gradio</p>
                <p>© 2024 Joshua Weg | <a href="https://github.com/Joshuaweg/AIgendetector">GitHub Repository</a></p>
            </div>
            """
        )

    return app

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI-Generated Video Detector - Web Application")
    print("=" * 60)

    # Initialize model
    try:
        initialize_model()
        print("\n✓ Model initialized successfully")
    except Exception as e:
        print(f"\n✗ Error initializing model: {e}")
        print("The app will start but predictions will fail until model is loaded.")

    # Create and launch app
    app = create_app()

    print("\n" + "=" * 60)
    print("Starting web server...")
    print("=" * 60 + "\n")

    # Launch with custom settings
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True for public Gradio link
        show_error=True,
        max_threads=4
    )

if __name__ == "__main__":
    main()
