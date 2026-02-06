"""
Flask API Server for AI-Generated Video Detection
Designed to integrate with Next.js frontend via REST API
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import cv2
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import base64
import io
from datetime import datetime
import uuid
import json

# Import model components
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
from interpret import load_video, load_model_correctly, visualize, save_attributions_video
from captum.attr import IntegratedGradients

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(folder, max_age_seconds=3600):
    """Remove files older than max_age_seconds"""
    now = datetime.now().timestamp()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > max_age_seconds:
                try:
                    os.remove(filepath)
                except:
                    pass

def process_video_from_path(video_path):
    """Process video file and return tensor"""
    try:
        video_tensor, frames, _, _ = load_video(video_path)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        return video_tensor, frames
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint

    Request:
        - file: video file (multipart/form-data)
        - generate_explanations: boolean (optional, default: false)

    Response:
        {
            "success": true,
            "prediction": {
                "class": "AI-Generated" or "Real",
                "confidence": 0.95,
                "probabilities": {
                    "ai_generated": 0.95,
                    "real": 0.05
                }
            },
            "video_id": "uuid",
            "attribution_video_url": "/api/download/uuid" (if explanations requested)
        }
    """
    global model, device

    # Initialize model if not already done
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model initialization failed: {str(e)}'
            }), 500

    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Get options
    generate_explanations = request.form.get('generate_explanations', 'false').lower() == 'true'

    # Generate unique ID for this request
    video_id = str(uuid.uuid4())

    # Save uploaded file
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Process video
        video_tensor, frames = process_video_from_path(filepath)
        video_tensor = video_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[pred].item()

        classes = ['AI-Generated', 'Real']

        # Build response
        response_data = {
            'success': True,
            'video_id': video_id,
            'prediction': {
                'class': classes[pred],
                'class_index': pred,
                'confidence': float(confidence),
                'probabilities': {
                    'ai_generated': float(probs[0]),
                    'real': float(probs[1])
                }
            },
            'timestamp': datetime.now().isoformat()
        }

        # Generate explanations if requested
        if generate_explanations:
            attribution_path = generate_attributions(
                video_tensor, frames, pred, video_id
            )
            if attribution_path:
                response_data['attribution_video_url'] = f'/api/download/{video_id}'

        # Cleanup uploaded file
        cleanup_old_files(UPLOAD_FOLDER)

        return jsonify(response_data)

    except Exception as e:
        # Cleanup on error
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze/frames', methods=['POST'])
def analyze_frames():
    """
    Analyze frame-level importance

    Request:
        - file: video file (multipart/form-data)

    Response:
        {
            "success": true,
            "frame_importance": [0.1, 0.5, 0.9, ...],
            "key_frames": [2, 10, 15],
            "statistics": {
                "mean": 0.45,
                "std": 0.25,
                "max_frame": 10
            }
        }
    """
    global model, device

    if model is None:
        try:
            initialize_model()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model initialization failed: {str(e)}'
            }), 500

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file'
        }), 400

    # Save temporary file
    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Process video
        video_tensor, frames = process_video_from_path(filepath)
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

        # Compute frame importance
        frame_importance = attributions.abs().mean(dim=[0, 2, 3, 4]).cpu().numpy()

        # Normalize
        frame_importance_norm = (frame_importance - frame_importance.min()) / \
                               (frame_importance.max() - frame_importance.min() + 1e-8)

        # Find key frames
        threshold = frame_importance_norm.mean() + frame_importance_norm.std()
        key_frames = np.where(frame_importance_norm > threshold)[0].tolist()

        # Cleanup
        os.remove(filepath)

        return jsonify({
            'success': True,
            'frame_importance': frame_importance_norm.tolist(),
            'key_frames': key_frames,
            'statistics': {
                'mean': float(frame_importance_norm.mean()),
                'std': float(frame_importance_norm.std()),
                'max_frame': int(frame_importance_norm.argmax()),
                'min_frame': int(frame_importance_norm.argmin())
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download/<video_id>', methods=['GET'])
def download_attribution(video_id):
    """Download attribution video"""
    filepath = os.path.join(RESULTS_FOLDER, f"{video_id}_attribution.mp4")

    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Attribution video not found'
        }), 404

    return send_file(
        filepath,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f'attribution_{video_id}.mp4'
    )

@app.route('/api/batch/predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple videos

    Request:
        - files[]: multiple video files

    Response:
        {
            "success": true,
            "results": [
                {
                    "filename": "video1.mp4",
                    "prediction": {...}
                },
                ...
            ]
        }
    """
    global model, device

    if model is None:
        try:
            initialize_model()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model initialization failed: {str(e)}'
            }), 500

    if 'files[]' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        }), 400

    files = request.files.getlist('files[]')

    if len(files) == 0:
        return jsonify({
            'success': False,
            'error': 'Empty file list'
        }), 400

    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'success': False,
                'error': 'Invalid file'
            })
            continue

        # Save temporary file
        video_id = str(uuid.uuid4())
        filename = f"{video_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Process and predict
            video_tensor, _ = process_video_from_path(filepath)
            video_tensor = video_tensor.to(device)

            with torch.no_grad():
                output = model(video_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(output, dim=1).item()
                confidence = probs[pred].item()

            classes = ['AI-Generated', 'Real']

            results.append({
                'filename': file.filename,
                'success': True,
                'prediction': {
                    'class': classes[pred],
                    'confidence': float(confidence),
                    'probabilities': {
                        'ai_generated': float(probs[0]),
                        'real': float(probs[1])
                    }
                }
            })

            # Cleanup
            os.remove(filepath)

        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })

            try:
                os.remove(filepath)
            except:
                pass

    return jsonify({
        'success': True,
        'total': len(files),
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

def generate_attributions(video_tensor, frames, pred_class, video_id):
    """Generate attribution visualizations"""
    temp_dir = tempfile.mkdtemp()

    try:
        # Enable gradients
        video_tensor.requires_grad = True

        # Create baseline
        baseline = torch.zeros_like(video_tensor)

        # Initialize IG
        ig = IntegratedGradients(model)

        # Calculate attributions
        attributions, _ = ig.attribute(
            video_tensor,
            baseline,
            target=pred_class,
            n_steps=50,
            internal_batch_size=1
        )

        # Visualize
        visualize(
            attributions,
            torch.tensor(frames),
            save_path=temp_dir,
            label=f"Attribution Analysis"
        )

        # Create video
        output_path = os.path.join(RESULTS_FOLDER, f"{video_id}_attribution.mp4")
        save_attributions_video(temp_dir, output_path, fps=8)

        return output_path

    except Exception as e:
        print(f"Error generating attributions: {e}")
        return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    return jsonify({
        'success': True,
        'model_info': {
            'architecture': 'CNN-Transformer Hybrid',
            'accuracy': '85.12%',
            'f1_score': '86.72%',
            'parameters': {
                'latent_encoder': 'FullLatentEncoder',
                'patch_encoder': 'FullPatchEncoder',
                'classifier': 'FullClassifier (12 layers, 12 heads)'
            }
        },
        'device': str(device) if device else 'not initialized',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024)
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AI-Generated Video Detector - API Server")
    print("=" * 60)

    # Initialize model
    try:
        initialize_model()
        print("\n✓ Model initialized successfully")
    except Exception as e:
        print(f"\n✗ Error initializing model: {e}")
        print("The server will start but predictions will fail until model is loaded.")

    print("\n" + "=" * 60)
    print("Starting API server...")
    print("API will be available at: http://localhost:5000")
    print("=" * 60 + "\n")

    # Configure and run
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
