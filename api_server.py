"""
Flask API Server for AI-Generated Video Detection
Designed to integrate with Next.js frontend via REST API
"""

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import sys
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
import threading
import time
import urllib.request

# Import model components
from full_scale_classifier import (
    FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier,
    FlowEncoder, FlowVideoClassifier,
)
from dataset import compute_flow_maps
from interpret import load_video, load_model_correctly, visualize, save_attributions_video
from captum.attr import IntegratedGradients

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
CORS(app, origins=['https://analystrix.com'], expose_headers=['Content-Range', 'Accept-Ranges', 'Content-Length'])  # Enable CORS for Next.js frontend

# ---------------------------------------------------------------------------
# Idle watchdog — stops the EC2 instance after N minutes of no requests
# ---------------------------------------------------------------------------
IDLE_TIMEOUT_SECONDS = int(os.environ.get('IDLE_TIMEOUT_MINUTES', '15')) * 60
_last_activity = time.time()


_IDLE_EXCLUDED_PATHS = {'/api/health', '/api/attributions/status'}

@app.before_request
def _record_activity():
    global _last_activity
    # Don't count health checks or status polls as real activity
    path = request.path
    if path in _IDLE_EXCLUDED_PATHS or path.startswith('/api/attributions/status/'):
        return
    _last_activity = time.time()


def _get_instance_metadata(path):
    """Fetch EC2 instance metadata (IMDSv2)."""
    token = urllib.request.urlopen(
        urllib.request.Request(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
            method='PUT',
        ),
        timeout=2,
    ).read().decode()
    return urllib.request.urlopen(
        urllib.request.Request(
            f'http://169.254.169.254/latest/meta-data/{path}',
            headers={'X-aws-ec2-metadata-token': token},
        ),
        timeout=2,
    ).read().decode()


def _idle_watchdog():
    """Background thread: stop the EC2 instance when idle too long."""
    import boto3
    print(f"Idle watchdog started — will stop instance after "
          f"{IDLE_TIMEOUT_SECONDS // 60} min of inactivity.")
    while True:
        time.sleep(60)
        idle_for = time.time() - _last_activity
        if idle_for >= IDLE_TIMEOUT_SECONDS:
            print(f"Idle for {idle_for:.0f}s — sending stop-instance command...")
            try:
                instance_id = _get_instance_metadata('instance-id')
                region = _get_instance_metadata('placement/region')
                boto3.client('ec2', region_name=region).stop_instances(
                    InstanceIds=[instance_id]
                )
                print(f"Stop command sent for {instance_id} in {region}.")
                return  # Stop looping — instance is shutting down
            except Exception as e:
                print(f"Watchdog stop failed: {e}")
                time.sleep(300)  # Back off 5 min before retrying


def _start_watchdog_if_ec2():
    """Only activate the watchdog when running on an actual EC2 instance."""
    try:
        urllib.request.urlopen(
            'http://169.254.169.254/latest/api/token', timeout=1
        )
        t = threading.Thread(target=_idle_watchdog, daemon=True)
        t.start()
    except Exception:
        print("Not running on EC2 — idle watchdog disabled.")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Ninox model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    'ninox1': {
        'display_name': 'Ninox 1',
        'architecture': 'FullVideoClassifier',
        'accuracy': '85.12%',
        'requires_flow': False,
        'attribution_supported': True,
    },
    'ninox1.1-flow': {
        'display_name': 'Ninox 1.1-Flow',
        'architecture': 'FlowVideoClassifier',
        'accuracy': '93.53%',
        'requires_flow': True,
        'attribution_supported': True,  # IG on video input; flow maps frozen at inference values
        'checkpoint': os.path.join(_SCRIPT_DIR, 'flow_stage2_checkpoints', 'checkpoint_epoch_0004.pt'),
    },
}

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'saved_attributions'
SAVED_VIDEOS_FOLDER = 'saved_videos'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

FEEDBACK_FOLDER = 'feedback'
ADMIN_API_KEY = os.environ.get('ADMIN_API_KEY', '')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SAVED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

# Multi-model cache: model_id -> model instance
_models: dict = {}
device = None

# Pre-computed IG baseline (shared across models that support attribution)
_ig_baseline = None

# Attribution job tracker: video_id -> {"status": "processing"|"ready"|"error", "error": str}
_attribution_jobs: dict = {}
_attribution_jobs_lock = threading.Lock()


def _compute_ig_baseline(saved_videos_dir, n_samples=10):
    """
    Build the IG baseline by averaging normalized real-video tensors.

    Real videos are identified by 'msrvtt' or 'real' in the filename — these
    are natural camera recordings absent of AI-generation artifacts.  Averaging
    N samples produces a stable reference point that lives inside the natural-
    video data distribution, so IG attributions measure 'deviation from natural'
    rather than 'deviation from black/zero', which is semantically correct for
    AI-artifact detection.  See: Bardhan et al. (2024), Distill.pub (2020).
    """
    real_files = [
        os.path.join(saved_videos_dir, f)
        for f in os.listdir(saved_videos_dir)
        if ("msrvtt" in f.lower() or "real" in f.lower())
        and f.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".MOV"))
    ]

    if not real_files:
        print("⚠  No real videos found for IG baseline — falling back to zeros.")
        return None

    # Deduplicate by MD5 to avoid skewing the mean with repeated uploads
    import hashlib
    seen, unique = set(), []
    for path in real_files:
        h = hashlib.md5(open(path, "rb").read(1 << 20)).hexdigest()  # first 1 MB
        if h not in seen:
            seen.add(h)
            unique.append(path)

    sample = unique[:n_samples]
    tensors = []
    for path in sample:
        try:
            tensor, _, _, _ = load_video(path)   # (24, 512, 512, 3), normalized
            tensors.append(tensor)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)} for baseline: {e}")

    if not tensors:
        print("⚠  Could not load any real videos — falling back to zeros.")
        return None

    baseline = torch.stack(tensors).mean(dim=0).unsqueeze(0)  # (1, 24, 512, 512, 3)
    print(f"✓ IG baseline computed from {len(tensors)} unique real video(s).")
    return baseline


def _get_ninox1_path():
    """Locate Ninox 1 checkpoint with fallbacks."""
    for candidate in [
        os.path.join(_SCRIPT_DIR, 'models', 'ninox_1.pt'),
        os.path.join(_SCRIPT_DIR, 'model', 'full_classifier_best.pt'),
        os.path.join(_SCRIPT_DIR, 'model', 'checkpoint_epoch_0004.pt'),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Ninox 1 checkpoint not found. Expected at models/ninox_1.pt or model/full_classifier_best.pt"
    )


def _load_flow_model(model_path, dev):
    """Instantiate and load FlowVideoClassifier from a Stage-2 checkpoint."""
    latent_encoder = FullLatentEncoder().to(dev)
    patch_encoder = FullPatchEncoder().to(dev)
    flow_encoder = FlowEncoder().to(dev)
    classifier = FullClassifier().to(dev)
    m = FlowVideoClassifier(latent_encoder, patch_encoder, flow_encoder, classifier).to(dev)

    try:
        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    except Exception:
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=dev, weights_only=True)

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    m.load_state_dict(state_dict)
    m.eval()
    print(
        f"Ninox 1.1-Flow loaded — epoch {checkpoint.get('epoch')}, "
        f"accuracy {checkpoint.get('best_accuracy', 'N/A'):.2f}%"
        if isinstance(checkpoint.get('best_accuracy'), float)
        else f"Ninox 1.1-Flow loaded — epoch {checkpoint.get('epoch')}"
    )
    return m


def initialize_model(model_id='ninox1'):
    """Load and cache a Ninox model by ID. Returns (model, device)."""
    global _models, device, _ig_baseline

    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model id '{model_id}'. Valid: {list(MODEL_REGISTRY)}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    if model_id not in _models:
        if model_id == 'ninox1':
            model_path = _get_ninox1_path()
            print(f"Loading Ninox 1 from {model_path}")
            _models['ninox1'] = load_model_correctly(model_path, device)
        elif model_id == 'ninox1.1-flow':
            model_path = MODEL_REGISTRY['ninox1.1-flow']['checkpoint']
            print(f"Loading Ninox 1.1-Flow from {model_path}")
            _models['ninox1.1-flow'] = _load_flow_model(model_path, device)

        _models[model_id].eval()
        print(f"{MODEL_REGISTRY[model_id]['display_name']} ready.")

        # Build IG baseline once (used by Ninox 1 attributions)
        if _ig_baseline is None:
            baseline_cpu = _compute_ig_baseline(SAVED_VIDEOS_FOLDER)
            _ig_baseline = baseline_cpu.to(device) if baseline_cpu is not None else None

    return _models[model_id], device

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_video_permanently(src_path, video_id, original_filename):
    """Copy an uploaded video to saved_videos/ for permanent storage."""
    ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else 'mp4'
    dest_filename = f"{video_id}__{original_filename}"
    dest_path = os.path.join(SAVED_VIDEOS_FOLDER, dest_filename)
    shutil.copy2(src_path, dest_path)
    return dest_path

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
        'models_loaded': list(_models.keys()),
        'device': str(device) if device else 'not initialized',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint

    Request (multipart/form-data):
        - file: video file
        - model: "ninox1" | "ninox1.1-flow"  (optional, default: "ninox1")
        - generate_explanations: "true" | "false"  (optional, Ninox 1 only)

    Response:
        {
            "success": true,
            "model": "ninox1",
            "model_name": "Ninox 1",
            "prediction": {
                "class": "AI-Generated" | "Real",
                "confidence": 0.95,
                "probabilities": {"ai_generated": 0.95, "real": 0.05}
            },
            "video_id": "uuid",
            "attribution_video_url": "/api/download/uuid"  (if explanations requested)
        }
    """
    model_id = request.form.get('model', 'ninox1')
    if model_id not in MODEL_REGISTRY:
        return jsonify({'success': False, 'error': f'Unknown model: {model_id}. Valid: {list(MODEL_REGISTRY)}'}), 400

    try:
        m, dev = initialize_model(model_id)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model initialization failed: {str(e)}'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    generate_explanations = request.form.get('generate_explanations', 'false').lower() == 'true'
    requires_flow = MODEL_REGISTRY[model_id]['requires_flow']
    attribution_supported = MODEL_REGISTRY[model_id]['attribution_supported']

    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    save_video_permanently(filepath, video_id, file.filename)

    try:
        video_tensor, frames = process_video_from_path(filepath)
        video_tensor = video_tensor.to(dev)

        with torch.no_grad():
            if requires_flow:
                frames_float = np.array(frames).astype(np.float32) / 255.0
                flow_maps = compute_flow_maps(frames_float).unsqueeze(0).to(dev)
                output = m(video_tensor, flow_maps)
            else:
                output = m(video_tensor)

            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[pred].item()

        classes = ['AI-Generated', 'Real']
        response_data = {
            'success': True,
            'video_id': video_id,
            'model': model_id,
            'model_name': MODEL_REGISTRY[model_id]['display_name'],
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

        if generate_explanations:
            with _attribution_jobs_lock:
                _attribution_jobs[video_id] = {'status': 'processing'}
            t = threading.Thread(
                target=_run_attributions_background,
                args=(
                    m,
                    video_tensor.detach().clone(),
                    flow_maps.detach().clone() if requires_flow else None,
                    frames,
                    pred,
                    video_id,
                ),
                daemon=True,
            )
            t.start()
            response_data['attribution_video_url'] = f'/api/download/{video_id}'
            response_data['attribution_status'] = 'processing'

        cleanup_old_files(UPLOAD_FOLDER)
        return jsonify(response_data)

    except Exception as e:
        try:
            os.remove(filepath)
        except:
            pass
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/frames', methods=['POST'])
def analyze_frames():
    """
    Analyze frame-level importance via IG attributions (Ninox 1 only).

    Request (multipart/form-data):
        - file: video file
        - model: "ninox1" | "ninox1.1-flow"  (optional, default: "ninox1")
    """
    model_id = request.form.get('model', 'ninox1')
    if model_id not in MODEL_REGISTRY:
        return jsonify({'success': False, 'error': f'Unknown model: {model_id}'}), 400

    try:
        m, dev = initialize_model(model_id)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model initialization failed: {str(e)}'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400

    requires_flow = MODEL_REGISTRY[model_id]['requires_flow']

    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    save_video_permanently(filepath, video_id, file.filename)

    try:
        video_tensor, frames = process_video_from_path(filepath)
        video_tensor = video_tensor.to(dev)
        video_tensor.requires_grad = True

        with torch.no_grad():
            if requires_flow:
                frames_float = np.array(frames).astype(np.float32) / 255.0
                flow_maps = compute_flow_maps(frames_float).unsqueeze(0).to(dev)
                output = m(video_tensor, flow_maps)
            else:
                flow_maps = None
                output = m(video_tensor)
            pred = torch.argmax(output, dim=1).item()

        baseline = (
            _ig_baseline.expand_as(video_tensor).clone()
            if _ig_baseline is not None
            else torch.zeros_like(video_tensor)
        )
        if flow_maps is not None:
            frozen_flow = flow_maps.detach()
            def _forward(videos):
                return m(videos, frozen_flow)
            ig = IntegratedGradients(_forward)
        else:
            ig = IntegratedGradients(m)
        attributions, _ = ig.attribute(video_tensor, baseline, target=pred, n_steps=50)

        frame_importance = attributions.abs().mean(dim=[0, 2, 3, 4]).cpu().numpy()
        frame_importance_norm = (frame_importance - frame_importance.min()) / \
                                (frame_importance.max() - frame_importance.min() + 1e-8)
        threshold = frame_importance_norm.mean() + frame_importance_norm.std()
        key_frames = np.where(frame_importance_norm > threshold)[0].tolist()

        os.remove(filepath)

        return jsonify({
            'success': True,
            'model': model_id,
            'model_name': MODEL_REGISTRY[model_id]['display_name'],
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<video_id>', methods=['GET'])
def download_attribution(video_id):
    """Stream attribution video with Range request support for browser playback"""
    filepath = os.path.join(RESULTS_FOLDER, f"{video_id}_attribution.mp4")

    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Attribution video not found'
        }), 404

    file_size = os.path.getsize(filepath)
    range_header = request.headers.get('Range')

    if range_header:
        # Parse Range: bytes=start-end
        byte_range = range_header.replace('bytes=', '').split('-')
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        length = end - start + 1

        with open(filepath, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        response = Response(
            data,
            status=206,
            mimetype='video/mp4',
            headers={
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(length),
            }
        )
        return response

    return send_file(filepath, mimetype='video/mp4', conditional=True)

@app.route('/api/batch/predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple videos.

    Request (multipart/form-data):
        - files[]: multiple video files
        - model: "ninox1" | "ninox1.1-flow"  (optional, default: "ninox1")
    """
    model_id = request.form.get('model', 'ninox1')
    if model_id not in MODEL_REGISTRY:
        return jsonify({'success': False, 'error': f'Unknown model: {model_id}'}), 400

    try:
        m, dev = initialize_model(model_id)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model initialization failed: {str(e)}'}), 500

    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    if len(files) == 0:
        return jsonify({'success': False, 'error': 'Empty file list'}), 400

    requires_flow = MODEL_REGISTRY[model_id]['requires_flow']
    classes = ['AI-Generated', 'Real']
    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({'filename': file.filename, 'success': False, 'error': 'Invalid file'})
            continue

        video_id = str(uuid.uuid4())
        filename = f"{video_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        save_video_permanently(filepath, video_id, file.filename)

        try:
            video_tensor, frames = process_video_from_path(filepath)
            video_tensor = video_tensor.to(dev)

            with torch.no_grad():
                if requires_flow:
                    frames_float = np.array(frames).astype(np.float32) / 255.0
                    flow_maps = compute_flow_maps(frames_float).unsqueeze(0).to(dev)
                    output = m(video_tensor, flow_maps)
                else:
                    output = m(video_tensor)

                probs = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(output, dim=1).item()
                confidence = probs[pred].item()

            results.append({
                'filename': file.filename,
                'video_id': video_id,
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
            os.remove(filepath)

        except Exception as e:
            results.append({'filename': file.filename, 'success': False, 'error': str(e)})
            try:
                os.remove(filepath)
            except:
                pass

    return jsonify({
        'success': True,
        'model': model_id,
        'model_name': MODEL_REGISTRY[model_id]['display_name'],
        'total': len(files),
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

def _run_attributions_background(m, video_tensor, flow_maps, frames, pred_class, video_id):
    """Run attribution generation in a background thread and update job status."""
    try:
        video_tensor = video_tensor.to(device)
        flow_maps = flow_maps.to(device) if flow_maps is not None else None
        result = generate_attributions(m, video_tensor, flow_maps, frames, pred_class, video_id)
        with _attribution_jobs_lock:
            if result:
                _attribution_jobs[video_id] = {'status': 'ready'}
            else:
                _attribution_jobs[video_id] = {'status': 'error', 'error': 'Attribution generation failed'}
    except Exception as e:
        with _attribution_jobs_lock:
            _attribution_jobs[video_id] = {'status': 'error', 'error': str(e)}


@app.route('/api/attributions/status/<video_id>', methods=['GET'])
def attribution_status(video_id):
    """Poll attribution generation status for a given video_id."""
    with _attribution_jobs_lock:
        job = _attribution_jobs.get(video_id)

    if job is None:
        # Check if the file already exists (e.g. from a previous run)
        filepath = os.path.join(RESULTS_FOLDER, f"{video_id}_attribution.mp4")
        if os.path.exists(filepath):
            return jsonify({'status': 'ready', 'attribution_video_url': f'/api/download/{video_id}'})
        return jsonify({'status': 'not_found'}), 404

    response = {'status': job['status']}
    if job['status'] == 'ready':
        response['attribution_video_url'] = f'/api/download/{video_id}'
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    return jsonify(response)


def generate_attributions(m, video_tensor, flow_maps, frames, pred_class, video_id):
    """Generate attribution visualizations.

    For FlowVideoClassifier (flow_maps is not None), IG is computed w.r.t. the
    video input only with flow maps frozen at their inference values.  The flow
    tokens and spatial tokens share the same transformer, so gradients flowing
    back through the video pathway already incorporate the joint attention signal.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Enable gradients
        video_tensor.requires_grad = True

        # Create baseline (distribution-matched real-frame average; falls back to zeros)
        baseline = (
            _ig_baseline.expand_as(video_tensor).clone()
            if _ig_baseline is not None
            else torch.zeros_like(video_tensor)
        )

        # For flow model: wrap forward so IG only differentiates w.r.t. video input
        if flow_maps is not None:
            frozen_flow = flow_maps.detach()
            def _forward(videos):
                return m(videos, frozen_flow)
            ig = IntegratedGradients(_forward)
        else:
            ig = IntegratedGradients(m)

        # Calculate attributions
        attributions, _ = ig.attribute(
            video_tensor,
            baseline,
            target=pred_class,
            n_steps=50,
            internal_batch_size=1,
            return_convergence_delta=True
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
        import traceback
        print(f"Error generating attributions: {e}\n{traceback.format_exc()}", flush=True, file=sys.stderr)
        raise

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def _require_admin():
    """Return a 401 response if the request lacks a valid admin API key, else None."""
    if not ADMIN_API_KEY:
        return jsonify({'success': False, 'error': 'Admin API key not configured on server'}), 500
    key = request.headers.get('X-API-Key', '')
    if not key or key != ADMIN_API_KEY:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    return None


@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """Return all feedback entries plus aggregate stats. Requires X-API-Key header."""
    err = _require_admin()
    if err:
        return err
    feedback_file = os.path.join(FEEDBACK_FOLDER, 'feedback.jsonl')

    entries = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total = len(entries)
    correct = sum(1 for e in entries if e.get('was_correct'))
    ai_entries = [e for e in entries if e.get('prediction') == 'AI-Generated']
    real_entries = [e for e in entries if e.get('prediction') == 'Real']
    model_counts: dict = {}
    for e in entries:
        m = e.get('model_used')
        if m:
            model_counts[m] = model_counts.get(m, 0) + 1

    return jsonify({
        'success': True,
        'stats': {
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': round(correct / total, 4) if total else None,
            'ai_generated_submissions': len(ai_entries),
            'real_submissions': len(real_entries),
            'model_counts': model_counts,
        },
        'entries': entries,
    })


@app.route('/api/feedback/export', methods=['GET'])
def export_feedback_csv():
    """Download feedback as a CSV file. Requires X-API-Key header."""
    err = _require_admin()
    if err:
        return err
    import csv, io
    feedback_file = os.path.join(FEEDBACK_FOLDER, 'feedback.jsonl')

    entries = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=['video_id', 'prediction', 'was_correct', 'model_used', 'timestamp'],
        extrasaction='ignore',
    )
    writer.writeheader()
    writer.writerows(entries)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=feedback.csv'},
    )


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Collect user feedback on classification accuracy.

    Request JSON:
        {
            "video_id": "uuid",
            "prediction": "AI-Generated" | "Real",
            "was_correct": true | false,
            "model_used": "Sora" (optional, for AI-generated videos)
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'JSON body required'}), 400

    video_id = data.get('video_id', '')
    prediction = data.get('prediction', '')
    was_correct = data.get('was_correct')
    model_used = data.get('model_used', '').strip()

    if not video_id or prediction not in ('AI-Generated', 'Real') or was_correct is None:
        return jsonify({'success': False, 'error': 'Missing or invalid fields'}), 400

    entry = {
        'video_id': video_id,
        'prediction': prediction,
        'was_correct': bool(was_correct),
        'model_used': model_used or None,
        'timestamp': datetime.now().isoformat(),
    }

    feedback_file = os.path.join(FEEDBACK_FOLDER, 'feedback.jsonl')
    with open(feedback_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

    return jsonify({'success': True})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API statistics and available models"""
    models_info = {}
    for model_id, info in MODEL_REGISTRY.items():
        models_info[model_id] = {
            'display_name': info['display_name'],
            'architecture': info['architecture'],
            'accuracy': info['accuracy'],
            'requires_flow': info['requires_flow'],
            'attribution_supported': info['attribution_supported'],
            'loaded': model_id in _models,
        }

    return jsonify({
        'success': True,
        'models': models_info,
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
    print("Ninox Model Family")
    print("=" * 60)

    # Pre-warm Ninox 1 at startup; Ninox 1.1-Flow loads on first use
    try:
        initialize_model('ninox1')
        print("\n✓ Ninox 1 initialized successfully")
    except Exception as e:
        print(f"\n✗ Error initializing Ninox 1: {e}")
        print("The server will start but predictions will fail until model is loaded.")

    # Start idle watchdog (no-op when not on EC2)
    _start_watchdog_if_ec2()

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
