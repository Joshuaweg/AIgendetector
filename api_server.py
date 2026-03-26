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
from full_scale_classifier import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier
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

# Global model instance
model = None
device = None

# Pre-computed IG baseline: average of real video frames (distribution-matched).
# Shape: (1, 24, 512, 512, 3) float32, on the active device.
# Falls back to zeros if no real videos are available.
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


def initialize_model(model_path=None):
    """Initialize the model once at startup"""
    global model, device, _ig_baseline

    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on device: {device}")

        # Use provided path or default
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'models', 'ninox_1.pt')

        # Check if model exists locally
        if not os.path.exists(model_path):
            # Try relative to this script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'model', 'full_classifier_best.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please place your trained model at this location."
            )

        model = load_model_correctly(model_path, device)
        model.eval()
        print("Model loaded successfully!")

        # Build IG baseline from real videos (one-time, cached for all requests)
        baseline_cpu = _compute_ig_baseline(SAVED_VIDEOS_FOLDER)
        if baseline_cpu is not None:
            _ig_baseline = baseline_cpu.to(device)
        else:
            _ig_baseline = None  # signals callers to use zeros_like fallback

    return model, device

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

    # Save uploaded file to temp location
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Persist a permanent copy immediately after saving
    save_video_permanently(filepath, video_id, file.filename)

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

        # Generate explanations if requested — runs in background to avoid timeout
        if generate_explanations:
            with _attribution_jobs_lock:
                _attribution_jobs[video_id] = {'status': 'processing'}
            t = threading.Thread(
                target=_run_attributions_background,
                args=(video_tensor.detach().clone(), frames, pred, video_id),
                daemon=True,
            )
            t.start()
            response_data['attribution_video_url'] = f'/api/download/{video_id}'
            response_data['attribution_status'] = 'processing'

        # Cleanup temp only (permanent copy is already saved)
        cleanup_old_files(UPLOAD_FOLDER)

        return jsonify(response_data)

    except Exception as e:
        # Cleanup temp on error (permanent copy is kept)
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

    # Persist a permanent copy immediately after saving
    save_video_permanently(filepath, video_id, file.filename)

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
        baseline = (
            _ig_baseline.expand_as(video_tensor).clone()
            if _ig_baseline is not None
            else torch.zeros_like(video_tensor)
        )
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

        # Cleanup temp only
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

        # Persist a permanent copy immediately after saving
        save_video_permanently(filepath, video_id, file.filename)

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

            # Cleanup temp only
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

def _run_attributions_background(video_tensor, frames, pred_class, video_id):
    """Run attribution generation in a background thread and update job status."""
    try:
        video_tensor = video_tensor.to(device)
        result = generate_attributions(video_tensor, frames, pred_class, video_id)
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


def generate_attributions(video_tensor, frames, pred_class, video_id):
    """Generate attribution visualizations"""
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

        # Initialize IG
        ig = IntegratedGradients(model)

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
