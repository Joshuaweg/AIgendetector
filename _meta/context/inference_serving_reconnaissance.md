# Flask API Inference/Serving Reconnaissance Report

Generated: 2026-03-23
Codebase: AIgendetector (AI-Generated Video Detection)

## CORE INFERENCE SERVING FILES

### Flask API Server
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/api_server.py (879 lines)

Key endpoints:
- POST /api/predict (lines 204-335) - Single video inference
- POST /api/batch/predict (lines 490-602) - Batch processing (NOTE: sequential, not true batch)
- POST /api/analyze/frames (lines 337-449) - Frame importance analysis
- GET /api/attributions/status/<video_id> (lines 619-637) - Attribution status polling
- GET /api/download/<video_id> (lines 451-488) - Video download with HTTP 206 Range support

### Model Loading & Architecture
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/full_scale_classifier.py (406 lines)
  - FullLatentEncoder (lines 104-188): 3x Conv2d stride-2, 512x512 -> 64x64
  - FullPatchEncoder (lines 190-282): 8x8 patch extraction, 768-dim embeddings
  - FullClassifier (lines 284-353): TransformerEncoder 12 layers, 12 heads
  - FullVideoClassifier (lines 355-410): Composition of all three

- /c/Users/joshu/Desktop/xai_projects/AIgendetector/interpret.py (479 lines)
  - load_model_correctly() (lines 145-188): Eager loading with device placement

### Web UI (Gradio)
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/app.py (479 lines)
  - Provides Gradio interface on port 7860

## PRODUCTION CONFIGURATION

### Gunicorn (systemd service)
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/deploy/api_server.service
  - Workers: 1 (GPU memory efficiency)
  - Threads: 4 (async lightweight work)
  - Timeout: 300s
  - Backlog: 64 (queue incoming)

### Nginx Reverse Proxy
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/deploy/nginx.conf
  - Timeouts: 300s read/send, 10s connect
  - Rate limiting: 10 req/min per IP
  - Range request support for video streaming

### Docker
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/Dockerfile.api
  - Base: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
  - Note: CMD uses 2 workers (inconsistent with systemd 1 worker)

- /c/Users/joshu/Desktop/xai_projects/AIgendetector/docker-compose.yml
  - GPU: 1 allocated, pinned to cuda:0
  - Volumes: read-only model mount, temp directories

### Dependencies
- /c/Users/joshu/Desktop/xai_projects/AIgendetector/requirements.txt
  - PyTorch CUDA 12.6
  - OpenCV, Pillow (vision)
  - Captum (Integrated Gradients)
  - Gunicorn, Flask, CORS
  - boto3 (EC2 idle watchdog)
  - imageio[ffmpeg] (video encoding)

## REQUEST HANDLING PATTERNS

### Single Inference Flow (1-3 seconds)
1. File upload + save to temp + permanent storage
2. load_video() extracts 24 frames
3. torch.no_grad() forward pass through model
4. Compute softmax + argmax
5. Return JSON with predictions

### Batch Inference (CRITICAL ISSUE)
- Sequential loop (lines 490-602)
- No torch.stack() batching
- No GPU parallelization
- Equivalent to N individual /api/predict calls

### Attribution Generation (2-5 minutes, background)
1. Client requests generate_explanations=true
2. Daemon thread spawned: _run_attributions_background()
3. Thread:
   - Clones video tensor
   - Runs IntegratedGradients with 50 steps
   - Visualizes frame-by-frame heatmaps
   - Encodes video via FFmpeg
4. Updates job status: processing -> ready/error
5. Client polls /api/attributions/status/<video_id>

ISSUE: GPU locked during attribution, blocks next inference

## MODEL LOADING

Strategy: EAGER (blocks startup)

Process (interpret.py lines 145-188):
1. Initialize FullLatentEncoder, FullPatchEncoder, FullClassifier
2. Compose FullVideoClassifier
3. Load checkpoint with weights_only fallback
4. Handle DataParallel format
5. Explicit .to(device) placement
6. Set eval mode

Device: Auto-detects CUDA, falls back to CPU

## ASYNC PATTERNS

Threading-based (not async/await):
- Gunicorn threads: 4 (light async work)
- Daemon threads: Attribution computation (blocks GPU)
- Job tracking: threading.Lock() with _attribution_jobs dict
- Polling: Client polls /api/attributions/status every 1-5s

## PERFORMANCE OBSERVATIONS

### GPU Memory
- Model load: 500MB-1GB
- Inference (batch=1): 1.5GB
- With gradients: 2-2.5GB total

### Concurrency Limitations
- Max ~1 inference + 1 attribution
- Batch endpoint provides no advantage (sequential)
- Attribution blocks GPU (daemon threads non-preemptive)

### Timeouts & Latency
- Inference: 500-1000ms model
- Attribution: 120-300s (50 IG steps + visualization + FFmpeg)
- Gunicorn timeout: 300s (may exceed)

## CRITICAL ISSUES

1. **No True Batching** (lines 490-602)
   - Fix: torch.stack() all videos, single batch forward

2. **Attribution Blocks GPU** (lines 312-616)
   - Fix: Separate worker process (Redis/Celery)

3. **Model Path Hardcoded** (lines 138-146)
   - Fix: Config file + explicit error

4. **Worker Count Inconsistency**
   - systemd: 1 worker | Dockerfile: 2 workers
   - Fix: Align configuration

5. **Synchronous Frame Loop**
   - FullLatentEncoder line 149: processes 24 frames sequentially
   - Fix: Vectorize all at once

6. **Attribution Timeout Risk**
   - Gunicorn 300s may not be enough
   - Fix: Increase to 600s or move to separate process

## DEPLOYMENT SETUP

- /c/Users/joshu/Desktop/xai_projects/AIgendetector/deploy/setup.sh
  - Bootstrap script for AWS Deep Learning AMI
  - Creates Python venv, installs deps, configures Nginx/systemd
  - Sets up IAM role for EC2 idle watchdog

## QUICK START

Development: python api_server.py
Production: gunicorn --workers 1 --threads 4 --timeout 300 -b 0.0.0.0:5000 api_server:app
Docker: docker-compose up --build
Health: curl http://localhost:5000/api/health
Predict: curl -X POST -F 'file=@video.mp4' http://localhost:5000/api/predict
Monitor: sudo journalctl -u api_server -f

