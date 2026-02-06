# Deployment Guide

Complete guide for deploying the AI-Generated Video Detector API and integrating with Next.js.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Next.js Integration](#nextjs-integration)
4. [Production Deployment](#production-deployment)
5. [Authentication](#authentication)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (6GB+ VRAM) or CPU
- Docker (optional)
- Node.js 18+ (for Next.js)

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install flask flask-cors gunicorn

# For Next.js (in your Next.js project)
npm install axios
```

### 2. Start the API Server

```bash
# Development mode
python api_server.py

# Production mode with Gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 300 api_server:app
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Get stats
curl http://localhost:5000/api/stats
```

---

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Build Standalone Docker Image

```bash
# Build image
docker build -t video-detector-api -f Dockerfile.api .

# Run container with GPU
docker run -d \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/model:/app/model:ro \
  --name video-detector \
  video-detector-api

# Run container (CPU only)
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/model:/app/model:ro \
  --name video-detector \
  video-detector-api
```

### Docker Best Practices

1. **Use volumes** for model files and temporary data
2. **Set resource limits** to prevent OOM errors
3. **Enable health checks** for automatic recovery
4. **Use multi-stage builds** to reduce image size

---

## Next.js Integration

### Step 1: Copy Integration Files

Copy these files to your Next.js project:

```
your-nextjs-app/
├── lib/
│   └── videoDetectorAPI.ts
├── components/
│   └── VideoDetector.tsx
└── app/
    └── detect/
        └── page.tsx
```

### Step 2: Configure Environment Variables

Create `.env.local` in your Next.js project:

```env
NEXT_PUBLIC_VIDEO_DETECTOR_API_URL=http://localhost:5000
```

### Step 3: Use the Component

```tsx
// app/detect/page.tsx
import VideoDetector from '@/components/VideoDetector';

export default function Page() {
  return (
    <div>
      <h1>AI Video Detection</h1>
      <VideoDetector />
    </div>
  );
}
```

### Step 4: Custom Integration

```typescript
// Custom usage with the API client
import { VideoDetectorAPI } from '@/lib/videoDetectorAPI';

const api = new VideoDetectorAPI(process.env.NEXT_PUBLIC_VIDEO_DETECTOR_API_URL);

async function handleUpload(file: File) {
  try {
    const result = await api.predict(file, true);
    console.log('Prediction:', result.prediction.class);
    console.log('Confidence:', result.prediction.confidence);
  } catch (error) {
    console.error('Prediction failed:', error);
  }
}
```

---

## Production Deployment

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/video-detector.service`:

```ini
[Unit]
Description=AI Video Detector API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/AIgendetector
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 api_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable video-detector
sudo systemctl start video-detector
sudo systemctl status video-detector
```

### Option 2: AWS Deployment

#### EC2 with Docker

```bash
# 1. Launch EC2 instance (g4dn.xlarge recommended)
# 2. Install Docker and NVIDIA Container Toolkit
# 3. Deploy

git clone https://github.com/Joshuaweg/AIgendetector.git
cd AIgendetector
docker-compose up -d
```

#### ECS with Fargate

```yaml
# task-definition.json
{
  "family": "video-detector",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/video-detector:latest",
      "memory": 8192,
      "cpu": 4096,
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

### Option 3: Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-detector-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: video-detector
  template:
    metadata:
      labels:
        app: video-detector
    spec:
      containers:
      - name: api
        image: video-detector-api:latest
        ports:
        - containerPort: 5000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
        volumeMounts:
        - name: model
          mountPath: /app/model
          readOnly: true
      volumes:
      - name: model
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: video-detector-service
spec:
  selector:
    app: video-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Configure Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/video-detector
server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

Enable and restart nginx:

```bash
sudo ln -s /etc/nginx/sites-available/video-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### HTTPS with Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Authentication

### Add API Key Authentication

Update `api_server.py`:

```python
from functools import wraps
from flask import request, jsonify

API_KEYS = {
    'your-api-key-here': 'client-name',
}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Apply to endpoints
@app.route('/api/predict', methods=['POST'])
@require_api_key
def predict():
    # ... existing code
```

### Use in Next.js

```typescript
const api = new VideoDetectorAPI('http://localhost:5000');

// Add API key to requests
const response = await fetch(`${apiUrl}/api/predict`, {
  method: 'POST',
  headers: {
    'X-API-Key': process.env.NEXT_PUBLIC_API_KEY,
  },
  body: formData,
});
```

---

## Monitoring

### Add Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('api.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
```

### Prometheus Metrics

```bash
# Install
pip install prometheus-flask-exporter

# Add to api_server.py
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

### Health Monitoring Script

```bash
#!/bin/bash
# monitor.sh

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health)

    if [ $response -eq 200 ]; then
        echo "$(date): API healthy"
    else
        echo "$(date): API unhealthy (status: $response)"
        # Restart service
        sudo systemctl restart video-detector
    fi

    sleep 60
done
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error:** `Model not found at /path/to/model`

**Solution:**
```bash
# Check model path
ls -la model/

# Update path in api_server.py or use environment variable
export MODEL_PATH=/absolute/path/to/model/full_classifier_best.pt
```

#### 2. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce workers in Gunicorn: `-w 1`
- Disable explanations by default
- Use smaller batch sizes
- Switch to CPU: `device='cpu'`

#### 3. CORS Errors

**Error:** `Access to fetch blocked by CORS policy`

**Solution:**
```python
# In api_server.py
from flask_cors import CORS

# Allow specific origin
CORS(app, origins=['https://yourdomain.com'])
```

#### 4. File Upload Fails

**Error:** `413 Request Entity Too Large`

**Solution:**
```python
# Increase file size limit
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

# In nginx
client_max_body_size 200M;
```

#### 5. Slow Predictions

**Solutions:**
- Use GPU instead of CPU
- Disable attribution generation
- Increase workers (if you have multiple GPUs)
- Implement caching with Redis

### Debug Mode

```bash
# Enable Flask debug mode (development only!)
export FLASK_DEBUG=1
python api_server.py

# Verbose logging
export LOG_LEVEL=DEBUG
```

### Check Logs

```bash
# Flask logs
tail -f api.log

# Docker logs
docker logs -f video-detector

# Systemd logs
sudo journalctl -u video-detector -f
```

---

## Performance Tuning

### Optimize for GPU

```python
# In api_server.py
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### Add Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_prediction(video_hash):
    # Cache predictions for frequently requested videos
    pass
```

### Load Balancing

Use multiple API instances with nginx:

```nginx
upstream api_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}
```

---

## Support

For issues and questions:
- **GitHub Issues:** https://github.com/Joshuaweg/AIgendetector/issues
- **Documentation:** See main README.md
- **API Health:** Check `/api/health` endpoint
