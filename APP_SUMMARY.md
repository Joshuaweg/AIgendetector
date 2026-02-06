# App Integration Summary

## 🎯 What Was Created

A complete full-stack integration for your AI-Generated Video Detector with Next.js support.

## 📁 Files Created

### 1. **API Server** (`api_server.py`)
Flask-based REST API with the following endpoints:

- **POST /api/predict** - Single video prediction with optional XAI
- **POST /api/analyze/frames** - Frame-level importance analysis
- **POST /api/batch/predict** - Batch process multiple videos
- **GET /api/download/<video_id>** - Download attribution videos
- **GET /api/health** - Health check endpoint
- **GET /api/stats** - Model information and statistics

**Features:**
- CORS enabled for Next.js integration
- File upload validation and size limits
- Automatic cleanup of temporary files
- Error handling and consistent response format
- GPU/CPU support with automatic detection

### 2. **Next.js Integration Files**

#### `nextjs-integration/videoDetectorAPI.ts`
TypeScript API client with:
- Type-safe interfaces for all API responses
- Methods for all API endpoints
- React hook for easy integration
- Error handling and validation

#### `nextjs-integration/VideoDetector.tsx`
React component featuring:
- Video upload with preview
- Real-time prediction display
- Confidence score visualization
- Frame analysis results
- Attribution video download
- Beautiful Tailwind CSS styling
- Loading states and error handling

#### `nextjs-integration/example-page.tsx`
Complete example page showing:
- How to use the VideoDetector component
- Custom callback handling
- Information sections
- Technical details display

#### `nextjs-integration/README.md`
Comprehensive integration guide with:
- Setup instructions
- API endpoint documentation
- Usage examples
- CORS configuration
- Production deployment tips

### 3. **Deployment Files**

#### `Dockerfile.api`
Production-ready Docker image with:
- NVIDIA CUDA 12.1 support
- Python 3.10
- All dependencies pre-installed
- Gunicorn for production serving
- Health checks

#### `docker-compose.yml`
Multi-container setup with:
- API service with GPU support
- Nginx reverse proxy (optional)
- Volume mounts for model and data
- Health checks and auto-restart

#### `nginx.conf`
Nginx configuration with:
- Rate limiting
- CORS headers
- Large file upload support
- SSL/HTTPS ready
- Proxy timeout configurations

#### `.env.example`
Environment variables template for:
- API configuration
- Model paths
- File upload settings
- Next.js integration
- Optional features (auth, caching, monitoring)

### 4. **Documentation**

#### `DEPLOYMENT.md`
Complete deployment guide covering:
- Quick start instructions
- Docker deployment
- Next.js integration steps
- Production deployment options (Systemd, AWS, Kubernetes)
- Authentication setup
- Monitoring and logging
- Troubleshooting common issues
- Performance tuning tips

#### `QUICKSTART.md`
5-minute setup guide with:
- Step-by-step installation
- API testing commands
- Next.js integration steps
- Quick usage examples
- Common troubleshooting

#### `APP_SUMMARY.md`
This file - overview of everything created

## 🚀 Quick Start

### Start API Server:
```bash
python api_server.py
# API runs on http://localhost:5000
```

### Integrate with Next.js:
```bash
# 1. Copy integration files to your Next.js project
cp nextjs-integration/videoDetectorAPI.ts YOUR_PROJECT/lib/
cp nextjs-integration/VideoDetector.tsx YOUR_PROJECT/components/

# 2. Add environment variable
echo "NEXT_PUBLIC_VIDEO_DETECTOR_API_URL=http://localhost:5000" >> .env.local

# 3. Use in your pages
import VideoDetector from '@/components/VideoDetector';
```

## 🔌 API Endpoints Reference

### Predict Video
```bash
POST /api/predict
Body: multipart/form-data
  - file: video file
  - generate_explanations: boolean (optional)

Response:
{
  "success": true,
  "video_id": "uuid",
  "prediction": {
    "class": "AI-Generated" | "Real",
    "confidence": 0.95,
    "probabilities": {
      "ai_generated": 0.95,
      "real": 0.05
    }
  },
  "attribution_video_url": "/api/download/uuid"
}
```

### Analyze Frames
```bash
POST /api/analyze/frames
Body: multipart/form-data
  - file: video file

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
```

### Batch Predict
```bash
POST /api/batch/predict
Body: multipart/form-data
  - files[]: array of video files

Response:
{
  "success": true,
  "total": 3,
  "results": [
    {
      "filename": "video1.mp4",
      "prediction": {...}
    },
    ...
  ]
}
```

## 💻 Usage Examples

### TypeScript/Next.js
```typescript
import { VideoDetectorAPI } from '@/lib/videoDetectorAPI';

const api = new VideoDetectorAPI('http://localhost:5000');

// Simple prediction
const result = await api.predict(videoFile);
console.log(result.prediction.class); // "AI-Generated" or "Real"

// With explanations
const result = await api.predict(videoFile, true);
const blob = await api.downloadAttribution(result.video_id);

// Frame analysis
const analysis = await api.analyzeFrames(videoFile);
console.log(analysis.key_frames); // [2, 10, 15]

// Batch processing
const results = await api.batchPredict([video1, video2, video3]);
```

### React Component
```tsx
import VideoDetector from '@/components/VideoDetector';

export default function Page() {
  const handlePrediction = (result) => {
    console.log('Prediction:', result);
    // Your custom logic
  };

  return (
    <VideoDetector
      apiUrl="http://localhost:5000"
      onPrediction={handlePrediction}
      showFrameAnalysis={true}
    />
  );
}
```

### Python
```python
import requests

# Predict
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict',
        files={'file': f}
    )

result = response.json()
print(f"Class: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']}")
```

### cURL
```bash
# Health check
curl http://localhost:5000/api/health

# Predict
curl -X POST http://localhost:5000/api/predict \
  -F "file=@video.mp4" \
  -F "generate_explanations=true"

# Frame analysis
curl -X POST http://localhost:5000/api/analyze/frames \
  -F "file=@video.mp4"
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build manually
docker build -t video-detector-api -f Dockerfile.api .
docker run -d --gpus all -p 5000:5000 video-detector-api
```

### With Nginx
```bash
# Full stack with reverse proxy
docker-compose up -d

# API on http://localhost:5000
# Nginx proxy on http://localhost:80
```

## 🔐 Production Checklist

- [ ] Update CORS settings for your domain
- [ ] Add API key authentication
- [ ] Configure HTTPS with SSL certificates
- [ ] Set up rate limiting
- [ ] Configure monitoring and logging
- [ ] Use Gunicorn with multiple workers
- [ ] Set up automated backups
- [ ] Configure firewall rules
- [ ] Test error handling
- [ ] Set up health check monitoring

## 🎨 Customization

### Modify API Response
Edit `api_server.py`:
```python
response_data = {
    'success': True,
    'prediction': {...},
    'custom_field': 'your_value'  # Add custom fields
}
```

### Customize UI Component
Edit `VideoDetector.tsx`:
```tsx
// Change colors, layout, features
<div className="your-custom-classes">
  {/* Your custom UI */}
</div>
```

### Add Authentication
```python
# In api_server.py
API_KEYS = {'your-key': 'client-name'}

@require_api_key
def predict():
    # Protected endpoint
```

## 📊 Monitoring

### Check API Health
```bash
# Health endpoint
curl http://localhost:5000/api/health

# Stats endpoint
curl http://localhost:5000/api/stats
```

### View Logs
```bash
# Flask logs
tail -f api.log

# Docker logs
docker-compose logs -f api

# Systemd logs
sudo journalctl -u video-detector -f
```

## 🐛 Troubleshooting

### Model Not Found
```bash
# Check path
ls -la model/full_classifier_best.pt

# Update in api_server.py
model_path = '/correct/path/to/model/full_classifier_best.pt'
```

### CORS Issues
```python
# In api_server.py
from flask_cors import CORS
CORS(app, origins=['https://yourdomain.com'])
```

### Out of Memory
```bash
# Reduce workers
gunicorn -w 1 api_server:app

# Or use CPU
device = torch.device('cpu')
```

## 📚 Additional Resources

- **Main README:** Model architecture and training details
- **DEPLOYMENT.md:** Comprehensive deployment guide
- **QUICKSTART.md:** 5-minute setup guide
- **Integration README:** Next.js integration details

## 🎯 What's Next?

1. **Try the API:**
   ```bash
   python api_server.py
   curl http://localhost:5000/api/health
   ```

2. **Integrate with Next.js:**
   - Copy files from `nextjs-integration/`
   - Add environment variables
   - Use the VideoDetector component

3. **Deploy to Production:**
   - Use Docker or Gunicorn
   - Configure nginx reverse proxy
   - Set up HTTPS with Let's Encrypt

4. **Customize:**
   - Add your branding
   - Implement authentication
   - Add custom features

## 💡 Key Features

✅ **REST API** - Full-featured Flask API with all endpoints
✅ **Next.js Integration** - Ready-to-use React components
✅ **Type Safety** - TypeScript interfaces for all API responses
✅ **XAI Support** - Optional attribution visualizations
✅ **Batch Processing** - Handle multiple videos at once
✅ **Docker Ready** - Production-ready containerization
✅ **Well Documented** - Comprehensive guides and examples
✅ **Error Handling** - Robust error responses
✅ **GPU Support** - CUDA acceleration when available
✅ **Production Ready** - Gunicorn, nginx, monitoring

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details

---

**🎉 You're all set!** Start the API server and integrate with your Next.js app.

For questions or issues, check the documentation or open a GitHub issue.
