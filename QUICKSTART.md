# Quick Start Guide

Get your AI Video Detector API running and integrated with Next.js in 5 minutes!

## 🚀 Step 1: Install Dependencies (2 min)

```bash
cd AIgendetector

# Install Python dependencies
pip install flask flask-cors gunicorn

# Or install everything
pip install -r requirements.txt
```

## 🔧 Step 2: Start the API Server (1 min)

```bash
# Simple start (development)
python api_server.py

# You should see:
# ✓ Model initialized successfully
# Starting API server...
# API will be available at: http://localhost:5000
```

**That's it!** Your API is now running on `http://localhost:5000`

## ✅ Step 3: Test the API (30 seconds)

Open another terminal and test:

```bash
# Check health
curl http://localhost:5000/api/health

# Get model stats
curl http://localhost:5000/api/stats
```

You should see a JSON response confirming the API is running!

## 🌐 Step 4: Integrate with Next.js (2 min)

### A. Copy Files to Your Next.js Project

```bash
# From the AIgendetector directory, copy these files:
cp nextjs-integration/videoDetectorAPI.ts YOUR_NEXTJS_PROJECT/lib/
cp nextjs-integration/VideoDetector.tsx YOUR_NEXTJS_PROJECT/components/
cp nextjs-integration/example-page.tsx YOUR_NEXTJS_PROJECT/app/detect/page.tsx
```

### B. Add Environment Variable

Create `.env.local` in your Next.js project:

```env
NEXT_PUBLIC_VIDEO_DETECTOR_API_URL=http://localhost:5000
```

### C. Install TailwindCSS (if not already)

The components use Tailwind CSS:

```bash
# In your Next.js project
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### D. Use the Component

```tsx
// app/page.tsx or any page
import VideoDetector from '@/components/VideoDetector';

export default function Home() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-4xl font-bold mb-8">AI Video Detector</h1>
      <VideoDetector />
    </div>
  );
}
```

### E. Start Your Next.js App

```bash
npm run dev
```

Visit `http://localhost:3000` and you'll see the video detector interface!

## 📝 Quick API Usage Examples

### From JavaScript/TypeScript

```typescript
// Upload and predict
const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result.prediction.class); // "AI-Generated" or "Real"
console.log(result.prediction.confidence); // 0.95
```

### From Python

```python
import requests

# Predict video
with open('video.mp4', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)

result = response.json()
print(f"Prediction: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']}")
```

### From cURL

```bash
# Upload and predict
curl -X POST http://localhost:5000/api/predict \
  -F "file=@video.mp4" \
  -F "generate_explanations=false"
```

## 🎯 Common Use Cases

### 1. Simple Prediction

```typescript
import { VideoDetectorAPI } from '@/lib/videoDetectorAPI';

const api = new VideoDetectorAPI('http://localhost:5000');
const result = await api.predict(videoFile);

if (result.prediction.class === 'AI-Generated') {
  alert(`AI-Generated with ${result.prediction.confidence * 100}% confidence`);
}
```

### 2. With Explanations

```typescript
const result = await api.predict(videoFile, true);

// Download attribution video
if (result.attribution_video_url) {
  const blob = await api.downloadAttribution(result.video_id);
  // Display or download the attribution video
}
```

### 3. Frame Analysis

```typescript
const analysis = await api.analyzeFrames(videoFile);

console.log('Key frames:', analysis.key_frames);
console.log('Most important frame:', analysis.statistics.max_frame);
```

### 4. Batch Processing

```typescript
const results = await api.batchPredict([video1, video2, video3]);

results.results.forEach(result => {
  console.log(`${result.filename}: ${result.prediction.class}`);
});
```

## 🐳 Docker Quick Start (Alternative)

If you prefer Docker:

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f api

# Test
curl http://localhost:5000/api/health
```

## 🔒 Production Deployment (5 minutes)

### Option 1: Gunicorn (Recommended)

```bash
# Install
pip install gunicorn

# Run with multiple workers
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 300 api_server:app
```

### Option 2: Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/video-detector.service

# Add configuration (see DEPLOYMENT.md)

# Start service
sudo systemctl enable video-detector
sudo systemctl start video-detector
```

### Option 3: Docker in Production

```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitor Your API

```bash
# Check health
curl http://localhost:5000/api/health

# Get statistics
curl http://localhost:5000/api/stats

# View logs
tail -f api.log  # or docker-compose logs -f
```

## 🛠️ Troubleshooting

### API Won't Start

**Problem:** Model not found

```bash
# Check if model file exists
ls -la model/full_classifier_best.pt

# Update path in api_server.py line 152
model_path = '/path/to/your/model/full_classifier_best.pt'
```

### CORS Errors in Browser

**Solution:** Flask-CORS is already configured. If still having issues:

```python
# In api_server.py
from flask_cors import CORS
CORS(app, origins=['http://localhost:3000'])  # Your Next.js URL
```

### Slow Predictions

**Solutions:**
- Use GPU (requires CUDA)
- Disable explanations: `generate_explanations=false`
- Use Gunicorn with multiple workers

### Out of Memory

**Solutions:**
```bash
# Reduce workers
gunicorn -w 1 api_server:app

# Or switch to CPU
# In api_server.py, change device to 'cpu'
```

## 📖 Next Steps

- **Full Documentation:** See `DEPLOYMENT.md`
- **API Reference:** Check `nextjs-integration/README.md`
- **Advanced Features:** Add authentication, caching, monitoring
- **Customize UI:** Modify `VideoDetector.tsx` component

## 💡 Tips

1. **GPU vs CPU:** GPU is 10-50x faster. Use CPU only if no GPU available.
2. **Explanations:** Enable only when needed (slower, ~30 seconds extra)
3. **File Size:** Default limit is 100MB, configurable in `api_server.py`
4. **Rate Limiting:** Add for production (see nginx.conf example)
5. **Caching:** Implement Redis for frequently requested videos

## 🎉 Success!

You now have:
- ✅ AI Video Detector API running
- ✅ Next.js integration ready
- ✅ Working example page
- ✅ API endpoints documented

Start detecting AI-generated videos! 🚀

---

**Need Help?**
- Read the full deployment guide: `DEPLOYMENT.md`
- Check API documentation: `nextjs-integration/README.md`
- Report issues: https://github.com/Joshuaweg/AIgendetector/issues
