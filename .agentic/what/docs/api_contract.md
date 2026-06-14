---
type: reference
created: 2026-06-13
updated: 2026-06-13
last_edited_by: agent_init
tags: [api, contract, integration, analystrix]
---

# AIgendetector API Contract

Defines the public API surface that external consumers (analystrix.com live demo) may call. All endpoints are served by `api_server.py` via Flask on port 5000, proxied through Nginx in production.

**Base URL (production)**: configured via environment variable on the consumer side  
**Auth**: none currently (add API key header before exposing publicly)

---

## Endpoints

### `GET /api/health`
Check if the service is up.

**Response**
```json
{ "status": "healthy" }
```

---

### `POST /api/predict`
Submit a video for real/AI-generated classification.

**Request**: `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | file | yes | Video file to classify |
| `run_attribution` | bool | no | Queue async Integrated Gradients attribution (default: false) |

**Response**
```json
{
  "video_id": "abc123",
  "prediction": "real" | "ai_generated",
  "confidence": 0.94,
  "attribution_queued": true | false
}
```

---

### `GET /api/attributions/status/<video_id>`
Poll for async attribution results (Integrated Gradients heatmaps).

**Response — pending**
```json
{ "status": "processing", "video_id": "abc123" }
```

**Response — complete**
```json
{
  "status": "complete",
  "video_id": "abc123",
  "attribution_url": "/api/download/abc123"
}
```

---

### `GET /api/download/<video_id>`
Download attribution visualization for a completed video.

**Response**: binary file (video or image overlay)

---

### `GET /api/stats`
Get aggregate usage statistics.

**Response**
```json
{
  "total_predictions": 1042,
  "real_count": 614,
  "ai_generated_count": 428,
  "accuracy": 0.8512
}
```

---

## Endpoints NOT for External Use

| Endpoint | Reason |
|----------|--------|
| `POST /api/analyze/frames` | Internal frame-level analysis, not stable |
| `POST /api/batch/predict` | Internal batch processing |
| `GET /api/feedback` | Internal feedback collection |
| `POST /api/feedback` | Internal feedback collection |
| `GET /api/feedback/export` | Internal export |

---

## Consumer Reference

analystrix.com integration guide: `analystrix.com/.agentic/what/docs/aigendetector_integration.md`

## Implementation Notes

- Attribution pipeline is **async** — `POST /api/predict` with `run_attribution=true` returns immediately; poll `/api/attributions/status/<video_id>` until `status: complete`
- Video file size limit: check `api_server.py` MAX_CONTENT_LENGTH (currently ~500MB)
- CORS: not configured by default — add Nginx CORS headers or Flask-CORS before exposing to browser
