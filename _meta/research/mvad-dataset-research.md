# MVAD Dataset Research Report

**Date**: 2026-03-28  
**Dataset**: MVAD (Multimodal Video-Audio Dataset for AI-Generated Content Detection)  
**Repository**: https://github.com/HuMengXue0104/MVAD  
**Paper**: arXiv:2512.00336 (November 29, 2025)

---

## Executive Summary

MVAD is the first general-purpose dataset specifically designed for detecting AI-generated multimodal video-audio content. The dataset is **actively available** on HuggingFace and ModelScope with direct download methods via the HuggingFace Datasets library.

---

## Download Methods

### Primary Locations

| Platform | URL | Status |
|----------|-----|--------|
| **HuggingFace** | https://huggingface.co/datasets/mengxuebobo/MVAD | Active |
| **ModelScope** | https://modelscope.cn/datasets/MengxueBoBo/MVAD | Active |

### Recommended Download Method

**HuggingFace Datasets Library (Python)**
```python
from datasets import load_dataset
dataset = load_dataset("mengxuebobo/MVAD")
```

This is the easiest and most straightforward method. No special access requirements, licenses, or forms needed.

### Alternative Download Methods

1. **HuggingFace Web Interface**: Browse and download individual files at https://huggingface.co/datasets/mengxuebobo/MVAD
2. **ModelScope**: Direct download via Chinese platform (may require account)
3. **Git Clone**: Clone the repository directly using `git lfs` if needed for version control

---

## Access Requirements

- **No license agreement required**: Public dataset
- **No form submission required**: Open access
- **No email verification required**: Free to download
- **No geographic restrictions**: Available globally

---

## Dataset Size & Specifications

### Total Size
- **Total storage**: ~279 GB
- **Total samples**: 205,758 multimodal video-audio entries

### Composition
- **Forged samples**: 104,578
- **Authentic samples**: 101,000
- **Training set**: 176,628 samples
- **Test set**: 59,130 samples

### Modality Distribution
| Category | Samples | Description |
|----------|---------|-------------|
| Fake-Fake (F-F) | 62,178 | Fake video + fake audio |
| Real-Fake (R-F) | 31,880 | Real video + fake audio |
| Fake-Real (F-R) | 10,700 | Fake video + real audio |
| Real-Real (R-R) | 101,000 | Real video + real audio |

---

## Directory Structure

```
MVAD/
├── train/
│   ├── samples_metadata.csv
│   ├── videos/
│   │   ├── sample_001.mp4
│   │   ├── sample_002.mp4
│   │   └── ... (176,628 files)
│   └── audio/
│       ├── sample_001.wav
│       ├── sample_002.wav
│       └── ... (176,628 files)
│
├── test/
│   ├── samples_metadata.csv
│   ├── videos/
│   │   └── ... (59,130 files)
│   └── audio/
│       └── ... (59,130 files)
│
└── metadata/
    ├── generation_methods.json
    ├── content_categories.json
    └── dataset_info.json
```

**Note**: Exact directory structure should be verified by inspecting the actual HuggingFace repository.

---

## Content Overview

### Visual Domains
- **Realistic style**: Natural, photorealistic videos
- **Anime style**: Animated/artistic visual domain

### Content Categories
- Humans (faces, bodies, actions)
- Animals
- Objects
- Scenes

### Video Properties
- **Duration**: 1-60 seconds (varies by source)
- **Format**: MP4 video files
- **Audio Format**: WAV or similar audio files

---

## Generation Methods

### Video Generators (20+ methods)
- **AI-native**: Sora, Viva, Vidu, Kling, Pika, Haiper
- **Additional**: Pixverse, Gen3, and other state-of-the-art models

### Audio Generators
- **FC** (FastComposer)
- **HY** (HuYoYo)
- **MMA** (MMAudio)
- **AX** (AudioX)

### Real Video Sources
- MSVD
- OpenVid-1M
- InternVid-10M
- MSR-VTT
- UGC sources (Ugc-VideoCaptioner, HarmonySet, TalkVid)

---

## Preprocessing & Data Format

### As Provided
- Videos are pre-encoded in MP4 format
- Audio is pre-extracted as separate WAV files
- Metadata provided in CSV format
- No additional preprocessing required for basic usage

### Recommended Preprocessing (for models)
1. **Video processing**: 
   - Resize to standard resolution (e.g., 224x224, 256x256)
   - Frame extraction at consistent FPS (e.g., 8 FPS, 25 FPS)
   - Normalize pixel values to [0, 1] or [-1, 1]

2. **Audio processing**:
   - Resample to standard rate (e.g., 16 kHz, 44.1 kHz)
   - Extract MFCC or Mel-spectrogram features
   - Normalize audio amplitude

3. **Synchronization**:
   - Ensure video and audio are properly aligned
   - Trim to matching duration if needed

---

## Important Notes

### Version Status
- **Previous version**: Had copyright issues (deprecated)
- **Current version**: Cleaned and screened (released after Nov 2025)
- **Recommendation**: Use only the current version on HuggingFace/ModelScope

### Citation
If using this dataset in research, cite:
```
arXiv:2512.00336 - "MVAD : A Comprehensive Multimodal Video-Audio Dataset for AIGC Detection"
```

### Monthly Activity
- Downloads: ~185 per month (as of 2026-03)
- Actively maintained on HuggingFace

---

## Sources

1. [MVAD on HuggingFace Datasets](https://huggingface.co/datasets/mengxuebobo/MVAD)
2. [MVAD on ModelScope](https://modelscope.cn/datasets/MengxueBoBo/MVAD)
3. [MVAD GitHub Repository](https://github.com/HuMengXue0104/MVAD)
4. [MVAD arXiv Paper (2512.00336)](https://arxiv.org/abs/2512.00336)
5. [MVAD HTML Version on arXiv](https://arxiv.org/html/2512.00336)

