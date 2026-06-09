"""
Phase 2 — TCAV Probe Dataset Construction
Builds single-concept video probe sets via signal-based selection.

Concepts:
  temporal_motion_inconsistency  optical flow variance in face region
  facial_geometry_drift          MediaPipe landmark distance variance (requires mediapipe)
  frequency_domain_artifact      DCT spectral divergence, face vs. background
  texture_boundary_artifact      ELA-inspired residual noise at region boundaries
  lighting_shadow_decoupling     gradient direction divergence, face vs. background

Usage:
  python tcav_probes.py --manifest data/flow_manifest.csv \
      --output _meta/probes/ \
      --n-per-concept 100

Dependencies (beyond requirements.txt):
  pip install mediapipe          (optional — geometry_drift only)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_flow_maps

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

N_FRAMES   = 24
FRAME_SIZE = 512
FLOW_H     = 64
FLOW_W     = 64


# ── video frame loader ─────────────────────────────────────────────────────────

def _load_frames_bgr(video_path: str, n: int = 16) -> list | None:
    """Load n evenly-spaced BGR uint8 frames. Returns None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None
    indices = set(int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n))
    frames, fi = [], 0
    while cap.isOpened() and len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in indices:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        fi += 1
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


# ── concept signal scoring functions ──────────────────────────────────────────

def compute_motion_inconsistency(video_path: str) -> float:
    """Optical flow magnitude variance in face region — higher = more inconsistency."""
    frames = _load_frames_bgr(video_path, n=16)
    if frames is None or len(frames) < 2:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    magnitudes = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h, w   = mag.shape
        face   = mag[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
        magnitudes.append(float(face.mean()))
    return float(np.var(magnitudes))


def compute_geometry_drift(video_path: str) -> float:
    """MediaPipe eye-corner distance variance across frames. Higher = more drift."""
    if not _MP_AVAILABLE:
        return -1.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    mp_mesh  = mp.solutions.face_mesh
    dists    = []
    with mp_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as mesh:
        while cap.isOpened() and len(dists) < N_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            small  = cv2.resize(frame, (256, 256))
            result = mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                dx = lm[33].x - lm[263].x
                dy = lm[33].y - lm[263].y
                dists.append((dx ** 2 + dy ** 2) ** 0.5)
    cap.release()
    return float(np.var(dists)) if len(dists) > 2 else 0.0


def compute_frequency_artifact(video_path: str) -> float:
    """DCT high-frequency energy divergence between face center and background corners."""
    frames = _load_frames_bgr(video_path, n=8)
    if frames is None:
        return 0.0
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        face = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        bg   = gray[: h // 4, : w // 4]
        if min(face.shape) < 8 or min(bg.shape) < 8:
            continue
        face_dct = cv2.dct(cv2.resize(face, (64, 64)))
        bg_dct   = cv2.dct(cv2.resize(bg, (64, 64)))
        half = 32
        face_hf = np.abs(face_dct[half:, half:]).sum()
        bg_hf   = np.abs(bg_dct[half:, half:]).sum()
        if bg_hf > 1e-6:
            scores.append(abs(face_hf - bg_hf) / bg_hf)
    return float(np.mean(scores)) if scores else 0.0


def compute_texture_boundary(video_path: str) -> float:
    """ELA residual std at face-boundary strip — higher = more boundary artifact."""
    frames = _load_frames_bgr(video_path, n=8)
    if frames is None:
        return 0.0
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        _, enc   = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        recomp   = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        residual = np.abs(gray - recomp)
        h, w     = residual.shape
        strip    = residual[h // 3: 2 * h // 3, 2 * w // 5: 3 * w // 5]
        scores.append(float(strip.std()))
    return float(np.mean(scores)) if scores else 0.0


def compute_lighting_decoupling(video_path: str) -> float:
    """Gradient direction divergence between face and background corners."""
    frames = _load_frames_bgr(video_path, n=8)
    if frames is None:
        return 0.0
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        angle      = np.arctan2(gy, gx)
        face_ang   = angle[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
        bg_ang     = (angle[: h // 5, : w // 5].mean() + angle[: h // 5, 4 * w // 5:].mean()) / 2
        scores.append(abs(face_ang - bg_ang))
    return float(np.mean(scores)) if scores else 0.0


SCORERS: dict = {
    'temporal_motion_inconsistency': compute_motion_inconsistency,
    'facial_geometry_drift':         compute_geometry_drift,
    'frequency_domain_artifact':     compute_frequency_artifact,
    'texture_boundary_artifact':     compute_texture_boundary,
    'lighting_shadow_decoupling':    compute_lighting_decoupling,
}


# ── probe set construction ─────────────────────────────────────────────────────

def score_videos(records: list, concept_name: str, output_dir: Path) -> list:
    """
    Score all videos for concept_name. Returns list of dicts sorted descending by score.
    Cached to disk — re-running skips already-scored videos.
    """
    cache = output_dir / f'scores_{concept_name}.json'
    if cache.exists():
        print(f"[score] {concept_name}: loading cache")
        with open(cache) as f:
            return json.load(f)

    scorer = SCORERS[concept_name]
    scored = []
    for path, label in tqdm(records, desc=f'Scoring {concept_name}', unit='vid'):
        try:
            s = scorer(path)
            if s >= 0:
                scored.append({'path': path, 'label': label, 'score': s})
        except Exception:
            pass

    scored.sort(key=lambda x: x['score'], reverse=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'w') as f:
        json.dump(scored, f)
    print(f"[score] {concept_name}: {len(scored)} scored → {cache}")
    return scored


def build_probe_set(all_scores: list, n: int) -> tuple[list, list]:
    """
    Positive: top-n AI-generated (label=0) by score.
    Negative: bottom-n real (label=1) by score (cleanest real examples).
    """
    ai_sorted   = [s for s in all_scores if s['label'] == 0]
    real_sorted = sorted([s for s in all_scores if s['label'] == 1],
                         key=lambda x: x['score'])
    return ai_sorted[:n], real_sorted[:n]


def check_contamination(
    probe_set: list,
    other_scores: dict,
    concept_name: str,
    top_pct: float = 0.40,
) -> list:
    """
    Remove videos whose scores on other concepts fall in the top top_pct.
    Single-concept probe sets produce better-quality CAVs.
    """
    high_sets: dict[str, set] = {}
    for other, scores in other_scores.items():
        if other == concept_name or not scores:
            continue
        cutoff = max(s['score'] for s in scores) * top_pct
        high_sets[other] = {s['path'] for s in scores if s['score'] >= cutoff}

    clean, removed = [], 0
    for entry in probe_set:
        if any(entry['path'] in hs for hs in high_sets.values()):
            removed += 1
        else:
            clean.append(entry)

    if removed:
        print(f"  [decontam] {concept_name}: removed {removed} cross-contaminated videos")
    return clean


# ── PyTorch Dataset for activation extraction ──────────────────────────────────

class ProbeVideoDataset(Dataset):
    """
    Loads probe videos on-the-fly for TCAV activation extraction.
    __getitem__ returns (frames [T,H,W,3] float32, flow [T-1,6,Hf,Wf] float32).
    Add batch dim with .unsqueeze(0) before model forward.
    """

    def __init__(
        self,
        records: list,
        n_frames: int = N_FRAMES,
        frame_size: int = FRAME_SIZE,
        flow_h: int = FLOW_H,
        flow_w: int = FLOW_W,
    ):
        self.records    = records
        self.n_frames   = n_frames
        self.frame_size = frame_size
        self.flow_h     = flow_h
        self.flow_w     = flow_w

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.records[idx]['path']
        cap  = cv2.VideoCapture(path)
        if not cap.isOpened():
            return self._zeros()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return self._zeros()
        indices = set(
            int(round(i * (total - 1) / max(self.n_frames - 1, 1)))
            for i in range(self.n_frames)
        )
        frames, fi = [], 0
        while cap.isOpened() and len(frames) < self.n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices:
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (self.frame_size, self.frame_size),
                )
                frames.append(frame.astype(np.float32) / 255.0)
            fi += 1
        cap.release()
        while len(frames) < self.n_frames:
            frames.append(frames[-1])
        frames_np = np.stack(frames[: self.n_frames])              # [T, H, W, 3]
        flow_maps = compute_flow_maps(frames_np, self.flow_h, self.flow_w)  # [T-1, 6, Hf, Wf]
        return torch.from_numpy(frames_np), flow_maps

    def _zeros(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.n_frames, self.frame_size, self.frame_size, 3),
            torch.zeros(self.n_frames - 1, 6, self.flow_h, self.flow_w),
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Build TCAV probe datasets')
    p.add_argument('--manifest',      default='data/flow_manifest.csv',
                   help='CSV with columns: path, label (0=AI, 1=real), generator, split')
    p.add_argument('--output',        default='_meta/probes/')
    p.add_argument('--n-per-concept', type=int, default=100,
                   help='Positive + negative examples per concept probe set')
    p.add_argument('--concepts',      nargs='+', default=list(SCORERS.keys()),
                   choices=list(SCORERS.keys()),
                   help='Which concepts to build (default: all)')
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.manifest, newline='') as f:
        for row in csv.DictReader(f):
            records.append((row['path'], int(row['label'])))
    print(f"[manifest] {len(records)} videos loaded")

    # Filter concepts that require unavailable dependencies
    concepts = args.concepts
    if 'facial_geometry_drift' in concepts and not _MP_AVAILABLE:
        print("[skip] facial_geometry_drift — install mediapipe to enable")
        concepts = [c for c in concepts if c != 'facial_geometry_drift']

    # Score all videos per concept
    all_scores: dict[str, list] = {}
    for concept in concepts:
        all_scores[concept] = score_videos(records, concept, output_dir)

    # Build and save probe sets
    for concept, scores in all_scores.items():
        pos, neg = build_probe_set(scores, args.n_per_concept)
        pos = check_contamination(pos, all_scores, concept)
        neg = check_contamination(neg, all_scores, concept)
        print(f"[probe] {concept}: {len(pos)} positive, {len(neg)} negative")

        out = output_dir / f'probe_{concept}.json'
        with open(out, 'w') as f:
            json.dump({'positive': pos, 'negative': neg}, f)
        print(f"  → {out}")

    print(f"\n[done] Probe sets written to {output_dir.resolve()}")
    print("Next: python tcav_interpret.py --probes _meta/probes/")


if __name__ == '__main__':
    main()
