"""
Comprehensive per-generator video analysis for optical flow detector efficacy.

Sources:
  F:/Gen-Video/GenVideo-Train/AI-Generated/<generator>/  (subdirs)
  F:/MVAD/videos/  (flat: ai_<generator>_<hash>.mp4)

Metrics per generator (up to 500 sampled videos):
  - FPS: mean, std, min, max
  - Frame count: mean, std, min, max
  - Duration (s): mean, std
  - Resolution: most common, n_unique
  - Decodability rate (%)
  - Codec (fourcc)

Flow metrics (subsample of 30 per generator via Farneback at 64x64):
  - Mean flow magnitude (pixels/frame)
  - Std of flow magnitude across frame pairs (temporal consistency)
  - Frozen frame rate: % pairs with near-zero flow (<0.5 px)
  - Max flow magnitude
"""

import cv2
import os
import sys
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GENVIDEO_DIR = Path("F:/Gen-Video/GenVideo-Train/AI-Generated")
MVAD_DIR     = Path("F:/MVAD/videos")

MAX_SAMPLE   = 500   # metadata sample per generator
FLOW_SAMPLE  = 30    # flow computation sample per generator
FLOW_H, FLOW_W = 64, 64
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_video_meta(path: str):
    """Return dict of metadata or None if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps        = cap.get(cv2.CAP_PROP_FPS)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec      = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)]).strip()

    # Try reading one frame to confirm decodability
    ret, _ = cap.read()
    cap.release()
    if not ret or fps <= 0 or n_frames <= 0:
        return None

    duration = n_frames / fps
    return dict(fps=fps, n_frames=n_frames, w=w, h=h, codec=codec, duration=duration)


def compute_flow_stats(path: str, max_frames=24):
    """Compute optical flow statistics for a single video. Returns dict or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    frames = []
    while len(frames) < max_frames * 2:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (FLOW_W, FLOW_H))
        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return None

    # Sample up to max_frames evenly
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames)-1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]

    magnitudes = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i+1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitudes.append(float(mag.mean()))

    if not magnitudes:
        return None

    mags = np.array(magnitudes)
    return dict(
        mean_mag   = float(mags.mean()),
        std_mag    = float(mags.std()),
        max_mag    = float(mags.max()),
        frozen_pct = float((mags < 0.5).mean() * 100),  # % near-zero pairs
    )


def stats(values):
    """Return mean/std/min/max dict for a list of numbers."""
    a = np.array(values, dtype=float)
    return dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max())


def print_generator_report(gen_name, meta_list, flow_list):
    n = len(meta_list)
    print(f"\n{'='*60}")
    print(f"  {gen_name}  (n={n})")
    print(f"{'='*60}")

    fps_vals      = [m['fps']      for m in meta_list]
    frame_vals    = [m['n_frames'] for m in meta_list]
    dur_vals      = [m['duration'] for m in meta_list]
    res_vals      = [f"{m['w']}x{m['h']}" for m in meta_list]
    codecs        = Counter(m['codec'] for m in meta_list)

    s = stats(fps_vals)
    print(f"  FPS         mean={s['mean']:.1f}  std={s['std']:.1f}  "
          f"min={s['min']:.0f}  max={s['max']:.0f}")

    s = stats(frame_vals)
    print(f"  Frames      mean={s['mean']:.0f}  std={s['std']:.0f}  "
          f"min={s['min']:.0f}  max={s['max']:.0f}")

    s = stats(dur_vals)
    print(f"  Duration(s) mean={s['mean']:.1f}  std={s['std']:.1f}  "
          f"min={s['min']:.1f}  max={s['max']:.1f}")

    top_res = Counter(res_vals).most_common(3)
    print(f"  Resolution  {top_res}  ({len(set(res_vals))} unique)")
    print(f"  Codec       {dict(codecs.most_common(3))}")

    if flow_list:
        mn  = np.mean([f['mean_mag']   for f in flow_list])
        sd  = np.mean([f['std_mag']    for f in flow_list])
        mx  = np.mean([f['max_mag']    for f in flow_list])
        frz = np.mean([f['frozen_pct'] for f in flow_list])
        print(f"  Flow(px/fr) mean={mn:.2f}  temporal_std={sd:.2f}  "
              f"max={mx:.2f}  frozen%={frz:.1f}")

        # Efficacy assessment
        if frz > 40:
            note = "HIGH frozen% → many static frames, low signal"
        elif mn < 0.3:
            note = "very low motion → may be hard for flow detector"
        elif sd > mn * 1.5:
            note = "high temporal variability → good discriminative signal"
        else:
            note = "stable motion → moderate flow signal"
        print(f"  Assessment  {note}")
    else:
        print(f"  Flow        (not computed)")


# ---------------------------------------------------------------------------
# Collect video paths per generator
# ---------------------------------------------------------------------------
def collect_genvideo(base_dir: Path):
    generators = {}
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        mp4s = list(subdir.rglob("*.mp4"))
        if mp4s:
            generators[subdir.name] = mp4s
    return generators


def collect_mvad(base_dir: Path):
    generators = defaultdict(list)
    for f in sorted(base_dir.glob("ai_*.mp4")):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            gen = parts[1]
            generators[gen].append(f)
    return dict(generators)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze(gen_name, all_paths, rng):
    # Sample for metadata
    sample = rng.sample(all_paths, min(MAX_SAMPLE, len(all_paths)))

    meta_list = []
    fail_count = 0
    for p in sample:
        m = get_video_meta(str(p))
        if m:
            meta_list.append(m)
        else:
            fail_count += 1

    if not meta_list:
        print(f"\n[{gen_name}] SKIP — 0/{len(sample)} decodable")
        return

    decodability = len(meta_list) / len(sample) * 100

    # Flow subsample
    flow_candidates = rng.sample(all_paths, min(FLOW_SAMPLE, len(all_paths)))
    flow_list = []
    for p in flow_candidates:
        fs = compute_flow_stats(str(p))
        if fs:
            flow_list.append(fs)

    print_generator_report(gen_name, meta_list, flow_list)
    print(f"  Decodable   {decodability:.1f}%  ({fail_count} failed of {len(sample)})")
    print(f"  Total avail {len(all_paths)}")


def main():
    rng = random.Random(RANDOM_SEED)

    print("Collecting video paths...")
    genvideo_gens = collect_genvideo(GENVIDEO_DIR) if GENVIDEO_DIR.exists() else {}
    mvad_gens     = collect_mvad(MVAD_DIR)         if MVAD_DIR.exists() else {}

    all_generators = {}
    for name, paths in genvideo_gens.items():
        all_generators[f"GV_{name}"] = [str(p) for p in paths]
    for name, paths in mvad_gens.items():
        all_generators[f"MVAD_{name}"] = [str(p) for p in paths]

    print(f"Found {len(all_generators)} generators, {sum(len(v) for v in all_generators.values()):,} total videos")
    print("Analyzing... (this will take ~10-15 min)\n")

    for gen_name, paths in sorted(all_generators.items()):
        sys.stdout.flush()
        analyze(gen_name, paths, rng)

    print(f"\n{'='*60}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
