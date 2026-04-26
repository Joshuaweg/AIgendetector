"""
Pre-compute optical flow maps for all videos in s2_flow_dataset.

Reads videos from F:\\s2_flow_dataset (HDD).
Writes .npy flow files to --out-root (default: C:\\flow_cache), mirroring
the same AI-Generated/<gen>/<name>.npy / Real/<gen>/<name>.npy structure.

Sampling: evenly-spaced --n-frames frames are drawn from each video.
  Each .npy: float32 [n_frames-1, 6, flow_h, flow_w]
  At n_frames=24, flow_h=64, flow_w=64: ~2.26 MB per video
  Total output: ~68 GB for 30k videos (fixed regardless of video length).

After this completes, upload C:\\flow_cache to S3:
    python scripts/upload_flow_cache_to_s3.py

Usage:
    python scripts/precompute_flow.py
    python scripts/precompute_flow.py --workers 4 --flow-h 64 --flow-w 64
    python scripts/precompute_flow.py --dry-run
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evenly_spaced_indices(total_frames, n):
    """Return n evenly-spaced frame indices covering [0, total_frames-1]."""
    if total_frames <= n:
        return list(range(total_frames))
    return [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Farneback — same params as compute_flow_maps in dataset.py
# ---------------------------------------------------------------------------

def compute_flow_for_video(args):
    """
    Worker function. Runs in a subprocess.
    args: (video_path, out_path, flow_h, flow_w, n_frames)
    Returns (video_path, ok, reason)

    Samples n_frames evenly-spaced frames from the video and computes
    n_frames-1 optical flow maps → shape [n_frames-1, 6, flow_h, flow_w].
    """
    import cv2

    video_path, out_path, flow_h, flow_w, n_frames = args

    # Skip if already computed
    if os.path.exists(out_path):
        return str(video_path), True, 'cached'

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return str(video_path), False, 'cannot open'

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total > 0:
            # Fast path: seek directly to evenly-spaced indices
            sample_idx = set(evenly_spaced_indices(total, n_frames))
            frames_gray = []
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i in sample_idx:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (flow_w, flow_h))
                    frames_gray.append(gray)
                i += 1
        else:
            # Frame count not reported — read all, then subsample
            all_gray = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (flow_w, flow_h))
                all_gray.append(gray)
            sample_idx = evenly_spaced_indices(max(len(all_gray), 1), n_frames)
            frames_gray = [all_gray[i] for i in sample_idx]

        cap.release()

        if len(frames_gray) < 2:
            return str(video_path), False, f'only {len(frames_gray)} sampled frames'

        flow_maps = []
        prev_u, prev_v = None, None

        for i in range(1, len(frames_gray)):
            flow = cv2.calcOpticalFlowFarneback(
                frames_gray[i - 1], frames_gray[i],
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            u, v   = flow[..., 0], flow[..., 1]
            mag    = np.sqrt(u ** 2 + v ** 2)
            angle  = np.arctan2(v, u)
            du     = u - prev_u if prev_u is not None else np.zeros_like(u)
            dv     = v - prev_v if prev_v is not None else np.zeros_like(v)

            flow_6ch = np.stack([u, v, mag, angle, du, dv], axis=0)  # [6, H, W]

            for c in range(6):
                p99 = np.percentile(np.abs(flow_6ch[c]), 99) + 1e-6
                flow_6ch[c] = np.clip(flow_6ch[c] / p99, -1.0, 1.0)

            flow_maps.append(flow_6ch)
            prev_u, prev_v = u, v

        flow_array = np.stack(flow_maps, axis=0).astype(np.float32)  # [n_frames-1, 6, H, W]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, flow_array)
        return str(video_path), True, ''

    except Exception as e:
        return str(video_path), False, f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src-root',  default=r'F:\s2_flow_dataset',
                   help='Root of video dataset (HDD)')
    p.add_argument('--out-root',  default=r'F:\flow_cache',
                   help='Root for .npy output (NVMe)')
    p.add_argument('--flow-h',    type=int, default=64)
    p.add_argument('--flow-w',    type=int, default=64)
    p.add_argument('--n-frames',  type=int, default=24,
                   help='Number of evenly-spaced frames to sample per video (default: 24)')
    p.add_argument('--workers',   type=int, default=4,
                   help='Parallel workers (4 recommended for HDD)')
    p.add_argument('--dry-run',   action='store_true',
                   help='Count videos only, no processing')
    p.add_argument('--resume',    action='store_true', default=True,
                   help='Skip already-computed .npy files (default: True)')
    p.add_argument('--limit',     type=int, default=None,
                   help='Process at most N videos (for testing)')
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.src_root)
    out  = Path(args.out_root)

    print(f"Source:  {src}  (HDD)")
    print(f"Output:  {out}  (NVMe)")
    print(f"Workers: {args.workers}")

    # Collect all videos, sorted by path for sequential HDD reads
    videos = sorted(src.rglob('*.mp4'))
    if args.limit:
        videos = videos[:args.limit]
    total  = len(videos)
    print(f"Found:   {total:,} videos")

    if args.dry_run:
        already = sum(1 for v in videos
                      if (out / v.relative_to(src)).with_suffix('.npy').exists())
        print(f"Already cached: {already:,} / {total:,}")
        n_flow = args.n_frames - 1
        size_gb = total * n_flow * 6 * args.flow_h * args.flow_w * 4 / 1e9
        print(f"Frames sampled per video: {args.n_frames}  =>  {n_flow} flow maps each")
        print(f"Estimated output size: {size_gb:.1f} GB")
        return

    # Build work list
    work = []
    for v in videos:
        rel      = v.relative_to(src)
        out_path = (out / rel).with_suffix('.npy')
        work.append((str(v), str(out_path), args.flow_h, args.flow_w, args.n_frames))

    already_cached = sum(1 for _, op, _, _, _ in work if os.path.exists(op))
    remaining      = len(work) - already_cached
    print(f"Already cached: {already_cached:,}  |  Remaining: {remaining:,}")
    print()

    start    = time.time()
    done     = 0
    failed   = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_flow_for_video, w): w[0] for w in work}
        for fut in as_completed(futures):
            path, ok, reason = fut.result()
            done += 1

            if not ok:
                failed.append((path, reason))
                print(f"  FAIL [{done}/{total}] {Path(path).name} — {reason}")
            elif reason == 'cached':
                pass  # silent skip
            elif done % 500 == 0 or done == total:
                elapsed = time.time() - start
                rate    = done / elapsed if elapsed > 0 else 1
                eta_min = (total - done) / rate / 60
                print(f"  [{done:>6}/{total}]  "
                      f"rate: {rate:.1f}/s  eta: {eta_min:.0f} min")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  OK:     {total - len(failed):,}")
    print(f"  Failed: {len(failed):,}")

    if failed:
        fail_log = out / 'precompute_failures.txt'
        os.makedirs(out, exist_ok=True)
        with open(fail_log, 'w') as f:
            for path, reason in failed:
                f.write(f"{path}\t{reason}\n")
        print(f"  Failures logged: {fail_log}")
    else:
        print(f"\nAll flow maps at: {out}")
        print(f"Next: python scripts/upload_flow_cache_to_s3.py")


if __name__ == '__main__':
    main()
