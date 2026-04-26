"""
Verify all videos in F:\\s2_flow_dataset are readable.

Checks each .mp4 can be opened and yields at least one valid frame.
Runs in parallel with a process pool.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --root F:\\s2_flow_dataset --workers 8
    python scripts/verify_dataset.py --dry-run   # just count files, no read
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def check_video(path: str) -> tuple[str, bool, str]:
    """Return (path, ok, reason). Import cv2 inside worker to avoid fork issues."""
    import cv2
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return path, False, "cannot open"
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return path, False, "no frames"
        return path, True, ""
    except Exception as e:
        return path, False, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"F:\s2_flow_dataset")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="data/verify_results.csv", help="CSV of failed videos")
    p.add_argument("--dry-run", action="store_true", help="Count files only, no read")
    args = p.parse_args()

    root = Path(args.root)
    videos = sorted(root.rglob("*.mp4"))
    total = len(videos)
    print(f"Found {total:,} videos under {root}")

    if args.dry_run:
        return

    failed = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_video, str(v)): v for v in videos}
        for fut in as_completed(futures):
            path, ok, reason = fut.result()
            done += 1
            if not ok:
                failed.append((path, reason))
                print(f"  FAIL [{done}/{total}] {path} — {reason}")
            elif done % 1000 == 0:
                print(f"  ok   [{done}/{total}] ...")

    print(f"\n{'='*60}")
    print(f"Total:  {total:,}")
    print(f"OK:     {total - len(failed):,}")
    print(f"Failed: {len(failed):,}")

    if failed:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "reason"])
            w.writerows(failed)
        print(f"\nFailed list written to: {out}")
        sys.exit(1)
    else:
        print("\nAll videos readable.")


if __name__ == "__main__":
    main()
