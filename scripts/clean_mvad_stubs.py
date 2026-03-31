"""
Remove 225-byte macOS resource fork stubs from F:/MVAD/videos.
These were created during ZIP extraction on Windows and are not real videos.

Safe threshold: files < 10KB are stubs. Valid videos are all >= 205KB.
Run with --dry-run first to preview, then without to delete.
"""

import os
import sys
from collections import Counter
from pathlib import Path

MVAD_DIR       = Path("F:/MVAD/videos")
STUB_THRESHOLD = 10 * 1024  # 10 KB — well above 225B stubs, well below 205KB real files
DRY_RUN        = "--dry-run" in sys.argv or "-n" in sys.argv


def main():
    if not MVAD_DIR.exists():
        print(f"ERROR: {MVAD_DIR} not found")
        sys.exit(1)

    all_mp4 = [f for f in MVAD_DIR.iterdir() if f.suffix == ".mp4"]
    stubs   = [f for f in all_mp4 if f.stat().st_size < STUB_THRESHOLD]
    real    = [f for f in all_mp4 if f.stat().st_size >= STUB_THRESHOLD]

    print(f"Total .mp4 files : {len(all_mp4):,}")
    print(f"Valid (>= 10KB)  : {len(real):,}")
    print(f"Stubs  (< 10KB)  : {len(stubs):,}  ({len(stubs)/len(all_mp4)*100:.1f}%)")
    print()

    stub_gens = Counter(f.stem.split("_")[1] for f in stubs)
    real_gens = Counter(f.stem.split("_")[1] for f in real)
    all_gens  = sorted(set(list(stub_gens) + list(real_gens)))

    print(f"{'Generator':<15} {'Real':>7} {'Stubs':>7} {'Stub%':>6}")
    print("-" * 40)
    for g in all_gens:
        r = real_gens[g]
        s = stub_gens[g]
        total = r + s
        print(f"{g:<15} {r:>7} {s:>7} {s/total*100:>5.1f}%")

    print()
    if DRY_RUN:
        print(f"DRY RUN — would delete {len(stubs):,} stub files. Re-run without --dry-run to delete.")
        return

    print(f"Deleting {len(stubs):,} stub files...")
    deleted = 0
    for f in stubs:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  ERROR deleting {f.name}: {e}")

    print(f"Done. Deleted {deleted:,} stubs. Remaining: {len(real):,} valid videos.")


if __name__ == "__main__":
    main()
