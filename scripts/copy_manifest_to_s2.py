"""
Copy all videos listed in flow_manifest.csv to F:\\s2_flow_dataset,
organized so FlowVideoDataset can infer labels from folder structure.

Output layout:
    F:\\s2_flow_dataset\\
        AI-Generated\\<generator>\\<filename>.mp4
        Real\\<generator>\\<filename>.mp4

Run from project root:
    python scripts/copy_manifest_to_s2.py
    python scripts/copy_manifest_to_s2.py --manifest data/flow_manifest.csv --dest F:\\s2_flow_dataset
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/flow_manifest.csv')
    p.add_argument('--dest',     default=r'F:\s2_flow_dataset')
    p.add_argument('--dry-run',  action='store_true', help='Print actions without copying')
    return p.parse_args()


def main():
    args = parse_args()
    dest = Path(args.dest)

    df = pd.read_csv(args.manifest)
    total = len(df)
    print(f"Manifest: {total} videos")
    print(f"Destination: {dest}")
    if args.dry_run:
        print("DRY RUN — no files will be copied\n")

    skipped = 0
    copied  = 0
    missing = 0
    errors  = 0

    for i, row in enumerate(df.itertuples(), 1):
        src = Path(row.path)

        # Build destination subfolder: AI-Generated/<generator>/ or Real/<generator>/
        label_dir = 'Real' if row.label == 1 else 'AI-Generated'
        dst_dir   = dest / label_dir / row.generator
        dst_file  = dst_dir / src.name

        if (i % 500) == 0 or i == total:
            print(f"[{i:>6}/{total}]  copied={copied}  skipped={skipped}  missing={missing}  errors={errors}")

        if dst_file.exists():
            skipped += 1
            continue

        if not src.exists():
            print(f"  MISSING: {src}")
            missing += 1
            continue

        if args.dry_run:
            copied += 1
            continue

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_file)
            copied += 1
        except Exception as e:
            print(f"  ERROR copying {src}: {e}")
            errors += 1

    print(f"\nDone.")
    print(f"  Copied:  {copied}")
    print(f"  Skipped: {skipped}  (already existed)")
    print(f"  Missing: {missing}  (source not found)")
    print(f"  Errors:  {errors}")
    if missing:
        print(f"\nWARNING: {missing} source files not found — check drive is mounted and paths are correct")


if __name__ == '__main__':
    main()
