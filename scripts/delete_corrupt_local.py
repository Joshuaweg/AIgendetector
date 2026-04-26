"""
Delete the 175 corrupt videos from F:\\s2_flow_dataset using verify_results.csv.

Usage:
    python scripts/delete_corrupt_local.py --dry-run
    python scripts/delete_corrupt_local.py
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',     default='data/verify_results.csv')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    df   = pd.read_csv(args.csv)

    deleted = 0
    missing = 0
    for path in df['path']:
        p = Path(path)
        if p.exists():
            if args.dry_run:
                print(f"  would delete: {p}")
            else:
                p.unlink()
                deleted += 1
        else:
            missing += 1

    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(df) - missing} files ({missing} already gone)")
    else:
        print(f"Deleted: {deleted}  |  Already gone: {missing}")


if __name__ == '__main__':
    main()
