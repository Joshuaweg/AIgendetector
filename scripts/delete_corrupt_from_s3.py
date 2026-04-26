"""
Delete the 175 corrupt videos from S3 using verify_results.csv.

Usage:
    python scripts/delete_corrupt_from_s3.py
    python scripts/delete_corrupt_from_s3.py --dry-run
"""

import argparse
from pathlib import Path

import boto3
import pandas as pd

BUCKET      = 'genvideo-complete'
LOCAL_ROOT  = r'F:\s2_flow_dataset'
S3_PREFIX   = 's2_flow_dataset'
REGION      = 'us-west-2'


def local_to_s3_key(local_path: str) -> str:
    rel = Path(local_path).relative_to(LOCAL_ROOT).as_posix()
    return f"{S3_PREFIX}/{rel}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',     default='data/verify_results.csv')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    keys = [local_to_s3_key(p) for p in df['path']]
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(keys)} corrupt files from s3://{BUCKET}/")

    if args.dry_run:
        for k in keys[:5]:
            print(f"  would delete: {k}")
        print(f"  ... and {len(keys)-5} more")
        return

    s3 = boto3.client('s3', region_name=REGION)

    # S3 batch delete accepts up to 1000 keys at a time
    deleted = 0
    errors  = []
    for i in range(0, len(keys), 1000):
        chunk = [{'Key': k} for k in keys[i:i+1000]]
        resp  = s3.delete_objects(Bucket=BUCKET, Delete={'Objects': chunk})
        deleted += len(resp.get('Deleted', []))
        errors  += resp.get('Errors', [])

    print(f"Deleted: {deleted}")
    if errors:
        print(f"Errors:  {len(errors)}")
        for e in errors:
            print(f"  {e['Key']} — {e['Message']}")
    else:
        print("All corrupt files removed from S3.")


if __name__ == '__main__':
    main()
