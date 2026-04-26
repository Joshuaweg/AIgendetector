"""
Upload pre-computed flow maps from C:\\flow_cache to S3.

Mirrors the same structure:
    C:\\flow_cache\\AI-Generated\\<gen>\\<name>.npy
    → s3://genvideo-complete/s2_flow_cache/AI-Generated/<gen>/<name>.npy

Resumable: checkpoint file tracks completed uploads.

Usage:
    python scripts/upload_flow_cache_to_s3.py
    python scripts/upload_flow_cache_to_s3.py --dry-run
    python scripts/upload_flow_cache_to_s3.py --workers 16
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET     = 'genvideo-complete'
S3_PREFIX  = 's2_flow_cache'
REGION     = 'us-west-2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src',        default=r'F:\flow_cache')
    p.add_argument('--bucket',     default=BUCKET)
    p.add_argument('--prefix',     default=S3_PREFIX)
    p.add_argument('--checkpoint', default='flow_cache_upload_checkpoint.json')
    p.add_argument('--workers',    type=int, default=16,
                   help='Upload threads (NVMe read → network, high concurrency fine)')
    p.add_argument('--dry-run',    action='store_true')
    p.add_argument('--yes',        action='store_true', help='Skip confirmation prompt')
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.src)

    if not src.exists():
        print(f"ERROR: {src} not found. Run precompute_flow.py first.")
        return

    files = sorted(src.rglob('*.npy'))
    total_size = sum(f.stat().st_size for f in files)
    print(f"Found {len(files):,} .npy files  ({total_size/1e9:.1f} GB)")
    print(f"Destination: s3://{args.bucket}/{args.prefix}/")

    if args.dry_run:
        print("\nDry run — not uploading.")
        return

    # Load checkpoint
    completed = set()
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            completed = set(json.load(f))
    ckpt_lock  = Lock()

    remaining = [(f, f.relative_to(src).as_posix()) for f in files
                 if f.relative_to(src).as_posix() not in completed]
    print(f"Already uploaded: {len(completed):,}  |  Remaining: {len(remaining):,}")

    if not remaining:
        print("All files already uploaded.")
        return

    if not args.yes:
        confirm = input(f"\nUpload {len(remaining):,} files? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    cfg = Config(
        region_name=REGION,
        retries={'max_attempts': 10, 'mode': 'adaptive'},
        tcp_keepalive=True,
    )
    s3 = boto3.client('s3', config=cfg)
    transfer_cfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=args.workers,
        use_threads=True,
    )

    start   = time.time()
    success = 0
    failed  = []
    lock    = Lock()

    def upload_one(local_path, rel):
        s3_key = f"{args.prefix}/{rel}"
        s3.upload_file(str(local_path), args.bucket, s3_key, Config=transfer_cfg)
        with ckpt_lock:
            completed.add(rel)
            with open(args.checkpoint, 'w') as f:
                json.dump(list(completed), f)
        return rel

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_one, lp, rel): rel for lp, rel in remaining}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
                with lock:
                    success += 1
            except Exception as e:
                rel = futures[fut]
                with lock:
                    failed.append((rel, str(e)))
                print(f"  FAILED: {rel} — {e}")

            if done % 1000 == 0 or done == len(remaining):
                elapsed = time.time() - start
                rate    = done / elapsed if elapsed > 0 else 1
                eta     = (len(remaining) - done) / rate / 60
                print(f"  [{done:>6}/{len(remaining)}]  "
                      f"success={success}  failed={len(failed)}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}min")

    elapsed = time.time() - start
    print(f"\nUpload complete in {elapsed/60:.1f} min")
    print(f"  Success: {success:,}  |  Failed: {len(failed):,}")

    if failed:
        with open('flow_cache_upload_failures.json', 'w') as f:
            json.dump(failed, f, indent=2)
        print("Re-run to retry — completed files are checkpointed.")
    else:
        print(f"\nFlow cache at: s3://{args.bucket}/{args.prefix}/")
        print("Next: update sm_train_v3.py to load pre-computed flow.")


if __name__ == '__main__':
    main()
