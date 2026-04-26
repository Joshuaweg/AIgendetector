"""
Upload F:\\s2_flow_dataset to s3://genvideo-complete/s2_flow_dataset/.

Timeout prevention:
  - Multipart upload (8 MB chunks) — each part has its own timeout, not the whole file
  - TCP keepalive enabled
  - Adaptive retry (up to 10 attempts with exponential backoff)
  - Local checkpoint file tracks completed uploads — safe to kill and resume

Usage:
    python scripts/upload_s2_to_s3.py
    python scripts/upload_s2_to_s3.py --source F:\\s2_flow_dataset --prefix s2_flow_dataset
    python scripts/upload_s2_to_s3.py --dry-run          # count files, no upload
    python scripts/upload_s2_to_s3.py --workers 20       # increase parallelism
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUCKET          = 'genvideo-complete'
DEFAULT_SOURCE  = r'F:\s2_flow_dataset'
DEFAULT_PREFIX  = 's2_flow_dataset'
DEFAULT_CHECKPOINT = 'upload_checkpoint.json'
REGION          = 'us-west-2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source',     default=DEFAULT_SOURCE)
    p.add_argument('--bucket',     default=BUCKET)
    p.add_argument('--prefix',     default=DEFAULT_PREFIX,
                   help='S3 key prefix (no leading slash)')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                   help='JSON file tracking completed uploads — enables resume')
    p.add_argument('--workers',    type=int, default=10,
                   help='Parallel upload threads (default 10)')
    p.add_argument('--dry-run',    action='store_true',
                   help='Count files and estimate size without uploading')
    return p.parse_args()


def build_s3_client():
    """S3 client configured to survive long uploads."""
    cfg = Config(
        region_name=REGION,
        retries={
            'max_attempts': 10,
            'mode': 'adaptive',       # backs off exponentially on throttle/timeout
        },
        tcp_keepalive=True,           # prevents idle connection timeout mid-upload
        connect_timeout=30,
        read_timeout=120,             # per-chunk timeout, not whole-file
    )
    return boto3.client('s3', config=cfg)


def build_transfer_config(workers):
    """Multipart config: 8 MB chunks, each chunk independently retried."""
    return TransferConfig(
        multipart_threshold  = 8 * 1024 * 1024,   # files > 8 MB use multipart
        multipart_chunksize  = 8 * 1024 * 1024,   # 8 MB per part
        max_concurrency      = workers,
        use_threads          = True,
    )


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path, completed, lock):
    with lock:
        with open(path, 'w') as f:
            json.dump(list(completed), f)


def collect_files(source_dir):
    """Walk source directory, return list of (local_path, relative_path) tuples."""
    source = Path(source_dir)
    files  = []
    for p in source.rglob('*.mp4'):
        rel = p.relative_to(source).as_posix()   # forward slashes for S3 key
        files.append((p, rel))
    return files


def upload_one(s3_client, transfer_cfg, bucket, prefix, local_path, rel_path):
    s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
    s3_client.upload_file(
        str(local_path),
        bucket,
        s3_key,
        Config=transfer_cfg,
    )
    return s3_key


def main():
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        print("Run scripts/copy_manifest_to_s2.py first.")
        return

    print(f"Scanning {source} ...")
    files = collect_files(source)
    if not files:
        print("No .mp4 files found. Is the copy step complete?")
        return

    total_size = sum(p.stat().st_size for p, _ in files)
    print(f"Found {len(files):,} videos  ({total_size / 1e9:.1f} GB)")
    print(f"Destination: s3://{args.bucket}/{args.prefix}/")

    if args.dry_run:
        print("\nDry run — not uploading.")
        # Show breakdown by subfolder
        from collections import Counter
        by_folder = Counter(rel.split('/')[0] for _, rel in files)
        for folder, count in sorted(by_folder.items()):
            print(f"  {folder:<20}  {count:>6} videos")
        return

    # Load checkpoint (resume from where we left off)
    completed     = load_checkpoint(args.checkpoint)
    already_done  = sum(1 for _, rel in files if rel in completed)
    remaining     = [(lp, rel) for lp, rel in files if rel not in completed]
    ckpt_lock     = Lock()

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  Already uploaded: {already_done:,}")
    print(f"  Remaining:        {len(remaining):,}")

    if not remaining:
        print("\nAll files already uploaded.")
        return

    confirm = input(f"\nUpload {len(remaining):,} files to s3://{args.bucket}/{args.prefix}/ ? [y/N] ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    s3_client    = build_s3_client()
    transfer_cfg = build_transfer_config(args.workers)

    start     = time.time()
    success   = 0
    failed    = 0
    failed_list = []
    print_lock = Lock()

    def upload_task(local_path, rel_path):
        try:
            upload_one(s3_client, transfer_cfg, args.bucket, args.prefix, local_path, rel_path)
            completed.add(rel_path)
            save_checkpoint(args.checkpoint, completed, ckpt_lock)
            return rel_path, None
        except Exception as e:
            return rel_path, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_task, lp, rel): rel for lp, rel in remaining}
        done_count = 0
        for future in as_completed(futures):
            rel_path, err = future.result()
            done_count += 1
            if err:
                failed += 1
                failed_list.append((rel_path, err))
                with print_lock:
                    print(f"  FAILED: {rel_path} — {err}")
            else:
                success += 1

            if done_count % 100 == 0 or done_count == len(remaining):
                elapsed = time.time() - start
                rate    = done_count / elapsed if elapsed > 0 else 0
                eta     = (len(remaining) - done_count) / rate if rate > 0 else 0
                with print_lock:
                    print(f"  [{done_count:>6}/{len(remaining)}]  "
                          f"success={success}  failed={failed}  "
                          f"rate={rate:.1f}/s  eta={eta/60:.0f}min")

    elapsed = time.time() - start
    print(f"\nUpload complete in {elapsed/60:.1f} min")
    print(f"  Success: {success:,}")
    print(f"  Failed:  {failed:,}")

    if failed_list:
        fail_log = args.checkpoint.replace('.json', '_failures.json')
        with open(fail_log, 'w') as f:
            json.dump(failed_list, f, indent=2)
        print(f"\nFailed files written to: {fail_log}")
        print("Re-run the script to retry — completed files are checkpointed and will be skipped.")
    else:
        print(f"\nAll videos at: s3://{args.bucket}/{args.prefix}/")
        print(f"Update DATA_PREFIX in launch_flow_sagemaker.py to '{args.prefix}/'")


if __name__ == '__main__':
    main()
