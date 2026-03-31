"""
Download MVAD AI-generated video ZIPs from HuggingFace, extract, and rename
to ai_<generator>_<id>.mp4 format for the detector dataset.

Output structure:
  F:/MVAD/downloads/     -- raw ZIPs kept for re-extraction
  F:/MVAD/videos/        -- extracted + renamed mp4s ready for training
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ID    = "mengxuebobo/MVAD"
DOWNLOAD_DIR = Path("F:/MVAD/downloads")
OUTPUT_DIR   = Path("F:/MVAD/videos")

# All fake-video ZIPs available (fake_fake = fake video + fake audio)
ZIPS = {
    "kling1_6":   "train/fake_fake/direct/kling1_6.zip",
    "emu3":       "train/fake_fake/indirect/emu3.zip",
    "gen3":       "train/fake_fake/indirect/gen3.zip",
    "haiper":     "train/fake_fake/indirect/haiper.zip",
    "moonvalley": "train/fake_fake/indirect/moonvalley.zip",
    "noisee":     "train/fake_fake/indirect/noisee.zip",
    "pika":       "train/fake_fake/indirect/pika.zip",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def short_hash(path: str, length=8) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:length]


def extract_and_rename(zip_path: Path, generator: str, out_dir: Path):
    """Extract zip, rename all mp4s to ai_<generator>_<hash>.mp4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = 0
    skipped  = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        mp4s = [n for n in zf.namelist() if n.lower().endswith('.mp4')]
        print(f"  {generator}: {len(mp4s)} mp4s in zip")

        for name in mp4s:
            h = short_hash(name)
            dest = out_dir / f"ai_{generator}_{h}.mp4"

            if dest.exists():
                skipped += 1
                continue

            # Extract to temp name, then rename
            data = zf.read(name)
            dest.write_bytes(data)
            accepted += 1

            if accepted % 500 == 0:
                print(f"    {accepted} extracted...")

    print(f"  {generator}: {accepted} new, {skipped} already existed")
    return accepted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_videos = 0

    for generator, repo_path in ZIPS.items():
        zip_dest = DOWNLOAD_DIR / f"{generator}.zip"

        # --- Download ---
        if zip_dest.exists():
            size_gb = zip_dest.stat().st_size / 1e9
            print(f"[{generator}] Already downloaded ({size_gb:.2f} GB), skipping download")
        else:
            print(f"[{generator}] Downloading {repo_path} ...")
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(DOWNLOAD_DIR),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves at local_dir/filename path, move to flat name
            downloaded = DOWNLOAD_DIR / repo_path
            if downloaded.exists():
                downloaded.rename(zip_dest)
            size_gb = zip_dest.stat().st_size / 1e9
            print(f"[{generator}] Downloaded ({size_gb:.2f} GB)")

        # --- Extract + rename ---
        print(f"[{generator}] Extracting...")
        n = extract_and_rename(zip_dest, generator, OUTPUT_DIR)
        total_videos += n

    # --- Final stats ---
    all_mp4s = list(OUTPUT_DIR.glob("ai_*.mp4"))
    print(f"\n=== Done ===")
    print(f"Total videos in {OUTPUT_DIR}: {len(all_mp4s)}")
    by_gen = {}
    for f in all_mp4s:
        gen = f.stem.split('_')[1]
        by_gen[gen] = by_gen.get(gen, 0) + 1
    for gen, count in sorted(by_gen.items()):
        print(f"  {gen:20s}: {count}")


if __name__ == "__main__":
    main()
