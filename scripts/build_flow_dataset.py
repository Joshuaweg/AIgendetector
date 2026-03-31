"""
Build flow training manifest CSV.

Scans all video sources, applies a 1000-video cap per generator,
balances real vs fake, and writes data/flow_manifest.csv.

Usage:
    python scripts/build_flow_dataset.py
"""

import random
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.model_selection import train_test_split

random.seed(42)

# ── Data sources ──────────────────────────────────────────────────────────────

FAKE_SOURCES = OrderedDict([
    # GenVideo-Train AI-Generated (flat dirs)
    ('dynamiccrafter', 'F:/Gen-Video/GenVideo-Train/AI-Generated/DynamicCrafter/*.mp4'),
    ('i2vgen_xl',      'F:/Gen-Video/GenVideo-Train/AI-Generated/I2VGEN_XL/*.mp4'),
    ('latte',          'F:/Gen-Video/GenVideo-Train/AI-Generated/Latte/*.mp4'),
    ('opensora',       'F:/Gen-Video/GenVideo-Train/AI-Generated/OpenSora/*.mp4'),
    ('pika_genvideo',  'F:/Gen-Video/GenVideo-Train/AI-Generated/pika/*.mp4'),
    ('sd',             'F:/Gen-Video/GenVideo-Train/AI-Generated/SD/*.mp4'),
    ('seine',          'F:/Gen-Video/GenVideo-Train/AI-Generated/SEINE/*.mp4'),
    ('svd',            'F:/Gen-Video/GenVideo-Train/AI-Generated/SVD/*.mp4'),
    ('videocrafter',   'F:/Gen-Video/GenVideo-Train/AI-Generated/VideoCrafter/*.mp4'),
    ('zeroscope',      'F:/Gen-Video/GenVideo-Train/AI-Generated/ZeroScope/*.mp4'),
    # MVAD prefix-based
    ('emu3',       'F:/MVAD/videos/ai_emu3_*.mp4'),
    ('gen3',       'F:/MVAD/videos/ai_gen3_*.mp4'),
    ('haiper',     'F:/MVAD/videos/ai_haiper_*.mp4'),
    ('kling1_6',   'F:/MVAD/videos/ai_kling1_6_*.mp4'),
    ('moonvalley', 'F:/MVAD/videos/ai_moonvalley_*.mp4'),
    ('noisee',     'F:/MVAD/videos/ai_noisee_*.mp4'),
    ('pika_mvad',  'F:/MVAD/videos/ai_pika_*.mp4'),
    # Sora (val set)
    ('sora', 'F:/Gen-Video/GenVideo-Val/GenVideo-Val/Fake/Sora/*.mp4'),
    # Veo3
    ('veo3', 'C:/Users/joshu/Downloads/Veo3/*.mp4'),
])

REAL_SOURCES = OrderedDict([
    ('kinetics', 'F:/Gen-Video/GenVideo-Train/Real/Kinetics/*.mp4'),
    ('youku',    'F:/Gen-Video/GenVideo-Train/Real/Youku_1M_10s/*/*.mp4'),
])

CAP = 1000
OUTPUT = Path('data/flow_manifest.csv')


def scan(glob_pattern: str) -> list[str]:
    """Glob for mp4 files, return sorted list of path strings."""
    parts = glob_pattern.replace('\\', '/')
    # Split into root and pattern
    p = Path(parts)
    # Use the parent of the first wildcard component as anchor
    anchor = p.parts[0]
    for i, part in enumerate(p.parts):
        if '*' in part or '?' in part:
            anchor = str(Path(*p.parts[:i]))
            pattern = str(Path(*p.parts[i:]))
            break
    else:
        anchor = str(p.parent)
        pattern = p.name

    results = sorted(str(f) for f in Path(anchor).glob(pattern))
    return results


def main():
    rows = []

    # ── Fake sources ──────────────────────────────────────────────────────
    print("Scanning fake sources...")
    for gen, pattern in FAKE_SOURCES.items():
        paths = scan(pattern)
        total = len(paths)
        if total > CAP:
            paths = sorted(random.sample(paths, CAP))
        print(f"  {gen:20s}  found={total:>6,d}  sampled={len(paths):>6,d}")
        for p in paths:
            rows.append({'path': p, 'label': 0, 'generator': gen})

    n_fake = sum(1 for r in rows if r['label'] == 0)
    print(f"\nTotal fake videos: {n_fake:,d}")

    # ── Real sources ──────────────────────────────────────────────────────
    print("\nScanning real sources...")
    real_pools = {}
    for gen, pattern in REAL_SOURCES.items():
        paths = scan(pattern)
        real_pools[gen] = paths
        print(f"  {gen:20s}  found={len(paths):>6,d}")

    # Target: n_fake total reals, split 50/50 kinetics/youku
    half = n_fake // 2
    kinetics_all = real_pools['kinetics']
    youku_all = real_pools['youku']

    if len(kinetics_all) >= half:
        kinetics_sample = sorted(random.sample(kinetics_all, half))
        youku_need = n_fake - half
    else:
        kinetics_sample = kinetics_all
        youku_need = n_fake - len(kinetics_all)

    youku_sample = sorted(random.sample(youku_all, min(youku_need, len(youku_all))))

    print(f"  kinetics sampled: {len(kinetics_sample):,d}")
    print(f"  youku sampled:    {len(youku_sample):,d}")

    for p in kinetics_sample:
        rows.append({'path': p, 'label': 1, 'generator': 'kinetics'})
    for p in youku_sample:
        rows.append({'path': p, 'label': 1, 'generator': 'youku'})

    # ── Build dataframe ───────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print(f"\nTotal videos: {len(df):,d}")

    # ── Stratified split 80/10/10 by generator ────────────────────────────
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['generator']
    )
    # Second split: 50/50 of temp -> 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['generator']
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── Write CSV ─────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT, index=False)
    print(f"\nManifest written to: {OUTPUT}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"\nTotal videos: {len(final_df):,d}")
    print(f"  Fake (label=0): {(final_df['label']==0).sum():,d}")
    print(f"  Real (label=1): {(final_df['label']==1).sum():,d}")

    print(f"\nSplit sizes:")
    for split in ['train', 'val', 'test']:
        n = (final_df['split'] == split).sum()
        print(f"  {split:6s}: {n:>6,d}")

    print(f"\nPer-generator counts:")
    gen_counts = final_df.groupby(['generator', 'label']).size().reset_index(name='count')
    for _, row in gen_counts.iterrows():
        label_str = 'fake' if row['label'] == 0 else 'real'
        print(f"  {row['generator']:20s} ({label_str}): {row['count']:>6,d}")


if __name__ == '__main__':
    main()
