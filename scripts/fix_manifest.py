"""Remove AV1-encoded Pika videos that precompute_flow.py could not decode."""
import os
import pandas as pd

MANIFEST = 'data/flow_manifest.csv'

failed_filenames = {
    'Pika_12988.mp4', 'Pika_25579.mp4', 'Pika_35715.mp4', 'Pika_45064.mp4',
    'Pika_47909.mp4', 'Pika_51916.mp4', 'Pika_70229.mp4', 'Pika_71247.mp4',
    'Pika_79763.mp4', 'Pika_85669.mp4', 'Pika_90586.mp4', 'Pika_93121.mp4',
    'Pika_97921.mp4', 'Pika_98217.mp4',
}

df = pd.read_csv(MANIFEST)
before = len(df)

df['fname'] = df['path'].apply(os.path.basename)
removed = df[df['fname'].isin(failed_filenames)]
print(f"Rows to remove ({len(removed)}):")
print(removed[['path', 'generator', 'split']].to_string())

df = df[~df['fname'].isin(failed_filenames)].drop(columns='fname')
after = len(df)

df.to_csv(MANIFEST, index=False)
print(f"\nManifest updated: {before} -> {after} rows (removed {before - after})")
print("\nGenerator counts after removal:")
print(df['generator'].value_counts().to_string())
