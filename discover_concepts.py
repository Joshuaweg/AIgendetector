"""
OBJ-1: Concept Discovery Pipeline
Produces an empirical concept taxonomy from high-confidence Ninox 1.1-Flow predictions.

Pipeline:
  1. Inference pass  — batch forward on GenVideo manifest, save confidence scores
  2. Selection       — 2,500 AI-generated + 2,500 real at >90% confidence
  3. Attribution     — IG on video frames (flow maps held fixed), compact [24,16,16] repr
  4. Clustering      — UMAP(50 dims) + HDBSCAN
  5. Output          — YAML concept candidates + top-20 videos per cluster

Checkpoint/resume: inference and attribution phases both write progress to disk.
Re-running the script skips already-completed stages.

Dependencies (not in requirements.txt — install once):
  pip install umap-learn hdbscan pyyaml

Usage:
  python discover_concepts.py
  python discover_concepts.py --manifest data/flow_manifest.csv \\
      --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt \\
      --output _meta/concepts/ --n-steps 50 --target-per-class 2500
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_flow_maps
from full_scale_classifier import (
    FullClassifier, FullLatentEncoder, FullPatchEncoder,
    FlowEncoder, FlowVideoClassifier,
)

# ── constants ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
POOL_SIZE    = 16      # avg-pool spatial size for attribution compact repr
N_FRAMES     = 24
FRAME_SIZE   = 512
FLOW_H       = 64
FLOW_W       = 64


# ── model loading ──────────────────────────────────────────────────────────────

def load_flow_model(checkpoint_path: str, device: torch.device) -> FlowVideoClassifier:
    m = FlowVideoClassifier(
        FullLatentEncoder(), FullPatchEncoder(), FlowEncoder(), FullClassifier()
    ).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        import numpy.core.multiarray
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    acc = ckpt.get('best_accuracy')
    print(f"Loaded checkpoint epoch {ckpt.get('epoch')} — accuracy {acc:.2f}%" if acc else
          f"Loaded checkpoint epoch {ckpt.get('epoch')}")
    return m


# ── video loading ──────────────────────────────────────────────────────────────

def load_video_for_inference(path: str):
    """Returns (frames_np [T,H,W,3] float32 in [0,1], flow_maps tensor [T-1,6,H_f,W_f])."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None, None
    indices = set(int(round(i * (total - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES))
    frames, fi = [], 0
    while cap.isOpened() and len(frames) < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in indices:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame.astype(np.float32) / 255.0)
        fi += 1
    cap.release()
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    frames_np = np.stack(frames[:N_FRAMES])  # [24, 512, 512, 3]
    flow_maps = compute_flow_maps(frames_np, FLOW_H, FLOW_W)  # [23, 6, 64, 64]
    return frames_np, flow_maps


# ── inference dataset ──────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, records):
        self.records = records  # list of (path, label)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        frames_np, flow_maps = load_video_for_inference(path)
        if frames_np is None:
            return None
        frames_t = torch.from_numpy(frames_np).unsqueeze(0)   # [1, 24, 512, 512, 3]
        flow_t   = flow_maps.unsqueeze(0)                      # [1, 23, 6, 64, 64]
        return {'path': path, 'label': label, 'frames': frames_t, 'flow': flow_t}


def inference_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'path':   [b['path']  for b in batch],
        'label':  [b['label'] for b in batch],
        'frames': torch.cat([b['frames'] for b in batch], dim=0),
        'flow':   torch.cat([b['flow']   for b in batch], dim=0),
    }


# ── Phase 1: inference pass ────────────────────────────────────────────────────

def run_inference(records, model, device, batch_size, cache_path):
    """Run model inference on all records. Returns list of result dicts."""
    if cache_path.exists():
        print(f"[inference] Loading cached results from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"[inference] Running on {len(records)} videos (batch_size={batch_size})")
    ds     = InferenceDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0,
                        collate_fn=inference_collate, shuffle=False)
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            frames = batch['frames'].to(device)
            flow   = batch['flow'].to(device)
            logits = model(frames, flow)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            for j, path in enumerate(batch['path']):
                results.append({
                    'path':      path,
                    'label':     batch['label'][j],
                    'pred':      int(preds[j]),
                    'conf':      float(probs[j][preds[j]]),
                    'prob_ai':   float(probs[j][0]),
                    'prob_real': float(probs[j][1]),
                })
            if (i + 1) % 50 == 0:
                print(f"  {sum(1 for r in results if r['path'])}/{len(records)} done")
            del frames, flow, logits, probs

    print(f"[inference] Complete: {len(results)} results")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(results, f)
    return results


# ── Phase 2: selection ─────────────────────────────────────────────────────────

def select_high_confidence(results, conf_threshold, target_per_class):
    """Return balanced high-confidence selection: target_per_class AI + target_per_class real."""
    ai_pool   = [r for r in results if r['pred'] == 0 and r['conf'] >= conf_threshold]
    real_pool = [r for r in results if r['pred'] == 1 and r['conf'] >= conf_threshold]
    print(f"[select] High-conf pool: {len(ai_pool)} AI, {len(real_pool)} real "
          f"(threshold={conf_threshold:.0%})")

    rng = np.random.default_rng(seed=42)
    ai_sel   = rng.choice(ai_pool,   min(target_per_class, len(ai_pool)),   replace=False).tolist()
    real_sel = rng.choice(real_pool, min(target_per_class, len(real_pool)), replace=False).tolist()
    selected = ai_sel + real_sel
    rng.shuffle(selected)
    print(f"[select] Selected: {len(ai_sel)} AI + {len(real_sel)} real = {len(selected)} total")
    return selected


# ── IG wrapper ─────────────────────────────────────────────────────────────────

class _VideoOnlyWrapper(nn.Module):
    """Wraps FlowVideoClassifier for IG: flow_maps are held constant."""
    def __init__(self, model: FlowVideoClassifier, flow_maps: torch.Tensor):
        super().__init__()
        self.model      = model
        self.flow_maps  = flow_maps  # [1, T-1, 6, H_f, W_f]

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return self.model(videos, self.flow_maps)


def compact_attribution(attr: torch.Tensor) -> np.ndarray:
    """
    Compress IG attribution to a fixed-size vector for clustering.

    Input:  attr [1, T, H, W, C]
    Output: flat [T * POOL_SIZE * POOL_SIZE] numpy float32

    Steps: abs-sum over C → permute → avg-pool2d → flatten
    """
    a = attr.squeeze(0).abs().sum(dim=-1)        # [T, H, W]
    a = a.unsqueeze(0)                           # [1, T, H, W] — fake batch for pool
    a = a.reshape(1, -1, FRAME_SIZE, FRAME_SIZE) # [1, T, H, W]
    a = F.avg_pool2d(a, kernel_size=FRAME_SIZE // POOL_SIZE)  # [1, T, P, P]
    return a.squeeze(0).cpu().numpy().flatten().astype(np.float32)


# ── Phase 3: attribution pass ──────────────────────────────────────────────────

def run_attributions(selected, model, device, n_steps, output_dir):
    """
    Run IG attribution on each selected video. Returns (paths, labels, compact_vectors).
    Checkpoint/resume: skips videos whose .npy already exists.
    """
    attr_dir = output_dir / 'attributions'
    attr_dir.mkdir(parents=True, exist_ok=True)

    ig_model   = None  # rebuilt per video (different flow_maps each time)
    paths, labels, vectors = [], [], []

    # Resume: load already-done
    done_index_path = output_dir / 'attribution_index.json'
    done = {}
    if done_index_path.exists():
        with open(done_index_path) as f:
            done = {e['path']: e for e in json.load(f)}
        print(f"[attr] Resuming — {len(done)} already computed")

    todo = [r for r in selected if r['path'] not in done]
    print(f"[attr] Computing IG for {len(todo)} videos (n_steps={n_steps})")

    for i, record in enumerate(todo):
        path = record['path']
        npy_name = f"{abs(hash(path)) % (10**9)}.npy"
        npy_path = attr_dir / npy_name

        frames_np, flow_maps = load_video_for_inference(path)
        if frames_np is None:
            print(f"  [skip] cannot load: {path}")
            continue

        frames_t = torch.from_numpy(frames_np).unsqueeze(0).to(device)  # [1,T,H,W,C]
        flow_t   = flow_maps.unsqueeze(0).to(device)                     # [1,T-1,6,H_f,W_f]
        frames_t.requires_grad_(True)

        wrapper  = _VideoOnlyWrapper(model, flow_t)
        ig       = IntegratedGradients(wrapper)
        baseline = torch.zeros_like(frames_t)

        with torch.no_grad():
            logits = model(frames_t, flow_t)
            pred   = int(torch.argmax(logits, dim=1).item())

        try:
            attr, _ = ig.attribute(
                frames_t, baseline,
                target=pred,
                n_steps=n_steps,
                return_convergence_delta=True,
                internal_batch_size=1,
            )
        except Exception as e:
            print(f"  [skip] IG failed for {path}: {e}")
            del frames_t, flow_t, wrapper
            torch.cuda.empty_cache()
            continue

        vec = compact_attribution(attr.detach())
        np.save(npy_path, vec)

        entry = {
            'path': path, 'label': record['label'], 'pred': pred,
            'conf': record['conf'], 'npy': str(npy_path),
        }
        done[path] = entry

        del attr, frames_t, flow_t, wrapper, ig
        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            with open(done_index_path, 'w') as f:
                json.dump(list(done.values()), f)
            print(f"  checkpoint: {i+1}/{len(todo)} done")

    with open(done_index_path, 'w') as f:
        json.dump(list(done.values()), f)

    # Compile results
    for entry in done.values():
        npy = np.load(entry['npy'])
        paths.append(entry['path'])
        labels.append(entry['label'])
        vectors.append(npy)

    return paths, labels, np.stack(vectors)


# ── Phase 4: clustering ────────────────────────────────────────────────────────

def cluster_attributions(vectors: np.ndarray, n_umap_components: int = 50):
    try:
        import umap
        import hdbscan as hdbscan_lib
    except ImportError:
        print("Missing: pip install umap-learn hdbscan")
        sys.exit(1)

    print(f"[cluster] UMAP: {vectors.shape} → {n_umap_components} dims")
    reducer    = umap.UMAP(n_components=n_umap_components, metric='cosine',
                           random_state=42, verbose=False)
    embeddings = reducer.fit_transform(vectors)

    print("[cluster] HDBSCAN clustering")
    clusterer  = hdbscan_lib.HDBSCAN(min_cluster_size=50, metric='euclidean',
                                      prediction_data=True)
    labels     = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = sum(1 for l in labels if l == -1)
    print(f"[cluster] {n_clusters} clusters, {n_noise} noise points")
    return labels, embeddings, clusterer


def top_videos_per_cluster(cluster_id, labels, embeddings, paths, video_labels, top_k=20):
    """Return top_k videos closest to cluster centroid."""
    mask     = np.where(np.array(labels) == cluster_id)[0]
    centroid = embeddings[mask].mean(axis=0)
    dists    = np.linalg.norm(embeddings[mask] - centroid, axis=1)
    ranked   = mask[np.argsort(dists)][:top_k]
    return [{'path': paths[i], 'label': int(video_labels[i]), 'dist': float(dists[np.where(mask == i)[0][0]])}
            for i in ranked]


# ── Phase 5: YAML output ───────────────────────────────────────────────────────

PLACEHOLDER_NAMES = {
    0: 'concept_cluster_0',
    1: 'concept_cluster_1',
    2: 'concept_cluster_2',
    3: 'concept_cluster_3',
    4: 'concept_cluster_4',
    5: 'concept_cluster_5',
    6: 'concept_cluster_6',
    7: 'concept_cluster_7',
    8: 'concept_cluster_8',
    9: 'concept_cluster_9',
    10: 'concept_cluster_10',
    11: 'concept_cluster_11',
}


def write_concept_yaml(cluster_ids, labels_arr, embeddings, paths, video_labels, output_dir):
    try:
        import yaml
    except ImportError:
        print("Missing: pip install pyyaml")
        sys.exit(1)

    candidates = []
    unique_clusters = sorted(c for c in set(labels_arr) if c != -1)
    for cid in unique_clusters:
        mask       = np.where(labels_arr == cid)[0]
        n_videos   = len(mask)
        ai_count   = sum(1 for i in mask if video_labels[i] == 0)
        real_count = sum(1 for i in mask if video_labels[i] == 1)
        top20      = top_videos_per_cluster(cid, labels_arr, embeddings, paths, video_labels, top_k=20)

        candidates.append({
            'cluster_id':   int(cid),
            'name':         PLACEHOLDER_NAMES.get(cid, f'concept_cluster_{cid}'),
            'description':  '[TODO: inspect top-20 videos and describe the spatial/temporal pattern]',
            'n_videos':     int(n_videos),
            'ai_count':     int(ai_count),
            'real_count':   int(real_count),
            'typical_onset': '[TODO: early / mid / late / consistent — from top-20 inspection]',
            'primary_region': '[TODO: face / background / boundary / motion region — from top-20]',
            'top_20_videos': top20,
        })

    out_path = output_dir / 'concept_candidates.yaml'
    with open(out_path, 'w') as f:
        yaml.dump({'concept_candidates': candidates}, f,
                  allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"[output] Concept candidates → {out_path}")
    return candidates


def write_cluster_json(labels_arr, paths, video_labels, embeddings, output_dir):
    records = [
        {'path': p, 'label': int(video_labels[i]), 'cluster': int(labels_arr[i]),
         'umap_x': float(embeddings[i, 0]), 'umap_y': float(embeddings[i, 1])}
        for i, p in enumerate(paths)
    ]
    out = output_dir / 'cluster_assignments.json'
    with open(out, 'w') as f:
        json.dump(records, f)
    print(f"[output] Cluster assignments → {out}")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',    default='data/flow_manifest.csv',
                   help='CSV with columns: path,label,generator,split')
    p.add_argument('--checkpoint',  default='flow_stage2_checkpoints/checkpoint_epoch_0004.pt')
    p.add_argument('--output',      default='_meta/concepts/')
    p.add_argument('--conf',        type=float, default=0.90,
                   help='Minimum confidence threshold for selection')
    p.add_argument('--target',      type=int,   default=2500,
                   help='Target videos per class (AI + real)')
    p.add_argument('--n-steps',     type=int,   default=50,
                   help='IG integration steps (50 is sufficient for clustering)')
    p.add_argument('--batch-size',  type=int,   default=4,
                   help='Inference batch size')
    p.add_argument('--umap-dims',   type=int,   default=50)
    p.add_argument('--skip-attr',   action='store_true',
                   help='Skip attribution phase (cluster from existing .npy files only)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    model = load_flow_model(args.checkpoint, device)

    # ── Read manifest ───────────────────────────────────────────────────────────
    import csv
    records = []
    with open(args.manifest, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['label'])  # 0=AI, 1=real
            records.append((row['path'], label))
    print(f"[manifest] {len(records)} videos")

    # ── Phase 1: inference ──────────────────────────────────────────────────────
    inference_cache = output_dir / 'inference_results.json'
    results = run_inference(records, model, device, args.batch_size, inference_cache)

    # ── Phase 2: selection ──────────────────────────────────────────────────────
    selected = select_high_confidence(results, args.conf, args.target)
    sel_path = output_dir / 'selected_videos.json'
    with open(sel_path, 'w') as f:
        json.dump(selected, f)

    if not selected:
        print("No high-confidence videos found. Lower --conf threshold or expand manifest.")
        return

    # ── Phase 3: attributions ───────────────────────────────────────────────────
    if args.skip_attr:
        print("[attr] --skip-attr: loading existing attributions")
        idx_path = output_dir / 'attribution_index.json'
        if not idx_path.exists():
            print("No attribution_index.json found. Run without --skip-attr first.")
            return
        with open(idx_path) as f:
            done = json.load(f)
        paths       = [e['path']  for e in done]
        video_labels = [e['label'] for e in done]
        vectors     = np.stack([np.load(e['npy']) for e in done])
    else:
        paths, video_labels, vectors = run_attributions(
            selected, model, device, args.n_steps, output_dir
        )

    print(f"[attr] Attribution matrix: {vectors.shape}")

    # ── Phase 4: clustering ─────────────────────────────────────────────────────
    labels_arr, embeddings, _ = cluster_attributions(vectors, args.umap_dims)

    # ── Phase 5: output ─────────────────────────────────────────────────────────
    write_cluster_json(labels_arr, paths, video_labels, embeddings, output_dir)
    write_concept_yaml(labels_arr, labels_arr, embeddings, paths, video_labels, output_dir)

    # Summary
    unique = sorted(c for c in set(labels_arr) if c != -1)
    print(f"\n{'='*60}")
    print(f"CONCEPT DISCOVERY COMPLETE")
    print(f"  Clusters found:   {len(unique)}")
    print(f"  Noise points:     {sum(1 for l in labels_arr if l == -1)}")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"\nNext step: open {output_dir/'concept_candidates.yaml'}")
    print("  For each cluster, watch the top-20 videos alongside their IG overlays.")
    print("  Fill in 'name', 'description', 'typical_onset', 'primary_region'.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
