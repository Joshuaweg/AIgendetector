"""
Phase 3 — TCAV Concept Probing
Identifies which concepts Ninox 1.1-Flow has internalized and in which transformer layers.

Algorithm:
  1. Layer selection  — quick single-split probe on candidate layers; pick highest sign count
  2. Concept validation — 5-split resampled TCAV on best layer; validate against thresholds
  3. Dual-pathway analysis — compare linear probe accuracy at frame vs. flow pathway endpoints

Outputs (written to --output dir):
  tcav_results.json        per-concept sign counts + CAV accuracy per split
  layer_selection.json     best layer chosen per concept
  dual_pathway.json        CAV accuracy at frame vs. flow pathway layers
  validated_concepts.yaml  concepts that pass all validation thresholds

Usage:
  python tcav_interpret.py \
      --checkpoint flow_stage2_checkpoints/checkpoint_epoch_0004.pt \
      --probes _meta/probes/ \
      --test-manifest data/flow_manifest.csv \
      --output _meta/tcav/

Thresholds (from ninox_tcav_plan.md):
  Sign count     > 0.65  (concept is predictive of AI class)
  Sign count std < 0.10  (consistent across probe splits)
  CAV accuracy   > 0.80  (concept is linearly separable in activation space)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from full_scale_classifier import (
    FullClassifier, FullLatentEncoder, FullPatchEncoder,
    FlowEncoder, FlowVideoClassifier,
)
from tcav_probes import ProbeVideoDataset

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

# ── layer config ───────────────────────────────────────────────────────────────

# Three representative transformer depths to probe (0-indexed, 12 layers total)
PROBE_LAYERS = [
    'classifier.transformer_encoder.layers.3',    # early-mid (layer 4)
    'classifier.transformer_encoder.layers.7',    # mid (layer 8)
    'classifier.transformer_encoder.layers.11',   # late (layer 12)
]

# Pathway-specific layers for dual-pathway analysis
FRAME_PATHWAY_LAYER = 'patch_encoder.norm'   # end of tubelet encoding
FLOW_PATHWAY_LAYER  = 'flow_encoder.norm'    # end of flow encoding

# Validation thresholds
SIGN_COUNT_MIN      = 0.65
SIGN_COUNT_STD_MAX  = 0.10
CAV_ACCURACY_MIN    = 0.80
TARGET_CLASS        = 0     # 0 = AI-generated


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> FlowVideoClassifier:
    model = FlowVideoClassifier(
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
    model.load_state_dict(sd)
    model.eval()
    acc = ckpt.get('best_accuracy')
    print(f"Model loaded: epoch {ckpt.get('epoch')}"
          + (f", accuracy {acc:.2f}%" if acc else ""))
    return model


def _get_nested(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Navigate dotted path like 'classifier.transformer_encoder.layers.3'."""
    m = model
    for part in path.split('.'):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


# ── activation extraction ──────────────────────────────────────────────────────

def extract_activations(
    model: FlowVideoClassifier,
    layer_path: str,
    dataset: ProbeVideoDataset,
    device: torch.device,
) -> np.ndarray:
    """
    Extract layer activations for all examples in dataset.
    Returns float32 array of shape [N, 768].

    For transformer layers (classifier.transformer_encoder.layers.N):
      runs the model manually up to that layer — avoids hook unreliability.
    For pathway layers (patch_encoder.norm, flow_encoder.norm):
      uses a forward hook since those are not inside the token-concat block.
    """
    if layer_path.startswith('classifier.transformer_encoder.layers.'):
        return _extract_transformer_acts(model, layer_path, dataset, device)
    return _extract_hook_acts(model, layer_path, dataset, device)


def _extract_transformer_acts(
    model: FlowVideoClassifier,
    layer_path: str,
    dataset: ProbeVideoDataset,
    device: torch.device,
) -> np.ndarray:
    layer_idx = int(layer_path.rsplit('.', 1)[-1])
    layers    = model.classifier.transformer_encoder.layers
    acts      = []

    model.eval()
    with torch.no_grad():
        for frames, flow in dataset:
            try:
                frames = frames.unsqueeze(0).to(device)   # [1,T,H,W,3]
                flow   = flow.unsqueeze(0).to(device)      # [1,T-1,6,Hf,Wf]
                latents = model.latent_encoder(frames)
                tokens  = model.patch_encoder(latents)
                del latents
                flow_tok = model.flow_encoder(flow)
                tokens   = torch.cat([tokens, flow_tok], dim=1)  # [1, N+T-1, 768]
                del flow_tok
                for i in range(layer_idx + 1):
                    tokens = layers[i](tokens)
                # Mean-pool over sequence dim → [768]
                acts.append(tokens.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32))
            except Exception:
                pass

    return np.stack(acts) if acts else np.zeros((0, 768), dtype=np.float32)


def _extract_hook_acts(
    model: FlowVideoClassifier,
    layer_path: str,
    dataset: ProbeVideoDataset,
    device: torch.device,
) -> np.ndarray:
    layer     = _get_nested(model, layer_path)
    collected = []

    def _hook(module, inp, out):
        t = out.detach()
        if t.dim() == 3:
            t = t.mean(dim=1)   # [B, D]
        elif t.dim() > 2:
            t = t.view(t.shape[0], -1)
        collected.append(t.squeeze(0).cpu().numpy().astype(np.float32))

    handle = layer.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for frames, flow in dataset:
            try:
                model(
                    frames.unsqueeze(0).to(device),
                    flow.unsqueeze(0).to(device),
                )
            except Exception:
                pass
    handle.remove()
    return np.stack(collected) if collected else np.zeros((0, 768), dtype=np.float32)


# ── CAV training ───────────────────────────────────────────────────────────────

def train_cav(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray | None, float]:
    """
    Train an L2-regularized logistic regression (CAV) on positive vs. negative activations.
    Returns (cav_unit_vector [768], held-out accuracy). Returns (None, 0.0) if too few examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n = min(len(pos_acts), len(neg_acts))
    if n < 4:
        return None, 0.0

    rng   = np.random.default_rng(seed)
    pos   = pos_acts[rng.choice(len(pos_acts), n, replace=False)]
    neg   = neg_acts[rng.choice(len(neg_acts), n, replace=False)]
    X     = np.concatenate([pos, neg])
    y     = np.array([1] * n + [0] * n)

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_s, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_val, y_val))

    cav = clf.coef_[0].copy()
    cav /= np.linalg.norm(cav) + 1e-10
    return cav.astype(np.float32), acc


# ── TCAV sign count ────────────────────────────────────────────────────────────

def compute_sign_count(
    model: FlowVideoClassifier,
    layer_path: str,
    cav: np.ndarray,
    test_dataset: ProbeVideoDataset,
    device: torch.device,
    target_class: int = TARGET_CLASS,
) -> float:
    """
    TCAV sign count: fraction of test examples where the directional derivative
    of the target class output along the CAV direction is positive.

    Only implemented for transformer layers — returns 0.0 for pathway layers
    (those use CAV accuracy as the interpretability metric instead).
    """
    if not layer_path.startswith('classifier.transformer_encoder.layers.'):
        return 0.0

    layer_idx  = int(layer_path.rsplit('.', 1)[-1])
    layers     = model.classifier.transformer_encoder.layers
    n_layers   = len(layers)
    classifier = model.classifier
    cav_t      = torch.tensor(cav, dtype=torch.float32, device=device)

    positive = 0
    total    = 0

    model.eval()
    for frames, flow in test_dataset:
        try:
            frames = frames.unsqueeze(0).to(device)
            flow   = flow.unsqueeze(0).to(device)

            # Build token sequence up to (and including) layer_idx without gradients
            with torch.no_grad():
                latents  = model.latent_encoder(frames)
                t_tok    = model.patch_encoder(latents)
                del latents
                f_tok    = model.flow_encoder(flow)
                tokens   = torch.cat([t_tok, f_tok], dim=1)   # [1, N, 768]
                del t_tok, f_tok
                for i in range(layer_idx + 1):
                    tokens = layers[i](tokens)
            # tokens: [1, seq_len, 768] — activation at layer_idx

            # Detach and re-enable gradient for directional derivative computation
            act = tokens.detach().requires_grad_(True)  # [1, seq_len, 768]

            # Forward from layer_idx+1 to output
            x = act
            for i in range(layer_idx + 1, n_layers):
                x = layers[i](x)
            pooled = x.mean(dim=1)                          # [1, 768]
            logits = classifier.fc(pooled)
            logits = (logits - logits.max(dim=1, keepdim=True)[0]).clamp(-15, 15)
            score  = logits[0, target_class]

            score.backward()

            if act.grad is not None:
                # Mean-pool gradient over sequence dim → [768]
                grad        = act.grad[0].mean(dim=0)
                directional = torch.dot(grad, cav_t).item()
                if directional > 0:
                    positive += 1
            total += 1

        except Exception:
            total += 1

    return float(positive / total) if total > 0 else 0.0


# ── layer selection (quick single split) ──────────────────────────────────────

def select_best_layer(
    concept_name: str,
    pos_records: list,
    neg_records: list,
    model: FlowVideoClassifier,
    candidate_layers: list,
    test_records: list,
    device: torch.device,
    n_probe: int = 30,
    n_test: int = 20,
) -> str:
    """Single-split probe on candidate layers. Returns layer with highest sign count."""
    print(f"\n[layer selection] {concept_name}")
    pos_ds  = ProbeVideoDataset(pos_records[:n_probe])
    neg_ds  = ProbeVideoDataset(neg_records[:n_probe])
    test_ds = ProbeVideoDataset(test_records[:n_test])

    best_layer = candidate_layers[0]
    best_sc    = -1.0

    for layer_path in candidate_layers:
        pos_acts = extract_activations(model, layer_path, pos_ds, device)
        neg_acts = extract_activations(model, layer_path, neg_ds, device)
        cav, acc = train_cav(pos_acts, neg_acts)
        if cav is None:
            print(f"  {layer_path.split('.')[-1]}: insufficient probes")
            continue
        sc = compute_sign_count(model, layer_path, cav, test_ds, device)
        print(f"  layer {layer_path.split('.')[-1]}: sc={sc:.3f}  cav_acc={acc:.3f}")
        if sc > best_sc:
            best_sc    = sc
            best_layer = layer_path

    print(f"  → best layer: {best_layer.split('.')[-1]}  (sc={best_sc:.3f})")
    return best_layer


# ── full concept validation (n splits) ────────────────────────────────────────

def validate_concept(
    concept_name: str,
    pos_records: list,
    neg_records: list,
    model: FlowVideoClassifier,
    layer_path: str,
    test_records: list,
    device: torch.device,
    n_splits: int = 5,
) -> dict:
    """
    Validate a concept over n_splits resampled probe splits.
    A concept is validated when:
      mean_sign_count > 0.65  AND  std < 0.10  AND  mean_cav_accuracy > 0.80
    """
    n         = min(len(pos_records), len(neg_records))
    if n < 10:
        return {'concept': concept_name, 'validated': False, 'reason': f'too few probes (n={n})'}

    test_ds    = ProbeVideoDataset(test_records)
    sign_counts = []
    cav_accs    = []

    for split_i in range(n_splits):
        rng     = np.random.default_rng(seed=split_i * 137 + 7)
        pos_sub = [pos_records[i] for i in rng.choice(len(pos_records),
                                                        min(n, len(pos_records)),
                                                        replace=False)]
        neg_sub = [neg_records[i] for i in rng.choice(len(neg_records),
                                                        min(n, len(neg_records)),
                                                        replace=False)]
        pos_acts = extract_activations(model, layer_path, ProbeVideoDataset(pos_sub), device)
        neg_acts = extract_activations(model, layer_path, ProbeVideoDataset(neg_sub), device)

        cav, acc = train_cav(pos_acts, neg_acts, seed=split_i)
        if cav is None:
            continue
        cav_accs.append(acc)

        sc = compute_sign_count(model, layer_path, cav, test_ds, device)
        sign_counts.append(sc)
        print(f"  split {split_i + 1}/{n_splits}: sc={sc:.3f}  cav_acc={acc:.3f}")

    if not sign_counts:
        return {'concept': concept_name, 'validated': False, 'reason': 'all splits failed'}

    mean_sc  = float(np.mean(sign_counts))
    std_sc   = float(np.std(sign_counts))
    mean_acc = float(np.mean(cav_accs)) if cav_accs else 0.0

    validated = (
        mean_sc  > SIGN_COUNT_MIN
        and std_sc  < SIGN_COUNT_STD_MAX
        and mean_acc > CAV_ACCURACY_MIN
    )

    return {
        'concept':           concept_name,
        'layer':             layer_path,
        'validated':         validated,
        'mean_sign_count':   round(mean_sc, 4),
        'std_sign_count':    round(std_sc, 4),
        'mean_cav_accuracy': round(mean_acc, 4),
        'n_splits':          len(sign_counts),
        'sign_counts':       [round(s, 4) for s in sign_counts],
        'cav_accuracies':    [round(a, 4) for a in cav_accs],
    }


# ── dual-pathway analysis ──────────────────────────────────────────────────────

def run_dual_pathway(
    validated_results: list,
    model: FlowVideoClassifier,
    probe_dir: Path,
    test_records: list,
    device: torch.device,
    n_probe: int = 30,
) -> dict:
    """
    For each validated concept, compare linear CAV accuracy at:
      - patch_encoder.norm   (frame-only pathway)
      - flow_encoder.norm    (flow-only pathway)
    Motion concepts should be more decodable from the flow pathway;
    structural concepts should be more decodable from the frame pathway.
    Pathway mismatches are scientifically interesting — document them.
    """
    pathway_layers = {
        'frame': FRAME_PATHWAY_LAYER,
        'flow':  FLOW_PATHWAY_LAYER,
    }
    results = {}

    for res in validated_results:
        concept  = res['concept']
        probe_f  = probe_dir / f'probe_{concept}.json'
        if not probe_f.exists():
            continue
        with open(probe_f) as f:
            probe = json.load(f)
        pos_ds   = ProbeVideoDataset(probe['positive'][:n_probe])
        neg_ds   = ProbeVideoDataset(probe['negative'][:n_probe])
        test_ds  = ProbeVideoDataset(test_records[:20])

        entry = {}
        for pathway_name, layer_path in pathway_layers.items():
            try:
                pos_acts = extract_activations(model, layer_path, pos_ds, device)
                neg_acts = extract_activations(model, layer_path, neg_ds, device)
                _, acc   = train_cav(pos_acts, neg_acts)
                entry[pathway_name] = round(acc, 4)
            except Exception as e:
                print(f"  [warn] {concept}/{pathway_name}: {e}")
                entry[pathway_name] = None

        # Dominant pathway = higher linear decodability
        if entry.get('frame') and entry.get('flow'):
            entry['dominant'] = 'frame' if entry['frame'] > entry['flow'] else 'flow'
        results[concept] = entry
        print(f"  {concept}: frame_acc={entry.get('frame')}  "
              f"flow_acc={entry.get('flow')}  dominant={entry.get('dominant', '?')}")

    return results


# ── display name lookup ────────────────────────────────────────────────────────

_DISPLAY_NAMES = {
    'temporal_motion_inconsistency': 'Unnatural movement pattern',
    'facial_geometry_drift':         'Face shape inconsistency over time',
    'frequency_domain_artifact':     'Digital texture irregularity',
    'texture_boundary_artifact':     'Region boundary artifact',
    'lighting_shadow_decoupling':    'Inconsistent lighting',
}


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='TCAV concept probing for Ninox FlowVideoClassifier')
    p.add_argument('--checkpoint', default='flow_stage2_checkpoints/checkpoint_epoch_0004.pt')
    p.add_argument('--probes',     default='_meta/probes/',
                   help='Directory containing probe_*.json files from tcav_probes.py')
    p.add_argument('--test-manifest', default='data/flow_manifest.csv')
    p.add_argument('--output',     default='_meta/tcav/')
    p.add_argument('--n-splits',   type=int, default=5)
    p.add_argument('--n-test',     type=int, default=50,
                   help='AI-generated test videos for sign count evaluation')
    p.add_argument('--skip-dual-pathway', action='store_true')
    return p.parse_args()


def main():
    args       = parse_args()
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe_dir  = Path(args.probes)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    # Load and sample test videos (AI-generated only for sign count)
    test_records = []
    with open(args.test_manifest, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['label']) == TARGET_CLASS:
                test_records.append({'path': row['path'], 'label': int(row['label'])})
    rng = np.random.default_rng(42)
    rng.shuffle(test_records)
    test_records = test_records[: args.n_test]
    print(f"[test] {len(test_records)} AI-generated test videos")

    # Discover probe sets
    probe_files = sorted(probe_dir.glob('probe_*.json'))
    print(f"[probes] {len(probe_files)} probe sets found in {probe_dir}")

    all_results       = []
    validated_results = []
    layer_selection   = {}

    for probe_file in probe_files:
        concept = probe_file.stem[len('probe_'):]
        with open(probe_file) as f:
            probe = json.load(f)
        pos_records = probe['positive']
        neg_records = probe['negative']

        if len(pos_records) < 10 or len(neg_records) < 10:
            print(f"[skip] {concept}: too few probes "
                  f"({len(pos_records)}+, {len(neg_records)}-)")
            continue

        print(f"\n{'=' * 55}")
        print(f"Concept: {concept}")
        print(f"  Probes: {len(pos_records)}+ / {len(neg_records)}-")

        # Step 1 — layer selection (quick single split)
        best_layer = select_best_layer(
            concept, pos_records, neg_records, model,
            PROBE_LAYERS, test_records, device,
        )
        layer_selection[concept] = best_layer

        # Step 2 — full n-split validation on best layer
        print(f"\n[validate] {concept} @ layer {best_layer.split('.')[-1]} "
              f"({args.n_splits} splits)")
        result = validate_concept(
            concept, pos_records, neg_records, model,
            best_layer, test_records, device, args.n_splits,
        )
        all_results.append(result)

        status = "VALIDATED ✓" if result['validated'] else "FAILED ✗"
        sc     = result.get('mean_sign_count', 0)
        std    = result.get('std_sign_count', 0)
        acc    = result.get('mean_cav_accuracy', 0)
        print(f"  {status}  sc={sc:.3f} ± {std:.3f}  cav_acc={acc:.3f}")

        if result['validated']:
            validated_results.append(result)

    # Save raw results
    with open(output_dir / 'tcav_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / 'layer_selection.json', 'w') as f:
        json.dump(layer_selection, f, indent=2)
    print(f"\n[saved] tcav_results.json, layer_selection.json → {output_dir}")

    # Dual-pathway analysis
    if validated_results and not args.skip_dual_pathway:
        print(f"\n[dual-pathway] {len(validated_results)} validated concepts")
        dual = run_dual_pathway(
            validated_results, model, probe_dir, test_records, device
        )
        with open(output_dir / 'dual_pathway.json', 'w') as f:
            json.dump(dual, f, indent=2)
        print(f"[saved] dual_pathway.json → {output_dir}")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"TCAV PROBING COMPLETE")
    print(f"  Concepts tested:   {len(all_results)}")
    print(f"  Validated:         {len(validated_results)}")
    for r in validated_results:
        layer_short = r['layer'].rsplit('.', 1)[-1]
        print(f"    ✓ {r['concept']}  "
              f"layer={layer_short}  "
              f"sc={r['mean_sign_count']:.3f} ± {r['std_sign_count']:.3f}  "
              f"cav_acc={r['mean_cav_accuracy']:.3f}")

    # Write validated concepts YAML (human-readable for Phase 5 narrative)
    if _YAML and validated_results:
        yaml_out = []
        for r in validated_results:
            yaml_out.append({
                'concept':           r['concept'],
                'display_name':      _DISPLAY_NAMES.get(r['concept'], r['concept']),
                'layer':             r['layer'],
                'mean_sign_count':   r['mean_sign_count'],
                'std_sign_count':    r['std_sign_count'],
                'mean_cav_accuracy': r['mean_cav_accuracy'],
            })
        with open(output_dir / 'validated_concepts.yaml', 'w') as f:
            yaml.dump({'validated_concepts': yaml_out}, f, sort_keys=False)
        print(f"\n[saved] validated_concepts.yaml → {output_dir}")
        print("Next: Phase 4 temporal timeline (tcav_timeline.py)")


if __name__ == '__main__':
    main()
