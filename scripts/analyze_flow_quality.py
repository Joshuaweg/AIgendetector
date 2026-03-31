"""
Correct optical flow analysis for AI video detection.

Measures flow STRUCTURE, not magnitude:
  - spatial_var:    variance of flow magnitude across pixels (low = unnaturally smooth)
  - hf_ratio:       high-freq power / total in FFT of flow field (low = AI over-smoothing)
  - temporal_corr:  correlation between consecutive flow fields (high = AI-like smoothness)
  - noise_floor:    std of flow in near-static regions (low = AI too clean)
  - laplacian_var:  variance of flow Laplacian (low = spatially over-smoothed)

Real videos: high spatial_var, high hf_ratio, low temporal_corr, high noise_floor
AI videos:   low spatial_var, low hf_ratio, high temporal_corr, low noise_floor
"""

import cv2
import sys
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_PER_GEN = 50
MAX_FRAMES     = 24
FLOW_SIZE      = 128   # higher res to preserve spatial noise structure
RANDOM_SEED    = 42

SOURCES = {
    "GenVideo-Train": {
        "type":    "subdirs",
        "path":    Path("F:/Gen-Video/GenVideo-Train/AI-Generated"),
    },
    "MVAD": {
        "type":    "flat_prefix",
        "path":    Path("F:/MVAD/videos"),
    },
    "Veo3": {
        "type":    "single",
        "path":    Path("C:/Users/joshu/Downloads/Veo3"),
        "name":    "Veo3",
    },
    "Sora-Val": {
        "type":    "single",
        "path":    Path("F:/Gen-Video/GenVideo-Val/GenVideo-Val/Fake/Sora"),
        "name":    "Sora",
    },
}

# Also include Real videos for baseline comparison
REAL_DIR = Path("F:/Gen-Video/dataset")

# ---------------------------------------------------------------------------
# Flow analysis
# ---------------------------------------------------------------------------
def compute_flow_metrics(path: str) -> dict | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    frames = []
    while len(frames) < MAX_FRAMES * 2:
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (FLOW_SIZE, FLOW_SIZE)))
    cap.release()

    if len(frames) < 4:
        return None

    if len(frames) > MAX_FRAMES:
        idx = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
        frames = [frames[i] for i in idx]

    flow_fields = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_fields.append(flow)

    if not flow_fields:
        return None

    spatial_vars, hf_ratios, noise_floors, laplacian_vars = [], [], [], []
    mag_means = []
    temp_corrs = []

    prev_mag = None
    for flow in flow_fields:
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u**2 + v**2)
        mag_means.append(float(mag.mean()))

        # 1. Spatial variance — how irregular is the flow field spatially?
        spatial_vars.append(float(mag.var()))

        # 2. High-frequency ratio via FFT
        fft = np.abs(np.fft.fftshift(np.fft.fft2(mag)))
        h, w = fft.shape
        cy, cx = h // 2, w // 2
        r_low = min(h, w) // 6   # low-freq radius (inner ~17%)
        y_g, x_g = np.ogrid[-cy:h - cy, -cx:w - cx]
        low_mask = (x_g**2 + y_g**2) <= r_low**2
        total = fft.sum() + 1e-10
        hf_ratios.append(float(fft[~low_mask].sum() / total))

        # 3. Noise floor: std of flow in the bottom-quartile (near-static) pixels
        q25 = np.percentile(mag, 25)
        static = mag[mag <= q25]
        if static.size > 0:
            noise_floors.append(float(static.std()))

        # 4. Laplacian variance — spatial second-derivative of flow (smoothness penalty)
        lap = cv2.Laplacian(mag.astype(np.float32), cv2.CV_32F)
        laplacian_vars.append(float(lap.var()))

        # 5. Temporal correlation between consecutive flow fields
        if prev_mag is not None:
            flat1, flat2 = prev_mag.flatten(), mag.flatten()
            if flat1.std() > 1e-6 and flat2.std() > 1e-6:
                temp_corrs.append(float(np.corrcoef(flat1, flat2)[0, 1]))
        prev_mag = mag

    return dict(
        mean_mag      = float(np.mean(mag_means)),
        spatial_var   = float(np.mean(spatial_vars)),
        hf_ratio      = float(np.mean(hf_ratios)),
        noise_floor   = float(np.mean(noise_floors)) if noise_floors else 0.0,
        laplacian_var = float(np.mean(laplacian_vars)),
        temporal_corr = float(np.mean(temp_corrs)) if temp_corrs else 0.0,
    )


# ---------------------------------------------------------------------------
# Collect paths
# ---------------------------------------------------------------------------
def collect_generators(source_cfg: dict) -> dict:
    """Returns {gen_name: [path_str, ...]}"""
    t    = source_cfg["type"]
    base = source_cfg["path"]
    gens = {}

    if not base.exists():
        print(f"  WARNING: path not found: {base}")
        return gens

    if t == "subdirs":
        for sub in sorted(base.iterdir()):
            if sub.is_dir():
                mp4s = [str(p) for p in sub.rglob("*.mp4")]
                if mp4s:
                    gens[sub.name] = mp4s

    elif t == "flat_prefix":
        buckets = defaultdict(list)
        for f in sorted(base.glob("ai_*.mp4")):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                buckets[parts[1]].append(str(f))
        gens.update(buckets)

    elif t == "single":
        name = source_cfg.get("name", base.name)
        mp4s = [str(p) for p in base.rglob("*.mp4")]
        # Deduplicate by filename
        seen, unique = set(), []
        for p in mp4s:
            fname = Path(p).name
            if fname not in seen:
                seen.add(fname)
                unique.append(p)
        if unique:
            gens[name] = unique

    return gens


# ---------------------------------------------------------------------------
# Analyse one generator
# ---------------------------------------------------------------------------
def analyse_generator(gen_name: str, paths: list, rng: random.Random) -> dict | None:
    sample = rng.sample(paths, min(SAMPLE_PER_GEN, len(paths)))
    results, fails = [], 0

    for p in sample:
        r = compute_flow_metrics(p)
        if r:
            results.append(r)
        else:
            fails += 1

    if not results:
        return None

    keys = ["mean_mag", "spatial_var", "hf_ratio", "noise_floor",
            "laplacian_var", "temporal_corr"]
    agg = {k: float(np.mean([r[k] for r in results])) for k in keys}
    agg["n"]         = len(results)
    agg["fail_rate"] = fails / len(sample) * 100
    agg["total"]     = len(paths)
    return agg


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
HEADER_PRINTED = False

def print_header():
    global HEADER_PRINTED
    if HEADER_PRINTED:
        return
    HEADER_PRINTED = True
    print(f"\n{'Generator':<30} {'n':>4}  {'MagMn':>6} {'SpatVar':>7} {'HFrat':>6} "
          f"{'NoisFl':>6} {'LapVar':>7} {'TmpCor':>6}  {'Fail%':>5}  Assessment")
    print("-" * 115)


def assess(r: dict) -> str:
    signals = 0
    notes = []

    # Low spatial var = AI-smooth
    if r["spatial_var"] < 0.05:
        signals += 1; notes.append("spatially smooth")
    # Low HF ratio = over-smoothed freq domain
    if r["hf_ratio"] < 0.70:
        signals += 1; notes.append("low HF power")
    # Low noise floor = too clean
    if r["noise_floor"] < 0.05:
        signals += 1; notes.append("clean noise floor")
    # Low laplacian = no spatial edges in flow
    if r["laplacian_var"] < 0.02:
        signals += 1; notes.append("smooth edges")
    # High temporal corr = predictable motion
    if r["temporal_corr"] > 0.90:
        signals += 1; notes.append("predictable motion")

    if signals >= 3:
        verdict = f"STRONG AI signal ({signals}/5): " + ", ".join(notes)
    elif signals >= 2:
        verdict = f"MODERATE AI signal ({signals}/5): " + ", ".join(notes)
    elif signals >= 1:
        verdict = f"WEAK AI signal ({signals}/5): " + ", ".join(notes)
    else:
        verdict = "No consistent AI signal"
    return verdict


def print_row(source: str, gen: str, r: dict):
    label = f"{source}/{gen}"[:29]
    print(f"{label:<30} {r['n']:>4}  "
          f"{r['mean_mag']:>6.3f} {r['spatial_var']:>7.4f} {r['hf_ratio']:>6.3f} "
          f"{r['noise_floor']:>6.4f} {r['laplacian_var']:>7.4f} {r['temporal_corr']:>6.3f}  "
          f"{r['fail_rate']:>4.0f}%  {assess(r)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(RANDOM_SEED)

    # Collect all generators across all sources
    all_results = {}   # (source, gen_name) -> agg dict

    print("Collecting paths and computing flow metrics...")
    print(f"Flow resolution: {FLOW_SIZE}x{FLOW_SIZE}  |  Sample: {SAMPLE_PER_GEN}/generator\n")

    for source_name, cfg in SOURCES.items():
        gens = collect_generators(cfg)
        if not gens:
            print(f"[{source_name}] No generators found")
            continue
        print(f"[{source_name}] {len(gens)} generators, analysing...")
        for gen_name, paths in sorted(gens.items()):
            sys.stdout.write(f"  {gen_name}... ")
            sys.stdout.flush()
            r = analyse_generator(gen_name, paths, rng)
            if r:
                all_results[(source_name, gen_name)] = r
                print(f"done ({r['n']} videos)")
            else:
                print("SKIP (no readable videos)")

    # Real baseline
    if REAL_DIR.exists():
        print(f"\n[Real-Kinetics] sampling for baseline...")
        real_paths = [str(p) for p in REAL_DIR.glob("real_*.mp4")]
        if real_paths:
            r = analyse_generator("Real", real_paths, rng)
            if r:
                all_results[("Real", "Kinetics")] = r
                print(f"  done ({r['n']} videos)")

    # Print table
    print("\n\n" + "=" * 115)
    print("OPTICAL FLOW STRUCTURE ANALYSIS")
    print("Metrics: SpatVar=spatial variance  HFrat=high-freq ratio  NoisFl=noise floor  "
          "LapVar=laplacian var  TmpCor=temporal corr")
    print("Real videos → HIGH SpatVar, HIGH HFrat, HIGH NoisFl, HIGH LapVar, LOW TmpCor")
    print("AI videos   → LOW SpatVar,  LOW HFrat,  LOW NoisFl,  LOW LapVar,  HIGH TmpCor")
    print("=" * 115)

    print_header()

    # Real first
    if ("Real", "Kinetics") in all_results:
        print_row("Real", "Kinetics", all_results[("Real", "Kinetics")])
        print("-" * 115)

    # AI sources
    for (source, gen), r in sorted(all_results.items()):
        if source == "Real":
            continue
        print_row(source, gen, r)

    print("\n")
    print("Interpretation guide:")
    print("  STRONG AI signal (3+/5) = flow features likely discriminative for this generator")
    print("  MODERATE (2/5)          = partial signal, may generalise with enough data")
    print("  WEAK/None               = flow features not useful for this generator")


if __name__ == "__main__":
    main()
