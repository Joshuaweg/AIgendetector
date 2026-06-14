"""
Production TCAV overlay generator.

Runs inside generate_attributions() immediately after IG completes, using the
same model instance and the already-computed attribution tensor.  Produces a
self-contained HTML file with:
  - 12 navigable annotated frames (raw frame + IG glow + concept contours + callouts)
  - Timeline strip coloured by per-segment concept strength
  - Spearman r between IG magnitude and CAS, displayed as spatial alignment score

Concept Activation Score (CAS) replaces sign count for single-video inference:
  cas[p] = dot(activation[p], cav[p])
This requires no backward pass and no test-set population — just a forward pass
to extract layer 7 activations and a vectorised dot product.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from full_scale_classifier import FlowVideoClassifier

# ── concept display config ─────────────────────────────────────────────────────

CONCEPT_COLORS = {
    'temporal_motion_inconsistency': '#FF8C00',
    'texture_boundary_artifact':     '#00CED1',
    'frequency_domain_artifact':     '#8B5CF6',
    'lighting_shadow_decoupling':    '#10B981',
    'facial_geometry_drift':         '#F43F5E',
}

CONCEPT_DISPLAY = {
    'temporal_motion_inconsistency': 'Unnatural movement pattern',
    'texture_boundary_artifact':     'Region boundary artifact',
    'frequency_domain_artifact':     'Digital texture irregularity',
    'lighting_shadow_decoupling':    'Inconsistent lighting',
    'facial_geometry_drift':         'Face shape inconsistency',
}

CONCEPT_DESCRIPTIONS = {
    'temporal_motion_inconsistency': (
        'Objects or background regions move in ways that violate physical motion '
        'continuity — a signature of frame-by-frame AI synthesis.'
    ),
    'texture_boundary_artifact':     (
        'Seams where separately synthesised image regions meet, visible as '
        'unnatural sharpness or blending discontinuities.'
    ),
    'frequency_domain_artifact':     (
        'Repeating digital texture patterns introduced by the generative model\'s '
        'upsampling layers.'
    ),
    'lighting_shadow_decoupling':    (
        'Lighting and shadow inconsistent with the direction or intensity of '
        'the apparent light source.'
    ),
    'facial_geometry_drift':         (
        'Facial proportions or geometry that shift between frames in ways '
        'impossible for a real face.'
    ),
}

# CAS threshold: positions above this (after 0-1 normalisation) are flagged
CAS_THRESHOLD    = 0.72         # raised from 0.60 — keeps only the strongest regions
CONTOUR_MIN_AREA = 1500         # px² — filters noise; keeps only substantive blobs
MAX_CONTOURS     = 2            # draw at most 2 contour regions per concept per frame
LAYER_IDX        = 7            # mid-network layer with preserved spatial structure
N_FRAME_TOKENS   = 768          # 12 segments × 8×8 spatial grid
N_SEGMENTS       = 12
SPATIAL_H        = 8
SPATIAL_W        = 8
FRAME_SIZE       = 512
IG_GLOW_ALPHA    = 0.55         # opacity for IG warm glow layer
CONTOUR_WIDTH    = 3            # px outline thickness
GAUSSIAN_SIGMA   = 12.0         # matches interpret.py


# ── CAV loading ────────────────────────────────────────────────────────────────

def load_cavs(tcav_dir: Path) -> dict[str, np.ndarray]:
    """
    Load pre-trained per-token CAVs produced by tcav_interpret.py --per-token.
    Returns {concept_name: cavs [768, 768]} for every concept whose
    cavs_layer{LAYER_IDX}.npy exists under tcav_dir/{concept}/.
    """
    cavs: dict[str, np.ndarray] = {}
    if not tcav_dir.exists():
        return cavs
    for concept_dir in sorted(tcav_dir.iterdir()):
        if not concept_dir.is_dir():
            continue
        cav_file = concept_dir / f'cavs_layer{LAYER_IDX}.npy'
        if cav_file.exists():
            cavs[concept_dir.name] = np.load(cav_file)
    return cavs


# ── CAS computation ────────────────────────────────────────────────────────────

def compute_cas_maps(
    model: FlowVideoClassifier,
    frames_tensor: torch.Tensor,
    flow_maps: torch.Tensor,
    cavs: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Single forward pass to layer LAYER_IDX; compute CAS = dot(act[p], cav[p])
    per frame-token position per concept.
    Returns {concept: [N_SEGMENTS, SPATIAL_H, SPATIAL_W]}.
    """
    layers = model.classifier.transformer_encoder.layers

    model.eval()
    with torch.no_grad():
        latents  = model.latent_encoder(frames_tensor)
        tokens   = model.patch_encoder(latents)
        del latents
        flow_tok = model.flow_encoder(flow_maps)
        tokens   = torch.cat([tokens, flow_tok], dim=1)   # [1, 791, 768]
        del flow_tok
        for i in range(LAYER_IDX + 1):
            tokens = layers[i](tokens)

    # [768, 768] frame-token activations (exclude the 23 flow tokens)
    act = tokens[0, :N_FRAME_TOKENS, :].cpu().numpy()   # [768, 768]

    result: dict[str, np.ndarray] = {}
    for concept, cav_matrix in cavs.items():
        # cav_matrix: [768 positions, 768 dims]
        cas = (act * cav_matrix).sum(axis=-1)            # [768]
        # Normalise to [0, 1] across the full video sequence
        cas_min, cas_max = cas.min(), cas.max()
        if cas_max > cas_min:
            cas = (cas - cas_min) / (cas_max - cas_min)
        else:
            cas = np.zeros_like(cas)
        result[concept] = cas.reshape(N_SEGMENTS, SPATIAL_H, SPATIAL_W)

    return result


def _upsample_cas(cas_grid: np.ndarray) -> np.ndarray:
    """
    Upsample [N_SEGMENTS, 8, 8] → [N_SEGMENTS, 512, 512].
    Bilinear then gaussian_filter(sigma=12) to match interpret.py smoothing.
    """
    out = np.zeros((N_SEGMENTS, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    for i in range(N_SEGMENTS):
        t   = torch.tensor(cas_grid[i]).unsqueeze(0).unsqueeze(0)
        up  = F.interpolate(t, size=(FRAME_SIZE, FRAME_SIZE),
                            mode='bilinear', align_corners=False)
        seg = gaussian_filter(up.squeeze().numpy(), sigma=GAUSSIAN_SIGMA)
        out[i] = seg
    return out


# ── Spearman correlation ───────────────────────────────────────────────────────

def compute_spearman(
    ig_attr: torch.Tensor,
    cas_upsampled: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Spearman r between IG attribution magnitude (mean across channels, frames)
    and mean CAS across segments.  Both are [512, 512] spatial maps.
    """
    # IG: [1, 24, 512, 512, 3] → absolute value, mean over frames and channels
    ig_mag = ig_attr.squeeze(0).abs().mean(dim=(0, 3)).cpu().numpy()  # [512, 512]

    scores: dict[str, float] = {}
    for concept, cas_up in cas_upsampled.items():
        cas_mean = cas_up.mean(axis=0)   # [512, 512]
        r, _ = spearmanr(ig_mag.flatten(), cas_mean.flatten())
        scores[concept] = round(float(r), 3)
    return scores


# ── Frame compositing ──────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _top_contours(mask: np.ndarray) -> list:
    """Return up to MAX_CONTOURS largest contours above CONTOUR_MIN_AREA."""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= CONTOUR_MIN_AREA]
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid[:MAX_CONTOURS]


def composite_frame(
    raw_frame: np.ndarray,
    ig_attr_frame: np.ndarray,
    cas_upsampled: dict[str, np.ndarray],
    seg_idx: int,
) -> tuple[Image.Image, list[dict]]:
    """
    Composite one 512×512 annotated frame.

    raw_frame:     [512, 512, 3] uint8 RGB
    ig_attr_frame: [512, 512, 3] float32 IG attributions for this frame
    cas_upsampled: {concept: [N_SEGMENTS, 512, 512]}
    seg_idx:       which temporal segment (0..11)

    Returns (PIL Image, callout list).
    """
    base = Image.fromarray(raw_frame.astype(np.uint8), 'RGB')

    # ── IG glow layer ───────────────────────────────────────────────────────
    ig_mag = np.abs(ig_attr_frame).mean(axis=-1)         # [512, 512]
    # Clip the bottom 70th percentile to zero — only the top 30% of
    # attribution pixels glow, suppressing background noise that would
    # otherwise wash out the overlay into an indistinct haze.
    p70     = np.percentile(ig_mag, 70)
    ig_mag  = np.clip(ig_mag - p70, 0, None)
    ig_norm = ig_mag / (ig_mag.max() + 1e-8)             # [0, 1]
    # Gamma boost: sqrt stretches mid-range values so moderate attributions
    # read visually rather than only the single hottest pixel.
    ig_norm = np.sqrt(ig_norm)
    cmap    = plt_inferno(ig_norm)                        # [512, 512, 3] float
    glow    = Image.fromarray((cmap * 255).astype(np.uint8), 'RGB')
    glow_a  = Image.fromarray((ig_norm * 255 * IG_GLOW_ALPHA).astype(np.uint8), 'L')
    base.paste(glow, mask=glow_a)

    # ── Concept contour overlays ─────────────────────────────────────────────
    # Corner anchors for labels: concept 0 → top-left, concept 1 → top-right
    # Keeps labels out of the frame centre regardless of where contours land.
    LABEL_CORNERS = [
        (8,  8),                          # top-left
        (FRAME_SIZE - 160, 8),            # top-right
        (8,  FRAME_SIZE - 28),            # bottom-left
        (FRAME_SIZE - 160, FRAME_SIZE - 28),  # bottom-right
    ]

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = ImageFont.load_default()

    active_labels: list[dict] = []

    for concept_idx, (concept, cas_up) in enumerate(cas_upsampled.items()):
        color_hex = CONCEPT_COLORS.get(concept, '#FFFFFF')
        color_rgb = _hex_to_rgb(color_hex)
        label     = CONCEPT_DISPLAY.get(concept, concept)

        seg_map = cas_up[seg_idx]                        # [512, 512]
        mask    = (seg_map > CAS_THRESHOLD).astype(np.uint8)
        contours = _top_contours(mask)
        if not contours:
            continue

        # Draw top-N contours only
        overlay = Image.new('RGBA', (FRAME_SIZE, FRAME_SIZE), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        for cnt in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            if len(pts) < 3:
                continue
            ov_draw.polygon(pts, fill=(*color_rgb, 35))
            ov_draw.line(pts + [pts[0]], fill=(*color_rgb, 230), width=CONTOUR_WIDTH)

        base = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')

        # Record label at its assigned corner
        lx, ly = LABEL_CORNERS[concept_idx % len(LABEL_CORNERS)]
        active_labels.append({'lx': lx, 'ly': ly, 'label': label,
                               'color': color_hex, 'color_rgb': color_rgb})

    # ── Fixed-corner callout labels ──────────────────────────────────────────
    draw = ImageDraw.Draw(base)
    for item in active_labels:
        lx, ly = item['lx'], item['ly']
        cr      = item['color_rgb']
        # Pill background
        tw = len(item['label']) * 7 + 10
        draw.rectangle([lx - 2, ly - 2, lx + tw, ly + 18],
                        fill=(0, 0, 0, 180))
        draw.rectangle([lx - 2, ly - 2, lx + 4, ly + 18],
                        fill=(*cr, 230))        # colour accent bar on left
        draw.text((lx + 6, ly), item['label'], fill=item['color'], font=font)

    return base, active_labels


def plt_inferno(arr: np.ndarray) -> np.ndarray:
    """Convert a [H, W] float [0,1] array to [H, W, 3] using the inferno colormap."""
    import matplotlib
    cmap = matplotlib.colormaps['inferno']
    return cmap(arr)[:, :, :3].astype(np.float32)


# ── HTML generation ────────────────────────────────────────────────────────────

def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html(
    frames_b64:   list[str],
    callouts:     list[list[dict]],
    cas_upsampled: dict[str, np.ndarray],
    concepts:     list[str],
    spearman:     dict[str, float],
    video_id:     str,
    prediction:   str,
    confidence:   float,
) -> str:
    """Build the self-contained HTML explanation page."""

    verdict_color = '#f85149' if prediction == 'AI-Generated' else '#3fb950'
    confidence_pct = f'{confidence * 100:.1f}%'

    # Timeline data: mean CAS per segment per concept, normalised to 0-1
    seg_strengths: list[list[float]] = []
    for concept in concepts:
        cas_up  = cas_upsampled[concept]
        strengths = [float(cas_up[s].mean()) for s in range(N_SEGMENTS)]
        # Normalise so timeline colours are relative within each concept
        mx = max(strengths) or 1.0
        seg_strengths.append([round(v / mx, 3) for v in strengths])

    # Legend items
    legend_items = []
    for c in concepts:
        display = CONCEPT_DISPLAY.get(c, c)
        color   = CONCEPT_COLORS.get(c, '#888')
        r_val   = spearman.get(c, 0)
        legend_items.append(
            f'<div class="legend-item">'
            f'<div class="swatch" style="background:{color}"></div>'
            f'<span>{display}</span>'
            f'<span class="r-val">IG align r={r_val:+.2f}</span>'
            f'</div>'
        )

    spearman_summary = ' · '.join(
        f'{CONCEPT_DISPLAY.get(c, c)}: r={v:+.2f}' for c, v in spearman.items()
    )

    concept_colors_js = json.dumps({c: CONCEPT_COLORS.get(c, '#888') for c in concepts})
    seg_strengths_js  = json.dumps(seg_strengths)
    frames_js         = json.dumps(frames_b64)
    concepts_js       = json.dumps([CONCEPT_DISPLAY.get(c, c) for c in concepts])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Detection Analysis — {video_id}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;
        padding:24px;max-width:900px;margin:0 auto}}
  .header{{display:flex;justify-content:space-between;align-items:flex-start;
           margin-bottom:20px;gap:12px;flex-wrap:wrap}}
  .verdict{{font-size:22px;font-weight:700;color:{verdict_color}}}
  .confidence{{font-size:13px;color:#8b949e;margin-top:4px}}
  .align-note{{font-size:12px;color:#8b949e;text-align:right;max-width:320px}}

  .viewer{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
  .frame-wrap{{position:relative;flex:1;border-radius:8px;overflow:hidden;
               box-shadow:0 4px 24px rgba(0,0,0,.5)}}
  .frame-wrap img{{width:100%;display:block}}
  .frame-badge{{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.75);
                color:#e6edf3;font-size:11px;padding:3px 8px;border-radius:4px}}
  .nav{{background:#21262d;border:1px solid #30363d;color:#e6edf3;
        width:40px;height:40px;border-radius:8px;cursor:pointer;font-size:20px;
        display:flex;align-items:center;justify-content:center;flex-shrink:0}}
  .nav:hover{{background:#30363d}}

  .counter{{text-align:center;color:#8b949e;font-size:12px;margin-bottom:14px}}

  .timeline{{display:flex;gap:3px;margin-bottom:18px;align-items:flex-end}}
  .seg{{flex:1;cursor:pointer;border-radius:3px;border:2px solid transparent;
        transition:border-color .15s}}
  .seg.active{{border-color:#58a6ff}}
  .seg-inner{{border-radius:2px;transition:height .2s}}
  .seg-num{{text-align:center;font-size:9px;color:#8b949e;margin-top:2px}}
  .tl-label{{font-size:11px;color:#8b949e;margin-bottom:6px}}

  .legend{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px}}
  .legend-item{{display:flex;align-items:center;gap:7px;font-size:13px}}
  .swatch{{width:14px;height:14px;border-radius:3px;flex-shrink:0}}
  .r-val{{color:#8b949e;font-size:11px;margin-left:4px}}
  .ig-legend{{display:flex;align-items:center;gap:7px;font-size:13px;
              margin-left:auto;color:#8b949e}}
  .ig-swatch{{width:32px;height:14px;border-radius:3px;
              background:linear-gradient(to right,#000004,#f8765c,#feca8d)}}

  .description{{margin-top:16px;padding:14px;background:#161b22;
               border:1px solid #21262d;border-radius:8px;font-size:12px;
               color:#8b949e;line-height:1.6}}
  .desc-concept{{color:#e6edf3;font-weight:600;margin-bottom:4px}}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="verdict">{prediction}</div>
    <div class="confidence">Confidence: {confidence_pct}</div>
  </div>
  <div class="align-note">
    IG ↔ concept spatial alignment<br>
    {spearman_summary}
  </div>
</div>

<div class="viewer">
  <button class="nav" id="prev" onclick="navigate(-1)">&#8249;</button>
  <div class="frame-wrap">
    <img id="frame-img" src="" alt="Annotated frame">
    <div class="frame-badge" id="frame-badge">Segment 00 · Frames 0–1</div>
  </div>
  <button class="nav" id="next" onclick="navigate(1)">&#8250;</button>
</div>

<div class="counter" id="counter">1 / {N_SEGMENTS}</div>

<div class="tl-label">Concept strength across video</div>
<div class="timeline" id="timeline"></div>

<div class="legend">
  {''.join(legend_items)}
  <div class="ig-legend">
    <div class="ig-swatch"></div>
    Model attention (IG)
  </div>
</div>

<div class="description" id="desc-panel"></div>

<script>
const FRAMES    = {frames_js};
const STRENGTHS = {seg_strengths_js};   // [n_concepts][n_segments]
const COLORS    = {concept_colors_js};
const CONCEPTS  = {concepts_js};
const N_SEG     = {N_SEGMENTS};
let   current   = 0;

function setFrame(idx) {{
  current = ((idx % N_SEG) + N_SEG) % N_SEG;
  document.getElementById('frame-img').src = 'data:image/png;base64,' + FRAMES[current];
  document.getElementById('frame-badge').textContent =
    'Segment ' + String(current).padStart(2,'0') +
    '  ·  Frames ' + (current*2) + '–' + (current*2+1);
  document.getElementById('counter').textContent = (current+1) + ' / ' + N_SEG;
  document.querySelectorAll('.seg').forEach((el,i) => {{
    el.classList.toggle('active', i === current);
  }});
  updateDesc();
}}

function navigate(dir) {{ setFrame(current + dir); }}

function buildTimeline() {{
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  for (let s = 0; s < N_SEG; s++) {{
    // Blend concept colours by their relative strength at this segment
    const div = document.createElement('div');
    div.className = 'seg' + (s===0?' active':'');
    div.onclick = () => setFrame(s);
    // Max strength across concepts for bar height
    const maxStr = Math.max(...STRENGTHS.map(c => c[s]));
    const h = Math.max(8, Math.round(maxStr * 48));
    // Dominant concept colour
    let domIdx = 0, domVal = 0;
    STRENGTHS.forEach((c,i) => {{ if(c[s] > domVal){{domVal=c[s];domIdx=i;}} }});
    const colorKey = Object.keys(COLORS)[domIdx];
    const color = COLORS[colorKey] || '#58a6ff';
    div.innerHTML = `<div class="seg-inner" style="height:${{h}}px;background:${{color}};opacity:${{0.4+maxStr*0.6}}"></div>`
                  + `<div class="seg-num">${{s}}</div>`;
    tl.appendChild(div);
  }}
}}

function updateDesc() {{
  const panel = document.getElementById('desc-panel');
  // Show concept with highest strength at current segment
  let maxStr = 0, maxIdx = 0;
  STRENGTHS.forEach((c,i) => {{ if(c[current] > maxStr){{maxStr=c[current];maxIdx=i;}} }});
  if(maxStr < 0.1){{panel.innerHTML='<span style="color:#8b949e">No strong concept signal in this segment.</span>';return;}}
  const conceptKey = Object.keys(COLORS)[maxIdx];
  const color = COLORS[conceptKey] || '#888';
  panel.innerHTML =
    `<div class="desc-concept" style="color:${{color}}">${{CONCEPTS[maxIdx]}}</div>` +
    `<div>${{maxStr > 0.5 ? 'Strong' : 'Moderate'}} signal in this segment.</div>`;
}}

buildTimeline();
setFrame(0);
</script>
</body>
</html>"""


# ── Main entry point ───────────────────────────────────────────────────────────

def run_overlay(
    model:         FlowVideoClassifier,
    frames_tensor: torch.Tensor,
    flow_maps:     torch.Tensor,
    ig_attributions: torch.Tensor,
    raw_frames:    list,
    cavs:          dict[str, np.ndarray],
    output_path:   str,
    video_id:      str,
    prediction:    str,
    confidence:    float,
    device:        torch.device,
) -> str | None:
    """
    Generate the TCAV overlay HTML for a single video.

    ig_attributions: [1, 24, 512, 512, 3] tensor from captum IG
    raw_frames:      list of 24 np.ndarray [512, 512, 3] uint8 RGB
    cavs:            loaded by load_cavs() at server startup

    Returns path to HTML file, or None if no concepts loaded.
    """
    if not cavs:
        return None

    concepts = list(cavs.keys())

    # 1. CAS maps per concept per segment
    cas_maps      = compute_cas_maps(model, frames_tensor, flow_maps, cavs, device)
    cas_upsampled = {c: _upsample_cas(v) for c, v in cas_maps.items()}

    # 2. Spearman correlation between IG and CAS
    spearman = compute_spearman(ig_attributions, cas_upsampled)

    # 3. Composite one annotated frame per segment (use first frame of each pair)
    ig_np    = ig_attributions.squeeze(0).cpu().numpy()  # [24, 512, 512, 3]
    frames_b64: list[str] = []
    all_callouts: list[list[dict]] = []

    for seg_i in range(N_SEGMENTS):
        frame_i  = seg_i * 2                        # first frame of the 2-frame segment
        raw_f    = np.array(raw_frames[frame_i])    # [512, 512, 3] uint8
        ig_frame = ig_np[frame_i]                   # [512, 512, 3] float

        composited, callouts = composite_frame(raw_f, ig_frame, cas_upsampled, seg_i)
        frames_b64.append(_img_to_b64(composited))
        all_callouts.append(callouts)

    # 4. Generate HTML
    html = generate_html(
        frames_b64, all_callouts, cas_upsampled,
        concepts, spearman, video_id, prediction, confidence,
    )

    Path(output_path).write_text(html, encoding='utf-8')
    return output_path
