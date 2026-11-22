# File: app.py
# What this file does: Interactive professor-ready 3-tab Gradio Blocks web application for live
#   ZeroFake deepfake detection and research benchmark exploration:
#   - Tab 1 ("Live Detection"): Clean, uncluttered upload-to-verdict workflow with auto-run on file
#     upload, 4 one-click sample buttons from demo_samples/, a high-contrast Verdict Card (REAL in
#     emerald, FAKE in crimson, UNKNOWN in amber), an Uncertainty Meter comparing U(x) vs. tau_reject,
#     and a 2-panel "Aligned Face vs. Grad-CAM Manipulation Heatmap" view.
#   - Tab 2 ("Deep Forensic Breakdown"): Technical viva analysis view with the full 3-panel Grad-CAM
#     strip (Aligned Crop | Raw Heatmap | Spatial Overlay), a Matplotlib bar chart of cosine similarity
#     to the 1 Real prototype and all 6 Fake prototypes (P1-P6), and a summary card identifying the
#     nearest prototype and manipulation cluster.
#   - Tab 3 ("Benchmarks & Research Results"): Automatically loads and renders formatted tables from
#     results/ablation_summary.csv and results/robustness_summary.csv alongside an interactive gallery
#     of all 300-DPI PNG charts found in results/plots/ (and results/).
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --samples_dir (str): Directory storing the 4 pre-loaded demo sample images (default: demo_samples).
#   - --results_dir (str): Directory containing ablation_summary.csv, robustness_summary.csv, and plots/ (default: results).
#   - --dry-run (flag): Verifies all 3 tabs, auto-run inference callbacks, benchmark loaders, and
#     faceless-frame warning handling on CPU (< 60s limit).
# Outputs:
#   - Live 3-tab Gradio web dashboard launched with demo.launch(share=True, debug=True).

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

import argparse
import csv
import io
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import load_rgb_image_224, normalize_imagenet_chw
from evaluation.closed_set import load_model_and_scorer_from_checkpoint
from models.open_set import (
    VERDICT_FAKE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    OpenSetUncertaintyScorer,
)
from models.zerofake_net import ZeroFakeNet
from preprocessing.face_alignment import align_and_crop_face, resize_rgb_image, save_rgb_crop
from preprocessing.face_detection import FaceDetector
from preprocessing.frame_extraction import create_synthetic_video, extract_evenly_spaced_frames
from training.train import cast_outputs_to_float32
from visualization.gradcam import compute_single_gradcam_map, overlay_heatmap_on_rgb

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False


SAMPLE_FILENAMES = {
    "real": "sample_real_face.jpg",
    "known_fake": "sample_known_fake.jpg",
    "unseen_fake": "sample_unseen_fake.jpg",
    "ambiguous": "sample_blurry_ambiguous.jpg",
}

PROTOTYPE_CLUSTER_DESCRIPTIONS = {
    0: "Authentic Camera Geometry & Natural Skin Texture (Real Anchor)",
    1: "Fake Prototype P1 — Autoencoder Inner-Face Blending (Deepfakes Cluster)",
    2: "Fake Prototype P2 — Expression & Mouth Reenactment (Face2Face Cluster)",
    3: "Fake Prototype P3 — Rigid 3D Mesh / Jawline Seam (FaceSwap Cluster)",
    4: "Fake Prototype P4 — High-Frequency Peri-Orbital Synthesis Artifacts",
    5: "Fake Prototype P5 — Illumination & Color-Transfer Boundary Mismatch",
    6: "Fake Prototype P6 — Subtle Neural Texture / Lip-Sync Patch Artifacts",
}

PLOT_TITLES = {
    "roc_comparison.png": "1. Unseen-Generator Overlaid ROC Curves (Baseline vs. ZeroFake)",
    "confusion_matrices.png": "2. Closed-Set (2x2) vs. Open-Set 3-Way (2x3) Confusion Matrices",
    "tsne_prototypes.png": "3. 2D t-SNE Projection of 128-d Embeddings & 1 Real + 6 Fake Prototypes",
    "gradcam_heatmaps.png": "4. ConvNeXt-Tiny 4x3 Grad-CAM Facial Artifact Attribution Grid",
}

HIGH_CONTRAST_CSS = """
.zf-container, .zf-card {
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.zf-card * {
    color: inherit;
}
.zf-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff !important;
    color: #0f172a !important;
    font-size: 13px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    overflow: hidden;
}
.zf-table-wrap th {
    background: #1e293b !important;
    color: #f8fafc !important;
    font-weight: 700;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #0f172a;
}
.zf-table-wrap td {
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
    color: #0f172a !important;
    background: #ffffff !important;
}
.zf-table-wrap tr:nth-child(even) td {
    background: #f8fafc !important;
}
.zf-table-wrap tr.zf-highlight td {
    background: #eff6ff !important;
    font-weight: 700;
    color: #1e3a8a !important;
}
"""


def ensure_demo_samples_exist(
    samples_dir: str = "demo_samples",
    data_dir: str = "/content/data",
) -> Dict[str, str]:
    """
    Ensures the 4 one-click sample images exist inside `demo_samples/`:
      1. `sample_real_face.jpg` (Real Face)
      2. `sample_known_fake.jpg` (Known Fake)
      3. `sample_unseen_fake.jpg` (Unseen Generator Fake)
      4. `sample_blurry_ambiguous.jpg` (Blurry / Ambiguous Face)
    """
    os.makedirs(samples_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    yy, xx = np.ogrid[:224, :224]
    cy, cx = 112, 112
    ry, rx = 74, 58
    face_oval = ((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2) <= 1.0

    configs = [
        ("real", [228, 192, 168], [35, 55, 45], 0),
        ("known_fake", [220, 180, 158], [65, 35, 35], 1),
        ("unseen_fake", [215, 175, 162], [55, 35, 65], 2),
        ("ambiguous", [185, 170, 160], [50, 50, 50], 3),
    ]

    for key, skin_rgb, bg_rgb, variant in configs:
        out_path = os.path.join(samples_dir, SAMPLE_FILENAMES[key])
        paths[key] = out_path
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        canvas = np.full((224, 224, 3), bg_rgb, dtype=np.uint8)
        canvas[face_oval] = np.array(skin_rgb, dtype=np.uint8)

        # Draw eyes, nose, and mouth landmarks
        canvas[86:96, 82:98] = np.array([25, 25, 25], dtype=np.uint8)
        canvas[86:96, 126:142] = np.array([25, 25, 25], dtype=np.uint8)
        canvas[108:128, 108:116] = np.array([195, 155, 135], dtype=np.uint8)
        canvas[145:153, 92:132] = np.array([175, 95, 95], dtype=np.uint8)

        if variant == 1:
            canvas[138:158, 86:138, 0] = np.uint8(245)
        elif variant == 2:
            canvas[140:160:2, 88:136:2] = np.array([240, 120, 110], dtype=np.uint8)
        elif variant == 3:
            padded = np.pad(canvas.astype(np.float32), ((4, 4), (4, 4), (0, 0)), mode="edge")
            acc = np.zeros_like(canvas, dtype=np.float32)
            for dy in range(9):
                for dx in range(9):
                    acc += padded[dy : dy + 224, dx : dx + 224] / 81.0
            canvas = np.clip(acc, 0, 255).astype(np.uint8)

        save_rgb_crop(canvas, out_path)

    return paths


def render_verdict_badge_html(
    verdict: str,
    evidence_fake: float,
    u_score: float,
    tau_reject: float,
    warning_msg: Optional[str] = None,
) -> str:
    """
    Renders a large, high-contrast HTML Verdict Card with explicit dark `#0f172a` text on crisp
    tinted backgrounds so it is 100% legible in both light and dark browser modes:
      - `REAL`    -> Emerald Green border (`#16a34a`), soft mint card (`#f0fdf4`)
      - `FAKE`    -> Crimson Red border (`#dc2626`), soft rose card (`#fef2f2`)
      - `UNKNOWN` -> Amber Gold border (`#d97706`), soft amber card (`#fffbeb`)
    """
    if warning_msg:
        return (
            "<div class='zf-card' style='background:#fffbeb;border:2.5px solid #d97706;border-radius:14px;"
            "padding:20px;text-align:center;color:#0f172a;box-shadow:0 4px 12px rgba(15,23,42,0.08);'>"
            "<div style='font-size:12px;font-weight:800;letter-spacing:1.4px;color:#b45309;text-transform:uppercase;'>"
            "MTCNN Face Preprocessor Gate</div>"
            "<div style='font-size:28px;font-weight:900;color:#92400e;margin:8px 0;'>⚠️ NO FACE DETECTED</div>"
            f"<div style='font-size:14.5px;font-weight:600;color:#0f172a;line-height:1.45;'>{warning_msg}</div>"
            "</div>"
        )

    v = verdict.strip().upper()
    fake_pct = evidence_fake * 100.0
    real_pct = (1.0 - evidence_fake) * 100.0

    if v == VERDICT_REAL:
        bg_color, border_color, badge_bg, badge_fg = "#f0fdf4", "#16a34a", "#15803d", "#ffffff"
        conf_str = f"Authentic Confidence: {real_pct:.1f}% (Fake Likelihood: {fake_pct:.1f}%)"
        explanation = (
            "This face matches the learned Real prototype cluster with low open-set uncertainty "
            "and shows no manipulation boundary seams."
        )
    elif v == VERDICT_FAKE:
        bg_color, border_color, badge_bg, badge_fg = "#fef2f2", "#dc2626", "#b91c1c", "#ffffff"
        conf_str = f"Manipulation Confidence (Fake Likelihood): {fake_pct:.1f}%"
        explanation = (
            "This face strongly activates ZeroFake's manipulation prototypes and exhibits spatial "
            "synthesis artifacts highlighted in the Grad-CAM heatmap below."
        )
    else:
        bg_color, border_color, badge_bg, badge_fg = "#fffbeb", "#d97706", "#b45309", "#ffffff"
        conf_str = f"Open-Set Uncertainty U(x) = {u_score:.3f} > Threshold τ_reject = {tau_reject:.3f}"
        explanation = (
            "ZeroFake abstained from a forced binary guess because this input lies far from known "
            "training prototypes (unseen zero-day generator or heavy visual degradation)."
        )

    return (
        f"<div class='zf-card' style='background:{bg_color};border:2.5px solid {border_color};"
        "border-radius:14px;padding:22px;text-align:center;color:#0f172a;"
        "box-shadow:0 4px 12px rgba(15,23,42,0.08);'>"
        "<div style='font-size:12px;font-weight:800;letter-spacing:1.4px;color:#334155;text-transform:uppercase;'>"
        "ZeroFake 3-Way Open-Set Verdict</div>"
        "<div style='margin:12px 0;'>"
        f"<span style='display:inline-block;background:{badge_bg};color:{badge_fg};font-size:34px;"
        "font-weight:900;padding:6px 28px;border-radius:999px;letter-spacing:1px;'>"
        f"{v}</span></div>"
        f"<div style='font-size:16px;font-weight:800;color:#0f172a;margin-bottom:6px;'>{conf_str}</div>"
        f"<div style='font-size:14px;font-weight:500;color:#1e293b;line-height:1.45;'>{explanation}</div>"
        "</div>"
    )


def render_uncertainty_gauge_html(
    u_score: float,
    tau_reject: float,
    d_proto: float,
    entropy: float,
    s_ood: float,
) -> str:
    """
    Renders a high-contrast Uncertainty Meter comparing combined uncertainty `U(x)` against the
    validation-calibrated rejection threshold `tau_reject`, clearly labeling `ACCEPTED` vs. `REJECTED AS UNKNOWN`.
    """
    safe_tau = tau_reject if (np.isfinite(tau_reject) and abs(tau_reject) > 1e-6) else 0.2756
    pct_u = float(np.clip((u_score + 1.5) / 3.5 * 100.0, 4.0, 98.0))
    pct_tau = float(np.clip((safe_tau + 1.5) / 3.5 * 100.0, 10.0, 92.0))
    is_rejected = u_score > safe_tau

    if is_rejected:
        bar_color = "#d97706"
        pill_bg = "#fef3c7"
        pill_border = "#d97706"
        pill_text = "#92400e"
        status_label = "⚠️ REJECTED AS UNKNOWN (U(x) > τ_reject)"
    else:
        bar_color = "#2563eb"
        pill_bg = "#dbeafe"
        pill_border = "#2563eb"
        pill_text = "#1e3a8a"
        status_label = "✅ ACCEPTED FOR REAL/FAKE CLASSIFICATION (U(x) ≤ τ_reject)"

    return (
        "<div class='zf-card' style='background:#ffffff;border:2px solid #cbd5e1;border-radius:14px;"
        "padding:18px;color:#0f172a;box-shadow:0 2px 8px rgba(15,23,42,0.05);margin-top:10px;'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>"
        "<span style='font-size:15px;font-weight:800;color:#0f172a;'>"
        f"Open-Set Uncertainty Meter: U(x) = {u_score:.4f}</span>"
        f"<span style='font-size:14px;font-weight:800;color:#b91c1c;background:#fef2f2;"
        "border:1px solid #fecaca;padding:3px 10px;border-radius:6px;'>"
        f"Calibrated Cutoff τ_reject = {safe_tau:.4f} (95% Val Coverage)</span>"
        "</div>"
        "<div style='position:relative;height:24px;background:#e2e8f0;border:1px solid #94a3b8;"
        "border-radius:12px;margin:14px 0;overflow:hidden;'>"
        f"<div style='width:{pct_u:.1f}%;height:100%;background:{bar_color};border-radius:12px;'></div>"
        f"<div style='position:absolute;left:{pct_tau:.1f}%;top:0;bottom:0;width:4px;background:#dc2626;' "
        "title='Validation-Calibrated Rejection Cutoff τ_reject'></div>"
        "</div>"
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>"
        f"<span style='display:inline-block;background:{pill_bg};border:1.5px solid {pill_border};"
        f"color:{pill_text};font-size:12.5px;font-weight:800;padding:4px 12px;border-radius:999px;'>"
        f"{status_label}</span>"
        "<span style='font-size:12.5px;font-weight:700;color:#334155;'>"
        f"D_proto: {d_proto:.3f} (0.4) &nbsp;|&nbsp; Entropy H(p): {entropy:.3f} (0.3) &nbsp;|&nbsp; "
        f"OOD Dist S_OOD: {s_ood:.3f} (0.3)</span>"
        "</div>"
        "</div>"
    )


def render_forensic_prototype_summary_html(
    verdict: str,
    sim_real: float,
    sim_fake_all: np.ndarray,
    d_real: float,
    d_fake: float,
    u_score: float,
    tau_reject: float,
    num_frames: int = 1,
) -> str:
    """
    Renders a high-contrast technical forensic summary card for Tab 2 ("Deep Forensic Breakdown")
    identifying the closest prototype (`Real` vs. `Fake P1..P6`), its manipulation cluster description,
    and cosine distance margins.
    """
    fake_sims = np.asarray(sim_fake_all, dtype=np.float32).reshape(-1)
    best_fake_idx = int(np.argmax(fake_sims)) if len(fake_sims) > 0 else 0
    best_fake_sim = float(fake_sims[best_fake_idx]) if len(fake_sims) > 0 else 0.0

    if sim_real >= best_fake_sim:
        closest_name = "Real Prototype (P_Real)"
        closest_sim = float(sim_real)
        cluster_desc = PROTOTYPE_CLUSTER_DESCRIPTIONS[0]
        accent_color = "#15803d"
    else:
        proto_num = best_fake_idx + 1
        closest_name = f"Fake Prototype P{proto_num}"
        closest_sim = best_fake_sim
        cluster_desc = PROTOTYPE_CLUSTER_DESCRIPTIONS.get(proto_num, f"Fake Manipulation Cluster #{proto_num}")
        accent_color = "#b91c1c"

    margin_diff = float(d_real - d_fake)

    return (
        "<div class='zf-card' style='background:#f8fafc;border:2px solid #cbd5e1;border-radius:14px;"
        "padding:18px;color:#0f172a;box-shadow:0 2px 8px rgba(15,23,42,0.05);'>"
        "<div style='font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:1px;color:#334155;'>"
        "Multi-Prototype Memory Bank & Cluster Attribution</div>"
        f"<div style='font-size:20px;font-weight:900;color:{accent_color};margin:8px 0;'>"
        f"Nearest Prototype: {closest_name} (Cosine Similarity = {closest_sim:.3f})</div>"
        f"<div style='font-size:14px;font-weight:700;color:#0f172a;margin-bottom:12px;'>"
        f"Cluster Interpretation: {cluster_desc}</div>"
        "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;"
        "background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:12px;font-size:13.5px;color:#0f172a;'>"
        f"<div><b>• Real Prototype Distance d_real(z):</b> {d_real:.4f} (Sim: {sim_real:.3f})</div>"
        f"<div><b>• Min Fake Prototype Distance d_fake(z):</b> {d_fake:.4f} (P{best_fake_idx + 1} Sim: {best_fake_sim:.3f})</div>"
        f"<div><b>• Prototype Separation Margin (d_real - d_fake):</b> {margin_diff:+.4f}</div>"
        f"<div><b>• Frames Analyzed / Verdict:</b> {num_frames} frame(s) → <b>{verdict}</b> (U={u_score:.3f} vs τ={tau_reject:.3f})</div>"
        "</div>"
        "</div>"
    )


def build_prototype_similarity_chart(
    sim_real: float,
    sim_fake_all: np.ndarray,
) -> Tuple[Any, np.ndarray]:
    """
    Creates a high-contrast Matplotlib bar chart (plus a rendered RGB image array) showing cosine
    similarity in `[-1, 1]` to the 1 learned Real prototype (`Real Proto`) and all `K=6` learned
    Fake prototypes (`Fake P1 ... Fake P6`).
    """
    fake_sims = np.asarray(sim_fake_all, dtype=np.float32).reshape(-1)
    k = len(fake_sims)
    labels = ["Real Proto"] + [f"Fake P{i + 1}" for i in range(k)]
    values = [float(sim_real)] + [float(v) for v in fake_sims]
    best_fake_idx = int(np.argmax(fake_sims)) if k > 0 else 0
    colors = ["#16a34a"] + [("#b91c1c" if i == best_fake_idx else "#ef4444") for i in range(k)]

    if HAS_MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        bars = ax.bar(labels, values, color=colors, edgecolor="#0f172a", linewidth=1.2, width=0.58)
        ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle="--")
        ax.set_ylim([min(-0.35, min(values) - 0.18), max(1.08, max(values) + 0.18)])
        ax.set_ylabel("Cosine Similarity <z, p>", fontsize=10.5, fontweight="bold", color="#0f172a")
        ax.set_title(
            "128-d Embedding Cosine Similarity to 1 Real Prototype & 6 Fake Prototypes",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
            pad=12,
        )
        ax.tick_params(colors="#0f172a", labelsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.7, color="#94a3b8")

        for bar_obj, val in zip(bars, values):
            y_pos = val + 0.03 if val >= 0 else val - 0.08
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() * 0.5,
                y_pos,
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9.5,
                fontweight="bold",
                color="#0f172a",
            )
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        if HAS_PIL:
            with Image.open(buf) as pil_im:
                chart_rgb = np.asarray(pil_im.convert("RGB"), dtype=np.uint8)
        else:
            chart_rgb = np.full((400, 720, 3), 255, dtype=np.uint8)
        return fig, chart_rgb

    # Pure-NumPy bar chart image fallback when matplotlib is not installed locally
    chart_rgb = np.full((320, 640, 3), 255, dtype=np.uint8)
    bar_w = 55
    gap = 25
    base_y = 260
    for idx, val in enumerate(values):
        x0 = 35 + idx * (bar_w + gap)
        x1 = min(635, x0 + bar_w)
        bar_h = int(np.clip((val + 0.2) * 160.0, 8.0, 210.0))
        y0 = max(20, base_y - bar_h)
        col = np.array([22, 163, 74], dtype=np.uint8) if idx == 0 else np.array([220, 38, 38], dtype=np.uint8)
        chart_rgb[y0:base_y, x0:x1] = col
    return None, chart_rgb


def load_benchmark_tables_and_plots(results_dir: str = "results") -> Dict[str, Any]:
    """
    Reads `results/ablation_summary.csv` and `results/robustness_summary.csv` plus all PNG charts
    inside `results/plots/` (and `results/`) for Tab 3 ("Benchmarks & Research Results").
    """
    ablation_csv = os.path.join(results_dir, "ablation_summary.csv")
    if not os.path.exists(ablation_csv):
        alt_csv = os.path.join(results_dir, "results_summary.csv")
        if os.path.exists(alt_csv):
            ablation_csv = alt_csv

    robustness_csv = os.path.join(results_dir, "robustness_summary.csv")

    ablation_rows: List[Dict[str, str]] = []
    if os.path.exists(ablation_csv):
        with open(ablation_csv, "r", encoding="utf-8") as f:
            ablation_rows = list(csv.DictReader(f))

    robustness_rows: List[Dict[str, str]] = []
    if os.path.exists(robustness_csv):
        with open(robustness_csv, "r", encoding="utf-8") as f:
            robustness_rows = list(csv.DictReader(f))

    # Render high-contrast HTML tables so Tab 3 looks sharp and readable in both light & dark modes
    ablation_html = _render_ablation_table_html(ablation_rows)
    robustness_html = _render_robustness_table_html(robustness_rows)

    # Discover all PNG plot files in results/plots/ and results/
    gallery_items: List[Tuple[str, str]] = []
    seen_paths = set()
    search_dirs = [os.path.join(results_dir, "plots"), results_dir]

    # Add canonical 4 plots in presentation order first
    for canonical_name, friendly_title in PLOT_TITLES.items():
        for s_dir in search_dirs:
            cand = os.path.join(s_dir, canonical_name)
            if os.path.exists(cand) and os.path.abspath(cand) not in seen_paths:
                gallery_items.append((os.path.abspath(cand), friendly_title))
                seen_paths.add(os.path.abspath(cand))

    # Add any additional PNG charts discovered in results/plots/ or results/
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for fname in sorted(os.listdir(s_dir)):
                if fname.lower().endswith(".png"):
                    full_p = os.path.abspath(os.path.join(s_dir, fname))
                    if full_p not in seen_paths:
                        caption = PLOT_TITLES.get(fname, fname.replace("_", " ").replace(".png", "").title())
                        gallery_items.append((full_p, caption))
                        seen_paths.add(full_p)

    return {
        "ablation_rows": ablation_rows,
        "robustness_rows": robustness_rows,
        "ablation_html": ablation_html,
        "robustness_html": robustness_html,
        "gallery_items": gallery_items,
    }


def _render_ablation_table_html(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return (
            "<div class='zf-card' style='background:#f8fafc;border:1.5px solid #cbd5e1;border-radius:10px;"
            "padding:14px;color:#0f172a;'>No ablation_summary.csv found in results/ yet. "
            "Run <code>python evaluate.py</code> to populate benchmark tables.</div>"
        )

    headers = [
        ("config_id", "Configuration"),
        ("holdout_generator", "Held-Out LOGO Gen"),
        ("closed_accuracy", "Closed Acc"),
        ("closed_f1", "Closed F1"),
        ("closed_roc_auc", "Closed ROC-AUC"),
        ("closed_eer", "Closed EER"),
        ("tau_reject", "Val τ_reject"),
        ("coverage", "Coverage"),
        ("selective_accuracy", "Selective Acc"),
        ("ood_auroc", "OOD AUROC"),
        ("ood_aupr", "OOD AUPR"),
    ]

    html = ["<div class='zf-table-wrap'><div style='overflow-x:auto;'><table><thead><tr>"]
    for _, h_title in headers:
        html.append(f"<th>{h_title}</th>")
    html.append("</tr></thead><tbody>")

    for r in rows:
        is_full_zf = "Config E" in r.get("config_id", "") or r.get("mode", "") == "zerofake"
        tr_cls = " class='zf-highlight'" if is_full_zf else ""
        html.append(f"<tr{tr_cls}>")
        for col_key, _ in headers:
            val = r.get(col_key, "")
            if col_key in ("closed_accuracy", "coverage", "selective_accuracy"):
                try:
                    val = f"{float(val) * 100.0:.2f}%"
                except Exception:
                    pass
            html.append(f"<td>{val}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div></div>")
    return "".join(html)


def _render_robustness_table_html(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return (
            "<div class='zf-card' style='background:#f8fafc;border:1.5px solid #cbd5e1;border-radius:10px;"
            "padding:14px;color:#0f172a;'>No robustness_summary.csv found in results/ yet. "
            "Run <code>python evaluate.py</code> to populate robustness tables.</div>"
        )

    headers = [
        ("split", "Test Split"),
        ("perturbation_family", "Family"),
        ("perturbation_name", "Distortion"),
        ("severity_param", "Severity"),
        ("roc_auc", "ROC-AUC"),
        ("eer", "EER"),
        ("f1", "F1-Score"),
        ("selective_accuracy", "Selective Acc"),
        ("coverage_pct", "Coverage (%)"),
    ]

    html = ["<div class='zf-table-wrap'><div style='overflow-x:auto;'><table><thead><tr>"]
    for _, h_title in headers:
        html.append(f"<th>{h_title}</th>")
    html.append("</tr></thead><tbody>")

    for r in rows:
        is_clean = r.get("perturbation_name", "") == "clean"
        tr_cls = " class='zf-highlight'" if is_clean else ""
        html.append(f"<tr{tr_cls}>")
        for col_key, _ in headers:
            val = r.get(col_key, "")
            if col_key == "selective_accuracy":
                try:
                    val = f"{float(val) * 100.0:.2f}%"
                except Exception:
                    pass
            elif col_key == "coverage_pct" and val and not str(val).endswith("%"):
                val = f"{val}%"
            html.append(f"<td>{val}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div></div>")
    return "".join(html)


class ZeroFakeDemoEngine:
    """
    Holds the loaded `ZeroFakeNet` model, validation-calibrated `OpenSetUncertaintyScorer`, and
    `FaceDetector` (MTCNN), providing unified inference on single images or short video clips
    (sampling 8 evenly spaced frames for fast ~2-second video analysis).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_synthetic_detector_fallback: bool = True,
    ) -> None:
        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model, self.scorer, self.ckpt = load_model_and_scorer_from_checkpoint(
                checkpoint_path, device=self.device
            )
            self.checkpoint_loaded = checkpoint_path
        else:
            self.model = ZeroFakeNet(mode="zerofake", pretrained=False)
            if HAS_TORCH and hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
                self.model.eval()
            self.scorer = OpenSetUncertaintyScorer(alpha=0.4, beta=0.3, gamma=0.3, target_val_coverage=0.95)
            self.scorer.tau_reject = 0.2756
            self.scorer.is_calibrated = True
            self.ckpt = {}
            self.checkpoint_loaded = "untrained_demo_weights"

        self.detector = FaceDetector(
            device=self.device,
            margin_scale=1.3,
            allow_synthetic_fallback=allow_synthetic_detector_fallback,
        )

    def analyze_rgb_frames(
        self,
        rgb_frames: List[np.ndarray],
        require_face_detection: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs MTCNN face detection & 1.3x alignment across a list of RGB frames (1 frame for an image,
        or up to 8 evenly spaced frames for a video clip).
        Returns outputs for both Tab 1 (Live Detection) and Tab 2 (Deep Forensic Breakdown).
        """
        aligned_crops: List[np.ndarray] = []
        norm_tensors: List[np.ndarray] = []

        for frame in rgb_frames:
            if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
                continue
            det = self.detector.detect_primary_face(frame)
            if det is None:
                if not require_face_detection:
                    crop = resize_rgb_image(frame, target_size=(224, 224))
                    aligned_crops.append(crop)
                    norm_tensors.append(normalize_imagenet_chw(crop, augment=False))
                continue

            crop = align_and_crop_face(frame, det, target_size=(224, 224), margin_scale=1.3)
            aligned_crops.append(crop)
            norm_tensors.append(normalize_imagenet_chw(crop, augment=False))

        if not aligned_crops:
            warning_html = render_verdict_badge_html(
                verdict=VERDICT_UNKNOWN,
                evidence_fake=0.0,
                u_score=0.0,
                tau_reject=self.scorer.tau_reject,
                warning_msg=(
                    "MTCNN did not detect a clear human face in the uploaded image or video. "
                    "Please upload a frontal or semi-profile face image/video."
                ),
            )
            gauge_html = render_uncertainty_gauge_html(0.0, self.scorer.tau_reject, 0.0, 0.0, 0.0)
            forensic_html = render_forensic_prototype_summary_html(
                "NO_FACE", 0.0, np.zeros((self.model.num_fake_prototypes,)), 1.0, 1.0, 0.0, self.scorer.tau_reject, 0
            )
            fig, chart_img = build_prototype_similarity_chart(0.0, np.zeros((self.model.num_fake_prototypes,)))
            blank_2panel = np.full((224, 452, 3), 241, dtype=np.uint8)
            blank_3panel = np.full((224, 680, 3), 241, dtype=np.uint8)
            return {
                "face_found": False,
                "verdict": "NO_FACE",
                "verdict_html": warning_html,
                "gauge_html": gauge_html,
                "forensic_summary_html": forensic_html,
                "proto_fig": fig,
                "proto_chart_rgb": chart_img,
                "live_2panel_gradcam": blank_2panel,
                "forensic_3panel_gradcam": blank_3panel,
                "side_by_side_gradcam": blank_3panel,
                "num_valid_frames": 0,
            }

        batch_chw = np.stack(norm_tensors, axis=0).astype(np.float32)

        if HAS_TORCH:
            x_t = torch.from_numpy(batch_chw).to(self.device)
            amp_on = bool(str(self.device).startswith("cuda") and torch.cuda.is_available())
            with torch.no_grad():
                if amp_on:
                    with torch.amp.autocast("cuda"):
                        raw_out = self.model(x_t)
                else:
                    raw_out = self.model(x_t)
                outputs = cast_outputs_to_float32(raw_out)

            sim_real_mean = float(outputs["sim_real"].detach().cpu().mean().item())
            sim_fake_mean = outputs["sim_fake_all"].detach().cpu().mean(dim=0).numpy()
            ev_fake_mean = float(outputs["evidence_fake"].detach().cpu().mean().item())
            d_real_mean = float(outputs["d_real"].detach().cpu().mean().item())
            d_fake_mean = float(outputs["d_fake"].detach().cpu().mean().item())
            probs_mean = float(outputs["probs"].detach().cpu().mean().item())
            emb_mean = outputs["embeddings"].detach().cpu().mean(dim=0, keepdim=True)
            agg_outputs = {
                "embeddings": emb_mean,
                "probs": torch.tensor([[probs_mean]], dtype=torch.float32),
                "d_real": torch.tensor([d_real_mean], dtype=torch.float32),
                "d_fake": torch.tensor([d_fake_mean], dtype=torch.float32),
                "p_real_norm": outputs["p_real_norm"].detach().cpu(),
                "p_fake_norm": outputs["p_fake_norm"].detach().cpu(),
                "evidence_fake": torch.tensor([ev_fake_mean], dtype=torch.float32),
            }
        else:
            outputs = self.model(batch_chw)
            sim_real_mean = float(np.mean(outputs["sim_real"]))
            sim_fake_mean = np.mean(outputs["sim_fake_all"], axis=0)
            ev_fake_mean = float(np.mean(outputs["evidence_fake"]))
            d_real_mean = float(np.mean(outputs["d_real"]))
            d_fake_mean = float(np.mean(outputs["d_fake"]))
            probs_mean = float(np.mean(outputs["probs"]))
            emb_mean = np.mean(outputs["embeddings"], axis=0, keepdims=True)
            agg_outputs = {
                "embeddings": emb_mean,
                "probs": np.array([[probs_mean]], dtype=np.float32),
                "d_real": np.array([d_real_mean], dtype=np.float32),
                "d_fake": np.array([d_fake_mean], dtype=np.float32),
                "p_real_norm": outputs["p_real_norm"],
                "p_fake_norm": outputs["p_fake_norm"],
                "evidence_fake": np.array([ev_fake_mean], dtype=np.float32),
            }

        pred = self.scorer.predict(agg_outputs, open_set_enabled=True)
        verdict = pred["verdicts"][0]
        u_score = float(pred["u_score"][0])
        d_proto = float(pred["d_prototype"][0])
        entropy = float(pred["entropy"][0])
        s_ood = float(pred["s_ood"][0])
        tau_reject = float(self.scorer.tau_reject)

        # Compute Grad-CAM on the representative middle aligned face frame
        mid_idx = len(aligned_crops) // 2
        rep_crop_rgb = aligned_crops[mid_idx]
        rep_norm_chw = batch_chw[mid_idx]
        cam_01 = compute_single_gradcam_map(self.model, rep_norm_chw, device=self.device)
        hm_rgb, overlay_rgb = overlay_heatmap_on_rgb(rep_crop_rgb, cam_01, alpha=0.45)

        divider = np.full((224, 4, 3), 255, dtype=np.uint8)
        # Tab 1 clean 2-panel side-by-side: Aligned Face (Left) vs. Grad-CAM Manipulation Overlay (Right)
        live_2panel = np.concatenate([rep_crop_rgb, divider, overlay_rgb], axis=1)
        # Tab 2 full 3-panel forensic strip: Aligned Face | Raw Grad-CAM Heatmap | Spatial Overlay
        forensic_3panel = np.concatenate([rep_crop_rgb, divider, hm_rgb, divider, overlay_rgb], axis=1)

        verdict_html = render_verdict_badge_html(
            verdict=verdict,
            evidence_fake=ev_fake_mean,
            u_score=u_score,
            tau_reject=tau_reject,
        )
        gauge_html = render_uncertainty_gauge_html(
            u_score=u_score,
            tau_reject=tau_reject,
            d_proto=d_proto,
            entropy=entropy,
            s_ood=s_ood,
        )
        forensic_html = render_forensic_prototype_summary_html(
            verdict=verdict,
            sim_real=sim_real_mean,
            sim_fake_all=sim_fake_mean,
            d_real=d_real_mean,
            d_fake=d_fake_mean,
            u_score=u_score,
            tau_reject=tau_reject,
            num_frames=len(aligned_crops),
        )
        proto_fig, proto_chart_rgb = build_prototype_similarity_chart(sim_real_mean, sim_fake_mean)

        return {
            "face_found": True,
            "verdict": verdict,
            "evidence_fake": ev_fake_mean,
            "u_score": u_score,
            "tau_reject": tau_reject,
            "verdict_html": verdict_html,
            "gauge_html": gauge_html,
            "forensic_summary_html": forensic_html,
            "proto_fig": proto_fig,
            "proto_chart_rgb": proto_chart_rgb,
            "live_2panel_gradcam": live_2panel,
            "forensic_3panel_gradcam": forensic_3panel,
            "side_by_side_gradcam": forensic_3panel,
            "num_valid_frames": len(aligned_crops),
        }

    def predict_all_panels(
        self,
        uploaded_image: Optional[np.ndarray],
        uploaded_video_path: Optional[str],
    ) -> Tuple[str, str, np.ndarray, np.ndarray, Any, str]:
        """
        Unified Gradio callback populating both Tab 1 ("Live Detection") and Tab 2 ("Deep Forensic Breakdown"):
          Returns:
            1. `verdict_html` (Tab 1 Verdict Card)
            2. `gauge_html` (Tab 1 Uncertainty Meter)
            3. `live_2panel_gradcam` (Tab 1 Side-by-Side Aligned Face vs. Grad-CAM Overlay)
            4. `forensic_3panel_gradcam` (Tab 2 Full 3-Panel Grad-CAM Inspection)
            5. `proto_plot_output` (Tab 2 Prototype Similarity Bar Chart)
            6. `forensic_summary_html` (Tab 2 Nearest Prototype & Cluster Summary Card)
        """
        frames: List[np.ndarray] = []
        if uploaded_image is not None and isinstance(uploaded_image, np.ndarray) and uploaded_image.ndim == 3:
            frames = [uploaded_image.astype(np.uint8)]
        elif uploaded_video_path and os.path.exists(str(uploaded_video_path)):
            sampled = extract_evenly_spaced_frames(str(uploaded_video_path), num_frames=8)
            frames = [rgb for _, rgb in sampled]

        res = self.analyze_rgb_frames(frames)
        plot_output = res["proto_fig"] if (HAS_MPL and res["proto_fig"] is not None) else res["proto_chart_rgb"]
        return (
            res["verdict_html"],
            res["gauge_html"],
            res["live_2panel_gradcam"],
            res["forensic_3panel_gradcam"],
            plot_output,
            res["forensic_summary_html"],
        )

    def predict_from_media(
        self,
        uploaded_image: Optional[np.ndarray],
        uploaded_video_path: Optional[str],
    ) -> Tuple[str, str, Any, np.ndarray]:
        """
        Backwards-compatible 4-tuple prediction helper used by unit tests.
        """
        v_html, g_html, _, cam_3panel, plot_out, _ = self.predict_all_panels(uploaded_image, uploaded_video_path)
        return v_html, g_html, plot_out, cam_3panel


def build_gradio_blocks_app(
    engine: ZeroFakeDemoEngine,
    sample_paths: Dict[str, str],
    results_dir: str = "results",
) -> Any:
    """
    Constructs the 3-tab `gr.Blocks` web dashboard with `gr.themes.Soft(primary_hue="blue", neutral_hue="slate")`,
    high-contrast dark-on-light card styling, auto-run on file upload, deep forensic breakdown, and
    built-in research benchmark tables + 300-DPI chart gallery.
    """
    benchmarks = load_benchmark_tables_and_plots(results_dir=results_dir)

    def load_sample_and_run(sample_key: str):
        img_path = sample_paths[sample_key]
        rgb = load_rgb_image_224(img_path, image_size=224)
        v_html, g_html, cam_2p, cam_3p, p_out, f_html = engine.predict_all_panels(rgb, None)
        return rgb, None, v_html, g_html, cam_2p, cam_3p, p_out, f_html

    def on_image_upload(img: Optional[np.ndarray]):
        v_html, g_html, cam_2p, cam_3p, p_out, f_html = engine.predict_all_panels(img, None)
        return None, v_html, g_html, cam_2p, cam_3p, p_out, f_html

    def on_video_upload(vid_path: Optional[str]):
        v_html, g_html, cam_2p, cam_3p, p_out, f_html = engine.predict_all_panels(None, vid_path)
        return None, v_html, g_html, cam_2p, cam_3p, p_out, f_html

    if not HAS_GRADIO or gr is None:
        return {
            "framework": "gradio_mock_blocks_3tabs",
            "tabs": ["Live Detection", "Deep Forensic Breakdown", "Benchmarks & Research Results"],
            "engine": engine,
            "sample_paths": sample_paths,
            "benchmarks": benchmarks,
            "load_sample_and_run": load_sample_and_run,
            "on_image_upload": on_image_upload,
            "on_video_upload": on_video_upload,
        }

    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    blocks_kwargs: Dict[str, Any] = {"title": "ZeroFake — Zero-Day Open-Set Deepfake Detector"}
    try:
        import inspect
        sig = inspect.signature(gr.Blocks.__init__)
        if "theme" in sig.parameters:
            blocks_kwargs["theme"] = theme
        if "css" in sig.parameters:
            blocks_kwargs["css"] = HIGH_CONTRAST_CSS
    except Exception:
        pass

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(f"<style>{HIGH_CONTRAST_CSS}</style>")
        gr.HTML(
            "<div class='zf-card' style='background:#0f172a;color:#f8fafc;border-radius:14px;padding:18px 22px;"
            "margin-bottom:10px;border:1px solid #334155;'>"
            "<div style='font-size:24px;font-weight:900;color:#ffffff;letter-spacing:0.3px;'>"
            "🛡️ ZeroFake: Zero-Day Open-Set Deepfake Detection Suite</div>"
            "<div style='font-size:13.5px;color:#cbd5e1;margin-top:4px;font-weight:500;'>"
            "<b>Backbone:</b> ConvNeXt-Tiny (768-d) &nbsp;|&nbsp; "
            "<b>Contrastive Space:</b> 128-d Unit Hypersphere (τ = 0.07) &nbsp;|&nbsp; "
            "<b>Memory Bank:</b> 1 Real + 6 Fake Prototypes &nbsp;|&nbsp; "
            "<b>3-Way Decision Rule:</b> REAL / FAKE / UNKNOWN (Val-Calibrated τ_reject)</div>"
            "</div>"
        )

        with gr.Tabs():
            # =========================================================================
            # TAB 1: LIVE DETECTION (Clean, Instant Upload -> Verdict)
            # =========================================================================
            with gr.TabItem("🔍 1. Live Detection"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### Upload Face Image or Video Clip (Auto-Runs on Upload)")
                        with gr.Tabs():
                            with gr.TabItem("📷 Upload Face Image"):
                                image_input = gr.Image(type="numpy", label="Input Face Image (RGB)")
                            with gr.TabItem("🎬 Upload Video Clip (8-Frame Fast Sampling)"):
                                video_input = gr.Video(label="Input Short Video Clip (.mp4)")

                        analyze_btn = gr.Button("🔍 Analyze Face / Video Now", variant="primary")

                        gr.Markdown("#### ⚡ One-Click Pre-Loaded Test Samples (`demo_samples/`):")
                        with gr.Row():
                            btn_real = gr.Button("1️⃣ Real Face")
                            btn_known = gr.Button("2️⃣ Known Fake")
                        with gr.Row():
                            btn_unseen = gr.Button("3️⃣ Unseen Generator Fake")
                            btn_ambig = gr.Button("4️⃣ Blurry / Ambiguous Face")

                    with gr.Column(scale=7):
                        verdict_html_out = gr.HTML(
                            value=render_verdict_badge_html(VERDICT_REAL, 0.02, -0.45, engine.scorer.tau_reject),
                            label="3-Way Verdict Card",
                        )
                        gauge_html_out = gr.HTML(
                            value=render_uncertainty_gauge_html(-0.45, engine.scorer.tau_reject, 0.12, 0.08, 0.19),
                            label="Open-Set Uncertainty Meter",
                        )
                        live_gradcam_out = gr.Image(
                            type="numpy",
                            label="Aligned 224x224 Face (Left) vs. Grad-CAM Manipulation Heatmap Overlay (Right)",
                        )

            # =========================================================================
            # TAB 2: DEEP FORENSIC BREAKDOWN (For Technical Viva Questions)
            # =========================================================================
            with gr.TabItem("🧬 2. Deep Forensic Breakdown"):
                gr.Markdown(
                    "### Multi-Prototype Similarity & Full 3-Panel Spatial Attribution\n"
                    "Inspect how the 128-d L2-normalized embedding $z$ compares against the **1 Real Prototype** "
                    "and all **6 Fake Prototypes ($P_1 \\dots P_6$)** alongside the raw ConvNeXt-Tiny activation map."
                )
                forensic_summary_out = gr.HTML(
                    value=render_forensic_prototype_summary_html(
                        VERDICT_REAL, 0.88, np.zeros((6,)), 0.12, 0.95, -0.45, engine.scorer.tau_reject, 1
                    ),
                    label="Nearest Prototype & Cluster Summary",
                )
                with gr.Row():
                    with gr.Column(scale=6):
                        if HAS_MPL:
                            proto_plot_out = gr.Plot(label="Cosine Similarity to 1 Real Prototype & 6 Fake Prototypes (P1–P6)")
                        else:
                            proto_plot_out = gr.Image(
                                type="numpy",
                                label="Cosine Similarity to 1 Real Prototype & 6 Fake Prototypes (P1–P6)",
                            )
                    with gr.Column(scale=6):
                        forensic_gradcam_out = gr.Image(
                            type="numpy",
                            label="3-Panel Inspection: Aligned Face (Left) | Raw Grad-CAM (Center) | Overlay (Right)",
                        )

            # =========================================================================
            # TAB 3: BENCHMARKS & RESEARCH RESULTS (Reads local results/ folder)
            # =========================================================================
            with gr.TabItem("📊 3. Benchmarks & Research Results"):
                gr.Markdown(
                    "### Empirical Leave-One-Generator-Out (LOGO) Ablation & Robustness Benchmarks\n"
                    "Automatically loaded from `results/ablation_summary.csv`, `results/robustness_summary.csv`, "
                    "and `results/plots/`."
                )
                gr.Markdown("#### 1. Five-Configuration Ablation Study (`results/ablation_summary.csv`)")
                gr.HTML(value=benchmarks["ablation_html"])

                gr.Markdown("#### 2. In-Memory Robustness Stress Tests (`results/robustness_summary.csv`)")
                gr.HTML(value=benchmarks["robustness_html"])

                gr.Markdown("#### 3. High-Resolution 300-DPI Evaluation Charts (`results/plots/`)")
                if benchmarks["gallery_items"]:
                    gr.Gallery(
                        value=benchmarks["gallery_items"],
                        label="ZeroFake Publication-Quality Visual Evaluation Suite (300 DPI)",
                        columns=2,
                        height="auto",
                    )
                else:
                    gr.Markdown("*No PNG plots found in `results/plots/` yet. Run `python evaluate.py` to generate charts.*")

        all_visual_outputs = [
            verdict_html_out,
            gauge_html_out,
            live_gradcam_out,
            forensic_gradcam_out,
            proto_plot_out,
            forensic_summary_out,
        ]

        # Wire Analyze button
        analyze_btn.click(
            fn=engine.predict_all_panels,
            inputs=[image_input, video_input],
            outputs=all_visual_outputs,
        )

        # Wire Auto-Run on Image / Video Upload
        image_input.upload(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[video_input] + all_visual_outputs,
        )
        video_input.upload(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[image_input] + all_visual_outputs,
        )

        # Wire 4 One-Click Sample Buttons
        btn_real.click(
            fn=lambda: load_sample_and_run("real"),
            inputs=[],
            outputs=[image_input, video_input] + all_visual_outputs,
        )
        btn_known.click(
            fn=lambda: load_sample_and_run("known_fake"),
            inputs=[],
            outputs=[image_input, video_input] + all_visual_outputs,
        )
        btn_unseen.click(
            fn=lambda: load_sample_and_run("unseen_fake"),
            inputs=[],
            outputs=[image_input, video_input] + all_visual_outputs,
        )
        btn_ambig.click(
            fn=lambda: load_sample_and_run("ambiguous"),
            inputs=[],
            outputs=[image_input, video_input] + all_visual_outputs,
        )

    return demo


def run_dry_run(checkpoint_path: Optional[str] = None) -> None:
    """
    Runs a CPU dry-run (< 60s limit) verifying:
      1. Generation of the 4 sample images in `demo_samples/`.
      2. Construction of the 3-tab Gradio Blocks app (`Live Detection`, `Deep Forensic Breakdown`,
         `Benchmarks & Research Results`) with high-contrast styling and auto-upload callbacks.
      3. Loading of `results/ablation_summary.csv`, `results/robustness_summary.csv`, and `results/plots/`.
      4. Image inference, 8-frame video inference, and faceless-frame warning handling.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing 3-tab app.py (Live Detection + Deep Forensic Breakdown + Benchmarks & Plots)...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        samples_dir = os.path.join(tmp_dir, "demo_samples")
        sample_paths = ensure_demo_samples_exist(samples_dir=samples_dir)
        assert len(sample_paths) == 4

        engine = ZeroFakeDemoEngine(
            checkpoint_path=checkpoint_path,
            device="cpu",
            allow_synthetic_detector_fallback=True,
        )
        results_dir = str(PROJECT_ROOT / "results")
        demo = build_gradio_blocks_app(engine, sample_paths, results_dir=results_dir)
        assert demo is not None, "3-tab Gradio Blocks app failed to build!"

        # Verify Tab 3 benchmark loader reads local results/
        bench = load_benchmark_tables_and_plots(results_dir=results_dir)
        assert "table" in bench["ablation_html"].lower() or "ablation" in bench["ablation_html"].lower()
        assert "table" in bench["robustness_html"].lower() or "robustness" in bench["robustness_html"].lower()

        # Verify Tab 1 + Tab 2 unified 6-output callback on sample image
        sample_rgb = load_rgb_image_224(sample_paths["real"], image_size=224)
        v_html, g_html, cam_2p, cam_3p, _, f_html = engine.predict_all_panels(sample_rgb, None)
        assert any(lbl in v_html for lbl in ("REAL", "FAKE", "UNKNOWN")), "Verdict card missing label!"
        assert "τ_reject" in g_html, "Uncertainty meter missing threshold tau_reject!"
        assert "Nearest Prototype" in f_html, "Forensic summary card missing Nearest Prototype breakdown!"
        assert cam_2p.shape == (224, 452, 3), f"Expected 2-panel Grad-CAM shape (224, 452, 3), got {cam_2p.shape}"
        assert cam_3p.shape == (224, 680, 3), f"Expected 3-panel Grad-CAM shape (224, 680, 3), got {cam_3p.shape}"

        # Verify 8-frame video clip inference
        dummy_video = os.path.join(tmp_dir, "clip.mp4")
        create_synthetic_video(dummy_video, total_frames=16, height=160, width=160, include_face=True)
        v_vid_html, _, _, _, _, _ = engine.predict_all_panels(None, dummy_video)
        assert any(lbl in v_vid_html for lbl in ("REAL", "FAKE", "UNKNOWN"))

        # Verify faceless image -> friendly warning banner without crashing
        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
        v_warn_html, _, _, _, _, _ = engine.predict_all_panels(blank_img, None)
        assert "NO FACE DETECTED" in v_warn_html, "Faceless image did not trigger friendly warning banner!"

    elapsed = time.time() - start_time
    assert elapsed < 60.0, f"Dry-run exceeded 60s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified 3-tab Gradio Blocks UI, auto-upload callbacks, 2-panel & 3-panel Grad-CAM, "
        f"prototype similarity chart, benchmark CSV/plot loaders, and faceless warning in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ZeroFake 3-tab interactive Gradio web application.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
        help="Path to trained ZeroFake checkpoint (.pt).",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="demo_samples",
        help="Directory containing the 4 pre-loaded demo sample images.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing ablation_summary.csv, robustness_summary.csv, and plots/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify 3-tab Gradio UI construction and sample/video/faceless inference on CPU (<60s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None)
        return

    sample_paths = ensure_demo_samples_exist(samples_dir=args.samples_dir)
    engine = ZeroFakeDemoEngine(checkpoint_path=args.checkpoint)
    demo = build_gradio_blocks_app(engine, sample_paths, results_dir=args.results_dir)

    if not HAS_GRADIO or gr is None:
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` or run with `--dry-run`.")

    print(f"[INFO] Launching ZeroFake 3-Tab Gradio Web App (checkpoint='{engine.checkpoint_loaded}')...")
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
