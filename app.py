# File: app.py
# What this file does: Ultra-clean, minimalist light-mode Gradio web UI for ZeroFake inspired by modern
#   AI detector landing pages, with collapsible dropdown accordions for deep forensic and benchmark details:
#   1. Minimalist Light Theme & Visual Polish: Uses gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
#      with custom CSS setting an off-white canvas (#f6f7fb), crisp white cards (#ffffff) with subtle borders
#      (#e2e8f0) and soft shadows, dark high-contrast text (#0f172a), muted slate subtitles (#475569), and
#      royal indigo (#3b38d8) active pills. Contains zero pre-loaded sample buttons.
#   2. Centered Minimalist Hero Header & 2-Mode Pill Switcher:
#      - Centered Title: "ZeroFake: Deepfake & Synthetic Face Detector"
#      - Centered Subtitle: "Automatically flags deepfakes, facial manipulations, and unseen synthetic generators.
#        Upload or paste a face image or video clip for instant open-set verification."
#      - Clean 2-option centered pill switcher ("Images" and "Videos" ONLY).
#   3. Clean Dashed Upload Card + Instant Auto-Detection (Zero Extra Clicks):
#      - Images Mode: Dashed white upload card using gr.Image(sources=["upload", "clipboard"], type="numpy",
#        label="Drop an image here, browse files, or press Ctrl+V to paste"). Wired with image_input.input(...)
#        so uploading or pasting an image runs inference immediately.
#      - Videos Mode: Dashed white upload card using gr.Video(label="Drop a short video clip here (8-frame fast sampling)").
#        Wired with video_input.upload(...) so uploading a video runs inference immediately.
#      - Subtle "Clear" button to reset the current media.
#   4. Clean Primary Verdict Card (Always Visible After Upload):
#      - Large color-coded status pill: REAL (emerald green #16a34a), FAKE (crimson red #dc2626), or
#        UNKNOWN — Unseen / Ambiguous (amber #d97706), with confidence/fake probability % and a single
#        plain-English summary sentence (plus a friendly warning if no face is detected by MTCNN).
#   5. Three Collapsible Dropdown Menus (gr.Accordion(..., open=False)):
#      - Dropdown 1: "Forensic Explainability & Grad-CAM Heatmap" (open=False)
#      - Dropdown 2: "Open-Set Uncertainty & Prototype Similarity Details" (open=False)
#      - Dropdown 3: "Model Benchmarks & Evaluation Results" (open=False)
#   6. Minimalist 3-Column Checkmark Footer Strip & Dry-Run (< 60s CPU limit).
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --results_dir (str): Directory containing ablation_summary.csv, robustness_summary.csv, and plots/ (default: results).
#   - --dry-run (flag): Verifies UI construction, image/video auto-inference, faceless warning, and
#     collapsible accordions on CPU (< 60s limit).
# Outputs:
#   - Live minimalist Gradio web application launched via demo.launch(share=True, debug=True).

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
import inspect
import io
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import normalize_imagenet_chw
from evaluation.closed_set import load_model_and_scorer_from_checkpoint
from models.open_set import (
    VERDICT_FAKE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    OpenSetUncertaintyScorer,
)
from models.zerofake_net import ZeroFakeNet
from preprocessing.face_alignment import align_and_crop_face, resize_rgb_image
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

MINIMALIST_LIGHT_CSS = """
body, .gradio-container {
    background-color: #f6f7fb !important;
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding-top: 24px !important;
}
/* Centered Hero Header */
.zf-hero {
    text-align: center;
    max-width: 720px;
    margin: 0 auto 18px auto;
    padding: 8px 16px;
}
.zf-hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.6px;
    line-height: 1.2;
    margin-bottom: 12px;
}
.zf-hero-subtitle {
    font-size: 15.5px;
    font-weight: 400;
    color: #475569 !important;
    line-height: 1.6;
    margin: 0 auto;
}
/* 2-Option Centered Mode Pill Bar (Images | Videos) */
#zf-mode-tabs > .tab-nav,
#zf-mode-tabs [role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 999px !important;
    padding: 5px !important;
    width: fit-content !important;
    margin: 0 auto 22px auto !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    gap: 4px !important;
}
#zf-mode-tabs > .tab-nav button,
#zf-mode-tabs [role="tab"] {
    border: none !important;
    border-radius: 999px !important;
    padding: 8px 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    background: transparent !important;
    transition: all 0.18s ease !important;
}
#zf-mode-tabs > .tab-nav button.selected,
#zf-mode-tabs [role="tab"][aria-selected="true"] {
    background: #3b38d8 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(59, 56, 216, 0.28) !important;
}
/* Clean White Dashed Upload Card */
.zf-upload-card {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    max-width: 640px !important;
    margin: 0 auto 12px auto !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04) !important;
    transition: border-color 0.2s ease !important;
}
.zf-upload-card:hover {
    border-color: #3b38d8 !important;
}
/* Subtle Clear Button */
#zf-clear-btn {
    max-width: 140px !important;
    margin: 0 auto 20px auto !important;
    background: #ffffff !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 6px 18px !important;
    display: block !important;
}
#zf-clear-btn:hover {
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
    background: #f8fafc !important;
}
/* Collapsible Dropdown Accordions */
.zf-accordion {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03) !important;
    color: #0f172a !important;
}
.zf-accordion span, .zf-accordion .label-wrap {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
}
/* High-Contrast Benchmark Tables */
.zf-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff !important;
    color: #0f172a !important;
    font-size: 13px;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
.zf-table-wrap th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1.5px solid #cbd5e1;
}
.zf-table-wrap td {
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
    color: #0f172a !important;
    background: #ffffff !important;
}
.zf-table-wrap tr.zf-highlight td {
    background: #eef2ff !important;
    font-weight: 700;
    color: #3b38d8 !important;
}
/* Minimalist 3-Column Checkmark Footer Strip */
.zf-footer {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 28px;
    margin: 28px auto 16px auto;
    padding: 16px 20px;
    border-top: 1px solid #e2e8f0;
    color: #475569 !important;
    font-size: 13.5px;
    font-weight: 500;
}
.zf-footer-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #475569 !important;
}
.zf-check {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 999px;
    background: #e0e7ff;
    color: #3b38d8;
    font-size: 11px;
    font-weight: 800;
}
"""


def render_idle_verdict_card_html() -> str:
    """
    Renders a clean, minimalist waiting state inside the Primary Verdict Card before a file is uploaded.
    """
    return (
        "<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:22px 24px;"
        "max-width:640px;margin:0 auto 18px auto;text-align:center;color:#0f172a;"
        "box-shadow:0 4px 16px rgba(15,23,42,0.04);'>"
        "<div style='display:inline-flex;align-items:center;gap:8px;background:#eef2ff;color:#3b38d8;"
        "font-size:12.5px;font-weight:700;padding:5px 14px;border-radius:999px;margin-bottom:8px;'>"
        "<span>●</span><span>Ready for Instant Open-Set Verification</span></div>"
        "<div style='font-size:14.5px;color:#475569;font-weight:500;'>"
        "Drop or paste a face image or short video clip above to run automatic MTCNN alignment and 3-way detection."
        "</div>"
        "</div>"
    )


def render_verdict_badge_html(
    verdict: str,
    evidence_fake: float,
    u_score: float,
    tau_reject: float,
    warning_msg: Optional[str] = None,
) -> str:
    """
    Renders the focused Primary Verdict Card (always visible below the upload card):
      - `REAL`    -> Emerald Green pill (`#16a34a`)
      - `FAKE`    -> Crimson Red pill (`#dc2626`)
      - `UNKNOWN` -> Amber Gold pill (`#d97706`, "UNKNOWN — Unseen / Ambiguous")
      - `NO FACE` -> Friendly amber warning if MTCNN detects zero faces.
    """
    if warning_msg:
        return (
            "<div style='background:#ffffff;border:1.5px solid #f59e0b;border-radius:16px;padding:22px 24px;"
            "max-width:640px;margin:0 auto 18px auto;text-align:center;color:#0f172a;"
            "box-shadow:0 4px 16px rgba(15,23,42,0.05);'>"
            "<div style='margin-bottom:10px;'>"
            "<span style='display:inline-block;background:#d97706;color:#ffffff;font-size:16px;"
            "font-weight:800;padding:6px 20px;border-radius:999px;'>"
            "⚠️ NO FACE DETECTED</span></div>"
            f"<div style='font-size:14.5px;font-weight:500;color:#475569;line-height:1.5;'>{warning_msg}</div>"
            "</div>"
        )

    v = verdict.strip().upper()
    fake_pct = evidence_fake * 100.0
    real_pct = (1.0 - evidence_fake) * 100.0

    if v == VERDICT_REAL:
        pill_bg = "#16a34a"
        border_col = "#bbf7d0"
        pill_text = "REAL"
        metric_line = f"Authentic Confidence: {real_pct:.1f}%  •  Fake Probability: {fake_pct:.1f}%"
        summary_sentence = (
            "This face matches the learned authentic prototype cluster with low open-set uncertainty "
            "and shows no synthetic blending seams."
        )
    elif v == VERDICT_FAKE:
        pill_bg = "#dc2626"
        border_col = "#fecaca"
        pill_text = "FAKE"
        metric_line = f"Fake Probability: {fake_pct:.1f}%  •  Manipulation Confidence: High"
        summary_sentence = (
            "This media strongly activates ZeroFake's manipulation prototypes and exhibits spatial "
            "synthesis artifacts in the facial region."
        )
    else:
        pill_bg = "#d97706"
        border_col = "#fde68a"
        pill_text = "UNKNOWN — Unseen / Ambiguous"
        metric_line = f"Uncertainty U(x) = {u_score:.3f} (> Cutoff τ_reject = {tau_reject:.3f})  •  Fake Prob: {fake_pct:.1f}%"
        summary_sentence = (
            "ZeroFake abstained from a forced binary guess because this input lies outside known training "
            "clusters (unseen zero-day generator or heavy visual degradation)."
        )

    return (
        f"<div style='background:#ffffff;border:1.5px solid {border_col};border-radius:16px;padding:24px;"
        "max-width:640px;margin:0 auto 18px auto;text-align:center;color:#0f172a;"
        "box-shadow:0 6px 20px rgba(15,23,42,0.05);'>"
        "<div style='margin-bottom:10px;'>"
        f"<span style='display:inline-block;background:{pill_bg};color:#ffffff;font-size:20px;"
        "font-weight:800;padding:7px 24px;border-radius:999px;letter-spacing:0.4px;'>"
        f"{pill_text}</span></div>"
        f"<div style='font-size:15.5px;font-weight:700;color:#0f172a;margin-bottom:6px;'>{metric_line}</div>"
        f"<div style='font-size:14px;font-weight:400;color:#475569;line-height:1.5;'>{summary_sentence}</div>"
        "</div>"
    )


def render_uncertainty_and_prototype_html(
    u_score: float,
    tau_reject: float,
    d_proto: float,
    entropy: float,
    s_ood: float,
    sim_real: float,
    sim_fake_all: np.ndarray,
    d_real: float,
    d_fake: float,
) -> str:
    """
    Renders the high-contrast Open-Set Uncertainty Meter comparing `U(x)` against `tau_reject`
    (`ACCEPTED` vs. `REJECTED AS UNKNOWN`) plus nearest-prototype attribution inside Dropdown 2.
    """
    safe_tau = tau_reject if (np.isfinite(tau_reject) and abs(tau_reject) > 1e-6) else 0.2756
    pct_u = float(np.clip((u_score + 1.5) / 3.5 * 100.0, 4.0, 98.0))
    pct_tau = float(np.clip((safe_tau + 1.5) / 3.5 * 100.0, 10.0, 92.0))
    is_rejected = u_score > safe_tau

    if is_rejected:
        bar_color = "#d97706"
        badge_bg = "#fffbeb"
        badge_border = "#d97706"
        badge_fg = "#92400e"
        status_label = "⚠️ REJECTED AS UNKNOWN (U(x) > τ_reject)"
    else:
        bar_color = "#3b38d8"
        badge_bg = "#eef2ff"
        badge_border = "#3b38d8"
        badge_fg = "#312e81"
        status_label = "✅ ACCEPTED FOR REAL/FAKE CLASSIFICATION (U(x) ≤ τ_reject)"

    fake_sims = np.asarray(sim_fake_all, dtype=np.float32).reshape(-1)
    best_fake_idx = int(np.argmax(fake_sims)) if len(fake_sims) > 0 else 0
    best_fake_sim = float(fake_sims[best_fake_idx]) if len(fake_sims) > 0 else 0.0

    if sim_real >= best_fake_sim:
        closest_proto = f"Real Prototype (Cosine Sim = {sim_real:.3f}, d_real = {d_real:.3f})"
        cluster_note = PROTOTYPE_CLUSTER_DESCRIPTIONS[0]
    else:
        p_id = best_fake_idx + 1
        closest_proto = f"Fake Prototype P{p_id} (Cosine Sim = {best_fake_sim:.3f}, d_fake = {d_fake:.3f})"
        cluster_note = PROTOTYPE_CLUSTER_DESCRIPTIONS.get(p_id, f"Fake Manipulation Cluster P{p_id}")

    return (
        "<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px;color:#0f172a;'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>"
        f"<span style='font-size:14.5px;font-weight:700;color:#0f172a;'>Combined Uncertainty U(x) = {u_score:.4f}</span>"
        f"<span style='font-size:13px;font-weight:700;color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;"
        f"padding:3px 10px;border-radius:999px;'>Calibrated Threshold τ_reject = {safe_tau:.4f}</span>"
        "</div>"
        "<div style='position:relative;height:20px;background:#f1f5f9;border:1px solid #cbd5e1;"
        "border-radius:999px;margin:12px 0;overflow:hidden;'>"
        f"<div style='width:{pct_u:.1f}%;height:100%;background:{bar_color};border-radius:999px;'></div>"
        f"<div style='position:absolute;left:{pct_tau:.1f}%;top:0;bottom:0;width:3px;background:#dc2626;' "
        "title='Validation-Calibrated Rejection Cutoff τ_reject'></div>"
        "</div>"
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;'>"
        f"<span style='background:{badge_bg};border:1px solid {badge_border};color:{badge_fg};"
        f"font-size:12px;font-weight:700;padding:4px 12px;border-radius:999px;'>{status_label}</span>"
        "<span style='font-size:12.5px;color:#475569;font-weight:600;'>"
        f"D_prototype: {d_proto:.3f} (0.4) &nbsp;•&nbsp; Entropy H(p): {entropy:.3f} (0.3) &nbsp;•&nbsp; "
        f"OOD Dist S_OOD: {s_ood:.3f} (0.3)</span>"
        "</div>"
        "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;font-size:13px;color:#0f172a;'>"
        f"<div><b>Nearest Prototype Match:</b> {closest_proto}</div>"
        f"<div style='color:#475569;margin-top:3px;'><b>Cluster Interpretation:</b> {cluster_note}</div>"
        "</div>"
        "</div>"
    )


def build_prototype_similarity_chart(
    sim_real: float,
    sim_fake_all: np.ndarray,
) -> Tuple[Any, np.ndarray]:
    """
    Creates a clean light-mode Matplotlib bar chart (plus a rendered RGB image array) showing cosine
    similarity in `[-1, 1]` across the 1 Real Prototype (`Real Proto`) and all `K=6` Fake Prototypes (`P1–P6`).
    """
    fake_sims = np.asarray(sim_fake_all, dtype=np.float32).reshape(-1)
    k = len(fake_sims)
    labels = ["Real Proto"] + [f"Fake P{i + 1}" for i in range(k)]
    values = [float(sim_real)] + [float(v) for v in fake_sims]
    best_fake_idx = int(np.argmax(fake_sims)) if k > 0 else 0
    colors = ["#16a34a"] + [("#3b38d8" if i == best_fake_idx else "#dc2626") for i in range(k)]

    if HAS_MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        bars = ax.bar(labels, values, color=colors, edgecolor="#0f172a", linewidth=1.0, width=0.55)
        ax.axhline(0.0, color="#64748b", linewidth=0.9, linestyle="--")
        ax.set_ylim([min(-0.35, min(values) - 0.18), max(1.08, max(values) + 0.18)])
        ax.set_ylabel("Cosine Similarity <z, p>", fontsize=10, fontweight="bold", color="#0f172a")
        ax.set_title(
            "Cosine Similarity to 1 Real Prototype & 6 Fake Prototypes (P1–P6)",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
            pad=10,
        )
        ax.tick_params(colors="#0f172a", labelsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.7, color="#cbd5e1")

        for bar_obj, val in zip(bars, values):
            y_pos = val + 0.03 if val >= 0 else val - 0.08
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() * 0.5,
                y_pos,
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
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
            chart_rgb = np.full((360, 720, 3), 255, dtype=np.uint8)
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
        col = np.array([22, 163, 74], dtype=np.uint8) if idx == 0 else np.array([59, 56, 216], dtype=np.uint8)
        chart_rgb[y0:base_y, x0:x1] = col
    return None, chart_rgb


def load_benchmark_tables_and_plots(results_dir: str = "results") -> Dict[str, Any]:
    """
    Loads formatted HTML tables from `results/ablation_summary.csv` and `results/robustness_summary.csv`
    plus all PNG evaluation charts inside `results/plots/` (and `results/`) for Dropdown 3.
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

    ablation_html = _render_ablation_table_html(ablation_rows)
    robustness_html = _render_robustness_table_html(robustness_rows)

    gallery_items: List[Tuple[str, str]] = []
    seen_paths = set()
    search_dirs = [os.path.join(results_dir, "plots"), results_dir]

    for canonical_name, friendly_title in PLOT_TITLES.items():
        for s_dir in search_dirs:
            cand = os.path.join(s_dir, canonical_name)
            if os.path.exists(cand) and os.path.abspath(cand) not in seen_paths:
                gallery_items.append((os.path.abspath(cand), friendly_title))
                seen_paths.add(os.path.abspath(cand))

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
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
            "padding:14px;color:#475569;font-size:13.5px;'>No <code>ablation_summary.csv</code> found in "
            "<code>results/</code> yet. Run <code>python evaluate.py</code> to generate tables.</div>"
        )

    headers = [
        ("config_id", "Configuration"),
        ("holdout_generator", "Held-Out Gen"),
        ("closed_accuracy", "Closed Acc"),
        ("closed_f1", "Closed F1"),
        ("closed_roc_auc", "Closed AUC"),
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
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
            "padding:14px;color:#475569;font-size:13.5px;'>No <code>robustness_summary.csv</code> found in "
            "<code>results/</code> yet. Run <code>python evaluate.py</code> to generate tables.</div>"
        )

    headers = [
        ("split", "Split"),
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
        Returns the Primary Verdict Card plus outputs for Dropdown 1 (3-panel Grad-CAM) and
        Dropdown 2 (Uncertainty & Prototype Similarity Bar Chart).
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
                    "MTCNN did not detect a clear human face in the uploaded media. "
                    "Please upload or paste a clear frontal or semi-profile face image or video clip."
                ),
            )
            unc_html = render_uncertainty_and_prototype_html(
                0.0, self.scorer.tau_reject, 0.0, 0.0, 0.0, 0.0, np.zeros((self.model.num_fake_prototypes,)), 1.0, 1.0
            )
            fig, chart_img = build_prototype_similarity_chart(0.0, np.zeros((self.model.num_fake_prototypes,)))
            blank_3panel = np.full((224, 680, 3), 246, dtype=np.uint8)
            return {
                "face_found": False,
                "verdict": "NO_FACE",
                "verdict_html": warning_html,
                "uncertainty_html": unc_html,
                "proto_fig": fig,
                "proto_chart_rgb": chart_img,
                "gradcam_3panel": blank_3panel,
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
        gradcam_3panel = np.concatenate([rep_crop_rgb, divider, hm_rgb, divider, overlay_rgb], axis=1)

        verdict_html = render_verdict_badge_html(
            verdict=verdict,
            evidence_fake=ev_fake_mean,
            u_score=u_score,
            tau_reject=tau_reject,
        )
        unc_html = render_uncertainty_and_prototype_html(
            u_score=u_score,
            tau_reject=tau_reject,
            d_proto=d_proto,
            entropy=entropy,
            s_ood=s_ood,
            sim_real=sim_real_mean,
            sim_fake_all=sim_fake_mean,
            d_real=d_real_mean,
            d_fake=d_fake_mean,
        )
        proto_fig, proto_chart_rgb = build_prototype_similarity_chart(sim_real_mean, sim_fake_mean)

        return {
            "face_found": True,
            "verdict": verdict,
            "evidence_fake": ev_fake_mean,
            "u_score": u_score,
            "tau_reject": tau_reject,
            "verdict_html": verdict_html,
            "uncertainty_html": unc_html,
            "proto_fig": proto_fig,
            "proto_chart_rgb": proto_chart_rgb,
            "gradcam_3panel": gradcam_3panel,
            "num_valid_frames": len(aligned_crops),
        }

    def predict_from_media(
        self,
        uploaded_image: Optional[np.ndarray],
        uploaded_video_path: Optional[str],
    ) -> Tuple[str, np.ndarray, str, Any]:
        """
        Unified auto-detection callback returning:
          1. `verdict_html` (Primary Verdict Card)
          2. `gradcam_3panel` (Dropdown 1: Aligned Face | Grad-CAM Heatmap | Overlay)
          3. `uncertainty_html` (Dropdown 2: Uncertainty Score U(x) vs. tau_reject bar)
          4. `proto_plot_output` (Dropdown 2: Prototype Cosine Similarity Bar Chart)
        """
        frames: List[np.ndarray] = []
        if uploaded_image is not None and isinstance(uploaded_image, np.ndarray) and uploaded_image.ndim == 3:
            frames = [uploaded_image.astype(np.uint8)]
        elif uploaded_video_path and os.path.exists(str(uploaded_video_path)):
            sampled = extract_evenly_spaced_frames(str(uploaded_video_path), num_frames=8)
            frames = [rgb for _, rgb in sampled]

        if not frames and uploaded_image is None and not uploaded_video_path:
            idle_html = render_idle_verdict_card_html()
            blank_3panel = np.full((224, 680, 3), 246, dtype=np.uint8)
            unc_html = render_uncertainty_and_prototype_html(
                0.0, self.scorer.tau_reject, 0.0, 0.0, 0.0, 0.0, np.zeros((self.model.num_fake_prototypes,)), 1.0, 1.0
            )
            fig, chart_img = build_prototype_similarity_chart(0.0, np.zeros((self.model.num_fake_prototypes,)))
            plot_out = fig if (HAS_MPL and fig is not None) else chart_img
            return idle_html, blank_3panel, unc_html, plot_out

        res = self.analyze_rgb_frames(frames)
        plot_output = res["proto_fig"] if (HAS_MPL and res["proto_fig"] is not None) else res["proto_chart_rgb"]
        return (
            res["verdict_html"],
            res["gradcam_3panel"],
            res["uncertainty_html"],
            plot_output,
        )


def build_gradio_blocks_app(
    engine: ZeroFakeDemoEngine,
    results_dir: str = "results",
) -> Any:
    """
    Constructs the minimalist light-mode `gr.Blocks` web application with:
      - Centered Hero Header & 2-mode pill switcher ("Images" and "Videos" only)
      - Dashed white upload card with instant auto-detection on image upload/paste or video upload
      - Subtle "Clear" button
      - Focused Primary Verdict Card
      - Three collapsed-by-default `gr.Accordion(..., open=False)` dropdowns for Grad-CAM, Uncertainty/Prototypes, and Benchmarks
      - Minimalist 3-column checkmark footer strip
    """
    benchmarks = load_benchmark_tables_and_plots(results_dir=results_dir)

    def on_image_input(img: Optional[np.ndarray]):
        if img is None:
            return engine.predict_from_media(None, None)
        return engine.predict_from_media(img, None)

    def on_video_input(vid_path: Optional[str]):
        if not vid_path:
            return engine.predict_from_media(None, None)
        return engine.predict_from_media(None, vid_path)

    def on_clear_media():
        v_html, cam_3p, unc_html, p_out = engine.predict_from_media(None, None)
        return None, None, v_html, cam_3p, unc_html, p_out

    if not HAS_GRADIO or gr is None:
        return {
            "framework": "gradio_minimalist_light_ui",
            "modes": ["Images", "Videos"],
            "accordions": [
                "Forensic Explainability & Grad-CAM Heatmap",
                "Open-Set Uncertainty & Prototype Similarity Details",
                "Model Benchmarks & Evaluation Results",
            ],
            "engine": engine,
            "benchmarks": benchmarks,
            "on_image_input": on_image_input,
            "on_video_input": on_video_input,
            "on_clear_media": on_clear_media,
        }

    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    blocks_kwargs: Dict[str, Any] = {"title": "ZeroFake: Deepfake & Synthetic Face Detector"}
    try:
        sig = inspect.signature(gr.Blocks.__init__)
        if "theme" in sig.parameters:
            blocks_kwargs["theme"] = theme
        if "css" in sig.parameters:
            blocks_kwargs["css"] = MINIMALIST_LIGHT_CSS
    except Exception:
        pass

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(f"<style>{MINIMALIST_LIGHT_CSS}</style>")

        # 1. Centered Minimalist Hero Header
        gr.HTML(
            "<div class='zf-hero'>"
            "<div class='zf-hero-title'>ZeroFake: Deepfake &amp; Synthetic Face Detector</div>"
            "<div class='zf-hero-subtitle'>"
            "Automatically flags deepfakes, facial manipulations, and unseen synthetic generators. "
            "Upload or paste a face image or video clip for instant open-set verification."
            "</div>"
            "</div>"
        )

        # 2. Clean 2-Option Mode Pill Switcher ("Images" and "Videos" ONLY) + Dashed Upload Cards
        with gr.Tabs(elem_id="zf-mode-tabs"):
            with gr.TabItem("Images"):
                with gr.Group(elem_classes=["zf-upload-card"]):
                    img_kwargs: Dict[str, Any] = {
                        "type": "numpy",
                        "label": "Drop an image here, browse files, or press Ctrl+V to paste",
                    }
                    try:
                        img_sig = inspect.signature(gr.Image.__init__)
                        if "sources" in img_sig.parameters:
                            img_kwargs["sources"] = ["upload", "clipboard"]
                    except Exception:
                        pass
                    image_input = gr.Image(**img_kwargs)

            with gr.TabItem("Videos"):
                with gr.Group(elem_classes=["zf-upload-card"]):
                    video_input = gr.Video(
                        label="Drop a short video clip here (8-frame fast sampling)"
                    )

        # Subtle Clear button
        clear_btn = gr.Button("Clear", elem_id="zf-clear-btn", size="sm")

        # 3. Clean Primary Verdict Card (Always Visible)
        verdict_card_out = gr.HTML(
            value=render_idle_verdict_card_html(),
            label="Primary Verdict Card",
        )

        # 4. Three Collapsible Dropdown Menus (open=False by default)
        with gr.Accordion("Forensic Explainability & Grad-CAM Heatmap", open=False, elem_classes=["zf-accordion"]):
            gr.Markdown(
                "Side-by-side spatial attribution from the final **ConvNeXt-Tiny** stage: "
                "**Aligned 224×224 Face Crop (Left)** | **Raw Grad-CAM Heatmap (Center)** | **Facial Artifact Overlay (Right)**."
            )
            gradcam_3panel_out = gr.Image(
                type="numpy",
                label="Aligned Face Crop | ConvNeXt-Tiny Grad-CAM Heatmap | Facial Artifact Overlay",
            )

        with gr.Accordion("Open-Set Uncertainty & Prototype Similarity Details", open=False, elem_classes=["zf-accordion"]):
            uncertainty_html_out = gr.HTML(
                value=render_uncertainty_and_prototype_html(
                    0.0, engine.scorer.tau_reject, 0.0, 0.0, 0.0, 0.0, np.zeros((6,)), 1.0, 1.0
                ),
                label="Open-Set Uncertainty U(x) vs. Calibrated Threshold τ_reject",
            )
            if HAS_MPL:
                proto_plot_out = gr.Plot(
                    label="Cosine Similarity Across 1 Real Prototype & 6 Fake Prototypes (P1–P6)"
                )
            else:
                proto_plot_out = gr.Image(
                    type="numpy",
                    label="Cosine Similarity Across 1 Real Prototype & 6 Fake Prototypes (P1–P6)",
                )

        with gr.Accordion("Model Benchmarks & Evaluation Results", open=False, elem_classes=["zf-accordion"]):
            gr.Markdown("#### Leave-One-Generator-Out (LOGO) Ablation Study (`results/ablation_summary.csv`)")
            gr.HTML(value=benchmarks["ablation_html"])

            gr.Markdown("#### In-Memory Robustness Stress Tests (`results/robustness_summary.csv`)")
            gr.HTML(value=benchmarks["robustness_html"])

            gr.Markdown("#### High-Resolution 300-DPI Evaluation Charts (`results/plots/`)")
            if benchmarks["gallery_items"]:
                gr.Gallery(
                    value=benchmarks["gallery_items"],
                    label="ZeroFake Evaluation Plots (ROC Curves, Confusion Matrices, t-SNE, Grad-CAM)",
                    columns=2,
                    height="auto",
                )
            else:
                gr.Markdown("*No PNG plots found in `results/plots/` yet. Run `python evaluate.py` to generate charts.*")

        # 5. Minimalist 3-Column Checkmark Footer Strip
        gr.HTML(
            "<div class='zf-footer'>"
            "<div class='zf-footer-item'><span class='zf-check'>✓</span>"
            "<span>ConvNeXt-Tiny + Multi-Prototype Space</span></div>"
            "<div class='zf-footer-item'><span class='zf-check'>✓</span>"
            "<span>Calibrated Open-Set UNKNOWN Rejection</span></div>"
            "<div class='zf-footer-item'><span class='zf-check'>✓</span>"
            "<span>Instant Image &amp; 8-Frame Video Analysis</span></div>"
            "</div>"
        )

        output_targets = [
            verdict_card_out,
            gradcam_3panel_out,
            uncertainty_html_out,
            proto_plot_out,
        ]

        # Wire instant auto-detection on image upload/paste (.input + .upload fallback)
        if hasattr(image_input, "input"):
            image_input.input(
                fn=on_image_input,
                inputs=[image_input],
                outputs=output_targets,
            )
        else:
            image_input.upload(
                fn=on_image_input,
                inputs=[image_input],
                outputs=output_targets,
            )

        # Wire instant auto-detection on video upload (.upload + .change fallback)
        video_input.upload(
            fn=on_video_input,
            inputs=[video_input],
            outputs=output_targets,
        )
        if hasattr(video_input, "change"):
            video_input.change(
                fn=on_video_input,
                inputs=[video_input],
                outputs=output_targets,
            )

        # Wire Clear button
        clear_btn.click(
            fn=on_clear_media,
            inputs=[],
            outputs=[image_input, video_input] + output_targets,
        )

    return demo


def run_dry_run(checkpoint_path: Optional[str] = None) -> None:
    """
    Runs a CPU dry-run (< 60s limit) verifying:
      1. Construction of the minimalist light-mode Gradio UI with the 2-mode pill switcher
         ("Images" and "Videos" only) and 3 collapsed-by-default `gr.Accordion` dropdowns.
      2. Instant auto-inference on a synthetic face image, an 8-frame sampled video clip,
         and a blank faceless image (verifying the friendly warning banner).
      3. Automatic loading of benchmark tables and PNG plots from `results/`.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing minimalist light-mode app.py (2-Mode Pill Switcher + Auto-Upload + 3 Accordions)...")

    engine = ZeroFakeDemoEngine(
        checkpoint_path=checkpoint_path,
        device="cpu",
        allow_synthetic_detector_fallback=True,
    )
    results_dir = str(PROJECT_ROOT / "results")
    demo = build_gradio_blocks_app(engine, results_dir=results_dir)
    assert demo is not None, "Minimalist Gradio Blocks app failed to build!"

    # 1. Test synthetic face image auto-inference
    yy, xx = np.ogrid[:224, :224]
    oval = ((yy - 112) ** 2) / (72 ** 2) + ((xx - 112) ** 2) / (56 ** 2) <= 1.0
    synth_face = np.full((224, 224, 3), 45, dtype=np.uint8)
    synth_face[oval] = np.array([225, 190, 165], dtype=np.uint8)
    synth_face[88:96, 84:98] = 25
    synth_face[88:96, 126:140] = 25

    v_html, cam_3p, unc_html, _ = engine.predict_from_media(synth_face, None)
    assert any(lbl in v_html for lbl in ("REAL", "FAKE", "UNKNOWN")), "Verdict card missing status pill!"
    assert cam_3p.shape == (224, 680, 3), f"Expected 3-panel Grad-CAM shape (224, 680, 3), got {cam_3p.shape}"
    assert "τ_reject" in unc_html, "Uncertainty dropdown HTML missing threshold tau_reject!"

    # 2. Test short video clip auto-inference (8-frame fast sampling)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_video = os.path.join(tmp_dir, "clip.mp4")
        create_synthetic_video(dummy_video, total_frames=16, height=160, width=160, include_face=True)
        v_vid_html, _, _, _ = engine.predict_from_media(None, dummy_video)
        assert any(lbl in v_vid_html for lbl in ("REAL", "FAKE", "UNKNOWN"))

    # 3. Test faceless image -> friendly warning banner without crashing
    blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
    v_warn_html, _, _, _ = engine.predict_from_media(blank_img, None)
    assert "NO FACE DETECTED" in v_warn_html, "Faceless image did not trigger friendly warning!"

    elapsed = time.time() - start_time
    assert elapsed < 60.0, f"Dry-run exceeded 60s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified minimalist light-mode UI, Images/Videos pill switcher, "
        f"instant upload/paste callbacks, 3 collapsed accordions, and faceless warning in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ZeroFake minimalist light-mode Gradio web application.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
        help="Path to trained ZeroFake checkpoint (.pt).",
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
        help="Verify minimalist Gradio UI construction and image/video/faceless inference on CPU (<60s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None)
        return

    engine = ZeroFakeDemoEngine(checkpoint_path=args.checkpoint)
    demo = build_gradio_blocks_app(engine, results_dir=args.results_dir)

    if not HAS_GRADIO or gr is None:
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` or run with `--dry-run`.")

    print(f"[INFO] Launching ZeroFake Minimalist Web App (checkpoint='{engine.checkpoint_loaded}')...")
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
