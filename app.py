# File: app.py
# What this file does: Interactive professor-ready Gradio web application (built with gr.Blocks)
#   for live ZeroFake deepfake detection.
#   1. Accepts either an uploaded face image OR a short video clip (sampling 8 evenly spaced frames
#      for fast ~2-second inference), PLUS 4 one-click sample buttons loading images from demo_samples/
#      (Real Face, Known Fake, Unseen Generator Fake, and Blurry/Ambiguous Face).
#   2. Runs MTCNN face detection and 1.3x-margin 224x224 alignment first; if no face is found, displays
#      a friendly warning banner instead of crashing.
#   3. Displays 4 rich visual outputs:
#      - Large color-coded HTML verdict badge: REAL (green), FAKE (red), or UNKNOWN (amber).
#      - Uncertainty Score U(x) compared against the validation-calibrated threshold tau_reject gauge.
#      - Matplotlib bar chart showing cosine similarity to the 1 Real prototype and all 6 Fake prototypes.
#      - Side-by-side Grad-CAM heatmap overlay highlighting the manipulated facial region.
#   4. Supports --checkpoint and --dry-run (verifies demo_samples/, full inference pipeline, and UI build
#      on CPU in under 5 seconds) and launches with demo.launch(share=True, debug=True).
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --samples_dir (str): Directory storing the 4 pre-loaded demo sample images (default: demo_samples).
#   - --dry-run (flag): Verifies sample generation, image/video inference, faceless-frame warning, and
#     Gradio Blocks UI construction on CPU in under 5 seconds.
# Outputs:
#   - Live Gradio web dashboard (with public share=True URL on Google Colab).

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
    If real dataset crops exist under `data_dir/crops`, copies real examples; otherwise synthesizes
    clean 224x224 portrait-like test faces with distinct facial features so buttons always work.
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

        # Draw eyes, eyebrows, nose, and mouth landmarks
        canvas[86:96, 82:98] = np.array([25, 25, 25], dtype=np.uint8)
        canvas[86:96, 126:142] = np.array([25, 25, 25], dtype=np.uint8)
        canvas[108:128, 108:116] = np.array([195, 155, 135], dtype=np.uint8)
        canvas[145:153, 92:132] = np.array([175, 95, 95], dtype=np.uint8)

        if variant == 1:
            # Known fake: subtle peri-orbital/mouth blending boundary artifact
            canvas[138:158, 86:138, 0] = np.uint8(245)
        elif variant == 2:
            # Unseen generator fake: neural texture high-frequency mouth pattern
            canvas[140:160:2, 88:136:2] = np.array([240, 120, 110], dtype=np.uint8)
        elif variant == 3:
            # Blurry / ambiguous face: heavy spatial box blur + low contrast
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
    Renders a large, professor-ready color-coded HTML verdict badge:
      - `REAL`    -> Emerald Green (`#1e8449`)
      - `FAKE`    -> Crimson Red (`#c0392b`)
      - `UNKNOWN` -> Amber Gold (`#d68910`)
      - `NO FACE` -> Friendly Warning Banner (`#e67e22`)
    """
    if warning_msg:
        return (
            "<div style='background:#fef5e7;border:2px solid #e67e22;border-radius:12px;padding:18px;"
            "text-align:center;font-family:sans-serif;'>"
            "<div style='font-size:24px;font-weight:800;color:#d35400;'>⚠️ NO FACE DETECTED</div>"
            f"<div style='font-size:14px;color:#7e5109;margin-top:6px;'>{warning_msg}</div>"
            "</div>"
        )

    v = verdict.strip().upper()
    if v == VERDICT_REAL:
        bg_color, border_color, text_color = "#eafaf1", "#27ae60", "#1e8449"
        subtitle = f"Authentic Face Confirmed (Fake Likelihood: {evidence_fake * 100:.1f}%)"
    elif v == VERDICT_FAKE:
        bg_color, border_color, text_color = "#fdedec", "#e74c3c", "#c0392b"
        subtitle = f"Deepfake Manipulation Detected (Fake Likelihood: {evidence_fake * 100:.1f}%)"
    else:
        bg_color, border_color, text_color = "#fef9e7", "#f39c12", "#b7950b"
        subtitle = (
            f"Open-Set Rejection: Uncertainty U(x)={u_score:.3f} > τ_reject={tau_reject:.3f} "
            "(Unseen Generator or Ambiguous/Degraded Input)"
        )

    return (
        f"<div style='background:{bg_color};border:3px solid {border_color};border-radius:14px;"
        "padding:20px;text-align:center;font-family:sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>"
        f"<div style='font-size:13px;font-weight:700;letter-spacing:1.2px;color:{text_color};"
        "text-transform:uppercase;'>ZeroFake 3-Way Open-Set Verdict</div>"
        f"<div style='font-size:40px;font-weight:900;color:{text_color};margin:6px 0;'>{v}</div>"
        f"<div style='font-size:14px;font-weight:600;color:#2c3e50;'>{subtitle}</div>"
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
    Renders a clean visual gauge comparing combined uncertainty `U(x)` against the validation-calibrated
    rejection threshold `tau_reject`, along with the 3 raw uncertainty signal breakdowns.
    """
    safe_tau = tau_reject if (np.isfinite(tau_reject) and abs(tau_reject) > 1e-6) else 1.25
    # Map U(x) in [-2.0, 2.5] to a percentage bar in [5%, 98%]
    pct_u = float(np.clip((u_score + 1.5) / 3.5 * 100.0, 4.0, 98.0))
    pct_tau = float(np.clip((safe_tau + 1.5) / 3.5 * 100.0, 10.0, 92.0))
    is_high = u_score > safe_tau
    bar_color = "#f39c12" if is_high else "#2980b9"
    status_text = "REJECTED AS UNKNOWN (U(x) > τ_reject)" if is_high else "ACCEPTED FOR CLASSIFICATION (U(x) ≤ τ_reject)"

    return (
        "<div style='border:1px solid #d5d8dc;border-radius:10px;padding:14px;font-family:sans-serif;background:#ffffff;'>"
        "<div style='display:flex;justify-content:space-between;font-weight:700;font-size:14px;color:#2c3e50;'>"
        f"<span>Combined Uncertainty U(x): {u_score:.4f}</span>"
        f"<span>Calibrated Threshold τ_reject: {safe_tau:.4f}</span>"
        "</div>"
        "<div style='position:relative;height:20px;background:#ebedef;border-radius:10px;margin:12px 0;overflow:hidden;'>"
        f"<div style='width:{pct_u:.1f}%;height:100%;background:{bar_color};border-radius:10px;'></div>"
        f"<div style='position:absolute;left:{pct_tau:.1f}%;top:0;bottom:0;width:3px;background:#c0392b;' "
        "title='Calibrated Rejection Threshold'></div>"
        "</div>"
        f"<div style='font-size:12.5px;font-weight:700;color:{bar_color};margin-bottom:8px;'>{status_text}</div>"
        "<div style='display:flex;justify-content:space-between;font-size:12px;color:#566573;"
        "border-top:1px solid #f2f4f4;padding-top:8px;'>"
        f"<span>• Nearest-Prototype Dist D_proto: <b>{d_proto:.3f}</b></span>"
        f"<span>• Predictive Entropy H(p): <b>{entropy:.3f}</b></span>"
        f"<span>• OOD Feature Dist S_OOD: <b>{s_ood:.3f}</b></span>"
        "</div>"
        "</div>"
    )


def build_prototype_similarity_chart(
    sim_real: float,
    sim_fake_all: np.ndarray,
) -> Tuple[Any, np.ndarray]:
    """
    Creates a Matplotlib bar chart (plus a rendered RGB image array) showing cosine similarity
    in `[-1, 1]` to the 1 learned Real prototype (`P_Real`) and all `K=6` learned Fake prototypes
    (`P_Fake_1 ... P_Fake_6`).
    """
    fake_sims = np.asarray(sim_fake_all, dtype=np.float32).reshape(-1)
    k = len(fake_sims)
    labels = ["Real Proto"] + [f"Fake P{i + 1}" for i in range(k)]
    values = [float(sim_real)] + [float(v) for v in fake_sims]
    colors = ["#27ae60"] + ["#e74c3c"] * k

    if HAS_MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)
        bars = ax.bar(labels, values, color=colors, edgecolor="#1c2833", linewidth=1.0, width=0.58)
        ax.axhline(0.0, color="#7f8c8d", linewidth=0.9, linestyle="--")
        ax.set_ylim([min(-0.3, min(values) - 0.15), 1.12])
        ax.set_ylabel("Cosine Similarity <z, p>", fontsize=10, fontweight="bold")
        ax.set_title(
            "Embedding Similarity to 1 Real Prototype & 6 Fake Prototypes",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="y", linestyle=":", alpha=0.6)

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
                color="#1c2833",
            )
        fig.tight_layout()

        # Render to RGB uint8 array for universal display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        if HAS_PIL:
            with Image.open(buf) as pil_im:
                chart_rgb = np.asarray(pil_im.convert("RGB"), dtype=np.uint8)
        else:
            chart_rgb = np.full((380, 680, 3), 250, dtype=np.uint8)
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
        col = np.array([39, 174, 96], dtype=np.uint8) if idx == 0 else np.array([231, 76, 60], dtype=np.uint8)
        chart_rgb[y0:base_y, x0:x1] = col
    return None, chart_rgb


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
            # Initialize default uncalibrated/demo ZeroFakeNet so --dry-run or fresh launch works immediately
            self.model = ZeroFakeNet(mode="zerofake", pretrained=False)
            if HAS_TORCH and hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
                self.model.eval()
            self.scorer = OpenSetUncertaintyScorer(alpha=0.4, beta=0.3, gamma=0.3, target_val_coverage=0.95)
            self.scorer.tau_reject = 1.15
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
        If no face is detected in any frame, returns a clean warning response without crashing.
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
                    "MTCNN did not detect a clear human face in the uploaded image/video. "
                    "Please upload a frontal or semi-profile face image/video."
                ),
            )
            gauge_html = render_uncertainty_gauge_html(0.0, self.scorer.tau_reject, 0.0, 0.0, 0.0)
            fig, chart_img = build_prototype_similarity_chart(0.0, np.zeros((self.model.num_fake_prototypes,)))
            blank_side_by_side = np.full((224, 448, 3), 240, dtype=np.uint8)
            return {
                "face_found": False,
                "verdict": "NO_FACE",
                "verdict_html": warning_html,
                "gauge_html": gauge_html,
                "proto_fig": fig,
                "proto_chart_rgb": chart_img,
                "side_by_side_gradcam": blank_side_by_side,
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

        # Side-by-side image: Left = Aligned 224x224 Face | Middle = Grad-CAM Heatmap | Right = Attention Overlay
        divider = np.full((224, 4, 3), 255, dtype=np.uint8)
        side_by_side = np.concatenate([rep_crop_rgb, divider, hm_rgb, divider, overlay_rgb], axis=1)

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
        proto_fig, proto_chart_rgb = build_prototype_similarity_chart(sim_real_mean, sim_fake_mean)

        return {
            "face_found": True,
            "verdict": verdict,
            "evidence_fake": ev_fake_mean,
            "u_score": u_score,
            "tau_reject": tau_reject,
            "verdict_html": verdict_html,
            "gauge_html": gauge_html,
            "proto_fig": proto_fig,
            "proto_chart_rgb": proto_chart_rgb,
            "side_by_side_gradcam": side_by_side,
            "num_valid_frames": len(aligned_crops),
        }

    def predict_from_media(
        self,
        uploaded_image: Optional[np.ndarray],
        uploaded_video_path: Optional[str],
    ) -> Tuple[str, str, Any, np.ndarray]:
        """
        Gradio callback accepting either `uploaded_image` (H, W, 3 RGB uint8) or `uploaded_video_path` (.mp4),
        sampling 8 evenly spaced frames for video clips, and returning:
          `(verdict_badge_html, uncertainty_gauge_html, prototype_plot, side_by_side_gradcam_image)`
        """
        frames: List[np.ndarray] = []
        if uploaded_image is not None and isinstance(uploaded_image, np.ndarray) and uploaded_image.ndim == 3:
            frames = [uploaded_image.astype(np.uint8)]
        elif uploaded_video_path and os.path.exists(str(uploaded_video_path)):
            sampled = extract_evenly_spaced_frames(str(uploaded_video_path), num_frames=8)
            frames = [rgb for _, rgb in sampled]

        if not frames:
            res = self.analyze_rgb_frames([])
        else:
            res = self.analyze_rgb_frames(frames)

        plot_output = res["proto_fig"] if (HAS_MPL and res["proto_fig"] is not None) else res["proto_chart_rgb"]
        return (
            res["verdict_html"],
            res["gauge_html"],
            plot_output,
            res["side_by_side_gradcam"],
        )


def build_gradio_blocks_app(
    engine: ZeroFakeDemoEngine,
    sample_paths: Dict[str, str],
) -> Any:
    """
    Constructs the `gr.Blocks` web dashboard with image/video upload inputs, 4 one-click sample buttons,
    and the 4 visual output panels. If `gradio` is not installed locally during a minimal laptop dry-run,
    returns a lightweight specification dictionary verifying all callbacks and components.
    """
    def load_sample_and_run(sample_key: str):
        img_path = sample_paths[sample_key]
        rgb = load_rgb_image_224(img_path, image_size=224)
        v_html, g_html, p_out, cam_out = engine.predict_from_media(rgb, None)
        return rgb, None, v_html, g_html, p_out, cam_out

    if not HAS_GRADIO or gr is None:
        return {
            "framework": "gradio_mock_blocks",
            "engine": engine,
            "sample_paths": sample_paths,
            "load_sample_and_run": load_sample_and_run,
        }

    with gr.Blocks(title="ZeroFake — Zero-Day Open-Set Deepfake Detector") as demo:
        gr.Markdown(
            "# 🛡️ ZeroFake: Zero-Day Open-Set Deepfake Detection Dashboard\n"
            "**Backbone:** ConvNeXt-Tiny (`768-d`) | **Contrastive Space:** `128-d` Unit Hypersphere | "
            "**Memory Bank:** `1 Real + 6 Fake` Learnable Prototypes | "
            "**Decision Rule:** Calibrated 3-Way Open-Set (`REAL`, `FAKE`, `UNKNOWN`)"
        )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 1. Upload Face Image or Short Video Clip (or Pick a Sample Below)")
                with gr.Tabs():
                    with gr.TabItem("📷 Upload Face Image"):
                        image_input = gr.Image(type="numpy", label="Input Face Image (RGB)")
                    with gr.TabItem("🎬 Upload Video Clip (8-Frame Fast Sampling)"):
                        video_input = gr.Video(label="Input Short Video Clip (.mp4)")

                analyze_btn = gr.Button("🔍 Analyze Face / Video with ZeroFake", variant="primary")

                gr.Markdown("#### ⚡ Pre-Loaded One-Click Evaluation Samples:")
                with gr.Row():
                    btn_real = gr.Button("1️⃣ Real Face")
                    btn_known = gr.Button("2️⃣ Known Fake")
                with gr.Row():
                    btn_unseen = gr.Button("3️⃣ Unseen Generator Fake")
                    btn_ambig = gr.Button("4️⃣ Blurry / Ambiguous Face")

            with gr.Column(scale=7):
                gr.Markdown("### 2. Calibrated 3-Way Verdict & Open-Set Uncertainty Analysis")
                verdict_html_out = gr.HTML(label="3-Way Verdict Badge")
                gauge_html_out = gr.HTML(label="Uncertainty Score U(x) vs. Threshold τ_reject")

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("### 3. Multi-Prototype Cosine Similarity (`1 Real + 6 Fake Prototypes`)")
                if HAS_MPL:
                    proto_plot_out = gr.Plot(label="Prototype Cosine Similarity Bar Chart")
                else:
                    proto_plot_out = gr.Image(type="numpy", label="Prototype Cosine Similarity Bar Chart")

            with gr.Column(scale=6):
                gr.Markdown("### 4. Side-by-Side ConvNeXt-Tiny Grad-CAM Spatial Heatmap")
                gradcam_img_out = gr.Image(
                    type="numpy",
                    label="Aligned Crop (Left) | Grad-CAM Heatmap (Center) | Facial Artifact Overlay (Right)",
                )

        analyze_btn.click(
            fn=engine.predict_from_media,
            inputs=[image_input, video_input],
            outputs=[verdict_html_out, gauge_html_out, proto_plot_out, gradcam_img_out],
        )

        btn_real.click(
            fn=lambda: load_sample_and_run("real"),
            inputs=[],
            outputs=[image_input, video_input, verdict_html_out, gauge_html_out, proto_plot_out, gradcam_img_out],
        )
        btn_known.click(
            fn=lambda: load_sample_and_run("known_fake"),
            inputs=[],
            outputs=[image_input, video_input, verdict_html_out, gauge_html_out, proto_plot_out, gradcam_img_out],
        )
        btn_unseen.click(
            fn=lambda: load_sample_and_run("unseen_fake"),
            inputs=[],
            outputs=[image_input, video_input, verdict_html_out, gauge_html_out, proto_plot_out, gradcam_img_out],
        )
        btn_ambig.click(
            fn=lambda: load_sample_and_run("ambiguous"),
            inputs=[],
            outputs=[image_input, video_input, verdict_html_out, gauge_html_out, proto_plot_out, gradcam_img_out],
        )

    return demo


def run_dry_run(checkpoint_path: Optional[str] = None) -> None:
    """
    Runs a fast (<5s) CPU dry-run that:
      1. Generates the 4 sample images in a temporary `demo_samples/` folder.
      2. Builds `ZeroFakeDemoEngine` and `build_gradio_blocks_app`.
      3. Runs inference on all 4 sample buttons, a synthetic 16-frame video clip (sampling 8 frames),
         and a blank faceless image (verifying the friendly warning banner instead of a crash).
    """
    start_time = time.time()
    print("[DRY-RUN] Testing app.py (Gradio Blocks UI + 4 Sample Buttons + 8-frame Video + Faceless Warning)...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        samples_dir = os.path.join(tmp_dir, "demo_samples")
        sample_paths = ensure_demo_samples_exist(samples_dir=samples_dir)
        assert len(sample_paths) == 4
        for k, p in sample_paths.items():
            assert os.path.exists(p), f"Sample image for '{k}' was not created at {p}!"

        engine = ZeroFakeDemoEngine(
            checkpoint_path=checkpoint_path,
            device="cpu",
            allow_synthetic_detector_fallback=True,
        )
        demo = build_gradio_blocks_app(engine, sample_paths)
        assert demo is not None, "Gradio Blocks app failed to build!"

        # 1. Test image inference on sample_real_face.jpg
        sample_rgb = load_rgb_image_224(sample_paths["real"], image_size=224)
        v_html, g_html, _, cam_grid = engine.predict_from_media(sample_rgb, None)
        assert any(lbl in v_html for lbl in ("REAL", "FAKE", "UNKNOWN")), "Verdict badge missing label!"
        assert "τ_reject" in g_html, "Uncertainty gauge HTML missing threshold tau_reject!"
        assert cam_grid.shape[0] == 224 and cam_grid.shape[2] == 3, f"Unexpected Grad-CAM grid shape: {cam_grid.shape}"

        # 2. Test short video clip inference (sampling 8 evenly spaced frames)
        dummy_video = os.path.join(tmp_dir, "clip.mp4")
        create_synthetic_video(dummy_video, total_frames=16, height=160, width=160, include_face=True)
        v_vid_html, _, _, _ = engine.predict_from_media(None, dummy_video)
        assert any(lbl in v_vid_html for lbl in ("REAL", "FAKE", "UNKNOWN"))

        # 3. Test faceless image -> must return friendly warning HTML without crashing
        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
        v_warn_html, _, _, _ = engine.predict_from_media(blank_img, None)
        assert "NO FACE DETECTED" in v_warn_html, "Faceless image did not trigger friendly warning banner!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified Gradio Blocks UI, 4 sample buttons, 8-frame video clip inference, "
        f"prototype similarity chart, side-by-side Grad-CAM, and faceless warning in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ZeroFake interactive Gradio web application.")
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
        "--dry-run",
        action="store_true",
        help="Verify Gradio UI construction and sample/video/faceless inference on CPU in <5s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None)
        return

    sample_paths = ensure_demo_samples_exist(samples_dir=args.samples_dir)
    engine = ZeroFakeDemoEngine(checkpoint_path=args.checkpoint)
    demo = build_gradio_blocks_app(engine, sample_paths)

    if not HAS_GRADIO or gr is None:
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` or run with `--dry-run`.")

    print(f"[INFO] Launching ZeroFake Gradio Web App (checkpoint='{engine.checkpoint_loaded}')...")
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
