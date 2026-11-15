"""Gradio demo: skin lesion triage with image + metadata SHAP explanations.

    .venv/bin/python app/app.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, URGENT  # noqa: E402
from src.explain import (gradcam, grouped_urgent_shap, image_shap, overlay,  # noqa: E402
                         region_name, urgent_fn)
from src.fusion import SITES  # noqa: E402
from src.triage import (BENIGN_ADVICE, DISCLAIMER, URGENT_ADVICE, TriagePipeline,  # noqa: E402
                        img_key, prepare, summary_text)

LIVE_MAX_EVALS = 300
EXAMPLES_DIR = ROOT / "app/examples"
NAMES = {"akiec": "actinic keratosis / intraepithelial carcinoma", "bcc": "basal cell carcinoma",
         "bkl": "benign keratosis", "df": "dermatofibroma", "mel": "melanoma", "nv": "nevus (mole)",
         "vasc": "vascular lesion"}

pipe = TriagePipeline(ROOT / "models")
f_urgent = urgent_fn(pipe.predictor)


def load_cache() -> dict:
    """Precomputed SHAP maps for the example images (made by app/build_examples.py)."""
    cache = {}
    if (EXAMPLES_DIR / "cache.npz").exists():
        d = np.load(EXAMPLES_DIR / "cache.npz")
        for k in d.files:
            cache[k] = d[k]
    return cache


CACHE = load_cache()
EXAMPLES = json.load(open(EXAMPLES_DIR / "examples.json")) if (EXAMPLES_DIR / "examples.json").exists() else []


def prob_chart(probs: np.ndarray):
    order = np.argsort(probs)
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#c0392b" if CLASSES[i] in URGENT else "#7f8c8d" for i in order]
    ax.barh([f"{CLASSES[i]} · {NAMES[CLASSES[i]]}" for i in order], probs[order], color=colors)
    for y, i in enumerate(order):
        ax.text(probs[i] + 0.01, y, f"{probs[i]:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_title("Class probabilities (red = urgent classes)", fontsize=10)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def waterfall_chart(expl):
    plt.close("all")
    shap.plots.waterfall(expl, show=False, max_display=5)
    fig = plt.gcf()
    fig.set_size_inches(5.5, 3)
    ax = fig.axes[0]
    ax.set_title("How each input moved the urgent score (probability units)", fontsize=10)
    fig.tight_layout()
    return fig


def badge_html(res) -> str:
    if res.urgent:
        head, color, advice = "🔴 SEE A DOCTOR SOON", "#c0392b", URGENT_ADVICE
    else:
        head, color, advice = "🟢 LOWER CONCERN", "#1e8449", BENIGN_ADVICE
    warn = ""
    if res.unlike_training:
        warn = ("<div style='margin-top:8px;padding:6px 10px;border-radius:6px;background:#f9e79f;"
                "color:#7d6608'>⚠ This image looks unlike the dermoscopy training data. "
                "Do not trust this result.</div>")
    elif res.low_confidence:
        warn = ("<div style='margin-top:8px;padding:6px 10px;border-radius:6px;background:#f9e79f;"
                "color:#7d6608'>⚠ Low confidence</div>")
    return (f"<div style='padding:14px;border-radius:10px;border:2px solid {color}'>"
            f"<div style='font-size:1.5em;font-weight:700;color:{color}'>{head}</div>"
            f"<div style='font-size:1.1em;margin-top:4px'>Urgent score: <b>{res.p_urgent:.2f}</b> "
            f"<span style='opacity:.7'>(decision threshold {res.tau:.2f})</span></div>"
            f"<div style='margin-top:6px'>{advice}</div>{warn}</div>")


def analyze(img, age, sex, site, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Please upload an image first.")
    img01 = prepare(img)
    res = pipe.run(img01, age, None if sex == "unknown" else sex, None if site == "unknown" else site)

    key = img_key(img01)
    if f"{key}_shap" in CACHE:
        sal, cam = CACHE[f"{key}_shap"], CACHE[f"{key}_cam"]
    else:
        progress(0.2, desc="Computing image SHAP (10-30 s on CPU)…")
        sal = image_shap(f_urgent, img01, max_evals=LIVE_MAX_EVALS)
        cam = gradcam(pipe.model, img01, pipe.T, pipe.device)

    tab = None
    if pipe.xgb is not None:
        tab = grouped_urgent_shap(pipe.xgb.predict_proba, res.extra["fusion_row"].iloc[0], pipe.background)

    shap_img = (overlay(img01, sal) * 255).astype(np.uint8)
    cam_img = (overlay(img01, cam, signed=False) * 255).astype(np.uint8)
    text = summary_text(res, region_name(sal), tab)
    if pipe.xgb is not None and not pipe.use_fusion:
        text += (" (The decision itself uses the image model alone. Age, sex and site are shown "
                 "for explanation, because adding them did not improve accuracy on held-out data.)")
    return (badge_html(res), prob_chart(res.probs), shap_img, cam_img,
            waterfall_chart(tab) if tab is not None else None, text)


with gr.Blocks(title="Skin lesion triage · explainable AI demo") as demo:
    gr.Markdown("# Skin lesion triage with explainable AI\n"
                f"**⚠ {DISCLAIMER}**")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Image(label="Dermoscopy image of a skin spot", type="numpy", height=300)
        with gr.Column(scale=1):
            age = gr.Number(label="Age", value=None, precision=0)
            sex = gr.Dropdown(["unknown", "male", "female"], value="unknown", label="Sex")
            site = gr.Dropdown(["unknown", *SITES], value="unknown", label="Body location")
            btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        badge = gr.HTML()
        probs = gr.Plot(label="Class probabilities")
    with gr.Row():
        with gr.Tabs():
            with gr.Tab("Image SHAP"):
                shap_out = gr.Image(label="Red = pushes toward 'see a doctor', blue = toward benign",
                                    height=300)
            with gr.Tab("Grad-CAM (comparison)"):
                cam_out = gr.Image(label="Grad-CAM: where the last conv layer was most active", height=300)
        water = gr.Plot(label="Metadata SHAP waterfall")
    summary = gr.Markdown()
    gr.Markdown(f"<small>{DISCLAIMER} Built on HAM10000 (Tschandl et al., 2018). The model was "
                "trained on dermoscopy images of mostly light skin; it is not validated for phone "
                "photos or darker skin tones.</small>")

    outputs = [badge, probs, shap_out, cam_out, water, summary]
    btn.click(analyze, [inp, age, sex, site], outputs)
    if EXAMPLES:
        gr.Examples([[str(EXAMPLES_DIR / e["file"]), e["age"], e["sex"], e["site"]] for e in EXAMPLES],
                    [inp, age, sex, site], label="Examples from the held-out test set "
                    "(their SHAP maps are precomputed)", examples_per_page=8)

if __name__ == "__main__":
    demo.launch()
