"""Runtime pipeline: CNN -> (optional) fusion -> triage level, OOD flag, plain-English summary."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import IMG_SIZE, URGENT_IDX, to_model_input
from .model import Predictor, get_device, load_model, logits_and_embeddings, softmax_np

URGENT_MSG = "See a doctor soon"
BENIGN_MSG = "Lower concern"
BENIGN_ADVICE = "Lower concern. If it changes in size, shape, or color, see a doctor."
URGENT_ADVICE = ("Features of this spot resemble skin cancers in the training data. "
                 "Please book an appointment with a doctor or dermatologist.")
DISCLAIMER = ("Student research demo, not a medical device and not a diagnosis. "
              "Images are processed in memory and never stored.")


def prepare(img: np.ndarray) -> np.ndarray:
    """uint8 HWC image (any size) -> (224, 224, 3) float32 in [0, 1], resized full-frame."""
    from PIL import Image

    pil = Image.fromarray(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def img_key(img01: np.ndarray) -> str:
    """Stable hash of a prepared image, used to look up precomputed example explanations."""
    import hashlib

    return hashlib.md5((img01 * 255).round().astype(np.uint8).tobytes()).hexdigest()


@dataclass
class OODDetector:
    """Flags inputs unlike the training data via Mahalanobis distance of the CNN embedding
    (shared covariance, per-class means) and, separately, low max-softmax confidence."""
    means: np.ndarray            # (7, d)
    precision: np.ndarray        # (d, d)
    maha_threshold: float
    msp_threshold: float

    def distance(self, emb: np.ndarray) -> np.ndarray:
        diffs = emb[:, None, :] - self.means[None]           # n, 7, d
        d = np.einsum("nkd,de,nke->nk", diffs, self.precision, diffs)
        return d.min(1)

    def flags(self, emb: np.ndarray, probs: np.ndarray) -> dict:
        dist = self.distance(emb)
        return {"maha": dist, "unlike_training": dist > self.maha_threshold,
                "low_confidence": probs.max(1) < self.msp_threshold}


@dataclass
class TriageResult:
    probs: np.ndarray                     # final 7-class probabilities
    cnn_probs: np.ndarray                 # calibrated CNN probabilities
    p_urgent: float
    tau: float
    urgent: bool
    unlike_training: bool
    low_confidence: bool
    maha: float
    extra: dict = field(default_factory=dict)


class TriagePipeline:
    def __init__(self, models_dir: str | Path = "models", device=None):
        d = Path(models_dir)
        self.device = device or get_device()
        cfg = json.load(open(d / "config.json"))
        self.cfg = cfg
        self.T, self.tau = cfg["T"], cfg["tau"]
        self.use_fusion = cfg.get("use_fusion", False)
        self.model = load_model(d / "cnn.pt", self.device)
        self.predictor = Predictor(self.model, self.T, self.device)
        ood = np.load(d / "ood.npz")
        self.ood = OODDetector(ood["means"], ood["precision"], float(ood["maha_threshold"]),
                               float(ood["msp_threshold"]))
        self.xgb = None
        if (d / "xgb.json").exists():
            from .fusion import make_xgb

            self.xgb = make_xgb()
            self.xgb.load_model(d / "xgb.json")
            self.background = pd.read_csv(d / "fusion_background.csv")

    @torch.no_grad()
    def cnn(self, img01: np.ndarray):
        x = to_model_input(img01[None]).to(self.device)
        logits, emb = logits_and_embeddings(self.model, x)
        logits, emb = logits.float().cpu().numpy(), emb.float().cpu().numpy()
        return softmax_np(logits, self.T), emb

    def fusion_row(self, cnn_probs: np.ndarray, age, sex, site) -> pd.DataFrame:
        from .fusion import fusion_features

        meta = pd.DataFrame([{"age": age if age not in ("", None) else np.nan,
                              "sex": sex or "unknown", "localization": site or "unknown"}])
        return fusion_features(cnn_probs, meta)

    def run(self, img01: np.ndarray, age=None, sex=None, site=None) -> TriageResult:
        cnn_probs, emb = self.cnn(img01)
        probs = cnn_probs
        row = None
        if self.xgb is not None:
            row = self.fusion_row(cnn_probs, age, sex, site)
            if self.use_fusion:
                probs = self.xgb.predict_proba(row)
        p_u = float(probs[0, URGENT_IDX].sum())
        f = self.ood.flags(emb, cnn_probs)
        return TriageResult(probs=probs[0], cnn_probs=cnn_probs[0], p_urgent=p_u, tau=self.tau,
                            urgent=p_u >= self.tau, unlike_training=bool(f["unlike_training"][0]),
                            low_confidence=bool(f["low_confidence"][0]), maha=float(f["maha"][0]),
                            extra={"fusion_row": row})


def summary_text(res: TriageResult, region: str, tab_expl=None) -> str:
    """Deterministic plain-English summary built from the largest SHAP region and the
    largest metadata contribution (no LLM, so the same input always gives the same text)."""
    toward = "see a doctor" if res.urgent else "lower concern"
    parts = [f"The model's urgent score is {res.p_urgent:.2f}, "
             f"{'above' if res.urgent else 'below'} the decision threshold of {res.tau:.2f}, "
             f"so it leans toward \"{toward}\"."]
    parts.append(f"The image regions that pushed the score up most were in the {region} of the photo.")
    if tab_expl is not None:
        names, vals = list(tab_expl.feature_names), np.asarray(tab_expl.values)
        meta = [(n, v) for n, v in zip(names, vals) if n != "image (CNN)"]
        n, v = max(meta, key=lambda t: abs(t[1]))
        size = "a small amount" if abs(v) < 0.05 else "a noticeable amount"
        if abs(v) >= 0.01:
            parts.append(f"{n.capitalize()} {'raised' if v > 0 else 'lowered'} the score by {size} "
                         f"({v:+.2f}).")
        else:
            parts.append("Age, sex and body site barely changed the score.")
    if res.unlike_training:
        parts.append("⚠ This image looks unlike the dermoscopy photos the model was trained on, "
                     "so the result is unreliable.")
    elif res.low_confidence:
        parts.append("⚠ The model is not confident about this image.")
    return " ".join(parts)
