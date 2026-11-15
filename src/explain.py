"""Explanations: image SHAP (Partition), Grad-CAM, tabular SHAP, and explanation checks."""
from __future__ import annotations

import cv2
import numpy as np
import shap
import torch

from .data import CLASSES, IMG_SIZE, URGENT_IDX, to_model_input


# ----------------------------------------------------------------------------- image SHAP
def urgent_fn(predictor):
    """Wrap a Predictor so it returns one output: p_urgent = p_akiec + p_bcc + p_mel."""
    def f(x):
        return predictor(np.asarray(x, dtype=np.float32))[:, URGENT_IDX].sum(1, keepdims=True)
    return f


def image_shap(fn, img01: np.ndarray, max_evals: int = 500, masker: str = "blur(64,64)",
               batch_size: int = 50, seed: int = 0) -> np.ndarray:
    """Partition SHAP for a single-output function on one (224, 224, 3) image in [0, 1].

    Returns a (224, 224) map summed over colour channels: positive = pushes the output up
    (toward "see a doctor" for urgent_fn), negative = pushes it down.
    """
    np.random.seed(seed)
    m = shap.maskers.Image(masker, img01.shape)
    explainer = shap.Explainer(fn, m, output_names=["urgent"])
    sv = explainer(img01[None], max_evals=max_evals, batch_size=batch_size, silent=True)
    return sv.values[0, ..., 0].sum(-1)


# ----------------------------------------------------------------------------- Grad-CAM
class _UrgentTarget:
    """Grad-CAM target: calibrated p_urgent computed from the logits."""

    def __init__(self, T: float):
        self.T = T

    def __call__(self, logits):
        return torch.softmax(logits / self.T, dim=-1)[..., URGENT_IDX].sum(-1)


def gradcam(model, img01: np.ndarray, T: float, device) -> np.ndarray:
    """Grad-CAM on EfficientNet's conv_head for p_urgent, as a (224, 224) map in [0, 1]."""
    from pytorch_grad_cam import GradCAM

    x = to_model_input(img01[None]).to(device)
    with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
        return cam(input_tensor=x, targets=[_UrgentTarget(T)])[0]


# ----------------------------------------------------------------------------- checks
def blurred(img01: np.ndarray, k: int = 63) -> np.ndarray:
    return cv2.GaussianBlur(img01, (k, k), 0)


def deletion_curve(fn, img01: np.ndarray, saliency: np.ndarray | None,
                   fracs=np.linspace(0, 0.5, 11), seed: int = 0) -> np.ndarray:
    """Replace the top-k% most salient pixels with a blurred copy and track fn's output.

    saliency=None deletes pixels in random order (the baseline a faithful map must beat).
    """
    n = img01.shape[0] * img01.shape[1]
    if saliency is None:
        order = np.random.default_rng(seed).permutation(n)
    else:
        order = np.argsort(-saliency.ravel(), kind="stable")
    base = blurred(img01)
    batch = []
    for f in fracs:
        mask = np.zeros(n, bool)
        mask[order[:int(f * n)]] = True
        mask = mask.reshape(img01.shape[:2])[..., None]
        batch.append(np.where(mask, base, img01))
    return fn(np.stack(batch))[:, 0]


def inside_mass(saliency: np.ndarray, lesion_mask: np.ndarray) -> float:
    """Share of positive saliency mass that falls inside the lesion mask."""
    pos = np.clip(saliency, 0, None)
    total = pos.sum()
    return float(pos[lesion_mask].sum() / total) if total > 0 else float("nan")


def map_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two saliency maps (downsampled to 28x28)."""
    from scipy.stats import spearmanr

    da = cv2.resize(a.astype(np.float32), (28, 28), interpolation=cv2.INTER_AREA).ravel()
    db = cv2.resize(b.astype(np.float32), (28, 28), interpolation=cv2.INTER_AREA).ravel()
    return float(spearmanr(da, db).statistic)


# ----------------------------------------------------------------------------- display
def overlay(img01: np.ndarray, sal: np.ndarray, signed: bool = True, alpha: float = 0.55) -> np.ndarray:
    """Blend a saliency map over the image. Signed maps: red = toward urgent, blue = away."""
    import matplotlib

    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    if signed:
        lim = np.abs(sal).max() + 1e-12
        rgba = matplotlib.colormaps["bwr"](0.5 + 0.5 * sal / lim)
        weight = np.abs(sal) / lim
    else:
        s = (sal - sal.min()) / (np.ptp(sal) + 1e-12)
        rgba = matplotlib.colormaps["jet"](s)
        weight = s
    a = (alpha * np.clip(weight * 1.6, 0, 1))[..., None]
    gray = np.repeat(img01.mean(-1, keepdims=True), 3, -1) * 0.35 + img01 * 0.65
    return np.clip(gray * (1 - a) + rgba[..., :3] * a, 0, 1)


def region_name(sal: np.ndarray) -> str:
    """Name the 3x3 grid cell that holds the most positive saliency ("upper left", ...)."""
    pos = np.clip(sal, 0, None)
    h, w = pos.shape
    cells = pos[: h // 3 * 3, : w // 3 * 3].reshape(3, h // 3, 3, w // 3).sum((1, 3))
    r, c = np.unravel_index(cells.argmax(), cells.shape)
    rows, cols = ["upper", "middle", "lower"], ["left", "centre", "right"]
    if (r, c) == (1, 1):
        return "centre"
    return f"{rows[r]} {cols[c]}".replace("middle ", "").replace(" centre", "")


# ----------------------------------------------------------------------------- tabular SHAP
def tabular_shap(xgb_model, X):
    """TreeExplainer on the fusion model; returns an Explanation of shape (n, features, 7)."""
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer(X)
    sv.output_names = CLASSES
    return sv


def feature_groups(columns) -> dict[str, list[str]]:
    """Group fusion features into the four things a patient would recognise."""
    cols = list(columns)
    return {
        "image (CNN)": [c for c in cols if c.startswith("cnn_p_")],
        "age": [c for c in cols if c == "age"],
        "sex": [c for c in cols if c.startswith("sex_")],
        "body site": [c for c in cols if c.startswith("site_")],
    }


def grouped_urgent_shap(prob_fn, x_row, background) -> shap.Explanation:
    """Exact Shapley values of p_urgent over 4 feature groups, in probability units.

    prob_fn maps a feature DataFrame to (n, 7) probabilities. A group that is "absent" takes
    its values from the background rows (interventional SHAP), so base_value is the average
    p_urgent over the background and base + sum(values) == p_urgent(x_row) exactly.
    """
    from itertools import combinations
    from math import factorial

    groups = feature_groups(background.columns)
    names = list(groups)
    k = len(names)

    def v(coalition):
        X = background.copy()
        for g in coalition:
            X[groups[g]] = x_row[groups[g]].to_numpy()
        return float(prob_fn(X)[:, URGENT_IDX].sum(1).mean())

    cache = {}
    for r in range(k + 1):
        for s in combinations(names, r):
            cache[frozenset(s)] = v(s)
    phi = []
    for g in names:
        rest = [n for n in names if n != g]
        total = 0.0
        for r in range(k):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for s in combinations(rest, r):
                total += w * (cache[frozenset(s) | {g}] - cache[frozenset(s)])
        phi.append(total)

    shown = [f"{x_row['age']:.0f}" if not np.isnan(x_row["age"]) else "unknown",
             next((c[4:] for c in groups["sex"] if x_row[c] == 1), "unknown"),
             next((c[5:].replace("_", " ") for c in groups["body site"] if x_row[c] == 1), "unknown")]
    p_img = x_row[groups["image (CNN)"]].to_numpy()[URGENT_IDX].sum()
    return shap.Explanation(values=np.array(phi), base_values=cache[frozenset()],
                            data=np.array([f"{p_img:.2f} urgent", *shown], dtype=object),
                            feature_names=names)
