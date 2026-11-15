"""Evaluation metrics: urgent sensitivity/specificity, ECE, bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .data import URGENT_IDX


def p_urgent(probs: np.ndarray) -> np.ndarray:
    return probs[:, URGENT_IDX].sum(axis=1)


def is_urgent(y: np.ndarray) -> np.ndarray:
    return np.isin(y, URGENT_IDX).astype(int)


def sens_spec(y_urgent: np.ndarray, score: np.ndarray, tau: float):
    pred = score >= tau
    sens = (pred & (y_urgent == 1)).sum() / max((y_urgent == 1).sum(), 1)
    spec = (~pred & (y_urgent == 0)).sum() / max((y_urgent == 0).sum(), 1)
    return float(sens), float(spec)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append((pred[m] == y[m]).mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def all_metrics(probs: np.ndarray, y: np.ndarray, tau: float) -> dict:
    yu, score = is_urgent(y), p_urgent(probs)
    sens, spec = sens_spec(yu, score, tau)
    return {
        "sensitivity_urgent": sens,
        "specificity_urgent": spec,
        "auroc_urgent": float(roc_auc_score(yu, score)) if 0 < yu.sum() < len(yu) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, probs.argmax(1))),
        "ece": ece(probs, y),
    }


def bootstrap_ci(probs: np.ndarray, y: np.ndarray, tau: float, n: int = 1000, seed: int = 0) -> dict:
    """Point estimate plus 95% percentile bootstrap CI for every metric in all_metrics."""
    rng = np.random.default_rng(seed)
    point = all_metrics(probs, y, tau)
    samples = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for k, v in all_metrics(probs[idx], y[idx], tau).items():
            samples[k].append(v)
    return {k: (point[k], float(np.nanpercentile(samples[k], 2.5)),
                float(np.nanpercentile(samples[k], 97.5))) for k in point}


def threshold_for_sensitivity(y_urgent: np.ndarray, score: np.ndarray, target: float = 0.90) -> float:
    """Highest threshold tau whose sensitivity on (y_urgent, score) is still >= target."""
    pos = np.sort(score[y_urgent == 1])
    k = int(np.floor((1 - target) * len(pos)))  # number of positives we may miss
    return float(pos[k])
