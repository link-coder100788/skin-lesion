"""Heuristic detectors for common dermoscopy artifacts (used for the shortcut hunt)."""
from __future__ import annotations

import cv2
import numpy as np


def dark_corner_score(img: np.ndarray, frac: float = 0.1) -> float:
    """Fraction of the four corner patches that are near-black (dermoscope vignette)."""
    h, w = img.shape[:2]
    k = int(min(h, w) * frac)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    return float(np.mean([c.mean() < 50 for c in corners]))


def hair_fraction(img: np.ndarray) -> float:
    """Fraction of pixels picked up by a black-hat filter (thin dark lines = hair)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return float((blackhat > 25).mean())


def load_mask(path, size: int | None = None) -> np.ndarray:
    """Load a HAM10000 segmentation PNG as a boolean size x size mask (default IMG_SIZE)."""
    from .data import IMG_SIZE

    size = size or IMG_SIZE
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127


def artifact_flags(img: np.ndarray) -> dict:
    # Ink marks and rulers were also tried, but colour thresholds could not separate ink
    # from blue-purple pigment or tinted vignettes reliably, so they are not auto-flagged.
    return {
        "dark_corners": dark_corner_score(img) >= 0.5,
        "hair": hair_fraction(img) > 0.03,
    }
