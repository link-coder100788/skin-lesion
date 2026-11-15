"""Metadata features and the XGBoost fusion model (CNN probabilities + age/sex/site)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data import CLASSES

SEXES = ["male", "female"]
SITES = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
         "lower extremity", "neck", "scalp", "trunk", "upper extremity"]
CNN_COLS = [f"cnn_p_{c}" for c in CLASSES]


def metadata_features(meta: pd.DataFrame) -> pd.DataFrame:
    """age (NaN kept, XGBoost handles it) + one-hot sex + one-hot body site.

    'unknown' sex/site map to all-zero one-hots, so a user can leave them blank in the app.
    """
    out = pd.DataFrame(index=meta.index)
    out["age"] = pd.to_numeric(meta["age"], errors="coerce")
    for s in SEXES:
        out[f"sex_{s}"] = (meta["sex"] == s).astype(int)
    for s in SITES:
        out[f"site_{s.replace(' ', '_')}"] = (meta["localization"] == s).astype(int)
    return out


def fusion_features(cnn_probs: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(cnn_probs, columns=CNN_COLS, index=meta.index)
    return pd.concat([X, metadata_features(meta)], axis=1)


def make_xgb(**kw) -> XGBClassifier:
    params = dict(objective="multi:softprob", max_depth=3, n_estimators=200, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4)
    params.update(kw)
    return XGBClassifier(**params)
