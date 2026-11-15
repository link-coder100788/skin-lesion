# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Baselines
# Every result needs something to beat.
# - **A:** always predict the majority class (`nv`).
# - **B:** logistic regression on metadata only (age, sex, body site). Shows how much the image adds.
#
# Both are fit on the train split. Validation scores are shown here; test predictions are
# saved for the single final evaluation in notebook 06.

# %%
import sys; sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data import CLASSES, load_metadata
from src.fusion import metadata_features
from src.metrics import all_metrics, is_urgent, p_urgent, threshold_for_sensitivity

df = load_metadata("../data/ham")
df["split"] = df["image_id"].map(pd.read_csv("../splits/split.csv").set_index("image_id")["split"])
X = metadata_features(df)
parts = {s: df.index[df.split == s] for s in ("train", "val", "test")}

# %% [markdown]
# ## Baseline A: majority class
# Its "urgent score" is constant, so no threshold can trade sensitivity for specificity.

# %%
prior = np.bincount(df.y[parts["train"]], minlength=7) / len(parts["train"])
probs_a = {s: np.tile(np.eye(7)[CLASSES.index("nv")], (len(idx), 1)) for s, idx in parts.items()}
print("accuracy on val:", (df.y[parts["val"]] == CLASSES.index("nv")).mean().round(3))
print(all_metrics(probs_a["val"], df.y[parts["val"]].to_numpy(), tau=0.5))

# %% [markdown]
# ## Baseline B: metadata-only logistic regression
# Class-balanced weights, so the model doesn't just learn "nv". The triage threshold is set on
# validation to keep sensitivity >= 90%, the same rule we use for the CNN.

# %%
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
clf.fit(X.loc[parts["train"]], df.y[parts["train"]])
probs_b = {s: clf.predict_proba(X.loc[idx]) for s, idx in parts.items()}
y_val = df.y[parts["val"]].to_numpy()
tau_b = threshold_for_sensitivity(is_urgent(y_val), p_urgent(probs_b["val"]))
print("tau_B =", round(tau_b, 4))
pd.Series(all_metrics(probs_b["val"], y_val, tau_b)).round(3)

# %%
np.savez("../results/baseline_preds.npz",
         **{f"{s}_{k}": v for k, d in [("A", probs_a), ("B", probs_b)] for s, v in d.items()},
         tau_B=tau_b, test_ids=df.image_id[parts["test"]].to_numpy())
