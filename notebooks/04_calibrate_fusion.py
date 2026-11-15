# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 · Calibration, triage threshold, fusion, and out-of-distribution detector
# Everything here is fit on the **validation** split (images the CNN never trained on).
# The test split is not touched until notebook 06.
#
# 1. Temperature scaling: one scalar `T`, `softmax(logits / T)`.
# 2. Triage threshold `τ`: highest value with urgent sensitivity ≥ 0.90 on validation.
# 3. Fusion: XGBoost on [7 calibrated CNN probs, age, sex, site], evaluated with 5-fold
#    lesion-grouped CV *within* validation and compared against the CNN alone.
# 4. OOD detector: Mahalanobis distance of the CNN embedding + low max-softmax confidence.

# %%
import sys; sys.path.insert(0, "..")
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from src.data import CLASSES, FITZ_TO_HAM, load_metadata, IMG_SIZE
from src.fusion import fusion_features, make_xgb
from src.metrics import (all_metrics, ece, is_urgent, p_urgent, reliability_curve,
                         sens_spec, threshold_for_sensitivity)
from src.model import fit_temperature, get_device, load_model, logits_and_embeddings, softmax_np
from src.data import to_model_input

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
out = np.load("../models/outputs.npz", allow_pickle=True)
df = load_metadata("../data/ham").set_index("image_id")
val = df.loc[out["val_ids"]]
y_val = val["y"].to_numpy()
print({k: out[k].shape for k in out.files if k.endswith("logits")})
print(json.load(open("../models/train_log.json"))["best_val_bacc"])

# %% [markdown]
# ## 1 · Temperature scaling

# %%
T = fit_temperature(out["val_logits"], y_val)
p_raw, p_cal = softmax_np(out["val_logits"]), softmax_np(out["val_logits"], T)
print(f"T = {T:.3f}  |  val ECE before {ece(p_raw, y_val):.4f} -> after {ece(p_cal, y_val):.4f}")

fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
for p, lab in [(p_raw, f"before (ECE {ece(p_raw, y_val):.3f})"), (p_cal, f"after T={T:.2f} (ECE {ece(p_cal, y_val):.3f})")]:
    x, y, _ = reliability_curve(p, y_val)
    ax.plot(x, y, "o-", label=lab)
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(fontsize=8)
ax.set_title("Reliability diagram (validation)")
fig.savefig("../figures/reliability.png")

# %% [markdown]
# ## 2 · Triage threshold τ (sensitivity ≥ 0.90 on validation)

# %%
yu_val, s_val = is_urgent(y_val), p_urgent(p_cal)
tau = threshold_for_sensitivity(yu_val, s_val, 0.90)
print(f"tau = {tau:.4f}", "val sens/spec =", np.round(sens_spec(yu_val, s_val, tau), 3))
pd.Series(all_metrics(p_cal, y_val, tau)).round(3)

# %% [markdown]
# ## 3 · Fusion with metadata
# The validation split is the only data with CNN predictions on unseen images, so fusion is
# tuned/evaluated with 5-fold **lesion-grouped** CV inside it. Out-of-fold predictions give a
# fair CNN-alone vs. CNN+metadata comparison. A null result is a valid result.

# %%
X_val = fusion_features(p_cal, val.reset_index())
oof = np.zeros_like(p_cal)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in sgkf.split(X_val, y_val, groups=val["lesion_id"]):
    m = make_xgb().fit(X_val.iloc[tr], y_val[tr])
    oof[te] = m.predict_proba(X_val.iloc[te])

tau_fusion = threshold_for_sensitivity(yu_val, p_urgent(oof), 0.90)
cmp = pd.DataFrame({
    "CNN alone": all_metrics(p_cal, y_val, tau),
    "CNN + metadata (OOF)": all_metrics(oof, y_val, tau_fusion),
}).round(3)
cmp

# %% [markdown]
# **Decision rule, fixed before looking at test:** use fusion for the triage decision only if
# it improves urgent AUROC by ≥ 0.005 *and* specificity at 90% sensitivity, without lowering
# balanced accuracy. Otherwise the CNN makes the decision and fusion is kept to *explain*
# how metadata shifts the score.

# %%
d = cmp["CNN + metadata (OOF)"] - cmp["CNN alone"]
use_fusion = bool(d["auroc_urgent"] >= 0.005 and d["specificity_urgent"] > 0 and d["balanced_accuracy"] >= 0)
print("use_fusion =", use_fusion, d.round(3).to_dict())

xgb = make_xgb().fit(X_val, y_val)
xgb.save_model("../models/xgb.json")
X_val.sample(100, random_state=0).to_csv("../models/fusion_background.csv", index=False)
np.save("../results/val_oof_fusion.npy", oof)

# %% [markdown]
# ## 4 · Out-of-distribution detector
# A phone photo or a cat is not a dermoscopy image. Mahalanobis distance of the 1280-d CNN
# embedding to the nearest class mean (shared, shrunk covariance fit on **train** embeddings).
# Threshold = 99th percentile of validation distances, so ~1% of real dermoscopy images get
# flagged. We also record a "low confidence" threshold on max softmax probability (10th
# percentile on validation).

# %%
from sklearn.covariance import LedoitWolf

emb_tr = out["train_emb"].astype(np.float64)
y_tr = df.loc[out["train_ids"], "y"].to_numpy()
means = np.stack([emb_tr[y_tr == k].mean(0) for k in range(7)])
lw = LedoitWolf().fit(emb_tr - means[y_tr])
precision = lw.precision_

from src.triage import OODDetector
det = OODDetector(means, precision, np.inf, 0.0)
d_val = det.distance(out["val_emb"].astype(np.float64))
maha_thr = float(np.percentile(d_val, 99))
msp_thr = float(np.percentile(p_cal.max(1), 10))
print(f"maha threshold {maha_thr:.1f}, msp threshold {msp_thr:.3f}")
np.savez("../models/ood.npz", means=means, precision=precision, maha_threshold=maha_thr,
         msp_threshold=msp_thr)

# %% [markdown]
# ### Test the detector on images that are clearly not dermoscopy
# Stock photos bundled with scikit-image (a cat, a person, coffee, a rocket, bricks…), plus
# a blank wall, pure noise, and a skin-coloured flat image. They must all be flagged.

# %%
from skimage import data as skd

def to01(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = Image.fromarray(a[..., :3].astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(a, dtype=np.float32) / 255

rng = np.random.default_rng(0)
ood_imgs = {
    "cat": skd.chelsea(), "astronaut": skd.astronaut(), "coffee": skd.coffee(),
    "rocket": skd.rocket(), "bricks": skd.brick(), "camera man": skd.camera(),
    "grass": skd.grass(), "hubble": skd.hubble_deep_field(),
    "blank wall": np.full((300, 300, 3), (225, 220, 210), np.uint8),
    "noise": rng.integers(0, 255, (300, 300, 3), dtype=np.uint8),
    "flat skin tone": np.full((300, 300, 3), (224, 172, 150), np.uint8),
    "black": np.zeros((300, 300, 3), np.uint8),
}
device = get_device()
model = load_model("../models/cnn.pt", device)
X = to_model_input(np.stack([to01(v) for v in ood_imgs.values()])).to(device)
lg, em = logits_and_embeddings(model, X)
pr = softmax_np(lg.cpu().numpy(), T)
det = OODDetector(means, precision, maha_thr, msp_thr)
f = det.flags(em.cpu().numpy().astype(np.float64), pr)
ood_table = pd.DataFrame({"maha": f["maha"].round(0), "flag_unlike_training": f["unlike_training"],
                          "max_softmax": pr.max(1).round(2), "flag_low_conf": f["low_confidence"],
                          "would_triage": np.where(p_urgent(pr) >= tau, "urgent", "benign")},
                         index=list(ood_imgs))
ood_table

# %%
fig, axes = plt.subplots(2, 6, figsize=(13, 5))
for ax, (name, img), (_, r) in zip(axes.flat, ood_imgs.items(), ood_table.iterrows()):
    ax.imshow(to01(img)); ax.axis("off")
    ax.set_title(f"{name}\n{'FLAGGED' if r.flag_unlike_training else 'not flagged'} (d={r.maha:.0f})",
                 fontsize=8, color="#1e8449" if r.flag_unlike_training else "#c0392b")
fig.suptitle(f"OOD check: Mahalanobis threshold {maha_thr:.0f} (99th pct of validation)", fontsize=10)
fig.savefig("../figures/ood_examples.png")

# %%
fig, ax = plt.subplots(figsize=(6, 3))
bins = np.linspace(0, max(d_val.max(), f["maha"].max()) * 1.05, 60)
ax.hist(d_val, bins=bins, alpha=.7, label="HAM10000 validation", density=True)
fz = det.distance(out["fitz_emb"].astype(np.float64)) if "fitz_emb" in out.files else None
if fz is not None:
    ax.hist(fz, bins=bins, alpha=.6, label="Fitzpatrick17k (clinical photos)", density=True)
for v in f["maha"]:
    ax.axvline(v, color="k", lw=.6, alpha=.6)
ax.axvline(maha_thr, color="r", ls="--", label="threshold")
ax.set_xlabel("Mahalanobis distance (black ticks = non-skin images)"); ax.legend(fontsize=8)
fig.savefig("../figures/ood_distances.png")
if fz is not None:
    print(f"Fitzpatrick17k images flagged as unlike training: {(fz > maha_thr).mean():.1%}")

# %%
cfg = {"T": T, "tau": tau, "tau_fusion": tau_fusion, "use_fusion": use_fusion,
       "maha_threshold": maha_thr, "msp_threshold": msp_thr}
if use_fusion:
    cfg["tau"] = tau_fusion  # the pipeline thresholds whichever probabilities it uses
cfg["tau_cnn"] = tau
json.dump(cfg, open("../models/config.json", "w"), indent=2)
json.dump({"val_comparison": cmp.to_dict(), "ood_nonskin": ood_table.reset_index().to_dict("records"),
           "fitz_flagged_frac": None if fz is None else float((fz > maha_thr).mean()),
           "ece_val_before": ece(p_raw, y_val), "ece_val_after": ece(p_cal, y_val)},
          open("../results/calibration_fusion.json", "w"), indent=2, default=str)
cfg
