"""Pick demo examples from the test set and precompute their SHAP / Grad-CAM maps.

    .venv/bin/python app/build_examples.py

One correctly classified test image per class, plus the dark-corner artifact image with the
most SHAP mass in its corners (from notebook 05's shortcut hunt). Images are copied into
app/examples/ (HAM10000, CC BY-NC-SA 4.0, Tschandl et al. 2018).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_metadata  # noqa: E402
from src.explain import gradcam, image_shap, urgent_fn  # noqa: E402
from src.model import softmax_np  # noqa: E402
from src.triage import TriagePipeline, img_key, prepare  # noqa: E402

MAX_EVALS = 2000
OUT = ROOT / "app/examples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = TriagePipeline(ROOT / "models")
    f_u = urgent_fn(pipe.predictor)
    out = np.load(ROOT / "models/outputs.npz", allow_pickle=True)
    df = load_metadata(ROOT / "data/ham").set_index("image_id")
    test = df.loc[out["test_ids"]].copy()
    probs = softmax_np(out["test_logits"], pipe.T)
    test["pred"] = np.array(CLASSES)[probs.argmax(1)]
    test["conf"] = probs.max(1)

    picks = []
    for c in ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"]:
        cand = test[(test.dx == c) & (test.pred == c) & test.age.notna() & (test.localization != "unknown")]
        # the median-confidence correct example, so the demo isn't cherry-picked to look perfect
        cand = cand.sort_values("conf")
        picks.append((cand.index[len(cand) // 2], f"{c} (correctly classified)"))
    hunt = ROOT / "results/shortcut_hunt.csv"
    if hunt.exists():
        h = pd.read_csv(hunt).set_index("image_id")
        h = h[h.dark_corners]
        if len(h):
            picks.append((h.corner_share.idxmax(), "dark-corner artifact"))

    cache, meta = {}, []
    for i, why in picks:
        f = f"{i}.jpg"
        shutil.copy(df.loc[i, "path"], OUT / f)
        # same path as an upload in the app: file -> uint8 array -> prepare() -> hash
        img01 = prepare(np.asarray(Image.open(OUT / f).convert("RGB")))
        key = img_key(img01)
        cache[f"{key}_shap"] = image_shap(f_u, img01, max_evals=MAX_EVALS)
        cache[f"{key}_cam"] = gradcam(pipe.model, img01, pipe.T, pipe.device)
        r = df.loc[i]
        meta.append({"file": f, "dx": r.dx, "why": why, "age": None if np.isnan(r.age) else int(r.age),
                     "sex": r.sex if r.sex in ("male", "female") else "unknown",
                     "site": r.localization})
        print(i, why)
    np.savez_compressed(OUT / "cache.npz", **cache)
    json.dump(meta, open(OUT / "examples.json", "w"), indent=2)


if __name__ == "__main__":
    main()
