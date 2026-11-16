"""Export the deployed pipeline for the iOS app (Core ML model + plain binary/JSON assets).

    .venv/bin/python ios/export_assets.py

Writes ios/Veritas/Generated/ (git-ignored, like models/cnn.pt):
  LesionNet.mlmodel     FastViT-T12 @ 384 px: [0,1] NCHW image -> logits (7) + embedding (1024)
  pipeline.json         T, tau, OOD thresholds, classes, sites, SHAP settings, test metrics
  xgb.json              the fusion model's trees, flattened for a small Swift evaluator
  background.json       fusion background rows (for the grouped metadata SHAP)
  ood_*.bin             little-endian float arrays: class means, precision, train embedding bank
  examples/             the 8 demo images, already resized to 384 px, plus their SHAP / Grad-CAM maps,
                        and one non-skin photo (scikit-image's cat) to show the OOD guard
  fixtures.json         Python pipeline outputs on the examples, for the Swift parity tests
"""
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, IMG_SIZE, URGENT_IDX  # noqa: E402
from src.explain import default_masker, grouped_urgent_shap  # noqa: E402
from src.fusion import SITES  # noqa: E402
from src.triage import TriagePipeline, img_key, prepare  # noqa: E402

OUT = ROOT / "ios/Veritas/Generated"
EXAMPLES = ROOT / "app/examples"


class Wrapped(torch.nn.Module):
    """ImageNet normalisation baked in, so the app feeds plain [0, 1] RGB."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        feats = self.model.forward_head(self.model.forward_features((x - self.mean) / self.std), pre_logits=True)
        return self.model.get_classifier()(feats), feats


def export_coreml(pipe, imgs):
    model = Wrapped(pipe.model.cpu().eval()).eval()
    example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(model, example)
    # "neuralnetwork" (.mlmodel) rather than an ML Program: coremltools has no native blob
    # writer for Python 3.14, which the ML Program format needs. The Swift tests check the
    # converted model against the PyTorch outputs in fixtures.json.
    ml = ct.convert(traced, inputs=[ct.TensorType("image", shape=example.shape)], convert_to="neuralnetwork")
    spec = ml.get_spec()
    names = [o.name for o in spec.description.output]
    for old, new in zip(names, ["logits", "embedding"]):
        ct.utils.rename_feature(spec, old, new)
    ml = ct.models.MLModel(spec)
    ml.short_description = "Skin lesion triage CNN (FastViT-T12, 384 px). Research demo, not a medical device."
    ml.save(OUT / "LesionNet.mlmodel")
    pipe.model.to(pipe.device)


def flatten_xgb(path):
    """xgb.json -> per-tree arrays. Each tree adds its leaf value to the margin of one class."""
    m = json.load(open(path))["learner"]
    gb = m["gradient_booster"]["model"]
    trees = []
    for t, cls in zip(gb["trees"], gb["tree_info"]):
        trees.append({"class": cls, "left": t["left_children"], "right": t["right_children"],
                      "feature": t["split_indices"], "threshold": t["split_conditions"],
                      "defaultLeft": [bool(v) for v in t["default_left"]]})
    base = json.loads(m["learner_model_param"]["base_score"])
    return {"baseScore": base, "features": m["feature_names"], "trees": trees}


def xgb_predict(flat, X):
    """Reference evaluator mirroring the Swift one; checked against xgboost below."""
    out = np.zeros((len(X), len(CLASSES)))
    for i, x in enumerate(X):
        margin = np.array(flat["baseScore"], dtype=np.float64)  # per-class intercept, in margin units
        for t in flat["trees"]:
            n = 0
            while t["left"][n] != -1:
                v = x[t["feature"][n]]
                if np.isnan(v):
                    n = t["left"][n] if t["defaultLeft"][n] else t["right"][n]
                else:
                    n = t["left"][n] if v < t["threshold"][n] else t["right"][n]
            margin[t["class"]] += t["threshold"][n]
        e = np.exp(margin - margin.max())
        out[i] = e / e.sum()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    pipe = TriagePipeline(ROOT / "models")
    meta = json.load(open(EXAMPLES / "examples.json"))
    cache = np.load(EXAMPLES / "cache.npz")

    # --- examples (pre-resized so the app sees exactly the pixels the Python pipeline saw)
    imgs = []
    for ex in meta:
        img01 = prepare(np.asarray(Image.open(EXAMPLES / ex["file"]).convert("RGB")))
        imgs.append(img01)
        stem = Path(ex["file"]).stem
        Image.fromarray((img01 * 255).round().astype(np.uint8)).save(OUT / "examples" / f"{stem}.png")
        key = img_key(img01)
        cache[f"{key}_shap"].astype("<f4").tofile(OUT / "examples" / f"{stem}_shap.bin")
        cache[f"{key}_cam"].astype("<f4").tofile(OUT / "examples" / f"{stem}_cam.bin")
        ex["id"], ex["hasMaps"] = stem, True
    # a non-skin photo with no precomputed maps: the app should answer "cannot assess"
    from skimage.data import chelsea

    cat = prepare(chelsea())
    imgs.append(cat)
    Image.fromarray((cat * 255).round().astype(np.uint8)).save(OUT / "examples/cat.png")
    meta.append({"id": "cat", "file": "cat.png", "dx": None, "why": "not a skin photo", "age": None,
                 "sex": "unknown", "site": "unknown", "hasMaps": False})
    json.dump(meta, open(OUT / "examples/examples.json", "w"), indent=2)

    export_coreml(pipe, imgs)

    # --- fusion model + background
    flat = flatten_xgb(ROOT / "models/xgb.json")
    X = pipe.background.to_numpy(dtype=np.float64)
    ref, mine = pipe.xgb.predict_proba(pipe.background), xgb_predict(flat, X)
    print(f"flattened XGBoost vs xgboost: max |Δp| = {np.abs(ref - mine).max():.2e}")
    assert np.abs(ref - mine).max() < 1e-4
    json.dump(flat, open(OUT / "xgb.json", "w"))
    json.dump({"columns": list(pipe.background.columns),
               "rows": [[None if np.isnan(v) else float(v) for v in r] for r in X]},
              open(OUT / "background.json", "w"))

    # --- OOD guard
    ood = pipe.ood
    ood.means.astype("<f4").tofile(OUT / "ood_means.bin")
    ood.precision.astype("<f4").tofile(OUT / "ood_precision.bin")
    ood.bank.astype("<f2").tofile(OUT / "ood_bank.bin")

    ev = json.load(open(ROOT / "results/eval.json"))
    cnn = ev["test_results"][ev["deployed"]]
    cfg = pipe.cfg
    blur = int(default_masker(IMG_SIZE)[5:].split(",")[0])
    json.dump({
        "imageSize": IMG_SIZE, "classes": CLASSES, "urgent": [CLASSES[i] for i in URGENT_IDX],
        "sites": SITES, "temperature": cfg["T"], "tau": cfg["tau"], "useFusion": cfg.get("use_fusion", False),
        "mahaThreshold": cfg["maha_threshold"], "knnThreshold": cfg["knn_threshold"],
        "mspThreshold": cfg["msp_threshold"], "knnK": ood.k, "embeddingDim": int(ood.bank.shape[1]),
        "bankSize": int(ood.bank.shape[0]), "shapBlur": blur, "shapMaxEvals": 300,
        "model": "FastViT-T12 · 384 px · HAM10000 + ISIC 2019",
        "testMetrics": {k: round(cnn[k][0], 3) for k in
                        ("sensitivity_urgent", "specificity_urgent", "auroc_urgent", "balanced_accuracy")},
    }, open(OUT / "pipeline.json", "w"), indent=2)

    # --- fixtures: what the Python pipeline says about each example (with and without metadata)
    fixtures = []
    for ex, img01 in zip(meta, imgs):
        for age, sex, site in [(ex["age"], ex["sex"], ex["site"]), (None, None, None)]:
            res = pipe.run(img01, age, sex, site)
            emb = pipe.cnn(img01)[1]
            tab = grouped_urgent_shap(pipe.xgb.predict_proba, res.extra["fusion_row"].iloc[0], pipe.background)
            fixtures.append({"id": ex["id"], "age": age, "sex": sex, "site": site,
                             "cnnProbs": res.cnn_probs.tolist(), "pUrgent": res.p_urgent,
                             "urgent": res.urgent, "unlikeTraining": res.unlike_training,
                             "maha": float(ood.distance(emb)[0]), "knn": float(ood.knn(emb)[0]),
                             "fusionProbs": pipe.xgb.predict_proba(res.extra["fusion_row"])[0].tolist(),
                             "metaShapBase": float(tab.base_values), "metaShap": tab.values.tolist()})
    json.dump(fixtures, open(OUT / "fixtures.json", "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
