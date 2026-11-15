# Skin lesion triage with explainable AI: hackathon plan

> **One-line pitch:** Upload a photo of a skin spot. The app says whether to see a dermatologist soon,
> shows *which pixels* drove that call, and shows *how* age and body location changed the answer.
> It also tells you honestly where the model is likely to be wrong.

> ⚠️ This is a student research demo. It is **not a medical device and not a diagnosis**. Say so on every
> screen and slide.

---

## 1. Why this project can win

| Judging criterion | How this project scores |
|---|---|
| Real problem | Melanoma is curable when caught early. Wait times for a dermatologist are often months. |
| Technical depth | Transfer learning, class imbalance, calibration, and **two layers of SHAP explainability** (image + metadata). |
| Demo | Live upload. Prediction, heatmap, and a waterfall chart appear in seconds. |
| Insight | SHAP can expose **shortcut learning** (the model "looks" at rulers, ink marks, or hair). Finding and fixing that is a story in itself. |
| Ethics | A fairness audit across skin tones, plus an out-of-distribution warning. |

Most hackathon classifiers stop at "92% accuracy". The extra parts here are explaining the model, checking the explanations, and being clear about its limits.

---

## 2. Problem definition

**Input:** one image of a skin lesion, plus optional metadata (age, sex, body location).

**Outputs:**
1. **7-class probabilities** (HAM10000 classes, see below).
2. **A triage level** derived from those probabilities:
   - 🔴 **See a doctor soon**: melanoma (`mel`), basal cell carcinoma (`bcc`), actinic keratosis / intraepithelial carcinoma (`akiec`)
   - 🟢 **Likely benign, monitor it**: nevus (`nv`), benign keratosis (`bkl`), dermatofibroma (`df`), vascular lesion (`vasc`)
3. **Explanations:**
   - an image SHAP map that shows the regions that pushed toward or away from "urgent"
   - a metadata SHAP waterfall that shows how much the image score, age, sex, and location each contributed
4. **A confidence or out-of-distribution flag** ("this image looks unlike the training data").

**Main metric:** **sensitivity (recall) on the urgent class**. Missing a cancer costs far more than a false alarm.
Set the triage threshold so that sensitivity is ≥ 90%, then report the specificity you get at that point.

---

## 3. Data

### 3.1 Primary dataset: HAM10000
- 10,015 dermoscopic images, 600×450 px, with `metadata.csv` (`lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`).
- Download: Kaggle `kmader/skin-cancer-mnist-ham10000` or Harvard Dataverse. License is CC BY-NC 4.0, which is fine for a non-commercial hackathon. Cite the paper (Tschandl et al., 2018).

| Class | Count | Triage |
|---|---|---|
| nv | 6,705 | benign |
| mel | 1,113 | **urgent** |
| bkl | 1,099 | benign |
| bcc | 514 | **urgent** |
| akiec | 327 | **urgent** |
| vasc | 142 | benign |
| df | 115 | benign |

The data is **heavily imbalanced**: 67% of the images are `nv`. A model that always says "nevus" scores 67% accuracy and is useless. Use this fact in the pitch.

### 3.2 Fairness and shift datasets (pick one)
- **Fitzpatrick17k**: clinical photos labeled with Fitzpatrick skin type (I–VI). Use it to measure performance by skin tone.
- **DDI (Stanford Diverse Dermatology Images)**: biopsy-confirmed and balanced across skin tones. It needs an access request, so ask for it **before** the hackathon.

HAM10000 contains mostly lighter skin and no skin-type labels. The fairness audit (section 8) exists to measure that gap honestly.

### 3.3 ⚠️ The data-leakage trap (read this)
HAM10000 has **several images of the same lesion** (same `lesion_id`). A random split by image puts photos of the same mole in both train and test, which inflates the scores.
**Split by `lesion_id`** with `GroupShuffleSplit` or `StratifiedGroupKFold`.

```python
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
```

**Split:** 70% train / 15% validation / 15% test, all grouped by lesion. Lock the test set away until the end.

---

## 4. System architecture

```
                ┌─────────────────────────┐
  image ──────► │ CNN (EfficientNet-B0)   │──► 7-class probs ──┐
                │ fine-tuned on HAM10000  │                    │
                └──────────┬──────────────┘                    ▼
                           │                      ┌──────────────────────────┐
                           │     age, sex, site ─►│ Fusion model (XGBoost)   │──► final probs + triage
                           │                      └──────────┬───────────────┘
                           ▼                                 ▼
               ┌────────────────────────┐        ┌────────────────────────────┐
               │ Image SHAP (Partition) │        │ Tabular SHAP (TreeExplainer)│
               │ + Grad-CAM (compare)   │        │ waterfall per prediction    │
               └────────────────────────┘        └────────────────────────────┘
                           │                                 │
                           └──────────────┬──────────────────┘
                                          ▼
                                  Gradio demo app
```

Why two stages? SHAP on 1,280 raw CNN embedding features tells you nothing. SHAP on
`[cnn_p_mel, cnn_p_bcc, …, age, sex, localization]` tells a clear story: "The image looked 0.62 melanoma.
Age 70 added +0.08. A lesion on the back added +0.03."

---

## 5. Step-by-step build

### Step 0: Setup (before the hackathon)
- Google Colab Pro or Kaggle Notebooks (free T4/P100 GPU). Kaggle already hosts HAM10000, so you skip the download.
- `pip install timm torch torchvision shap xgboost scikit-learn pandas matplotlib gradio opencv-python`
- Create a GitHub repo with the structure in section 11.
- Ask for DDI access if you plan to use it.

### Step 1: Exploratory data analysis (1–2 h)
- Plot class counts, age histograms per class, and body location per class.
- Look at about 50 images per class. Note **artifacts**: rulers, ink marks, hair, dark vignette corners, gel bubbles. Write them down; you will look for them again in the SHAP maps.
- Check for missing `age` (a few rows) and `localization == "unknown"`.

### Step 2: Baseline (1 h)
Every result needs a baseline to compare against.
- **Baseline A:** always predict the majority class (`nv`). Report its balanced accuracy, which is about 14%.
- **Baseline B:** logistic regression on metadata only (age, sex, localization). This shows how much the image adds.
- **Baseline C (optional):** logistic regression on frozen ImageNet ResNet features.

### Step 3: Train the CNN (4–6 h)
- **Backbone:** `timm` `efficientnet_b0` (fast, ~5M params). Use `resnet50` or `efficientnet_b3` if time allows.
- **Input:** resize to 224×224 and use ImageNet normalization.
- **Augmentation** (lesions have no "up"): random flips (H+V), 90° rotations, `RandomResizedCrop(224, scale=(0.7,1.0))`, light `ColorJitter`. Do not over-jitter the color, because color carries diagnostic information.
- **Class imbalance**, pick one and say why in the slides:
  - weighted cross-entropy (`weight = 1 / sqrt(class_freq)`)
  - `WeightedRandomSampler`
  - focal loss (stretch)
- **Schedule:** freeze the backbone and train the head for 2 epochs. Then unfreeze everything and train 10–15 epochs with AdamW (lr 3e-4 head / 1e-4 backbone) and a cosine schedule. Use mixed precision.
- **Checkpoint** on best validation **balanced accuracy**, not plain accuracy.

```python
import timm, torch.nn as nn
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=7)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### Step 4: Calibrate and set the triage threshold (1 h)
- **Temperature scaling** on the validation set: learn one scalar `T` and use `softmax(logits / T)`. A 0.8 output should then be right about 80% of the time. Show a reliability diagram before and after.
- **Triage score:** `p_urgent = p_mel + p_bcc + p_akiec`.
- Choose the threshold `τ` on the validation set as the highest value that keeps sensitivity ≥ 0.90. Freeze it.

### Step 5: Fusion model with metadata (2 h)
- Features: the CNN's 7 calibrated probabilities + `age` + `sex` (one-hot) + `localization` (one-hot or native categorical).
- **Avoid leakage.** The fusion model must train on CNN predictions for images **the CNN did not train on**. Two options:
  - *Simple:* run the CNN on the validation split. Train XGBoost on that split with 5-fold cross-validation for tuning. Evaluate on test.
  - *Better:* get out-of-fold CNN predictions with 5-fold CNN training. Only do this if your GPU time allows.
- Use `XGBClassifier(objective="multi:softprob", max_depth=3, n_estimators=200, learning_rate=0.05)`. Keep it small, because the validation split has only ~1,500 rows.
- Report whether fusion beats the CNN alone. **A null result is fine, and you should say so.**

### Step 6: Explainability (see section 6) (4–5 h)

### Step 7: Evaluation and fairness (see sections 7–8) (2–3 h)

### Step 8: Demo app (see section 9) (3–4 h)

### Step 9: Slides and rehearsal (2 h)

---

## 6. Explainability: the core of the project

You get **three** explanation views. Build them in this order.

### 6.1 Image SHAP with the Partition explainer (primary)
SHAP gives each **region** of the image a contribution to the predicted class. It works by hiding (blurring) regions and measuring how the output changes, and Shapley values from game theory decide how to share the credit fairly. The Partition explainer splits the image into a hierarchy of regions, so it is fast enough for a live demo.

```python
import numpy as np, shap, torch

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def predict_fn(x_hwc: np.ndarray) -> np.ndarray:
    """x: (N, 224, 224, 3) float in [0,1]  ->  (N, 7) calibrated probabilities."""
    x = torch.tensor(x_hwc).permute(0, 3, 1, 2).float()
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        logits = model(x.to(device))
    return torch.softmax(logits / T, dim=1).cpu().numpy()

masker = shap.maskers.Image("blur(64,64)", (224, 224, 3))
explainer = shap.Explainer(predict_fn, masker, output_names=CLASSES)

sv = explainer(
    images[:4],                      # (4, 224, 224, 3)
    max_evals=500,                   # more = sharper but slower (~2–5 s/img on GPU)
    batch_size=50,
    outputs=shap.Explanation.argsort.flip[:2],   # top-2 predicted classes
)
shap.image_plot(sv)
```

**Triage-level SHAP.** Explain `p_urgent` directly by wrapping the model so that it returns one output:

```python
def predict_urgent(x):
    p = predict_fn(x)
    return (p[:, 0] + p[:, 1] + p[:, 4])[:, None]   # akiec + bcc + mel
```

This gives one map: **red = "pushes toward see a doctor", blue = "pushes toward benign"**. It is the easiest view for a judge to read.

**Tuning:**
- `blur(64,64)` is a reasonable "missing" value. Try `"inpaint_telea"` too, because a blurred lesion still looks a little like a lesion. Comparing maskers is a good mini-experiment for the slides.
- `max_evals` trades sharpness for speed. Use 300 in the live demo and 2,000 for slide figures.

### 6.2 Grad-CAM (comparison and sanity check)
Grad-CAM uses the gradients of the last conv layer. It is fast and smooth, but coarse.
Use the `pytorch-grad-cam` package with `target_layers=[model.conv_head]` for EfficientNet in timm.
Show SHAP and Grad-CAM **side by side**. When the two agree, you can trust the explanation more. When they disagree, discuss why.

### 6.3 Tabular SHAP on the fusion model
```python
explainer_tab = shap.TreeExplainer(xgb_model)
sv_tab = explainer_tab(X_test_df)               # shape: (n, features, 7 classes)

i, mel = 12, CLASSES.index("mel")
shap.plots.waterfall(sv_tab[i, :, mel])         # one patient: why melanoma?
shap.plots.beeswarm(sv_tab[:, :, mel])          # global: what drives melanoma calls?
```
Per-patient waterfall example: *"Base rate 0.11 → image says melanoma +0.41 → age 72 +0.07 → back +0.02 → sex −0.01 = 0.60."*
The global beeswarm shows whether the model learned real patterns from the dataset, such as the higher melanoma rate at older ages.

### 6.4 Check the explanations (this is what gives the project depth)
An explanation can look convincing and still be wrong. Run these checks and put the results on a slide:

1. **Shortcut hunt.** Collect test images that contain rulers, ink marks, or dark corners (from your EDA notes). Measure how much SHAP mass falls **outside the lesion**. You can get a rough lesion mask from Otsu thresholding, or from the **HAM10000 segmentation masks** (ISIC 2018 Task 1).
   - Metric: `% of positive |SHAP| inside lesion mask`. Report it per class.
   - If the model leans on artifacts, **fix it**: crop around the lesion, inpaint hair (DullRazor-style morphology), or add artifact augmentation. Then retrain and show the before/after maps. **This is your best slide.**
2. **Deletion test (faithfulness).** Remove the top-k% SHAP pixels and plot how fast `p_predicted` drops. Do the same with random pixels. A faithful explanation makes the curve drop faster than random. Report the area under the deletion curve for SHAP and for Grad-CAM.
3. **Model-randomization sanity check** (Adebayo et al., 2018). Randomize the weights of the classifier head and recompute the maps. If the maps barely change, the explainer is showing image edges, not the model's reasoning.
4. **Stability.** Add tiny noise or flip the image. The maps should move with the image and not change their content.

---

## 7. Evaluation (on the locked test set, once)

Report all of these. Do not show accuracy alone.

| Metric | Why |
|---|---|
| **Sensitivity on urgent @ τ** | The main metric. How many cancers do we catch? |
| Specificity on urgent @ τ | How many unneeded doctor visits? |
| AUROC (urgent vs. benign) | Quality independent of the threshold |
| Balanced accuracy (7-class) | Fair across the imbalanced classes |
| Per-class recall + confusion matrix | Shows which classes the model confuses (typically mel ↔ nv, bkl ↔ mel) |
| Expected calibration error (ECE) before/after temperature scaling | Shows whether the confidence numbers can be trusted |
| Comparison table: Baseline A / B / CNN / CNN+fusion | Shows what each part adds |

Add **95% confidence intervals** with a bootstrap over the test set (1,000 resamples). Judges notice it, and it takes about 10 lines of code.

**Reference point:** published HAM10000 / ISIC 2018 models reach balanced accuracy of roughly 0.75–0.88. A weekend EfficientNet-B0 at about 0.70–0.80 is a good result. Say that honestly.

---

## 8. Fairness and safety

1. **Skin-tone audit.** Run the trained model (no retraining) on Fitzpatrick17k or DDI images of the matching classes. Report urgent-class sensitivity for skin types I–II, III–IV, and V–VI.
   Expect a drop, partly from skin tone and partly because clinical photos differ from dermoscopy images.
   **Show it, explain both causes, and propose fixes** (more diverse data, domain adaptation). That honesty earns points.
2. **Out-of-distribution warning.** A phone photo is **not** a dermoscopy image. Add a simple detector:
   - maximum softmax probability below a threshold → "low confidence", **or**
   - Mahalanobis distance of the CNN embedding from the training embeddings → "this image looks unlike the training data".
   Test it on non-lesion images (a cat, a hand, a blank wall). They must be flagged.
3. **Asymmetric output.** The app never says "you are fine". The benign message reads: *"Lower concern. If it changes in size, shape, or color, see a doctor."*
4. **Privacy.** The demo processes images in memory and stores nothing. Say so on the screen.

---

## 9. Demo app (Gradio, deployed to Hugging Face Spaces)

**Layout:**
```
┌───────────────────────────────────────────────────────────────┐
│  [Upload image]   Age [__]  Sex [▼]  Body location [▼]         │
│  [Analyze]                                                     │
├───────────────────────────┬───────────────────────────────────┤
│  🔴 SEE A DOCTOR SOON      │  Class probabilities (bar chart)   │
│  Urgent score: 0.71        │  mel ████████ 0.52                 │
│  ⚠ Low-confidence badge    │  bcc ██ 0.12   nv ███ 0.20 …       │
├───────────────────────────┼───────────────────────────────────┤
│  Image SHAP overlay        │  Metadata SHAP waterfall           │
│  (toggle: SHAP / Grad-CAM) │  image +0.41, age +0.07, …         │
├───────────────────────────┴───────────────────────────────────┤
│  "What the model looked at": plain-English summary             │
│  ⚠ Research demo, not a diagnosis. Images are not stored.      │
└───────────────────────────────────────────────────────────────┘
```

- Preload 6–8 **example images** from the test set (one per class, plus one image with a ruler artifact). Venue Wi-Fi and lighting will fail you during a live demo.
- Cache the SHAP results for the example images so they appear at once. Compute SHAP live only for uploads.
- A CPU-only Space runs EfficientNet-B0 inference in under 1 s. Live SHAP at `max_evals=300` takes 10–30 s on CPU, so show a spinner. The free GPU tier, if available, makes it 2–5 s.
- Plain-English summary template: *"The model's 'urgent' score came mostly from the dark, uneven border on the left side of the spot. Age added a small amount."* Build it from the SHAP outputs (largest region + largest metadata feature). Do not use an LLM, so the output stays deterministic.

---

## 10. Timeline

### 24-hour version
| Hours | Task | Owner |
|---|---|---|
| 0–1 | Setup, data loaded, lesion-grouped split | ML |
| 1–3 | EDA + baselines A/B | Data |
| 1–8 | CNN training (runs in the background) | ML |
| 3–6 | Gradio skeleton with a dummy model | App |
| 8–9 | Calibration + threshold | ML |
| 9–11 | Fusion XGBoost + TreeExplainer | Data |
| 9–13 | Image SHAP + Grad-CAM | ML |
| 13–16 | Explanation checks: shortcut hunt + deletion test | ML + Data |
| 13–17 | Wire the real model and explanations into the app, deploy | App |
| 16–19 | Test evaluation, bootstrap CIs, fairness audit | Data |
| 19–22 | Slides + figures | All |
| 22–24 | Rehearse the demo twice, buffer time | All |

### 48-hour version: add these
- Shortcut fix (crop, hair removal, or artifact augmentation) + retrain + before/after SHAP
- Out-of-fold CNN predictions for the fusion model
- Mahalanobis OOD detector
- Model-randomization sanity check
- A bigger backbone or test-time augmentation (average over flips)

### Team roles (3–4 people)
- **ML lead:** CNN, calibration, image SHAP
- **Data / eval lead:** EDA, baselines, fusion + tabular SHAP, metrics, fairness
- **App lead:** Gradio, deployment, UX copy
- **Story lead** (or shared): slides, ethics, rehearsal, citations

---

## 11. Repo structure

```
skin-lesion-triage/
├── PLAN.md
├── README.md                  # how to run, results table, disclaimer, citations
├── requirements.txt
├── data/                      # git-ignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_train_cnn.ipynb
│   ├── 04_calibrate_fusion.ipynb
│   ├── 05_explain_shap.ipynb
│   └── 06_eval_fairness.ipynb
├── src/
│   ├── data.py                # dataset, transforms, grouped split
│   ├── model.py               # timm model, temperature scaling
│   ├── fusion.py              # XGBoost fusion
│   ├── explain.py             # SHAP image/tabular, Grad-CAM, deletion test
│   └── triage.py              # threshold logic, OOD flag, summary text
├── app/
│   └── app.py                 # Gradio
├── models/                    # cnn.pt, T.json, xgb.json, tau.json (git-ignored or HF Hub)
└── figures/                   # slide-ready PNGs
```

Fix random seeds, and save `tau`, `T`, and the split indices to files. Anyone must be able to reproduce your numbers.

---

## 12. Pitch outline (5 minutes)

1. **Hook (30 s):** "Melanoma caught early: ~99% five-year survival. Caught late: far lower. The wait for a dermatologist can be months."
2. **What we built (30 s):** a triage tool that explains itself.
3. **Live demo (90 s):** a cached example → triage → SHAP map → waterfall. Then the ruler-artifact example.
4. **How it works (45 s):** one architecture slide. Mention the lesion-level split ("we avoided a common leakage bug").
5. **Results (45 s):** baseline vs. model table, sensitivity @ τ with CI, confusion matrix.
6. **What SHAP taught us (45 s):** the shortcut finding + fix, before/after maps, deletion curve. **The key slide.**
7. **Limits and ethics (30 s):** the skin-tone gap, dermoscopy vs. phone, "not a diagnosis".
8. **Next steps (15 s):** diverse data, clinician feedback, prospective testing.

Expect these judge questions: *Why not just report accuracy? How do you know the heatmap is right? Would this work on my phone? On darker skin? What happens on a false negative?* Every one of them has an answer in this plan.

---

## 13. Risks and fallbacks

| Risk | Fallback |
|---|---|
| GPU time runs out | Train on 128×128 images, or freeze the backbone and train only the head. It still works for the demo. |
| Live SHAP is too slow | Lower `max_evals`, cache the examples, show Grad-CAM live and SHAP for the examples. |
| Fusion does not beat the CNN | Report the null result. Keep tabular SHAP: it still explains how the metadata shifts each prediction. |
| No fairness dataset access | Use Fitzpatrick17k (public), or state the limit clearly and show HAM10000's demographic breakdown. |
| Demo Wi-Fi fails | Run the app locally plus a recorded backup video. |
| SHAP maps look noisy | Raise `max_evals`, try `inpaint_telea`, show the triage-level map instead of 7 per-class maps. |

---

## 14. Stretch ideas
- **Concept-level explanations:** score each image for ABCD features (asymmetry, border irregularity, color variety, diameter) with classic computer vision. Feed those to the fusion model, so SHAP speaks a dermatologist's language: "border irregularity +0.15".
- **Counterfactuals:** "What would have to change for this to be classed as benign?"
- **Uncertainty:** MC-dropout or a 3-model ensemble. Show a confidence band on the urgent score.
- **On-device:** export to ONNX / Core ML and run it on a phone with no image upload, for privacy.

---

## 15. Key references
- Tschandl, Rosendahl, Kittler. *The HAM10000 dataset.* Scientific Data, 2018.
- Lundberg & Lee. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS, 2017.
- Selvaraju et al. *Grad-CAM.* ICCV, 2017.
- Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS, 2018.
- Guo et al. *On Calibration of Modern Neural Networks* (temperature scaling). ICML, 2017.
- Groh et al. *Fitzpatrick17k.* CVPR Workshops, 2021.
- Daneshjou et al. *Disparities in dermatology AI performance on a diverse, curated clinical image set (DDI).* Science Advances, 2022.
- Winkler et al. *Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning CNN.* JAMA Dermatology, 2019. (This is the evidence for the shortcut-learning slide.)
