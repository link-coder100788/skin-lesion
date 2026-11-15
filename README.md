# Skin lesion triage with explainable AI

> ⚠️ **Student research demo. Not a medical device and not a diagnosis.**

Upload a dermoscopy photo of a skin spot. The app says whether to see a dermatologist soon,
shows which regions of the image drove that call (image SHAP, with Grad-CAM alongside),
shows how age, sex and body site moved the score (a metadata SHAP waterfall), and refuses to
give a verdict on images that look unlike its training data.

See [PLAN.md](PLAN.md) for the full plan and [slides/pitch.pptx](slides/pitch.pptx) for the pitch deck.

## Results (locked HAM10000 test split, 1,503 images, lesion-grouped)

95% CIs from 1,000 bootstrap resamples. `T`, `τ` and the fusion decision were all fixed on
validation before the test set was scored.

| Model | Urgent sensitivity @ τ | Specificity | AUROC (urgent) | Balanced acc. (7-class) |
|---|---|---|---|---|
| A · always "nevus" | 0.00 | 1.00 | 0.50 | 0.14 |
| B · metadata-only logistic regression | 0.92 [0.89, 0.95] | 0.37 [0.34, 0.40] | 0.77 | 0.38 |
| **CNN (EfficientNet-B0, calibrated), deployed** | **0.93 [0.90, 0.96]** | **0.72 [0.69, 0.74]** | **0.90 [0.89, 0.92]** | **0.66 [0.61, 0.72]** |
| CNN + metadata (XGBoost fusion) | 0.92 [0.89, 0.95] | 0.70 [0.68, 0.73] | 0.90 | 0.67 |

- "Urgent" = melanoma, basal cell carcinoma, actinic keratosis/intraepithelial carcinoma.
  τ = 0.148 is the highest threshold that keeps validation sensitivity ≥ 0.90.
- Temperature scaling (T = 1.47) lowers the test calibration error (ECE) from 0.089 to 0.025.
- Per-class recall: akiec 0.43, bcc 0.74, bkl 0.69, df 0.53, mel 0.59, nv 0.84, vasc 0.81.
  Published HAM10000 models reach a balanced accuracy of about 0.75–0.88, and this weekend
  EfficientNet-B0 sits below that range.
- **Fusion is a null result.** Adding age, sex and body site did not improve the triage on
  validation (lesion-grouped CV), so the CNN makes the decision. Fusion is kept to *explain*
  how the metadata shifts the score.

### What checking the explanations showed (notebook 05)
- **Shortcut hunt.** Dark vignette corners appear in 10% of melanomas but only 2% of nevi.
  SHAP puts a median of 1.6% of its positive mass in the corners, and the corners cover 3.9%
  of the image. **We found no evidence that the model uses them.** SHAP mass is only modestly
  concentrated in the lesion (melanoma: 40% inside the mask vs. 34% lesion area).
- **Faithful.** Deleting the top-SHAP pixels lowers the urgent score faster than deleting
  random pixels (deletion AUC 0.31 vs. 0.47). Grad-CAM does even better (0.21).
- **Fragile.** A map is deterministic for a fixed input, but a mirror flip (rank corr. 0.14),
  faint noise (0.05), or a different masker (blur vs. inpaint, ≈ 0) changes it a lot. SHAP
  and Grad-CAM barely agree (median 0.01). Read the maps as coarse hints, not ground truth.
- The metadata matters little. The mean |contribution| to p_urgent is 0.22 for the image,
  0.03 for age, 0.02 for body site and 0.003 for sex.

### Fairness and safety (notebook 06, 04)
| Data | Urgent sensitivity @ τ | n urgent |
|---|---|---|
| HAM10000 test (dermoscopy) | 0.93 [0.90, 0.96] | 295 |
| Fitzpatrick17k types I–II (clinical photos) | 0.60 [0.56, 0.63] | 783 |
| Fitzpatrick17k types III–IV | 0.55 [0.50, 0.59] | 422 |
| Fitzpatrick17k types V–VI | 0.59 [0.49, 0.70] | 92 |

On clinical photos, sensitivity drops by about a third for *every* skin type. That points to
photo type (clinical vs. dermoscopy) as the dominant shift. HAM10000 has no skin-type labels
and very little dark skin, so a skin-tone gap cannot be ruled out either. The fix is diverse
*dermoscopic* data.

**Out-of-distribution guard.** An image is flagged if its kNN cosine distance to the training
embeddings, or its Mahalanobis distance, exceeds the validation 99th percentile (2.1% of real
validation images are flagged). It flagged 12/12 non-skin development images and 9/10 unseen
ones; it missed a retina photo. It also flags about 25% of Fitzpatrick17k clinical photos. A
flagged image gets "cannot assess", never a verdict. The app never says "you are fine".

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install libomp            # macOS: OpenMP runtime for xgboost
```

**Data** (Kaggle API; put `KAGGLE_TOKEN=...` in `.env`):
```bash
set -a; . ./.env; set +a; export KAGGLE_API_TOKEN="$KAGGLE_TOKEN"
.venv/bin/kaggle datasets download kmader/skin-cancer-mnist-ham10000 -p data/ham --unzip
.venv/bin/kaggle datasets download tschandl/ham10000-lesion-segmentations -p data/masks --unzip
.venv/bin/kaggle datasets download -f fitzpatrick17k.csv asosenge/fitzpatrick17k -p data/fitz
```

**Pipeline** (heavy steps run as private Kaggle GPU kernels):
```bash
notebooks/run.sh 01 02                                          # EDA + baselines (local, CPU)
.venv/bin/python kaggle/build_kernel.py train --user <you>      # bundle src/ into a kernel
.venv/bin/kaggle kernels push -p kaggle/build/train             # train the CNN on a GPU
.venv/bin/python kaggle/build_kernel.py explain --user <you>    # notebooks 03–06 + app examples
.venv/bin/kaggle kernels push -p kaggle/build/explain
.venv/bin/kaggle kernels output <you>/skin-lesion-triage-explain -p kaggle/out/explain
# copy kaggle/out/explain/proj/{models,results,figures,notebooks,app/examples} into the repo
```
`models/cnn.pt`, `models/outputs.npz` and `models/ood.npz` are git-ignored (they're large), so
they come from the kernel output.

**Demo app:** `.venv/bin/python app/app.py` → http://127.0.0.1:7860. The 8 test-set examples
have precomputed SHAP maps. Uploads compute SHAP live (300 evaluations, a few seconds on an M-series Mac).

**Slides:** `.venv/bin/python slides/build_slides.py` → `slides/pitch.pptx` (all numbers are read from `results/`).

## Repo layout
```
src/        data.py (lesion-grouped split, transforms) · model.py (timm, temperature scaling)
            train.py · fusion.py · explain.py (SHAP, Grad-CAM, checks) · triage.py (pipeline,
            OOD guard, summary text) · metrics.py · artifacts.py (dark-corner / hair flags)
notebooks/  01_eda · 02_baselines · 03_train_cnn · 04_calibrate_fusion · 05_explain_shap ·
            06_eval_fairness   (.py = source, .ipynb = executed)
kaggle/     build_kernel.py: bundles the repo into self-contained Kaggle kernels
app/        app.py (Gradio) · build_examples.py · examples/
splits/     split.csv (70/15/15, StratifiedGroupKFold by lesion_id, seed 42)
models/     config.json (T, τ, OOD thresholds) · xgb.json · fusion_background.csv · train_log.json
results/ figures/ slides/
```

## Design notes and known limitations
- **Split by `lesion_id`.** HAM10000 has several images of the same lesion, so a random image
  split leaks them across splits. No lesion appears in more than one split.
- **Imbalance.** Weighted cross-entropy with `w = 1/sqrt(freq)`; the checkpoint is chosen on
  validation balanced accuracy.
- **Fusion training data.** The fusion model is trained only on validation-split CNN outputs
  (images the CNN never trained on).
- **Grouped metadata SHAP.** The per-patient waterfall uses exact Shapley values over four
  feature groups (image, age, sex, site) in probability units. The global beeswarm uses
  TreeExplainer (log-odds).
- **Artifact flags.** Ink marks and rulers are *not* auto-flagged: colour thresholds confused
  ink with blue-purple pigment.
- **The OOD detector design** was chosen after Mahalanobis alone failed on the development
  non-skin images. That is why we report the second, unseen set separately.
- **Not done** (48-hour stretch items in the plan): a shortcut fix + retrain, out-of-fold CNN
  predictions for fusion, the model-randomization sanity check, test-time augmentation, and
  Hugging Face Spaces deployment.
- **macOS.** torch and xgboost each bundle libomp, and multithreaded OpenMP then segfaults.
  `src/__init__.py` sets `OMP_NUM_THREADS=1` on macOS.

## Citations
Tschandl, Rosendahl & Kittler, *The HAM10000 dataset*, Scientific Data 2018 (CC BY-NC-SA 4.0; the
example images in `app/examples/` are from it) · Lundberg & Lee, SHAP, NeurIPS 2017 · Selvaraju
et al., Grad-CAM, ICCV 2017 · Guo et al., calibration, ICML 2017 · Sun et al., deep kNN OOD, ICML
2022 · Groh et al., Fitzpatrick17k, CVPRW 2021 · Daneshjou et al., DDI, Sci. Adv. 2022 · Winkler
et al., JAMA Dermatology 2019.
