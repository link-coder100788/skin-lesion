# Skin lesion triage with explainable AI

> ⚠️ **Student research demo. Not a medical device and not a diagnosis.**

Upload a dermoscopy photo of a skin spot. The app says whether to see a dermatologist soon,
shows which regions of the image drove that call (image SHAP, with Grad-CAM alongside),
shows how age, sex and body site moved the score (a metadata SHAP waterfall), and refuses to
give a verdict on images that look unlike its training data.

See [PLAN.md](PLAN.md) for the full plan and [slides/pitch.pptx](slides/pitch.pptx) for the pitch deck.

## Model (chosen on validation only)

| Validation split (HAM10000, held-out lesions) | Balanced acc. | Urgent AUROC | Specificity @ 90% sens. |
|---|---|---|---|
| EfficientNet-B0, 224 px (first model) | 0.676 | 0.897 | 0.70 |
| FastViT-T12, 224 px, HAM10000 only (best of 6 architectures, [slides](slides/pitch.pptx)) | 0.786 | 0.931 | 0.79 |
| **FastViT-T12, 384 px, + 15,316 ISIC 2019 training images (deployed)** | **0.832** | **0.948** | **0.89** |

The architecture sweep (EfficientNet-B0, EfficientNetV2-B0, FastViT-T12/S12, ConvNeXt V2-Femto,
MobileNetV4-Hybrid-M) used one recipe (label smoothing 0.1, drop path 0.2, 20 epochs) and is in
`results/sweep.json`. MobileNetV4 did not converge with that recipe. The winner was then retrained
at 384 px for 2 + 30 epochs (2.9 h on a Kaggle GPU) with extra ISIC 2019 images that are not in
HAM10000 (SCC merged into akiec). Only the *training* split grew: validation and test are
still HAM10000, so the numbers stay comparable.

## Results (locked HAM10000 test split, 1,503 images, lesion-grouped)

95% CIs from 1,000 bootstrap resamples. `T`, `τ` and the fusion decision were all fixed on
validation before the test set was scored. The test set was scored for the first model and then
once more for the final model, never for the sweep.

| Model | Urgent sensitivity @ τ | Specificity | AUROC (urgent) | Balanced acc. (7-class) |
|---|---|---|---|---|
| A · always "nevus" | 0.00 | 1.00 | 0.50 | 0.14 |
| B · metadata-only logistic regression | 0.92 [0.89, 0.95] | 0.37 [0.34, 0.40] | 0.77 | 0.38 |
| **CNN (FastViT-T12, calibrated), deployed** | **0.80 [0.76, 0.85]** | **0.87 [0.85, 0.89]** | **0.93 [0.91, 0.95]** | **0.82 [0.80, 0.85]** |
| CNN + metadata (XGBoost fusion) | 0.90 [0.86, 0.93] | 0.82 [0.79, 0.84] | 0.94 | 0.82 |
| *first model: EfficientNet-B0, calibrated* | *0.93 [0.90, 0.96]* | *0.72 [0.69, 0.74]* | *0.90* | *0.66* |

- "Urgent" = melanoma, basal cell carcinoma, actinic keratosis/intraepithelial carcinoma.
  τ = 0.150 is the highest threshold that keeps validation sensitivity ≥ 0.90.
- **The 90% sensitivity target was missed on test (0.80).** Validation was used to pick the
  architecture, the checkpoint epoch, T and τ, so the threshold sits on an optimistic curve. We did
  not move τ, or switch to fusion, after seeing the test result: either would make the test set a
  second validation set. The fix is a separate calibration split for τ. The final model ranks
  cases much better (AUROC 0.90 → 0.93, balanced accuracy 0.66 → 0.82), but at this τ it trades
  sensitivity for specificity compared with the first model.
- Temperature scaling (T = 0.66) lowers the test calibration error (ECE) from 0.129 to 0.054.
- Per-class recall: akiec 0.59, bcc 0.77, bkl 0.87, df 1.00, mel 0.64, nv 0.89, vasc 1.00.
  Published HAM10000 models reach a balanced accuracy of about 0.75–0.88; this one (0.82) is inside that range.
- **Fusion is a null result on validation.** Adding age, sex and body site did not improve the
  triage on validation (lesion-grouped CV), so the CNN makes the decision. Fusion is kept to
  *explain* how the metadata shifts the score.

### What checking the explanations showed (notebook 05)
- **Shortcut hunt.** Dark vignette corners appear in 10% of melanomas but only 2% of nevi.
  SHAP puts a median of 1.4% of its positive mass in the corners, and the corners cover 3.9%
  of the image. **We found no evidence that the model uses them.** SHAP mass is concentrated in
  the lesion (melanoma: 82% inside the mask vs. 34% lesion area; it was 40% for EfficientNet-B0).
- **Faithful.** Deleting the top-SHAP pixels lowers the urgent score much faster than deleting
  random pixels (deletion AUC 0.26 vs. 0.58; Grad-CAM 0.34).
- **Moderately stable.** A map is deterministic for a fixed input. A mirror flip keeps it fairly
  similar (rank corr. 0.69), faint noise less so (0.39), and SHAP and Grad-CAM agree moderately
  (median 0.35). The EfficientNet-B0 maps were far less stable (0.14 / 0.05 / 0.01). Still,
  read the maps as hints, not ground truth.
- The metadata matters little. The mean |contribution| to p_urgent is 0.25 for the image,
  0.013 for age, 0.023 for body site and 0.002 for sex.

### Fairness and safety (notebook 06, 04)
| Data | Urgent sensitivity @ τ | Specificity | n urgent | Flagged as OOD |
|---|---|---|---|---|
| HAM10000 test (dermoscopy) | 0.80 [0.76, 0.84] | 0.87 | 295 | – |
| Fitzpatrick17k types I–II (clinical photos) | 0.71 [0.68, 0.75] | 0.63 | 783 | 40% |
| Fitzpatrick17k types III–IV | 0.69 [0.65, 0.73] | 0.49 | 422 | 38% |
| Fitzpatrick17k types V–VI | 0.77 [0.68, 0.86] | 0.36 | 92 | 30% |

On clinical photos sensitivity drops for every skin type, and **specificity falls steeply with
darker skin** (0.63 → 0.36), so darker skin gets many more false alarms. Photo type
(clinical vs. dermoscopy) and skin tone are mixed in this comparison, and HAM10000 has no
skin-type labels and very little dark skin. The fix is diverse *dermoscopic* data.

**Out-of-distribution guard.** An image is flagged if its kNN cosine distance to the training
embeddings, or its Mahalanobis distance, exceeds the validation 99th percentile (1.5% of real
validation images are flagged). It flagged 10/12 non-skin development images (missed: a
photo of a man with a camera and a Hubble star field) and 9/10 unseen ones (missed: a
histology slide). It also flags about 39% of Fitzpatrick17k clinical photos. A flagged image
gets "cannot assess", never a verdict. The app never says "you are fine".

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
The training kernels attach `cdeotte/jpeg-isic2019-512x512` (ISIC 2019) and the Fitzpatrick17k
images on Kaggle directly, so they are not downloaded locally.

**Pipeline** (heavy steps run as private Kaggle GPU kernels):
```bash
notebooks/run.sh 01 02                                          # EDA + baselines (local, CPU)
.venv/bin/python kaggle/build_kernel.py sweep --sweep final --user <you>   # bundle src/ into a kernel
.venv/bin/kaggle kernels push -p kaggle/build/sweep-final       # train FastViT-T12 @ 384 px (~3 h GPU)
.venv/bin/python kaggle/build_kernel.py explain --user <you> --from-sweep final/fastvit_t12_384
                                                                # notebooks 03–06 + app examples (~2 h)
.venv/bin/kaggle kernels push -p kaggle/build/explain
.venv/bin/kaggle kernels output <you>/skin-lesion-triage-explain -p kaggle/out/explain
# copy kaggle/out/explain/proj/{models,results,figures,notebooks,app/examples} into the repo
```
`models/cnn.pt`, `models/outputs.npz` and `models/ood.npz` are git-ignored (they're large), so
they come from the kernel output.

**Demo app:** `.venv/bin/python app/app.py` → http://127.0.0.1:7860. The 8 test-set examples
have precomputed SHAP maps. Uploads compute SHAP live (300 evaluations at 384 px, roughly 10–30 s on an M-series Mac).

**Slides:** `.venv/bin/python slides/build_slides.py` → `slides/pitch.pptx` (all numbers are read from `results/`).

## Repo layout
```
src/        data.py (lesion-grouped split, transforms, ISIC 2019 mapping) · model.py (timm, temperature scaling)
            train.py · fusion.py · explain.py (SHAP, Grad-CAM, checks) · triage.py (pipeline,
            OOD guard, summary text) · metrics.py · artifacts.py (dark-corner / hair flags)
notebooks/  01_eda · 02_baselines · 03_train_cnn · 04_calibrate_fusion · 05_explain_shap ·
            06_eval_fairness   (.py = source, .ipynb = executed)
kaggle/     build_kernel.py: bundles the repo into self-contained Kaggle kernels (train / sweep / explain)
            compare_sweeps.py: validation-only comparison of sweep runs
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
- **Validation did too many jobs.** It picked the architecture, the epoch, T and τ. That is why
  the 90% sensitivity target held on validation but not on test (see Results).
- **Not done** (48-hour stretch items in the plan): a separate calibration split, a shortcut fix +
  retrain, out-of-fold CNN predictions for fusion, the model-randomization sanity check,
  test-time augmentation, and Hugging Face Spaces deployment.
- **macOS.** torch and xgboost each bundle libomp, and multithreaded OpenMP then segfaults.
  `src/__init__.py` sets `OMP_NUM_THREADS=1` on macOS.

## Citations
Tschandl, Rosendahl & Kittler, *The HAM10000 dataset*, Scientific Data 2018 (CC BY-NC-SA 4.0; the
example images in `app/examples/` are from it) · Lundberg & Lee, SHAP, NeurIPS 2017 · Selvaraju
et al., Grad-CAM, ICCV 2017 · Guo et al., calibration, ICML 2017 · Sun et al., deep kNN OOD, ICML
2022 · Groh et al., Fitzpatrick17k, CVPRW 2021 · Daneshjou et al., DDI, Sci. Adv. 2022 · Winkler
et al., JAMA Dermatology 2019.
