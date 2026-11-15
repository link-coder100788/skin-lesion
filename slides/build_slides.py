"""Build the 5-minute pitch deck (PLAN.md section 12) from results/*.json and figures/.

    .venv/bin/python slides/build_slides.py      # -> slides/pitch.pptx

Every number on the slides is read from the results files, so the deck always matches the
latest run.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
R = {n: json.load(open(ROOT / f"results/{n}.json")) for n in ("eda", "calibration_fusion", "explain", "eval")}
CFG = json.load(open(ROOT / "models/config.json"))

INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x5F, 0x6B, 0x76)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x84, 0x49)
LIGHT = RGBColor(0xF4, 0xF6, 0xF7)
W, H = Inches(13.333), Inches(7.5)
DISCLAIMER = "Student research demo · not a medical device · not a diagnosis"

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for k, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.color.rgb, run.font.name = Pt(size), bold, color, font
    return box


def bullets(slide, x, y, w, h, items, size=18, color=INK, space=8):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for k, item in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        bold = item.startswith("**")
        run = p.add_run()
        run.text = "•  " + item.strip("*")
        run.font.size, run.font.color.rgb, run.font.bold, run.font.name = Pt(size), color, bold, "Calibri"
    return box


def new_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    bar.fill.solid(); bar.fill.fore_color.rgb = RED; bar.line.fill.background()
    if kicker:
        text(s, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4), kicker.upper(), 12, True, RED)
    text(s, Inches(0.6), Inches(0.62), Inches(12.2), Inches(0.9), title, 30, True)
    text(s, Inches(0.6), Inches(7.05), Inches(12), Inches(0.3), DISCLAIMER, 10, color=MUTED)
    return s


def image(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def box(slide, x, y, w, h, fill=LIGHT, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.adjustments[0] = 0.08
    return shp


def stat(slide, x, y, value, label, color=INK):
    box(slide, x, y, Inches(2.9), Inches(1.45))
    text(slide, x, y + Inches(0.12), Inches(2.9), Inches(0.7), value, 34, True, color, PP_ALIGN.CENTER)
    text(slide, x + Inches(0.15), y + Inches(0.85), Inches(2.6), Inches(0.6), label, 12, color=MUTED,
         align=PP_ALIGN.CENTER)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


ev = R["eval"]["test_results"]
cnn = ev["CNN (calibrated)"]
fus = ev["CNN + metadata (XGBoost)"]
ci = lambda m: f"{m[0]:.2f} (95% CI {m[1]:.2f}–{m[2]:.2f})"
fair = {r["group"]: r for r in R["eval"]["fairness"]}
ex = R["explain"]
counts = R["eda"]["class_counts"]
nv_share = counts["nv"] / sum(counts.values())

# 1 · Title ---------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H); bg.fill.solid(); bg.fill.fore_color.rgb = INK
bg.line.fill.background()
text(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.2), "Skin lesion triage\nthat explains itself", 50, True,
     RGBColor(0xFF, 0xFF, 0xFF))
text(s, Inches(0.9), Inches(4.25), Inches(11), Inches(0.8),
     "Upload a photo of a skin spot → should you see a dermatologist soon? Which pixels drove that call? "
     "How did age and body site change it? And where is the model likely to be wrong?", 18,
     color=RGBColor(0xD5, 0xDB, 0xDF))
text(s, Inches(0.9), Inches(6.5), Inches(11), Inches(0.4), "⚠ " + DISCLAIMER, 14, True, RGBColor(0xF5, 0xB0, 0x41))
notes(s, "Title. Say the disclaimer out loud.")

# 2 · Hook ----------------------------------------------------------------------------------
s = new_slide("Caught early, melanoma is very treatable.\nThe wait to see a dermatologist can be months.",
              "The problem")
bullets(s, Inches(0.6), Inches(2.3), Inches(6.4), Inches(4), [
    "Localised melanoma: ~99% five-year survival; once it has spread, far lower",
    "Dermatology waiting lists often run to months",
    "A triage tool can help decide who should be seen first — if people can trust it",
    "Trust needs more than a score: it needs reasons, and honest limits",
], 20)
stat(s, Inches(7.3), Inches(2.4), f"{nv_share:.0%}", "of HAM10000 images are harmless moles (nv)")
stat(s, Inches(10.35), Inches(2.4), "0%", "of cancers caught by a model that always says “mole”", RED)
text(s, Inches(7.3), Inches(4.1), Inches(5.9), Inches(1.4),
     f"“Always predict nevus” scores {nv_share:.0%} accuracy and is useless. That is why we never report "
     "plain accuracy.", 16, color=MUTED)
notes(s, "Survival figure: American Cancer Society / SEER, localised melanoma 5-year relative survival ~99%.")

# 3 · What we built -------------------------------------------------------------------------
s = new_slide("A triage tool that shows its reasoning", "What we built")
cols = [("1 · Triage", "“See a doctor soon” or “lower concern”, from a calibrated urgent score "
         f"(mel + bcc + akiec) with a threshold set for ≥90% sensitivity."),
        ("2 · Where it looked", "Image SHAP map: red regions pushed toward “see a doctor”, blue away. "
         "Grad-CAM shown alongside as a cross-check."),
        ("3 · Why, in numbers", "Waterfall: base rate → image → age → sex → body site, in probability "
         "units."),
        ("4 · When not to trust it", "Out-of-distribution guard: non-dermoscopy images get "
         "“cannot assess”, never a verdict.")]
for k, (h_, b_) in enumerate(cols):
    x = Inches(0.6 + k * 3.15)
    box(s, x, Inches(2.0), Inches(2.95), Inches(2.6))
    text(s, x + Inches(0.2), Inches(2.15), Inches(2.6), Inches(0.5), h_, 18, True, RED)
    text(s, x + Inches(0.2), Inches(2.7), Inches(2.6), Inches(1.9), b_, 14)
image(s, FIG / "shap_vs_gradcam.png", Inches(3.4), Inches(4.75), h=Inches(2.25))
text(s, Inches(0.6), Inches(5.2), Inches(2.7), Inches(1.6), "Test images: original, image SHAP, Grad-CAM. Red = pushes toward “see a doctor”.", 12, color=MUTED)
notes(s, "Everything runs in memory; nothing is stored.")

# 4 · Live demo -----------------------------------------------------------------------------
s = new_slide("Live demo", "90 seconds")
bullets(s, Inches(0.6), Inches(1.9), Inches(6), Inches(4.5), [
    "Cached example (melanoma) → triage → SHAP map → waterfall",
    "Toggle to Grad-CAM: do they agree?",
    "The dark-corner artifact example: what the model “looked at”",
    "Upload a cat photo → “cannot assess this image”",
    "Backup: run locally (python app/app.py) + recorded video",
], 20)
image(s, FIG / "tabular_waterfall_example.png", Inches(7.0), Inches(1.9), w=Inches(5.8))
notes(s, "If Wi-Fi fails, the app runs fully offline on a laptop.")

# 5 · How it works --------------------------------------------------------------------------
s = new_slide("Two stages, two layers of explanation", "How it works")
steps = [("Image", Inches(0.6)), ("EfficientNet-B0\nfine-tuned on\nHAM10000", Inches(2.55)),
         ("7 calibrated\nprobabilities\n(temperature T)", Inches(4.9)),
         ("XGBoost fusion\n+ age, sex, site", Inches(7.25)), ("Triage +\nOOD guard", Inches(9.6))]
for k, (label, x) in enumerate(steps):
    b = box(s, x, Inches(2.0), Inches(2.0), Inches(1.3), fill=LIGHT if k else RGBColor(0xFD, 0xED, 0xEC))
    text(s, x, Inches(2.12), Inches(2.0), Inches(1.1), label, 14, True, align=PP_ALIGN.CENTER)
    if k:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x - Inches(0.38), Inches(2.5), Inches(0.32), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = MUTED; a.line.fill.background()
text(s, Inches(2.55), Inches(3.45), Inches(4.3), Inches(0.6), "↓ Image SHAP (Partition) + Grad-CAM", 14, True, RED)
text(s, Inches(7.25), Inches(3.45), Inches(4.3), Inches(0.6), "↓ Tabular SHAP (exact, grouped)", 14, True, RED)
bullets(s, Inches(0.6), Inches(4.3), Inches(12.2), Inches(2.6), [
    "**Split by lesion, not by image: HAM10000 has several photos of the same mole. A random split leaks them "
    "into the test set and inflates scores.",
    "Weighted cross-entropy (1/√frequency) for the 58:1 class imbalance; checkpoint on balanced accuracy",
    f"Temperature scaling (T = {CFG['T']:.2f}) so a 0.8 means ~80%; threshold τ = {CFG['tau_cnn']:.3f} chosen on "
    "validation for ≥90% sensitivity, then frozen",
    "Fusion trained only on images the CNN never saw (validation split, lesion-grouped 5-fold CV)",
], 16)
notes(s, "Why two stages: SHAP on 1,280 raw embedding features says nothing; SHAP on 7 probabilities + age/sex/site tells a story.")

# 6 · Results -------------------------------------------------------------------------------
s = new_slide(f"We catch {cnn['sensitivity_urgent'][0]:.0%} of cancers on held-out lesions", "Results · locked test set")
rows = [("", "Sensitivity\n(urgent)", "Specificity", "AUROC", "Balanced acc.")]
for name in ["A · always nv", "B · metadata LR", "CNN (calibrated)", "CNN + metadata (XGBoost)"]:
    m = ev[name]
    rows.append((name, f"{m['sensitivity_urgent'][0]:.2f}", f"{m['specificity_urgent'][0]:.2f}",
                 f"{m['auroc_urgent'][0]:.2f}", f"{m['balanced_accuracy'][0]:.2f}"))
tbl = s.shapes.add_table(len(rows), 5, Inches(0.6), Inches(1.9), Inches(7.4), Inches(2.6)).table
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = v or " "
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(13)
                run.font.bold = r == 0 or row[0].startswith("CNN (")
                run.font.color.rgb = INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xFD, 0xED, 0xEC) if row[0].startswith("CNN (") else (
            LIGHT if r == 0 else RGBColor(0xFF, 0xFF, 0xFF))
tbl.columns[0].width = Inches(2.6)
for c in range(1, 5):
    tbl.columns[c].width = Inches(1.2)
bullets(s, Inches(0.6), Inches(4.75), Inches(7.4), Inches(2.3), [
    f"Sensitivity {ci(cnn['sensitivity_urgent'])} at specificity {ci(cnn['specificity_urgent'])}",
    f"Balanced accuracy {cnn['balanced_accuracy'][0]:.2f}; published HAM10000 models reach ~0.75–0.88. "
    "A weekend EfficientNet-B0 sits just below that range.",
    f"Calibration error (ECE) {R['eval']['ece_test_before']:.3f} → {R['eval']['ece_test_after']:.3f} after temperature scaling",
    "Metadata fusion did not help: a null result, reported as one",
], 14)
image(s, FIG / "confusion_matrix.png", Inches(8.3), Inches(1.75), h=Inches(4.0))
text(s, Inches(8.3), Inches(5.85), Inches(4.6), Inches(1),
     "Most confusions: mel ↔ nv and bkl ↔ mel, as in the literature.", 13, color=MUTED)
notes(s, "95% CIs: 1,000 bootstrap resamples over test images. Test set was scored once, after every choice was fixed on validation.")

# 7 · What SHAP taught us -------------------------------------------------------------------
hunt = {r["group"]: r for r in ex["shortcut_by_group"]}
byc = {r["dx"]: r for r in ex["shortcut_by_class"]}
CORNER_AREA = 4 * 22 * 22 / 224 ** 2  # the four 10% corner patches used in notebook 05
s = new_slide("We checked the explanations: faithful, but fragile", "What SHAP taught us")
del_auc = ex["deletion_auc"]
image(s, FIG / "shortcut_dark_corners.png", Inches(0.6), Inches(1.75), w=Inches(7.6))
image(s, FIG / "deletion_curve.png", Inches(8.5), Inches(1.65), w=Inches(4.4))
dc = hunt.get("dark corners", {})
dr = R["eda"]["artifact_rate_by_class"]["dark_corners"]
bullets(s, Inches(0.6), Inches(4.85), Inches(12.3), Inches(2.2), [
    f"**Shortcut hunt: dark vignette corners are in {dr['mel']:.0%} of melanomas but {dr['nv']:.0%} of moles, a "
    f"possible shortcut. SHAP puts only {dc.get('corner_share', float('nan')):.1%} of its positive mass in the corners "
    f"(they cover {CORNER_AREA:.1%} of the image), so there is no sign the model uses them.",
    f"Faithful: deleting the top SHAP pixels lowers the urgent score faster than random (deletion AUC "
    f"{del_auc['SHAP']:.2f} vs {del_auc['random']:.2f}), but Grad-CAM does even better ({del_auc['Grad-CAM']:.2f}).",
    f"Fragile: the same image always gives the same map, but a mirror flip (rank corr. "
    f"{ex['stability_median']['flip']:.2f}) or faint noise ({ex['stability_median']['noise']:.2f}) changes it a lot, "
    f"and SHAP and Grad-CAM barely agree ({ex['shap_gradcam_agreement_median']:.2f}). Read the maps as coarse hints.",
], 14, space=4)
notes(s, "Melanoma: median %.0f%% of positive SHAP mass inside the lesion vs %.0f%% lesion area: only modestly lesion-focused. "
         "Winkler et al. 2019 (JAMA Dermatology) showed ink markings fooled a commercial CNN; that is why we looked."
      % (100 * byc["mel"]["inside"], 100 * byc["mel"]["lesion_area"]))

# 8 · Limits and ethics ---------------------------------------------------------------------
s = new_slide("Where the model is likely to be wrong", "Limits and ethics")
image(s, FIG / "fairness_sensitivity.png", Inches(0.6), Inches(1.65), h=Inches(3.55))
image(s, FIG / "ood_examples.png", Inches(6.95), Inches(1.75), w=Inches(6.0))
ham_s = fair["HAM10000 test (dermoscopy)"]["sensitivity"]
fz_s = [fair[g]["sensitivity"] for g in ("I–II", "III–IV", "V–VI") if g in fair]
cf = R["calibration_fusion"]
fresh = cf.get("ood_fresh", [])
bullets(s, Inches(0.6), Inches(5.35), Inches(12.3), Inches(1.7), [
    f"On clinical photos (Fitzpatrick17k), sensitivity falls from {ham_s:.0%} to {min(fz_s):.0%}–{max(fz_s):.0%} for "
    "every skin type. The photo type (clinical vs. dermoscopy) is the bigger shift, and HAM10000 has almost no "
    "dark skin, so the skin-tone gap cannot be ruled out. Fix: diverse, dermoscopic data.",
    f"OOD guard flags {sum(r['flag_unlike_training'] for r in cf['ood_nonskin'])}/{len(cf['ood_nonskin'])} non-skin "
    f"test images and {sum(r['flag_unlike_training'] for r in fresh)}/{len(fresh)} unseen ones (it missed a retina photo). "
    "The app never says “you are fine”.",
], 14, space=4)
notes(s, "Two causes are mixed in the Fitzpatrick result: skin tone and clinical-vs-dermoscopy. We cannot separate them with this data.")

# 9 · Next steps ----------------------------------------------------------------------------
s = new_slide("Next steps", "Where this goes")
bullets(s, Inches(0.6), Inches(2.0), Inches(12), Inches(4.5), [
    "Diverse data: dermoscopy across all skin types (e.g. DDI), then re-run the fairness audit",
    "Shortcut-proofing: crop to the lesion, artifact augmentation, retrain, re-run the shortcut hunt",
    "Clinician feedback on the explanations: do dermatologists find the maps useful?",
    "Uncertainty: ensembles / MC-dropout, shown as a confidence band",
    "Prospective testing before anything clinical: this is a research demo",
], 22)

# 10 · References ---------------------------------------------------------------------------
s = new_slide("References", "Sources")
bullets(s, Inches(0.6), Inches(1.8), Inches(12.2), Inches(5), [
    "Tschandl, Rosendahl, Kittler. The HAM10000 dataset. Scientific Data, 2018. (CC BY-NC-SA 4.0)",
    "Lundberg & Lee. A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS, 2017.",
    "Selvaraju et al. Grad-CAM. ICCV, 2017.",
    "Adebayo et al. Sanity Checks for Saliency Maps. NeurIPS, 2018.",
    "Guo et al. On Calibration of Modern Neural Networks. ICML, 2017.",
    "Sun et al. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML, 2022.",
    "Groh et al. Fitzpatrick17k. CVPR Workshops, 2021.",
    "Daneshjou et al. Disparities in dermatology AI performance (DDI). Science Advances, 2022.",
    "Winkler et al. Surgical skin markings and CNN diagnostic performance. JAMA Dermatology, 2019.",
], 15, space=6)

out = ROOT / "slides/pitch.pptx"
prs.save(out)
print("wrote", out, len(prs.slides), "slides")
