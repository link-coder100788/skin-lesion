# Spot Check (iOS)

A SwiftUI iPhone app that runs the deployed triage pipeline fully on device: no server, no Python.

| Piece | On device |
|---|---|
| FastViT-T12 @ 384 px | Core ML (`LesionNet.mlmodel`, converted from `models/cnn.pt`) |
| Temperature scaling, τ, triage text | Swift (`Pipeline/Triage.swift`) |
| OOD guard (kNN cosine + Mahalanobis) | Swift + Accelerate (`Pipeline/OODGuard.swift`) |
| Metadata SHAP (exact, 4 groups) and XGBoost | Swift tree evaluator (`Pipeline/Fusion.swift`) |
| Image SHAP | Swift Owen-value partition explainer, blur masker, 300 evaluations (`Pipeline/ImageShap.swift`) |
| Grad-CAM | Precomputed for the bundled examples only: Core ML can't compute gradients on device |

## Build
```bash
.venv/bin/pip install coremltools          # once
.venv/bin/python ios/export_assets.py      # needs models/ from the Kaggle explain kernel
cd ios && xcodegen                         # brew install xcodegen; regenerates Veritas.xcodeproj
open Veritas.xcodeproj                     # pick an iPhone simulator or your iPhone (set a signing team)
```
`ios/Veritas/Generated/` is git-ignored (about 55 MB) and rebuilt by the export script.

## Tests
`xcodebuild -project ios/Veritas.xcodeproj -scheme Veritas -destination 'platform=iOS Simulator,name=iPhone 17' test`

`VeritasTests/ParityTests.swift` checks the phone against the Python pipeline (`fixtures.json`) on the
bundled examples. The checks are: Core ML vs PyTorch probabilities (±0.02) with the same triage and OOD
decisions; XGBoost and metadata SHAP (±1e-4); the image SHAP map's sum (Owen efficiency) and its
agreement with the Python SHAP map; and a non-skin photo flagged as "cannot assess".

Debug builds accept `-demo <example id>` (e.g. `-demo cat`) and `-scrollEnd 1` as launch arguments, for screenshots.
