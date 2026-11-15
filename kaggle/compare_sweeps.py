"""Compare sweep outputs on the VALIDATION split only (never test).

    python kaggle/compare_sweeps.py models kaggle/out/sweep/b0_reg kaggle/out/sweep-arch-a/*/
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_metadata  # noqa: E402
from src.metrics import is_urgent, p_urgent, sens_spec, threshold_for_sensitivity  # noqa: E402
from src.model import fit_temperature, softmax_np  # noqa: E402

df = load_metadata(Path(__file__).resolve().parent.parent / "data/ham").set_index("image_id")
print(f"{'model':22s} {'val bAcc':>8s} {'AUROC':>6s} {'spec@90':>7s} {'T':>5s} {'finite':>6s}")
for d in sys.argv[1:]:
    o = np.load(Path(d) / "outputs.npz", allow_pickle=True)
    lg = o["val_logits"].astype(np.float64)
    ok = np.isfinite(lg).all()
    if not ok:
        print(f"{Path(d).name:22s} non-finite logits, skipped"); continue
    y = df.loc[o["val_ids"], "y"].to_numpy()
    T = fit_temperature(lg, y)
    p = softmax_np(lg, T)
    yu, s = is_urgent(y), p_urgent(p)
    tau = threshold_for_sensitivity(yu, s)
    print(f"{Path(d).name:22s} {balanced_accuracy_score(y, p.argmax(1)):8.3f} {roc_auc_score(yu, s):6.3f} "
          f"{sens_spec(yu, s, tau)[1]:7.3f} {T:5.2f} {str(ok):>6s}")
