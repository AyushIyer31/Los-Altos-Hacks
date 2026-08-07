"""Bootstrap 95% confidence intervals for the headline metrics, reusing saved
predictions (no retraining). S669 for the mutation model, BRENDA for the screener."""
import csv
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
B = 2000
HERE = os.path.dirname(os.path.abspath(__file__))


def ci(vals):
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot(fn, n):
    return [fn(rng.integers(0, n, n)) for _ in range(B)]


print("=" * 60)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (B=2000)")
print("=" * 60)

# ---------- S669 (mutation model) ----------
y = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)["y_s669"]
preds = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "models/*_s669_pred.npy")))
         if len(np.load(f)) == len(y)]
ens = np.mean(np.stack(preds), 0)
yt = (y < 0).astype(int)

pear = boot(lambda i: np.corrcoef(ens[i], y[i])[0, 1], len(y))
auc = boot(lambda i: roc_auc_score(yt[i], -ens[i]) if 0 < yt[i].sum() < len(i) else np.nan, len(y))
auc = [a for a in auc if not np.isnan(a)]
m, lo, hi = ci(pear)
print(f"\nS669  Pearson : {m:.3f}  (95% CI {lo:.3f} – {hi:.3f})   n={len(y)}")
m, lo, hi = ci(auc)
print(f"S669  AUC     : {m:.3f}  (95% CI {lo:.3f} – {hi:.3f})   [stabilizing detection]")

# ---------- BRENDA (screener model) ----------
rows = list(csv.DictReader(open(os.path.join(HERE, "models_tm/brenda_tm_predictions.csv"))))
pred = np.array([float(r["pred_tm_ensemble"]) for r in rows])
temp = np.array([float(r["true_stable_temp_c"]) for r in rows])
lab = np.array([r["label"] for r in rows])
keep = np.isin(lab, ["positive", "negative"])
p_k, yk = pred[keep], (lab[keep] == "positive").astype(int)

aucb = boot(lambda i: roc_auc_score(yk[i], p_k[i]) if 0 < yk[i].sum() < len(i) else np.nan, len(yk))
aucb = [a for a in aucb if not np.isnan(a)]
pearb = boot(lambda i: np.corrcoef(pred[i], temp[i])[0, 1], len(pred))
m, lo, hi = ci(aucb)
print(f"\nBRENDA AUC    : {m:.3f}  (95% CI {lo:.3f} – {hi:.3f})   n={int(keep.sum())} (pos/neg)")
m, lo, hi = ci(pearb)
print(f"BRENDA Pearson: {m:.3f}  (95% CI {lo:.3f} – {hi:.3f})   n={len(pred)} (all)")
print("\ndone.")
