"""Threshold sweep for the ddG mutation model on S669, treated as a STABILIZING-
mutation classifier (positive = stabilizing). Predict 'stabilizing' if predicted
ddG < threshold; sweep the threshold to show the precision/recall trade-off."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")
y = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)["y_s669"]

# ensemble = mean of all saved per-model S669 predictions
preds = []
for f in sorted(glob.glob(os.path.join(M, "*_s669_pred.npy"))):
    p = np.load(f)
    if len(p) == len(y):
        preds.append(p)
ens = np.mean(np.stack(preds), 0)

yt = (y < 0).astype(int)          # positive class = stabilizing
P, N = int(yt.sum()), int((1 - yt).sum())

from sklearn.metrics import roc_auc_score
auc = float(roc_auc_score(yt, -ens))   # more-negative ddG = more stabilizing


def at(thr):
    pp = (ens < thr).astype(int)        # predict 'stabilizing' if ddG < thr
    tp = int(((pp == 1) & (yt == 1)).sum()); fp = int(((pp == 1) & (yt == 0)).sum())
    fn = int(((pp == 0) & (yt == 1)).sum()); tn = int(((pp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(yt)
    return dict(thr=round(thr, 2), precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), accuracy=round(acc, 3), tp=tp, fp=fp, fn=fn, tn=tn)


print(f"S669 stabilizing/destabilizing = {P}/{N}  | ensemble AUC = {auc:.3f}\n")
print(f"{'thr':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}   (tp,fp,fn,tn)")
rows = [at(t) for t in np.arange(-1.5, 1.01, 0.25)]
best = max(rows, key=lambda r: r["f1"])
for r in rows:
    star = "  <- best F1" if r["thr"] == best["thr"] else ""
    print(f"{r['thr']:>6} {r['precision']:>6} {r['recall']:>7} {r['f1']:>6} "
          f"{r['accuracy']:>6}   ({r['tp']},{r['fp']},{r['fn']},{r['tn']}){star}")
print(f"\nBest-F1 threshold = {best['thr']} kcal/mol: "
      f"P={best['precision']} R={best['recall']} F1={best['f1']} acc={best['accuracy']}")
json.dump({"auc": round(auc, 3), "best": best, "sweep": rows},
          open(os.path.join(M, "s669_threshold_sweep.json"), "w"), indent=2)
print("\nwrote", os.path.join(M, "s669_threshold_sweep.json"))
