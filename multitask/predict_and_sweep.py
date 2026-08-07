"""Load the trained Tm boosting models, predict on the BRENDA test set, write
per-protein predictions, and sweep the stable/unstable decision threshold to find
the best precision/recall/F1 trade-off (instead of the fixed 60C cut)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "features_tm.npz")
M = os.path.join(HERE, "models_tm")
PRED_OUT = os.path.join(M, "brenda_tm_predictions.csv")
SWEEP_OUT = os.path.join(M, "threshold_sweep.json")

import joblib

d = np.load(F, allow_pickle=True)
Xb = d["X_brenda"]
temp = d["brenda_temp"].astype(float)
label = d["brenda_label"].astype(str)
basis = d["brenda_basis"].astype(str)

# light, equally-good boosting ensemble
names = ["tm_lightgbm", "tm_xgboost", "tm_catboost"]
preds = {}
for n in names:
    m = joblib.load(os.path.join(M, f"{n}.joblib"))
    preds[n] = m.predict(Xb)
ens = np.mean(np.stack(list(preds.values())), 0)

# write per-protein predictions
with open(PRED_OUT, "w") as f:
    f.write("idx,true_stable_temp_c,basis,label,pred_tm_ensemble," + ",".join(names) + "\n")
    for i in range(len(ens)):
        row = [i, temp[i], basis[i], label[i], round(float(ens[i]), 2)]
        row += [round(float(preds[n][i]), 2) for n in names]
        f.write(",".join(map(str, row)) + "\n")

# threshold sweep on positive/negative subset
keep = np.isin(label, ["positive", "negative"])
yt = (label[keep] == "positive").astype(int)
score = ens[keep]
P = int(yt.sum()); N = int((1 - yt).sum())


def at(thr):
    pp = (score >= thr).astype(int)
    tp = int(((pp == 1) & (yt == 1)).sum()); fp = int(((pp == 1) & (yt == 0)).sum())
    fn = int(((pp == 0) & (yt == 1)).sum()); tn = int(((pp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(yt)
    return dict(thr=thr, precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), accuracy=round(acc, 3), tp=tp, fp=fp, fn=fn, tn=tn)

from sklearn.metrics import roc_auc_score
auc = float(roc_auc_score(yt, score))
print(f"BRENDA pos/neg = {P}/{N}  | ensemble AUC = {auc:.3f}\n")
print(f"{'thr':>4} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}   (tp,fp,fn,tn)")
rows = [at(t) for t in range(40, 86, 2)]
best = max(rows, key=lambda r: r["f1"])
for r in rows:
    star = "  <- best F1" if r["thr"] == best["thr"] else ""
    print(f"{r['thr']:>4} {r['precision']:>6} {r['recall']:>7} {r['f1']:>6} "
          f"{r['accuracy']:>6}   ({r['tp']},{r['fp']},{r['fn']},{r['tn']}){star}")

print(f"\nBest-F1 threshold = {best['thr']}C: "
      f"P={best['precision']} R={best['recall']} F1={best['f1']} acc={best['accuracy']}")
json.dump({"auc": round(auc, 3), "best": best, "sweep": rows},
          open(SWEEP_OUT, "w"), indent=2)
print(f"\nwrote {PRED_OUT}\nwrote {SWEEP_OUT}")
