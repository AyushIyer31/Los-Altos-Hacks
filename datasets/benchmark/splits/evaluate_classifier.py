"""Defensible evaluation of the degrader-finder on the MMseqs2 (30% id) splits.

Reports, for the strict-review fixes:
  #4  cluster-grouped 5-fold CV  -> AUROC / AUPRC as mean +/- SD
  #3  AUPRC (prevalence-robust) + precision re-estimated at realistic low prevalence
      (so in-test 32% positive doesn't overstate real-world screening precision)
  #4  bootstrap 95% CIs on the experimentally-confirmed PlasticEnz holdout

Model = same degrader-finder design already in the repo: XGBoost on 25
composition features.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

HERE = Path(__file__).resolve().parent
M = HERE / "mmseqs"
SEED = 42
rng = np.random.default_rng(SEED)

# ---- 25 composition features (matches validate_degrader_finder.feats) ----
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_I = {a: i for i, a in enumerate(AA)}
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

def feats(seq):
    n = max(len(seq), 1)
    comp = np.zeros(20)
    for c in seq:
        if c in AA_I:
            comp[AA_I[c]] += 1
    comp /= n
    arom = sum(seq.count(a) for a in "FWY") / n
    pos = sum(seq.count(a) for a in "KRH") / n
    neg = sum(seq.count(a) for a in "DE") / n
    gravy = sum(KD.get(c, 0) for c in seq) / n
    return np.concatenate([comp, [len(seq), arom, pos, neg, gravy]])

def X_of(frame):
    return np.vstack([feats(s) for s in frame["sequence"].values])

def model(y):
    spw = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=SEED,
        scale_pos_weight=spw)

# ---------------------------------------------------------------- load
comb = pd.read_csv(M / "combined.csv")
pool = comb[comb.partition == "pool"].reset_index(drop=True)
indep = comb[comb.partition == "independent"].reset_index(drop=True)
Xp, yp, fold = X_of(pool), pool.activity_label.values, pool.cv_fold.values.astype(int)
Xi, yi = X_of(indep), indep.activity_label.values
print(f"pool {len(pool)}  independent {len(indep)}")

# ---------------------------------------------------------------- #4 cluster CV
roc_f, pr_f = [], []
oof = np.zeros(len(pool))
for k in range(5):
    tr, te = fold != k, fold == k
    m = model(yp[tr]).fit(Xp[tr], yp[tr])
    p = m.predict_proba(Xp[te])[:, 1]
    oof[te] = p
    roc_f.append(roc_auc_score(yp[te], p))
    pr_f.append(average_precision_score(yp[te], p))
cv = {
    "auroc_mean": round(float(np.mean(roc_f)), 4), "auroc_sd": round(float(np.std(roc_f)), 4),
    "auprc_mean": round(float(np.mean(pr_f)), 4), "auprc_sd": round(float(np.std(pr_f)), 4),
    "auroc_per_fold": [round(x, 4) for x in roc_f],
    "auprc_per_fold": [round(x, 4) for x in pr_f],
    "pooled_oof_auroc": round(float(roc_auc_score(yp, oof)), 4),
    "pooled_oof_auprc": round(float(average_precision_score(yp, oof)), 4),
}

# ---------------------------------------------------------------- #3/#4 independent + bootstrap CI
final = model(yp).fit(Xp, yp)
pi = final.predict_proba(Xi)[:, 1]
auroc_i = roc_auc_score(yi, pi)
auprc_i = average_precision_score(yi, pi)

def boot(metric, n=2000):
    vals = []
    idx = np.arange(len(yi))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yi[s])) < 2:
            continue
        vals.append(metric(yi[s], pi[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)

auroc_ci = boot(roc_auc_score)
auprc_ci = boot(average_precision_score)

# ---------------------------------------------------------------- #3 precision at realistic prevalence
# operating point: threshold 0.5; derive TPR/FPR, then precision at chosen prevalence p
thr = 0.5
pred = (pi >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yi, pred).ravel()
tpr = tp / max(tp + fn, 1)
fpr = fp / max(fp + tn, 1)
def precision_at(p):
    num = p * tpr
    den = p * tpr + (1 - p) * fpr
    return round(float(num / den), 4) if den > 0 else None
test_prev = float(yi.mean())
prevalence = {
    "operating_threshold": thr, "tpr": round(tpr, 4), "fpr": round(fpr, 4),
    "precision_at_test_prevalence_%.2f" % test_prev: precision_at(test_prev),
    "precision_at_prevalence_0.05": precision_at(0.05),
    "precision_at_prevalence_0.01": precision_at(0.01),
    "note": "real-world degrader prevalence is ~1%; precision drops sharply vs the 30% test mix",
}

# ---------------------------------------------------------------- report
report = {
    "model": "XGBoost on 25 composition features (repo degrader-finder design)",
    "split": "MMseqs2 30% identity; CV is cluster-grouped; PlasticEnz = confirmed holdout",
    "in_distribution_cluster_cv_5fold": cv,
    "independent_plasticenz_confirmed": {
        "n": int(len(yi)), "n_pos": int(yi.sum()),
        "auroc": round(float(auroc_i), 4), "auroc_95ci": auroc_ci,
        "auprc": round(float(auprc_i), 4), "auprc_95ci": auprc_ci,
        "auprc_random_baseline": round(test_prev, 4),
    },
    "precision_vs_prevalence": prevalence,
}
(M / "eval_report.json").write_text(json.dumps(report, indent=2))
print("\n===== EVALUATION REPORT =====")
print(json.dumps(report, indent=2))
print(f"\nwritten to {M}/eval_report.json")
