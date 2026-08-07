"""HARD source-holdout test: train the degrader-finder on every source EXCEPT
PlasticEnz, then test on the full PlasticEnz set (994 sequences).

PlasticEnz sequences (and near-duplicate homologs) are scrubbed from training
so the model has genuinely never seen them -> cross-database generalization test.
This is NOT the 19K stability model (that scores mutations, not sequences); it's
the degrader-finder (sequence -> degrader/non-degrader).
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
import xgboost as xgb

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from validate_degrader_finder import feats   # reuse composition features

VALID = set("ACDEFGHIKLMNPQRSTVWY")


def valid(s):
    s = str(s).strip().upper()
    return s if (s and set(s) <= VALID and 80 <= len(s) <= 1200) else None


# ---- PlasticEnz test set (full 994) ----
pe_rows = []
for f in ["train.csv", "test.csv"]:
    d = pd.read_csv(HERE / "plasticenz" / f)
    for r in d.itertuples():
        v = valid(r.sequence)
        if v:
            pe_rows.append((v, int(r.label)))
pe = pd.DataFrame(pe_rows, columns=["sequence", "label"]).drop_duplicates("sequence")
pe_seqs = set(pe.sequence)
print(f"PlasticEnz test set: {len(pe)} ({int(pe.label.sum())} deg / {int((pe.label==0).sum())} non)")

# ---- training = benchmark minus PlasticEnz source ----
bench = pd.read_csv(HERE / "benchmark_v3.csv")
train = bench[bench.source != "PlasticEnz"][["sequence", "activity_label"]].copy()
train.columns = ["sequence", "label"]
train["sequence"] = train.sequence.str.upper()
print(f"training pool (non-PlasticEnz): {len(train)}")

# ---- scrub PlasticEnz exact + near-duplicates from training ----
def km(s, k=6):
    return {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else {s}
pe_km = [km(s) for s in pe_seqs]
idx = defaultdict(set)
for i, kk in enumerate(pe_km):
    for m in kk:
        idx[m].add(i)
pe_list = list(pe_seqs)
def leaks(s):
    sk = km(s); cand = set()
    for m in sk:
        cand |= idx.get(m, set())
    for ci in cand:
        t = pe_list[ci]
        if s == t or s in t or t in s or len(sk & pe_km[ci]) / len(sk | pe_km[ci]) > 0.3:
            return True
    return False
before = len(train)
train = train[~train.sequence.map(leaks)].reset_index(drop=True)
print(f"  scrubbed {before-len(train)} PlasticEnz overlaps/homologs from training -> {len(train)} train rows")

# ---- features + train ----
print("building features...")
Xtr = np.vstack([feats(s) for s in train.sequence])
ytr = train.label.values
Xte = np.vstack([feats(s) for s in pe.sequence])
yte = pe.label.values

spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=spw, eval_metric="logloss", n_jobs=4, verbosity=0)
m.fit(Xtr, ytr)
proba = m.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(yte, pred)

print("\n===== DEGRADER-FINDER on PlasticEnz (held-out source, leak-free) =====")
print(f"  Accuracy : {accuracy_score(yte, pred):.3f}")
print(f"  Precision: {precision_score(yte, pred, zero_division=0):.3f}")
print(f"  Recall   : {recall_score(yte, pred, zero_division=0):.3f}")
print(f"  F1       : {f1_score(yte, pred, zero_division=0):.3f}")
print(f"  ROC-AUC  : {roc_auc_score(yte, proba):.3f}")
print(f"  PR-AUC   : {average_precision_score(yte, proba):.3f}")
print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
print(f"\n  (test n={len(pe)}, trained on {len(train)} sequences from PMBD/UniProt/PDG_DB/SergejB)")
