"""HYPOTHESIS TEST: does sequence-based thermostability predict whether an
enzyme degrades PET at HIGH temperature (>=60 C)?

Outcome  : Erickson measured degradation at 60-70 C  (degrader vs not)
Predictor: thermostability features from sequence ONLY (independent of the
           degradation measurements) -- IVYWREL index, charged content, etc.
Eval     : leave-one-out CV (only ~63 enzymes). Reports all metrics + correlation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).parent
e = pd.read_csv(HERE / "erickson2022_degradation.csv")

# ---- per-enzyme high-temp degradation (max at >=60 C), dedup by sequence ----
hi = e[e.temp_C >= 60].groupby("sequence")["aromatic_products_mg_per_L"].max()
df = hi.reset_index().rename(columns={"aromatic_products_mg_per_L": "hi_deg"})
df["label"] = (df["hi_deg"] > 0).astype(int)   # degrades at high temp?
print(f"enzymes (unique sequences): {len(df)}")
print(f"  high-temp degraders (label 1): {df.label.sum()}")
print(f"  non-degraders at high temp (0): {(df.label==0).sum()}")

# ---- thermostability features from sequence (independent of degradation) ----
def thermo_feats(s):
    s = "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    n = max(len(s), 1)
    f = lambda chars: sum(s.count(c) for c in chars) / n
    ivywrel = f("IVYWREL")                 # Zeldovich OGT index (thermophilicity)
    charged = f("DEKR")
    ek = f("EK")                           # salt-bridge formers
    hydroph = f("AILMFWV")
    aromatic = f("FWY")
    pro = f("P")
    gly = f("G")
    # aliphatic index (Ikai) ~ thermostability
    ai = (f("A")*100 + 2.9*f("V")*100 + 3.9*(f("I")+f("L"))*100)
    return [ivywrel, charged, ek, hydroph, aromatic, pro, gly, ai]

FNAMES = ["IVYWREL", "charged", "E+K", "hydrophobic", "aromatic", "Pro", "Gly", "aliphatic_idx"]
X = np.array([thermo_feats(s) for s in df.sequence])
y = df.label.values

# ---- leave-one-out CV with logistic regression ----
pipe_pred = cross_val_predict(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    StandardScaler().fit_transform(X), y, cv=LeaveOneOut())
proba = cross_val_predict(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    StandardScaler().fit_transform(X), y, cv=LeaveOneOut(), method="predict_proba")[:, 1]

print("\n========== HYPOTHESIS TEST RESULTS (leave-one-out CV) ==========")
print(f"  Accuracy : {accuracy_score(y, pipe_pred):.3f}")
print(f"  Precision: {precision_score(y, pipe_pred, zero_division=0):.3f}")
print(f"  Recall   : {recall_score(y, pipe_pred, zero_division=0):.3f}")
print(f"  F1       : {f1_score(y, pipe_pred, zero_division=0):.3f}")
print(f"  ROC-AUC  : {roc_auc_score(y, proba):.3f}")
cm = confusion_matrix(y, pipe_pred)
print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

# ---- continuous correlation: thermostability vs actual degradation ----
log_deg = np.log1p(df.hi_deg.values)
pr = pearsonr(X[:, 0], log_deg); sp = spearmanr(X[:, 0], log_deg)
print("\n  Correlation of IVYWREL thermostability index vs log(high-temp degradation):")
print(f"    Pearson r = {pr[0]:.3f} (p={pr[1]:.3g}) | Spearman rho = {sp[0]:.3f} (p={sp[1]:.3g})")

# single-feature AUCs (which thermostability signal matters most)
print("\n  Single-feature ROC-AUC for predicting high-temp degrader:")
for i, name in enumerate(FNAMES):
    try:
        a = roc_auc_score(y, X[:, i])
        print(f"    {name:14s} AUC={max(a,1-a):.3f}")
    except Exception:
        pass

verdict = roc_auc_score(y, proba)
print("\nVERDICT:", "supports hypothesis" if verdict >= 0.65 else
      ("weak/no support" if verdict < 0.6 else "modest support"),
      f"(AUC={verdict:.3f})")
