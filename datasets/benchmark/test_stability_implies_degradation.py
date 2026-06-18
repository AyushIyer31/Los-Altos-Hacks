"""HYPOTHESIS: does stability imply degradability?

Test on the HARD test set (degraders vs look-alike non-degraders). Predictor =
a per-enzyme thermostability score from sequence (IVYWREL index + stability
features). If stability implies degradability, the score should separate
degraders (label 1) from non-degraders (label 0).

We evaluate the stability score DIRECTLY as the predictor (no classifier
training) -> a clean test of the hypothesis.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from scipy.stats import pointbiserialr

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "hard_test_set.csv")
y = df.activity_label.values
print(f"hard test set: {len(df)}  ({int(y.sum())} degrader / {int((y==0).sum())} look-alike)")


def stability_score(s):
    """Higher = predicted more thermostable (IVYWREL index, Zeldovich 2007)."""
    s = "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    n = max(len(s), 1)
    return sum(s.count(c) for c in "IVYWREL") / n


def aliphatic_index(s):
    s = "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    n = max(len(s), 1)
    return (s.count("A") + 2.9*s.count("V") + 3.9*(s.count("I")+s.count("L"))) / n * 100


print("computing per-enzyme stability scores...")
ivy = np.array([stability_score(s) for s in df.sequence])
ali = np.array([aliphatic_index(s) for s in df.sequence])

for name, score in [("IVYWREL thermostability index", ivy), ("aliphatic index", ali)]:
    auc = roc_auc_score(y, score)
    auc = max(auc, 1 - auc)                       # direction-agnostic separability
    r = pointbiserialr(y, score)
    # threshold at the value that best separates (median split for acc/prec/rec)
    thr = np.median(score)
    # orient so higher score -> predict degrader if degraders score higher on avg
    direction = 1 if score[y == 1].mean() >= score[y == 0].mean() else -1
    pred = ((score - thr) * direction >= 0).astype(int)
    print(f"\n=== predictor: {name} ===")
    print(f"  ROC-AUC (stability separates degrader vs not): {auc:.3f}")
    print(f"  point-biserial r with label: {r.correlation:+.3f} (p={r.pvalue:.2g})")
    print(f"  Accuracy : {accuracy_score(y, pred):.3f}")
    print(f"  Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"  Recall   : {recall_score(y, pred, zero_division=0):.3f}")
    print(f"  F1       : {f1_score(y, pred, zero_division=0):.3f}")

best = max(roc_auc_score(y, ivy), 1-roc_auc_score(y, ivy),
          roc_auc_score(y, ali), 1-roc_auc_score(y, ali))
print("\nVERDICT:", "stability implies degradability" if best >= 0.65 else
      ("WEAK/NO support -- stability does NOT imply degradability" if best < 0.6 else "modest"),
      f"(best AUC={best:.3f}; 0.5 = no relationship)")
