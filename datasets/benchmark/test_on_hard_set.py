"""Train the degrader-finder on benchmark_v3 (now contains hard look-alike
negatives) and evaluate on the held-out HARD test set. Reports all metrics.

NOTE: this is the degrader-finder (sequence -> degrader/non-degrader), NOT the
19K stability model (which scores mutations and cannot classify sequences).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
import xgboost as xgb

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from validate_degrader_finder import feats

train = pd.read_csv(HERE / "benchmark_v3.csv")
test = pd.read_csv(HERE / "hard_test_set.csv")
print(f"TRAIN (benchmark_v3): {len(train)}  ({int((train.activity_label==1).sum())} deg / {int((train.activity_label==0).sum())} non)")
print(f"TEST  (hard set)    : {len(test)}  ({int((test.activity_label==1).sum())} deg / {int((test.activity_label==0).sum())} look-alike)")

print("building features...")
Xtr = np.vstack([feats(s) for s in train.sequence.str.upper()])
ytr = train.activity_label.values
Xte = np.vstack([feats(s) for s in test.sequence.str.upper()])
yte = test.activity_label.values

spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                      scale_pos_weight=spw, eval_metric="logloss", n_jobs=4, verbosity=0)
m.fit(Xtr, ytr)
proba = m.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(yte, pred)

print("\n===== DEGRADER-FINDER on the HARD TEST SET =====")
print(f"  Accuracy : {accuracy_score(yte, pred):.3f}")
print(f"  Precision: {precision_score(yte, pred, zero_division=0):.3f}")
print(f"  Recall   : {recall_score(yte, pred, zero_division=0):.3f}")
print(f"  F1       : {f1_score(yte, pred, zero_division=0):.3f}")
print(f"  ROC-AUC  : {roc_auc_score(yte, proba):.3f}")
print(f"  PR-AUC   : {average_precision_score(yte, proba):.3f}")
print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
