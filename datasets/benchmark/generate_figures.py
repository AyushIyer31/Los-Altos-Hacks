"""Generate presentation figures for the degrader-finder validation:
ROC curve, Precision-Recall curve, metrics bar chart, and confusion matrix.
Uses the honest clustered split. Saves PNGs to figures/.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from validate_degrader_finder import feats, cluster   # reuse the same logic

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3})
ACC, BLUE, RED, GREEN = "#1b9e8a", "#0b5c8a", "#d64045", "#2e8b57"


def main():
    df = pd.read_csv(HERE / "benchmark_v3.csv")
    seqs = df["sequence"].str.upper().tolist()
    y = df["activity_label"].values
    print("building features + clustering...")
    X = np.vstack([feats(s) for s in seqs])
    cid = cluster(seqs)

    rng = np.random.RandomState(0)
    clusters = np.unique(cid); rng.shuffle(clusters)
    test_clusters = set(clusters[:int(0.2 * len(clusters))])
    te = np.array([c in test_clusters for c in cid])
    tr = ~te

    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, eval_metric="logloss", n_jobs=4, verbosity=0)
    m.fit(X[tr], y[tr])
    proba = m.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yte = y[te]

    # ---- 1. ROC curve ----
    fpr, tpr, _ = roc_curve(yte, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5.5))
    plt.plot(fpr, tpr, color=ACC, lw=2.5, label=f"Degrader-finder (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Degrader-Finder\n(clustered split, leak-free)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG / "roc_curve.png"); plt.close()

    # ---- 2. Precision-Recall curve ----
    prec, rec, _ = precision_recall_curve(yte, proba)
    ap = average_precision_score(yte, proba)
    base = yte.mean()
    plt.figure(figsize=(6, 5.5))
    plt.plot(rec, prec, color=BLUE, lw=2.5, label=f"Degrader-finder (PR-AUC = {ap:.3f})")
    plt.axhline(base, ls="--", color="gray", lw=1, label=f"Baseline (prevalence = {base:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Degrader-Finder\n(clustered split, leak-free)")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(FIG / "precision_recall_curve.png"); plt.close()

    # ---- 3. Metrics bar chart ----
    metrics = {"Accuracy": accuracy_score(yte, pred),
               "Precision": precision_score(yte, pred, zero_division=0),
               "Recall": recall_score(yte, pred, zero_division=0),
               "F1": f1_score(yte, pred, zero_division=0),
               "ROC-AUC": roc_auc, "PR-AUC": ap}
    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=ACC, edgecolor="black", width=0.6)
    for b, v in zip(bars, metrics.values()):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.ylim(0, 1.08); plt.ylabel("Score")
    plt.title("Degrader-Finder Performance\n(clustered / leak-free split, n=%d test)" % te.sum())
    plt.tight_layout(); plt.savefig(FIG / "metrics_bar.png"); plt.close()

    # ---- 4. Confusion matrix ----
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.xticks([0, 1], ["non-degrader", "degrader"]); plt.yticks([0, 1], ["non-degrader", "degrader"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix — Degrader-Finder")
    plt.grid(False); plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png"); plt.close()

    print("saved figures:")
    for f in ["roc_curve", "precision_recall_curve", "metrics_bar", "confusion_matrix"]:
        print(f"  figures/{f}.png")
    print(f"\nmetrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
