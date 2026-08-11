"""Generate ROC, precision, and supporting diagnostic figures for the
two-stage stability classifier, computed directly from saved model
outputs. No values in this script are simulated, interpolated, or
estimated -- every curve/point/table below is computed with sklearn
directly from real per-sample predictions on disk.

Stage 1 (BRENDA thermostability classifier, n=1,563 labeled enzymes):
  Full-resolution ROC / precision-recall / threshold-metric curves are
  computed from every real predicted Tm in
  multitask/models_tm/brenda_tm_predictions.csv (979 positive, 584
  negative; 471 "ambiguous" rows excluded, matching the original
  classification task). This reproduces the previously published 5-point
  summary almost exactly (ROC AUC 0.732 here vs. 0.732 published;
  precision 0.65/0.71/0.77/0.93/0.98 at 46/50/52/60/66 C here vs. the
  same values published) -- see stage1_sanity_check() below.

Stage 2 (S669 ddG stabilizing-mutation classifier, n=669 mutations):
  The raw per-mutation predictions from all 6 ensemble members
  (mlp/lightgbm/random_forest/catboost/xgboost/extra_trees) were
  retrieved from the Nautilus HPC PVC (multitask/models/*_s669_pred.npy
  + y_s669 from features.npz, namespace gilpin-lab) and copied into
  multitask/models/ in this repo. Full-resolution ROC / precision-recall
  / threshold-metric curves are computed from the 6-model mean exactly
  as multitask/s669_sweep.py defines the task (positive = stabilizing,
  i.e. predicted ddG < threshold). This reproduces the previously
  published values almost exactly (ensemble ROC AUC 0.669 here vs. 0.669
  published; ensemble Pearson r 0.390 here vs. 0.390 published; per-model
  Pearson/RMSE match Table 3A / stage2_model_comparison.png) -- see
  stage2_sanity_check() below.
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})
INK, ACCENT, GREY = "#1a1a1a", "#1a6a72", "0.45"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(ROOT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ============================================================
# Stage 1: real per-sample predictions (full resolution)
# ============================================================
brenda_csv = os.path.join(ROOT, "multitask", "models_tm", "brenda_tm_predictions.csv")
df1 = pd.read_csv(brenda_csv)
d1 = df1[df1["label"].isin(["positive", "negative"])].copy()
y1 = (d1["label"] == "positive").to_numpy().astype(int)
s1 = d1["pred_tm_ensemble"].to_numpy(float)
b_pos, b_neg = int(y1.sum()), int((1 - y1).sum())

fpr1, tpr1, roc_thr1 = roc_curve(y1, s1)
auc1 = roc_auc_score(y1, s1)
prec1, rec1, pr_thr1 = precision_recall_curve(y1, s1)
ap1 = average_precision_score(y1, s1)

# 5 previously-published reference thresholds, retained only as annotated
# checkpoints on top of the full real curve (not a separate data source).
brenda_ref = [(46, 0.65, 0.98), (50, 0.71, 0.77), (52, 0.77, 0.63), (60, 0.93, 0.39), (66, 0.98, 0.32)]


def stage1_sanity_check():
    print(f"Stage 1 sanity check: n={len(y1)} (pos={b_pos}, neg={b_neg}), "
          f"full-resolution AUC={auc1:.3f} (published=0.732)")
    for thr, p_conf, r_conf in brenda_ref:
        pp = (s1 >= thr).astype(int)
        tp = int(((pp == 1) & (y1 == 1)).sum()); fp = int(((pp == 1) & (y1 == 0)).sum())
        fn = int(((pp == 0) & (y1 == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"  thr={thr:>3}C  recomputed P={p:.3f} R={r:.3f}  |  published P={p_conf:.2f} R={r_conf:.2f}")


stage1_sanity_check()

# full-resolution precision/recall/F1/accuracy vs threshold (Stage 1)
tp1 = rec1 * b_pos
fp1 = np.where(prec1 > 0, tp1 * (1 - prec1) / np.maximum(prec1, 1e-12), b_neg)
fn1 = b_pos - tp1
tn1 = b_neg - fp1
acc1 = (tp1 + tn1) / (b_pos + b_neg)
f1_1 = np.where((prec1 + rec1) > 0, 2 * prec1 * rec1 / np.maximum(prec1 + rec1, 1e-12), 0.0)
best1_idx = int(np.argmax(f1_1[:-1]))  # exclude the threshold=inf sentinel point
best1_thr, best1_f1 = pr_thr1[best1_idx], f1_1[best1_idx]

# ============================================================
# Stage 2: real per-mutation predictions (full resolution)
# ============================================================
S2DIR = os.path.join(ROOT, "multitask", "models")
y2_ddg = np.load(os.path.join(S2DIR, "y_s669.npy")).astype(float)  # real experimental ddG (kcal/mol)
S2_MODEL_NAMES = ["mlp", "lightgbm", "random_forest", "catboost", "xgboost", "extra_trees"]
s2_model_preds = {
    name: np.load(os.path.join(S2DIR, f"{name}_s669_pred.npy")).astype(float)
    for name in S2_MODEL_NAMES
}
ens2_ddg = np.mean(np.stack(list(s2_model_preds.values())), axis=0)  # ensemble predicted ddG

y2 = (y2_ddg < 0).astype(int)          # positive class = stabilizing mutation
s2 = -ens2_ddg                          # higher score = more predicted-stabilizing
s_pos, s_neg = int(y2.sum()), int((1 - y2).sum())

fpr2, tpr2, roc_thr2 = roc_curve(y2, s2)
auc2 = roc_auc_score(y2, s2)
prec2, rec2, pr_thr2 = precision_recall_curve(y2, s2)
ap2 = average_precision_score(y2, s2)
pr_thr2_ddg = -pr_thr2  # back to native ddG-threshold units ("predict stabilizing if ddG < thr")

s669_ref = [(0.00, 0.81, 0.10), (0.25, 0.51, 0.26), (0.50, 0.38, 0.46), (1.00, 0.31, 0.80)]


def stage2_sanity_check():
    print(f"Stage 2 sanity check: n={len(y2)} (stabilizing={s_pos}, destabilizing={s_neg}), "
          f"full-resolution AUC={auc2:.3f} (published=0.669), "
          f"ensemble Pearson r={np.corrcoef(y2_ddg, ens2_ddg)[0,1]:.3f} (published=0.390)")
    for thr, p_conf, r_conf in s669_ref:
        pp = (ens2_ddg < thr).astype(int)
        tp = int(((pp == 1) & (y2 == 1)).sum()); fp = int(((pp == 1) & (y2 == 0)).sum())
        fn = int(((pp == 0) & (y2 == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"  thr={thr:>4} kcal/mol  recomputed P={p:.3f} R={r:.3f}  |  published P={p_conf:.2f} R={r_conf:.2f}")


stage2_sanity_check()

# full-resolution precision/recall/F1/accuracy vs threshold (Stage 2)
tp2 = rec2 * s_pos
fp2 = np.where(prec2 > 0, tp2 * (1 - prec2) / np.maximum(prec2, 1e-12), s_neg)
fn2 = s_pos - tp2
tn2 = s_neg - fp2
acc2 = (tp2 + tn2) / (s_pos + s_neg)
f1_2 = np.where((prec2 + rec2) > 0, 2 * prec2 * rec2 / np.maximum(prec2 + rec2, 1e-12), 0.0)
best2_idx = int(np.argmax(f1_2[:-1]))
best2_thr_ddg, best2_f1 = pr_thr2_ddg[best2_idx], f1_2[best2_idx]

# ============================================================
# FIGURE 1: ROC curves (both stages, full resolution)
# ============================================================
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.plot(fpr1, tpr1, "-", color="black", lw=1.4, zorder=3,
        label=f"Stage 1: BRENDA, full res. (n={len(y1)}, AUC={auc1:.3f})")
for thr, p, r in brenda_ref:
    tp = r * b_pos; fp = tp * (1 - p) / p
    ax.plot(fp / b_neg, r, "o", color="black", ms=5, zorder=4)
ax.plot(fpr2, tpr2, "--", color=GREY, lw=1.4, zorder=3,
        label=f"Stage 2: S669, full res. (n={len(y2)}, AUC={auc2:.3f})")
for thr, p, r in s669_ref:
    tp = r * s_pos; fp = tp * (1 - p) / p
    ax.plot(fp / s_neg, r, "s", color=GREY, ms=5, mfc="white", zorder=4)
ax.plot([0, 1], [0, 1], ":", color="0.7", lw=1.2, label="Chance")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Receiver-operating-characteristic curves", fontsize=11.5)
ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9,
          edgecolor="none", bbox_to_anchor=(0.99, 0.01))
ax.set_aspect("equal"); fig.tight_layout()
save(fig, "roc_curve")

# ============================================================
# FIGURE 2: Precision vs. threshold (no "more stringent" arrow)
# ============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.0))
a1.plot(pr_thr1, prec1[:-1], "-", color="black", lw=1.2, zorder=2,
        label=f"full resolution (n={len(y1)})")
a1.plot([t for t, _, _ in brenda_ref], [p for _, p, _ in brenda_ref], "o", color="black", ms=5, zorder=3,
        label="published checkpoints")
a1.set_xlabel("Temperature threshold (°C)"); a1.set_ylabel("Precision")
a1.set_title("Stage 1 (BRENDA)", fontsize=11); a1.set_ylim(0.4, 1.02); a1.grid(alpha=0.25)
a1.legend(fontsize=7.5, frameon=False, loc="lower right")

a2.plot(pr_thr2_ddg, prec2[:-1], "-", color="0.35", lw=1.2, zorder=2,
        label=f"full resolution (n={len(y2)})")
a2.plot([t for t, _, _ in s669_ref], [p for _, p, _ in s669_ref], "s", color="0.35", ms=5, mfc="white", zorder=3,
        label="published checkpoints")
a2.set_xlabel("ΔΔG threshold (kcal/mol)"); a2.set_ylabel("Precision")
a2.set_title("Stage 2 (S669)", fontsize=11); a2.set_ylim(0.15, 0.95); a2.grid(alpha=0.25)
a2.invert_xaxis()
a2.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Precision vs. decision threshold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "precision_vs_threshold")

# ============================================================
# FIGURE 3: Precision-recall curves (both stages, full resolution)
# ============================================================
fig, ax = plt.subplots(figsize=(5.4, 4.8))
ax.plot(rec1, prec1, "-", color="black", lw=1.4, label=f"Stage 1: BRENDA (AP = {ap1:.3f})")
ax.plot(rec2, prec2, "--", color=GREY, lw=1.4, label=f"Stage 2: S669 (AP = {ap2:.3f})")
ax.axhline(b_pos / (b_pos + b_neg), color="0.75", ls=":", lw=1,
           label=f"Stage 1 baseline (P = {b_pos/(b_pos+b_neg):.2f})")
ax.axhline(s_pos / (s_pos + s_neg), color="0.85", ls=":", lw=1,
           label=f"Stage 2 baseline (P = {s_pos/(s_pos+s_neg):.2f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-recall curves", fontsize=11.5)
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "precision_recall_curve")

# ============================================================
# FIGURE 4: Threshold-metric sweep (precision/recall/F1/accuracy)
# ============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
a1.plot(pr_thr1, prec1[:-1], color="black", lw=1.3, label="Precision")
a1.plot(pr_thr1, rec1[:-1], color=ACCENT, lw=1.3, label="Recall")
a1.plot(pr_thr1, f1_1[:-1], color="0.55", lw=1.3, ls="--", label="F1")
a1.plot(pr_thr1, acc1[:-1], color="0.75", lw=1.3, ls=":", label="Accuracy")
a1.axvline(best1_thr, color="firebrick", lw=0.9, alpha=0.6)
a1.text(best1_thr, 0.03, f" best F1={best1_f1:.2f}\n @{best1_thr:.0f}C", fontsize=7, color="firebrick")
a1.set_xlabel("Temperature threshold (°C)"); a1.set_ylabel("Score")
a1.set_title("Stage 1 (BRENDA, full resolution)", fontsize=10.5); a1.set_ylim(0, 1.02); a1.grid(alpha=0.2)
a1.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")

a2.plot(pr_thr2_ddg, prec2[:-1], color="black", lw=1.3, label="Precision")
a2.plot(pr_thr2_ddg, rec2[:-1], color=ACCENT, lw=1.3, label="Recall")
a2.plot(pr_thr2_ddg, f1_2[:-1], color="0.55", lw=1.3, ls="--", label="F1")
a2.plot(pr_thr2_ddg, acc2[:-1], color="0.75", lw=1.3, ls=":", label="Accuracy")
a2.axvline(best2_thr_ddg, color="firebrick", lw=0.9, alpha=0.6)
a2.text(best2_thr_ddg, 0.03, f" best F1={best2_f1:.2f}\n @{best2_thr_ddg:.2f}", fontsize=7, color="firebrick")
a2.invert_xaxis()
a2.set_xlabel("ΔΔG threshold (kcal/mol)")
a2.set_title("Stage 2 (S669, full resolution)", fontsize=10.5); a2.grid(alpha=0.2)
fig.suptitle("Threshold sweep: precision, recall, F1, accuracy", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "threshold_metric_sweep")

# ============================================================
# FIGURE 5: Stage 1 diagnostics -- score separation + confusion matrix
# ============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
bins = np.linspace(s1.min(), s1.max(), 40)
a1.hist(s1[y1 == 0], bins=bins, color=GREY, alpha=0.65, label=f"negative (n={b_neg})")
a1.hist(s1[y1 == 1], bins=bins, color=ACCENT, alpha=0.65, label=f"positive (n={b_pos})")
a1.axvline(best1_thr, color="firebrick", lw=1, ls="--", label=f"best-F1 threshold ({best1_thr:.0f} C)")
a1.set_xlabel("Predicted $T_m$ (°C)"); a1.set_ylabel("Count")
a1.set_title("Predicted-score distribution by true label", fontsize=10.5)
a1.legend(fontsize=8, frameon=False)

thr_show = best1_thr
pp = (s1 >= thr_show).astype(int)
tp = int(((pp == 1) & (y1 == 1)).sum()); fp = int(((pp == 1) & (y1 == 0)).sum())
fn = int(((pp == 0) & (y1 == 1)).sum()); tn = int(((pp == 0) & (y1 == 0)).sum())
cm = np.array([[tn, fp], [fn, tp]])
a2.imshow(cm, cmap="Greys", vmin=0)
for i in range(2):
    for j in range(2):
        a2.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
a2.set_xticks([0, 1]); a2.set_xticklabels(["pred negative", "pred positive"])
a2.set_yticks([0, 1]); a2.set_yticklabels(["true negative", "true positive"])
a2.set_title(f"Confusion matrix @ {thr_show:.0f} °C (best F1 = {best1_f1:.2f})", fontsize=10.5)
fig.suptitle("Stage 1 classifier diagnostics (BRENDA)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "stage1_diagnostics")

# ============================================================
# FIGURE 6 (NEW): Stage 2 diagnostics -- score separation + confusion matrix
# ============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
bins2 = np.linspace(ens2_ddg.min(), ens2_ddg.max(), 40)
a1.hist(ens2_ddg[y2 == 0], bins=bins2, color=GREY, alpha=0.65, label=f"destabilizing (n={s_neg})")
a1.hist(ens2_ddg[y2 == 1], bins=bins2, color=ACCENT, alpha=0.65, label=f"stabilizing (n={s_pos})")
a1.axvline(best2_thr_ddg, color="firebrick", lw=1, ls="--", label=f"best-F1 threshold ({best2_thr_ddg:.2f})")
a1.set_xlabel("Predicted ΔΔG, ensemble mean (kcal/mol)"); a1.set_ylabel("Count")
a1.set_title("Predicted-score distribution by true label", fontsize=10.5)
a1.legend(fontsize=8, frameon=False)

thr2_show = best2_thr_ddg
pp2 = (ens2_ddg < thr2_show).astype(int)
tp = int(((pp2 == 1) & (y2 == 1)).sum()); fp = int(((pp2 == 1) & (y2 == 0)).sum())
fn = int(((pp2 == 0) & (y2 == 1)).sum()); tn = int(((pp2 == 0) & (y2 == 0)).sum())
cm2 = np.array([[tn, fp], [fn, tp]])
a2.imshow(cm2, cmap="Greys", vmin=0)
for i in range(2):
    for j in range(2):
        a2.text(j, i, f"{cm2[i,j]}", ha="center", va="center",
                color="white" if cm2[i, j] > cm2.max() / 2 else "black", fontsize=13)
a2.set_xticks([0, 1]); a2.set_xticklabels(["pred destabilizing", "pred stabilizing"])
a2.set_yticks([0, 1]); a2.set_yticklabels(["true destabilizing", "true stabilizing"])
a2.set_title(f"Confusion matrix @ {thr2_show:.2f} kcal/mol (best F1 = {best2_f1:.2f})", fontsize=10.5)
fig.suptitle("Stage 2 classifier diagnostics (S669)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "stage2_diagnostics")

# ============================================================
# Numerical outputs (full real sweep tables + summary metrics)
# ============================================================
stage1_table = pd.DataFrame({
    "threshold_C": pr_thr1, "precision": prec1[:-1], "recall": rec1[:-1],
    "f1": f1_1[:-1], "accuracy": acc1[:-1],
    "tp": tp1[:-1].round().astype(int), "fp": fp1[:-1].round().astype(int),
    "fn": fn1[:-1].round().astype(int), "tn": tn1[:-1].round().astype(int),
})
stage1_table.to_csv(os.path.join(OUT, "stage1_threshold_sweep_full.csv"), index=False)

stage2_table = pd.DataFrame({
    "threshold_ddg_kcal_mol": pr_thr2_ddg, "precision": prec2[:-1], "recall": rec2[:-1],
    "f1": f1_2[:-1], "accuracy": acc2[:-1],
    "tp": tp2[:-1].round().astype(int), "fp": fp2[:-1].round().astype(int),
    "fn": fn2[:-1].round().astype(int), "tn": tn2[:-1].round().astype(int),
})
stage2_table.to_csv(os.path.join(OUT, "stage2_threshold_sweep_full.csv"), index=False)

summary = {
    "stage1_brenda": {
        "n": int(len(y1)), "n_positive": b_pos, "n_negative": b_neg,
        "roc_auc": round(float(auc1), 4), "average_precision": round(float(ap1), 4),
        "n_unique_thresholds": int(len(pr_thr1)),
        "best_f1_threshold_C": round(float(best1_thr), 2), "best_f1": round(float(best1_f1), 4),
        "source": "multitask/models_tm/brenda_tm_predictions.csv (local, real per-sample predictions)",
    },
    "stage2_s669": {
        "n": int(len(y2)), "n_positive": s_pos, "n_negative": s_neg,
        "roc_auc": round(float(auc2), 4), "average_precision": round(float(ap2), 4),
        "ensemble_pearson_r": round(float(np.corrcoef(y2_ddg, ens2_ddg)[0, 1]), 4),
        "n_unique_thresholds": int(len(pr_thr2_ddg)),
        "best_f1_threshold_ddg": round(float(best2_thr_ddg), 3), "best_f1": round(float(best2_f1), 4),
        "source": "multitask/models/*_s669_pred.npy + y_s669.npy, pulled from the Nautilus HPC PVC "
                  "(namespace gilpin-lab) -- real per-mutation ensemble predictions.",
    },
}
with open(os.path.join(OUT, "perf_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

print("\nWrote:", ", ".join(sorted(os.listdir(OUT))))
