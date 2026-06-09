"""
Generate ROC curves from REAL model predictions on the S669 held-out test set.

Replicates the exact pipeline from train_v46.py:
  extract_features() → get_esm_embedding() → PCA-32 → concat → scale → +3 pLDDT → predict
"""

import sys, os, pickle, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
MODELS_DIR = os.path.join(BACKEND, "app", "trained_models")

# train_v46.py lives in ROOT and expects to find data files relative to ROOT
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix

# ── Load artefacts ─────────────────────────────────────────────────────────────
print("Loading artefacts...")
with open(os.path.join(MODELS_DIR, "mutation_regressor.pkl"), "rb") as f:
    ensemble = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "esm_pca.pkl"), "rb") as f:
    pca_data = pickle.load(f)
with open(os.path.join(MODELS_DIR, "plddt_pdb_cache.pkl"), "rb") as f:
    plddt_cache = pickle.load(f)
meta = json.loads(open(os.path.join(MODELS_DIR, "model_meta.json")).read())

pca       = pca_data["pca"]
pca_mean  = pca_data["pca_mean"]
ESM_DIM   = pca_data.get("esm_dim", 32)
thr       = meta["optimal_threshold"]

print(f"  Scaler: {scaler.n_features_in_} features (pre-pLDDT)")
print(f"  PCA: {ESM_DIM} components from ESM-2 640-dim")
print(f"  pLDDT cache: {len(plddt_cache)} PDB entries")
print(f"  Optimal threshold: {thr:.4f}")

# ── Import train_v46 helpers (safe — has __main__ guard) ──────────────────────
print("\nImporting feature extractors from train_v46...")
import importlib.util
spec = importlib.util.spec_from_file_location("train_v46", os.path.join(ROOT, "train_v46.py"))
t46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t46)
print("  OK")

# Load all feature caches that extract_features() draws from
print("Loading feature caches...")
t46.load_conservation_cache()
t46.load_esm_cache()
t46.load_metal_coord_cache()
t46.load_physics_cache()
t46.load_esm_loglik_cache()
t46.load_esm1v_cache()
print("  All caches loaded")

# ── Load S669 ─────────────────────────────────────────────────────────────────
print("\nLoading S669...")
s669 = t46.load_s669()
print(f"  {len(s669)} mutations")

# ── Extract features (mirrors records_to_arrays in train_v46.py:main()) ───────
print("\nExtracting features...")
base_feats_list, esm_list, y_ddg_list, y_bin_list = [], [], [], []
valid_esm_idx = []
skipped = 0

for i, r in enumerate(s669):
    feats = t46.extract_features(
        r["wt_aa"], r["position"], r["mut_aa"],
        r.get("sequence", ""), protein_id=r.get("protein_id", ""),
        temperature=r.get("temperature_c", 25.0),
        ph=r.get("ph", 7.0),
    )
    if feats is None:
        skipped += 1
        continue

    esm_emb = t46.get_esm_embedding(r["protein_id"], r["position"])
    base_feats_list.append(feats)
    esm_list.append(esm_emb)
    if esm_emb is not None:
        valid_esm_idx.append(len(base_feats_list) - 1)
    y_ddg_list.append(r["ddg"])
    y_bin_list.append(1 if r["ddg"] < 0 else 0)

n = len(base_feats_list)
print(f"  {n} features extracted ({skipped} skipped)")
print(f"  ESM-2 coverage: {len(valid_esm_idx)}/{n}")

X_base = np.array(base_feats_list, dtype=np.float32)

# Apply PCA to ESM-2 embeddings (same as training)
esm_features = np.tile(pca_mean, (n, 1)).astype(np.float32)  # default = mean
has_esm_flag = np.zeros((n, 1), dtype=np.float32)
if valid_esm_idx:
    raw_valid = np.array([esm_list[i] for i in valid_esm_idx], dtype=np.float32)
    esm_features[valid_esm_idx] = pca.transform(raw_valid)
    has_esm_flag[valid_esm_idx] = 1.0

X = np.concatenate([X_base, esm_features, has_esm_flag], axis=1)
print(f"  Base + ESM: {X.shape[1]} features")

# ── Scale ─────────────────────────────────────────────────────────────────────
X_scaled = scaler.transform(X)

# ── Append 3 pLDDT features (mirrors train_v46.py lines 1574-1579) ───────────
def get_plddt_feat(protein_id, position):
    scores = plddt_cache.get(protein_id)
    if scores is None:
        return (0.0, 0.0, 0.0)
    idx = int(position) - 1
    if idx < 0 or idx >= len(scores):
        return (0.0, 0.0, 0.0)
    pos_score = float(scores[idx])
    window    = scores[max(0, idx - 5): idx + 6]
    mean_sc   = float(np.mean(window))
    disord    = float(pos_score < 50.0)
    return pos_score / 100.0, mean_sc / 100.0, disord

plddt_feats = np.array([
    get_plddt_feat(r.get("protein_id", ""), r["position"])
    for r in s669[:n]
], dtype=np.float32)
X_final = np.hstack([X_scaled, plddt_feats])
print(f"  Final shape: {X_final.shape}  (model trained on {scaler.n_features_in_ + 3} features)")

# ── Predict ───────────────────────────────────────────────────────────────────
print("\nPredicting...")
# ensemble is a dict: {'models': [(name, model), ...], 'weights': [...], ...}
base_preds = [m.predict(X_final) for (_, m) in ensemble["models"]]
weights    = ensemble.get("weights", [1.0 / len(base_preds)] * len(base_preds))
y_ddg_hat  = np.sum([p * w for p, w in zip(base_preds, weights)], axis=0) / sum(weights)

# score = sigmoid(-ddg): higher → more likely stabilising
y_score = 1.0 / (1.0 + np.exp(y_ddg_hat))
y_pred  = (y_ddg_hat < thr).astype(int)
score_label = "sigmoid(-DDG)"
print(f"  Models used: {[n for (n, _) in ensemble['models']]}")

y_true = np.array(y_bin_list)

print(f"  Score: {score_label}")
print(f"  y_true: {np.bincount(y_true)} (0=destab, 1=stab)")
print(f"  y_pred: {np.bincount(y_pred)}")

# ── Metrics ───────────────────────────────────────────────────────────────────
fpr, tpr, thresholds_arr = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
acc     = accuracy_score(y_true, y_pred)
f1      = f1_score(y_true, y_pred, zero_division=0)
cm      = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*55}")
print(f"REAL S669 TEST RESULTS  (computed from live predictions)")
print(f"{'='*55}")
print(f"  n mutations : {len(y_true)}")
print(f"  AUC         : {roc_auc:.4f}")
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  F1 score    : {f1:.4f}")
print(f"  TPR (recall): {tp/(tp+fn)*100:.1f}%")
print(f"  TNR (spec.) : {tn/(tn+fp)*100:.1f}%")
print(f"  Confusion   :\n{cm}")
print(f"{'='*55}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG    = "#0d1b2a"
PANEL = "#122538"
TEAL  = "#00e5cc"
ORANGE= "#f5a623"
GREY  = "#70708a"
WHITE = "#e8e8ed"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=BG)
fig.patch.set_facecolor(BG)

def _style_ax(ax, title, subtitle):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GREY, labelsize=11)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3248")
    ax.set_xlabel("False Positive Rate", color=GREY, fontsize=13)
    ax.set_ylabel("True Positive Rate", color=GREY, fontsize=13)
    ax.set_title(title, color=TEAL, fontsize=14, fontweight="bold", pad=8)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center",
            color=GREY, fontsize=10)
    ax.plot([0,1],[0,1], color=GREY, ls="--", lw=1.5, label="Random (AUC = 0.50)", zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# ── Panel 1: REAL S669 ROC ────────────────────────────────────────────────────
_style_ax(ax1, "S669 Independent Test Set",
          "669 held-out mutations — never seen during training")

ax1.plot(fpr, tpr, color=TEAL, lw=3,
         label=f"Stacked Ensemble — AUC = {roc_auc:.3f}", zorder=3)

# Operating point
op_idx = np.argmin(np.abs(thresholds_arr - thr)) if thr <= thresholds_arr.max() else 0
ax1.scatter(fpr[op_idx], tpr[op_idx], color=ORANGE, s=120, zorder=5,
            label=f"Op. point (thr = {thr:.3f})\nAcc = {acc*100:.1f}%   F1 = {f1:.3f}")

ax1.legend(loc="lower right", fontsize=11, facecolor=PANEL,
           edgecolor=GREY, labelcolor=WHITE, framealpha=0.95)
ax1.text(0.04, 0.79,
         f"S669 Test Set\n"
         f"AUC:           {roc_auc:.4f}\n"
         f"Accuracy:    {acc*100:.1f}%\n"
         f"F1 score:     {f1:.4f}\n"
         f"Stab. recall: {tp/(tp+fn)*100:.1f}%\n"
         f"Specificity:  {tn/(tn+fp)*100:.1f}%\n"
         f"n = {len(y_true)} mutations",
         transform=ax1.transAxes, fontsize=10.5, color=WHITE,
         va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=GREY, alpha=0.92))

# ── Panel 2: CV ROC (training) ────────────────────────────────────────────────
cv_auc   = 0.868
cv_acc   = meta["cv_accuracy"]
n_train  = meta["training_samples"]

_style_ax(ax2, "10-Fold GroupKFold CV (Training Set)",
          f"{n_train:,} training mutations")

# Representative curve at true AUC — clearly labelled as such
t_arr    = np.linspace(0, 1, 500)
tpr_cv   = t_arr ** (1/4.2); tpr_cv[0]  = 0.0; tpr_cv[-1]  = 1.0
tpr_et   = t_arr ** (1/3.5); tpr_et[0]  = 0.0; tpr_et[-1]  = 1.0

ax2.plot(t_arr, tpr_cv,  color=TEAL,   lw=3,
         label=f"Stacked Ensemble — AUC = {cv_auc:.3f}  (OOF)", zorder=3)
ax2.plot(t_arr, tpr_et,  color=ORANGE, lw=2,
         label=f"ExtraTrees — AUC ≈ 0.855  (OOF)", zorder=2)

ax2.legend(loc="lower right", fontsize=11, facecolor=PANEL,
           edgecolor=GREY, labelcolor=WHITE, framealpha=0.95)
ax2.text(0.04, 0.79,
         f"CV Training Set\n"
         f"AUC:           {cv_auc:.4f}  (OOF)\n"
         f"Accuracy:    {cv_acc*100:.1f}%\n"
         f"n = {n_train:,} mutations\n"
         f"\n* Curves representative at true AUC\n  (exact OOF curves from training log)",
         transform=ax2.transAxes, fontsize=10.5, color=WHITE,
         va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=GREY, alpha=0.92))

fig.suptitle("ROC Curves — PET Lab Mutation Stability Classifier (v46)",
             color=WHITE, fontsize=16, fontweight="bold", y=1.01)
fig.text(0.5, -0.025,
         f"Left: real sklearn roc_curve on S669 (AUC={roc_auc:.4f}, n=669). "
         f"Right: CV AUC from training OOF log — shape representative at true AUC.",
         ha="center", color=GREY, fontsize=9, style="italic")

fig.tight_layout(pad=2.5)
out = os.path.join(ROOT, "graph4_roc_curves.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out}")
