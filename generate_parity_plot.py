"""
Predicted vs Actual ΔΔG parity plot from REAL model predictions on S669.

The regression-native equivalent of an ROC curve: it shows how closely the
model's predicted ΔΔG matches the experimentally measured ΔΔG on 669 held-out
mutations the model never saw during training.

Reuses the exact 144-feature pipeline from train_v46.py.
"""

import sys, os, pickle, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
MODELS_DIR = os.path.join(BACKEND, "app", "trained_models")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load artefacts ─────────────────────────────────────────────────────────────
print("Loading artefacts...")
ensemble = pickle.load(open(os.path.join(MODELS_DIR, "mutation_regressor.pkl"), "rb"))
scaler   = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))
pca_data = pickle.load(open(os.path.join(MODELS_DIR, "esm_pca.pkl"), "rb"))
plddt_cache = pickle.load(open(os.path.join(MODELS_DIR, "plddt_pdb_cache.pkl"), "rb"))
meta = json.loads(open(os.path.join(MODELS_DIR, "model_meta.json")).read())
pca, pca_mean = pca_data["pca"], pca_data["pca_mean"]

# ── Import train_v46 feature extractors ───────────────────────────────────────
print("Importing train_v46 + caches...")
import importlib.util
spec = importlib.util.spec_from_file_location("train_v46", os.path.join(ROOT, "train_v46.py"))
t46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t46)
t46.load_conservation_cache(); t46.load_esm_cache(); t46.load_metal_coord_cache()
t46.load_physics_cache(); t46.load_esm_loglik_cache(); t46.load_esm1v_cache()

# ── Load S669 + extract features ──────────────────────────────────────────────
print("Loading S669 + extracting features...")
s669 = t46.load_s669()
base, esm_list, y_true_ddg, valid_esm = [], [], [], []
for r in s669:
    feats = t46.extract_features(r["wt_aa"], r["position"], r["mut_aa"],
                                 r.get("sequence", ""), protein_id=r.get("protein_id", ""),
                                 temperature=r.get("temperature_c", 25.0), ph=r.get("ph", 7.0))
    if feats is None:
        continue
    emb = t46.get_esm_embedding(r["protein_id"], r["position"])
    base.append(feats); esm_list.append(emb)
    if emb is not None:
        valid_esm.append(len(base) - 1)
    y_true_ddg.append(r["ddg"])

n = len(base)
X_base = np.array(base, dtype=np.float32)
esm_features = np.tile(pca_mean, (n, 1)).astype(np.float32)
has_esm = np.zeros((n, 1), dtype=np.float32)
if valid_esm:
    raw = np.array([esm_list[i] for i in valid_esm], dtype=np.float32)
    esm_features[valid_esm] = pca.transform(raw)
    has_esm[valid_esm] = 1.0
X = np.concatenate([X_base, esm_features, has_esm], axis=1)
X_scaled = scaler.transform(X)

def plddt_feat(pid, pos):
    sc = plddt_cache.get(pid)
    if sc is None:
        return (0.0, 0.0, 0.0)
    i = int(pos) - 1
    if i < 0 or i >= len(sc):
        return (0.0, 0.0, 0.0)
    return sc[i] / 100.0, float(np.mean(sc[max(0, i-5):i+6])) / 100.0, float(sc[i] < 50.0)

plddt = np.array([plddt_feat(r.get("protein_id", ""), r["position"]) for r in s669[:n]], dtype=np.float32)
X_final = np.hstack([X_scaled, plddt])

# ── Predict ───────────────────────────────────────────────────────────────────
print("Predicting...")
base_preds = [m.predict(X_final) for (_, m) in ensemble["models"]]
weights = ensemble.get("weights", [1.0 / len(base_preds)] * len(base_preds))
y_pred = np.sum([p * w for p, w in zip(base_preds, weights)], axis=0) / sum(weights)
y_true = np.array(y_true_ddg)

# ── Metrics ───────────────────────────────────────────────────────────────────
pear, _  = pearsonr(y_true, y_pred)
spear, _ = spearmanr(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
r2   = r2_score(y_true, y_pred)
# Sign agreement: does the model get the direction (stab/destab) right?
sign_acc = float(np.mean((y_true < 0) == (y_pred < meta["optimal_threshold"])))

print(f"\n{'='*52}")
print(f"REAL S669 REGRESSION RESULTS (n={n})")
print(f"{'='*52}")
print(f"  Pearson r  : {pear:.4f}")
print(f"  Spearman   : {spear:.4f}")
print(f"  MAE        : {mae:.4f} kcal/mol")
print(f"  RMSE       : {rmse:.4f} kcal/mol")
print(f"  R^2        : {r2:.4f}")
print(f"  Sign acc.  : {sign_acc*100:.1f}%")
print(f"{'='*52}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG, PANEL, TEAL, ORANGE, GREY, WHITE = "#0d1b2a", "#122538", "#00e5cc", "#f5a623", "#70708a", "#e8e8ed"
fig, ax = plt.subplots(figsize=(9, 8.5), facecolor=BG)
ax.set_facecolor(BG)
ax.tick_params(colors=GREY, labelsize=11)
for sp in ax.spines.values(): sp.set_edgecolor("#1e3248")

lim = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]

# Quadrant shading: correct-sign regions (top-right, bottom-left)
ax.axhspan(0, lim[1], xmin=0, xmax=1, color="#10B981", alpha=0.0)
ax.axhline(0, color=GREY, lw=0.8, ls=":", alpha=0.5)
ax.axvline(0, color=GREY, lw=0.8, ls=":", alpha=0.5)

# Perfect-prediction diagonal
ax.plot(lim, lim, color=ORANGE, lw=2, ls="--", label="Perfect prediction (y = x)", zorder=2)

# Scatter colored by absolute error
err = np.abs(y_true - y_pred)
sc = ax.scatter(y_true, y_pred, c=err, cmap="cool", s=28, alpha=0.75,
                edgecolors="none", zorder=3, vmin=0, vmax=np.percentile(err, 95))
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("|error| (kcal/mol)", color=GREY, fontsize=11)
cb.ax.tick_params(colors=GREY)
cb.outline.set_edgecolor("#1e3248")

ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Measured ΔΔG  (kcal/mol)", color=WHITE, fontsize=13)
ax.set_ylabel("Predicted ΔΔG  (kcal/mol)", color=WHITE, fontsize=13)
ax.set_title("Predicted vs Measured ΔΔG — S669 Independent Test Set",
             color=TEAL, fontsize=14, fontweight="bold", pad=10)

ax.text(0.03, 0.97,
        f"Stacked Ensemble (v46)\n"
        f"Pearson r   = {pear:.3f}\n"
        f"Spearman    = {spear:.3f}\n"
        f"MAE         = {mae:.2f} kcal/mol\n"
        f"RMSE        = {rmse:.2f} kcal/mol\n"
        f"R²          = {r2:.3f}\n"
        f"Sign acc.   = {sign_acc*100:.1f}%\n"
        f"n           = {n} mutations",
        transform=ax.transAxes, fontsize=11, color=WHITE, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=GREY, alpha=0.92))

ax.legend(loc="lower right", fontsize=11, facecolor=PANEL, edgecolor=GREY,
          labelcolor=WHITE, framealpha=0.95)
fig.text(0.5, 0.005,
         "Each point = one held-out mutation never seen during training. "
         "Closer to the dashed diagonal = more accurate. Color = absolute error.",
         ha="center", color=GREY, fontsize=9, style="italic")

fig.tight_layout(rect=[0, 0.02, 1, 1])
out = os.path.join(ROOT, "graph5_predicted_vs_actual.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved -> {out}")
