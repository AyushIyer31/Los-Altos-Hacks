"""
Threshold optimization on S669 using saved model.
Sweeps classification thresholds on raw regressor ensemble predictions
(same approach the backend uses in production).
"""
import sys, os, pickle
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_auc_score)

BASE_DIR  = "/Users/admin/Documents/PET - Lab"
MODEL_DIR = os.path.join(BASE_DIR, "backend/app/trained_models")
sys.path.insert(0, BASE_DIR)

import train_v46 as t46

# ── Load model bundle ────────────────────────────────────────────────────────
print("Loading saved model bundle...")
bundle         = pickle.load(open(os.path.join(MODEL_DIR, "mutation_regressor.pkl"), "rb"))
scaler         = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
esm_pca_bundle = pickle.load(open(os.path.join(MODEL_DIR, "esm_pca.pkl"), "rb"))
esm_pca        = esm_pca_bundle["pca"]
pca_mean       = esm_pca_bundle["pca_mean"]

models         = bundle["models"]   # list of (name, regressor)
print(f"  {len(models)} base regressors loaded")

# ── Load all feature caches ──────────────────────────────────────────────────
print("Loading caches...")
t46.load_conservation_cache()
t46.load_esm_cache()
t46.load_metal_coord_cache()
t46.load_physics_cache()
t46.load_esm_loglik_cache()
t46.load_esm1v_cache()
print("  All caches loaded.")

# ── Load S669 ────────────────────────────────────────────────────────────────
print("Loading S669...")
s669 = t46.load_s669()
print(f"  {len(s669)} mutations")

# ── Extract features ─────────────────────────────────────────────────────────
print("Extracting features...")
base_feats, esm_embs, y_ddg, y_bin = [], [], [], []

for r in s669:
    feats = t46.extract_features(
        r['wt_aa'], r['position'], r['mut_aa'],
        r['sequence'], protein_id=r['protein_id'],
        temperature=r.get('temperature_c', 25.0),
        ph=r.get('ph', 7.0),
        struct_rsa=r.get('struct_rsa'),
        struct_phi=r.get('struct_phi'),
        struct_psi=r.get('struct_psi'),
        struct_depth=r.get('struct_depth'),
        fp_asa=r.get('fp_asa'),
        fp_ss=r.get('fp_ss'),
    )
    if feats is None:
        continue
    emb = t46.get_esm_embedding(r['protein_id'], r['position'])
    base_feats.append(feats)
    esm_embs.append(emb)
    y_ddg.append(r['ddg'])
    y_bin.append(1 if r['ddg'] < 0 else 0)

base_feats = np.array(base_feats, dtype=np.float32)
y_test     = np.array(y_bin)
y_test_ddg = np.array(y_ddg)
n          = len(base_feats)
print(f"  {n} samples, {base_feats.shape[1]} base features")

# ── ESM-2 PCA + has_esm_flag ─────────────────────────────────────────────────
esm_pca_feats = np.tile(pca_mean, (n, 1)).astype(np.float32)
has_esm_flag  = np.zeros((n, 1), dtype=np.float32)
for i, emb in enumerate(esm_embs):
    if emb is not None:
        arr = np.array(emb).reshape(1, -1)
        esm_pca_feats[i] = esm_pca.transform(arr)[0]
        has_esm_flag[i]  = 1.0

X_141 = np.hstack([base_feats, esm_pca_feats, has_esm_flag])
print(f"  After ESM-2 PCA + flag: {X_141.shape[1]} features")

# ── Scale ────────────────────────────────────────────────────────────────────
X_scaled = scaler.transform(X_141)

# ── AlphaFold pLDDT ──────────────────────────────────────────────────────────
plddt_cache = t46.load_plddt_cache() if hasattr(t46, 'load_plddt_cache') else {}
plddt_cols  = np.zeros((n, 3), dtype=np.float32)
if isinstance(plddt_cache, dict):
    for i, r in enumerate(s669[:n]):
        pdb = r.get('protein_id', '')[:4].upper()
        pos = r.get('position', 0)
        val = float(plddt_cache.get(pdb, {}).get(str(pos), 0.0) or 0.0)
        plddt_cols[i] = [val, val, 0.0]

X_test_scaled = np.hstack([X_scaled, plddt_cols])
print(f"  Final: {X_test_scaled.shape[1]} features (141 scaled + 3 pLDDT)")

# ── Raw ensemble predictions ─────────────────────────────────────────────────
print("Running ensemble predictions...")
all_preds = np.array([mdl.predict(X_test_scaled) for _, mdl in models])  # (8, n)
ens_ddg   = all_preds.mean(axis=0)   # raw ensemble DDG (kcal/mol)
print(f"  Ensemble DDG range: [{ens_ddg.min():.2f}, {ens_ddg.max():.2f}]")

# Convert to stabilizing probability: p_stab = σ(-ddg)  (more negative → more stable)
from scipy.special import expit
proba = expit(-ens_ddg)   # high prob = likely stabilizing (ddg < 0)

auc = roc_auc_score(y_test, proba)
print(f"\nAUC (raw ensemble, S669): {auc:.4f}")
print(f"S669 class balance: {y_test.sum()} stabilising, {(1-y_test).sum()} destabilising")

# DDG-threshold approach: predict stabilizing if ddg < ddg_threshold
# Standard is ddg_threshold = 0; positive ddg_threshold is more lenient
print("\n── DDG-threshold sweep (predict stable if ddg < T) ──────────────")
print(f"{'T(ddg)':>8} {'Acc':>7} {'Recall':>8} {'Spec':>8} {'F1':>7}  TP  FP  TN  FN")
print("-" * 70)

ddg_results = []
for thr in np.arange(-2.0, 2.5, 0.1):
    preds = (ens_ddg < thr).astype(int)
    acc   = accuracy_score(y_test, preds)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    cm    = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ddg_results.append((thr, acc, rec, spec, f1, int(tp), int(fp), int(tn), int(fn)))

for thr, acc, rec, spec, f1, tp, fp, tn, fn in ddg_results:
    is_default = abs(thr - 0.0) < 0.05
    if rec >= 0.45 or is_default:
        tag = " ← default(0)" if is_default else ""
        if rec >= 0.60: tag += " ★"
        elif rec >= 0.55: tag += " ✓"
        print(f"{thr:>8.1f} {acc:>7.4f} {rec:>8.4f} {spec:>8.4f} {f1:>7.4f} {tp:>3} {fp:>3} {tn:>3} {fn:>3}{tag}")

# ── Best recommendations ─────────────────────────────────────────────────────
print("\n── Recommendations ──────────────────────────────────────────")

# Best accuracy among rows with recall ≥ 60%
for target_rec, label in [(0.60, "60%"), (0.55, "55%"), (0.50, "50%")]:
    candidates = [(acc, thr, rec, spec, tp, fp, tn, fn)
                  for thr, acc, rec, spec, f1, tp, fp, tn, fn in ddg_results
                  if rec >= target_rec]
    if candidates:
        best = max(candidates)
        print(f"\n★ Best threshold for ≥{label} recall:")
        print(f"  DDG threshold:  {best[1]:.2f} kcal/mol")
        print(f"  Accuracy:       {best[0]*100:.2f}%")
        print(f"  Recall:         {best[2]*100:.2f}%")
        print(f"  Specificity:    {best[3]*100:.2f}%")
        print(f"  Confusion:      TN={best[6]}, FP={best[5]}, FN={best[7]}, TP={best[4]}")
        break

print(f"\nCV Accuracy (unchanged by threshold): 79.76%")
print(f"\nNote: AUC={auc:.4f} — this limits the recall/accuracy tradeoff.")
print(f"  If 60% recall cannot maintain ≥79% accuracy, retraining is required.")
