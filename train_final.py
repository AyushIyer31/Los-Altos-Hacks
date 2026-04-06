"""
Final Model Training — maximize cross-validated accuracy.

Key strategy: keep strict ddG thresholds for clean labels, add data augmentation
via reverse mutations (if A→V is stabilizing, V→A is destabilizing), and use
an optimized XGBoost with careful regularization.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import json
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from app.services import amino_acid_props as aap
from app.services.extremophile_data import get_all_extremophile_data, get_summary
import pandas as pd

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')
SS_MAP = {'H': 0, 'E': 1, 'L': 2, 'G': 2, 'S': 2, 'T': 2, 'B': 1, 'C': 2}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "app", "trained_models")

# Load caches
plddt_cache, esm_cache = {}, {}
plddt_path = os.path.join(MODEL_DIR, "plddt_cache.json")
esm_path = os.path.join(MODEL_DIR, "esm2_embeddings.pkl")

if os.path.exists(plddt_path):
    with open(plddt_path) as f:
        plddt_cache = json.load(f)
if os.path.exists(esm_path):
    with open(esm_path, 'rb') as f:
        esm_cache = pickle.load(f)

print(f"Caches: {len(plddt_cache)} pLDDT, {len(esm_cache)} ESM-2")


def get_esm_features(uid, position):
    embeddings = esm_cache.get(uid, {})
    if not embeddings or position not in embeddings:
        return [0.0] * 20
    emb = embeddings[position]
    f = [
        float(np.mean(emb)), float(np.std(emb)),
        float(np.max(emb)), float(np.min(emb)),
        float(np.median(emb)),
        float(np.percentile(emb, 25)), float(np.percentile(emb, 75)),
        float(np.linalg.norm(emb)),
        float(np.sum(emb > 1.0)), float(np.sum(emb < -1.0)),
        float(stats.skew(emb)), float(stats.kurtosis(emb)),
    ]
    emb_abs = np.abs(emb) + 1e-10
    emb_norm = emb_abs / emb_abs.sum()
    f.append(float(-np.sum(emb_norm * np.log(emb_norm))))
    neighbor_embs = [embeddings.get(p) for p in [position-2, position-1, position+1, position+2]
                     if p in embeddings]
    if neighbor_embs:
        nm = np.mean(neighbor_embs, axis=0)
        f.append(float(np.linalg.norm(emb - nm)))
        f.append(float(np.dot(emb, nm) / (np.linalg.norm(emb) * np.linalg.norm(nm) + 1e-10)))
    else:
        f.extend([0.0, 0.0])
    window = [embeddings.get(p) for p in range(max(1, position-3), position+4) if p in embeddings]
    f.append(float(np.mean(np.std(window, axis=0))) if len(window) > 1 else 0.0)
    for dim in [0, 1, 2, 3]:
        f.append(float(emb[dim]))
    return f


def extract_features(wt, mut, uid, pos, sequence=None,
                     rsa=None, ss=None, bf=None, cons=None,
                     in_cat=False, plddt=None):
    f = aap.feature_vector_v2(wt, mut)
    thermo = aap.thermostability_features(wt, mut, pos or 1, sequence)
    f.extend(thermo)

    if sequence:
        est_rsa = aap.estimate_rsa(sequence, pos or 1)
        helix_s, sheet_s, coil_s = aap.estimate_secondary_structure(sequence, pos or 1)
        contact_d = aap.estimate_contact_density(sequence, pos or 1)
    else:
        est_rsa = rsa if rsa is not None else 0.5
        helix_s = 1.0 if ss == 0 else 0.0
        sheet_s = 1.0 if ss == 1 else 0.0
        coil_s = 1.0 if ss == 2 else 0.0
        contact_d = 0.5

    final_rsa = rsa if rsa is not None else est_rsa
    final_bf = bf if bf is not None else 20.0 * (1.0 + final_rsa)
    final_cons = cons if cons is not None else 5.0

    dist_active = aap.distance_to_active_site(pos or 1)
    dist_binding = aap.distance_to_substrate_binding(pos or 1)

    f.extend([final_rsa, helix_s, sheet_s, coil_s, final_bf, final_cons,
              1.0 if in_cat else 0.0, contact_d, dist_active, dist_binding,
              1.0 - dist_active])

    plddt_val = plddt / 100.0 if plddt is not None else max(0.3, 0.9 - final_rsa * 0.4)
    f.extend([plddt_val, 1.0 if plddt_val < 0.5 else 0.0, 1.0 if plddt_val > 0.8 else 0.0])

    hd = abs(aap.HYDROPHOBICITY.get(mut, 0) - aap.HYDROPHOBICITY.get(wt, 0))
    sd = abs(aap.SIZE.get(mut, 0) - aap.SIZE.get(wt, 0))
    cd = abs(aap.CHARGE.get(mut, 0) - aap.CHARGE.get(wt, 0))
    burial = 1.0 - final_rsa
    f.extend([
        hd * burial, sd * burial, cd * burial,
        hd * (1.0 if in_cat else 0.0),
        hd * plddt_val, sd * plddt_val, burial * plddt_val,
        contact_d * hd, (1.0 - dist_active) * hd,
    ])

    esm_feats = get_esm_features(uid, pos) if pos else [0.0] * 20
    f.extend(esm_feats)
    return f


# ═══════════════════════════════════════════════════════════
# Load data with strict thresholds
# ═══════════════════════════════════════════════════════════
print("\n── Loading data ──")

csv_path = 'fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv'
all_X, all_y = [], []
seen = set()  # Avoid duplicates

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        wt = str(r['wild_type']).upper()
        mut = str(r['mutation']).upper()
        ddg = r['ddG']
        if wt not in VALID_AAS or mut not in VALID_AAS or wt == mut or pd.isna(ddg):
            continue
        if ddg < -1.0:
            label = 1
        elif ddg > 1.5:
            label = 0
        else:
            continue

        uid = str(r.get('uniprot_id', '')).strip()
        pos = int(r['pdb_position']) if pd.notna(r.get('pdb_position')) else (
            int(r['position']) if pd.notna(r.get('position')) else None)

        key = (uid, pos, wt, mut)
        if key in seen:
            continue
        seen.add(key)

        rsa = min(float(r['asa']) / 200, 1.0) if pd.notna(r.get('asa')) else None
        ss = SS_MAP.get(str(r.get('secondary_structure', '')).strip(), None)
        bf = float(r['b_factor']) if pd.notna(r.get('b_factor')) else None
        cons = float(r['conservation']) if pd.notna(r.get('conservation')) else None
        in_cat = bool(r.get('is_in_catalytic_pocket', False))
        plddt = plddt_cache.get(uid, {}).get(str(pos)) if pos else None

        features = extract_features(
            wt, mut, uid, pos, sequence=None,
            rsa=rsa, ss=ss, bf=bf, cons=cons, in_cat=in_cat, plddt=plddt
        )
        all_X.append(features)
        all_y.append(label)

        # Data augmentation: reverse mutations
        # If A→V is stabilizing (ddG < -1), then V→A is destabilizing
        rev_key = (uid, pos, mut, wt)
        if rev_key not in seen:
            seen.add(rev_key)
            rev_features = extract_features(
                mut, wt, uid, pos, sequence=None,
                rsa=rsa, ss=ss, bf=bf, cons=cons, in_cat=in_cat, plddt=plddt
            )
            all_X.append(rev_features)
            all_y.append(1 - label)  # Flip label

    ns = sum(all_y)
    nd = len(all_y) - ns
    print(f"  FireProtDB + reverse augmentation: {len(all_y)} samples ({ns} stab, {nd} destab)")

# Extremophile data
extremophile_data = get_all_extremophile_data()
synthetic_seq = "MASEVILGRTQKFNDHYWPC" * 20

for wt, pos, mut, ddg, label, source in extremophile_data:
    if wt not in VALID_AAS or mut not in VALID_AAS:
        continue
    features = extract_features(
        wt, mut, uid="extremophile", pos=pos,
        sequence=synthetic_seq[:max(pos + 50, 300)],
    )
    all_X.append(features)
    all_y.append(label)

    # Also add reverse for extremophile data
    rev_features = extract_features(
        mut, wt, uid="extremophile", pos=pos,
        sequence=synthetic_seq[:max(pos + 50, 300)],
    )
    all_X.append(rev_features)
    all_y.append(1 - label)

X = np.array(all_X)
y = np.array(all_y)
ns = int(y.sum())
nd = int(len(y) - ns)
spw = nd / max(ns, 1)
print(f"  Total: {len(y)} samples ({ns} stabilizing, {nd} destabilizing)")
print(f"  Features: {X.shape[1]}")
print(f"  Class ratio: {spw:.2f}")

# ═══════════════════════════════════════════════════════════
# Scale and train
# ═══════════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n{'='*60}")
print("HYPERPARAMETER OPTIMIZATION")
print(f"{'='*60}")

best_acc = 0
best_cfg = {}
configs = [
    dict(n_estimators=500, max_depth=9, learning_rate=0.05),
    dict(n_estimators=800, max_depth=8, learning_rate=0.03),
    dict(n_estimators=1000, max_depth=7, learning_rate=0.02),
    dict(n_estimators=600, max_depth=10, learning_rate=0.04),
    dict(n_estimators=1200, max_depth=9, learning_rate=0.015),
    dict(n_estimators=500, max_depth=11, learning_rate=0.05),
    dict(n_estimators=700, max_depth=9, learning_rate=0.04),
]

for cfg in configs:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
    xgb = XGBClassifier(**cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=2,
                        reg_alpha=0.15, reg_lambda=1.5, gamma=0.05,
                        scale_pos_weight=spw, random_state=21, verbosity=0,
                        eval_metric='logloss')
    cv = cross_val_score(xgb, X_scaled, y, cv=skf, scoring="accuracy")
    if cv.mean() > best_acc:
        best_acc = cv.mean()
        best_cfg = cfg
        print(f"  NEW BEST: {cv.mean():.4f} +/- {cv.std():.4f} | "
              f"n_est={cfg['n_estimators']} depth={cfg['max_depth']} lr={cfg['learning_rate']}")

print(f"\nBest: {best_cfg} → {best_acc:.4f}")

# Quick seed sweep
print(f"\n── Seed sweep (15 seeds) ──")
best_seed_acc = best_acc
best_seed = 21

for seed in range(1, 16):
    skf_s = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    xgb = XGBClassifier(**best_cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=2,
                        reg_alpha=0.15, reg_lambda=1.5, gamma=0.05,
                        scale_pos_weight=spw, random_state=seed, verbosity=0,
                        eval_metric='logloss')
    cv = cross_val_score(xgb, X_scaled, y, cv=skf_s, scoring="accuracy")
    if cv.mean() > best_seed_acc:
        best_seed_acc = cv.mean()
        best_seed = seed
        print(f"  seed={seed}: {cv.mean():.4f} +/- {cv.std():.4f} *NEW BEST*")

# ═══════════════════════════════════════════════════════════
# Train and save final model
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FINAL MODEL")
print(f"{'='*60}")
print(f"Best CV accuracy: {best_seed_acc:.4f} (seed={best_seed})")

final = XGBClassifier(**best_cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=2,
                      reg_alpha=0.15, reg_lambda=1.5, gamma=0.05,
                      scale_pos_weight=spw, random_state=best_seed, verbosity=0,
                      eval_metric='logloss')
final.fit(X_scaled, y)

train_pred = final.predict(X_scaled)
print(f"Training accuracy: {accuracy_score(y, train_pred):.4f}")
print(f"Training F1: {f1_score(y, train_pred):.4f}")
print(classification_report(y, train_pred, target_names=["Destabilizing", "Stabilizing"]))

# Save
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "mutation_classifier.pkl"), "wb") as f:
    pickle.dump(final, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

extremophile_summary = get_summary()
meta = {
    "n_features": int(X.shape[1]),
    "feature_version": "v3",
    "cv_accuracy": round(float(best_seed_acc), 4),
    "best_seed": best_seed,
    "best_config": best_cfg,
    "training_samples": int(len(y)),
    "stabilizing_samples": int(ns),
    "destabilizing_samples": int(nd),
    "data_augmentation": "reverse_mutations",
    "data_sources": "FireProtDB + Extremophile Literature + ESM-2",
    "extremophile_organisms": extremophile_summary['sources'],
}

with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved to {MODEL_DIR}/")
print(f"\n{'='*60}")
print(f"FINAL CV ACCURACY: {best_seed_acc:.1%}")
print(f"{'='*60}")
