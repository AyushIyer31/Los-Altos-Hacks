"""Train the production model using FireProtDB + AlphaFold pLDDT features.

Run this script once to generate the trained model files.
The app will load the cached model at startup.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import pandas as pd, numpy as np, json, urllib.request, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from app.services import amino_acid_props as aap

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')
SS_MAP = {'H':0,'E':1,'L':2,'G':2,'S':2,'T':2,'B':1}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "app", "trained_models")

# ── AlphaFold pLDDT fetcher ──
plddt_cache = {}
PLDDT_CACHE_PATH = os.path.join(MODEL_DIR, "plddt_cache.json")

def load_plddt_cache():
    global plddt_cache
    if os.path.exists(PLDDT_CACHE_PATH):
        with open(PLDDT_CACHE_PATH) as f:
            plddt_cache = json.load(f)
        print(f"  Loaded pLDDT cache: {len(plddt_cache)} proteins")

def save_plddt_cache():
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PLDDT_CACHE_PATH, 'w') as f:
        json.dump(plddt_cache, f)

def fetch_plddt(uniprot_id):
    if uniprot_id in plddt_cache:
        return
    if not uniprot_id or uniprot_id == 'nan':
        plddt_cache[uniprot_id] = {}
        return
    try:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data and 'pdbUrl' in data[0]:
                with urllib.request.urlopen(data[0]['pdbUrl'], timeout=10) as pdb_resp:
                    scores = {}
                    for line in pdb_resp.read().decode().split('\n'):
                        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                            scores[int(line[22:26].strip())] = float(line[60:66].strip())
                    plddt_cache[uniprot_id] = scores
            else:
                plddt_cache[uniprot_id] = {}
    except:
        plddt_cache[uniprot_id] = {}


def extract_features(wt, mut, rsa=None, ss=None, bf=None, cons=None, in_cat=False, plddt=None):
    """44-feature vector: biochemical (27) + structural (7) + AlphaFold (3) + interactions (7)."""
    f = aap.feature_vector_v2(wt, mut)
    f.append(rsa if rsa is not None else 0.5)
    f.append(1.0 if ss == 0 else 0.0)
    f.append(1.0 if ss == 1 else 0.0)
    f.append(1.0 if ss == 2 else 0.0)
    f.append(bf if bf is not None else 20.0)
    f.append(cons if cons is not None else 5.0)
    f.append(1.0 if in_cat else 0.0)
    plddt_val = plddt / 100.0 if plddt is not None else 0.7
    f.append(plddt_val)
    f.append(1.0 if plddt is not None and plddt < 50 else 0.0)
    f.append(1.0 if plddt is not None and plddt > 90 else 0.0)
    hd = abs(aap.HYDROPHOBICITY.get(mut, 0) - aap.HYDROPHOBICITY.get(wt, 0))
    sd = abs(aap.SIZE.get(mut, 0) - aap.SIZE.get(wt, 0))
    cd = abs(aap.CHARGE.get(mut, 0) - aap.CHARGE.get(wt, 0))
    burial = 1.0 - (rsa if rsa is not None else 0.5)
    f.extend([hd * burial, sd * burial, cd * burial, hd * (1.0 if in_cat else 0.0)])
    f.append(hd * plddt_val)
    f.append(sd * plddt_val)
    f.append(burial * plddt_val)
    cons_val = (cons if cons is not None else 5.0) / 9.0
    f.append(hd * cons_val)
    f.append(sd * cons_val)
    return f


if __name__ == "__main__":
    print("=== Training Production Model ===\n")

    # Load pLDDT cache
    load_plddt_cache()

    # Fetch AlphaFold pLDDT for all proteins
    df = pd.read_csv('fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv')
    uids = list(set(str(u).strip() for u in df['uniprot_id'].dropna().unique() if str(u).strip() != 'nan'))
    new_fetches = [u for u in uids if u not in plddt_cache]
    if new_fetches:
        print(f"Fetching AlphaFold pLDDT for {len(new_fetches)} new proteins...")
        for i, uid in enumerate(new_fetches):
            fetch_plddt(uid)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(new_fetches)} done")
        save_plddt_cache()
    fetched = sum(1 for u in uids if plddt_cache.get(u))
    print(f"AlphaFold pLDDT available for {fetched}/{len(uids)} proteins\n")

    # Build training set
    X, y = [], []
    for _, r in df.iterrows():
        wt, mut, ddg = str(r['wild_type']).upper(), str(r['mutation']).upper(), r['ddG']
        if wt not in VALID_AAS or mut not in VALID_AAS or wt == mut or pd.isna(ddg):
            continue
        if ddg < -1.0:
            label = 1  # stabilizing
        elif ddg > 1.5:
            label = 0  # destabilizing
        else:
            continue

        rsa = min(float(r['asa']) / 200, 1.0) if pd.notna(r.get('asa')) else None
        ss = SS_MAP.get(str(r.get('secondary_structure', '')).strip(), None)
        bf = float(r['b_factor']) if pd.notna(r.get('b_factor')) else None
        cons = float(r['conservation']) if pd.notna(r.get('conservation')) else None
        in_cat = bool(r.get('is_in_catalytic_pocket', False))
        uid = str(r.get('uniprot_id', '')).strip()
        pos = int(r['pdb_position']) if pd.notna(r.get('pdb_position')) else (
            int(r['position']) if pd.notna(r.get('position')) else None)
        plddt = plddt_cache.get(uid, {}).get(str(pos), plddt_cache.get(uid, {}).get(pos)) if pos else None

        X.append(extract_features(wt, mut, rsa, ss, bf, cons, in_cat, plddt))
        y.append(label)

    X, y = np.array(X), np.array(y)
    ns, nd = int(y.sum()), int(len(y) - y.sum())
    spw = nd / max(ns, 1)
    print(f"Dataset: {len(y)} samples ({ns} stabilizing, {nd} destabilizing)")
    print(f"Features: {X.shape[1]}")
    print(f"Scale pos weight: {spw:.2f}\n")

    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
        reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=spw, random_state=58, verbosity=0,
    )
    clf.fit(X_scaled, y)

    # Evaluate
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=40)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
    train_pred = clf.predict(X_scaled)

    print(f"Training accuracy: {accuracy_score(y, train_pred):.4f}")
    print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Training F1: {f1_score(y, train_pred):.4f}\n")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "mutation_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Model saved to backend/app/trained_models/")
    print(f"\nFINAL CV ACCURACY: {cv_scores.mean():.1%}")
