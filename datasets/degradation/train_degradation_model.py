"""Train a PET-degradation-efficiency regressor on the Erickson 2022 dataset.

Target : log1p(aromatic_products_uM)  -- real PET hydrolysis readout
Inputs : sequence composition/physicochemistry + temp_C + pH + substrate

Two honest CV schemes are reported:
  * random KFold   -> condition interpolation for KNOWN enzymes (optimistic)
  * GroupKFold(enz)-> generalization to UNSEEN enzymes (the hard, honest metric)

Saves model + metadata into backend/app/trained_models/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "backend"))
from app.services.degradation_features import build_feature_vector, FEATURE_NAMES  # noqa: E402

DATA = HERE / "erickson2022_degradation.csv"
MODEL_OUT = REPO / "backend" / "app" / "trained_models" / "degradation_regressor.pkl"
META_OUT = REPO / "backend" / "app" / "trained_models" / "degradation_model_meta.json"


def build_matrix(df):
    X = np.vstack([
        build_feature_vector(r.sequence, r.temp_C, r.pH, r.substrate, r.crystallinity_pct)
        for r in df.itertuples()
    ])
    y = np.log1p(df["aromatic_products_mg_per_L"].values)  # compress heavy right tail
    groups = df["enzyme_id"].values
    return X, y, groups


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0,
    )


def cv_report(X, y, groups, splitter, name):
    r2s, maes = [], []
    for tr, te in splitter:
        m = make_model().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        # MAE reported in mg/L (back-transformed) for interpretability
        maes.append(mean_absolute_error(np.expm1(y[te]), np.expm1(pred)))
    print(f"  {name:28s} R2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}"
          f" | MAE = {np.mean(maes):.2f} mg/L")
    return {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "mae_mg_per_L_mean": float(np.mean(maes))}


def main():
    df = pd.read_csv(DATA)
    X, y, groups = build_matrix(df)
    print(f"Dataset: {len(df)} rows, {df['enzyme_id'].nunique()} enzymes, "
          f"{X.shape[1]} features")

    print("\nCross-validation:")
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rand = cv_report(X, y, groups, kf.split(X), "random KFold (known enz)")
    gkf = GroupKFold(n_splits=5)
    grp = cv_report(X, y, groups, gkf.split(X, y, groups), "GroupKFold (unseen enz)")

    # Final model trained on all data
    model = make_model().fit(X, y)
    joblib.dump(model, MODEL_OUT)

    meta = {
        "model": "HistGradientBoostingRegressor",
        "target": "log1p(aromatic_products_mg_per_L)",
        "n_rows": int(len(df)),
        "n_enzymes": int(df["enzyme_id"].nunique()),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "cv_random_known_enzymes": rand,
        "cv_groupkfold_unseen_enzymes": grp,
        "temp_range_C": [float(df.temp_C.min()), float(df.temp_C.max())],
        "pH_range": [float(df.pH.min()), float(df.pH.max())],
        "substrates": sorted(df.substrate.unique().tolist()),
        "units": "mg/L aromatic products (TPA+MHET+BHET) released",
        "substrate_loading_mg_per_L": 29000,
        "source": "Erickson et al. 2022, Nat Commun 13:7850 (10.1038/s41467-022-35237-x)",
        "note": ("Predicts mg/L aromatic products (TPA+MHET+BHET) released at "
                 "2.9% (29 g/L) PET loading, 0.7 mg enzyme/g PET, 96 h. "
                 "GroupKFold metric reflects generalization to novel sequences; "
                 "small-enzyme-count data, treat unseen-enzyme predictions as indicative."),
    }
    META_OUT.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved meta  -> {META_OUT}")


if __name__ == "__main__":
    main()
