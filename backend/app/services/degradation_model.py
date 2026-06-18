"""Inference for the PET-degradation-efficiency regressor.

Loads the HistGradientBoosting model trained on Erickson et al. 2022 and
predicts uM aromatic products (TPA+MHET+BHET) released for a given
(sequence, temperature, pH, substrate). This is real, experimentally-grounded
degradation -- the replacement for the frontend's thermal-Gaussian "efficiency".
"""
import json
import os

import numpy as np
import joblib

from .degradation_features import build_feature_vector

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "degradation_regressor.pkl")
META_PATH = os.path.join(MODEL_DIR, "degradation_model_meta.json")

_model = None
_meta = None

# --- % depolymerization conversion (Erickson 2022 assay) ---
# Standard screen: 2.9% (w/v) PET = 29 g/L = 29,000 mg/L substrate.
# % depolymerization = mol aromatic released / mol PET repeat units x 100.
# Aromatic products (mg/L) -> mol via TPA-equivalent MW (intermediates MHET/BHET
# are slightly heavier, so this is a clearly-labelled approximation).
PET_LOADING_MG_PER_L = 29000.0
PET_REPEAT_MW = 192.17     # g/mol, PET monomer (ethylene terephthalate) unit
TPA_MW = 166.13            # g/mol, terephthalic acid (fully-hydrolysed product)


def percent_depolymerization(products_mg_per_L: float) -> float:
    """Convert mg/L aromatic products to approx % of PET depolymerized."""
    mol_aromatic = products_mg_per_L / TPA_MW
    mol_pet = PET_LOADING_MG_PER_L / PET_REPEAT_MW
    return round(max(0.0, mol_aromatic / mol_pet * 100.0), 4)


def _load():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def predict_degradation(sequence: str, temp_C: float, pH: float,
                        substrate: str = "amorphous_film",
                        crystallinity_pct: float = 0.0) -> float:
    """Predicted mg/L aromatic products released (>= 0)."""
    model, _ = _load()
    x = build_feature_vector(sequence, temp_C, pH, substrate, crystallinity_pct).reshape(1, -1)
    return float(max(0.0, np.expm1(model.predict(x)[0])))


def temperature_profile(sequence: str, pH: float = 7.5,
                        substrate: str = "amorphous_film",
                        crystallinity_pct: float = 0.0,
                        t_min: int = 30, t_max: int = 70, step: int = 10):
    """Degradation vs temperature curve.

    Returns list of {temp_C, products_mg_per_L, percent_depolymerized}.
    """
    model, _ = _load()
    temps = list(range(t_min, t_max + 1, step))
    X = np.vstack([
        build_feature_vector(sequence, t, pH, substrate, crystallinity_pct) for t in temps
    ])
    preds = np.maximum(0.0, np.expm1(model.predict(X)))
    return [{"temp_C": t,
             "products_mg_per_L": round(float(p), 3),
             "percent_depolymerized": percent_depolymerization(float(p))}
            for t, p in zip(temps, preds)]


def model_info() -> dict:
    _, meta = _load()
    return {
        "model": meta.get("model"),
        "trained_on": meta.get("source"),
        "n_rows": meta.get("n_rows"),
        "n_enzymes": meta.get("n_enzymes"),
        "r2_known_enzymes": meta.get("cv_random_known_enzymes", {}).get("r2_mean"),
        "r2_unseen_enzymes": meta.get("cv_groupkfold_unseen_enzymes", {}).get("r2_mean"),
        "temp_range_C": meta.get("temp_range_C"),
        "pH_range": meta.get("pH_range"),
        "substrates": meta.get("substrates"),
        "units": "mg/L aromatic products (TPA+MHET+BHET) released",
        "substrate_loading_mg_per_L": PET_LOADING_MG_PER_L,
        "percent_basis": "mol aromatic / mol PET repeat units (TPA-equivalent) at 2.9% loading",
    }
