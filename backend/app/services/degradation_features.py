"""Feature builder for the PET-degradation regressor.

Shared by the training script and the inference service so features are
computed identically in both places. Pure-python + numpy; no ESM/structure
dependency, so it runs anywhere the API runs.

Features = sequence composition/physicochemistry + reaction conditions.
"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}

AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

SUBSTRATES = ["amorphous_film", "amorphous_powder", "crystalline_powder"]

# Order matters — must stay fixed once the model is trained.
FEATURE_NAMES = (
    [f"aa_{a}" for a in AA]
    + ["seq_len", "frac_aromatic", "frac_positive", "frac_negative",
       "net_charge_frac", "gravy"]
    + ["temp_C", "pH", "crystallinity_pct"]
    + [f"substrate_{s}" for s in SUBSTRATES]
)


def sequence_features(sequence: str) -> dict:
    seq = "".join(c for c in str(sequence).upper() if c in AA_INDEX)
    n = max(len(seq), 1)
    comp = np.zeros(len(AA))
    for c in seq:
        comp[AA_INDEX[c]] += 1
    comp /= n
    feats = {f"aa_{a}": comp[i] for i, a in enumerate(AA)}
    feats["seq_len"] = float(len(seq))
    feats["frac_aromatic"] = sum(1 for c in seq if c in AROMATIC) / n
    feats["frac_positive"] = sum(1 for c in seq if c in POSITIVE) / n
    feats["frac_negative"] = sum(1 for c in seq if c in NEGATIVE) / n
    feats["net_charge_frac"] = (
        sum(1 for c in seq if c in POSITIVE) - sum(1 for c in seq if c in NEGATIVE)
    ) / n
    feats["gravy"] = sum(KD.get(c, 0.0) for c in seq) / n
    return feats


def build_feature_vector(sequence: str, temp_C: float, pH: float,
                         substrate: str = "amorphous_film",
                         crystallinity_pct: float = 0.0) -> np.ndarray:
    """Return a feature vector aligned to FEATURE_NAMES."""
    feats = sequence_features(sequence)
    feats["temp_C"] = float(temp_C)
    feats["pH"] = float(pH)
    feats["crystallinity_pct"] = float(crystallinity_pct)
    for s in SUBSTRATES:
        feats[f"substrate_{s}"] = 1.0 if substrate == s else 0.0
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)
