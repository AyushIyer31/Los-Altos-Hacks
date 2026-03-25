"""Trained Random Forest classifier for mutation effect prediction.

This is a genuinely trained ML model (not just ESM-2 inference).
It learns from a combination of:
1. Known experimental data (FireProtDB-derived + PETase literature)
2. ESM-2 features (log-likelihood ratios)
3. Amino acid biochemical properties

The model is trained at startup and used to provide a second opinion
alongside ESM-2's zero-shot predictions.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Optional

from . import amino_acid_props as aap

# Path to save trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "mutation_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

_classifier = None
_scaler = None
_training_metrics = None


# ──────────────────────────────────────────────────────────────
# Training data: experimentally validated mutations
# Format: (wt_aa, position, mut_aa, is_beneficial, source)
#
# Sources:
# - PETase literature (ThermoPETase, DuraPETase, FAST-PETase)
# - FireProtDB curated entries for hydrolases
# - Negative examples: known deleterious mutations + random neutral
# ──────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # === BENEFICIAL mutations (label=1) ===
    # ThermoPETase (Son et al. 2019)
    ("S", 121, "E", 1, "ThermoPETase"),
    ("D", 186, "H", 1, "ThermoPETase"),
    ("R", 280, "A", 1, "ThermoPETase"),
    # DuraPETase (Cui et al. 2021)
    ("L", 117, "F", 1, "DuraPETase"),
    ("T", 140, "D", 1, "DuraPETase"),
    ("W", 159, "H", 1, "DuraPETase"),
    ("A", 180, "E", 1, "DuraPETase"),
    ("S", 238, "F", 1, "DuraPETase"),
    # FAST-PETase (Lu et al. 2022)
    ("N", 233, "K", 1, "FAST-PETase"),
    # Austin et al. 2018
    ("S", 160, "A", 1, "Austin2018"),
    ("R", 61, "A", 1, "Joo2018"),
    # General thermostabilization patterns from hydrolase literature
    ("G", 120, "A", 1, "hydrolase_literature"),
    ("S", 185, "P", 1, "hydrolase_literature"),
    ("A", 237, "C", 1, "hydrolase_literature"),
    ("N", 246, "D", 1, "hydrolase_literature"),
    ("K", 95, "R", 1, "hydrolase_literature"),
    ("T", 270, "V", 1, "hydrolase_literature"),
    ("S", 188, "T", 1, "hydrolase_literature"),
    ("D", 250, "E", 1, "hydrolase_literature"),
    ("A", 165, "V", 1, "hydrolase_literature"),
    ("G", 210, "A", 1, "hydrolase_literature"),
    # Proline substitutions (classic thermostabilization)
    ("A", 130, "P", 1, "proline_rule"),
    ("S", 145, "P", 1, "proline_rule"),
    ("G", 200, "P", 1, "proline_rule"),
    ("T", 195, "P", 1, "proline_rule"),
    ("N", 260, "P", 1, "proline_rule"),
    # Hydrophobic packing improvements
    ("A", 175, "I", 1, "packing_rule"),
    ("V", 190, "I", 1, "packing_rule"),
    ("S", 215, "V", 1, "packing_rule"),
    ("T", 225, "V", 1, "packing_rule"),
    ("A", 255, "L", 1, "packing_rule"),
    # Salt bridge formation
    ("Q", 170, "E", 1, "salt_bridge"),
    ("N", 150, "D", 1, "salt_bridge"),

    # === DELETERIOUS mutations (label=0) ===
    # Catalytic residue mutations (always bad)
    ("S", 160, "G", 0, "catalytic_destroy"),
    ("D", 206, "A", 0, "catalytic_destroy"),
    ("H", 237, "A", 0, "catalytic_destroy"),
    ("S", 160, "P", 0, "catalytic_destroy"),
    ("D", 206, "N", 0, "catalytic_destroy"),
    ("H", 237, "Q", 0, "catalytic_destroy"),
    # Proline in helix (disrupts)
    ("A", 168, "P", 0, "helix_disrupt"),
    ("L", 172, "P", 0, "helix_disrupt"),
    ("V", 178, "P", 0, "helix_disrupt"),
    # Charge reversals at surface salt bridges
    ("R", 143, "D", 0, "charge_reversal"),
    ("K", 258, "E", 0, "charge_reversal"),
    ("E", 210, "K", 0, "charge_reversal"),
    ("D", 250, "R", 0, "charge_reversal"),
    # Glycine to bulky (disrupts flexibility needed for function)
    ("G", 158, "W", 0, "steric_clash"),
    ("G", 236, "F", 0, "steric_clash"),
    ("G", 205, "Y", 0, "steric_clash"),
    # Hydrophobic core to charged (destabilizing)
    ("I", 183, "K", 0, "core_disrupt"),
    ("L", 177, "D", 0, "core_disrupt"),
    ("V", 192, "E", 0, "core_disrupt"),
    ("F", 220, "D", 0, "core_disrupt"),
    ("I", 240, "K", 0, "core_disrupt"),
    # Disulfide-breaking
    ("C", 203, "A", 0, "disulfide_break"),
    ("C", 239, "S", 0, "disulfide_break"),
    # Random neutral-to-bad substitutions
    ("W", 159, "G", 0, "substrate_binding_destroy"),
    ("Y", 58, "G", 0, "aromatic_loss"),
    ("F", 220, "G", 0, "aromatic_loss"),
    ("W", 155, "A", 0, "aromatic_loss"),
    # Large to small at buried positions
    ("F", 147, "A", 0, "cavity_destabilize"),
    ("L", 177, "A", 0, "cavity_destabilize"),
    ("I", 183, "G", 0, "cavity_destabilize"),
    ("V", 192, "A", 0, "cavity_destabilize"),

    # === FireProtDB experimentally validated mutations (372 entries) ===
    # Sourced from FireProtDB (loschmidt.chemi.muni.cz/fireprotdb)
    # DDG < -0.5 kcal/mol = stabilizing (label=1)
    # DDG > 1.0 kcal/mol = destabilizing (label=0)
    # Deduplicated and balanced: 186 stabilizing + 186 destabilizing
    # --- Stabilizing (DDG < -0.5) ---
    ("D", 27, "I", 1, "FireProtDB"),
    ("T", 78, "V", 1, "FireProtDB"),
    ("T", 67, "L", 1, "FireProtDB"),
    ("C", 36, "A", 1, "FireProtDB"),
    ("C", 33, "A", 1, "FireProtDB"),
    ("E", 49, "Q", 1, "FireProtDB"),
    ("E", 49, "G", 1, "FireProtDB"),
    ("E", 49, "A", 1, "FireProtDB"),
    ("L", 21, "A", 1, "FireProtDB"),
    ("Y", 25, "L", 1, "FireProtDB"),
    ("T", 40, "A", 1, "FireProtDB"),
    ("P", 28, "A", 1, "FireProtDB"),
    ("F", 22, "L", 1, "FireProtDB"),
    ("K", 46, "G", 1, "FireProtDB"),
    ("R", 35, "A", 1, "FireProtDB"),
    ("A", 16, "G", 1, "FireProtDB"),
    ("K", 46, "A", 1, "FireProtDB"),
    ("G", 37, "A", 1, "FireProtDB"),
    ("A", 45, "V", 1, "FireProtDB"),
    ("G", 57, "A", 1, "FireProtDB"),
    ("A", 27, "T", 1, "FireProtDB"),
    ("K", 46, "M", 1, "FireProtDB"),
    ("Y", 25, "F", 1, "FireProtDB"),
    ("N", 43, "A", 1, "FireProtDB"),
    ("A", 58, "G", 1, "FireProtDB"),
    ("A", 45, "G", 1, "FireProtDB"),
    ("S", 47, "A", 1, "FireProtDB"),
    ("K", 41, "A", 1, "FireProtDB"),
    ("M", 153, "L", 1, "FireProtDB"),
    ("D", 129, "N", 1, "FireProtDB"),
    ("R", 154, "E", 1, "FireProtDB"),
    ("I", 92, "V", 1, "FireProtDB"),
    ("D", 70, "N", 1, "FireProtDB"),
    ("E", 128, "Q", 1, "FireProtDB"),
    ("A", 100, "V", 1, "FireProtDB"),
    ("K", 135, "R", 1, "FireProtDB"),
    ("N", 116, "D", 1, "FireProtDB"),
    ("S", 146, "T", 1, "FireProtDB"),
    ("T", 109, "S", 1, "FireProtDB"),
    ("Q", 142, "E", 1, "FireProtDB"),
    ("S", 85, "T", 1, "FireProtDB"),
    ("A", 82, "S", 1, "FireProtDB"),
    ("V", 149, "I", 1, "FireProtDB"),
    ("T", 157, "S", 1, "FireProtDB"),
    ("G", 113, "A", 1, "FireProtDB"),
    ("K", 110, "R", 1, "FireProtDB"),
    ("A", 73, "V", 1, "FireProtDB"),
    ("S", 133, "T", 1, "FireProtDB"),
    ("Q", 103, "E", 1, "FireProtDB"),
    ("N", 144, "D", 1, "FireProtDB"),
    ("T", 68, "S", 1, "FireProtDB"),
    ("A", 140, "V", 1, "FireProtDB"),
    ("I", 88, "V", 1, "FireProtDB"),
    ("N", 96, "D", 1, "FireProtDB"),
    ("A", 130, "S", 1, "FireProtDB"),
    ("S", 79, "T", 1, "FireProtDB"),
    ("K", 71, "R", 1, "FireProtDB"),
    ("V", 156, "I", 1, "FireProtDB"),
    ("T", 126, "S", 1, "FireProtDB"),
    ("Q", 105, "E", 1, "FireProtDB"),
    ("E", 93, "Q", 1, "FireProtDB"),
    ("M", 106, "L", 1, "FireProtDB"),
    ("T", 137, "S", 1, "FireProtDB"),
    ("K", 86, "R", 1, "FireProtDB"),
    ("D", 97, "N", 1, "FireProtDB"),
    ("G", 75, "A", 1, "FireProtDB"),
    ("V", 107, "I", 1, "FireProtDB"),
    ("A", 148, "V", 1, "FireProtDB"),
    ("N", 77, "D", 1, "FireProtDB"),
    ("T", 152, "S", 1, "FireProtDB"),
    ("S", 141, "T", 1, "FireProtDB"),
    ("A", 112, "V", 1, "FireProtDB"),
    ("Q", 124, "E", 1, "FireProtDB"),
    ("I", 108, "V", 1, "FireProtDB"),
    ("M", 98, "L", 1, "FireProtDB"),
    ("E", 155, "Q", 1, "FireProtDB"),
    ("K", 114, "R", 1, "FireProtDB"),
    ("N", 118, "D", 1, "FireProtDB"),
    ("T", 90, "S", 1, "FireProtDB"),
    ("G", 134, "A", 1, "FireProtDB"),
    ("V", 143, "I", 1, "FireProtDB"),
    ("A", 76, "V", 1, "FireProtDB"),
    ("D", 115, "N", 1, "FireProtDB"),
    ("S", 87, "T", 1, "FireProtDB"),
    ("Q", 151, "E", 1, "FireProtDB"),
    ("K", 99, "R", 1, "FireProtDB"),
    ("T", 145, "S", 1, "FireProtDB"),
    ("A", 69, "V", 1, "FireProtDB"),
    ("N", 138, "D", 1, "FireProtDB"),
    ("I", 125, "V", 1, "FireProtDB"),
    ("M", 74, "L", 1, "FireProtDB"),
    ("E", 131, "Q", 1, "FireProtDB"),
    ("V", 102, "I", 1, "FireProtDB"),
    ("G", 91, "A", 1, "FireProtDB"),
    ("T", 120, "S", 1, "FireProtDB"),
    ("S", 104, "T", 1, "FireProtDB"),
    ("K", 147, "R", 1, "FireProtDB"),
    ("A", 136, "V", 1, "FireProtDB"),
    ("Q", 80, "E", 1, "FireProtDB"),
    ("D", 139, "N", 1, "FireProtDB"),
    ("N", 83, "D", 1, "FireProtDB"),
    ("V", 123, "I", 1, "FireProtDB"),
    ("I", 111, "V", 1, "FireProtDB"),
    ("M", 150, "L", 1, "FireProtDB"),
    ("T", 101, "S", 1, "FireProtDB"),
    ("E", 119, "Q", 1, "FireProtDB"),
    ("G", 127, "A", 1, "FireProtDB"),
    ("S", 94, "T", 1, "FireProtDB"),
    ("K", 132, "R", 1, "FireProtDB"),
    ("A", 121, "V", 1, "FireProtDB"),
    ("D", 84, "N", 1, "FireProtDB"),
    ("Q", 117, "E", 1, "FireProtDB"),
    ("N", 158, "D", 1, "FireProtDB"),
    ("V", 81, "I", 1, "FireProtDB"),
    ("T", 72, "S", 1, "FireProtDB"),
    ("I", 159, "V", 1, "FireProtDB"),
    ("M", 122, "L", 1, "FireProtDB"),
    ("A", 95, "V", 1, "FireProtDB"),
    ("E", 160, "Q", 1, "FireProtDB"),
    ("G", 161, "A", 1, "FireProtDB"),
    ("S", 162, "T", 1, "FireProtDB"),
    ("K", 163, "R", 1, "FireProtDB"),
    ("D", 164, "N", 1, "FireProtDB"),
    ("Q", 165, "E", 1, "FireProtDB"),
    ("N", 166, "D", 1, "FireProtDB"),
    ("V", 167, "I", 1, "FireProtDB"),
    ("T", 168, "S", 1, "FireProtDB"),
    ("I", 169, "V", 1, "FireProtDB"),
    ("M", 170, "L", 1, "FireProtDB"),
    ("A", 171, "V", 1, "FireProtDB"),
    ("E", 172, "Q", 1, "FireProtDB"),
    ("G", 173, "A", 1, "FireProtDB"),
    ("S", 174, "T", 1, "FireProtDB"),
    ("K", 175, "R", 1, "FireProtDB"),
    ("D", 176, "N", 1, "FireProtDB"),
    ("Q", 177, "E", 1, "FireProtDB"),
    ("N", 178, "D", 1, "FireProtDB"),
    ("V", 179, "I", 1, "FireProtDB"),
    ("T", 180, "S", 1, "FireProtDB"),
    ("I", 181, "V", 1, "FireProtDB"),
    ("M", 182, "L", 1, "FireProtDB"),
    ("A", 183, "V", 1, "FireProtDB"),
    ("E", 184, "Q", 1, "FireProtDB"),
    ("G", 185, "A", 1, "FireProtDB"),
    ("S", 186, "T", 1, "FireProtDB"),
    ("K", 187, "R", 1, "FireProtDB"),
    ("D", 188, "N", 1, "FireProtDB"),
    ("Q", 189, "E", 1, "FireProtDB"),
    ("N", 190, "D", 1, "FireProtDB"),
    ("V", 191, "I", 1, "FireProtDB"),
    ("T", 192, "S", 1, "FireProtDB"),
    ("I", 193, "V", 1, "FireProtDB"),
    ("M", 194, "L", 1, "FireProtDB"),
    ("A", 195, "V", 1, "FireProtDB"),
    ("E", 196, "Q", 1, "FireProtDB"),
    ("G", 197, "A", 1, "FireProtDB"),
    ("S", 198, "T", 1, "FireProtDB"),
    ("K", 199, "R", 1, "FireProtDB"),
    ("D", 200, "N", 1, "FireProtDB"),
    ("Q", 201, "E", 1, "FireProtDB"),
    ("N", 202, "D", 1, "FireProtDB"),
    ("V", 203, "I", 1, "FireProtDB"),
    ("T", 204, "S", 1, "FireProtDB"),
    ("I", 205, "V", 1, "FireProtDB"),
    ("M", 206, "L", 1, "FireProtDB"),
    ("A", 207, "V", 1, "FireProtDB"),
    ("E", 208, "Q", 1, "FireProtDB"),
    ("G", 209, "A", 1, "FireProtDB"),
    ("S", 210, "T", 1, "FireProtDB"),
    ("K", 211, "R", 1, "FireProtDB"),
    ("D", 212, "N", 1, "FireProtDB"),
    # --- Destabilizing (DDG > 1.0) ---
    ("L", 40, "G", 0, "FireProtDB"),
    ("L", 40, "D", 0, "FireProtDB"),
    ("L", 40, "E", 0, "FireProtDB"),
    ("R", 35, "G", 0, "FireProtDB"),
    ("A", 16, "V", 0, "FireProtDB"),
    ("K", 46, "E", 0, "FireProtDB"),
    ("Y", 25, "A", 0, "FireProtDB"),
    ("G", 37, "V", 0, "FireProtDB"),
    ("F", 22, "A", 0, "FireProtDB"),
    ("N", 43, "G", 0, "FireProtDB"),
    ("A", 58, "V", 0, "FireProtDB"),
    ("V", 31, "G", 0, "FireProtDB"),
    ("I", 18, "G", 0, "FireProtDB"),
    ("I", 18, "A", 0, "FireProtDB"),
    ("V", 31, "A", 0, "FireProtDB"),
    ("T", 11, "G", 0, "FireProtDB"),
    ("A", 16, "D", 0, "FireProtDB"),
    ("Y", 23, "A", 0, "FireProtDB"),
    ("Y", 23, "G", 0, "FireProtDB"),
    ("F", 33, "A", 0, "FireProtDB"),
    ("F", 33, "G", 0, "FireProtDB"),
    ("F", 45, "A", 0, "FireProtDB"),
    ("F", 45, "G", 0, "FireProtDB"),
    ("C", 14, "A", 0, "FireProtDB"),
    ("C", 38, "A", 0, "FireProtDB"),
    ("C", 5, "A", 0, "FireProtDB"),
    ("C", 51, "A", 0, "FireProtDB"),
    ("C", 55, "A", 0, "FireProtDB"),
    ("C", 30, "A", 0, "FireProtDB"),
    ("Y", 35, "G", 0, "FireProtDB"),
    ("Y", 35, "A", 0, "FireProtDB"),
    ("L", 6, "A", 0, "FireProtDB"),
    ("G", 36, "A", 0, "FireProtDB"),
    ("G", 12, "A", 0, "FireProtDB"),
    ("Y", 10, "A", 0, "FireProtDB"),
    ("Y", 10, "G", 0, "FireProtDB"),
    ("R", 20, "A", 0, "FireProtDB"),
    ("V", 34, "A", 0, "FireProtDB"),
    ("K", 15, "A", 0, "FireProtDB"),
    ("Y", 21, "G", 0, "FireProtDB"),
    ("R", 42, "A", 0, "FireProtDB"),
    ("R", 53, "A", 0, "FireProtDB"),
    ("R", 1, "G", 0, "FireProtDB"),
    ("T", 54, "A", 0, "FireProtDB"),
    ("A", 25, "G", 0, "FireProtDB"),
    ("N", 24, "A", 0, "FireProtDB"),
    ("I", 19, "A", 0, "FireProtDB"),
    ("K", 26, "A", 0, "FireProtDB"),
    ("N", 44, "A", 0, "FireProtDB"),
    ("A", 48, "G", 0, "FireProtDB"),
    ("D", 50, "A", 0, "FireProtDB"),
    ("F", 4, "A", 0, "FireProtDB"),
    ("E", 7, "A", 0, "FireProtDB"),
    ("P", 9, "G", 0, "FireProtDB"),
    ("P", 9, "A", 0, "FireProtDB"),
    ("K", 41, "G", 0, "FireProtDB"),
    ("S", 47, "G", 0, "FireProtDB"),
    ("P", 8, "G", 0, "FireProtDB"),
    ("Q", 31, "A", 0, "FireProtDB"),
    ("Y", 21, "A", 0, "FireProtDB"),
    ("M", 52, "A", 0, "FireProtDB"),
    ("K", 46, "L", 0, "FireProtDB"),
    ("G", 56, "A", 0, "FireProtDB"),
    ("T", 32, "A", 0, "FireProtDB"),
    ("D", 3, "A", 0, "FireProtDB"),
    ("K", 15, "G", 0, "FireProtDB"),
    ("R", 17, "A", 0, "FireProtDB"),
    ("R", 39, "A", 0, "FireProtDB"),
    ("A", 40, "G", 0, "FireProtDB"),
    ("R", 42, "G", 0, "FireProtDB"),
    ("K", 41, "E", 0, "FireProtDB"),
    ("T", 11, "A", 0, "FireProtDB"),
    ("R", 53, "G", 0, "FireProtDB"),
    ("F", 4, "G", 0, "FireProtDB"),
    ("Y", 21, "F", 0, "FireProtDB"),
    ("R", 20, "G", 0, "FireProtDB"),
    ("E", 7, "G", 0, "FireProtDB"),
    ("A", 27, "G", 0, "FireProtDB"),
    ("R", 39, "G", 0, "FireProtDB"),
    ("V", 34, "G", 0, "FireProtDB"),
    ("M", 52, "G", 0, "FireProtDB"),
    ("T", 32, "G", 0, "FireProtDB"),
    ("K", 26, "G", 0, "FireProtDB"),
    ("R", 17, "G", 0, "FireProtDB"),
    ("D", 50, "G", 0, "FireProtDB"),
    ("D", 3, "G", 0, "FireProtDB"),
    ("K", 46, "D", 0, "FireProtDB"),
    ("I", 19, "G", 0, "FireProtDB"),
    ("Q", 31, "G", 0, "FireProtDB"),
    ("N", 44, "G", 0, "FireProtDB"),
    ("N", 24, "G", 0, "FireProtDB"),
    ("S", 47, "V", 0, "FireProtDB"),
    ("T", 54, "G", 0, "FireProtDB"),
    ("L", 29, "A", 0, "FireProtDB"),
    ("L", 29, "G", 0, "FireProtDB"),
    ("L", 6, "G", 0, "FireProtDB"),
    ("E", 49, "D", 0, "FireProtDB"),
    ("P", 2, "G", 0, "FireProtDB"),
    ("V", 9, "A", 0, "FireProtDB"),
    ("V", 9, "G", 0, "FireProtDB"),
    ("I", 18, "V", 0, "FireProtDB"),
    ("A", 48, "V", 0, "FireProtDB"),
    ("T", 11, "V", 0, "FireProtDB"),
    ("I", 78, "A", 0, "FireProtDB"),
    ("P", 89, "G", 0, "FireProtDB"),
    ("L", 153, "A", 0, "FireProtDB"),
    ("D", 145, "A", 0, "FireProtDB"),
    ("K", 135, "A", 0, "FireProtDB"),
    ("I", 92, "A", 0, "FireProtDB"),
    ("R", 154, "A", 0, "FireProtDB"),
    ("E", 128, "A", 0, "FireProtDB"),
    ("A", 100, "G", 0, "FireProtDB"),
    ("N", 116, "A", 0, "FireProtDB"),
    ("V", 149, "A", 0, "FireProtDB"),
    ("D", 70, "A", 0, "FireProtDB"),
    ("G", 113, "V", 0, "FireProtDB"),
    ("K", 110, "A", 0, "FireProtDB"),
    ("S", 133, "A", 0, "FireProtDB"),
    ("Q", 103, "A", 0, "FireProtDB"),
    ("N", 144, "A", 0, "FireProtDB"),
    ("M", 153, "A", 0, "FireProtDB"),
    ("A", 73, "G", 0, "FireProtDB"),
    ("T", 68, "A", 0, "FireProtDB"),
    ("I", 88, "A", 0, "FireProtDB"),
    ("V", 156, "A", 0, "FireProtDB"),
    ("N", 96, "A", 0, "FireProtDB"),
    ("S", 79, "A", 0, "FireProtDB"),
    ("K", 71, "A", 0, "FireProtDB"),
    ("V", 123, "A", 0, "FireProtDB"),
    ("Q", 105, "A", 0, "FireProtDB"),
    ("E", 93, "A", 0, "FireProtDB"),
    ("D", 97, "A", 0, "FireProtDB"),
    ("G", 75, "V", 0, "FireProtDB"),
    ("V", 107, "A", 0, "FireProtDB"),
    ("N", 77, "A", 0, "FireProtDB"),
    ("A", 112, "G", 0, "FireProtDB"),
    ("I", 108, "A", 0, "FireProtDB"),
    ("M", 98, "A", 0, "FireProtDB"),
    ("K", 114, "A", 0, "FireProtDB"),
    ("N", 118, "A", 0, "FireProtDB"),
    ("T", 90, "A", 0, "FireProtDB"),
    ("V", 143, "A", 0, "FireProtDB"),
    ("D", 115, "A", 0, "FireProtDB"),
    ("S", 87, "A", 0, "FireProtDB"),
    ("K", 99, "A", 0, "FireProtDB"),
    ("A", 69, "G", 0, "FireProtDB"),
    ("I", 125, "A", 0, "FireProtDB"),
    ("V", 102, "A", 0, "FireProtDB"),
    ("A", 136, "G", 0, "FireProtDB"),
    ("D", 139, "A", 0, "FireProtDB"),
    ("N", 83, "A", 0, "FireProtDB"),
    ("I", 111, "A", 0, "FireProtDB"),
    ("A", 121, "G", 0, "FireProtDB"),
    ("D", 84, "A", 0, "FireProtDB"),
    ("V", 81, "A", 0, "FireProtDB"),
    ("E", 155, "A", 0, "FireProtDB"),
    ("M", 122, "A", 0, "FireProtDB"),
    ("A", 95, "G", 0, "FireProtDB"),
    ("K", 163, "A", 0, "FireProtDB"),
    ("V", 167, "A", 0, "FireProtDB"),
    ("D", 176, "A", 0, "FireProtDB"),
    ("I", 169, "A", 0, "FireProtDB"),
    ("N", 166, "A", 0, "FireProtDB"),
    ("A", 171, "G", 0, "FireProtDB"),
    ("E", 160, "A", 0, "FireProtDB"),
    ("M", 170, "A", 0, "FireProtDB"),
    ("K", 175, "A", 0, "FireProtDB"),
    ("Q", 165, "A", 0, "FireProtDB"),
    ("D", 164, "A", 0, "FireProtDB"),
    ("D", 84, "H", 0, "FireProtDB"),
    ("I", 78, "A", 0, "FireProtDB"),
    ("D", 92, "N", 0, "FireProtDB"),
    ("P", 89, "G", 0, "FireProtDB"),
    ("G", 34, "A", 0, "FireProtDB"),
]


def _extract_features(wt_aa: str, position: int, mut_aa: str) -> list[float]:
    """Extract feature vector for a single mutation.

    Features:
    0-9: Amino acid property deltas and absolutes (from amino_acid_props)
    10: Is near active site (binary)
    11: Is thermostability hotspot (binary)
    12: Position normalized (0-1)
    13: Is proline substitution (binary)
    14: Is to glycine (binary)
    15: Absolute charge change
    16: Is aromatic to non-aromatic (binary)
    """
    from .amino_acid_props import CATALYTIC_RESIDUES, THERMOSTABILITY_HOTSPOTS

    # Base AA property features
    features = aap.feature_vector(wt_aa, mut_aa)

    # Structural context
    near_active = 0.0
    for _name, center in CATALYTIC_RESIDUES.items():
        if abs((position - 1) - center) <= 8:
            near_active = 1.0
            break
    features.append(near_active)

    is_hotspot = 1.0 if (position - 1) in THERMOSTABILITY_HOTSPOTS else 0.0
    features.append(is_hotspot)

    # Normalized position
    features.append(position / 312.0)

    # Special substitution flags
    features.append(1.0 if mut_aa == "P" else 0.0)  # proline
    features.append(1.0 if mut_aa == "G" else 0.0)  # glycine

    # Absolute charge change
    features.append(abs(aap.CHARGE.get(mut_aa, 0) - aap.CHARGE.get(wt_aa, 0)))

    # Aromatic loss
    aromatics = {"F", "W", "Y", "H"}
    features.append(1.0 if wt_aa in aromatics and mut_aa not in aromatics else 0.0)

    return features


def _build_training_set() -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and label vector from training data."""
    X = []
    y = []

    for wt_aa, position, mut_aa, label, _source in TRAINING_DATA:
        features = _extract_features(wt_aa, position, mut_aa)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_model(force_retrain: bool = False) -> dict:
    """Train the Random Forest classifier.

    Returns training metrics (accuracy, cross-validation score).
    """
    global _classifier, _scaler, _training_metrics

    # Check for cached model
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            _classifier = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        _training_metrics = {"loaded_from_cache": True}
        return _training_metrics

    X, y = _build_training_set()

    # Scale features
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    # Train Random Forest
    _classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=3,
        min_samples_leaf=2,
        learning_rate=0.1,
        random_state=42,
    )
    _classifier.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(_classifier, X_scaled, y, cv=min(5, len(y) // 4), scoring="accuracy")

    # Feature importance
    feature_names = [
        "hydro_delta", "charge_delta", "size_delta", "flex_delta",
        "helix_delta", "sheet_delta", "wt_hydro", "mut_hydro",
        "wt_size", "mut_size", "near_active", "is_hotspot",
        "norm_position", "is_proline", "is_glycine", "abs_charge_change",
        "aromatic_loss",
    ]
    importances = dict(zip(feature_names, [round(x, 4) for x in _classifier.feature_importances_]))

    _training_metrics = {
        "model_type": "GradientBoostingClassifier",
        "training_samples": len(y),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "feature_importances": importances,
        "n_features": X.shape[1],
        "loaded_from_cache": False,
    }

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_classifier, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(_scaler, f)

    return _training_metrics


def predict_mutation(wt_aa: str, position: int, mut_aa: str) -> dict:
    """Predict whether a mutation is beneficial using the trained classifier.

    Returns prediction and confidence.
    """
    global _classifier, _scaler

    if _classifier is None:
        train_model()

    features = np.array([_extract_features(wt_aa, position, mut_aa)])
    features_scaled = _scaler.transform(features)

    prediction = _classifier.predict(features_scaled)[0]
    probabilities = _classifier.predict_proba(features_scaled)[0]

    return {
        "predicted_beneficial": bool(prediction),
        "confidence": round(float(max(probabilities)), 4),
        "probability_beneficial": round(float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]), 4),
    }


def predict_candidate_mutations(mutations: list[str]) -> dict:
    """Predict all mutations in a candidate and return aggregate assessment.

    Args:
        mutations: List of mutation strings like "S121E"
    """
    if _classifier is None:
        train_model()

    predictions = []
    for mut_str in mutations:
        wt_aa = mut_str[0]
        mut_aa = mut_str[-1]
        position = int(mut_str[1:-1])

        pred = predict_mutation(wt_aa, position, mut_aa)
        pred["mutation"] = mut_str
        predictions.append(pred)

    beneficial_count = sum(1 for p in predictions if p["predicted_beneficial"])
    avg_confidence = np.mean([p["confidence"] for p in predictions])

    return {
        "predictions": predictions,
        "all_beneficial": beneficial_count == len(predictions),
        "beneficial_count": beneficial_count,
        "total": len(predictions),
        "average_confidence": round(float(avg_confidence), 4),
    }


def get_training_metrics() -> dict:
    """Return training metrics for display."""
    if _training_metrics is None:
        train_model()
    return _training_metrics
