# PET Degradation-Efficiency Dataset

Real, experimentally-measured PET hydrolysis data — the degradation efficiency
the model needs, as opposed to the stability (ΔΔG) data the rest of the repo
trains on. Curated from primary literature, standardized to one tidy schema.

## Why this exists

The ML pipeline (`backend/`) is trained on protein **stability** (FireProtDB +
ThermoMutDB, ~17k ΔΔG points). It contains **no degradation data** — the only
degradation numbers in the repo were 5 hardcoded enzyme profiles in
`backend/app/services/pet_degradation_simulator.py`. The "efficiency %" shown in
the UI is a temperature-fit Gaussian, not a measured degradation rate. This
dataset is the foundation for predicting/validating **actual** PET breakdown.

## Schema (`*_degradation.csv`)

One row = one measurement of (enzyme × substrate × temperature × pH).

| column | meaning |
|---|---|
| `enzyme_id` | enzyme identifier or name from the source |
| `sequence` | full protein sequence (joined from source SI) |
| `substrate` | `amorphous_film` / `amorphous_powder` / `crystalline_powder` |
| `crystallinity_pct` | approximate % crystallinity of substrate |
| `temp_C` | reaction temperature (°C) |
| `pH` | reaction pH |
| `buffer` | buffer code from source |
| `enzyme_load_mg_per_g` | enzyme loading (mg enzyme / g PET) |
| `substrate_loading_pct` | substrate loading (% w/v) |
| `time_h` | reaction time (hours) |
| `aromatic_products_mg_per_L` | **degradation readout**: sum of aromatic products (TPA+MHET+BHET) released, **mg/L** (source: "Sum of Aromatic Products (mg/L)") |
| `aromatic_products_stdev` | std. dev. across replicates |
| `src_table` | source supplementary table |
| `source` | citation |
| `doi` | source DOI |

`aromatic_products_mg_per_L` is the degradation-efficiency target: the mass of
aromatic monomers released, the standard quantitative measure of PET hydrolysis.
Higher = more PET broken down under that condition.

**% depolymerization.** Because the assay is 2.9% w/v PET (29,000 mg/L), the
mass readout converts to a fraction of PET degraded:
`% ≈ (mg/L ÷ 166.13 g/mol TPA) ÷ (29,000 ÷ 192.17 g/mol PET) × 100 ≈ mg/L × 0.00399`.
The best engineered enzyme in the data (LCC-ICCG, ~10,400 mg/L) ≈ **42%**.

## Sources ingested

| source | rows | enzymes | what |
|---|---|---|---|
| Erickson et al. 2022, *Nat Commun* 13:7850 (`10.1038/s41467-022-35237-x`) | 3,616 | 65 | **Training set.** Thermotolerance screen: 51 natural-diversity hydrolases + controls (LCC, LCC-ICCG, IsPETase, BTA-1…) across 30–70 °C, pH 4.5–9.0, three substrate forms. Standard conditions: 0.7 mg enzyme/g PET, 2.9 % loading, 96 h. |
| Lu et al. 2022, *Nature* 604:662 (`10.1038/s41586-022-04599-z`) | 6 | 3 | **External validation only** (`lu2022_validation.csv`). Fig. 2b variant activities in *mM* under a single condition — different unit/assay, so NOT merged into training. |

Sources that did **not** yield mergeable quantitative rows:
- **Tournier et al. 2020, *Nature*** — supplementary information is PDF-only; no
  machine-readable degradation matrices.
- **PlasticDB / PAZy** — qualitative/semi-quantitative breadth (which microbe
  degrades which polymer), not standardized kinetic measurements.

Regenerate:
```bash
python ingest_erickson2022.py   # -> erickson2022_degradation.csv (training)
python ingest_lu2022.py         # -> lu2022_validation.csv (held-out)
python train_degradation_model.py   # -> backend trained_models/degradation_regressor.pkl
```

## Model

`train_degradation_model.py` trains a `HistGradientBoostingRegressor`:
`sequence composition + temp + pH + substrate -> log1p(mg/L aromatic products)`.
The endpoint also returns `percent_depolymerized` (see above).

| CV scheme | what it measures | R² | MAE |
|---|---|---|---|
| random KFold | known enzymes, new conditions | **0.93** | 41 mg/L |
| GroupKFold (by enzyme) | **novel sequences** (honest) | **0.48** | 101 mg/L |

Served by `backend/app/services/degradation_model.py` via the `POST /simulate`
endpoint. The 0.48 unseen-enzyme R² is the realistic small-data ceiling with
composition-only features (65 enzymes); novel-sequence predictions are
indicative, not validated. Adding ESM embeddings is the obvious next lever.

## Important caveats (read before training)

- **Cross-paper comparability.** Degradation readouts are *not* directly
  comparable across labs — substrate form, enzyme loading, and assay time differ.
  Within Erickson 2022 conditions are standardized; when adding other sources,
  keep `substrate`, `enzyme_load`, `time_h` as features, do not pool blindly.
- **Zeros are real.** ~1,665 of 3,616 rows are 0 mg/L (no measurable activity at
  that condition) — these are informative negatives, not missing data.
- **Scale.** This is the realistic ceiling for *clean* PET-degradation data:
  low thousands of measurements from tens of enzymes. There is no public
  "thousands-of-thousands" degradation dataset — those experiments don't exist.
  Use this for validation and small-data modeling, not deep nets from scratch.

## To expand

Candidate next sources (each adds tens–hundreds of rows; require per-paper SI parsing):
- Lu et al. 2022, *Nature* (FAST-PETase) — ML-guided variants, temp/pH curves
- Tournier et al. 2020, *Nature* (LCC engineering) — kinetics
- Bell et al. 2022, *Nat Catal* (HotPETase) — thermostability + activity
- PlasticDB (plasticdb.org) — ~329 proteins, breadth (qualitative/semi-quant)
- PAZy (pazy.eu) — curated plastic-active enzymes
