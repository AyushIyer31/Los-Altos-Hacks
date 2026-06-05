# PET Lab — AI-Powered PETase Enzyme Optimization

PET Lab is a full-stack platform for computationally engineering plastic-degrading PETase enzymes. It combines a machine learning ensemble trained on 17,791 experimentally validated mutations with physics-based condition corrections and an interactive 3D web interface.

**Live site:** [https://ayushiyer31.github.io/PET-Lab/PET_Lab_Website/](https://ayushiyer31.github.io/PET-Lab/PET_Lab_Website/)

## What It Does

- Predicts mutational ΔΔG (change in protein stability) for any PETase variant
- Generates ranked multi-mutant candidates optimized for thermostability
- Adjusts predictions for real assay conditions: temperature, pH, ionic strength, Ca²⁺ concentration
- Visualizes mutations on 3D protein structures via 3Dmol.js

## Model

The current model (v44) is a 6-model stacking ensemble with 144 features:

| Metric | Value |
|---|---|
| CV Accuracy | 79.68% |
| CV MAE | 0.95 kcal/mol |
| Pearson r | 0.76 |
| Spearman ρ | 0.73 |

**Ensemble members:** GradientBoosting, XGBoost, LightGBM, CatBoost, HistGradientBoosting, Ridge meta-learner

**Key features:** ESM-1v evolutionary fitness, AlphaFold pLDDT scores, ESM-2 PCA embeddings, PSSM conservation, real RSA/SS from DSSP, metal coordination geometry, Debye-Hückel ionic screening, Hill-equation Ca²⁺ thermodynamics

**Training data:** FireProtDB (6,798 mutations) + ThermoMutDB (10,993 mutations) — real experimental data only, no synthetic augmentation.

## Architecture

```
PET_Lab_Website/       Frontend (static HTML/JS, GitHub Pages)
backend/               FastAPI REST API (Cloud Run)
  app/
    services/          ML inference, PDB fetching, ESM embeddings
    trained_models/    Model weights (auto-downloaded from Hugging Face)
    models/            Pydantic schemas
train_v44.py           Training script for current model
train_v45.py           Experimental: +Optuna classifier tuning
train_v46.py           Experimental: +RF/ExtraTrees ensemble
```

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements-deploy.txt
python start.py
```

The server starts on `http://localhost:10000`. Model files are auto-downloaded from Hugging Face on first startup.

### Frontend

Open `PET_Lab_Website/index.html` in a browser, or serve it:

```bash
cd PET_Lab_Website
python3 -m http.server 8000
```

Update the `API` constant in `index.html` to `http://localhost:10000` for local development.

## Feature Generation Scripts

These scripts generate the cached feature files used during training:

- `compute_esm1v.py` / `compute_esm1v_ensemble.py` — ESM-1v evolutionary fitness scores
- `compute_esm_embeddings.py` / `generate_esm_embeddings.py` — ESM-2 embeddings
- `generate_esm_loglik.py` — ESM log-likelihood ratios
- `generate_metal_coord_cache.py` — Metal coordination geometry from PDB
- `generate_physics_features.py` — Debye-Hückel, Born solvation, Hill-equation features
- `generate_pssm_conservation.py` — PSSM conservation scores
- `fetch_plddt_cache.py` — AlphaFold pLDDT confidence scores

## Deployment

The backend is deployed on Google Cloud Run. To redeploy:

```bash
cd backend
gcloud run deploy pet-lab-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

Set the `HF_TOKEN` environment variable on Cloud Run for model downloads from the private Hugging Face repository.

## References

- Yoshida et al. (2016). A bacterium that degrades and assimilates poly(ethylene terephthalate). *Science*
- Lu et al. (2022). Machine learning-aided engineering of hydrolases for PET depolymerization. *Nature*
- Tournier et al. (2020). An engineered PET depolymerase to break down and recycle plastic bottles. *Nature*
