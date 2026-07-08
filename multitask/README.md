# Multi-task stability model

A single network that predicts four stability measurements from sequence:
**ddG · dTm · Tm · abundance** — one shared ESM-2 encoder, four heads, FiLM
conditioning on assay (temperature, pH). Built to train on
`../stability_dataset_multitask.csv` (~1M rows, ~24.5K proteins) on a single A100.

## Files
| File | What it is |
|------|------------|
| `model.py` | The network: shared encoder + 4 heads + FiLM(T, pH). Run directly for a shape smoke-test. |
| `train_multitask.py` | Loads the CSV, precomputes & caches ESM-2 embeddings, trains with per-head/per-source weighting, evaluates the ddG head on S669, saves to `runs/`. |
| `nautilus_pvc.yaml` | One-time PersistentVolumeClaim for code + data + cache. |
| `nautilus_job.yaml` | The A100 training Job (Kubernetes). |
| `requirements.txt` | `torch`, `fair-esm`, `pandas`, `numpy`. |

## Design choices (and where to tune)
- **Embeddings are precomputed once** (`esm_meanpool_cache.pkl`) so the GPU runs ESM-2
  a single time, not every epoch. Sequences > 1022 aa are truncated to the ESM context limit.
- **Targets are z-scored per head** so the large-magnitude Tm head can't dominate the loss.
- **Emphasis knobs** live at the top of `train_multitask.py`:
  - `HEAD_WEIGHTS` — abundance proxy down-weighted to 0.3.
  - `SOURCE_WEIGHTS` — gold thermodynamic sources (FireProtDB/ProDDG/ThermoMutDB) lifted to 1.5.
- **Split is protein-grouped** — no wild-type sequence appears in both train and val.

## Quick local sanity check (no A100 needed)
```bash
python model.py                       # verify the network wires up
python train_multitask.py --limit 5000 --epochs 2   # tiny end-to-end run (needs torch + fair-esm)
```

## Run on Nautilus
```bash
kubectl apply -f nautilus_pvc.yaml          # 1. create storage (once)
# ... copy multitask/, stability_dataset_multitask.csv, s669_full.tsv onto the PVC ...
kubectl apply -f nautilus_job.yaml          # 2. submit the A100 job
kubectl logs -f job/petlab-multitask-train  # 3. watch
```

## Known upgrade paths (intentionally left simple in this scaffold)
- Mean-pooled embeddings lose mutation-site detail → swap in per-residue / mutation-site
  embeddings or a small attention pool for a likely accuracy gain.
- Embeddings are frozen → add optional end-to-end ESM fine-tuning (much slower, more GPU).
- FiLM conditioning is only meaningful on ddG/dTm rows (the only ones with real T/pH).
