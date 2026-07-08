# Data Manifest

Large data files (>100 MB) are **not** stored in this repository (GitHub's file limit).
This manifest records, for every large file, whether it is **regenerable** (rebuild from
scripts in this repo), **re-downloadable** (fetch from a public source), or **archived**
(irreplaceable — back up to durable storage). With this file, the project is fully
reproducible without committing the raw bytes.

Small, final artifacts (the leakage-clean pH-optimum datasets, all scripts, reports, and
slides) **are** committed to the repo.

---

## Bucket 1 — Regenerable (do not store; rebuild from scripts)

Outputs of pipeline scripts that live in this repo. Delete freely; recreate on demand.

| File | Size | Rebuild with |
|------|------|--------------|
| `stability_dataset_multitask.csv` | 353 MB | `python build_stability_csv.py` (imports the source loaders) |
| `datasets/staging/staging_all.csv` | 408 MB | `python build_staging_table.py` |
| `datasets/staging/staging_clean.csv` | 357 MB | `python build_staging_table.py` (after `run_mmseqs_audit.py`) |
| `datasets/staging/mmseqs/` | ~20 MB | `python run_mmseqs_audit.py` |
| `stability_dataset_19k.csv`, `stability_dataset_19k_mt.csv` | 41 / 59 MB | `build_stability_csv.py`, `convert_19k_to_mt.py` |
| `datasets/downloads/ph_opt/` intermediates (`ref.fasta`, `trainpool.*`, `*.m8`, `brenda_*_seqs*.json`) | ≤13 MB | pH-optimum build steps (see Bucket 2 sources + UniProt REST) |

> Rebuild order: download Bucket-2 sources → `run_mmseqs_audit.py` → `build_staging_table.py` → `build_stability_csv.py`.

---

## Bucket 2 — Re-downloadable (do not store; fetch from public source)

Public database releases. Record of exact source + version.

| File | Size | Source | Notes |
|------|------|--------|-------|
| `datasets/downloads/brenda_2026_1.json` | 677 MB | https://www.brenda-enzymes.org/download.php | Release **2026.1**, `brenda_2026_1.json.tar.gz`; **login-gated** (free account); CC BY 4.0 |
| `datasets/stability_megascale/Processed_K50_dG_datasets.zip` | 967 MB | Zenodo **7992926** | Tsuboyama et al. 2023, [Nature](https://www.nature.com/articles/s41586-023-06328-6) |
| `datasets/stability_megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | 665 MB | (extracted from the zip above) | 776K variants |
| `datasets/downloads/fireprotdb_dump.zip` | 388 MB | FireProtDB — https://loschmidt.chemi.muni.cz/fireprotdb/ | [Bioinformatics btaa1059](https://doi.org/10.1093/bioinformatics/btaa1059) |
| `datasets/downloads/domainome_table2.txt` | 111 MB | Zenodo **13629491** (Table 2) | Domainome aPCA abundance, [Nature 2024](https://www.nature.com/articles/s41586-024-08370-4) |
| `datasets/downloads/domainome_stability.csv` | 96 MB | (derived from Domainome Table 2 via `convert_domainome.py`) | — |
| Meltome (`meltome_mixed.fasta`, `meltome_tm.csv`) | ~16 MB | FLIP mirror `mixed_split.fasta` | Meltome Atlas, [Nat. Methods 2020](https://www.nature.com/articles/s41592-020-0801-4) |
| ThermoMutDB (`thermomutdb.json`) | — | https://biosig.lab.uq.edu.au/thermomutdb/ | — |
| ProDDG / S2648 (`proddg_s2648.tsv`) | — | ProDDG / S2648 published set | — |
| UniProt sequences & pH-optimum | — | https://rest.uniprot.org/ | resolved on the fly (accession → sequence; `cc_bpcp_ph_dependence:optimum`) |
| EpHod pH-optimum + pHenv (if used) | — | Zenodo **14252615** | [Nat. Mach. Intell. 2025](https://www.nature.com/articles/s42256-025-01026-6) |

---

## Bucket 3 — Irreplaceable (back up to durable storage) ⚠️

Expensive to regenerate (requires GPU/HPC time). **These currently exist only on the local
machine — back them up.**

| File | Size | What it is | Backup target |
|------|------|-----------|---------------|
| `backend/app/trained_models/mutation_regressor.pkl` | 2.0 GB | trained ΔΔG mutation model | Zenodo / cloud / HPC storage |
| `backend/app/trained_models/esm_embeddings_cache.pkl` | 286 MB | cached ESM-2 embeddings | Zenodo / cloud / HPC storage |
| `pdb_structures/` | 371 MB | structure files (re-fetchable from RCSB, but slow) | cloud / HPC storage |

**Recommended:** archive Bucket 3 + the final clean datasets to **Zenodo** (free, ≤50 GB,
gives a citable DOI) and keep a working copy on the **HPC cluster** where training runs.

---

## Committed to the repo (for reference)

- `datasets/downloads/ph_opt/FINAL_train_BRENDA.csv` — pH-optimum training set (BRENDA), homology-clean
- `datasets/downloads/ph_opt/FINAL_test_UNIPROT.csv` — pH-optimum test set (UniProt), homology-clean
- `datasets/downloads/ph_opt/LEAKAGE_REPORT.txt` — leakage-audit report
- all `*.py` pipeline scripts, `*.html` reports, project PDFs and slides

Leakage standard used throughout: **mmseqs2, ≥30% identity / ≥50% coverage, bidirectional.**
