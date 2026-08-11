# PET-Lab: Condition-Aware Machine-Learning Models and Datasets for Engineering Thermostable PET-Degrading Enzymes

> **Note for depositor:** fields marked `<< FILL >>` need your input (author list,
> affiliations, ORCID, advisor). Everything else is ready to paste into Zenodo.

## Authors / Creators
- Ayush Iyer — University of California, Santa Cruz — `<< ORCID >>`
- `<< co-authors / advisor, affiliations >>`

## License
Creative Commons Attribution 4.0 International (**CC BY 4.0**)

## Description (paste into Zenodo "Description")

This record archives the trained models and leakage-audited datasets from the PET-Lab
project, a machine-learning pipeline for engineering thermostable PET-degrading enzymes
(e.g. PETase, LCC, cutinases). Enzymatic PET recycling is most efficient at high temperature,
where the plastic softens, yet most natural PET hydrolases denature well below that point.
The pipeline pairs two complementary models built on ESM-2 protein-language-model embeddings:
a **screening model** that predicts an enzyme's melting temperature (Tm) directly from
sequence, and a **mutation model** that predicts the stability change (ΔΔG) of point
mutations. The pipeline is condition-aware (temperature), with pH conditioning in progress.

A central contribution is **data rigor**: training data assembled from seven public stability
resources (~1M measurements) and independent benchmarks audited for sequence-homology leakage
using mmseqs2 (≥30% identity / ≥50% coverage, bidirectional), so reported performance reflects
generalization to genuinely novel enzymes.

## Contents of this record

| File | What it is |
|------|-----------|
| `mutation_regressor.pkl` | Trained ΔΔG mutation-effect model (gradient-boosted ensemble on ESM-2 features) |
| `esm_embeddings_cache.pkl` | Cached ESM-2 embeddings used by the pipeline |
| `ph_opt_train_BRENDA.csv` | pH-optimum **training** set (BRENDA-sourced), homology-clean |
| `ph_opt_test_UNIPROT.csv` | pH-optimum **test** set (UniProt-sourced), homology-clean, independent |
| `ph_opt_LEAKAGE_REPORT.txt` | Leakage-audit report for the pH-optimum split |

## Methods / provenance

- Embeddings: ESM-2 (`esm2_t30_150M`).
- Training stability data: FireProtDB, ProDDG/S2648, ThermoMutDB, Tsuboyama 2023
  (Zenodo 7992926), Domainome (Zenodo 13629491), Meltome Atlas.
- pH-optimum labels: BRENDA 2026.1 (CC BY 4.0) and UniProt (`ph_dependence: optimum`).
- Leakage standard: mmseqs2, ≥30% id / ≥50% coverage, bidirectional.
- Code: https://github.com/AyushIyer31/PET-Lab

## Related identifiers (add under Zenodo "Related/alternate identifiers")
- isSupplementTo → GitHub repo: https://github.com/AyushIyer31/PET-Lab
- references → Tsuboyama et al. 2023, Nature (doi:10.1038/s41586-023-06328-6)
- references → BRENDA (brenda-enzymes.org)

## Keywords
PETase, protein stability, enzyme engineering, ESM-2, machine learning, thermostability,
plastic recycling, ddG, melting temperature, pH optimum

## Suggested citation
Iyer, A. et al. (2026). *PET-Lab: Condition-Aware Machine-Learning Models and Datasets for
Engineering Thermostable PET-Degrading Enzymes.* Zenodo. https://doi.org/<< assigned on publish >>
