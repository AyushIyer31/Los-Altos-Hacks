# Section 7 — Data Availability (for *Scientific Reports*)

> Owner: JA (James Ponzio). Compiled from `DATA.md`. Zenodo DOI now assigned:
> **10.5281/zenodo.21257369** (record "PET training set", Iyer, Ayush; one file, ~2.5 GB).
> See the confirmation notes at the bottom before final submission (access + license).

---

## Data Availability statement (ready to paste)

The processed training table, the leakage-audited benchmark datasets, saved predictions, and the
trained model artifacts generated in this study are available through Zenodo
(https://doi.org/10.5281/zenodo.21257369). All source code for data assembly, leakage auditing,
model training, and evaluation is available at https://github.com/AyushIyer31/PET-Lab.

This work reuses the following publicly available datasets:

- **BRENDA** (release 2026.1), from which the whole-protein temperature-stability benchmark was
  constructed — https://www.brenda-enzymes.org (CC BY 4.0).
- **Tsuboyama et al. (2023)** mega-scale folding-stability dataset — Zenodo,
  https://doi.org/10.5281/zenodo.7992926.
- **Domainome** aPCA abundance dataset — Zenodo, https://doi.org/10.5281/zenodo.13629491.
- **FireProtDB** — https://loschmidt.chemi.muni.cz/fireprotdb/.
- **Meltome Atlas** melting-temperature data (Jarzab et al., 2020).
- **ThermoMutDB** — https://biosig.lab.uq.edu.au/thermomutdb/.
- **ProDDG / S2648** stability-change dataset (published).
- **S669** benchmark of single-point ΔΔG measurements (Pancotti et al., 2022), used as the
  independent test set for the mutation model.
- **EpHod** enzyme pH-optimum dataset (used for the pH-optimum extension) — Zenodo,
  https://doi.org/10.5281/zenodo.14252615.

Protein sequences not stored inline in the source databases were resolved from **UniProt**
(https://www.uniprot.org).

---

### Open confirmations on the Zenodo deposit (DOI 10.5281/zenodo.21257369)
- **Contents:** the record currently shows **one file (~2.5 GB)**. Confirm it contains all cited
  artifacts (processed training table, leakage-audited benchmark datasets, saved predictions, split
  identifiers, leakage-audit outputs, trained models) and add a README/data dictionary inside it.
- **Access:** the files currently **require login (restricted)**. For submission, either make the
  files openly downloadable or provide a reviewer-access mechanism; adjust the wording if access
  stays restricted.
- **License:** the deposit is currently **GPL-3.0** (a software license). Confirm this is intended
  for the data + model, or apply a data license such as **CC BY 4.0**.

### Notes for JA
- *Scientific Reports* **requires** a Data Availability statement; the paragraph above satisfies it.
- If the paper is scoped **temperature-only**, drop the EpHod line (pH-optimum) from the list.
- Confirm each source's exact release/version matches what was used (see `DATA.md`).
