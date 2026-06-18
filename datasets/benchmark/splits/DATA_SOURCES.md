# Data Sources & Provenance — Plastic-Degrader Classifier Splits

Combined dataset: **41,279** unique sequences (deduped) from **5 upstream databases**, split for
the degrader-vs-non-degrader classifier. Labels were assigned by **enzyme-annotation rules**
(EC number + protein name + source database), not by re-running experiments — see
"Label reliability" below.

| Split | File | Rows | Degraders | Non-degraders |
|---|---|---|---|---|
| Training | `train.csv` | 28,263 | 9,678 | 18,585 |
| In-distribution test | `test_indist.csv` | 9,839 | 4,443 | 5,396 |
| Independent test | `test_independent_plasticenz.csv` | 709 | 214 | 495 |

---

## Which source feeds which split

### Independent test set (709) — held out, model never trains on it
| Source | Rows | Reference |
|---|---|---|
| **PlasticEnz** | 709 | *PlasticEnz: An integrated database and screening tool combining homology and machine learning to identify plastic-degrading enzymes in meta-omics datasets.* PLOS Computational Biology (2025). DOI: 10.1371/journal.pcbi.1013892. Preprint: bioRxiv 2025.10.28.685028. |

> 100% experimentally **confirmed** labels — this is the gold-standard "final exam."
> *(First-author name to be confirmed from the DOI page before citing.)*

### Training set (28,263)
| Source | Rows | Reference |
|---|---|---|
| **UniProt-lookalike** | 15,872 | The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* Nucleic Acids Research (accessed via UniProt REST API, 2025). |
| **PMBD** | 8,120 | Gan Z. & Zhang H. (2019). *PMBD: a Comprehensive Plastics Microbial Biodegradation Database.* Database (Oxford) 2019:baz119. DOI: 10.1093/database/baz119. |
| **UniProtKB** | 4,136 | The UniProt Consortium (as above). |
| **SergejB** | 85 | Curated compilation (597 entries; not independently published) citing ~330 primary articles — e.g. Danso et al. 2018 AEM 84:e02773-17; Bollinger et al. 2020 Front. Microbiol. 11:114; Carniel et al. 2017 Process Biochem. 59:84-90. |
| **PDG_DB** | 50 | Zrimec J., Kokina M., Jonasson S., Zorrilla F., Zelezniak A. (2021). *Plastic-Degrading Potential across the Global Microbiome Correlates with Recent Pollution Trends.* mBio 12(5):e02155-21. DOI: 10.1128/mbio.02155-21. |

### In-distribution test set (9,839)
| Source | Rows | Reference |
|---|---|---|
| **UniProt-lookalike** | 3,958 | The UniProt Consortium (as above). |
| **PDG_DB** | 3,289 | Zrimec et al. 2021 (as above). |
| **PMBD** | 2,157 | Gan & Zhang 2019 (as above). |
| **UniProtKB** | 420 | The UniProt Consortium (as above). |
| **SergejB** | 15 | Curated compilation (as above). |

---

## Full citation list

1. **PMBD** — Gan Z. & Zhang H. (2019). *PMBD: a Comprehensive Plastics Microbial Biodegradation Database.* **Database (Oxford)** 2019:baz119. https://doi.org/10.1093/database/baz119
2. **PDG_DB** — Zrimec J., Kokina M., Jonasson S., Zorrilla F., Zelezniak A. (2021). *Plastic-Degrading Potential across the Global Microbiome Correlates with Recent Pollution Trends.* **mBio** 12(5):e02155-21. https://doi.org/10.1128/mbio.02155-21
3. **PlasticEnz** — *PlasticEnz: An integrated database and screening tool combining homology and machine learning to identify plastic-degrading enzymes in meta-omics datasets.* **PLOS Computational Biology** (2025). https://doi.org/10.1371/journal.pcbi.1013892
4. **UniProtKB / UniProt-lookalike** — The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* **Nucleic Acids Research** (accessed via UniProt REST API, 2025). https://www.uniprot.org
5. **SergejB** — Curated compilation (unpublished; 597 entries) citing ~330 primary articles. Full reference list available in `datasets/benchmark/extra/sergej_db.xlsx` ("Vir" column).

---

## Label reliability (important for any publication)

All **sequences** are real. Source databases are credible (4 peer-reviewed + UniProt).
The **labels** (degrader=1 / non-degrader=0), however, are mostly inferred, not experimentally proven:

| Tier | Train | In-dist test | Independent | Meaning |
|---|---|---|---|---|
| Experimentally confirmed | 73 | 16 | **709** | measured in a lab |
| Literature-curated | 85 | 15 | — | from primary papers |
| Predicted / homology | 9,520 | 4,412 | — | computational guess from sequence similarity |
| Assumed negatives | 18,585 | 5,396 | — | lookalike enzymes assumed non-degrading |

**How labels were assigned** (rule-based, in `build_benchmark.py` / `build_hard_test_set.py`):
- **Degrader (1):** annotated with a plastic-degrading EC number (e.g. 3.1.1.101 PETase, 3.1.1.74
  cutinase, 3.1.1.75/76 PHB/PHA depolymerase, 3.5.1.46 nylonase) or name keyword, or listed in a
  plastic-degradation database.
- **Non-degrader (0):** same α/β-hydrolase fold but a non-plastic EC (lipase 3.1.1.3, carboxylesterase
  3.1.1.1, arylesterase 3.1.1.2), with any plastic-annotated entries disqualified from the negatives.

**Safe methods wording:** *"trained on ~38k sequences from curated and computationally predicted
plastic-degradation databases (PMBD, PDG_DB/Zrimec et al., UniProt) with hard-negative decoys, and
evaluated on an independent, experimentally-confirmed hold-out (PlasticEnz)."*
Do **not** claim "trained on experimentally validated plastic degraders."
