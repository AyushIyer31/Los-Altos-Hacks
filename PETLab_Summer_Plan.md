# PETLab — Lean Summer Plan (Final)

**Window:** mid-June → end of August 2026 (~11 weeks)

**Main goal:** Build a defensible computational validation showing PETLab can **retrieve/rank real polymer-degrading enzymes with measurable precision and recall** under target conditions — and compare it honestly against simple baselines.

**Deferred to fall:** full wet lab, long molecular dynamics, QM/MM, final experimental PET degradation.

---

## The framing shift

- **Old:** "We predict stabilizing PETase mutations."
- **New:** "PETLab is a **condition-aware ranking pipeline** for plastic-degrading enzyme candidates. The ΔΔG model is a **stability module**, not the final validation."

**The summer win (one sentence):**
> We built a functional polymer-degrader benchmark, compared PETLab against lookup/similarity/stability baselines, and measured precision/recall on held-out degraders — using the 65 Erickson enzymes as a quantitative, condition-aware anchor.

---

## Three risks to walk in *aware of* (inherent, not plan flaws)

1. **The similarity baseline is the real test.** Positives are largely defined by enzyme family, which sequence-similarity detects well — so a BLAST/nearest-neighbor baseline may be hard to beat. Decide the narrative **before** seeing the number:
   - *Beats similarity* → "learned signal beyond homology."
   - *Ties similarity* → contribution is the **benchmark + condition-awareness + honest baselining** (still real; it's the literal answer to "why not a lookup?").
2. **Negatives are "untested," not confirmed non-degraders.** Treat them as **presumed-negative / unlabeled** (PU-learning framing). For the **held-out test set**, prefer **experimentally-confirmed** positives (`evidence_level`) so the headline metric isn't polluted by predicted-only annotations.
3. **Condition-awareness rests on the 65 Erickson enzymes.** Public temp/pH/Ca²⁺ metadata is likely too sparse in UniProt. Frame condition results as **proof-of-concept**, not a powered benchmark.

---

## Two process rules

- **Freeze a "v1 benchmark" by ~Week 3**, even if imperfect, so baselines aren't blocked. Keep improving it in parallel.
- **Week 1 is an explicit go/no-go gate:** after counting condition metadata, decide in writing — *"if <~30 positives have real temp/pH, condition-validation = Erickson only; don't sink time into UniProt condition fields."*

---

## The repeating laptop/HPC pattern

```
LAPTOP: collect data, clean tables, build FASTA/CSV
   ── scp ──▶ HPC: ESM-2 embeddings only (GPU job)
   ◀─ scp ── LAPTOP: train small models, metrics, figures
```
**HPC = only ESM-2 embeddings (+ optional docking later). Laptop = everything else.**

---

# CORE SPINE (must finish)

## Phase 0 — Reframe the claim
**Time:** 3–5 days · **Where:** laptop · **Status:** mostly done

- Rewrite headline to the condition-aware ranking framing.
- Build a **proven-vs-predicted** table.
- Relabel S669 → **"independent stability validation only."**
- Write a one-paragraph professor-facing summary + a **"what not to claim"** section.

**Outputs:** `proven_vs_predicted.md`, corrected S669 slide, `professor_summary.md`

---

## Phase 1 — Build the polymer-degrader benchmark ⭐ (highest priority, the bottleneck)
**Time:** 3–4 weeks · **Where:** laptop

Build `benchmark_polymer_conditions.csv` with columns:
```
enzyme_id, enzyme_name, uniprot_id, sequence, enzyme_family,
substrate_material, activity_label, activity_value, activity_unit,
temperature_c, ph, salt_type, calcium_mM, source, evidence_level, notes
```

**Positives** (label=1): PETases, cutinases, polyester hydrolases, PHB / PLA / PCL depolymerases, polyurethane esterases, nylon-related hydrolases.
**Negatives:**
- easy: random proteins (no enzyme/polymer annotation)
- medium: hydrolases/lipases/esterases/proteases with no polymer evidence
- **hard:** close family relatives without reported plastic activity *(most important)*

**Most important Week-1 task — the go/no-go table:**

| Source | # positives | # w/ temp | # w/ pH | # w/ salt/Ca²⁺ | usable for condition validation? |
|---|---|---|---|---|---|

**Outputs:** `benchmark_polymer_conditions.csv`, `degraders.fasta`, `hard_negatives.fasta`, `condition_metadata_availability_report.md`, dataset-summary figures.

### Phase 1B — Keep the 65 Erickson enzymes as the condition anchor *(parallel)*
Erickson = the **quantitative condition calibration set** (real degradation values at real conditions), not a side note.

**Possible metrics:** correlation (PETLab score vs degradation amount), top-k enrichment for high performers, precision/recall after binarizing high vs low degradation, condition-specific ranking sanity check.

**Outputs:** `erickson_condition_dataset.csv`, `erickson_quantitative_results.csv`

---

## Phase 2 — Baselines
**Time:** 1 week · **Where:** laptop · **Goal:** answer "why not just use a database?"

1. **Lookup** (substrate + temp + pH → known working enzymes) — the database-only baseline.
2. **Similarity** (BLAST / embedding nearest-neighbor) — the close-homolog baseline. *(The crux comparison.)*
3. **Stability-only** (ΔΔG model alone) — is stability enough?
4. **ESM degrader classifier** — does learned representation help?

**Outputs:** `lookup_baseline_results.csv`, `similarity_baseline_results.csv`, `stability_only_results.csv`, `esm_degrader_results.csv`, `baseline_comparison_table.csv`

---

## Phase 3 — ESM embeddings + degrader-finder ⭐
**Time:** 2 weeks · **Where:** laptop + HPC

- **[LAPTOP]** prepare `degraders_and_negatives.fasta` (+ metadata).
- **[HPC]** SLURM job: `facebook/esm2_t33_650M_UR50D` → embeddings. *(Not 3B unless everything else is done.)*
- **[LAPTOP]** train: logistic regression, Extra Trees / Random Forest, XGBoost/LightGBM. Most important model: **positives vs hard negatives**. Use PU-learning framing.
- **Validation splits:** random, **clustered holdout**, substrate holdout (if data), family holdout (if data).

**Outputs:** `degrader_embeddings.h5`, `degrader_finder_model.pkl`, `degrader_finder_metrics.csv`, `precision_recall_curve.png`, `clustered_holdout_results.csv`

---

## Phase 4 — Professor-answering validation ⭐
**Time:** 2 weeks · **Where:** laptop

**Question:** Can PETLab retrieve known polymer-degraders better than lookup, similarity, or stability-only?

| Method | What it proves |
|---|---|
| Lookup | database-only |
| Similarity | close-homolog |
| Stability-only | ΔΔG proxy alone |
| ESM degrader-finder | learned degrader signal |
| PETLab-lite | degrader + stability/condition |

**Metrics:** precision@10/@20, recall@10/@20, PR-AUC, ROC-AUC, enrichment over random.
**Erickson (quantitative):** correlation with degradation amount, top-k enrichment, high-vs-low precision/recall.
**Test-set rule:** held-out, experimentally-confirmed positives; clustered so no close homologs leak.

**Outputs:** `final_professor_validation_results.csv`, `final_precision_recall_curve.png`, `condition_specific_retrieval_results.csv`, `professor_response.md`

---

# STRETCH GOALS (only after the spine works)

- **A — Stability upgrade:** add ESM mutation-position embeddings, pLDDT, DSSP/RSA, FoldX. *(Original ΔΔG model is already enough.)*
- **B — Combined PETLab-lite score:** interpretable combination of degrader-likeness + stability + available condition info. **Do not pretend the weights are optimized.**
- **C — Docking (top 10–20 only):** AutoDock Vina/Smina, HPC. **Skip MD/QM-MM. Docking ≠ proof of activity.**

# CUT / DOWNGRADE (not needed to answer the professor)
- active-site preservation score (unless clean catalytic-residue annotations → simple warning flag only)
- specificity/safety filter (warning flag only)
- weighted final score as a *required* step
- any full MD / QM/MM

---

# 11-week timeline

| Weeks | Focus | Main deliverable |
|---|---|---|
| **1–4** | Benchmark: reframe, schema, positives, negatives, **condition-metadata count**, clean Erickson. **Freeze v1 by ~Wk 3.** | `benchmark_polymer_conditions.csv`, `erickson_condition_dataset.csv`, `condition_metadata_availability_report.md` |
| **5** | Baselines (lookup, similarity, stability-only) | `baseline_comparison_table.csv` |
| **6–7** | ESM-2 embeddings (HPC) + degrader-finder + hard-negative/clustered holdout | `degrader_finder_metrics.csv` |
| **8–9** | Final validation: compare all methods, precision/recall, Erickson quantitative, figures | `final_professor_validation_results.csv` |
| **10** | Positive controls (FAST-PETase/ThermoPETase) + cleanup + optional docking | `positive_control_results.csv`, `limitations.md` |
| **11** | Package: deck, professor response, GitHub cleanup, results summary, fall wet-lab plan | `PETLab_updated_deck.pdf`, `professor_response.md` |

---

# Success criteria (end of August)

> PETLab now goes beyond ΔΔG stability prediction. We built a polymer-degrader benchmark with hard negatives, compared against lookup/similarity/stability baselines, and evaluated retrieval using precision and recall. Because public condition metadata is limited, we separately use the 65 Erickson enzymes as the quantitative condition-aware anchor. The result is a ranked shortlist for wet-lab testing — **not** a claim of completed enzyme engineering.

## Do NOT claim
- PETLab created a working PET-degrading enzyme
- S669 proves degradation
- stability automatically means degradation
- docking proves activity
- environmental safety is solved

## You CAN claim
- the ΔΔG module is independently validated for **stability**
- PETLab includes a **functional polymer-degrader benchmark**
- PETLab is evaluated with **precision/recall** for degrader retrieval
- PETLab is compared against **lookup and similarity baselines**
- the output is a **ranked wet-lab shortlist**, not a finished enzyme

---

# Start today (no HPC needed)
Phase 0 + Phase 1 Week-1: pull UniProt (EC-filtered) + PlasticDB/PAZy positives and produce the **condition-metadata availability report** (the go/no-go table). This is the most important early answer.
