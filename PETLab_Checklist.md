# PETLab — Guided Execution Checklist

Companion to [PETLab_Summer_Plan.md](PETLab_Summer_Plan.md). Work top to bottom. Tags:
**[LAPTOP]** local work · **[HPC]** GPU job · **[BG]** background (start early, runs in parallel) · **[GATE]** decision point — stop and decide.

---

## ⏱️ DAY 1 — Pre-flight (do these first, today)

- [ ] **[BG]** Submit the **HPC access / GPU allocation request** (even though first use is ~Week 6). *This is the hidden 1-week calendar cost.*
- [ ] **[BG]** Set up a **cloud-GPU fallback** account (Colab Pro / RunPod) in case HPC queue stalls.
- [ ] **[LAPTOP]** Create the repo folders:
  ```
  PETLab/data/{raw,processed,embeddings}  scripts/  notebooks/  models/  results/  figures/  reports/
  ```
- [ ] **[LAPTOP]** Confirm you already have `erickson2022_degradation.csv` (built earlier) → copy into `data/processed/`.

---

## WEEK 1 — Reframe + benchmark schema + the GATE

### Phase 0 — Reframe (2–3 days)
- [ ] **[LAPTOP]** Rewrite project headline to the condition-aware ranking framing.
- [ ] **[LAPTOP]** Write `proven_vs_predicted.md` (two-column table).
- [ ] **[LAPTOP]** Relabel the S669 slide → "independent **stability** validation only."
- [ ] **[LAPTOP]** Write `professor_summary.md` (one paragraph) + a "what not to claim" list.

### Phase 1 start — Benchmark schema + first pull
- [ ] **[LAPTOP]** Create empty `benchmark_polymer_conditions.csv` with the 16 columns.
- [ ] **[LAPTOP]** Pull **positives** from **UniProt REST API** by EC: `3.1.1.101` (PETase), `3.1.1.74` (cutinase), polyester hydrolases. Save sequences + annotations.
- [ ] **[LAPTOP]** Pull from **PlasticDB** and **PAZy** (other plastics).
- [ ] **[LAPTOP]** For each positive, record whatever **temperature / pH / salt / Ca²⁺** metadata exists.

### 🚦 [GATE 1] — Condition-metadata go/no-go (END OF WEEK 1)
- [ ] **[LAPTOP]** Produce `condition_metadata_availability_report.md`:

  | Source | # positives | # w/ temp | # w/ pH | # w/ salt/Ca²⁺ | usable? |
- [ ] **[GATE]** **Decide in writing:** if **< ~30** positives have real temp/pH →
  *"Condition validation = Erickson only. Do NOT sink more time into UniProt condition fields."*

---

## WEEK 2 — Collect positives (all materials)

- [ ] **[LAPTOP]** Expand positives to all classes: PETases, cutinases, polyester hydrolases, **PHB / PLA / PCL depolymerases, polyurethane esterases, nylon hydrolases**.
- [ ] **[LAPTOP]** Add `enzyme_family`, `substrate_material`, `source`, `evidence_level` for each.
- [ ] **[LAPTOP]** Mark which positives are **experimentally confirmed** vs **predicted-only** (`evidence_level`).
- [ ] **[LAPTOP]** Dedupe sequences (e.g., MMseqs2 / CD-HIT at 90–95% identity).
- [ ] **Output:** `positives_cleaned.csv`

---

## WEEK 3 — Negatives + v1 FREEZE

- [ ] **[LAPTOP]** Collect **easy negatives** (random UniProt proteins, no enzyme/polymer annotation).
- [ ] **[LAPTOP]** Collect **medium negatives** (hydrolases/lipases/esterases/proteases, no polymer evidence).
- [ ] **[LAPTOP]** Collect **hard negatives** (close family relatives, no reported plastic activity). ← *most important*
- [ ] **[LAPTOP]** Label as **presumed-negative / unlabeled** (PU framing), not "confirmed negative."
- [ ] **[LAPTOP]** Merge positives + negatives → finalize `benchmark_polymer_conditions.csv`.
- [ ] **[LAPTOP]** Export `degraders.fasta` and `hard_negatives.fasta`.

### 🚦 [GATE 2] — Freeze v1 benchmark (END OF WEEK 3)
- [ ] **[GATE]** **Freeze `benchmark_polymer_conditions.csv` v1** — even if imperfect. Downstream starts now. Keep improving in a *separate* v2 copy.

---

## WEEK 4 — Erickson anchor + benchmark figures

### Phase 1B — Erickson (parallel, mostly done)
- [ ] **[LAPTOP]** Reformat `erickson2022_degradation.csv` → `erickson_condition_dataset.csv` (condition columns aligned to benchmark schema).
- [ ] **[LAPTOP]** Define **high-vs-low degradation** labels (binarize) for precision/recall use.
- [ ] **[LAPTOP]** Save `erickson_quantitative_results.csv` template (for Phase 4 metrics).

### Benchmark QC
- [ ] **[LAPTOP]** Make dataset-summary figures: substrate distribution, temp/pH distribution, positive/negative counts.
- [ ] **[LAPTOP]** Write `dataset_summary.md`.
- [ ] **[BG]** Confirm HPC access is live; test a tiny SLURM job + load ESM-2 650M.

---

## WEEK 5 — Baselines (Phase 2)

- [ ] **[LAPTOP]** **Lookup baseline:** table mapping (substrate, temp, pH) → known working enzymes. → `lookup_baseline_results.csv`
- [ ] **[LAPTOP]** **Similarity baseline:** BLAST / MMseqs2 nearest-neighbor to known degraders. → `similarity_baseline_results.csv` *(the crux comparison)*
- [ ] **[LAPTOP]** **Stability-only baseline:** rank by existing ΔΔG model. → `stability_only_results.csv`
- [ ] **[LAPTOP]** Build `baseline_comparison_table.csv` (precision/recall placeholders).

---

## WEEKS 6–7 — ESM embeddings + degrader-finder (Phase 3)

### Laptop prep
- [ ] **[LAPTOP]** Build `degraders_and_negatives.fasta` + label table.
- [ ] **[LAPTOP]** Write the embedding script (`scripts/run_esm_embeddings.py`) + SLURM batch file.

### Hand-off + HPC
- [ ] **[LAPTOP→HPC]** `scp degraders_and_negatives.fasta user@hpc:/scratch/petlab/`
- [ ] **[HPC]** `sbatch run_esm.sh` → run **`facebook/esm2_t33_650M_UR50D`** → `degrader_embeddings.h5` *(hours–2 days incl. queue)*.
- [ ] **[HPC→LAPTOP]** `scp` embeddings back to `data/embeddings/`.

### Train + validate
- [ ] **[LAPTOP]** Train: logistic regression, Extra Trees / Random Forest, XGBoost/LightGBM.
- [ ] **[LAPTOP]** Train the key model: **positives vs hard negatives**.
- [ ] **[LAPTOP]** Run **clustered holdout** (cluster by similarity, hold out whole clusters).
- [ ] **[LAPTOP]** Run **substrate holdout** + **family holdout** (if enough data).
- [ ] **Outputs:** `degrader_finder_model.pkl`, `degrader_finder_metrics.csv`, `precision_recall_curve.png`, `clustered_holdout_results.csv`

---

## WEEKS 8–9 — Final professor-answering validation (Phase 4)

- [ ] **[LAPTOP]** Build a **held-out test set** of **experimentally-confirmed** positives, clustered so no close homologs leak.
- [ ] **[LAPTOP]** Score the test set with **all 5 methods**: lookup, similarity, stability-only, ESM degrader-finder, PETLab-lite.
- [ ] **[LAPTOP]** Compute **precision@10/@20, recall@10/@20, PR-AUC, ROC-AUC, enrichment**.
- [ ] **[LAPTOP]** **Erickson quantitative:** correlation (score vs degradation), top-k enrichment, high-vs-low precision/recall.
- [ ] **[LAPTOP]** Make `final_precision_recall_curve.png` + `final_model_comparison.png`.
- [ ] 🚦 **[GATE 3] Pick the narrative** (before spinning numbers): *beats similarity* → "signal beyond homology"; *ties similarity* → "benchmark + condition-awareness + honest baselining."
- [ ] **[LAPTOP]** Write `professor_response.md`.
- [ ] **Outputs:** `final_professor_validation_results.csv`, `condition_specific_retrieval_results.csv`

---

## WEEK 10 — Positive controls + cleanup

- [ ] **[LAPTOP]** Sanity-check known engineered variants (FAST-PETase, ThermoPETase) — *controls, not validation*. → `positive_control_results.csv`
- [ ] **[LAPTOP]** Write `limitations.md` (incl. similarity-baseline caveat, PU negatives, Erickson-only condition data).
- [ ] **[HPC] (optional, only if core done)** Dock PET/MHET/BHET against top 10–20 → `optional_docking_results.csv`. *Docking ≠ proof of activity.*

---

## WEEK 11 — Final package

- [ ] **[LAPTOP]** Update slide deck (suggested order: problem → why stability alone isn't enough → PETLab → ΔΔG module + S669 → degrader-finder → benchmark → baselines → precision/recall → condition ranking → top candidates → limitations + fall plan).
- [ ] **[LAPTOP]** Finalize `professor_response.md`.
- [ ] **[LAPTOP]** Clean GitHub repo + `README.md`.
- [ ] **[LAPTOP]** Write `final_results_summary.md` + the **fall wet-lab plan** (top-5 variants → Tm + degradation assays).

---

## ✅ Done-when (end-of-August success criteria)
- [ ] Benchmark with hard negatives exists and is documented.
- [ ] Baselines (lookup, similarity, stability-only) computed.
- [ ] Precision/recall on held-out degraders reported vs all baselines.
- [ ] Erickson used as the quantitative condition anchor.
- [ ] Honest claims only (no "we built a working PET-eater").
- [ ] Ranked wet-lab shortlist produced.

---

## 🔑 The 3 things that protect the timeline
1. **Start the HPC access request on Day 1** (background) — or it adds a week.
2. **Hard-freeze the benchmark at end of Week 3** — Phase 1 is the only row likely to overrun.
3. **Start the benchmark pull today** — the 11 weeks only fits if Week 1 begins now.
