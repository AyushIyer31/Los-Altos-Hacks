d<!--
Section 5. Results  [JA]
Drafted for the two-stage (Stage 1 = temperature/Tm; Stage 2 = folding ΔΔG) manuscript
described in Manuscript_Section_Guide.pdf. NOT for technical_paper.md (that is the older
single-model paper). Cautions are set apart as labeled Note/Warning callouts (blockquotes);
they can instead be typeset as numbered footnotes if the journal prefers.
Exact values sourced from backend/make_perf_figures.py and the section guide.
-->

## 5. Results

### 5.1 Dataset composition and leakage auditing

After harmonization, the multi-source staging table comprised **1,001,888 mutation
records** drawn from seven experimental sources (**Table 1**). Because the framework
treats temperature stability and folding stability as distinct problems, records were
routed by measurement type: **29,654 melting-temperature (Tm) records** were assigned to
Stage 1 and **508,693 folding free-energy change (ΔΔG) records** to Stage 2. Abundance-proxy
and ΔTm records were retained in the table but were **not used** for model training.

To prevent optimistic bias, every candidate training record was audited against the
independent test proteins at two levels: exact-match (identical sequence and mutation) and
sequence-homology (shared homologous background). Auditing removed **approximately 132,000
records**, of which **131,479 were flagged by homology alone** and would have been invisible
to exact-match filtering. This indicates that homology-level leakage, rather than duplicate
records, is the dominant contamination risk in this setting.

### 5.2 Stage 1 — temperature-stability performance (BRENDA)

*Internal validation (development only).* During development, the Stage 1 temperature model
achieved an internal held-out RMSE of **approximately 6.5 °C** and a validation Pearson
correlation of **approximately 0.80**. These figures describe performance on data drawn from
the training distribution and are reported solely to characterize model fitting; they are not
evidence of generalization and are kept separate from the independent results below.

*Independent testing (BRENDA benchmark).* Independent evaluation used a curated BRENDA
whole-protein temperature-stability benchmark. During curation, **412 records were removed** to
yield **2,034 proteins**; a further **471 ambiguous cases were excluded**, leaving **1,563
proteins (979 positive, 584 negative)**. On this independent set the classifier reached a
**ROC AUC of 0.732 (95% CI 0.708–0.755)**. Evaluated as a regressor against measured Tm, it
obtained a **Pearson correlation of 0.54 (95% CI 0.50–0.57)**, rising to **0.60 on the subset
with directly measured true-Tm values** (**Table 2**).

*Threshold behaviour.* Classification performance depended strongly on the decision threshold
(**Fig. 1**). As the temperature cutoff was made more stringent, precision increased while
recall fell: precision rose from **0.65 (recall 0.98)** at the most permissive cutoff to
**0.93 (recall 0.39)** and **0.98 (recall 0.32)** at the strictest cutoffs (**Table 2**). The
strict-threshold regime therefore identifies a small, high-confidence set of stabilizing
candidates at the cost of missing most true positives.

> **⚠ Warning —** A precision value measured at a strict threshold is not an estimate of
> experimental success. The 0.93 precision is obtained at **39% recall**; **do not equate 0.93
> precision with 93% experimental success**, and do not read it as overall model reliability.
> The corresponding recall, F1, accuracy, and ROC AUC in **Table 2** give the complete picture.

### 5.3 Stage 2 — ΔΔG stabilization performance (S669)

Independent folding-stability performance was assessed on the S669 benchmark of **669 single
mutations (168 stabilizing, 501 non-stabilizing)**. Across **2,000 bootstrap resamples**, the
gradient-boosted ensemble achieved a **Pearson correlation of 0.390 (95% CI 0.32–0.46)** and a
**ROC AUC of 0.669 (95% CI 0.62–0.72)** for stabilizing/non-stabilizing classification
(**Table 3**). Among individual models, **Extra Trees — not the ensemble — produced the
strongest single-model regression metrics**.

As in Stage 1, precision and recall traded off with threshold stringency (**Fig. 1**). At the
strict ΔΔG threshold, precision reached **0.81 at a recall of 0.10**; relaxing the threshold
lowered precision to **0.51 (recall 0.26)**, **0.38 (recall 0.46)**, and **0.31 (recall 0.80)**
(**Table 3**). The model thus operates as a high-precision, low-recall filter at strict cutoffs:
the top-ranked candidates are enriched for true stabilizers, but the majority of stabilizing
mutations are not recovered.

> **Note —** The Stage 2 low recall is a reported property of the model, not an omission. The
> headline **0.81 precision corresponds to only 10% recall**; it indicates that a small set of
> top-ranked mutations is enriched for stabilizers and is intended to support prioritization of
> candidates for experimental testing, not exhaustive recovery of every stabilizing mutation.

### 5.4 Cross-stage comparison

Stage 1 achieved a higher independent ROC AUC (**0.732**) than Stage 2 (**0.669**). However,
the two stages were trained and evaluated on **different datasets, prediction targets
(temperature stability vs. folding ΔΔG), class labels, and sample sizes (1,563 vs. 669
records)**. The comparison is therefore descriptive only: it is **not a controlled head-to-head
evaluation, and no statistical significance test was performed** between the two stages. Any
difference in the reported metrics should be interpreted in light of these differences rather
than as evidence that one stage is intrinsically superior.
