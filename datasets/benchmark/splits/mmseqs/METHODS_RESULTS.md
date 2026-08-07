# Degrader-Finder: Leakage-Aware Methodology & Results

## Methodology (addresses strict-review fixes)

1. **Combine + dedupe.** benchmark_v3 + hard_test_set -> 41,279 unique sequences
   (exact-sequence dedup; PlasticEnz retained on ties).
2. **Cluster at 30% identity (fix #2/#6).** `mmseqs easy-cluster --min-seq-id 0.3
   -c 0.8 --cov-mode 0` -> 3,616 clusters. **All partitions share <30% pairwise
   identity** — a defined, field-standard cutoff (not a heuristic).
3. **Independent holdout.** PlasticEnz (experimentally **confirmed** labels, n=709)
   held out entirely. **9,471** of its homologs were found and scrubbed from the
   training pool — vs only 2,468 caught by the prior k-mer method, i.e. the old
   split leaked ~7,000 near-homologs of the test set into training.
4. **Cluster-grouped 5-fold CV + exact 80/20 (fix #4/#7).** StratifiedGroupKFold;
   no cluster spans folds; 80/20 = folds 1-4 vs fold 0, class-balanced.
5. **Honest metrics (fix #3).** AUROC + AUPRC + bootstrap 95% CIs; precision
   re-estimated at realistic (~1%) prevalence.

Model: repo degrader-finder design — XGBoost on 25 composition features.

## Results

| Evaluation | AUROC | AUPRC |
|---|---|---|
| In-distribution, cluster 5-fold CV | **0.971 ± 0.009** | **0.954 ± 0.016** |
| Independent (PlasticEnz, confirmed) | **0.627** [0.577–0.672] | **0.489** [0.422–0.559] |
| Random baseline (independent) | 0.50 | 0.302 |

**Precision vs. prevalence (independent set, threshold 0.5):** TPR 0.69, FPR 0.54.
Precision = 0.35 at the 30% test mix, **0.06 at 5% prevalence, 0.013 at 1%**.

## Honest interpretation (what to tell the reviewer)

- **The model generalizes well within the source distribution** (0.97 AUROC across
  sequence-separated clusters) **but only weakly to a different, experimentally-
  confirmed database** (0.63 AUROC; CI lower bound 0.58 > 0.5, so above chance but
  modest). This 0.97 -> 0.63 gap is the real generalization story.
- **The drop is expected and is the point of the protocol.** It reflects (a)
  annotation circularity — in-distribution labels come from EC/homology rules the
  composition model partly re-learns (fix #1), and (b) cross-database domain shift
  (fix #5). The rigorous split *surfaces* this instead of hiding it.
- **At realistic prevalence (~1%), screening precision is ~1%** — i.e. most top
  hits would be false positives. Composition features alone are not sufficient for
  real-world degrader discovery; richer features (e.g. ESM embeddings) are the
  likely next step.

**Bottom line:** the methodology is now defensible (defined identity cutoff,
cluster-grouped CV, CIs, confirmed external holdout, prevalence-aware precision).
The honest performance is modest on novel confirmed sequences — better to report
this than to present a leakage-inflated 0.97.
