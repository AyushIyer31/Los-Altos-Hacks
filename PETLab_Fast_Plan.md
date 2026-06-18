# PETLab — Fast Plan (Before the Professor)

**Goal:** walk into the meeting with a real **precision/recall result** showing PETLab can rank polymer-degrading enzymes — in **~2–3 weeks**, not 11.
**Why it's fast:** we auto-pull the benchmark instead of hand-curating it, reuse models we already built, and use Colab instead of waiting on HPC.
**Full rigorous version:** [PETLab_Summer_Plan.md](PETLab_Summer_Plan.md) — that's the fall follow-on.

---

## ⚠️ The warning — what this speed costs you

Going fast means the benchmark is **auto-pulled from databases, not hand-verified.** That's fine, but it creates 5 risks you must design around:

1. **It's noisier.** Databases label some enzymes as "predicted cutinase," not lab-confirmed degraders. So the benchmark is bigger but rougher → **call every result "preliminary."**
2. **Data leakage is the silent killer.** Auto-pulled enzymes come in families of near-identical sequences. If a close cousin sits in *both* your training and test sets, your scores look amazing but mean nothing → **you must split by sequence-similarity clusters, not randomly.**
3. **A simple similarity search may match your model.** Because your positives are defined by enzyme family, a plain BLAST "find the closest known degrader" baseline could score nearly as well. Decide your story in advance: if you beat it → "real signal beyond homology"; if you tie it → your contribution is the benchmark + honest baselining.
4. **Your negatives aren't truly negative.** A protein with "no reported activity" might just be untested. Treat them as *presumed-negative*, and for the test set use *experimentally-confirmed* positives.
5. **Condition-awareness rests only on the 65 Erickson enzymes.** Public temp/pH data is too sparse, so condition results are proof-of-concept, not a powered benchmark.

> **Bottom line:** rushing is fine *if* you (a) do the clustered split, (b) say "preliminary," and (c) know the similarity-baseline story. It only hurts you if you skip the leakage check or oversell.

---

## What we reuse (already built — saves the most time)
- ✅ **`erickson2022_degradation.csv`** — 65 enzymes with real degradation + conditions → your quantitative anchor.
- ✅ **ΔΔG / stability model** → ready-made stability-only baseline.
- ✅ **degradation model + `/simulate`** → for the Erickson quantitative check.

---

## The 5 steps

**1. Reframe the claim (½ day · laptop).**
Restate PETLab as a condition-aware *ranking* pipeline; relabel S669 as "stability validation only" so nothing oversells degradation.

**2. Auto-build the benchmark (2–4 days · laptop).**
Script-pull degraders from UniProt by EC number (`3.1.1.101` PETase, `3.1.1.74` cutinase, polyester hydrolases) plus PlasticDB; auto-pull related enzymes with no plastic activity as hard negatives; dedupe and tag each as confirmed vs predicted. → `benchmark_v1.csv`

**3. Embeddings + degrader-finder + baselines (3–4 days · laptop + Colab).**
Run ESM-2 650M on Colab to turn sequences into fingerprints (a few hours); train a simple classifier (logistic/XGBoost) to recognize degraders; build the 3 baselines — lookup, similarity, stability-only — to compare against.

**4. Validation (2–3 days · laptop) — ⚠️ the clustered split happens here.**
Hold out whole similarity clusters (no leakage), then compute precision/recall@10/@20, PR-AUC, ROC-AUC for every method; separately check whether the score tracks real degradation on Erickson; make the comparison figures. → `precision_recall_results.csv`

**5. Package for the meeting (1–2 days · laptop).**
Slides + `professor_response.md` framed as "preliminary result + fall roadmap," with limitations stated plainly.

---

## Timeline (~2–3 weeks focused)

| Days | Focus | Deliverable |
|---|---|---|
| 1 | reframe + start the pull | proven-vs-predicted table |
| 2–4 | benchmark + Erickson anchor | `benchmark_v1.csv` |
| 5–7 | embeddings + finder + baselines | trained finder + baseline scores |
| 8–10 | validation (clustered split) + figures | `precision_recall_results.csv` |
| 11–12 | slides + professor response + buffer | meeting-ready package |

---

## For the meeting

**You can say:** we built a first (preliminary) polymer-degrader benchmark, compared PETLab against lookup/similarity/stability baselines, reported precision/recall on a leak-free clustered set, used Erickson as a condition anchor, and here's the fall plan.

**Don't claim:** that PETLab made a working PET-eater, that the benchmark is final, that stability proves degradation, or that condition-awareness is fully powered.

---

## Start today
Run **Step 2 — the auto-pull** (UniProt by EC + PlasticDB) and report the counts. That gives you `benchmark_v1` and shows empirically whether the benchmark is a 2-day or 2-week job.
