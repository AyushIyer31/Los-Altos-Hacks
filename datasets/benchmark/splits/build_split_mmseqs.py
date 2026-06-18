"""Leakage-aware split for the plastic-degrader classifier, using MMseqs2 at
30% sequence identity (field-standard redundancy cutoff).

Fixes vs. the MinHash version:
  #2/#6  real %-identity clustering (MMseqs2 --min-seq-id 0.3 -c 0.8), so we can
         state: train/test/independent share <30% pairwise identity.
  #4     cluster-level StratifiedGroupKFold -> 5 CV folds (no cluster spans folds).
  #7     exact 80/20 train/test taken as fold 0 vs folds 1-4 -> balanced + exact.

Outputs -> datasets/benchmark/splits/mmseqs/
  combined.csv (with columns: cluster, cv_fold, partition)
  train.csv, test_indist.csv, test_independent_plasticenz.csv
  split_report.json
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
OUT = HERE / "mmseqs"
OUT.mkdir(exist_ok=True)

SEED = 42
MIN_SEQ_ID = 0.30      # 30% identity
COV = 0.80             # 80% coverage
N_FOLDS = 5

# ---------------------------------------------------------------- combine+dedupe
a = pd.read_csv(BENCH / "benchmark_v3.csv")
b = pd.read_csv(BENCH / "hard_test_set.csv")
df = pd.concat([a, b], ignore_index=True)
df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
# PlasticEnz wins on dedupe so confirmed entries are never absorbed into train
df["_pe"] = (df["source"] == "PlasticEnz").astype(int)
df = (df.sort_values("_pe", ascending=False)
        .drop_duplicates("sequence")
        .drop(columns="_pe")
        .reset_index(drop=True))
N = len(df)
print(f"combined + deduped: {N} rows "
      f"({int((df.activity_label==1).sum())} deg / {int((df.activity_label==0).sum())} non)")

# ---------------------------------------------------------------- MMseqs2 cluster
tmp = Path(tempfile.mkdtemp(prefix="mmseqs_"))
fasta = tmp / "in.fasta"
with open(fasta, "w") as fh:
    for i, s in enumerate(df["sequence"].values):
        fh.write(f">{i}\n{s}\n")

res = tmp / "clu"
cmd = ["mmseqs", "easy-cluster", str(fasta), str(res), str(tmp / "work"),
       "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COV), "--cov-mode", "0",
       "--threads", "4", "-v", "1"]
print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# parse rep<TAB>member -> cluster id per row
member2rep = {}
with open(f"{res}_cluster.tsv") as fh:
    for line in fh:
        rep, mem = line.rstrip("\n").split("\t")
        member2rep[int(mem)] = int(rep)
df["cluster"] = [member2rep[i] for i in range(N)]
n_clusters = df["cluster"].nunique()
print(f"clusters @ {int(MIN_SEQ_ID*100)}% id: {n_clusters}  (avg {N/n_clusters:.1f}/cluster)")
shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- carve sets
is_pe = (df["source"] == "PlasticEnz").values
pe_clusters = set(df.loc[is_pe, "cluster"].unique())
scrub_mask = (~is_pe) & df["cluster"].isin(pe_clusters).values   # PE homologs -> drop
df["partition"] = "pool"
df.loc[is_pe, "partition"] = "independent"
df.loc[scrub_mask, "partition"] = "scrubbed"

pool = df[df.partition == "pool"].copy()
print(f"independent (PlasticEnz): {int(is_pe.sum())}")
print(f"scrubbed PlasticEnz homologs: {int(scrub_mask.sum())}")
print(f"clean pool: {len(pool)}")

# ---------------------------------------------------------------- CV folds (#4) + 80/20 (#7)
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold = np.empty(len(pool), dtype=int)
for k, (_, te) in enumerate(sgkf.split(pool, pool["activity_label"], pool["cluster"])):
    fold[te] = k
pool["cv_fold"] = fold
df.loc[pool.index, "cv_fold"] = fold
df["cv_fold"] = df["cv_fold"].astype("Int64")

train = pool[pool.cv_fold != 0].copy()        # folds 1-4  (80%)
test_indist = pool[pool.cv_fold == 0].copy()  # fold 0     (20%)

# ---------------------------------------------------------------- verify + save
def bal(d):
    return {"rows": int(len(d)),
            "degraders": int((d.activity_label == 1).sum()),
            "non_degraders": int((d.activity_label == 0).sum()),
            "pct_degrader": round(100 * float((d.activity_label == 1).mean()), 1)}

df.to_csv(OUT / "combined.csv", index=False)
train.to_csv(OUT / "train.csv", index=False)
test_indist.to_csv(OUT / "test_indist.csv", index=False)
df[df.partition == "independent"].to_csv(OUT / "test_independent_plasticenz.csv", index=False)

train_cl, test_cl = set(train.cluster), set(test_indist.cluster)
fold_sizes = pool.groupby("cv_fold").size().to_dict()
report = {
    "method": f"MMseqs2 easy-cluster --min-seq-id {MIN_SEQ_ID} -c {COV} --cov-mode 0",
    "identity_cutoff": "<30% pairwise identity between partitions",
    "seed": SEED, "n_folds": N_FOLDS,
    "combined": bal(df),
    "n_clusters": int(n_clusters),
    "independent_plasticenz": bal(df[df.partition == "independent"]),
    "scrubbed_homologs": bal(df[df.partition == "scrubbed"]),
    "train": bal(train),
    "test_indist": bal(test_indist),
    "cv_fold_sizes": {int(k): int(v) for k, v in fold_sizes.items()},
    "train_test_ratio": f"{round(100*len(train)/len(pool))}/{round(100*len(test_indist)/len(pool))}",
    "no_cluster_straddles_train_test": bool(train_cl.isdisjoint(test_cl)),
    "no_cluster_straddles_any_fold": bool(
        all(pool.groupby("cluster")["cv_fold"].nunique() == 1)),
}
(OUT / "split_report.json").write_text(json.dumps(report, indent=2))
print("\n===== SPLIT REPORT =====")
print(json.dumps(report, indent=2))
print(f"\nwritten to {OUT}/")
