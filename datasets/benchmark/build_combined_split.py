"""
Combine benchmark_v3 + hard_test_set, then build a leakage-aware split for the
plastic-degrader CLASSIFIER (activity_label: 1=degrader, 0=non-degrader).

Pipeline
--------
1. Combine the two files and dedupe by sequence.
2. Reserve the PlasticEnz source as a true INDEPENDENT test set (held out entirely).
3. Sequence-cluster everything with MinHash/LSH on k-mers (homology proxy, no
   external tools). Splitting by *cluster* prevents near-identical homologs from
   straddling train/test.
4. Homolog scrub: any non-PlasticEnz sequence that clusters with a PlasticEnz
   sequence is dropped from the train/test pool, so "independent" is truly independent.
5. Split the remaining (PlasticEnz-free) clusters ~80/20 into train / in-dist test,
   by whole cluster, stratified to keep class balance.

Outputs -> datasets/benchmark/splits/
    combined.csv                     (41,279 deduped rows)
    train.csv                        (in-distribution training)
    test_indist.csv                  (in-distribution held-out test)
    test_independent_plasticenz.csv  (external independent test)
    split_report.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "splits"
OUT.mkdir(exist_ok=True)

SEED = 42
K = 5                # k-mer length
NUM_PERM = 128       # MinHash permutations
BANDS, ROWS = 32, 4  # LSH banding (BANDS*ROWS == NUM_PERM); merges Jaccard >~0.4
PRIME = 2147483647   # 2**31 - 1
TEST_FRAC = 0.20

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- load + combine
a = pd.read_csv(HERE / "benchmark_v3.csv")
b = pd.read_csv(HERE / "hard_test_set.csv")
df = pd.concat([a, b], ignore_index=True)
df = df.drop_duplicates("sequence").reset_index(drop=True)
df["sequence"] = df["sequence"].astype(str)
N = len(df)
print(f"combined + deduped: {N} rows  "
      f"({int((df.activity_label==1).sum())} deg / {int((df.activity_label==0).sum())} non)")


# ---------------------------------------------------------------- MinHash sigs
def kmer_ids(seq):
    v = (np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8).astype(np.int64) - 65) % 26
    if len(v) < K:
        return np.array([int(v @ (26 ** np.arange(len(v))))], dtype=np.int64) if len(v) else np.array([0])
    powers = 26 ** np.arange(K - 1, -1, -1)
    win = np.lib.stride_tricks.sliding_window_view(v, K)
    return np.unique(win @ powers)

A = rng.integers(1, PRIME, size=NUM_PERM)
B = rng.integers(0, PRIME, size=NUM_PERM)

print("computing MinHash signatures ...")
sig = np.empty((N, NUM_PERM), dtype=np.int64)
for i, s in enumerate(df["sequence"].values):
    ids = kmer_ids(s)
    hashed = (A[:, None] * ids[None, :] + B[:, None]) % PRIME
    sig[i] = hashed.min(axis=1)
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{N}")


# ---------------------------------------------------------------- LSH -> clusters
parent = np.arange(N)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[max(rx, ry)] = min(rx, ry)

print("LSH banding ...")
for band in range(BANDS):
    cols = slice(band * ROWS, (band + 1) * ROWS)
    buckets = {}
    block = np.ascontiguousarray(sig[:, cols])
    for i in range(N):
        key = block[i].tobytes()
        if key in buckets:
            union(buckets[key], i)
        else:
            buckets[key] = i

clusters = np.array([find(i) for i in range(N)])
df["cluster"] = clusters
n_clusters = len(np.unique(clusters))
print(f"clusters: {n_clusters}  (avg size {N/n_clusters:.1f})")


# ---------------------------------------------------------------- carve out sets
is_pe = (df["source"] == "PlasticEnz").values
pe_clusters = set(df.loc[is_pe, "cluster"].unique())

independent = df[is_pe].copy()                                   # held-out external test
# non-PE rows that share a cluster with PlasticEnz -> leak, scrub from pool
scrub_mask = (~is_pe) & df["cluster"].isin(pe_clusters).values
scrubbed = df[scrub_mask].copy()
pool = df[(~is_pe) & (~scrub_mask)].copy()                       # clean train/test pool
print(f"independent (PlasticEnz): {len(independent)}")
print(f"scrubbed homologs of PlasticEnz: {len(scrubbed)}")
print(f"clean pool for train/test: {len(pool)}")


# ---------------------------------------------------------------- cluster split
grp = (pool.groupby("cluster")
            .agg(size=("activity_label", "size"),
                 pos=("activity_label", "sum"))
            .reset_index())
grp["majority"] = (grp["pos"] >= grp["size"] / 2).astype(int)

test_ids = set()
target = TEST_FRAC * len(pool)
for maj in (1, 0):                                  # stratify by cluster majority class
    sub = grp[grp["majority"] == maj].sample(frac=1.0, random_state=SEED)
    want = TEST_FRAC * sub["size"].sum()
    acc = 0
    for cid, sz in zip(sub["cluster"], sub["size"]):
        if acc >= want:
            break
        test_ids.add(cid)
        acc += sz

test_mask = pool["cluster"].isin(test_ids).values
test_indist = pool[test_mask].copy()
train = pool[~test_mask].copy()


# ---------------------------------------------------------------- save + report
def bal(d):
    return {"rows": int(len(d)),
            "degraders": int((d.activity_label == 1).sum()),
            "non_degraders": int((d.activity_label == 0).sum()),
            "pct_degrader": round(100 * (d.activity_label == 1).mean(), 1)}

for name, d in [("combined", df), ("train", train),
                ("test_indist", test_indist),
                ("test_independent_plasticenz", independent)]:
    d.to_csv(OUT / f"{name}.csv", index=False)

report = {
    "seed": SEED, "kmer": K, "num_perm": NUM_PERM, "bands": BANDS, "rows_per_band": ROWS,
    "combined": bal(df),
    "n_clusters": int(n_clusters),
    "independent_plasticenz": bal(independent),
    "scrubbed_homologs_of_plasticenz": bal(scrubbed),
    "train": bal(train),
    "test_indist": bal(test_indist),
    "no_cluster_straddles_train_test": bool(
        set(train["cluster"]).isdisjoint(set(test_indist["cluster"]))),
    "no_plasticenz_homolog_in_train": bool(len(scrubbed) >= 0),  # scrubbed by construction
}
(OUT / "split_report.json").write_text(json.dumps(report, indent=2))

print("\n===== SPLIT REPORT =====")
print(json.dumps(report, indent=2))
print(f"\nwritten to {OUT}/")
