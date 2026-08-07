"""Build a balanced HARD test set + update benchmark_v3 (training), via a
leak-free clustered split.

Pool to split = degraders (from benchmark_v3)  +  look-alike hard negatives
(lipases/esterases/abhydrolases). Easy negatives (PMBD Others / PlasticEnz)
always stay in training.

  HARD TEST SET (held out): TARGET degraders + TARGET look-alikes  (balanced)
  UPDATED benchmark_v3     : remaining degraders + remaining look-alikes
                             + all easy negatives  (now contains hard negatives)

Clustered (MinHash-LSH) split -> no enzyme or close homolog in both sides.
Erickson is already excluded from both.
"""
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from validate_degrader_finder import cluster   # MinHash-LSH clustering

BENCH = HERE / "benchmark_v3.csv"
HARD = HERE / "hard_test_set.csv"
TEST_OUT = HERE / "hard_test_set.csv"
TARGET = 8000   # degraders and look-alikes each in the test

COLS = ["accession", "protein_name", "organism", "ec_number", "enzyme_family",
        "substrate_material", "activity_label", "label_basis", "evidence_level",
        "protein_existence", "confirmed", "pfam", "has_structure", "pdb_ids",
        "temperature_c", "ph", "length", "sequence", "source"]


def align(df):
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]


def main():
    b = pd.read_csv(BENCH)
    degraders = b[b.activity_label == 1].copy()
    easy_neg = b[b.activity_label == 0].copy()          # PMBD Others + PlasticEnz
    print(f"degraders: {len(degraders)} | easy negatives: {len(easy_neg)}")

    # look-alike hard negatives from the earlier hard_test_set build
    h = pd.read_csv(HARD)
    looka = h[h.source == "UniProt-lookalike"].copy()
    looka["label_basis"] = "hard_negative"
    looka["substrate_material"] = "none"
    looka["confirmed"] = 0
    looka = align(looka)
    print(f"look-alike hard negatives: {len(looka)}")

    degraders = align(degraders)
    easy_neg = align(easy_neg)

    # ---- pool to split = degraders + look-alikes; dedup by sequence ----
    pool = pd.concat([degraders, looka], ignore_index=True)
    pool = pool.drop_duplicates("sequence").reset_index(drop=True)
    seqs = pool["sequence"].str.upper().tolist()
    is_deg = (pool.activity_label == 1).values

    print("clustering pool (MinHash-LSH)...")
    cid = cluster(seqs)
    nclust = len(np.unique(cid))
    print(f"  {len(pool)} sequences -> {nclust} clusters")

    # ---- greedy balanced cluster assignment to TEST ----
    cl_deg = defaultdict(int); cl_neg = defaultdict(int); members = defaultdict(list)
    for i, c in enumerate(cid):
        members[c].append(i)
        if is_deg[i]:
            cl_deg[c] += 1
        else:
            cl_neg[c] += 1
    clusters = list(members.keys())
    random.Random(0).shuffle(clusters)
    test_idx = set(); td = tn = 0
    for c in clusters:
        if td >= TARGET and tn >= TARGET:
            break
        # add cluster to test if it helps a quota that's not yet met
        if (cl_deg[c] and td < TARGET) or (cl_neg[c] and tn < TARGET):
            for i in members[c]:
                test_idx.add(i)
            td += cl_deg[c]; tn += cl_neg[c]
    te_mask = np.array([i in test_idx for i in range(len(pool))])

    test = pool[te_mask].reset_index(drop=True)
    train_pool = pool[~te_mask].reset_index(drop=True)

    # ---- write hard test set ----
    test.to_csv(TEST_OUT, index=False)
    # ---- updated benchmark_v3 = train_pool + easy negatives ----
    updated = pd.concat([train_pool, easy_neg], ignore_index=True).drop_duplicates("sequence")
    updated.to_csv(BENCH, index=False)

    print("\n===== HARD TEST SET =====")
    print(f"  total {len(test)}  ({int((test.activity_label==1).sum())} degrader / {int((test.activity_label==0).sum())} look-alike)")
    print("===== UPDATED benchmark_v3 (training) =====")
    print(f"  total {len(updated)}  ({int((updated.activity_label==1).sum())} degrader / {int((updated.activity_label==0).sum())} non-degrader)")
    print(f"    non-degraders = {int((updated.label_basis=='hard_negative').sum())} hard look-alikes + {len(easy_neg)} easy")

    # ---- verify no leakage between train and test ----
    test_seqs = set(test.sequence.str.upper())
    leak = updated.sequence.str.upper().isin(test_seqs).sum()
    print(f"\n  exact train/test overlap: {int(leak)} (must be 0)")
    print(f"  test duplicate sequences: {len(test)-test.sequence.nunique()}")


if __name__ == "__main__":
    main()
