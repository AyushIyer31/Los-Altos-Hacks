"""Baseline degrader-finder validation on benchmark_v3.

Trains a classifier (sequence-composition features -- no ESM yet) to predict
degrader vs non-degrader, and reports accuracy + precision/recall/F1/AUC.

Two splits:
  * RANDOM     -> optimistic (homologs leak across train/test)
  * CLUSTERED  -> honest (similar sequences kept on the same side)

This is a BASELINE (composition features); ESM embeddings would be stronger.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
import xgboost as xgb

HERE = Path(__file__).parent
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_I = {a: i for i, a in enumerate(AA)}
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}


def feats(seq):
    n = max(len(seq), 1)
    comp = np.zeros(20)
    for c in seq:
        if c in AA_I:
            comp[AA_I[c]] += 1
    comp /= n
    arom = sum(seq.count(a) for a in "FWY") / n
    pos = sum(seq.count(a) for a in "KRH") / n
    neg = sum(seq.count(a) for a in "DE") / n
    gravy = sum(KD.get(c, 0) for c in seq) / n
    return np.concatenate([comp, [len(seq), arom, pos, neg, gravy]])


def cluster(seqs, k=5, num_hashes=12, bands=6):
    """Fast MinHash-LSH clustering -> cluster id per seq (homologs grouped).

    Union-find over LSH bands: sequences sharing any banded MinHash signature
    are merged. O(N * num_hashes), scales to 20k+ sequences.
    """
    import random
    rnd = random.Random(0)
    masks = [rnd.getrandbits(64) for _ in range(num_hashes)]

    def sig(s):
        kms = {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else {s}
        hs = [hash(m) for m in kms] or [0]
        return [min((h ^ msk) & 0xFFFFFFFFFFFFFFFF for h in hs) for msk in masks]

    parent = list(range(len(seqs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows = num_hashes // bands
    buckets = defaultdict(list)
    for i, s in enumerate(seqs):
        sg = sig(s)
        for b in range(bands):
            key = (b, tuple(sg[b*rows:(b+1)*rows]))
            buckets[key].append(i)
    for members in buckets.values():
        for j in members[1:]:
            union(members[0], j)
    roots = [find(i) for i in range(len(seqs))]
    remap = {r: c for c, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


def evaluate(Xtr, ytr, Xte, yte):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, eval_metric="logloss",
                          n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, p),
        "pr_auc": average_precision_score(yte, p),
    }


def main():
    df = pd.read_csv(HERE / "benchmark_v3.csv")
    seqs = df["sequence"].str.upper().tolist()
    y = df["activity_label"].values
    print(f"benchmark: {len(df)} ({(y==1).sum()} pos / {(y==0).sum()} neg)")
    print("building features...")
    X = np.vstack([feats(s) for s in seqs])

    rng = np.random.RandomState(0)
    # ---- RANDOM split (optimistic) ----
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]
    rand = evaluate(X[tr], y[tr], X[te], y[te])

    # ---- CLUSTERED split (honest) ----
    print("clustering sequences (homology-aware split)...")
    cid = cluster(seqs)
    clusters = np.unique(cid)
    rng.shuffle(clusters)
    test_clusters = set(clusters[:int(0.2 * len(clusters))])
    te_mask = np.array([c in test_clusters for c in cid])
    n_clusters = len(clusters)
    clus = evaluate(X[~te_mask], y[~te_mask], X[te_mask], y[te_mask])

    print(f"\n  distinct sequence clusters: {n_clusters}")
    print(f"  clustered test size: {int(te_mask.sum())}  ({int(y[te_mask].sum())} pos / {int((y[te_mask]==0).sum())} neg)\n")
    hdr = f"{'metric':<12}{'RANDOM (leaky)':>16}{'CLUSTERED (honest)':>20}"
    print(hdr); print("-" * len(hdr))
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{k:<12}{rand[k]:>16.3f}{clus[k]:>20.3f}")
    print("\nNote: CLUSTERED is the honest number; RANDOM is inflated by homolog leakage.")
    print("Baseline = composition features only; ESM embeddings would raise this.")


if __name__ == "__main__":
    main()
