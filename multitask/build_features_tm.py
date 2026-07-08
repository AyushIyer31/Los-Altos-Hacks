"""Build features for a WHOLE-PROTEIN stability model: sequence -> Tm (deg C).

Unlike build_features.py (ddG, which needs a WT+mutant PAIR per row), this model
takes a SINGLE sequence and predicts its melting temperature. Training rows are
the Tm measurements in the staging set (Meltome + FireProtDB Tm). It also embeds
the leakage-audited BRENDA temperature-stability test set so the trainer can score
generalization on enzymes the model never saw.

Feature vector = ESM-2 mean-pooled embedding (640-d) + log(length). No PCA (the
Tm set is ~23K rows, so the forests stay tractable on full embeddings).

Reuses the embedding cache + helpers from train_multitask.py (single source of
truth for the ESM model, cleaning, and the grouped split).

Run on a GPU node (needs torch + fair-esm). Output: features_tm.npz
"""
import os

import numpy as np
import pandas as pd

from train_multitask import (
    DATA_CSV, MAX_SEQ_LEN, _clean, get_or_build_embeddings, grouped_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "features_tm.npz")
BRENDA = os.path.join(BASE, "brenda_temp_stability_clean.csv")


def tm_table(df):
    """{sequence: mean Tm} over all Tm rows (a sequence can recur -> average)."""
    acc = {}
    for r in df.itertuples(index=False):
        if r.measurement_type != "Tm":
            continue
        s = _clean(r.wt_sequence)
        if not s:
            continue
        s = s[:MAX_SEQ_LEN]
        try:
            y = float(r.measured_value)
        except (TypeError, ValueError):
            continue
        acc.setdefault(s, []).append(y)
    return {s: float(np.mean(v)) for s, v in acc.items()}


def featurize(seq, emb):
    return np.concatenate([emb[seq], [np.log1p(len(seq))]]).astype(np.float32)


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("[1/4] loading Tm rows ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    tm = tm_table(df)
    print(f"  unique Tm sequences: {len(tm)}")

    print("[2/4] loading BRENDA test set ...")
    bdf = pd.read_csv(BRENDA)
    bdf["_seq"] = [(_clean(s) or "")[:MAX_SEQ_LEN] for s in bdf["sequence"]]
    bdf = bdf[bdf["_seq"].str.len() >= 5].reset_index(drop=True)
    print(f"  BRENDA test proteins: {len(bdf)}")

    print("[3/4] embeddings (reusing cache) ...")
    emb = get_or_build_embeddings(set(tm) | set(bdf["_seq"]), device)

    print("[4/4] assembling matrices ...")
    seqs = [s for s in tm if s in emb]
    X = np.stack([featurize(s, emb) for s in seqs])
    y = np.asarray([tm[s] for s in seqs], np.float32)
    grp = np.arange(len(seqs))            # one sample per unique sequence
    tr, va = grouped_split(grp, 0.1, 42)

    bX = np.stack([featurize(s, emb) for s in bdf["_seq"]])
    np.savez_compressed(
        OUT,
        X_tr=X[tr], y_tr=y[tr], X_va=X[va], y_va=y[va],
        X_brenda=bX,
        brenda_temp=bdf["stable_temp_c"].astype(float).to_numpy(),
        brenda_label=bdf["label"].to_numpy(),
        brenda_basis=bdf["basis"].to_numpy(),
    )
    print(f"saved -> {OUT}  (train {int(tr.sum())}, val {int(va.sum())}, "
          f"brenda {len(bX)}, dims {X.shape[1]})")


if __name__ == "__main__":
    main()
