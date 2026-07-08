"""Build the shared feature matrix for the classical-model zoo (run ONCE).

All the tree/boosting/MLP models predict ddG from a fixed feature vector, so we
build that matrix a single time here and save it to `features.npz`. Both the GPU
job (XGBoost/LightGBM/CatBoost) and the CPU job (RandomForest/ExtraTrees/MLP)
then just LOAD this file — features are never rebuilt three times.

Feature vector per row = [ PCA(emb_block) | engineered_block ]:
  • emb_block      = [wt_emb, mut_emb, mut_emb - wt_emb]  (ESM-2 mean-pooled),
                     reduced with PCA so 500K+ rows stay tractable for the forests.
  • engineered_block (NOT PCA'd — kept at full strength):
      - ESM-2 ΔLL: masked/wt-marginal log-likelihood of the substitution AT the
        mutated residue (log P(mut) − log P(wt)) + log P(wt), log P(mut), entropy.
        This is the mutation-LOCAL signal; mean-pooled embeddings wash it out, so
        we add it explicitly. (Same score family as ESM-1v, Meier 2021.)
      - biochemical Δ between wt/mut residue (charge, H-bonds, volume, hydropathy,
        MW, polarity, aromatic, Pro/Gly involvement).
      - assay conditions (temperature, pH).

The engineered block bypasses PCA on purpose: PCA keeps the high-variance
between-protein directions and would otherwise crush the small but decisive
per-mutation signal. Only ddG rows are used (that is what S669 tests). Split is
protein-grouped (no WT sequence in both sides).

Needs torch + fair-esm (for the one-time ESM passes). Run on a GPU node.
Output: features.npz  (X_tr, y_tr, w_tr, grp_tr, X_va, y_va, X_s669, y_s669, meta)
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

# Reuse the embedding + cleaning helpers from the neural pipeline (single source of truth).
from train_multitask import (
    DATA_CSV, S669_TSV, MAX_SEQ_LEN, SOURCE_WEIGHTS,
    _clean, get_or_build_embeddings, embed_sequences, grouped_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "features.npz")
LLR_CACHE = os.path.join(HERE, "esm_llr_cache.pkl")  # {wt_seq: {pos1: np.float32[20] logprobs}}

ESM_LLR_MODEL = "esm2_t30_150M_UR50D"  # same model family as the embeddings
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# ---- amino-acid property tables (for the biochemical-Δ features) ------------
_HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
          "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
          "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
          "Y": -1.3, "V": 4.2}
_VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
        "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
        "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
        "Y": 193.6, "V": 140.0}
_MW = {"A": 71.1, "R": 156.2, "N": 114.1, "D": 115.1, "C": 103.1, "Q": 128.1,
       "E": 129.1, "G": 57.1, "H": 137.1, "I": 113.2, "L": 113.2, "K": 128.2,
       "M": 131.2, "F": 147.2, "P": 97.1, "S": 87.1, "T": 101.1, "W": 186.2,
       "Y": 163.2, "V": 99.1}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
_HBD = {"R": 3, "K": 1, "W": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1, "C": 1}
_HBA = {"D": 2, "E": 2, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1}
_POLAR = set("RNDCQEHKSTYW")
_AROM = set("FWYH")


def _biochem(wa, ma):
    """12-d biochemical delta (mut − wt) vector; zeros if residues unknown."""
    if wa not in _MW or ma not in _MW:
        return np.zeros(12, np.float32)
    dq = _CHARGE.get(ma, 0.0) - _CHARGE.get(wa, 0.0)
    return np.asarray([
        _HYDRO[ma] - _HYDRO[wa],
        _VOL[ma] - _VOL[wa],
        _MW[ma] - _MW[wa],
        dq,
        float(dq > 0),                                   # charge gain
        float(dq < 0),                                   # charge loss
        _HBD.get(ma, 0) - _HBD.get(wa, 0),
        _HBA.get(ma, 0) - _HBA.get(wa, 0),
        float(ma in _POLAR) - float(wa in _POLAR),
        float(wa == "P" or ma == "P"),                   # proline involved
        float(wa == "G" or ma == "G"),                   # glycine involved
        float(ma in _AROM) - float(wa in _AROM),
    ], np.float32)


def _cond(temp, ph):
    """3-d condition vector [T/100, pH/14, has_both]; neutral defaults if missing."""
    t = pd.to_numeric(temp, errors="coerce")
    p = pd.to_numeric(ph, errors="coerce")
    has = float(not (pd.isna(t) or pd.isna(p)))
    return np.asarray([0.25 if pd.isna(t) else t / 100.0,
                       0.50 if pd.isna(p) else p / 14.0,
                       has], np.float32)


def _mut_fields(wt, mut, pos, wa, ma):
    """Resolve (1-indexed pos, wt_aa, mut_aa). Prefer the dataset columns; fall
    back to diffing wt vs mut for a single substitution. Returns (None,None,None)
    if the position cannot be trusted."""
    try:
        p = int(float(pos))
        a = str(wa).strip().upper()[:1]
        b = str(ma).strip().upper()[:1]
        if 1 <= p <= len(wt) and wt[p - 1] == a and b in _MW:
            return p, a, b
    except (TypeError, ValueError):
        pass
    if mut is not None and len(wt) == len(mut):
        diffs = [k for k in range(len(wt)) if wt[k] != mut[k]]
        if len(diffs) == 1:
            k = diffs[0]
            if wt[k] in _MW and mut[k] in _MW:
                return k + 1, wt[k], mut[k]
    return None, None, None


# ============================ ESM-2 ΔLL ===================================== #
def _llr_feat(cache, wt, pos, wa, ma):
    """5-d ΔLL feature [has_llr, llr, logp_wt, logp_mut, entropy]."""
    if pos is None:
        return np.zeros(5, np.float32)
    lp = cache.get(wt, {}).get(pos)
    if lp is None:
        return np.zeros(5, np.float32)
    wi, mi = AA_ORDER.index(wa), AA_ORDER.index(ma)
    p = np.exp(lp)
    ent = float(-(p * lp).sum())
    return np.asarray([1.0, float(lp[mi] - lp[wi]), float(lp[wi]), float(lp[mi]), ent], np.float32)


def build_llr(needed, device, batch_tokens=4096):
    """needed: {wt_seq: set(pos1)} -> {wt_seq: {pos1: np.float32[20] log-probs}}.

    wt-marginal: one ESM-2 forward pass per UNIQUE wt sequence yields the LM-head
    logits at every position at once, so we read off all mutated positions from a
    single pass (far cheaper than masking each position separately). Cached to disk.
    """
    cache = {}
    if os.path.exists(LLR_CACHE):
        with open(LLR_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"  loaded ΔLL cache: {len(cache)} sequences")
    todo = sorted((s for s in needed if s and s not in cache), key=len)
    if not todo:
        return cache

    import torch
    import esm
    print(f"  ΔLL: ESM-2 forward pass for {len(todo)} new sequences on {device} ...")
    model, alphabet = getattr(esm.pretrained, ESM_LLR_MODEL)()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    aa_idx = torch.tensor([alphabet.get_idx(a) for a in AA_ORDER], device=device)

    i, n, t0 = 0, len(todo), time.time()
    with torch.no_grad():
        while i < n:
            batch, toks = [], 0
            while i < n:
                s = todo[i][:MAX_SEQ_LEN]
                L = len(s) + 2
                if batch and toks + L > batch_tokens:
                    break
                batch.append(s)
                toks += L
                i += 1
            _, _, tokens = bc([(str(k), s) for k, s in enumerate(batch)])
            logits = model(tokens.to(device))["logits"]  # [B, maxL+2, vocab]
            for j, s in enumerate(batch):
                d = {}
                for pos in needed[s]:           # 1-indexed; token index = pos (BOS at 0)
                    if 1 <= pos <= len(s):
                        row = logits[j, pos, aa_idx]
                        d[pos] = torch.log_softmax(row, dim=0).float().cpu().numpy()
                cache[s] = d
            if (i % 2000) < len(batch):
                print(f"    ΔLL {i}/{n}  ({(time.time()-t0)/60:.1f} min)")
    with open(LLR_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  ΔLL cache now {len(cache)} sequences -> {LLR_CACHE}")
    return cache


# ============================ feature rows ================================== #
def _ddg_records(df):
    """Yield (wt, mut, pos, wa, ma, y, weight, temp, ph) for usable ddG rows."""
    for r in df.itertuples(index=False):
        if r.measurement_type != "ddG":
            continue
        wt, mut = _clean(r.wt_sequence), _clean(r.mut_sequence)
        if wt is None or mut is None:
            continue
        wt, mut = wt[:MAX_SEQ_LEN], mut[:MAX_SEQ_LEN]
        try:
            y = float(r.measured_value)
        except (TypeError, ValueError):
            continue
        pos, wa, ma = _mut_fields(wt, mut, r.position, r.wt_aa, r.mut_aa)
        yield (wt, mut, pos, wa, ma, y,
               SOURCE_WEIGHTS.get(r.source_dataset, 1.0), r.assay_temperature_c, r.ph)


def _feature_rows(df, emb, device):
    """Return X_emb, X_eng, y, weight, group for ddG rows that have both seqs."""
    seqs = sorted(emb.keys())
    s2i = {s: i for i, s in enumerate(seqs)}
    E = np.stack([emb[s] for s in seqs]).astype(np.float32)

    recs = [r for r in _ddg_records(df) if r[0] in s2i and r[1] in s2i]
    needed = {}
    for wt, _, pos, _, _, _, _, _, _ in recs:
        if pos is not None:
            needed.setdefault(wt, set()).add(pos)
    llr = build_llr(needed, device)

    Xe, Xg, y, w, grp, n_llr = [], [], [], [], [], 0
    for wt, mut, pos, wa, ma, yi, wi, temp, ph in recs:
        wv, mv = E[s2i[wt]], E[s2i[mut]]
        lf = _llr_feat(llr, wt, pos, wa, ma)
        n_llr += int(lf[0] > 0)
        Xe.append(np.concatenate([wv, mv, mv - wv]))
        Xg.append(np.concatenate([lf, _biochem(wa, ma), _cond(temp, ph)]))
        y.append(yi)
        w.append(wi)
        grp.append(s2i[wt])
    print(f"  ddG feature rows: {len(y)}  (ΔLL available for {n_llr})")
    return (np.asarray(Xe, np.float32), np.asarray(Xg, np.float32),
            np.asarray(y, np.float32), np.asarray(w, np.float32), np.asarray(grp))


def _s669_rows(emb_s669, device):
    s = pd.read_csv(S669_TSV, sep="\t").dropna(subset=["wt_sequence", "mutant_sequence", "ddG"])
    recs = []
    for r in s.itertuples(index=False):
        wt, mut = _clean(r.wt_sequence), _clean(r.mutant_sequence)
        if not wt or not mut:
            continue
        wt, mut = wt[:MAX_SEQ_LEN], mut[:MAX_SEQ_LEN]
        if wt not in emb_s669 or mut not in emb_s669:
            continue
        # S669 'mutation' col is like "S11A"; fall back to seq-diff inside _mut_fields.
        m = str(r.mutation).strip()
        wa = m[0] if m else None
        ma = m[-1] if m else None
        posraw = m[1:-1] if len(m) > 2 else None
        pos, wa, ma = _mut_fields(wt, mut, posraw, wa, ma)
        recs.append((wt, mut, pos, wa, ma, float(r.ddG), r.T, r.pH))

    needed = {}
    for wt, _, pos, _, _, _, _, _ in recs:
        if pos is not None:
            needed.setdefault(wt, set()).add(pos)
    llr = build_llr(needed, device)

    Xe, Xg, y, n_llr = [], [], [], 0
    for wt, mut, pos, wa, ma, ddg, temp, ph in recs:
        wv, mv = emb_s669[wt], emb_s669[mut]
        lf = _llr_feat(llr, wt, pos, wa, ma)
        n_llr += int(lf[0] > 0)
        Xe.append(np.concatenate([wv, mv, mv - wv]))
        Xg.append(np.concatenate([lf, _biochem(wa, ma), _cond(temp, ph)]))
        y.append(ddg)
    print(f"  S669 feature rows: {len(y)}  (ΔLL available for {n_llr})")
    return np.asarray(Xe, np.float32), np.asarray(Xg, np.float32), np.asarray(y, np.float32)


def _pca_fit_transform(X_tr, others, n):
    """Simple PCA via SVD on standardized training features; apply to all sets."""
    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-8
    Xc = (X_tr - mu) / sd
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:n].T  # [feat, n]
    def tr(X):
        return ((X - mu) / sd) @ comp
    return tr(X_tr), [tr(X) for X in others], (mu, sd, comp)


def _standardize(X_tr, others):
    """Z-score the engineered block (fit on train, apply to all) — no PCA."""
    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-8
    f = lambda X: (X - mu) / sd
    return f(X_tr), [f(o) for o in others]


def main(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("[1/5] loading dataset ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)

    print("[2/5] embeddings (reusing cache) ...")
    seqs = set(df.wt_sequence.map(_clean).dropna()) | set(df.mut_sequence.map(_clean).dropna())
    seqs = {s[:MAX_SEQ_LEN] for s in seqs}
    emb = get_or_build_embeddings(seqs, device)

    print("[3/5] building train feature rows (emb + ΔLL + biochem + conditions) ...")
    X_emb, X_eng, y, w, grp = _feature_rows(df, emb, device)
    tr, va = grouped_split(grp, args.val_frac, 42)

    print("[4/5] building S669 feature rows ...")
    s = pd.read_csv(S669_TSV, sep="\t").dropna(subset=["wt_sequence", "mutant_sequence"])
    s_seqs = {x[:MAX_SEQ_LEN] for x in (set(s.wt_sequence.map(_clean).dropna()) |
                                        set(s.mutant_sequence.map(_clean).dropna()))}
    Xs_emb, Xs_eng, y_s669 = _s669_rows(embed_sequences(s_seqs, device), device)

    print(f"[5/5] PCA emb -> {args.pca} dims, concat engineered ({X_eng.shape[1]}-d), saving ...")
    Etr, (Eva, Es), _ = _pca_fit_transform(X_emb[tr], [X_emb[va], Xs_emb], args.pca)
    Gtr, (Gva, Gs) = _standardize(X_eng[tr], [X_eng[va], Xs_eng])
    Xtr = np.concatenate([Etr, Gtr], 1)
    Xva = np.concatenate([Eva, Gva], 1)
    Xs = np.concatenate([Es, Gs], 1)
    np.savez_compressed(
        OUT,
        X_tr=Xtr, y_tr=y[tr], w_tr=w[tr], grp_tr=grp[tr],
        X_va=Xva, y_va=y[va],
        X_s669=Xs, y_s669=y_s669,
        meta=np.array([f"pca={args.pca}", f"eng_dims={X_eng.shape[1]}",
                       f"total_dims={Xtr.shape[1]}", f"ddG_rows={len(y)}"]))
    print(f"saved -> {OUT}  (train {tr.sum()}, val {va.sum()}, s669 {len(y_s669)}, "
          f"dims {Xtr.shape[1]})")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pca", type=int, default=256, help="PCA dims for the embedding block")
    a.add_argument("--val_frac", type=float, default=0.05)
    a.add_argument("--limit", type=int, default=0, help="subsample rows for a quick test")
    main(a.parse_args())
