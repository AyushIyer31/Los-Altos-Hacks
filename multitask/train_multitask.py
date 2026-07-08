"""Train the multi-task stability model on stability_dataset_multitask.csv.

Pipeline (all driven from this one file):
  1. Precompute ESM-2 mean-pooled embeddings for every UNIQUE sequence (WT + mut),
     cache to disk so the GPU runs the language model only once, not every epoch.
  2. Build per-row tensors: (wt_vec, mut_vec, condition, head_index, target, weight).
  3. Per-head target standardisation (z-score) so the large-magnitude Tm head does
     not dominate the loss of the small-magnitude ddG head.
  4. Per-head + per-source confidence weighting so the 458K-row abundance PROXY and
     the megascale Tsuboyama data do not drown the ~20K rows of gold thermodynamic
     signal (see SOURCE_WEIGHTS / HEAD_WEIGHTS below — these are the knobs to tune).
  5. Protein-grouped train/val split (no wild-type sequence appears in both).
  6. Train with early stopping; evaluate the ddG head on the held-out S669 set.
  7. Save model weights + config (scalers, weights, args) to OUT_DIR.

NOTE: requires `torch` and `fair-esm` (pip install fair-esm). The embedding step
needs a GPU to be fast; everything after it is light. Nothing here auto-runs on
import — call via `python train_multitask.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from model import MEASUREMENT_TYPES, TYPE_TO_IDX, COND_DIM, MultiTaskStabilityNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "stability_dataset_multitask.csv")
S669_TSV = os.path.join(BASE_DIR, "s669_full.tsv")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
EMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esm_meanpool_cache.pkl")

ESM_MODEL = "esm2_t30_150M_UR50D"  # 640-dim, matches the rest of the repo
ESM_LAYER = 30
EMB_DIM = 640
MAX_SEQ_LEN = 1022  # ESM-2 context limit (minus BOS/EOS); longer seqs are truncated
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---- the tunable emphasis knobs --------------------------------------------
# Down-weight the weak abundance proxy; lift the small gold thermodynamic sets.
HEAD_WEIGHTS = {"ddG": 1.0, "dTm": 1.0, "Tm": 0.5, "abundance": 0.3}
SOURCE_WEIGHTS = {
    "FireProtDB": 1.5, "ProDDG_S2648": 1.5, "ThermoMutDB": 1.5,  # gold thermodynamic
    "Tsuboyama2023": 1.0, "Tsuboyama2023_double": 1.0,           # silver (megascale)
    "Domainome": 1.0, "Meltome": 1.0,                            # proxy / Tm (head-weighted)
}


# ============================ ESM embedding ================================= #
_esm_state = {}


def _load_esm(device):
    if "model" not in _esm_state:
        import esm
        print(f"  loading {ESM_MODEL} ...")
        model, alphabet = getattr(esm.pretrained, ESM_MODEL)()
        model = model.to(device).eval()
        _esm_state["model"] = model
        _esm_state["bc"] = alphabet.get_batch_converter()
    return _esm_state["model"], _esm_state["bc"]


def _clean(seq):
    if not isinstance(seq, str):
        return None
    seq = "".join(a for a in seq.upper() if a in VALID_AA)
    return seq if len(seq) >= 5 else None


def embed_sequences(seqs, device, batch_tokens=4096):  # 4096 fits ~11GB GPUs
    """Return {sequence: np.float32[640]} mean-pooled ESM-2 embeddings.

    Sequences are length-sorted and packed into batches under a token budget so
    short Tsuboyama domains run fast while long proteins stay within memory.
    """
    import torch

    uniq = sorted({s for s in seqs if s}, key=len)
    model, bc = _load_esm(device)
    out = {}
    i, n = 0, len(uniq)
    t0 = time.time()
    with torch.no_grad():
        while i < n:
            # greedily fill a batch under the token budget
            batch, toks = [], 0
            while i < n:
                s = uniq[i][:MAX_SEQ_LEN]
                L = len(s) + 2
                if batch and toks + L > batch_tokens:
                    break
                batch.append((str(len(batch)), s))
                toks += L
                i += 1
            _, _, tokens = bc(batch)
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
            for j, (_, s) in enumerate(batch):
                v = reps[j, 1:len(s) + 1].mean(0).float().cpu().numpy()  # mean-pool residues
                out[s] = v
            if (i % 2000) < len(batch):
                print(f"    embedded {i}/{n}  ({(time.time()-t0)/60:.1f} min)")
    return out


def get_or_build_embeddings(seqs, device):
    cache = {}
    if os.path.exists(EMB_CACHE):
        with open(EMB_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"  loaded embedding cache: {len(cache)} sequences")
    missing = [s for s in seqs if s and s not in cache]
    if missing:
        print(f"  embedding {len(missing)} new sequences on {device} ...")
        cache.update(embed_sequences(missing, device))
        with open(EMB_CACHE, "wb") as f:
            pickle.dump(cache, f)
        print(f"  cache now {len(cache)} sequences -> {EMB_CACHE}")
    return cache


# ============================ data assembly ================================= #
def build_rows(df, emb):
    """Turn the dataframe into aligned numpy arrays for training."""
    wt_idx, mut_idx, conds, head_idx, targets, weights, groups = [], [], [], [], [], [], []
    # Map each unique embedded sequence to a row in a stacked matrix E.
    seqs = sorted(emb.keys())
    s2i = {s: i for i, s in enumerate(seqs)}
    E = np.stack([emb[s] for s in seqs]).astype(np.float32)

    skipped = 0
    for r in df.itertuples(index=False):
        mt = r.measurement_type
        if mt not in TYPE_TO_IDX:
            skipped += 1
            continue
        wt = _clean(r.wt_sequence)
        mut = _clean(r.mut_sequence)
        if wt is None:
            skipped += 1
            continue
        wt = wt[:MAX_SEQ_LEN]
        mut = (mut[:MAX_SEQ_LEN] if mut is not None else wt)
        if wt not in s2i or mut not in s2i:
            skipped += 1
            continue
        try:
            y = float(r.measured_value)
        except (TypeError, ValueError):
            skipped += 1
            continue

        t = pd.to_numeric(r.assay_temperature_c, errors="coerce")
        p = pd.to_numeric(r.ph, errors="coerce")
        has = float(not (pd.isna(t) or pd.isna(p)))
        cond = [0.0 if pd.isna(t) else t / 100.0,
                0.0 if pd.isna(p) else p / 14.0,
                has]

        wt_idx.append(s2i[wt])
        mut_idx.append(s2i[mut])
        conds.append(cond)
        head_idx.append(TYPE_TO_IDX[mt])
        targets.append(y)
        weights.append(HEAD_WEIGHTS[mt] * SOURCE_WEIGHTS.get(r.source_dataset, 1.0))
        groups.append(s2i[wt])  # group by WT protein for the split

    print(f"  assembled {len(targets)} rows  (skipped {skipped})")
    return (E, np.array(wt_idx), np.array(mut_idx), np.array(conds, np.float32),
            np.array(head_idx), np.array(targets, np.float32),
            np.array(weights, np.float32), np.array(groups))


def standardize_targets(targets, head_idx):
    """Per-head z-score. Returns standardized targets + {head: (mean, std)}."""
    scalers, y = {}, targets.copy()
    for t, hi in TYPE_TO_IDX.items():
        m = head_idx == hi
        if m.sum() == 0:
            scalers[t] = (0.0, 1.0)
            continue
        mu, sd = float(targets[m].mean()), float(targets[m].std() or 1.0)
        scalers[t] = (mu, sd)
        y[m] = (targets[m] - mu) / sd
    return y, scalers


def grouped_split(groups, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val_groups = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    return ~is_val, is_val


# ============================ training ====================================== #
def run(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(args.out, exist_ok=True)

    print("\n[1/5] loading dataset ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    print(f"  {len(df):,} rows")

    print("\n[2/5] embeddings ...")
    all_seqs = set(df.wt_sequence.map(_clean).dropna())
    all_seqs |= set(df.mut_sequence.map(_clean).dropna())
    all_seqs = {s[:MAX_SEQ_LEN] for s in all_seqs}
    emb = get_or_build_embeddings(all_seqs, device)

    print("\n[3/5] assembling tensors ...")
    E, wt_idx, mut_idx, conds, head_idx, targets, weights, groups = build_rows(df, emb)
    y, scalers = standardize_targets(targets, head_idx)
    tr, va = grouped_split(groups, args.val_frac, args.seed)
    print(f"  train {tr.sum():,} | val {va.sum():,}")

    E_t = torch.from_numpy(E).to(device)

    def loader(mask, shuffle):
        ds = TensorDataset(
            torch.from_numpy(wt_idx[mask]), torch.from_numpy(mut_idx[mask]),
            torch.from_numpy(conds[mask]), torch.from_numpy(head_idx[mask]),
            torch.from_numpy(y[mask]), torch.from_numpy(weights[mask]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, drop_last=False)

    tr_dl, va_dl = loader(tr, True), loader(va, False)

    print("\n[4/5] training ...")
    net = MultiTaskStabilityNet(emb_dim=EMB_DIM, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    def weighted_mse(pred_all, hidx, target, w):
        pred = pred_all.gather(1, hidx.view(-1, 1)).squeeze(1)
        se = (pred - target) ** 2 * w
        return se.sum() / (w.sum() + 1e-8)

    best, bad = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        net.train()
        for wi, mi, c, hi, tgt, w in tr_dl:
            wi, mi, c, hi, tgt, w = [x.to(device) for x in (wi, mi, c, hi, tgt, w)]
            opt.zero_grad()
            out = net(E_t[wi], E_t[mi], c)
            loss = weighted_mse(out, hi, tgt, w)
            loss.backward()
            opt.step()

        net.eval()
        tot, den = 0.0, 0.0
        with torch.no_grad():
            for wi, mi, c, hi, tgt, w in va_dl:
                wi, mi, c, hi, tgt, w = [x.to(device) for x in (wi, mi, c, hi, tgt, w)]
                out = net(E_t[wi], E_t[mi], c)
                tot += weighted_mse(out, hi, tgt, w).item() * len(tgt)
                den += len(tgt)
        val = tot / max(den, 1)
        print(f"  epoch {ep:3d}  val_wmse {val:.4f}")
        if val < best - 1e-4:
            best, bad = val, 0
            torch.save(net.state_dict(), os.path.join(args.out, "model_best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (no improvement for {args.patience} epochs)")
                break

    # save config
    cfg = dict(esm_model=ESM_MODEL, emb_dim=EMB_DIM, hidden=args.hidden,
               measurement_types=MEASUREMENT_TYPES, scalers=scalers,
               head_weights=HEAD_WEIGHTS, source_weights=SOURCE_WEIGHTS,
               cond_layout=["temp/100", "pH/14", "has_condition"], args=vars(args))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved -> {args.out}/model_best.pt + config.json")

    print("\n[5/5] S669 evaluation (ddG head) ...")
    evaluate_s669(net, scalers, device)


def evaluate_s669(net, scalers, device):
    import torch
    if not os.path.exists(S669_TSV):
        print("  s669_full.tsv not found — skipping.")
        return
    s = pd.read_csv(S669_TSV, sep="\t")
    s = s.dropna(subset=["wt_sequence", "mutant_sequence", "ddG"]).copy()
    seqs = set(s.wt_sequence.map(_clean).dropna()) | set(s.mutant_sequence.map(_clean).dropna())
    emb = embed_sequences({x[:MAX_SEQ_LEN] for x in seqs}, device)

    mu, sd = scalers["ddG"]
    preds, obs = [], []
    net.eval()
    with torch.no_grad():
        for r in s.itertuples(index=False):
            wt, mut = _clean(r.wt_sequence), _clean(r.mutant_sequence)
            if not wt or not mut:
                continue
            wt, mut = wt[:MAX_SEQ_LEN], mut[:MAX_SEQ_LEN]
            if wt not in emb or mut not in emb:
                continue
            t = pd.to_numeric(r.T, errors="coerce")
            p = pd.to_numeric(r.pH, errors="coerce")
            has = float(not (pd.isna(t) or pd.isna(p)))
            cond = torch.tensor([[0.0 if pd.isna(t) else t / 100.0,
                                  0.0 if pd.isna(p) else p / 14.0, has]],
                                dtype=torch.float32, device=device)
            wv = torch.tensor(emb[wt][None], device=device)
            mv = torch.tensor(emb[mut][None], device=device)
            out = net(wv, mv, cond)[0, TYPE_TO_IDX["ddG"]].item()
            preds.append(out * sd + mu)  # de-standardize back to kcal/mol
            obs.append(float(r.ddG))
    if len(preds) < 5:
        print("  too few overlapping S669 rows to score.")
        return
    preds, obs = np.array(preds), np.array(obs)
    pear = np.corrcoef(preds, obs)[0, 1]
    rank = np.corrcoef(np.argsort(np.argsort(preds)), np.argsort(np.argsort(obs)))[0, 1]
    rmse = float(np.sqrt(((preds - obs) ** 2).mean()))
    print(f"  S669 n={len(preds)}  Pearson={pear:.3f}  Spearman={rank:.3f}  RMSE={rmse:.3f} kcal/mol")


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--epochs", type=int, default=60)
    a.add_argument("--batch", type=int, default=1024)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--wd", type=float, default=1e-5)
    a.add_argument("--hidden", type=int, default=512)
    a.add_argument("--dropout", type=float, default=0.2)
    a.add_argument("--val_frac", type=float, default=0.05)
    a.add_argument("--patience", type=int, default=8)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--limit", type=int, default=0, help="subsample N rows for a quick smoke test")
    a.add_argument("--out", type=str, default=DEFAULT_OUT)
    return a.parse_args()


if __name__ == "__main__":
    run(parse_args())
