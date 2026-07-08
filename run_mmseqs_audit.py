"""Sequence-homology leakage audit: staging WT sequences vs the S669 test set.

Exact-key matching cannot catch Tsuboyama domain homology to S669 (no shared
UniProt/PDB). This runs mmseqs2 to find sequence-similar proteins and flags
any staging row whose WT sequence is homologous to an S669 WT sequence.

Reports hit counts at multiple identity thresholds (defensible), and writes the
set of homology-leaking WT-sequence hashes so the final clean table can exclude
them. A staging WT seq is flagged as homologous leakage at the chosen cutoff:
    pident >= IDENT  AND  query-coverage >= COV
"""
import csv
import hashlib
import os
import subprocess
import sys

import pandas as pd

STAGING = "datasets/staging/staging_all.csv"   # ALL candidates (stable, order-independent)
S669_FILE = "s669_full.tsv"
WORK = "datasets/staging/mmseqs"
OUT_FLAGS = "datasets/staging/homology_leak_hashes.txt"

IDENT = 0.30   # >=30% identity ...
COV = 0.50     # ... over >=50% of the (shorter) query -> treat as homologous


def sha1(s):
    return hashlib.sha1(str(s).strip().upper().encode()).hexdigest() if s else ""


def write_fasta(path, seqs):
    with open(path, "w") as f:
        for sid, seq in seqs.items():
            f.write(f">{sid}\n{seq}\n")


def main():
    os.makedirs(WORK, exist_ok=True)

    # unique WT sequences from staging (id = seq hash)
    q = {}
    df = pd.read_csv(STAGING, low_memory=False)
    for s in df["wt_sequence"].dropna().astype(str).unique():
        if len(s) >= 10:
            q[sha1(s)] = s
    # S669 WT sequences (targets)
    t = {}
    s669 = pd.read_csv(S669_FILE, sep="\t")
    for s in s669["wt_sequence"].dropna().astype(str).unique():
        t[sha1(s)] = s
    print(f"unique staging WT seqs: {len(q)}  |  S669 WT seqs: {len(t)}")

    qf, tf = f"{WORK}/query.fasta", f"{WORK}/target.fasta"
    write_fasta(qf, q)
    write_fasta(tf, t)
    res = f"{WORK}/hits.m8"
    cmd = [
        "mmseqs", "easy-search", qf, tf, res, f"{WORK}/tmp",
        "-s", "7.5", "-e", "1000", "--max-seqs", "500",
        "--format-output", "query,target,pident,qcov,tcov,evalue,bits",
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)

    # best hit per query
    best = {}
    with open(res) as f:
        for line in f:
            qid, tid, pid, qcov, tcov, ev, bits = line.rstrip("\n").split("\t")
            pid, qcov = float(pid) / 100.0 if float(pid) > 1 else float(pid), float(qcov)
            cur = best.get(qid)
            if cur is None or pid > cur[0]:
                best[qid] = (pid, qcov, tid)

    # threshold distribution
    def count(idn, cov):
        return sum(1 for pid, qcov, _ in best.values() if pid >= idn and qcov >= cov)

    print("\nStaging WT seqs homologous to S669 (by identity threshold, qcov>=0.5):")
    for idn in (0.90, 0.50, 0.30, 0.25):
        print(f"  >= {int(idn*100):>3d}% id : {count(idn, COV):>5d} unique WT seqs")

    flagged = {qid for qid, (pid, qcov, _) in best.items() if pid >= IDENT and qcov >= COV}
    with open(OUT_FLAGS, "w") as f:
        f.write("\n".join(sorted(flagged)) + "\n")

    # how many staging rows do those sequences account for?
    df["wt_hash"] = df["wt_sequence"].fillna("").astype(str).map(sha1)
    n_rows = int(df["wt_hash"].isin(flagged).sum())
    print(f"\nChosen cutoff: >= {int(IDENT*100)}% id over >= {int(COV*100)}% qcov")
    print(f"Flagged unique WT seqs : {len(flagged)}")
    print(f"Staging rows they cover: {n_rows}  ({100*n_rows/len(df):.2f}% of {len(df)})")
    print(f"-> wrote {OUT_FLAGS}")


if __name__ == "__main__":
    main()
