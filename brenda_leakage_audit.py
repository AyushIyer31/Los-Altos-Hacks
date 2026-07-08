"""Leakage audit: BRENDA temperature-stability test proteins vs the model's
TRAINING set (stability_dataset_multitask.csv).

Two layers, mirroring the project's S669 audit:
  1. EXACT keys   : UniProt accession match, or exact WT/mut sequence-hash match.
  2. HOMOLOGY     : mmseqs2, BRENDA seq vs every training seq; flag >=30% identity
                    over >=50% query coverage (same cutoff as run_mmseqs_audit.py).

Writes a CLEAN test set with every leaked protein removed, so it can serve as a
genuinely independent benchmark.

  python brenda_leakage_audit.py
"""
import csv
import hashlib
import os
import subprocess
import sys

TRAIN = "stability_dataset_multitask.csv"
TEST = "datasets/downloads/brenda_temp_stability_labeled.csv"
CLEAN_OUT = "datasets/downloads/brenda_temp_stability_clean.csv"
WORK = "datasets/downloads/brenda_mmseqs"
IDENT, COV = 0.30, 0.50

csv.field_size_limit(10**7)


def h(s):
    s = (s or "").strip().upper()
    return hashlib.md5(s.encode()).hexdigest() if s else None


def write_fasta(path, seqs):
    with open(path, "w") as f:
        for sid, seq in seqs.items():
            f.write(f">{sid}\n{seq}\n")


def main():
    os.makedirs(WORK, exist_ok=True)

    # ---- training pool: accessions + sequence hashes + unique seqs for mmseqs ----
    train_acc, train_seq_h, train_seqs = set(), set(), {}
    for r in csv.DictReader(open(TRAIN)):
        if r.get("uniprot_id", "").strip():
            train_acc.add(r["uniprot_id"].strip())
        for k in ("wt_sequence", "mut_sequence"):
            s = (r.get(k) or "").strip().upper()
            if len(s) >= 10:
                train_seq_h.add(h(s))
                train_seqs[h(s)] = s
    print(f"training: {len(train_acc)} accessions | {len(train_seqs)} unique seqs")

    # ---- BRENDA test set ----
    test = list(csv.DictReader(open(TEST)))
    for r in test:
        r["_acc_leak"] = r["accession"].strip() in train_acc
        r["_seq_leak"] = h(r["sequence"]) in train_seq_h
    print(f"test proteins: {len(test)}")

    # ---- exact-key leakage ----
    n_acc = sum(r["_acc_leak"] for r in test)
    n_seq = sum(r["_seq_leak"] for r in test)
    print(f"  exact accession leak: {n_acc}")
    print(f"  exact sequence  leak: {n_seq}")

    # ---- homology via mmseqs2 ----
    qf, tf, res = f"{WORK}/query.fasta", f"{WORK}/target.fasta", f"{WORK}/hits.m8"
    write_fasta(qf, {str(i): r["sequence"] for i, r in enumerate(test)
                     if len(r["sequence"]) >= 10})
    write_fasta(tf, train_seqs)
    cmd = ["mmseqs", "easy-search", qf, tf, res, f"{WORK}/tmp",
           "-s", "7.5", "-e", "1000", "--max-seqs", "300",
           "--format-output", "query,target,pident,qcov,tcov,evalue,bits"]
    print("running mmseqs easy-search (this can take a few minutes)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); sys.exit(1)

    best = {}
    with open(res) as f:
        for line in f:
            qid, tid, pid, qcov, tcov, ev, bits = line.rstrip("\n").split("\t")
            pid = float(pid) / 100.0 if float(pid) > 1 else float(pid)
            qcov = float(qcov)
            if qid not in best or pid > best[qid][0]:
                best[qid] = (pid, qcov)

    print("\nBRENDA seqs homologous to TRAINING (qcov>=0.5):")
    for idn in (0.90, 0.50, 0.30, 0.25):
        c = sum(1 for pid, qcov in best.values() if pid >= idn and qcov >= COV)
        print(f"  >= {int(idn*100):>3d}% id : {c:>5d}")

    for i, r in enumerate(test):
        pid, qcov = best.get(str(i), (0.0, 0.0))
        r["_homology_leak"] = pid >= IDENT and qcov >= COV
        r["_max_pident"] = round(pid, 3)

    # ---- combine ----
    for r in test:
        r["_leak"] = r["_acc_leak"] or r["_seq_leak"] or r["_homology_leak"]
    n_homol = sum(r["_homology_leak"] for r in test)
    n_leak = sum(r["_leak"] for r in test)
    clean = [r for r in test if not r["_leak"]]
    print(f"\n  homology leak (>=30% id, >=50% cov): {n_homol}")
    print(f"  TOTAL leaked (any layer):            {n_leak}  ({100*n_leak/len(test):.1f}%)")
    print(f"  CLEAN test proteins:                 {len(clean)}")

    import collections
    lab = collections.Counter(r["label"] for r in clean)
    print(f"  clean by label: pos {lab.get('positive',0)} | "
          f"neg {lab.get('negative',0)} | ambig {lab.get('ambiguous',0)}")

    cols = [c for c in test[0] if not c.startswith("_")]
    with open(CLEAN_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(clean)
    print(f"\nwrote {len(clean)} clean rows -> {CLEAN_OUT}")


if __name__ == "__main__":
    main()
