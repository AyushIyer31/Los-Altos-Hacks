"""Leakage audit: does the older 19K training data overlap the S669 test set?
Exact (protein, mutation, mutant-seq) + mmseqs2 homology (>=30% id, >=50% qcov).
Mirrors the project's standard audit so the 19K model's S669 score can be judged."""
import csv
import hashlib
import os
import subprocess

csv.field_size_limit(10**7)
S669 = "s669_full.tsv"
K19 = "stability_dataset_19k.csv"
WORK = "datasets/downloads/s669_19k_mmseqs"
IDENT, COV = 0.30, 0.50


def h(s):
    s = (s or "").strip().upper()
    return hashlib.md5(s.encode()).hexdigest() if s else None


# ---- S669 ----
s_wt, s_mut_seq, s_mutkey, s_wt_list = set(), set(), set(), {}
with open(S669) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        wt = (r.get("wt_sequence") or "").strip().upper()
        mut = (r.get("mutant_sequence") or "").strip().upper()
        m = (r.get("mutation") or "").strip()
        if wt:
            s_wt.add(h(wt)); s_wt_list[h(wt)] = wt
        if mut:
            s_mut_seq.add(h(mut))
        if wt and len(m) >= 3 and m[1:-1].isdigit():
            s_mutkey.add((h(wt), int(m[1:-1]), m[0].upper(), m[-1].upper()))
print(f"S669: {len(s_wt)} unique WT seqs, {len(s_mutkey)} mutation keys")

# ---- 19K ----
k_wt, k_mut_seq, k_mutkey, k_wt_uniq = set(), set(), set(), {}
with open(K19) as f:
    for r in csv.DictReader(f):
        wt = (r.get("sequence") or "").strip().upper()
        if not wt:
            continue
        k_wt.add(h(wt)); k_wt_uniq[h(wt)] = wt
        try:
            pos = int(float(r["position"]))
        except (ValueError, TypeError):
            continue
        wa, ma = r["wt_aa"].strip().upper(), r["mut_aa"].strip().upper()
        if 1 <= pos <= len(wt) and wt[pos - 1] == wa:
            k_mutkey.add((h(wt), pos, wa, ma))
            k_mut_seq.add(h(wt[:pos - 1] + ma + wt[pos:]))
print(f"19K:  {len(k_wt)} unique WT seqs, {len(k_mutkey)} mutation keys\n")

# ---- EXACT overlap ----
wt_ov = s_wt & k_wt
mut_ov = s_mutkey & k_mutkey
mutseq_ov = s_mut_seq & k_mut_seq
print("=== EXACT overlap (S669 ∩ 19K) ===")
print(f"  S669 WT proteins also in 19K : {len(wt_ov)} / {len(s_wt)} "
      f"({100*len(wt_ov)/len(s_wt):.1f}%)")
print(f"  S669 exact mutations in 19K  : {len(mut_ov)} / {len(s_mutkey)} "
      f"({100*len(mut_ov)/max(len(s_mutkey),1):.1f}%)")
print(f"  S669 mutant-seq in 19K       : {len(mutseq_ov)} / {len(s_mut_seq)}")

# ---- HOMOLOGY (mmseqs2): S669 WT (query) vs 19K WT (target) ----
os.makedirs(WORK, exist_ok=True)
qf, tf, res = f"{WORK}/q.fasta", f"{WORK}/t.fasta", f"{WORK}/hits.m8"
with open(qf, "w") as f:
    for hh, s in s_wt_list.items():
        if len(s) >= 10:
            f.write(f">{hh}\n{s}\n")
with open(tf, "w") as f:
    for hh, s in k_wt_uniq.items():
        if len(s) >= 10:
            f.write(f">{hh}\n{s}\n")
print("\nrunning mmseqs2 (S669 WT vs 19K WT) ...")
r = subprocess.run(["mmseqs", "easy-search", qf, tf, res, f"{WORK}/tmp",
                    "-s", "7.5", "-e", "1000", "--max-seqs", "300",
                    "--format-output", "query,target,pident,qcov"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-1000:])
best = {}
with open(res) as f:
    for line in f:
        q, t, pid, qcov = line.rstrip("\n").split("\t")
        pid = float(pid) / 100 if float(pid) > 1 else float(pid)
        qcov = float(qcov)
        if q not in best or pid > best[q][0]:
            best[q] = (pid, qcov)
print("\n=== HOMOLOGY: S669 WT proteins with a homolog in 19K (qcov>=0.5) ===")
for idn in (0.90, 0.50, 0.30):
    c = sum(1 for pid, qcov in best.values() if pid >= idn and qcov >= COV)
    print(f"  >= {int(idn*100):>3d}% id : {c:>4d} / {len(s_wt)} ({100*c/len(s_wt):.1f}%)")
leaked = sum(1 for pid, qcov in best.values() if pid >= IDENT and qcov >= COV)
print(f"\n=> S669 proteins homologous to 19K training: {leaked}/{len(s_wt)} "
      f"({100*leaked/len(s_wt):.1f}%) at >=30% id, >=50% cov")
