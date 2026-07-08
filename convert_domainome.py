"""Convert Human Domainome 1.0 (aPCA abundance) into the staging schema.

Source: Zenodo 13629491, Extended_data_Table_2 (per-variant aPCA fitness).
Measurement: normalized_fitness = abundance-based stability PROXY (not ddG).

Quirk: `aa_seq` is the MUTANT domain sequence (residue at the mutated site is
already the mutant), and `position` is protein-numbered with a per-domain
offset. We solve each domain's offset (protein_pos -> aa_seq index) from the
data, then reconstruct WT by reverting the mutated residue.
"""
import csv
from collections import Counter

import pandas as pd

SRC = "datasets/downloads/domainome_table2.txt"
OUT = "datasets/downloads/domainome_stability.csv"
AA = set("ACDEFGHIKLMNPQRSTVWY")
COLS = ["domain_ID", "uniprot_ID", "wt_sequence", "mut_sequence",
        "mutation", "position", "wt_aa", "mut_aa", "measured_value", "source"]


def solve_offset(sub):
    """Most consistent s where protein_pos p maps to aa_seq index (p - s)."""
    c = Counter()
    for aa_seq, p, mut_aa in zip(sub["aa_seq"], sub["position"], sub["mut_aa"]):
        s = str(aa_seq)
        for idx, ch in enumerate(s):
            if ch == mut_aa:
                c[int(p) - idx] += 1
    return c.most_common(1)[0][0] if c else None


def main():
    df = pd.read_csv(SRC, sep="\t",
                     usecols=["domain_ID", "uniprot_ID", "aa_seq", "wt_aa",
                              "position", "mut_aa", "STOP", "normalized_fitness"])
    df = df[(df["STOP"] == False) & df["mut_aa"].isin(AA) & df["wt_aa"].isin(AA)]
    df = df.dropna(subset=["normalized_fitness"]).copy()
    df["position"] = df["position"].astype(int)

    n_written = n_skipped = 0
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for dom, sub in df.groupby("domain_ID"):
            s = solve_offset(sub)
            if s is None:
                n_skipped += len(sub); continue
            for _, r in sub.iterrows():
                mut_seq = str(r["aa_seq"])
                idx = int(r["position"]) - s
                if not (0 <= idx < len(mut_seq) and mut_seq[idx] == r["mut_aa"]):
                    n_skipped += 1; continue
                wt_seq = mut_seq[:idx] + r["wt_aa"] + mut_seq[idx + 1:]
                w.writerow({
                    "domain_ID": dom, "uniprot_ID": r["uniprot_ID"],
                    "wt_sequence": wt_seq, "mut_sequence": mut_seq,
                    "mutation": f"{r['wt_aa']}{r['position']}{r['mut_aa']}",
                    "position": r["position"], "wt_aa": r["wt_aa"], "mut_aa": r["mut_aa"],
                    "measured_value": round(float(r["normalized_fitness"]), 5),
                    "source": "Domainome",
                })
                n_written += 1

    print(f"Written: {n_written}   Skipped (offset/align): {n_skipped}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
