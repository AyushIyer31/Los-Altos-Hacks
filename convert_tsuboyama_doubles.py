"""Extract Tsuboyama2023 DOUBLE mutants (multi-substitution rows) from the local
mega-scale file into a flat CSV for staging.

Source: datasets/stability_megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
Multi rows use 'A5G:K10R' notation (':' separator); all are exactly 2 subs.

Same conventions as convert_tsuboyama.py:
  - keep only rows with numeric ddG_ML (drop unreliable "-")
  - NEGATE ddG_ML  (Tsuboyama positive=stabilizing -> model negative=stabilizing)
  - aa_seq is the MUTANT sequence (both subs applied); reconstruct WT by reverting
  - neutral nominal conditions (25 C / pH 7.0), proteolysis assay
"""
import csv
import re

import pandas as pd

SRC = "datasets/stability_megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
OUT = "datasets/stability_megascale/tsuboyama_doubles.csv"
COLS = ["protein_id", "mutation", "ddg", "temperature_c", "ph",
        "wt_sequence", "mut_sequence", "source"]
SUB = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def main():
    df = pd.read_csv(SRC, usecols=["aa_seq", "mut_type", "ddG_ML", "WT_name"])
    df = df[df["mut_type"].str.contains(":", na=False)].copy()
    df["ddg_raw"] = pd.to_numeric(df["ddG_ML"], errors="coerce")
    df = df.dropna(subset=["ddg_raw"])

    n_written = n_skipped = 0
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for mut_seq, mut_type, ddg_raw, wtname in zip(
            df["aa_seq"], df["mut_type"], df["ddg_raw"], df["WT_name"]
        ):
            if not isinstance(mut_seq, str):
                n_skipped += 1
                continue
            subs = [SUB.fullmatch(s) for s in mut_type.split(":")]
            if any(s is None for s in subs):
                n_skipped += 1
                continue
            wt_seq = mut_seq
            ok = True
            for s in subs:
                wt_aa, pos, m_aa = s.group(1), int(s.group(2)), s.group(3)
                if not (1 <= pos <= len(mut_seq) and mut_seq[pos - 1] == m_aa):
                    ok = False
                    break
                wt_seq = wt_seq[:pos - 1] + wt_aa + wt_seq[pos:]   # revert this sub
            if not ok:
                n_skipped += 1
                continue
            w.writerow({
                "protein_id": wtname,
                "mutation": mut_type,
                "ddg": round(-float(ddg_raw), 4),
                "temperature_c": 25.0,
                "ph": 7.0,
                "wt_sequence": wt_seq,
                "mut_sequence": mut_seq,
                "source": "Tsuboyama2023_double",
            })
            n_written += 1

    print(f"Written (double, numeric ddG): {n_written}")
    print(f"Skipped (mismatch/parse)     : {n_skipped}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
