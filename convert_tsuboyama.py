"""Convert the Tsuboyama et al. 2023 mega-scale folding-stability dataset
(Dataset2+Dataset3) into this project's stability schema and write a flat CSV.

Source: Zenodo 7992926, Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
~776K variants of ~300 mini-protein domains (cDNA-display proteolysis).

We keep clean SINGLE substitutions with a numeric ddG_ML.

Sign convention: Tsuboyama ddG_ML is positive = stabilizing (verified: raw mean
-0.76, only 22% positive -> negative = destabilizing). This pipeline uses
negative = stabilizing, so we NEGATE ddG_ML to match (same as ProDDG/ThermoMutDB).

aa_seq in mutant rows is the MUTANT sequence (aa_seq[pos-1] == mut_aa, verified
100%), so the WT sequence is reconstructed by placing the WT residue back.

Output columns match stability_dataset_19k.csv:
  protein_id, position, wt_aa, mut_aa, ddg, temperature_c, ph, sequence, source

NOTE: the proteolysis assay does not report per-mutation T/pH, so neutral
defaults (25.0 C, pH 7.0) are used — consistent with the ProDDG loader.
"""
import csv
import re

SRC = "datasets/stability_megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
OUT = "datasets/stability_megascale/tsuboyama_stability.csv"
COLS = ["protein_id", "position", "wt_aa", "mut_aa", "ddg",
        "temperature_c", "ph", "sequence", "source"]
SINGLE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def main():
    import pandas as pd
    df = pd.read_csv(SRC, usecols=["aa_seq", "mut_type", "ddG_ML", "WT_name"])
    total = len(df)

    m = df["mut_type"].str.extract(SINGLE)        # NaN for non-single rows
    df["wt_aa"], df["pos_s"], df["mut_aa"] = m[0], m[1], m[2]
    df = df[df["wt_aa"].notna()].copy()           # keep single substitutions
    df["position"] = df["pos_s"].astype(int)
    df["ddg_raw"] = pd.to_numeric(df["ddG_ML"], errors="coerce")
    df = df.dropna(subset=["ddg_raw"])            # drop unreliable "-" ddG

    n_written = 0
    n_skipped = 0
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for mut_seq, wt, pos, mut, ddg_raw, wtname in zip(
            df["aa_seq"], df["wt_aa"], df["position"], df["mut_aa"],
            df["ddg_raw"], df["WT_name"]
        ):
            # sanity: mutant sequence must carry the mutant residue at pos
            if not (isinstance(mut_seq, str) and 1 <= pos <= len(mut_seq)
                    and mut_seq[pos - 1] == mut):
                n_skipped += 1
                continue
            wt_seq = mut_seq[:pos - 1] + wt + mut_seq[pos:]   # revert to WT
            w.writerow({
                "protein_id": wtname,
                "position": pos,
                "wt_aa": wt,
                "mut_aa": mut,
                "ddg": round(-float(ddg_raw), 4),   # NEGATE -> model convention
                "temperature_c": 25.0,
                "ph": 7.0,
                "sequence": wt_seq,
                "source": "Tsuboyama2023",
            })
            n_written += 1

    print(f"Total rows in source : {total}")
    print(f"Written (single, numeric ddG): {n_written}")
    print(f"Skipped (position mismatch)  : {n_skipped}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
