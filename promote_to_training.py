"""Promote the audited staging table -> the final multi-task TRAINING dataset.

Reads datasets/staging/staging_clean.csv (already leak-free + de-duplicated),
runs hard integrity guards, drops audit-only flag columns, and writes the
canonical multi-task training file.

Does NOT overwrite the legacy single-column stability_dataset_19k.csv.
Output: stability_dataset_multitask.csv
"""
import sys

import pandas as pd

SRC = "datasets/staging/staging_clean.csv"
OUT = "stability_dataset_multitask.csv"
DROP = ["leak_flag", "leak_reason", "dup_flag"]  # audit-only, redundant once clean


def main():
    df = pd.read_csv(SRC, low_memory=False)

    # ---- hard integrity guards: refuse to promote contaminated data ----
    assert (df["leak_flag"].astype(str) == "0").all(), "ABORT: leaked rows present!"
    assert (df["dup_flag"].astype(str) == "0").all(), "ABORT: duplicate rows present!"
    bad_type = set(df["measurement_type"]) - {
        "ddG", "dTm", "Tm", "dH", "dCp", "thermal_shift", "abundance", "proteolysis"}
    assert not bad_type, f"ABORT: non-stability measurement types: {bad_type}"
    assert df["measured_value"].notna().all(), "ABORT: rows with no measured value!"

    out = df.drop(columns=[c for c in DROP if c in df.columns])
    out.to_csv(OUT, index=False)

    print(f"Promoted {len(out)} clean rows -> {OUT}")
    print(f"Columns: {list(out.columns)}")
    print("\nBy measurement_type:")
    print(out["measurement_type"].value_counts().to_string())
    print("\nBy source_dataset:")
    print(out["source_dataset"].value_counts().to_string())
    up = out["wt_seq_hash"].fillna("").astype(str)
    up = up.where(up != "", out["uniprot_id"].fillna(out["protein_name"]).astype(str))
    print(f"\nUnique proteins: {up.nunique()}")
    print("Legacy stability_dataset_19k.csv: left UNTOUCHED.")


if __name__ == "__main__":
    main()
