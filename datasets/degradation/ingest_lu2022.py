"""Ingest the schema-compatible slice of Lu et al. 2022 (Nature 604:662)
as an EXTERNAL VALIDATION set -- NOT merged into the Erickson training data.

Why separate: Lu's source data is per-figure and heterogeneous. The one clean
enzyme-activity table (Fig. 2b) reports PET-hydrolytic activity in *mM* monomer
under a single assay condition, whereas Erickson reports *uM* aromatic products
across a temperature/pH grid. Mixing units/assays would corrupt training, so we
keep this as a held-out sanity-check of relative enzyme ranking.

Source: Lu et al. 2022, Nature 604:662-667. DOI 10.1038/s41586-022-04599-z
File:   41586_2022_4599_MOESM4_ESM.xlsx (Fig. 2b)
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw" / "lu2022_MOESM4.xlsx"
OUT = HERE / "lu2022_validation.csv"
SOURCE = "Lu et al. 2022, Nature 604:662"
DOI = "10.1038/s41586-022-04599-z"

# Fig. 2b layout: rows = enzymes; two construct blocks (WT cols 1-3, LM cols 5-7),
# each three replicates. "LM" = last-mutant engineered construct from the paper.
CONSTRUCT_COLS = {"WT": [1, 2, 3], "engineered": [5, 6, 7]}


def main():
    df = pd.read_excel(RAW, "Figure 2b", header=None)
    # data rows start where col0 is an enzyme name (after the 'Unit: mM' row)
    start = next(i for i in range(len(df)) if str(df.iloc[i, 0]).startswith("Unit")) + 1
    records = []
    for r in range(start, len(df)):
        enzyme = df.iloc[r, 0]
        if pd.isna(enzyme):
            continue
        for construct, cols in CONSTRUCT_COLS.items():
            vals = [df.iloc[r, c] for c in cols if pd.notna(df.iloc[r, c])]
            vals = [float(v) for v in vals]
            if vals:
                records.append({
                    "enzyme_id": f"{str(enzyme).strip()} ({construct})",
                    "construct": construct,
                    "activity_mM_monomer": sum(vals) / len(vals),
                    "activity_stdev": pd.Series(vals).std(),
                    "n_replicates": len(vals),
                    "substrate": "amorphous_film",
                    "note": "single-condition assay; units mM (not comparable to Erickson uM)",
                    "source": SOURCE,
                    "doi": DOI,
                })
    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} validation rows -> {OUT}")
    print(out[["enzyme_id", "activity_mM_monomer", "n_replicates"]].to_string(index=False))


if __name__ == "__main__":
    main()
