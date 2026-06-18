"""Ingest Erickson et al. 2022 (Nat Commun 13:7850) Source Data into a tidy
PET-degradation-efficiency dataset.

Source file: 41467_2022_35237_MOESM3_ESM.xlsx (Springer static CDN)
DOI: 10.1038/s41467-022-35237-x

The source tables are wide enzyme x (temperature, pH) matrices of
"Sum of Aromatic Products" (uM TPA + MHET + BHET released) -- the direct
measure of PET hydrolysis. This script melts them to one row per
(enzyme, substrate, temperature, pH) measurement and joins protein
sequences from Tables D1/D2.

Standard endpoint screen conditions (Erickson 2022, Methods):
    enzyme loading   = 0.7 mg enzyme / g PET
    substrate loading= 2.9 % (w/v)
    reaction time    = 96 h
"""
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw" / "erickson2022_source_data.xlsx"
OUT = HERE / "erickson2022_degradation.csv"

DOI = "10.1038/s41467-022-35237-x"
SOURCE = "Erickson et al. 2022, Nat Commun 13:7850"
ENZYME_LOAD = 0.7        # mg enzyme / g PET
SUBSTRATE_LOADING = 2.9  # % w/v
ENDPOINT_TIME_H = 96.0

SUBSTRATE_LABELS = {
    "Amorphous PET Film": ("amorphous_film", 0.0),
    "Amorphous PET Powder": ("amorphous_powder", 0.0),
    "Crystalline PET Powder": ("crystalline_powder", 30.0),
}

# Per-sheet default substrate when no explicit substrate label precedes the block.
SHEET_DEFAULT_SUBSTRATE = {
    "Table D3": ("amorphous_film", 0.0),   # primary thermotolerance screen
    "Table D4": ("amorphous_film", 0.0),   # low-pH screen
    "Table D6": ("amorphous_film", 0.0),   # overridden by section labels
}


def buffer_to_ph(code: str):
    """Map a buffer code like 'NP7.5', 'C6', 'NaAc4.5', 'B8' to numeric pH."""
    if code is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(code))
    return float(m.group(1)) if m else None


def temp_to_c(val):
    if val is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    return float(m.group(1)) if m else None


# Screen-column label -> Table D2 control name (labels differ between sheets).
CONTROL_ALIASES = {
    "IsPETase": "IsPETase (WT)",
    "LCC": "LCC (WT)",
    "W159H/S238F": "IsPETase (W159H/S238F)",
}


def load_sequences(xl):
    """Build {enzyme_id_str: sequence} from Tables D1 (numeric IDs) and D2 (named)."""
    seqs = {}
    for sheet, id_col in [("Table D1", "Enzyme ID"), ("Table D2", "Name")]:
        df = xl.parse(sheet, header=None)
        # find header row containing the id column label
        hdr = next(i for i in range(len(df)) if df.iloc[i, 0] == id_col)
        body = xl.parse(sheet, header=hdr)
        for _, row in body.iterrows():
            eid = row[id_col]
            seq = row.get("Protein Sequence")
            if pd.notna(eid) and pd.notna(seq):
                seqs[str(eid).strip()] = str(seq).strip()
    # resolve control aliases used in the screen tables
    for alias, name in CONTROL_ALIASES.items():
        if name in seqs:
            seqs[alias] = seqs[name]
    return seqs


def parse_matrix_sheet(xl, sheet):
    """Melt one wide enzyme x (temp,pH) matrix sheet into tidy records."""
    df = xl.parse(sheet, header=None)
    nrows, ncols = df.shape
    records = []
    cur_substrate = SHEET_DEFAULT_SUBSTRATE.get(sheet, ("amorphous_film", 0.0))

    r = 0
    while r < nrows:
        c0 = str(df.iloc[r, 0]).strip()
        # track substrate section labels
        if c0 in SUBSTRATE_LABELS:
            cur_substrate = SUBSTRATE_LABELS[c0]
        # detect an enzyme-header block: col0 == 'Enzyme ID'
        if c0 == "Enzyme ID":
            enzyme_cols = {}
            for c in range(1, ncols):
                v = df.iloc[r, c]
                if pd.notna(v) and str(v).strip() not in ("", "St. Dev.", "nan"):
                    enzyme_cols[c] = str(v).strip()
            # data starts 2 rows below (skip the Rxn Temp / Rxn pH subheader)
            d = r + 2
            last_temp = None
            while d < nrows:
                t_raw = df.iloc[d, 0]
                p_raw = df.iloc[d, 1]
                t_str = str(t_raw).strip()
                # stop conditions: new block / substrate label / fully blank row
                if t_str == "Enzyme ID" or t_str in SUBSTRATE_LABELS:
                    break
                if pd.isna(t_raw) and pd.isna(p_raw):
                    break
                if pd.notna(t_raw) and re.search(r"\d", t_str):
                    last_temp = temp_to_c(t_raw)
                temp_c = last_temp
                ph = buffer_to_ph(p_raw)
                if temp_c is not None and ph is not None:
                    for c, eid in enzyme_cols.items():
                        val = df.iloc[d, c]
                        std = df.iloc[d, c + 1] if c + 1 < ncols else None
                        if pd.notna(val):
                            try:
                                fval = float(val)
                            except (ValueError, TypeError):
                                continue
                            records.append({
                                "enzyme_id": eid,
                                "substrate": cur_substrate[0],
                                "crystallinity_pct": cur_substrate[1],
                                "temp_C": temp_c,
                                "pH": ph,
                                "buffer": str(p_raw).strip(),
                                # Source units are mg/L ("Sum of Aromatic Products (mg/L)")
                                "aromatic_products_mg_per_L": fval,
                                "aromatic_products_stdev": float(std) if pd.notna(std) else None,
                                "src_table": sheet,
                            })
                d += 1
            r = d
            continue
        r += 1
    return records


def main():
    xl = pd.ExcelFile(RAW)
    seqs = load_sequences(xl)
    print(f"Loaded {len(seqs)} sequences from Tables D1/D2")

    all_records = []
    for sheet in ["Table D3", "Table D4", "Table D6"]:
        recs = parse_matrix_sheet(xl, sheet)
        print(f"  {sheet}: {len(recs)} measurements")
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df["sequence"] = df["enzyme_id"].map(seqs)
    df["enzyme_load_mg_per_g"] = ENZYME_LOAD
    df["substrate_loading_pct"] = SUBSTRATE_LOADING
    df["time_h"] = ENDPOINT_TIME_H
    df["source"] = SOURCE
    df["doi"] = DOI

    cols = ["enzyme_id", "sequence", "substrate", "crystallinity_pct",
            "temp_C", "pH", "buffer", "enzyme_load_mg_per_g",
            "substrate_loading_pct", "time_h", "aromatic_products_mg_per_L",
            "aromatic_products_stdev", "src_table", "source", "doi"]
    df = df[cols].sort_values(["enzyme_id", "substrate", "temp_C", "pH"])
    df.to_csv(OUT, index=False)

    n_seq = df["sequence"].notna().sum()
    print(f"\nWrote {len(df)} rows -> {OUT}")
    print(f"  unique enzymes : {df['enzyme_id'].nunique()}")
    print(f"  with sequence  : {n_seq}/{len(df)} rows")
    print(f"  substrates     : {sorted(df['substrate'].unique())}")
    print(f"  temp range     : {df['temp_C'].min():.0f}-{df['temp_C'].max():.0f} C")
    print(f"  pH range       : {df['pH'].min()}-{df['pH'].max()}")
    print(f"  nonzero activity: {(df['aromatic_products_mg_per_L'] > 0).sum()} rows")


if __name__ == "__main__":
    main()
