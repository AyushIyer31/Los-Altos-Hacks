"""Merge the three stability sources (FireProtDB + ProDDG/S2648 + ThermoMutDB)
into ONE flat CSV for easy inspection.

Reuses the EXACT loaders from train_v46.py so the parsing, ΔΔG sign convention
(negative = stabilizing), and per-source filtering match what training sees.

This does NOT train anything — it only imports the loader functions and writes
a CSV. Output: stability_dataset_19k.csv
"""
import csv
import os
import train_v46 as t  # safe: train_v46 has a __main__ guard; only BLOSUM parse runs at import

OUT = "stability_dataset_19k.csv"
TSUBOYAMA_CSV = "datasets/stability_megascale/tsuboyama_stability.csv"  # built by convert_tsuboyama.py
COLS = ["protein_id", "position", "wt_aa", "mut_aa", "ddg",
        "temperature_c", "ph", "sequence", "source"]


def main():
    fireprot, _fp_dtm = t.load_fireprotdb()      # (records, dtm_records)
    proddg = t.load_proddg()                       # records
    thermomutdb, _dtm = t.load_thermomutdb()       # (records, dtm_records)

    all_records = fireprot + proddg + thermomutdb
    print(f"\nCombined raw records (curated DBs): {len(all_records)}")
    print(f"  FireProtDB : {len(fireprot)}")
    print(f"  ProDDG     : {len(proddg)}")
    print(f"  ThermoMutDB: {len(thermomutdb)}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in all_records:
            w.writerow({c: r.get(c, "") for c in COLS})

        # Append Tsuboyama mega-scale rows (already in this schema), if present
        n_tsu = 0
        if os.path.exists(TSUBOYAMA_CSV):
            with open(TSUBOYAMA_CSV, newline="") as tf:
                for row in csv.DictReader(tf):
                    w.writerow({c: row.get(c, "") for c in COLS})
                    n_tsu += 1
            print(f"  Tsuboyama2023: {n_tsu}")
        else:
            print(f"  Tsuboyama2023: 0 (run convert_tsuboyama.py first)")

    print(f"\nWrote {len(all_records) + n_tsu} rows -> {OUT}")


if __name__ == "__main__":
    main()
