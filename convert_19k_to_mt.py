"""Convert the older flat stability_dataset_19k.csv into the multitask schema so
the EXACT Stage-2 feature pipeline (build_features.py) can train on it unchanged.
Reconstructs the mutant sequence by applying the substitution (no fabrication)."""
import csv

csv.field_size_limit(10**7)
SRC = "stability_dataset_19k.csv"
OUT = "stability_dataset_19k_mt.csv"
COLS = ["measurement_type", "wt_sequence", "mut_sequence", "measured_value",
        "position", "wt_aa", "mut_aa", "assay_temperature_c", "ph", "source_dataset"]

n_in = n_out = 0
with open(SRC) as f, open(OUT, "w", newline="") as g:
    w = csv.DictWriter(g, fieldnames=COLS)
    w.writeheader()
    for r in csv.DictReader(f):
        n_in += 1
        wt = (r["sequence"] or "").strip().upper()
        try:
            pos = int(float(r["position"]))
        except (ValueError, TypeError):
            continue
        wa, ma = r["wt_aa"].strip().upper(), r["mut_aa"].strip().upper()
        if not wt or pos < 1 or pos > len(wt) or wt[pos - 1] != wa:
            continue                      # position/WT residue must agree
        if r["ddg"].strip() == "":
            continue
        mut = wt[:pos - 1] + ma + wt[pos:]
        w.writerow({
            "measurement_type": "ddG", "wt_sequence": wt, "mut_sequence": mut,
            "measured_value": r["ddg"], "position": pos, "wt_aa": wa, "mut_aa": ma,
            "assay_temperature_c": r["temperature_c"], "ph": r["ph"],
            "source_dataset": r["source"],
        })
        n_out += 1
print(f"in {n_in} -> out {n_out} ddG rows  ({OUT})")
