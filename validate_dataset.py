"""Full accuracy/integrity audit of stability_dataset_multitask.csv.
Read-only. Reports PASS/FAIL per check with violation counts. Does not modify.
"""
import pandas as pd

F = "stability_dataset_multitask.csv"
AA = set("ACDEFGHIKLMNPQRSTVWY")
VALID_TYPES = {"ddG", "dTm", "Tm", "dH", "dCp", "thermal_shift", "abundance", "proteolysis"}
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def n_diff(wt, mut):
    if not isinstance(wt, str) or not isinstance(mut, str) or len(wt) != len(mut):
        return -1
    return sum(a != b for a, b in zip(wt, mut))


def main():
    d = pd.read_csv(F, low_memory=False)
    d["val"] = pd.to_numeric(d["measured_value"], errors="coerce")
    print(f"Rows: {len(d)}\n")

    print("== schema / values ==")
    check("no missing measured_value", d["val"].notna().all(),
          f"{d['val'].isna().sum()} NaN")
    check("all measurement_types valid", set(d.measurement_type) <= VALID_TYPES,
          str(set(d.measurement_type) - VALID_TYPES))
    check("no leftover leakage flags", ("leak_flag" not in d) or (d.get("leak_flag", 0) == 0).all())

    print("\n== sign convention (negative=stabilizing ddG; most muts destabilize) ==")
    for s, g in d[d.measurement_type == "ddG"].groupby("source_dataset"):
        pct = 100 * (g["val"] > 0).mean()
        check(f"ddG {s} majority-destabilizing(+)", pct > 50, f"{pct:.1f}% positive")

    print("\n== value ranges (units sanity) ==")
    ddg = d[d.measurement_type == "ddG"]["val"]
    check("ddG within [-25,25] kcal/mol", ddg.between(-25, 25).all(),
          f"{(~ddg.between(-25,25)).sum()} out of range")
    tm = d[d.measurement_type == "Tm"]["val"]
    check("Tm within [0,120] C", tm.between(0, 120).all(), f"{(~tm.between(0,120)).sum()} out")
    dtm = d[d.measurement_type == "dTm"]["val"]
    check("dTm within [-80,80] C", dtm.between(-80, 80).all(), f"{(~dtm.between(-80,80)).sum()} out")

    print("\n== conditions ==")
    ph = pd.to_numeric(d["ph"], errors="coerce").dropna()
    check("pH within [0,14]", ph.between(0, 14).all(), f"{(~ph.between(0,14)).sum()} out")
    t = pd.to_numeric(d["assay_temperature_c"], errors="coerce").dropna()
    check("assay T within [-20,150] C", t.between(-20, 150).all(), f"{(~t.between(-20,150)).sum()} out")

    print("\n== sequences ==")
    has_wt = d["wt_sequence"].notna() & (d["wt_sequence"].astype(str).str.len() > 0)
    wt = d.loc[has_wt, "wt_sequence"].astype(str)
    bad_aa = wt.apply(lambda s: bool(set(s.upper()) - AA)).sum()
    check("WT sequences only valid residues", bad_aa == 0, f"{bad_aa} with non-standard chars")

    # single-substitution rows: wt/mut must differ at exactly 1 position
    single = d[(d.measurement_type.isin(["ddG", "dTm", "abundance"])) &
               d.mutation.astype(str).str.match(r"^[A-Z]\d+[A-Z]$") &
               d.mut_sequence.notna() & (d.mut_sequence.astype(str).str.len() > 0)].copy()
    diffs = single.apply(lambda r: n_diff(r.wt_sequence, r.mut_sequence), axis=1)
    check("single muts: WT/mut differ at exactly 1 residue", (diffs == 1).all(),
          f"{(diffs != 1).sum()} of {len(single)} violate (incl len-mismatch)")

    # double-substitution rows (Tsuboyama doubles): differ at exactly 2
    dbl = d[d.mutation.astype(str).str.contains(":", na=False) &
            d.mut_sequence.notna() & (d.mut_sequence.astype(str).str.len() > 0)].copy()
    if len(dbl):
        dd = dbl.apply(lambda r: n_diff(r.wt_sequence, r.mut_sequence), axis=1)
        check("double muts: WT/mut differ at exactly 2 residues", (dd == 2).all(),
              f"{(dd != 2).sum()} of {len(dbl)} violate")

    # mutation notation internal consistency (wt_aa/mut_aa match the code)
    sm = d[d.mutation.astype(str).str.match(r"^[A-Z]\d+[A-Z]$") & d.wt_aa.notna()].copy()
    code_wt = sm.mutation.str[0]
    code_mut = sm.mutation.str.extract(r"([A-Z])$")[0]
    check("mutation code matches wt_aa", (code_wt == sm.wt_aa.astype(str)).all(),
          f"{(code_wt != sm.wt_aa.astype(str)).sum()} mismatched")
    check("mutation code matches mut_aa", (code_mut.values == sm.mut_aa.astype(str).values).all())
    check("wt_aa != mut_aa (no self-mutations)", (sm.wt_aa.astype(str) != sm.mut_aa.astype(str)).all(),
          f"{(sm.wt_aa.astype(str) == sm.mut_aa.astype(str)).sum()} self-muts")

    print("\n== duplicates ==")
    key = ["protein_name", "uniprot_id", "mutation", "measurement_type", "measured_value"]
    ndup = d.duplicated(subset=key).sum()
    check("no exact duplicate measurements", ndup == 0, f"{ndup} dups")

    print("\n" + "=" * 50)
    if fails:
        print(f"RESULT: {len(fails)} CHECK(S) FAILED -> {fails}")
    else:
        print("RESULT: ALL CHECKS PASSED — dataset is internally accurate.")


if __name__ == "__main__":
    main()
