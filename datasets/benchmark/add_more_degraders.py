"""Add MORE degrader-labelled sequences to the benchmark from every source we
could obtain: PMBD positive classes, PDG_DB (validated + predicted), the
SergejB-BF curated DB, and the PlasticEnz full DB.

All are degraders -> activity_label = 1. Merge with the existing benchmark,
keep existing rows on conflict, dedup by exact sequence, then scrub Erickson
(exact + substring + near-duplicate) so the held-out final exam stays clean.
"""
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
EXTRA = HERE / "extra"
PMBD = HERE / "pmbd"
PE = HERE / "plasticenz"
BENCH = HERE / "benchmark_v3.csv"
ERICKSON = HERE.parent / "degradation" / "erickson2022_degradation.csv"

VALID = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 80, 1200
COLS = ["accession", "protein_name", "organism", "ec_number", "enzyme_family",
        "substrate_material", "activity_label", "label_basis", "evidence_level",
        "protein_existence", "confirmed", "pfam", "has_structure", "pdb_ids",
        "temperature_c", "ph", "length", "sequence", "source"]


def row(acc, seq, substrate, source, evidence, confirmed=0):
    return {"accession": acc, "protein_name": "", "organism": "", "ec_number": "",
            "enzyme_family": source, "substrate_material": substrate, "activity_label": 1,
            "label_basis": source, "evidence_level": evidence, "protein_existence": "",
            "confirmed": confirmed, "pfam": "", "has_structure": 0, "pdb_ids": "",
            "temperature_c": None, "ph": None, "length": len(seq), "sequence": seq,
            "source": source}


def valid_seq(s):
    s = re.sub(r"\s", "", s).upper()
    s = re.sub(r"^H{4,}", "", s)  # strip His-tags
    return s if (s and set(s) <= VALID and MIN_LEN <= len(s) <= MAX_LEN) else None


def read_fasta(path):
    acc, seq, out = None, [], []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith(">"):
            if acc:
                out.append((acc, "".join(seq)))
            m = re.match(r">(\S+)", ln)
            acc, seq = (m.group(1) if m else "unknown"), []
        else:
            seq.append(ln.strip())
    if acc:
        out.append((acc, "".join(seq)))
    return out


def ingest():
    recs = []
    # 1. PMBD positive classes (substrate = filename)
    for sub in ["PHA", "PHB", "PU", "PVA", "Phthalate"]:
        for acc, s in read_fasta(PMBD / f"{sub}.fasta"):
            v = valid_seq(s)
            if v:
                recs.append(row(acc.split("|")[1] if "|" in acc else acc, v, sub, "PMBD", "curated_predicted"))
    # 2. PDG_DB validated
    for acc, s in read_fasta(EXTRA / "PDG_DB_protein.faa"):
        v = valid_seq(s)
        if v:
            recs.append(row(acc, v, "plastic", "PDG_DB", "validated", confirmed=1))
    # 3. PDG_DB predicted (substrate from gene_type)
    gt = {}
    for ln in open(EXTRA / "Predicted_PDGs_gene_type.tsv"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 5:
            gt[p[0]] = p[4]
    for acc, s in read_fasta(EXTRA / "Predicted_PDGs_Non-redundant.fa"):
        v = valid_seq(s)
        if v:
            recs.append(row(acc, v, gt.get(acc, "plastic"), "PDG_DB", "predicted"))
    # 4. SergejB curated (sequence embedded in a FASTA-text column)
    sdf = pd.read_excel(EXTRA / "sergej_db.xlsx")
    seqcol = [c for c in sdf.columns if "zaporedje" in c.lower()][0]
    subcol = [c for c in sdf.columns if "plastike" in c.lower()][0]
    acccol = [c for c in sdf.columns if "accession" in c.lower()][0]
    for _, r in sdf.iterrows():
        text = str(r[seqcol])
        # take the longest valid AA stretch in the cell
        best = None
        for chunk in re.split(r">[^\n]*\n?", text):
            v = valid_seq(chunk)
            if v and (best is None or len(v) > len(best)):
                best = v
        if best:
            recs.append(row(str(r[acccol])[:20], best, str(r[subcol])[:20], "SergejB", "curated"))
    # 5. PlasticEnz full DB
    for acc, s in read_fasta(PE / "PlastEnz_db.fasta"):
        v = valid_seq(s)
        if v:
            recs.append(row(acc.split("|")[1] if "|" in acc else acc, v, "plastic", "PlasticEnz", "confirmed", confirmed=1))
    return pd.DataFrame(recs, columns=COLS)


def main():
    bench = pd.read_csv(BENCH)
    new = ingest()
    print("new degrader rows ingested (pre-dedup):", len(new))
    print(new.groupby("source").size().to_string())

    # existing benchmark first -> existing wins on duplicate sequence
    merged = pd.concat([bench, new], ignore_index=True)
    before_dup = len(merged)
    merged = merged.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    print(f"\nafter dedup: {len(merged)} (removed {before_dup-len(merged)} duplicate sequences)")

    # scrub Erickson (exact + substring + near-duplicate)
    eri = sorted(set(pd.read_csv(ERICKSON)["sequence"].dropna().str.upper()))
    def kmers(s, k=6): return {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else {s}
    ek = [kmers(e) for e in eri]; idx = defaultdict(set)
    for i, km in enumerate(ek):
        for m in km: idx[m].add(i)
    def leak(b):
        b = b.upper(); bk = kmers(b); cand = set()
        for m in bk: cand |= idx.get(m, set())
        for ci in cand:
            e = eri[ci]
            if e in b or b in e or len(bk & ek[ci]) / len(bk | ek[ci]) > 0.3: return True
        return False
    before = len(merged)
    merged = merged[~merged.sequence.map(leak)].reset_index(drop=True)
    print(f"scrubbed {before-len(merged)} Erickson exact/near-duplicates")

    merged.to_csv(BENCH, index=False)
    pos = int((merged.activity_label == 1).sum()); neg = int((merged.activity_label == 0).sum())
    print(f"\nUPDATED benchmark: {len(merged)} rows  ({pos} pos / {neg} neg, {round(pos/len(merged)*100)}% pos)")
    print("  by source:", merged.source.value_counts().to_dict())


if __name__ == "__main__":
    main()
