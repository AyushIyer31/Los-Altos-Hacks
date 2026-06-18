"""Build the PETLab polymer-degrader benchmark in ONE pass (-> benchmark_v3.csv).

Replaces the old v1->v2->v3->enrich chain. No intermediate files.

Composition:
  POSITIVES  : UniProt EC/name-annotated degraders  +  PlasticEnz positives
  NEGATIVES  : PMBD curated "Others"  +  PlasticEnz negatives
  DEDUP      : exact sequence; PlasticEnz (experimentally curated) wins
  SCRUB      : remove any sequence in the Erickson 65 (held-out final exam)
  ENRICH     : optimum temperature / pH from UniProt biophysicochemical (sparse)

Sources: UniProtKB (live), PlasticEnz (plasticenz/), PMBD CNN "Others" (pmbd/).
"""
import re
import time
import urllib.parse
import urllib.request
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
PE_DIR = HERE / "plasticenz"
PMBD_OTHERS = HERE / "pmbd" / "Others.fasta"
ERICKSON = HERE.parent / "degradation" / "erickson2022_degradation.csv"
OUT = HERE / "benchmark_v3.csv"

VALID = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 80, 1200
BASE = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = ("accession,reviewed,protein_name,organism_name,ec,length,"
          "protein_existence,ph_dependence,temp_dependence,xref_pdb,xref_pfam,sequence")

# positive degraders, EC/name only (no Pfam fold-family — too weak a label)
POSITIVES = [
    ("PET", "PET hydrolase", "ec:3.1.1.101", "EC"),
    ("PET", "PET hydrolase", "protein_name:PETase", "name"),
    ("PET", "MHET hydrolase", "ec:3.1.1.102", "EC"),
    ("polyester", "cutinase", "ec:3.1.1.74", "EC"),
    ("polyester", "polyester hydrolase", 'protein_name:"polyester hydrolase"', "name"),
    ("PHB", "PHB depolymerase", "ec:3.1.1.75", "EC"),
    ("PHO", "PHA depolymerase", "ec:3.1.1.76", "EC"),
    ("nylon", "nylon hydrolase", "ec:3.5.1.46", "EC"),
    ("nylon", "nylon endohydrolase", "ec:3.5.1.117", "EC"),
    ("polyurethane", "polyurethanase", "protein_name:polyurethan*", "name"),
    ("PLA", "PLA depolymerase", 'protein_name:polylactic* OR protein_name:"PLA depolymerase"', "name"),
    ("PCL", "PCL depolymerase", "protein_name:polycaprolacton*", "name"),
]

COLS = ["accession", "protein_name", "organism", "ec_number", "enzyme_family",
        "substrate_material", "activity_label", "label_basis", "evidence_level",
        "protein_existence", "confirmed", "pfam", "has_structure", "pdb_ids",
        "temperature_c", "ph", "length", "sequence", "source"]


def http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace"), r.headers.get("Link", "")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def parse_cond(temp_s, ph_s):
    t = re.search(r"[Oo]ptimum temperature is (\d+(?:\.\d+)?)", temp_s or "")
    p = re.search(r"[Oo]ptimum pH is (\d+(?:\.\d+)?)", ph_s or "")
    return (float(t.group(1)) if t else None, float(p.group(1)) if p else None)


def fetch_uniprot(query):
    rows, header = [], None
    url = f"{BASE}?query={urllib.parse.quote(query)}&format=tsv&fields={FIELDS}&size=500"
    while url:
        text, link = http(url)
        lines = text.splitlines()
        if header is None:
            header = lines[0].split("\t")
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) == len(header):
                rows.append(dict(zip(header, p)))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return rows


def clean(r, label, substrate, family, basis, source="UniProtKB"):
    seq = (r.get("Sequence") or "").strip().upper()
    if not seq or not set(seq) <= VALID or not (MIN_LEN <= len(seq) <= MAX_LEN):
        return None
    temp, ph = parse_cond(r.get("Temperature dependence", ""), r.get("pH dependence", ""))
    reviewed = r.get("Reviewed") == "reviewed"
    existence = r.get("Protein existence", "")
    pdb = (r.get("PDB") or "").strip().strip(";")
    return {
        "accession": r.get("Entry", ""), "protein_name": (r.get("Protein names", "") or "")[:120],
        "organism": (r.get("Organism", "") or "")[:80], "ec_number": r.get("EC number", ""),
        "enzyme_family": family, "substrate_material": substrate, "activity_label": label,
        "label_basis": basis, "evidence_level": "reviewed" if reviewed else "unreviewed",
        "protein_existence": existence,
        "confirmed": int(reviewed and existence == "Evidence at protein level"),
        "pfam": (r.get("Pfam") or "").strip().strip(";"), "has_structure": int(bool(pdb)),
        "pdb_ids": pdb[:60], "temperature_c": temp, "ph": ph, "length": len(seq),
        "sequence": seq, "source": source,
    }


def load_plasticenz():
    out = []
    for f in ["train.csv", "test.csv"]:
        for r in pd.read_csv(PE_DIR / f).itertuples():
            seq = str(r.sequence).strip().upper()
            if not seq or not set(seq) <= VALID or not (MIN_LEN <= len(seq) <= MAX_LEN):
                continue
            acc = str(r.id).split("|")[1] if "|" in str(r.id) else str(r.id)
            ec = re.search(r"EC=([\d.]+)", str(r.id))
            out.append({"accession": acc, "protein_name": "", "organism": "",
                        "ec_number": ec.group(1) if ec else "", "enzyme_family": "PlasticEnz",
                        "substrate_material": "plastic" if r.label == 1 else "none",
                        "activity_label": int(r.label), "label_basis": "PlasticEnz",
                        "evidence_level": "confirmed", "protein_existence": "", "confirmed": 1,
                        "pfam": "", "has_structure": 0, "pdb_ids": "", "temperature_c": None,
                        "ph": None, "length": len(seq), "sequence": seq, "source": "PlasticEnz"})
    return out


def load_pmbd_negatives():
    out, acc, seq = [], None, []
    for ln in open(PMBD_OTHERS):
        if ln.startswith(">"):
            if acc:
                out.append((acc, "".join(seq)))
            m = re.match(r">\w+\|([^|]+)\|", ln)
            acc, seq = (m.group(1) if m else ln[1:].split()[0]), []
        else:
            seq.append(ln.strip())
    if acc:
        out.append((acc, "".join(seq)))
    rows = []
    for a, s in out:
        s = s.upper()
        if not s or not set(s) <= VALID or not (MIN_LEN <= len(s) <= MAX_LEN):
            continue
        rows.append({"accession": a, "protein_name": "", "organism": "", "ec_number": "",
                     "enzyme_family": "PMBD-Others", "substrate_material": "none",
                     "activity_label": 0, "label_basis": "PMBD-Others",
                     "evidence_level": "curated_negative", "protein_existence": "", "confirmed": 0,
                     "pfam": "", "has_structure": 0, "pdb_ids": "", "temperature_c": None,
                     "ph": None, "length": len(s), "sequence": s, "source": "PMBD"})
    return rows


def enrich_conditions(df):
    accs = df["accession"].dropna().unique().tolist()
    tmap, pmap = {}, {}
    for i in range(0, len(accs), 90):
        q = " OR ".join(f"accession:{a}" for a in accs[i:i + 90])
        url = (f"{BASE}?query={urllib.parse.quote(q)}&format=tsv"
               f"&fields=accession,ph_dependence,temp_dependence&size=500")
        try:
            text, _ = http(url)
        except Exception:
            continue
        for ln in text.splitlines()[1:]:
            p = ln.split("\t")
            if len(p) < 3:
                continue
            t, ph = parse_cond(p[2], p[1])
            if t is not None:
                tmap[p[0]] = t
            if ph is not None:
                pmap[p[0]] = ph
    df["temperature_c"] = df.apply(lambda r: r["temperature_c"] if pd.notna(r["temperature_c"])
                                   else tmap.get(r["accession"]), axis=1)
    df["ph"] = df.apply(lambda r: r["ph"] if pd.notna(r["ph"]) else pmap.get(r["accession"]), axis=1)
    return df


def main():
    records = []
    print("Pulling UniProt positives...")
    for substrate, family, query, basis in POSITIVES:
        kept = [clean(r, 1, substrate, family, basis) for r in fetch_uniprot(query)]
        kept = [k for k in kept if k]
        print(f"  {substrate:12s} {family:22s} [{basis}] {len(kept)}")
        records += kept

    print("Loading PlasticEnz + PMBD...")
    records += load_plasticenz()
    records += load_pmbd_negatives()

    df = pd.DataFrame(records, columns=COLS)

    # dedup: PlasticEnz (curated) wins, then keep first
    df["_pref"] = (df.source == "PlasticEnz").astype(int)
    df = (df.sort_values("_pref", ascending=False)
            .drop_duplicates(subset="sequence", keep="first").drop(columns="_pref"))

    # scrub Erickson (held-out final exam) — exact + substring + near-duplicate.
    # Exact match alone misses construct/signal-peptide variants and close homologs,
    # so we remove anything substring-contained or 6-mer Jaccard > 0.3 to any Erickson seq.
    from collections import defaultdict
    eri = sorted(set(pd.read_csv(ERICKSON)["sequence"].dropna().str.upper()))

    def kmers(s, k=6):
        return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}

    ek = [kmers(e) for e in eri]
    idx = defaultdict(set)
    for i, km in enumerate(ek):
        for m in km:
            idx[m].add(i)

    def is_erickson_leak(b):
        b = b.upper()
        bk = kmers(b)
        cand = set()
        for m in bk:
            cand |= idx.get(m, set())
        for ci in cand:
            e = eri[ci]
            if e in b or b in e:
                return True
            if len(bk & ek[ci]) / len(bk | ek[ci]) > 0.3:
                return True
        return False

    before = len(df)
    df = df[~df.sequence.map(is_erickson_leak)].reset_index(drop=True)
    print(f"Scrubbed {before - len(df)} Erickson exact/near-duplicate overlaps.")

    print("Enriching temperature/pH from UniProt (this takes a couple minutes)...")
    df = enrich_conditions(df)

    df.to_csv(OUT, index=False)
    pos = int((df.activity_label == 1).sum())
    neg = int((df.activity_label == 0).sum())
    print(f"\nWrote {OUT}: {len(df)} rows  ({pos} pos / {neg} neg, {pos/len(df)*100:.0f}% pos)")
    print(f"  sources: {df.source.value_counts().to_dict()}")
    print(f"  with temp: {int(df.temperature_c.notna().sum())} | with pH: {int(df.ph.notna().sum())}")


if __name__ == "__main__":
    main()
