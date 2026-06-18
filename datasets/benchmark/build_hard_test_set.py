"""Build a HARD, independent test set for the degrader-finder.

  POSITIVES : PlasticEnz degraders (491) + PlasticEnz negatives (503) seed
  HARD NEG  : look-alike enzymes (lipases, carboxylesterases, other
              alpha/beta-hydrolases = same fold as cutinases/PETases) with NO
              plastic-degradation annotation -> the decoys that make it HARD.
  RULES     : no repeats (exact dedup), independent of the training benchmark
              and Erickson (near-duplicate scrub).

Target ~20k unique sequences (mostly hard negatives -- the realistic task of
finding rare real degraders among many look-alikes).
"""
import re
import time
import urllib.parse
import urllib.request
import pandas as pd
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
PE = HERE / "plasticenz"
BENCH = HERE / "benchmark_v3.csv"
ERICKSON = HERE.parent / "degradation" / "erickson2022_degradation.csv"
OUT = HERE / "hard_test_set.csv"

VALID = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 80, 1200
BASE = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,reviewed,protein_name,organism_name,ec,length,sequence"
# look-alike (same alpha/beta-hydrolase fold), NOT plastic degraders:
HARD_NEG = [
    ("lipase",            "ec:3.1.1.3", 10000),
    ("carboxylesterase",  "ec:3.1.1.1", 6000),
    ("arylesterase",      "ec:3.1.1.2", 2000),
    ("ab-hydrolase fold", "xref:pfam-PF00561", 4000),
]
# disqualify anything that looks like an actual degrader
PLASTIC = re.compile(r"cutin|petase|polyester|depolymeras|terephthalat|plastic|"
                     r"polyurethan|polycaprolac|polylactic|nylon|hydroxybutyrate|"
                     r"hydroxyalkanoate", re.I)
DEG_EC = ("3.1.1.101", "3.1.1.74", "3.1.1.75", "3.1.1.76", "3.5.1.46", "3.1.1.102")


def valid(s):
    s = str(s).strip().upper()
    return s if (s and set(s) <= VALID and MIN_LEN <= len(s) <= MAX_LEN) else None


def http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace"), r.headers.get("Link", "")
        except Exception:
            if a == 2:
                raise
            time.sleep(2)


def fetch(query, cap):
    rows, header = [], None
    url = f"{BASE}?query={urllib.parse.quote(query)}&format=tsv&fields={FIELDS}&size=500"
    while url and len(rows) < cap:
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
    return rows[:cap]


def load_plasticenz():
    out = []
    for f in ["train.csv", "test.csv"]:
        for r in pd.read_csv(PE / f).itertuples():
            v = valid(r.sequence)
            if v:
                acc = str(r.id).split("|")[1] if "|" in str(r.id) else str(r.id)
                out.append({"accession": acc, "protein_name": "", "organism": "",
                            "ec_number": "", "enzyme_family": "PlasticEnz",
                            "activity_label": int(r.label), "evidence_level": "confirmed",
                            "length": len(v), "sequence": v, "source": "PlasticEnz"})
    return out


def main():
    records = load_plasticenz()
    print(f"PlasticEnz seed: {len(records)}")

    print("pulling look-alike hard negatives from UniProt...")
    for fam, query, cap in HARD_NEG:
        raw = fetch(query, cap)
        kept = 0
        for r in raw:
            v = valid(r.get("Sequence", ""))
            name = r.get("Protein names", "") or ""
            ec = r.get("EC number", "") or ""
            if not v or PLASTIC.search(name) or any(d in ec for d in DEG_EC):
                continue
            records.append({"accession": r.get("Entry", ""), "protein_name": name[:100],
                            "organism": (r.get("Organism", "") or "")[:80], "ec_number": ec,
                            "enzyme_family": fam, "activity_label": 0,
                            "evidence_level": "hard_negative", "length": len(v),
                            "sequence": v, "source": "UniProt-lookalike"})
            kept += 1
        print(f"  {fam:20s} pulled {len(raw):5d}  kept {kept:5d}")

    df = pd.DataFrame(records)

    # no repeats: exact sequence dedup (PlasticEnz wins)
    df["_p"] = (df.source == "PlasticEnz").astype(int)
    df = df.sort_values("_p", ascending=False).drop_duplicates("sequence", keep="first").drop(columns="_p")

    # Independence (source-holdout protocol):
    #   * hard negatives (UniProt-lookalike): must NOT be in training -> scrub vs
    #     (training benchmark minus PlasticEnz) AND Erickson.
    #   * PlasticEnz rows: KEEP all (they're the held-out positives); only scrub vs
    #     Erickson so the final exam stays separate. Independence from training is
    #     enforced by EXCLUDING PlasticEnz from training when this test is used.
    def km(s, k=6): return {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else {s}
    def make_checker(ref_seqs):
        ref = sorted(set(ref_seqs)); rk = [km(s) for s in ref]; idx = defaultdict(set)
        for i, kk in enumerate(rk):
            for m in kk: idx[m].add(i)
        def leaks(s):
            s = s.upper(); sk = km(s); cand = set()
            for m in sk: cand |= idx.get(m, set())
            for ci in cand:
                t = ref[ci]
                if s == t or s in t or t in s or len(sk & rk[ci]) / len(sk | rk[ci]) > 0.3:
                    return True
            return False
        return leaks

    eri = set(pd.read_csv(ERICKSON).sequence.dropna().str.upper())
    train_ref = set(pd.read_csv(BENCH).query("source != 'PlasticEnz'").sequence.str.upper())
    chk_eri = make_checker(eri)
    chk_train = make_checker(train_ref | eri)

    is_pe = df.source == "PlasticEnz"
    drop_pe = is_pe & df.sequence.map(chk_eri)                  # PlasticEnz: only vs Erickson
    drop_hn = (~is_pe) & df.sequence.map(chk_train)             # hard negs: vs training + Erickson
    before = len(df)
    df = df[~(drop_pe | drop_hn)].reset_index(drop=True)
    print(f"\nscrubbed {int(drop_pe.sum())} PlasticEnz (vs Erickson) + {int(drop_hn.sum())} hard-neg (vs training/Erickson) = {before-len(df)} total")

    df.to_csv(OUT, index=False)
    pos = int((df.activity_label == 1).sum()); neg = int((df.activity_label == 0).sum())
    print(f"\nHARD TEST SET: {len(df)} unique sequences  ({pos} degrader / {neg} hard-negative)")
    print(f"  by source: {df.source.value_counts().to_dict()}")
    print(f"  duplicate sequences: {len(df)-df.sequence.nunique()}")
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
