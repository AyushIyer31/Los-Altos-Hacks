"""Extract BRENDA proteins that have BOTH pH stability AND temperature stability,
then resolve their amino-acid sequences from UniProt.

BRENDA's bulk JSON (download.php -> "BRENDA JSON file", CC BY 4.0) stores, per EC
number, a set of proteins (id -> organism + UniProt/GenBank accessions) plus
`ph_stability` and `temperature_stability` arrays. Each stability record is a
{value, comment, proteins:[ids], references:[ids]} object. There is NO sequence
string in BRENDA -- only accessions -- so we fetch sequences from UniProt.

Pipeline:
  1. For each EC: protein ids cited by ph_stability  (set A)
                  protein ids cited by temperature_stability (set B)
  2. Keep proteins in A & B  (the "both stabilities" intersection).
  3. Keep those with a UniProt accession.
  4. Batch-fetch sequences from UniProt REST.
  5. Write datasets/downloads/brenda_ph_thermal.csv

Usage:
  python convert_brenda.py --inspect datasets/downloads/brenda_2026_1.json   # dump structure, no fetch
  python convert_brenda.py datasets/downloads/brenda_2026_1.json             # full run
"""
import csv
import json
import sys
import time

OUT = "datasets/downloads/brenda_ph_thermal.csv"
SEQ_CACHE = "datasets/downloads/brenda_uniprot_seqs.json"
COLS = ["ec", "brenda_protein_id", "organism", "accession", "source",
        "ph_stability", "temperature_stability", "sequence"]

# BRENDA key names can drift between releases; accept the known variants.
PH_KEYS = ["ph_stability", "phStability", "ph_st"]
TEMP_KEYS = ["temperature_stability", "temperatureStability", "ts"]
PROT_KEYS = ["proteins", "protein"]


def first_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def stab_text(record):
    """Render one stability record as 'value [comment]'."""
    v = str(record.get("value", "")).strip()
    c = str(record.get("comment", "")).strip()
    return f"{v} [{c}]" if c else v


def inspect(path):
    """Print top-level + one EC entry so we can confirm the real structure."""
    with open(path) as f:
        doc = json.load(f)
    print("top-level keys:", list(doc.keys()))
    data = doc.get("data", doc)
    ecs = list(data.keys())
    print(f"EC entries: {len(ecs)}  e.g. {ecs[:5]}")
    # find an EC that actually has both stability fields, to show their shape
    for ec in ecs:
        e = data[ec]
        if not isinstance(e, dict):
            continue
        pk, tk = first_key(e, PH_KEYS), first_key(e, TEMP_KEYS)
        if pk and tk:
            print(f"\nEC {ec} keys: {list(e.keys())}")
            print(f"  ph key   = {pk!r}, sample: {json.dumps(e[pk][:1], indent=2)[:600]}")
            print(f"  temp key = {tk!r}, sample: {json.dumps(e[tk][:1], indent=2)[:600]}")
            prk = first_key(e, PROT_KEYS)
            print(f"  protein key = {prk!r}, sample: "
                  f"{json.dumps(list(e[prk].items())[:1] if isinstance(e[prk], dict) else e[prk][:1], indent=2)[:600]}")
            return
    print("No EC entry had both stability fields with the expected key names.")


def get_proteins(entry):
    """Return {id(str) -> {organism, accessions[list], source}} for one EC entry."""
    prk = first_key(entry, PROT_KEYS)
    if prk is None:
        return {}
    raw = entry[prk]
    out = {}
    items = raw.items() if isinstance(raw, dict) else ((str(p.get("id")), p) for p in raw)
    for pid, p in items:
        accs = p.get("accessions") or ([p["accession"]] if p.get("accession") else [])
        out[str(pid)] = {"organism": p.get("organism", ""),
                         "accessions": accs,
                         "source": p.get("source", "")}
    return out


def parse(path, mode="either"):
    """mode='either' -> protein needs pH OR temp stability (union, default).
       mode='both'   -> protein needs pH AND temp stability (intersection)."""
    with open(path) as f:
        doc = json.load(f)
    data = doc.get("data", doc)
    rows = []
    n_ec = n_both = 0
    for ec, entry in data.items():
        if not isinstance(entry, dict):
            continue
        n_ec += 1
        pk, tk = first_key(entry, PH_KEYS), first_key(entry, TEMP_KEYS)
        if not (pk and tk):
            continue
        proteins = get_proteins(entry)

        # map protein-id -> list of stability texts
        ph_by_pid, temp_by_pid = {}, {}
        for rec in entry[pk]:
            for pid in rec.get("proteins", []):
                ph_by_pid.setdefault(str(pid), []).append(stab_text(rec))
        for rec in entry[tk]:
            for pid in rec.get("proteins", []):
                temp_by_pid.setdefault(str(pid), []).append(stab_text(rec))

        if mode == "both":
            keep = set(ph_by_pid) & set(temp_by_pid)
        else:
            keep = set(ph_by_pid) | set(temp_by_pid)
        for pid in keep:
            n_both += 1
            info = proteins.get(pid, {})
            # prefer a UniProt-style accession
            accs = info.get("accessions", [])
            acc = next((a for a in accs if a and a[0].isalpha()), accs[0] if accs else "")
            rows.append({
                "ec": ec,
                "brenda_protein_id": pid,
                "organism": info.get("organism", ""),
                "accession": acc,
                "source": info.get("source", ""),
                "ph_stability": " ; ".join(ph_by_pid.get(pid, [])),
                "temperature_stability": " ; ".join(temp_by_pid.get(pid, [])),
                "sequence": "",
            })
    label = "BOTH pH AND thermal" if mode == "both" else "pH OR thermal"
    print(f"EC entries scanned: {n_ec}")
    print(f"proteins with {label} stability: {n_both}")
    with_acc = sum(1 for r in rows if r["accession"])
    print(f"  of which have an accession: {with_acc}")
    return rows


def fetch_sequences(rows):
    """Batch-resolve UniProt accessions -> sequence. Caches to SEQ_CACHE."""
    import os
    import requests
    cache = {}
    if os.path.exists(SEQ_CACHE):
        cache = json.load(open(SEQ_CACHE))
    accs = sorted({r["accession"] for r in rows if r["accession"] and r["accession"] not in cache})
    print(f"fetching {len(accs)} new sequences from UniProt ({len(cache)} cached)...")
    BATCH = 100
    for i in range(0, len(accs), BATCH):
        chunk = accs[i:i + BATCH]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        try:
            r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                             params={"query": q, "format": "fasta", "size": BATCH}, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  batch {i//BATCH} failed: {e}")
            continue
        acc = seq = None
        for line in r.text.splitlines():
            if line.startswith(">"):
                if acc:
                    cache[acc] = seq
                # header: >sp|P12345|NAME ...
                parts = line.split("|")
                acc, seq = (parts[1] if len(parts) > 1 else None), ""
            else:
                seq = (seq or "") + line.strip()
        if acc:
            cache[acc] = seq
        print(f"  {min(i+BATCH, len(accs))}/{len(accs)}")
        time.sleep(0.3)
        json.dump(cache, open(SEQ_CACHE, "w"))
    for r in rows:
        r["sequence"] = cache.get(r["accession"], "")
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--inspect" in sys.argv:
        inspect(args[0])
        return
    if not args:
        print(__doc__)
        return
    rows = parse(args[0])
    rows = fetch_sequences(rows)
    rows = [r for r in rows if r["sequence"]]
    with open(OUT, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows (with sequence) -> {OUT}")


if __name__ == "__main__":
    main()
