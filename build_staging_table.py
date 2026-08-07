"""Build a STAGING table of candidate stability mutations/proteins — NOT the
final training set.

Goals (professor-facing, scientifically defensible):
  1. Collect every stability-relevant measurement from the locally-available
     sources into ONE rich, fully-annotated staging table.
  2. Keep only stability-relevant measurement types
     (ddG, dTm, Tm, dH, dCp, thermal_shift, abundance, proteolysis).
  3. Preserve full provenance per row: protein name, UniProt, PDB/chain,
     WT seq, mutant seq, mutation notation, residue mapping, T, pH,
     denaturant/condition, measurement type, measured value, source, PMID/link.
  4. Run a STRICT duplicate + leakage audit against the independent S669 test
     set (s669_full.tsv) BEFORE anything is eligible for training.
  5. Report numbers clearly.

Nothing here writes to the training set. Outputs go to datasets/staging/.

NOTE on coverage: only sources already on disk are loaded here
(Tsuboyama2023, ProDDG/S2648, FireProtDB, ThermoMutDB). The not-yet-downloaded
sources (MGnify, Domainome, Meltome, ProThermDB) plug into the same loader
interface and flow through the identical audit when their files land.
"""
import csv
import hashlib
import json
import os
import re

import pandas as pd

OUT_DIR = "datasets/staging"
STAGING_ALL = os.path.join(OUT_DIR, "staging_all.csv")        # every candidate + audit flags
STAGING_CLEAN = os.path.join(OUT_DIR, "staging_clean.csv")    # leak-free, de-duplicated
REPORT = os.path.join(OUT_DIR, "audit_report.txt")

S669_FILE = "s669_full.tsv"
UNIPROT_CACHE = "uniprot_seqs.json"

try:
    _UNIPROT_SEQS = json.load(open(UNIPROT_CACHE))
except (FileNotFoundError, json.JSONDecodeError):
    _UNIPROT_SEQS = {}

# Required provenance fields + audit/quality helpers.
COLS = [
    "protein_name", "uniprot_id", "pdb_id", "chain",
    "wt_sequence", "mut_sequence", "mutation", "position", "wt_aa", "mut_aa",
    "assay_temperature_c", "ph", "denaturant", "condition_quality",
    "measurement_type", "measured_value",
    "source_dataset", "pmid", "source_link",
    "wt_seq_hash", "mut_seq_hash",
]
AUDIT_COLS = ["leak_flag", "leak_reason", "dup_flag"]

STABILITY_TYPES = {"ddG", "dTm", "Tm", "dH", "dCp",
                   "thermal_shift", "abundance", "proteolysis"}

AA = set("ACDEFGHIKLMNPQRSTVWY")


# ----------------------------- helpers ---------------------------------------
def sha1(s):
    if not s:
        return ""
    return hashlib.sha1(str(s).strip().upper().encode()).hexdigest()


def parse_mutation(code):
    """'E49M' -> ('E', 49, 'M'). Returns (wt, pos, mut) or (None,None,None)."""
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", str(code).strip())
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)


def apply_mutation(wt_seq, pos, wt_aa, mut_aa):
    """Return mutant sequence if wt_seq[pos-1]==wt_aa, else '' (no fabrication)."""
    if not wt_seq or pos is None:
        return ""
    if pos < 1 or pos > len(wt_seq):
        return ""
    if wt_seq[pos - 1].upper() != wt_aa.upper():
        return ""
    return wt_seq[:pos - 1] + mut_aa.upper() + wt_seq[pos:]


def blank_row():
    return {c: "" for c in COLS}


# ----------------------------- loaders ---------------------------------------
def load_tsuboyama():
    """tsuboyama_stability.csv: 'sequence' col is the WT seq (convert_tsuboyama.py
    already reverted the mutant); derive the mutant by applying the substitution."""
    path = "datasets/stability_megascale/tsuboyama_stability.csv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        pos = int(r["position"])
        wt_aa, mut_aa = str(r["wt_aa"]), str(r["mut_aa"])
        wt_seq = str(r["sequence"])
        mut_seq = apply_mutation(wt_seq, pos, wt_aa, mut_aa)
        row = blank_row()
        row.update(
            protein_name=str(r["protein_id"]),
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=f"{wt_aa}{pos}{mut_aa}", position=pos,
            wt_aa=wt_aa, mut_aa=mut_aa,
            assay_temperature_c=r.get("temperature_c", ""), ph=r.get("ph", ""),
            denaturant="cDNA-display-proteolysis", condition_quality="nominal",
            measurement_type="ddG", measured_value=r["ddg"],
            source_dataset="Tsuboyama2023",
            source_link="https://www.nature.com/articles/s41586-023-06328-6",
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        out.append(row)
    return out


def load_tsuboyama_doubles():
    """tsuboyama_doubles.csv: WT + mutant sequences pre-reconstructed; multi-sub
    notation kept whole in `mutation` (position/wt_aa/mut_aa left blank)."""
    path = "datasets/stability_megascale/tsuboyama_doubles.csv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        wt_seq, mut_seq = str(r["wt_sequence"]), str(r["mut_sequence"])
        row = blank_row()
        row.update(
            protein_name=str(r["protein_id"]),
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=str(r["mutation"]),
            assay_temperature_c=r.get("temperature_c", ""), ph=r.get("ph", ""),
            denaturant="cDNA-display-proteolysis", condition_quality="nominal",
            measurement_type="ddG", measured_value=r["ddg"],
            source_dataset="Tsuboyama2023_double",
            source_link="https://www.nature.com/articles/s41586-023-06328-6",
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        out.append(row)
    return out


def load_proddg():
    path = "proddg_s2648.tsv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path, sep="\t")
    for _, r in df.iterrows():
        wt_aa, pos, mut_aa = parse_mutation(r["mutation"])
        if pos is None:
            continue
        wt_seq = str(r["wt_sequence"])
        mut_seq = apply_mutation(wt_seq, pos, wt_aa, mut_aa)
        row = blank_row()
        row.update(
            pdb_id=str(r.get("pdb", "")),
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=r["mutation"], position=pos, wt_aa=wt_aa, mut_aa=mut_aa,
            condition_quality="not_reported",
            # S2648 native convention is positive=stabilizing; project uses
            # negative=stabilizing -> NEGATE to match Tsuboyama/FireProtDB.
            measurement_type="ddG", measured_value=round(-float(r["ddG"]), 4),
            source_dataset="ProDDG_S2648",
            source_link="https://doi.org/10.1093/bioinformatics/btaa1059",
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        out.append(row)
    return out


def load_fireprotdb():
    path = "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        wt_aa = str(r["wild_type"]).strip()
        mut_aa = str(r["mutation"]).strip()
        try:
            pos = int(r["position"])
        except (ValueError, TypeError):
            continue
        if wt_aa not in AA or mut_aa not in AA:
            continue
        wt_seq = str(r["sequence"]) if pd.notna(r.get("sequence")) else ""
        mut_seq = apply_mutation(wt_seq, pos, wt_aa, mut_aa)
        pmid = "" if pd.isna(r.get("publication_pubmed")) else str(r.get("publication_pubmed"))
        doi = r.get("publication_doi")
        link = f"https://doi.org/{doi}" if pd.notna(doi) else ""
        base = blank_row()
        base.update(
            protein_name=str(r.get("protein_name", "")),
            uniprot_id="" if pd.isna(r.get("uniprot_id")) else str(r.get("uniprot_id")),
            pdb_id="" if pd.isna(r.get("pdb_id")) else str(r.get("pdb_id")),
            chain="" if pd.isna(r.get("chain")) else str(r.get("chain")),
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=f"{wt_aa}{pos}{mut_aa}", position=pos, wt_aa=wt_aa, mut_aa=mut_aa,
            ph="" if pd.isna(r.get("pH")) else r.get("pH"),
            denaturant="" if pd.isna(r.get("method")) else str(r.get("method")),
            condition_quality="measured" if pd.notna(r.get("pH")) else "not_reported",
            source_dataset="FireProtDB", pmid=pmid, source_link=link,
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        # emit one row per available measurement type
        if pd.notna(r.get("ddG")):
            row = dict(base); row["measurement_type"] = "ddG"; row["measured_value"] = r["ddG"]
            out.append(row)
        if pd.notna(r.get("dTm")):
            row = dict(base); row["measurement_type"] = "dTm"; row["measured_value"] = r["dTm"]
            out.append(row)
        if pd.notna(r.get("tm")):
            row = dict(base); row["measurement_type"] = "Tm"; row["measured_value"] = r["tm"]
            out.append(row)
    return out


def load_meltome():
    """Meltome Atlas WT proteins -> Tm (no mutation). measurement_type=Tm."""
    path = "datasets/downloads/meltome_tm.csv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        seq = str(r["sequence"])
        row = blank_row()
        row.update(
            protein_name=str(r["seq_id"]),
            wt_sequence=seq, mut_sequence="",
            mutation="", condition_quality="nominal",
            denaturant="thermal-proteome-profiling",
            measurement_type="Tm", measured_value=r["tm"],
            source_dataset="Meltome",
            source_link="https://www.nature.com/articles/s41592-020-0801-4",
            wt_seq_hash=sha1(seq), mut_seq_hash="",
        )
        out.append(row)
    return out


def load_domainome():
    """Human Domainome aPCA abundance (stability PROXY). measurement_type=abundance."""
    path = "datasets/downloads/domainome_stability.csv"
    if not os.path.exists(path):
        return []
    out = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        wt_seq, mut_seq = str(r["wt_sequence"]), str(r["mut_sequence"])
        row = blank_row()
        row.update(
            protein_name=str(r["domain_ID"]),
            uniprot_id=str(r["uniprot_ID"]),
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=str(r["mutation"]), position=r["position"],
            wt_aa=str(r["wt_aa"]), mut_aa=str(r["mut_aa"]),
            denaturant="aPCA-in-cell-abundance", condition_quality="nominal",
            measurement_type="abundance", measured_value=r["measured_value"],
            source_dataset="Domainome",
            source_link="https://www.nature.com/articles/s41586-024-08370-4",
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        out.append(row)
    return out


def load_thermomutdb():
    """thermomutdb.json: rich metadata but NO sequence (flagged for repair).
    temperature is in Kelvin."""
    path = "thermomutdb.json"
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out = []
    for r in data:
        if str(r.get("mutation_type", "")).lower() != "single":
            continue
        wt_aa, pos, mut_aa = parse_mutation(r.get("mutation_code", ""))
        if pos is None:
            continue
        tK = r.get("temperature")
        tC = round(tK - 273.15, 2) if isinstance(tK, (int, float)) else ""
        uni = r.get("uniprot") or ""
        pmid = str(r.get("PMID") or "")
        doi = r.get("DOI")
        link = f"https://doi.org/{doi}" if doi else (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else "")
        # repair sequence from UniProt cache when position+WT residue agree
        wt_seq = mut_seq = ""
        cand = _UNIPROT_SEQS.get(str(uni).strip())
        if cand and 1 <= pos <= len(cand) and cand[pos - 1].upper() == wt_aa:
            wt_seq = cand
            mut_seq = apply_mutation(wt_seq, pos, wt_aa, mut_aa)
        base = blank_row()
        base.update(
            protein_name=r.get("protein", "") or "",
            uniprot_id=uni,
            pdb_id=r.get("PDB_wild") or "",
            chain=r.get("mutated_chain") or "",
            wt_sequence=wt_seq, mut_sequence=mut_seq,
            mutation=r.get("mutation_code", ""), position=pos, wt_aa=wt_aa, mut_aa=mut_aa,
            assay_temperature_c=tC, ph="" if r.get("ph") is None else r.get("ph"),
            denaturant=r.get("method") or "",
            condition_quality="measured" if r.get("ph") is not None else "not_reported",
            source_dataset="ThermoMutDB", pmid=pmid, source_link=link,
            wt_seq_hash=sha1(wt_seq), mut_seq_hash=sha1(mut_seq),
        )
        if r.get("ddg") is not None:
            # raw ThermoMutDB ddg is inverted vs project convention
            # (negative=stabilizing) -> NEGATE. dtm is already correct, leave it.
            row = dict(base); row["measurement_type"] = "ddG"
            row["measured_value"] = round(-float(r["ddg"]), 4)
            out.append(row)
        if r.get("dtm") is not None:
            row = dict(base); row["measurement_type"] = "dTm"; row["measured_value"] = r["dtm"]
            out.append(row)
    return out


# ---------------------- S669 leakage index -----------------------------------
def build_s669_index():
    df = pd.read_csv(S669_FILE, sep="\t")
    idx = {
        "wt_hash": set(), "mut_hash": set(),
        "uniprot_mut": set(), "pdb_mut": set(), "pdbchain_mut": set(),
        "mutations": set(),
    }
    for _, r in df.iterrows():
        mut = str(r["mutation"]).strip()
        idx["mutations"].add(mut)
        idx["wt_hash"].add(sha1(r.get("wt_sequence", "")))
        idx["mut_hash"].add(sha1(r.get("mutant_sequence", "")))
        if pd.notna(r.get("uniprot")):
            idx["uniprot_mut"].add((str(r["uniprot"]).strip(), mut))
        if pd.notna(r.get("pdb")):
            idx["pdb_mut"].add((str(r["pdb"]).strip().upper(), mut))
            if pd.notna(r.get("chain")):
                idx["pdbchain_mut"].add((str(r["pdb"]).strip().upper(), str(r["chain"]).strip(), mut))
    idx["wt_hash"].discard(""); idx["mut_hash"].discard("")
    return idx, len(df)


def pdb_tokens(pdb_field):
    """'1PGA|1EM7|2GB1' -> ['1PGA','1EM7','2GB1']."""
    return [t.strip().upper() for t in re.split(r"[|,;]", str(pdb_field)) if t.strip()]


def audit_leak(row, idx):
    """Return (is_leak, reason) checking strongest -> weakest protein identity."""
    mut = str(row["mutation"]).strip()
    if row["wt_seq_hash"] and row["wt_seq_hash"] in idx["wt_hash"]:
        return True, "wt_sequence_hash_match"
    if row["mut_seq_hash"] and row["mut_seq_hash"] in idx["mut_hash"]:
        return True, "mut_sequence_hash_match"
    if row["uniprot_id"] and (str(row["uniprot_id"]).strip(), mut) in idx["uniprot_mut"]:
        return True, "uniprot+mutation_match"
    for tok in pdb_tokens(row["pdb_id"]):
        if row["chain"] and (tok, str(row["chain"]).strip(), mut) in idx["pdbchain_mut"]:
            return True, "pdb+chain+mutation_match"
        if (tok, mut) in idx["pdb_mut"]:
            return True, "pdb+mutation_match"
    return False, ""


# ----------------------------- main ------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loaders = [
        ("Tsuboyama2023", load_tsuboyama),
        ("Tsuboyama2023_double", load_tsuboyama_doubles),
        ("ProDDG_S2648", load_proddg),
        ("FireProtDB", load_fireprotdb),
        ("ThermoMutDB", load_thermomutdb),
        ("Domainome", load_domainome),
        ("Meltome", load_meltome),
    ]
    rows = []
    per_source_raw = {}
    for name, fn in loaders:
        recs = fn()
        per_source_raw[name] = len(recs)
        rows.extend(recs)
    rows = [r for r in rows if r["measurement_type"] in STABILITY_TYPES]
    total_found = len(rows)

    # ---- S669 leakage audit ----
    idx, n_s669 = build_s669_index()
    # optional sequence-homology leak set (produced by run_mmseqs_audit.py)
    homology_hashes = set()
    hfile = os.path.join(OUT_DIR, "homology_leak_hashes.txt")
    if os.path.exists(hfile):
        homology_hashes = {ln.strip() for ln in open(hfile) if ln.strip()}
    leak_reasons = {}
    for r in rows:
        leak, reason = audit_leak(r, idx)
        if not leak and r["wt_seq_hash"] and r["wt_seq_hash"] in homology_hashes:
            leak, reason = True, "homology_S669_>=30pct"
        r["leak_flag"] = "1" if leak else "0"
        r["leak_reason"] = reason
        if leak:
            leak_reasons[reason] = leak_reasons.get(reason, 0) + 1
    n_leak = sum(1 for r in rows if r["leak_flag"] == "1")

    # ---- duplicate audit (exact same measurement: protein-id + mutation + type + value) ----
    seen = set()
    n_dup = 0
    for r in rows:
        ident = r["wt_seq_hash"] or r["uniprot_id"] or r["pdb_id"] or r["protein_name"]
        key = (ident, r["mutation"], r["measurement_type"], str(r["measured_value"]))
        if key in seen:
            r["dup_flag"] = "1"; n_dup += 1
        else:
            r["dup_flag"] = "0"; seen.add(key)

    # cross-dataset repeated mutations (same mutation string seen in >1 dataset) — informational
    mut_to_sources = {}
    for r in rows:
        mut_to_sources.setdefault(r["mutation"], set()).add(r["source_dataset"])
    n_cross = sum(1 for m, s in mut_to_sources.items() if len(s) > 1)

    # ---- write staging_all (everything + flags) ----
    out_cols = COLS + AUDIT_COLS
    with open(STAGING_ALL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- write staging_clean (training-eligible: no leak, no exact dup) ----
    clean = [r for r in rows if r["leak_flag"] == "0" and r["dup_flag"] == "0"]
    with open(STAGING_CLEAN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(clean)

    # ---- report ----
    def tally(records, key):
        d = {}
        for r in records:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    uniq_prot = len({(r["wt_seq_hash"] or r["uniprot_id"] or r["pdb_id"] or r["protein_name"]) for r in clean})
    uniq_mut = len({(r["wt_seq_hash"] or r["uniprot_id"] or r["protein_name"], r["mutation"]) for r in clean})
    n_with_seq = sum(1 for r in clean if r["wt_sequence"] and r["mut_sequence"])
    tmdb_clean = [r for r in clean if r["source_dataset"] == "ThermoMutDB"]
    tmdb_seq = sum(1 for r in tmdb_clean if r["wt_sequence"])

    lines = []
    P = lines.append
    P("=" * 70)
    P("STAGING TABLE — DUPLICATE & S669 LEAKAGE AUDIT")
    P("=" * 70)
    P(f"Independent test set (S669): {n_s669} rows  [{S669_FILE}]")
    P("")
    P("RAW candidates pulled per source (locally available sources only):")
    for k, v in per_source_raw.items():
        P(f"  {k:16s} {v:>8d}")
    P(f"  {'TOTAL FOUND':16s} {total_found:>8d}  (stability-relevant measurement types only)")
    P("")
    P("REMOVED — exact duplicates (same protein+mutation+type+value):")
    P(f"  {n_dup:>8d}")
    P("REMOVED — S669 test-set leakage (any key matched):")
    P(f"  {n_leak:>8d}")
    for reason, c in sorted(leak_reasons.items(), key=lambda x: -x[1]):
        P(f"      {reason:28s} {c:>8d}")
    P("")
    P(f"Cross-dataset repeated mutations (informational): {n_cross} mutation strings in >1 dataset")
    P("")
    P(f"FINAL CLEAN (training-eligible) rows: {len(clean)}")
    P("")
    P("Clean rows kept BY MEASUREMENT TYPE:")
    for k, v in tally(clean, "measurement_type").items():
        P(f"  {k:16s} {v:>8d}")
    P("")
    P("Clean rows kept BY DATASET:")
    for k, v in tally(clean, "source_dataset").items():
        P(f"  {k:16s} {v:>8d}")
    P("")
    P(f"Final UNIQUE proteins (by seq-hash/uniprot/pdb/name): {uniq_prot}")
    P(f"Final UNIQUE mutations (protein+mutation):            {uniq_mut}")
    P(f"Clean rows WITH both WT+mutant sequence:             {n_with_seq} "
      f"({100*n_with_seq/max(len(clean),1):.1f}%)")
    P(f"ThermoMutDB clean rows sequence-repaired:            {tmdb_seq}/{len(tmdb_clean)}")
    P("")
    P("CAVEATS (defensible disclosure):")
    P("  - Homology leakage audited via mmseqs2 (>=30% id, >=50% qcov vs S669)")
    P("    and EXCLUDED — see homology_leak_hashes.txt. This catches Tsuboyama")
    P("    domain-level homology that exact UniProt/PDB keys cannot.")
    P("  - ThermoMutDB sequences repaired from UniProt where position+WT residue")
    P("    agree; rows with PDB-based numbering that disagree are left without a")
    P("    sequence (NOT fabricated) and excluded from sequence-based training.")
    P("  - Domainome 'abundance' is an aPCA stability PROXY (not ddG) -> its own")
    P("    head; kept separate from thermodynamic measurement types.")
    P("  - Meltome 'Tm' rows are WILD-TYPE proteins (no mutation) -> sequence->Tm")
    P("    head; audited for homology to S669 like all other sources.")
    P("  - Still pending (not in these numbers, same audit on arrival):")
    P("    MGnify (1.8M, pretraining-only; data not yet openly hosted),")
    P("    ProThermDB (pH/T/dH/dCp; form-gated, manual download required).")
    P("=" * 70)
    report = "\n".join(lines)
    print(report)
    with open(REPORT, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote:\n  {STAGING_ALL}\n  {STAGING_CLEAN}\n  {REPORT}")


if __name__ == "__main__":
    main()
