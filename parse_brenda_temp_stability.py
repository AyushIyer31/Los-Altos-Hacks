"""Parse BRENDA's free-text temperature_stability into a numeric stability ceiling
and a stable/unstable label, so brenda_ph_thermal.csv can serve as an INDEPENDENT
test set for a whole-protein (sequence -> stability) model.

BRENDA temperature_stability format (one protein, ';'-separated):
    "55 [1 h, stable] ; 90 [60 min, loss of 20% activity] ; 76.8 [melting temperature]"
Each entry = <temp or lo-hi range> [<comment>].  -999 = no specific temperature.

Per entry we read the temperature and classify the COMMENT as:
  - Tm        : an explicit melting temperature (cleanest stability number)
  - stable    : enzyme retains activity at that temperature
  - unstable  : enzyme loses activity / denatures / has a measured half-life there
Then per protein we derive one numeric ceiling `stable_temp_c` (prefer a wild-type
Tm; else the hottest 'stable' temperature; else the onset of inactivation) and a
3-way label using tunable thresholds.

NOTE: heuristic and approximate. Tm (unfolding midpoint) and "stable up to"
(retains activity) are different physical quantities put on one coarse axis; the
`basis` column records which was used so you can filter.

  python parse_brenda_temp_stability.py            # parse + stats
"""
import csv
import re

SRC = "datasets/downloads/brenda_ph_thermal.csv"
OUT = "datasets/downloads/brenda_temp_stability_labeled.csv"

# label thresholds (deg C)
T_POS = 60.0   # stable_temp_c >= this -> positive (thermostable)
T_NEG = 45.0   # stable_temp_c <= this -> negative (not thermostable)

ENTRY = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*\[(.*)\]\s*$")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

STABLE_NEGATIONS = ("no loss", "no activity loss", "no significant loss",
                    "without loss", "no detectable loss", "no appreciable loss",
                    "no change", "no significant change")
UNSTABLE_KW = ("loss", "lost", "denatur", "inactiv", "unstable",
               "half-life", "half life")
STABLE_KW = ("stable", "retain", "remaining", "residual", "fully active", "active for")
STORAGE_KW = ("stored", "storage", "freez", "-20", "-80", "frozen")
DURATION_KW = ("day", "week", "month")
IMMOB_KW = ("immobiliz", "encapsulat", "nanoflower", "cross-link", "crosslink")
TM_MAX = 125.0   # free-protein Tm above this is implausible -> drop
MIN_T = 20.0     # below room temp = cold-activity/storage, NOT heat tolerance -> ignore


def classify(comment):
    """Return (kind, is_wt, is_mut, is_storage, is_immob).
    kind in {'Tm','stable','unstable',''}. Uses % residual activity when present."""
    c = comment.lower()
    is_mut = "mutant" in c
    is_wt = any(k in c for k in ("wild-type", "wild type", "wildtype", "native"))
    is_immob = any(k in c for k in IMMOB_KW)
    is_storage = any(k in c for k in STORAGE_KW) or \
        ("stable" in c and any(d in c for d in DURATION_KW))
    if "melting temp" in c or "apparent melting" in c or re.search(r"\btm\b", c) \
            or "tm-value" in c or "tm," in c:
        return "Tm", is_wt, is_mut, is_storage, is_immob
    # explicit "no loss / no change" phrases are STABLE despite the word 'loss'
    if any(neg in c for neg in STABLE_NEGATIONS):
        return "stable", is_wt, is_mut, is_storage, is_immob
    # "% residual/remaining activity" -> stable if >=50%, else unstable
    has_resid = any(k in c for k in ("residual", "remaining", "retain"))
    pct = PCT.search(c)
    if has_resid and pct:
        return ("stable" if float(pct.group(1)) >= 50 else "unstable",
                is_wt, is_mut, is_storage, is_immob)
    if any(k in c for k in UNSTABLE_KW):
        return "unstable", is_wt, is_mut, is_storage, is_immob
    if any(k in c for k in STABLE_KW):
        return "stable", is_wt, is_mut, is_storage, is_immob
    return "", is_wt, is_mut, is_storage, is_immob


def parse_field(text):
    """Return dict with stable_temp_c, basis, and the raw evidence lists."""
    tms_wt, tms_any, stable_t, unstable_t = [], [], [], []
    for part in text.split(";"):
        m = ENTRY.match(part.strip())
        if not m:
            continue
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        if lo == -999:                      # BRENDA "no temperature" sentinel
            continue
        kind, is_wt, is_mut, is_storage, is_immob = classify(m.group(3))
        if kind == "Tm":
            if is_immob or is_mut and not is_wt or not (MIN_T <= lo <= TM_MAX):
                continue                    # immobilized / mutant-only / implausible
            (tms_wt if is_wt else tms_any).append(lo)
        elif kind == "stable":
            if is_storage or hi < MIN_T:
                continue                    # cold-storage/sub-ambient, NOT heat tolerance
            stable_t.append(hi)             # hottest temp it's still OK at
        elif kind == "unstable":
            if lo < MIN_T:
                continue                    # cold-temp entry, not a thermal onset
            unstable_t.append(lo)           # onset = low end of an inactivation range

    if tms_wt:
        return dict(stable_temp_c=max(tms_wt), basis="Tm_wt",
                    n_stable=len(stable_t), n_unstable=len(unstable_t))
    if tms_any:
        return dict(stable_temp_c=max(tms_any), basis="Tm",
                    n_stable=len(stable_t), n_unstable=len(unstable_t))
    if stable_t:
        return dict(stable_temp_c=max(stable_t), basis="stable_evidence",
                    n_stable=len(stable_t), n_unstable=len(unstable_t))
    if unstable_t:
        return dict(stable_temp_c=min(unstable_t), basis="inactivation_onset",
                    n_stable=0, n_unstable=len(unstable_t))
    return None


def label(t):
    if t >= T_POS:
        return "positive"
    if t <= T_NEG:
        return "negative"
    return "ambiguous"


def main():
    rows = list(csv.DictReader(open(SRC)))
    out_cols = ["accession", "organism", "ec", "stable_temp_c", "basis",
                "label", "n_stable", "n_unstable", "temperature_stability", "sequence"]
    out, parsed, unparsed = [], 0, 0
    by_basis, by_label = {}, {}
    for r in rows:
        txt = r["temperature_stability"].strip()
        if not txt:
            continue
        res = parse_field(txt)
        if not res:
            unparsed += 1
            continue
        parsed += 1
        lab = label(res["stable_temp_c"])
        by_basis[res["basis"]] = by_basis.get(res["basis"], 0) + 1
        by_label[lab] = by_label.get(lab, 0) + 1
        out.append({
            "accession": r["accession"], "organism": r["organism"], "ec": r["ec"],
            "stable_temp_c": round(res["stable_temp_c"], 1), "basis": res["basis"],
            "label": lab, "n_stable": res["n_stable"], "n_unstable": res["n_unstable"],
            "temperature_stability": txt, "sequence": r["sequence"],
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out)

    n_temp = sum(1 for r in rows if r["temperature_stability"].strip())
    print(f"rows with temperature_stability text : {n_temp}")
    print(f"  parsed to a numeric ceiling        : {parsed} ({100*parsed/n_temp:.1f}%)")
    print(f"  could not parse                    : {unparsed}")
    print(f"\nby basis (how the number was derived):")
    for k, v in sorted(by_basis.items(), key=lambda x: -x[1]):
        print(f"  {k:20} {v}")
    print(f"\nlabel split (POS>= {T_POS}C, NEG<= {T_NEG}C):")
    for k in ("positive", "negative", "ambiguous"):
        print(f"  {k:10} {by_label.get(k,0)}")
    ts = sorted(r["stable_temp_c"] for r in out)
    import statistics
    print(f"\nstable_temp_c distribution: min {ts[0]} | "
          f"median {statistics.median(ts)} | mean {round(statistics.mean(ts),1)} | max {ts[-1]}")
    for lo, hi in [(-50, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 1000)]:
        c = sum(1 for t in ts if lo <= t < hi)
        print(f"    {lo:>4}-{hi:<4}C : {c}")
    print(f"\nwrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
