"""Generate stabilizing-mutation candidates for wild-type IsPETase at 50 C
using the trained mutation model. Temperature-only (pH held neutral)."""
import sys, json
import numpy as np
from app.services import trained_classifier as clf

# --- Exact WT IsPETase (UniProt A0A0K8P6T7), 290 aa ---
WT = ("MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPS"
      "GYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSS"
      "RSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQA"
      "PWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGN"
      "SNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS")
assert len(WT) == 290, len(WT)

TEMP = 50.0          # target test temperature (C) -- temperature only
PH   = 7.0           # neutral, held constant (not optimized)
AAs  = "ACDEFGHIKLMNPQRSTVWY"

# positions to PROTECT (1-indexed on full sequence):
SIGNAL   = set(range(1, 28))            # signal peptide 1-27
CATALYTIC = {160, 206, 237}            # Ser-Asp-His catalytic triad
DISULFIDE = {203, 239, 273, 289}       # cysteines in the two disulfide bonds
PROTECT = SIGNAL | CATALYTIC | DISULFIDE

print("Loading trained model (mutation_regressor.pkl + ESM-2)...", file=sys.stderr)
clf.train_model()   # loads pickled ensemble + scaler + PCA (name is legacy)

# Build all allowed single-mutation tuples (wt_aa, pos_1indexed, mut_aa)
tuples, meta = [], []
for i, wt_aa in enumerate(WT):
    pos = i + 1
    if pos in PROTECT:
        continue
    for mut in AAs:
        if mut == wt_aa:
            continue
        tuples.append((wt_aa, pos, mut))
        meta.append((pos, wt_aa, mut))
print(f"Scoring {len(tuples)} single mutations at {TEMP} C ...", file=sys.stderr)

ddg, prob, _extra = clf.predict_mutations_batch_raw(tuples, sequence=WT, temperature=TEMP, ph=PH)
dtm = clf.predict_dtm_batch(tuples, sequence=WT, temperature=TEMP, ph=PH)
ddg, prob, dtm = np.asarray(ddg), np.asarray(prob), np.asarray(dtm)

rows = []
for k, (pos, wt_aa, mut) in enumerate(meta):
    rows.append({
        "mutation": f"{wt_aa}{pos}{mut}",
        "pos": pos, "wt": wt_aa, "mut": mut,
        "ddg": round(float(ddg[k]), 4),      # negative = stabilizing
        "prob_stab": round(float(prob[k]), 4),
        "pred_dTm": round(float(dtm[k]), 3), # positive = raises Tm
    })

# Rank: strongest predicted stabilization. Primary = predicted dTm (interpretable),
# require model also calls it stabilizing (ddg<0 & prob>0.5) for confidence.
conf = [r for r in rows if r["ddg"] < 0 and r["prob_stab"] > 0.5]
conf.sort(key=lambda r: (-r["pred_dTm"], r["ddg"]))

print("\n=== TOP 15 confident stabilizing single mutations (50 C) ===")
print(f"{'mut':<8}{'pred_dTm':>9}{'ddg':>9}{'prob':>7}")
for r in conf[:15]:
    print(f"{r['mutation']:<8}{r['pred_dTm']:>9}{r['ddg']:>9}{r['prob_stab']:>7}")

json.dump({"wt": WT, "temp": TEMP, "ranked": conf[:40]},
          open("candidates_raw.json", "w"), indent=2)
print(f"\nWrote candidates_raw.json ({len(conf)} confident stabilizers total)", file=sys.stderr)
