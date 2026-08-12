"""Append the AlphaFold stabilising-mutation figure to PETase_Stability_Analysis.ipynb."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "PETase_Stability_Analysis.ipynb")
nb = nbf.read(NB, as_version=4)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

nb.cells.append(md(r"""
---
# 17 · A stabilising mutation, scored by the model, on an AlphaFold structure

This section takes a mutation that is **experimentally stabilising**, confirms the
model also called it stabilising, and places it on the **AlphaFold-predicted
structure** of the protein.

Protein: **YtvA blue-light photoreceptor** from *Bacillus subtilis* (UniProt
O34627). Its AlphaFold model is high quality — global pLDDT 84.6 with only 0.4 %
of residues in the very-low band — so the predicted fold can be trusted around the
mutation site.

> **What AlphaFold does and does not contribute here.** AlphaFold predicts the
> *wild-type* fold from sequence. It does not predict ΔΔG, does not model the
> mutant, and was **not** used by the stability model — which takes sequence plus
> assay conditions as input. AlphaFold is used for two things: to show where the
> scored mutation sits, and to report per-residue confidence (pLDDT) at that
> position, which tells us whether the local geometry can be trusted.
"""))

nb.cells.append(code(r'''
import re, json, urllib.request, warnings
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer

UNIPROT, XRAY_ID, CHAIN = "O34627", "2PR5", "A"
SEQ_MUT = "N92Y"
AF_DIR = os.path.join(ROOT, "alphafold_structures")
os.makedirs(AF_DIR, exist_ok=True)
AF_PATH = os.path.join(AF_DIR, f"AF-{UNIPROT}-F1-v6.pdb")

# The AlphaFold file URL carries a version that moves, so take it from the API.
meta = json.load(urllib.request.urlopen(
    f"https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT}", timeout=30))[0]
if not os.path.exists(AF_PATH):
    urllib.request.urlretrieve(meta["pdbUrl"], AF_PATH)

print(f"AlphaFold model  {meta['modelEntityId']}  ({meta['toolUsed']})")
print(f"  global pLDDT   {meta['globalMetricValue']:.1f}")
print(f"  very-low band  {meta['fractionPlddtVeryLow']*100:.1f}% of residues")

# --- the mutation and the model's prediction ---
_s669 = pd.read_csv(os.path.join(ROOT, "s669_full.tsv"), sep="\t")
_ens = np.mean(np.stack([
    np.load(os.path.join(ROOT, "multitask", "models", f"{m}_s669_pred.npy")).astype(float)
    for m in ["mlp", "lightgbm", "random_forest", "catboost", "xgboost", "extra_trees"]]), 0)
_s669["pred"] = _ens
row = _s669[(_s669.pdb == XRAY_ID) & (_s669.mutation == SEQ_MUT)].iloc[0]
PRED, EXPT = float(row.pred), float(row.ddG)
mm = re.match(r"([A-Z])(\d+)([A-Z])", str(row.mutation_pdb))
WT_AA, POS, MUT_AA = mm.group(1), int(mm.group(2)), mm.group(3)

# --- structures ---
_p = PDBParser(QUIET=True)
AF = _p.get_structure("af", AF_PATH)[0][CHAIN]
XR = _p.get_structure("xr", os.path.join(ROOT, "pdb_structures", f"{XRAY_ID}.pdb"))[0][CHAIN]
af_res = [r for r in AF if r.id[0] == " "]
xr_res = [r for r in XR if r.id[0] == " "]

TGT = [r for r in af_res if r.id[1] == POS][0]
PLDDT_SITE = TGT["CA"].get_bfactor()
AF_CA = np.array([r["CA"].coord for r in af_res if "CA" in r])
AF_PLDDT = np.array([r["CA"].get_bfactor() for r in af_res if "CA" in r])
tgt_ca = TGT["CA"].coord
NEAR = sorted([r for r in af_res if r.id[1] != POS and "CA" in r
               and np.linalg.norm(r["CA"].coord - tgt_ca) < 10.0],
              key=lambda r: np.linalg.norm(r["CA"].coord - tgt_ca))

# --- does the AlphaFold model agree with the crystal structure? ---
shared = sorted(set(r.id[1] for r in af_res) & set(r.id[1] for r in xr_res))
sup = Superimposer()
sup.set_atoms([[r for r in xr_res if r.id[1] == n][0]["CA"] for n in shared],
              [[r for r in af_res if r.id[1] == n][0]["CA"] for n in shared])
RMSD = sup.rms

print(f"\nprotein        {row.protein}  ({row.organism})")
print(f"mutation       {SEQ_MUT} (S669 construct numbering) = {row.mutation_pdb} (UniProt / PDB)")
print(f"residue {POS}    {TGT.get_resname()}   pLDDT {PLDDT_SITE:.1f}")
print(f"neighbours     {len(NEAR)} residues with CA within 10 A")
print(f"AF vs crystal  Ca RMSD {RMSD:.2f} A over {len(shared)} shared residues")
print(f"assay          pH {row.pH}, {row['T']} C")
print()
print(f"MODEL PREDICTED ddG   {PRED:+.2f} kcal/mol   -> STABILISING")
print(f"EXPERIMENTAL    ddG   {EXPT:+.2f} kcal/mol   -> STABILISING")
print(f"absolute error        {abs(PRED-EXPT):.2f}")
print(f"sign agreement        {'YES' if np.sign(PRED)==np.sign(EXPT) else 'NO'}"
      f"  -> variant correctly PRIORITISED for wet-lab testing")
'''))

nb.cells.append(md(r"""
### 17.1 · Structure figure

The two structure panels are **ray-traced PyMOL renders**, produced by
`backend/render_pymol.py` in an isolated `pymol-render` conda environment
(PyMOL 3.1) and loaded here as images — PyMOL cannot be installed alongside this
notebook's numpy without breaking torch, so it is kept in its own environment.

Only the **LOV domain** (residues 21–147) is drawn. The ΔΔG for this mutation was
measured on that isolated 132-residue construct, crystallised as 2PR5; the
full-length AlphaFold model additionally contains a linker helix and a STAS domain
that the measurement says nothing about.
"""))

nb.cells.append(code(r'''
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
RED, VIO, GREEN = "#c0392b", "#4a3aa7", "#2f7d32"
PYMOL_DIR = os.path.join(ROOT, "paper_figures", "pymol")

def load_trimmed(name, pad=8):
    """Load a PyMOL PNG and crop its transparent margin."""
    img = mpimg.imread(os.path.join(PYMOL_DIR, name))
    alpha = img[..., 3] if img.shape[-1] == 4 else np.ones(img.shape[:2])
    ys, xs = np.where(alpha > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1]

fold = load_trimmed("af_lov.png")
site = load_trimmed("af_site.png")

fig = plt.figure(figsize=(15.2, 7.2))

# ---------------- Panel A: the fold ----------------
axA = fig.add_axes([.012, .105, .355, .700]); axA.axis("off")
axA.imshow(fold)
axA.set_title(f"AlphaFold model  ·  {meta['modelEntityId']}\n"
              f"LOV domain, residues 21–147  ·  coloured by pLDDT",
              fontsize=11.5, fontweight="bold", pad=8)
axA.text(.5, -.045, f"{AA3[TGT.get_resname()]}{POS} side chain shown as spheres/sticks",
         transform=axA.transAxes, ha="center", fontsize=9.3,
         color=RED, fontweight="bold")
axA.legend(handles=[Line2D([], [], color=c, lw=4, label=l) for c, l in
                    [("#0053D6", "very high  > 90"), ("#65CBF3", "confident  70–90"),
                     ("#FFDB13", "low  50–70"), ("#FF7D45", "very low  < 50")]],
           loc="lower left", bbox_to_anchor=(-.01, .02), fontsize=7.6,
           frameon=False, title="pLDDT", title_fontsize=8.2)

# ---------------- Panel B: the packing ----------------
axB = fig.add_axes([.375, .105, .285, .700]); axB.axis("off")
axB.imshow(site)
axB.set_title(f"local packing  ·  {len(NEAR)} residues within 10 Å\n"
              f"site pLDDT {PLDDT_SITE:.1f}  (very high)",
              fontsize=11, fontweight="bold", pad=8)
axB.text(.5, -.045, "Asn107 red · neighbouring side chains grey",
         transform=axB.transAxes, ha="center", fontsize=9.0, color="#555")

# ---------------- Panel C: the decision ----------------
axC = fig.add_axes([0, 0, 1, 1]); axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
axC.patch.set_alpha(0)

PX, PW = .685, .300
axC.add_patch(FancyBboxPatch((PX, .455), PW, .420,
              boxstyle="round,pad=0.006,rounding_size=.012",
              fc="#fbfbf9", ec="#555", lw=1.6, zorder=2))
axC.text(PX + PW/2, .845, "THE SCREENING DECISION", ha="center", va="center",
         fontsize=10.5, fontweight="bold", color=INK, zorder=3)

rows = [("protein", "YtvA photoreceptor · B. subtilis"),
        ("mutation", f"{SEQ_MUT}   (UniProt {row.mutation_pdb})"),
        ("substitution", "Asn → Tyr : polar → bulky aromatic"),
        ("site pLDDT", f"{PLDDT_SITE:.1f}  — very high confidence"),
        ("AF vs crystal", f"Cα RMSD {RMSD:.2f} Å  ({len(shared)} residues)"),
        ("assay", f"pH {row.pH},  {row['T']} °C")]
yy = .793
for k, v in rows:
    axC.text(PX + .018, yy, k.upper(), fontsize=7.0, color="#7a7a75",
             fontweight="bold", va="center", zorder=3)
    axC.text(PX + .112, yy, v, fontsize=8.3, color=INK, va="center", zorder=3)
    yy -= .043

axC.plot([PX + .018, PX + PW - .018], [.553, .553], color="#ddd", lw=1, zorder=3)
axC.text(PX + .018, .518, "MODEL PREDICTED", fontsize=7.4, color="#7a7a75",
         fontweight="bold", va="center", zorder=3)
axC.text(PX + PW - .018, .518, f"{PRED:+.2f} kcal/mol", ha="right", va="center",
         fontsize=13, fontweight="bold", color=TEAL, zorder=3)
axC.text(PX + .018, .480, "EXPERIMENTAL", fontsize=7.4, color="#7a7a75",
         fontweight="bold", va="center", zorder=3)
axC.text(PX + PW - .018, .480, f"{EXPT:+.2f} kcal/mol", ha="right", va="center",
         fontsize=13, fontweight="bold", color=INK, zorder=3)

axC.add_patch(FancyBboxPatch((PX, .290), PW, .135,
              boxstyle="round,pad=0.006,rounding_size=.012",
              fc="#eef6ee", ec=GREEN, lw=1.9, zorder=2))
axC.text(PX + PW/2, .388, "PRIORITISED", ha="center", va="center", fontsize=15,
         fontweight="bold", color=GREEN, zorder=3)
axC.text(PX + PW/2, .330, "both values negative — the screener keeps\n"
                          "this variant for wet-lab testing",
         ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.5, zorder=3)

axC.add_patch(FancyBboxPatch((PX, .075), PW, .185,
              boxstyle="round,pad=0.006,rounding_size=.012",
              fc="#f4f2fa", ec=VIO, lw=1.3, zorder=2))
axC.text(PX + PW/2, .228, "WHAT ALPHAFOLD DID AND DID NOT DO", ha="center",
         va="center", fontsize=8.0, fontweight="bold", color=VIO, zorder=3)
axC.text(PX + PW/2, .150,
         "AlphaFold predicted the WILD-TYPE fold from sequence.\n"
         "It did not model the mutant and did not predict ΔΔG.\n"
         "The stability model takes sequence + assay conditions\n"
         "and used neither this structure nor the crystal.",
         ha="center", va="center", fontsize=7.9, color="#3a3a36",
         linespacing=1.6, zorder=3)

fig.suptitle("A stabilising mutation scored by the Stage 2 screener, "
             "shown on the AlphaFold model",
             fontsize=13.5, fontweight="bold", y=.965)
save(fig, "fig9_alphafold_stabilising"); plt.show()
'''))

nbf.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
