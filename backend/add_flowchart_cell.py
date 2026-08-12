"""Append the architecture flowchart section to PETase_Stability_Analysis.ipynb."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "PETase_Stability_Analysis.ipynb")

nb = nbf.read(NB, as_version=4)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

nb.cells.append(md(r"""
---
# 14 · System architecture

Industrial PET depolymerisation runs near the polymer's glass transition (~70 °C)
in a buffered process stream, so an enzyme has to survive **both** a temperature and
a pH it did not evolve for. That splits enzyme selection into three separate
prediction problems, and this project builds screeners for them.

The figure below is drawn directly from the code paths in `multitask/`. Solid
borders mark screeners that are trained and benchmarked; the dashed border marks
the pH screener, whose leakage-audited dataset is complete but which has not yet
been trained.
"""))

nb.cells.append(md(r"""
### 14.1 · How much pH information does the training data actually contain?

Before drawing pH into the architecture it is worth measuring how much pH signal
exists to learn from. This reads the harmonised staging table directly.
"""))

nb.cells.append(code(r'''
_ph = pd.read_csv(os.path.join(ROOT, "datasets", "staging", "staging_clean.csv"),
                  usecols=["measurement_type", "ph", "source_dataset"], low_memory=False)
_dd = _ph[_ph.measurement_type == "ddG"]
_v  = pd.to_numeric(_dd["ph"], errors="coerce")

N_DDG    = len(_dd)
PH7      = int((_v == 7.0).sum())
PH7_FRAC = PH7 / N_DDG
PH_VAR   = int((_v.notna() & (_v != 7.0)).sum())

print(f"ddG training records            {N_DDG:>10,}")
print(f"  measured at pH 7.0 exactly    {PH7:>10,}   ({PH7_FRAC:6.1%})")
print(f"  with real pH variation        {PH_VAR:>10,}   ({PH_VAR/N_DDG:6.1%})")
print(f"  pH missing entirely           {int(_v.isna().sum()):>10,}")
print("\nSources supplying real pH variation:")
print(_dd[_v.notna() & (_v != 7.0)].source_dataset.value_counts().to_string())
print(f"\npH range where it does vary: {_v[_v != 7.0].min():.1f} - {_v[_v != 7.0].max():.1f}")
del _ph, _dd
'''))

nb.cells.append(md(r"""
Public ΔΔG measurements are overwhelmingly recorded at neutral pH. A model given a
feature that reads 7.0 in ~98 % of rows has almost no variation to learn from — which
is precisely why the process-window question needs a **separate screener trained on a
dedicated pH-optimum dataset**, rather than being folded into the ΔΔG model.
"""))

nb.cells.append(code(r'''
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, TEAL, VIOLET = "#111111", "#1a6a72", "#4a3aa7"
GREY, RED = "#8a8a85", "#c0392b"
TEAL_L, VIOLET_L, GREY_L = "#e2eef0", "#e9e7f4", "#f2f2f0"

fig, ax = plt.subplots(figsize=(13.2, 8.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.6, ls="-", r=0.014, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.004,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def arrow(p0, p1, color=GREY, lw=1.5, rad=0.0, style="-|>", z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle=style, mutation_scale=13, lw=lw, color=color, zorder=z))

# ─────────────────────────────── title
ax.text(.5, .985, "Screening enzymes for industrial PET depolymerisation",
        ha="center", va="top", fontsize=15.5, fontweight="bold", color=INK)
ax.text(.5, .945, "Three condition-driven questions · two screeners trained, one dataset-ready",
        ha="center", va="top", fontsize=10.5, color="#555", style="italic")

# ─────────────────────────────── process constraint
box(.20, .845, .60, .068, "#fbfbf9", INK, lw=1.5)
ax.text(.5, .879, "PROCESS CONDITIONS   ~70 °C  ·  buffered pH  ·  aqueous stream",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

W, H, Y0 = .295, .565, .235
XS = [.030, .3525, .675]
TOP = Y0 + H

QS = ["Q1  Is the enzyme stable enough?",
      "Q2  Does the mutation help?",
      "Q3  Does it work at process pH?"]
for x, q in zip(XS, QS):
    arrow((x + W/2, .843), (x + W/2, TOP + .038), color=INK, lw=1.4)
    ax.text(x + W/2, .826, q, ha="center", va="top", fontsize=10, color=INK,
            fontweight="bold")

# ─────────────────────────────── the three screeners
panels = [
    dict(x=XS[0], fc=GREY_L, ec=INK, ls="-", title="STAGE 1", sub="scaffold screener",
         status="TRAINED  ·  BENCHMARKED", sc=INK,
         rows=[("INPUT", "one protein sequence"),
               ("FEATURES", "ESM-2 $t30$ mean-pooled (640-d)\n+ log(length)\n= 641 features"),
               ("ENSEMBLE", "3 models — mean\nLightGBM · XGBoost · CatBoost"),
               ("OUTPUT", "predicted $T_m$  (°C)"),
               ("BENCHMARK", "BRENDA  n = 1,563\nROC AUC 0.732 · precision 0.978")]),
    dict(x=XS[1], fc=TEAL_L, ec=TEAL, ls="-", title="STAGE 2", sub="mutation screener",
         status="TRAINED  ·  BENCHMARKED", sc=TEAL,
         rows=[("INPUT", "wild-type + mutant sequence\n+ assay temperature and pH"),
               ("FEATURES", "ESM-2 [WT | mut | Δ] → PCA 256-d\n+ ESM-2 ΔLL at mutation site\n"
                            "+ biochemical Δ  + (T, pH)"),
               ("ENSEMBLE", "6 models — mean\nLGBM · XGB · CatBoost · RF · ET · MLP"),
               ("OUTPUT", "predicted ΔΔG  (kcal/mol)"),
               ("BENCHMARK", "S669  n = 669\nROC AUC 0.669 · precision 0.810")]),
    dict(x=XS[2], fc=VIOLET_L, ec=VIOLET, ls="--", title="pH SCREENER", sub="process-window screener",
         status="DATASET READY  ·  NOT YET TRAINED", sc=VIOLET,
         rows=[("INPUT", "one protein sequence"),
               ("FEATURES", "ESM-2 embedding\n(same encoder as Stage 1)"),
               ("ENSEMBLE", "to be trained"),
               ("OUTPUT", "predicted pH optimum"),
               ("BENCHMARK", "BRENDA / UniProt split\nhomology-audited ≥30% identity")]),
]

for p in panels:
    x, ec = p["x"], p["ec"]
    box(x, Y0, W, H, p["fc"], ec, lw=2.0, ls=p["ls"])
    box(x, TOP - .055, W, .055, ec, ec, lw=0, r=.012, z=3)
    ax.text(x + W/2, TOP - .0275, p["title"], ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="white", zorder=4)
    ax.text(x + W/2, TOP - .078, p["sub"], ha="center", va="center",
            fontsize=9.5, color=ec, style="italic")
    ax.text(x + W/2, TOP - .107, p["status"], ha="center", va="center",
            fontsize=7.6, color=ec, fontweight="bold")
    yy = TOP - .148
    for label, val in p["rows"]:
        ax.text(x + .016, yy, label, ha="left", va="top", fontsize=7.4,
                color="#7a7a75", fontweight="bold")
        ax.text(x + .016, yy - .023, val, ha="left", va="top", fontsize=8.5,
                color=INK, linespacing=1.5)
        yy -= .024 + .028 * (val.count("\n") + 1)

# pH emphasis: mark where pH actually enters the trained screeners
ax.text(XS[1] + W - .016, Y0 + .011, "↑  pH enters here", ha="right", va="bottom",
        fontsize=8.2, color=VIOLET, fontweight="bold", style="italic")

# ─────────────────────────────── why Q3 needs its own dataset
box(.030, .118, .618, .098, "#fdf6f4", RED, lw=1.4)
ax.text(.044, .198, "WHY pH NEEDS A DEDICATED SCREENER", ha="left", va="center",
        fontsize=8.6, fontweight="bold", color=RED)
ax.text(.044, .152, f"{PH7_FRAC:.1%} of all {N_DDG:,} public ΔΔG records were measured at pH 7.0 —\n"
                    f"only {PH_VAR:,} ({PH_VAR/N_DDG:.1%}) carry real pH variation.",
        ha="left", va="center", fontsize=9.0, color=INK, linespacing=1.5)

box(.668, .118, .302, .098, "#f7f7f5", GREY, lw=1.3, ls="--")
ax.text(.819, .198, "SHARED FOUNDATION", ha="center", va="center",
        fontsize=8.6, fontweight="bold", color="#6a6a65")
ax.text(.819, .150, f"{CLEAN:,} leakage-audited records\n"
                    f"{LEAK_HOM:,} homology leaks removed\n({LEAK_HOM/LEAK_EXACT:.0f}× what exact matching finds)",
        ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.5)

# ─────────────────────────────── application chain
chain = [("candidate\nenzymes", GREY), ("STAGE 1", INK), ("best\nscaffold", GREY),
         ("STAGE 2", TEAL), ("~20 variants\nto assay", GREY)]
cw, cy, ch = .148, .030, .070
gap = (.94 - 5 * cw) / 4
for i, (label, col) in enumerate(chain):
    cx = .030 + i * (cw + gap)
    box(cx, cy, cw, ch, "white", col, lw=1.5, r=.010)
    ax.text(cx + cw/2, cy + ch/2, label, ha="center", va="center",
            fontsize=8.8, color=col, linespacing=1.4,
            fontweight="bold" if col != GREY else "normal")
    if i < 4:
        arrow((cx + cw + .004, cy + ch/2), (cx + cw + gap - .004, cy + ch/2),
              color=GREY, lw=1.4)
ax.text(.5, .003, "application workflow — the two trained screeners chained in sequence",
        ha="center", va="bottom", fontsize=8, color="#7a7a75", style="italic")

save(fig, "fig6_architecture"); plt.show()
'''))

nbf.write(nb, NB)
print(f"appended 2 cells -> {len(nb.cells)} total")
