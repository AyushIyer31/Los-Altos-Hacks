"""Append the two-tier leakage-audit diagram to PETase_Stability_Analysis.ipynb."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "PETase_Stability_Analysis.ipynb")
nb = nbf.read(NB, as_version=4)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

nb.cells.append(md(r"""
---
# 16 · The two-tier leakage audit

Every candidate record is tested against the S669 benchmark twice, by two methods
that fail in different ways.

**Tier 1 — exact-key matching.** Five identifier tests, applied strongest-identity
first: wild-type sequence hash, mutant sequence hash, UniProt + mutation,
PDB + chain + mutation, PDB + mutation. Multi-ID fields such as `1PGA|1EM7|2GB1`
are split so a record cannot hide behind its third PDB code.

**Tier 2 — MMseqs2 homology search.** Every staging wild-type sequence is searched
against every S669 wild-type sequence; a row is flagged at ≥30 % identity over
≥50 % query coverage.

Tier 2 runs only on rows that survived tier 1, so the two counts are mutually
exclusive and sum exactly. The audit runs on `staging_all.csv` — all candidates,
before any filtering — so the result is independent of row order.
"""))

nb.cells.append(code(r'''
_fl = pd.read_csv(os.path.join(ROOT, "datasets", "staging", "staging_all.csv"),
                  usecols=["leak_flag", "dup_flag", "leak_reason"], low_memory=False)
L = _fl.leak_flag.astype(bool); D = _fl.dup_flag.astype(bool)

N_CAND   = len(_fl)
N_DUP    = int(D.sum())
N_LEAK   = int(L.sum())
N_BOTH   = int((L & D).sum())
N_REMOVE = int((L | D).sum())
N_CLEAN  = int((~L & ~D).sum())

REASONS = _fl.loc[L, "leak_reason"].value_counts()
TIER2 = int(REASONS.get("homology_S669_>=30pct", 0))
TIER1_BREAKDOWN = REASONS.drop("homology_S669_>=30pct", errors="ignore")
TIER1 = int(TIER1_BREAKDOWN.sum())

print(f"candidate records        {N_CAND:>10,}")
print(f"  tier 1 — exact keys    {TIER1:>10,}")
for k, v in TIER1_BREAKDOWN.items():
    print(f"      {k:<28} {v:>6,}")
print(f"  tier 2 — homology      {TIER2:>10,}")
print(f"  exact duplicates       {N_DUP:>10,}")
print(f"  (rows hit by both)     {N_BOTH:>10,}")
print(f"  total removed (union)  {N_REMOVE:>10,}")
print(f"clean, training-eligible {N_CLEAN:>10,}")
print(f"\ncheck: {N_CAND:,} - {N_REMOVE:,} = {N_CAND - N_REMOVE:,}  "
      f"({'OK' if N_CAND - N_REMOVE == N_CLEAN else 'MISMATCH'})")
print(f"homology / exact-key ratio = {TIER2 / TIER1:.0f}x")
del _fl
'''))

nb.cells.append(md("### 16.1 · Audit diagram"))

nb.cells.append(code(r'''
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, VIO, TEALC, GREYC = "#c0392b", "#4a3aa7", "#1a6a72", "#8a8a85"

fig = plt.figure(figsize=(14.6, 8.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def bx(x, y, w, h, fc, ec, lw=1.8, ls="-", r=.012, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.004,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def down(x, y0, y1, color=INK, lw=2.0):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=16, lw=lw, color=color, zorder=4))

def right(x0, x1, y, color=RED, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=14, lw=lw, color=color, zorder=4))

ax.text(.5, .975, "Two-tier test-set leakage audit",
        ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
ax.text(.5, .935, "every candidate record checked against the S669 benchmark twice, "
                  "by two methods that fail differently",
        ha="center", va="top", fontsize=10.5, color="#555", style="italic")

FX, FW = .045, .285          # funnel column
SX, SW = .365, .215          # side-callout column

# ---- funnel ----------------------------------------------------------------
bx(FX, .795, FW, .080, "#f2f2f0", GREYC, lw=1.6)
ax.text(FX + FW/2, .855, "CANDIDATE RECORDS", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#5a5a55")
ax.text(FX + FW/2, .822, f"{N_CAND:,}", ha="center", va="center",
        fontsize=17, fontweight="bold", color=INK)

down(FX + FW/2, .793, .742)

# tier 1
bx(FX, .625, FW, .115, "#fdf6f4", RED, lw=2.0)
ax.text(FX + FW/2, .718, "TIER 1 · EXACT-KEY MATCHING", ha="center", va="center",
        fontsize=9.8, fontweight="bold", color=RED)
ax.text(FX + FW/2, .687, "5 identifier tests, strongest first", ha="center",
        va="center", fontsize=8.4, color="#666", style="italic")
ax.text(FX + FW/2, .650, "sequence hash · UniProt+mut · PDB+chain+mut · PDB+mut",
        ha="center", va="center", fontsize=7.6, color="#444")
right(FX + FW + .004, SX - .004, .682)

bx(SX, .625, SW, .115, "white", RED, lw=1.5)
ax.text(SX + SW/2, .716, f"− {TIER1:,}", ha="center", va="center",
        fontsize=15, fontweight="bold", color=RED)
yy = .690
for k, v in TIER1_BREAKDOWN.items():
    ax.text(SX + .012, yy, k.replace("_match", "").replace("_", " "),
            ha="left", va="center", fontsize=7.0, color="#555")
    ax.text(SX + SW - .012, yy, f"{v:,}", ha="right", va="center",
            fontsize=7.0, color=INK, fontweight="bold")
    yy -= .0155

down(FX + FW/2, .623, .572)

# tier 2
bx(FX, .455, FW, .115, "#f0eefa", VIO, lw=2.0)
ax.text(FX + FW/2, .548, "TIER 2 · MMseqs2 HOMOLOGY SEARCH", ha="center", va="center",
        fontsize=9.8, fontweight="bold", color=VIO)
ax.text(FX + FW/2, .517, "every WT sequence vs every S669 sequence", ha="center",
        va="center", fontsize=8.4, color="#666", style="italic")
ax.text(FX + FW/2, .480, "≥ 30 % identity   ·   ≥ 50 % query coverage",
        ha="center", va="center", fontsize=8.6, color=INK, fontweight="bold")
right(FX + FW + .004, SX - .004, .512, color=VIO)

bx(SX, .455, SW, .115, "white", VIO, lw=1.5)
ax.text(SX + SW/2, .534, f"− {TIER2:,}", ha="center", va="center",
        fontsize=17, fontweight="bold", color=VIO)
ax.text(SX + SW/2, .488, f"{TIER2/TIER1:.0f}× what tier 1\ncan reach",
        ha="center", va="center", fontsize=8.6, color=VIO,
        fontweight="bold", linespacing=1.45)

down(FX + FW/2, .453, .402)

# duplicates
bx(FX, .318, FW, .082, "#f7f7f5", GREYC, lw=1.6)
ax.text(FX + FW/2, .377, "EXACT-DUPLICATE REMOVAL", ha="center", va="center",
        fontsize=9.3, fontweight="bold", color="#5a5a55")
ax.text(FX + FW/2, .344, "same protein + mutation + type + value", ha="center",
        va="center", fontsize=7.8, color="#666", style="italic")
right(FX + FW + .004, SX - .004, .359, color=GREYC)
bx(SX, .318, SW, .082, "white", GREYC, lw=1.4)
ax.text(SX + SW/2, .372, f"− {N_DUP:,}", ha="center", va="center",
        fontsize=13.5, fontweight="bold", color="#5a5a55")
ax.text(SX + SW/2, .337, f"{N_BOTH:,} of these were\nalso leak hits",
        ha="center", va="center", fontsize=7.4, color="#777", linespacing=1.4)

down(FX + FW/2, .316, .262)

# clean
bx(FX, .150, FW, .112, "#eef4ea", "#2f4a22", lw=2.2)
ax.text(FX + FW/2, .242, "CLEAN · TRAINING-ELIGIBLE", ha="center", va="center",
        fontsize=9.8, fontweight="bold", color="#2f4a22")
ax.text(FX + FW/2, .205, f"{N_CLEAN:,}", ha="center", va="center",
        fontsize=20, fontweight="bold", color="#2f4a22")
ax.text(FX + FW/2, .170, f"{N_CAND:,} − {N_REMOVE:,} removed (union)",
        ha="center", va="center", fontsize=8.2, color="#4a5a44")

# ---- right panel: log-scale comparison -------------------------------------
axb = fig.add_axes([.655, .585, .30, .265])
bars = axb.bar(["Tier 1\nexact keys", "Tier 2\nhomology"], [TIER1, TIER2],
               color=[RED, VIO], width=.55, zorder=3)
axb.set_yscale("log"); axb.set_ylim(100, TIER2 * 7)
axb.set_ylabel("records caught (log scale)", fontsize=9)
axb.tick_params(labelsize=8.5)
axb.grid(axis="y", alpha=.22, zorder=0); axb.set_axisbelow(True)
for b, v, c in zip(bars, [TIER1, TIER2], [RED, VIO]):
    axb.text(b.get_x() + b.get_width()/2, v * 1.3, f"{v:,}", ha="center",
             fontsize=11, fontweight="bold", color=c)
axb.annotate("", xy=(1, TIER2 * 2.6), xytext=(0, TIER2 * 2.6),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
axb.text(.5, TIER2 * 3.3, f"{TIER2/TIER1:.0f}×", ha="center", fontsize=13,
         fontweight="bold", color=INK)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

# ---- right panel: why -------------------------------------------------------
bx(.638, .105, .325, .430, "#fbfbf9", GREYC, lw=1.4)
ax.text(.8005, .500, "WHY TIER 1 CANNOT SEE IT", ha="center", va="center",
        fontsize=9.6, fontweight="bold", color=INK)

wy = .455
for title, body, col in [
    ("Tsuboyama mega-scale data is domain-level",
     "small protein domains that share no UniProt\naccession and no PDB entry with any S669 entry —\n"
     "yet are clearly homologous by sequence", VIO),
    ("Meltome carries no identifiers at all",
     "the FLIP mirror stripped accessions and renamed\nevery protein Sequence0…N, so all 27,884 records\n"
     "are invisible to identifier-based matching", VIO),
    ("Every tier-1 key needs an identifier",
     "so the failure is structural, not a tuning problem —\n"
     "no threshold on an identifier test can recover these", RED),
]:
    ax.text(.655, wy, "▸", fontsize=9, color=col, va="top", fontweight="bold")
    ax.text(.672, wy, title, fontsize=8.8, color=INK, va="top", fontweight="bold")
    ax.text(.672, wy - .026, body, fontsize=7.8, color="#555", va="top",
            linespacing=1.55)
    wy -= .118

ax.text(.8005, .138, "Sensitivity was reported at several identity thresholds,\n"
                     "not only 30 %, so the cutoff was not tuned to flatter the result.",
        ha="center", va="center", fontsize=7.6, color="#777", style="italic",
        linespacing=1.5)

save(fig, "fig8_leakage_audit"); plt.show()
'''))

nbf.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
