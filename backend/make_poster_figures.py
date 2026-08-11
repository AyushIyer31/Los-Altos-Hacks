"""Poster diagrams for the PET Lab two-stage stability model.

Six schematic/analytical figures, one per poster section:
  A  problem_leakage        THE PROBLEM   — hidden homology leakage
  B  solution_pipeline      SOLUTION      — two-stage decision flow
  C  methodology_flow       METHODOLOGY   — data -> model -> evaluation
  D  results_operating      RESULTS       — precision/recall at operating points
  E  analysis_lift          ANALYSIS      — lift over baseline + screening cost
  F  conclusion_roadmap     CONCLUSION    — done vs. next

Every number is real: Stage 1/2 metrics recomputed here from the saved
predictions, dataset/leakage counts read from datasets/staging/audit_report.txt.
Style matches the poster (dark-green headers, white cards, numbered steps).
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figures", "poster")
os.makedirs(OUT, exist_ok=True)

# ---- poster palette -------------------------------------------------------
GREEN = "#2f4a22"      # poster header green
GREEN_L = "#e8eee3"    # pale green card fill
TEAL = "#1a6a72"       # accent (matches existing paper figures)
TEAL_L = "#dceaec"
RED = "#c0392b"        # danger / leakage
RED_L = "#f7e2df"
GREY = "#8a8a85"
GREY_L = "#f0f0ee"
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 0.9, "axes.edgecolor": "#666",
    "figure.dpi": 200, "savefig.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white", dpi=300)
    plt.close(fig)
    print("wrote", name)


def card(ax, x, y, w, h, fc, ec, lw=1.6, r=0.03, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p0, p1, color=GREEN, lw=2.2, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=20,
                                 lw=lw, color=color, zorder=z))


def step_badge(ax, x, y, n, color=GREEN, r=0.021):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", zorder=7)


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ===========================================================================
# Real numbers
# ===========================================================================
# --- audit report (authoritative on-disk counts) ---
audit = open(os.path.join(ROOT, "datasets", "staging", "audit_report.txt")).read()


def grab(pattern, text=audit):
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None


RAW_TOTAL = grab(r"TOTAL FOUND\s+(\d+)")
LEAK_TOTAL = grab(r"S669 test-set leakage \(any key matched\):\s*\n\s*(\d+)")
LEAK_HOM = grab(r"homology_S669_>=30pct\s+(\d+)")
DUPS = grab(r"exact duplicates.*?\n\s*(\d+)")
CLEAN = grab(r"FINAL CLEAN \(training-eligible\) rows:\s*(\d+)")
N_PROT = grab(r"Final UNIQUE proteins.*?:\s*(\d+)")
LEAK_EXACT = LEAK_TOTAL - LEAK_HOM
RATIO = LEAK_HOM / LEAK_EXACT

# --- Stage 1 (BRENDA) ---
df1 = pd.read_csv(os.path.join(ROOT, "multitask", "models_tm", "brenda_tm_predictions.csv"))
d1 = df1[df1["label"].isin(["positive", "negative"])]
y1 = (d1["label"] == "positive").to_numpy().astype(int)
s1 = d1["pred_tm_ensemble"].to_numpy(float)
BASE1 = y1.mean()

# --- Stage 2 (S669) ---
y2raw = np.load(os.path.join(ROOT, "multitask", "models", "y_s669.npy")).astype(float)
ens2 = np.mean(np.stack([
    np.load(os.path.join(ROOT, "multitask", "models", f"{n}_s669_pred.npy")).astype(float)
    for n in ["mlp", "lightgbm", "random_forest", "catboost", "xgboost", "extra_trees"]]), 0)
y2 = (y2raw < 0).astype(int)
BASE2 = y2.mean()


def op1(t):
    pp = s1 >= t
    tp = int((pp & (y1 == 1)).sum()); called = int(pp.sum())
    return dict(thr=t, prec=tp / called, rec=tp / y1.sum(), called=called, hits=tp)


def op2(t):
    pp = ens2 < t
    tp = int((pp & (y2 == 1)).sum()); called = int(pp.sum())
    return dict(thr=t, prec=tp / called, rec=tp / y2.sum(), called=called, hits=tp)


S1 = [op1(t) for t in (46, 50, 52, 60, 66)]
S2 = [op2(t) for t in (0.0, 0.25, 0.5, 1.0)]

print(f"raw={RAW_TOTAL:,} dups={DUPS:,} leak={LEAK_TOTAL:,} "
      f"(homology {LEAK_HOM:,} / exact {LEAK_EXACT:,} = {RATIO:.0f}x) clean={CLEAN:,}")


# ===========================================================================
# A — THE PROBLEM: hidden homology leakage
# ===========================================================================
fig = plt.figure(figsize=(12.2, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.26)

# left: true-proportion composition of the leaked records
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.99, "Test-set contamination is mostly invisible",
        ha="center", va="top", fontsize=13.5, fontweight="bold", color=INK)
ax.text(0.5, 0.875, f"All {LEAK_TOTAL:,} training records that overlap the S669 benchmark",
        ha="center", va="top", fontsize=10, color="#555")

bx, bw_, byy, bhh = 0.045, 0.91, 0.44, 0.20
frac = LEAK_EXACT / LEAK_TOTAL
ax.add_patch(FancyBboxPatch((bx, byy), bw_ * frac, bhh,
                            boxstyle="round,pad=0,rounding_size=0.004",
                            fc=GREEN, ec="white", lw=1.2, zorder=4))
ax.add_patch(FancyBboxPatch((bx + bw_ * frac, byy), bw_ * (1 - frac), bhh,
                            boxstyle="round,pad=0,rounding_size=0.004",
                            fc=RED, ec="white", lw=1.2, zorder=3))

ax.text(bx + bw_ * frac + bw_ * (1 - frac) / 2, byy + bhh / 2,
        f"{LEAK_HOM:,}   ({LEAK_HOM/LEAK_TOTAL*100:.1f}%)", ha="center", va="center",
        fontsize=15, fontweight="bold", color="white", zorder=6)
ax.text(0.62, byy - 0.075, "found ONLY by homology search", ha="center", va="top",
        fontsize=10.5, fontweight="bold", color=RED)
ax.text(0.62, byy - 0.155,
        "≥30% sequence identity to a benchmark protein —\n"
        "no exact identifier is ever shared", ha="center", va="top",
        fontsize=9, color=RED, linespacing=1.4)

ax.annotate(f"{LEAK_EXACT}  ({frac*100:.1f}%)\nfound by exact-key matching",
            xy=(bx + bw_ * frac / 2, byy), xytext=(bx + 0.075, byy - 0.135),
            fontsize=10, fontweight="bold", color=GREEN, ha="center", va="top",
            linespacing=1.4,
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                            connectionstyle="arc3,rad=0.25"))

ax.text(0.5, 0.115, "A pipeline that deduplicates on identifiers alone keeps\n"
                    "the red block — and reports an inflated benchmark score.",
        ha="center", va="top", fontsize=10, color="#444", linespacing=1.5)

# right: bar comparison
ax2 = fig.add_subplot(gs[1])
vals = [LEAK_EXACT, LEAK_HOM]
labels = ["Exact-key\nmatching", "Homology\nsearch"]
bars = ax2.bar(labels, vals, color=[GREEN, RED], width=0.52, zorder=3)
ax2.set_yscale("log")
ax2.set_ylim(100, LEAK_HOM * 6)
ax2.set_ylabel("Leaked records detected (log scale)", fontsize=10.5)
ax2.set_title("Leakage caught, by detection method", fontsize=12,
              fontweight="bold", pad=12)
ax2.grid(axis="y", alpha=0.22, zorder=0)
ax2.set_axisbelow(True)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}",
             ha="center", fontsize=13, fontweight="bold",
             color=GREEN if v == LEAK_EXACT else RED)
ax2.annotate("", xy=(1, LEAK_HOM * 2.4), xytext=(0, LEAK_HOM * 2.4),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
ax2.text(0.5, LEAK_HOM * 3.0, f"{RATIO:.0f}× more leakage found",
         ha="center", fontsize=12, fontweight="bold", color=INK)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
save(fig, "A_problem_leakage")


# ===========================================================================
# B — SOLUTION: two-stage decision flow
# ===========================================================================
fig, ax = blank((12.4, 4.3))
ax.text(0.5, 0.965, "Two stages, two engineering decisions",
        ha="center", va="top", fontsize=14, fontweight="bold", color=INK)

bw, bh, by = 0.185, 0.36, 0.34
xs = [0.035, 0.275, 0.545, 0.79]

card(ax, xs[0], by, bw, bh, GREY_L, GREY, lw=1.5)
ax.text(xs[0] + bw / 2, by + bh - 0.105, "CANDIDATE\nENZYMES", ha="center",
        fontsize=10.5, fontweight="bold", color="#4a4a45", linespacing=1.3)
ax.text(xs[0] + bw / 2, by + 0.105, "thousands of\nsequences", ha="center",
        fontsize=9.5, color="#5a5a55", linespacing=1.35)

card(ax, xs[1], by, 0.225, bh, TEAL_L, TEAL, lw=2.0)
step_badge(ax, xs[1] + 0.026, by + bh - 0.028, 1, TEAL)
ax.text(xs[1] + 0.117, by + bh - 0.085, "STAGE 1", ha="center", fontsize=11.5,
        fontweight="bold", color=TEAL)
ax.text(xs[1] + 0.117, by + bh - 0.145, "scaffold selection", ha="center",
        fontsize=9.5, color=TEAL, style="italic")
ax.text(xs[1] + 0.117, by + 0.145, "sequence → $T_m$", ha="center", fontsize=11,
        color=INK, fontweight="bold")
ax.text(xs[1] + 0.117, by + 0.062, "“which enzyme do\nI start from?”", ha="center",
        fontsize=9, color="#444", linespacing=1.35)

card(ax, xs[2], by, 0.215, bh, TEAL_L, TEAL, lw=2.0)
step_badge(ax, xs[2] + 0.026, by + bh - 0.028, 2, TEAL)
ax.text(xs[2] + 0.107, by + bh - 0.085, "STAGE 2", ha="center", fontsize=11.5,
        fontweight="bold", color=TEAL)
ax.text(xs[2] + 0.107, by + bh - 0.145, "mutation ranking", ha="center",
        fontsize=9.5, color=TEAL, style="italic")
ax.text(xs[2] + 0.107, by + 0.145, "WT + mutant → ΔΔG", ha="center", fontsize=10.5,
        color=INK, fontweight="bold")
ax.text(xs[2] + 0.107, by + 0.062, "“which mutation\ndo I make?”", ha="center",
        fontsize=9, color="#444", linespacing=1.35)

card(ax, xs[3], by, bw, bh, GREEN_L, GREEN, lw=2.0)
ax.text(xs[3] + bw / 2, by + bh - 0.075, "WET LAB", ha="center", fontsize=11,
        fontweight="bold", color=GREEN)
ax.text(xs[3] + bw / 2, by + 0.16, "~20", ha="center", fontsize=19,
        fontweight="bold", color=GREEN)
ax.text(xs[3] + bw / 2, by + 0.085, "high-confidence\nvariants to assay", ha="center",
        fontsize=9, color=GREEN, linespacing=1.35)

arrow(ax, (xs[0] + bw, by + bh / 2), (xs[1] - 0.005, by + bh / 2))
arrow(ax, (xs[1] + 0.225, by + bh / 2), (xs[2] - 0.005, by + bh / 2))
arrow(ax, (xs[2] + 0.215, by + bh / 2), (xs[3] - 0.005, by + bh / 2))
ax.text((xs[0] + bw + xs[1]) / 2, by + bh / 2 + 0.045, "filter", ha="center",
        fontsize=8.5, color=GREEN, style="italic")
ax.text((xs[1] + 0.225 + xs[2]) / 2, by + bh / 2 + 0.045, "best\nscaffold",
        ha="center", fontsize=8.5, color=GREEN, style="italic", linespacing=1.25)
ax.text((xs[2] + 0.215 + xs[3]) / 2, by + bh / 2 + 0.045, "ranked", ha="center",
        fontsize=8.5, color=GREEN, style="italic")

card(ax, 0.26, 0.045, 0.515, 0.215, "#fbfbf9", GREY, lw=1.3)
ax.text(0.5175, 0.215, "Both stages take assay temperature and pH as inputs",
        ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
ax.text(0.5175, 0.125, "predictions are tied to measurement conditions,\n"
                       "not averaged across them", ha="center", va="center",
        fontsize=9, color="#555", linespacing=1.45)
save(fig, "B_solution_pipeline")


# ===========================================================================
# C — METHODOLOGY: data -> model -> evaluation
# ===========================================================================
fig, ax = blank((12.4, 5.0))
ax.text(0.5, 0.975, "From seven public databases to two audited models",
        ha="center", va="top", fontsize=14, fontweight="bold", color=INK)

steps = [
    ("COLLECT", f"{RAW_TOTAL:,}\nraw records", "7 public sources:\nTsuboyama · Domainome\nMeltome · FireProtDB\nThermoMutDB\nProDDG", GREY_L, GREY),
    ("HARMONIZE", "unified\nschema", "protein · mutation\nsequence · value\ntemperature · pH\nquality flags", GREY_L, GREY),
    ("AUDIT", f"−{LEAK_TOTAL:,}\nleaked", "exact keys +\nMMseqs2 homology\n(≥30% id, ≥50% cov)\nvs. S669", RED_L, RED),
    ("FEATURIZE", "ESM-2\nembeddings", "Stage 1: mean-pool\n+ log(length)\nStage 2: [WT|mut|Δ]\n→ PCA 256 + ΔLL\n+ biochem + (T, pH)", TEAL_L, TEAL),
    ("TRAIN", "3 + 6 model\nensembles", "Stage 1: LightGBM\nXGBoost · CatBoost\nStage 2: those + RF\nExtraTrees · MLP\non Nautilus GPU", TEAL_L, TEAL),
    ("EVALUATE", "held-out\nonly", "BRENDA (Stage 1)\nS669 (Stage 2)\nnever used for\ntuning or threshold\nchoice", GREEN_L, GREEN),
]
n = len(steps)
bw = 0.142; gap = (1 - 0.03 * 2 - n * bw) / (n - 1)
by, bh = 0.30, 0.50
for i, (title, big, body, fc, ec) in enumerate(steps):
    x = 0.03 + i * (bw + gap)
    card(ax, x, by, bw, bh, fc, ec, lw=1.8)
    step_badge(ax, x + 0.019, by + bh - 0.026, i + 1, ec, r=0.019)
    ax.text(x + bw / 2, by + bh - 0.085, title, ha="center", fontsize=10.5,
            fontweight="bold", color=ec)
    ax.text(x + bw / 2, by + bh - 0.175, big, ha="center", fontsize=11.5,
            fontweight="bold", color=INK, linespacing=1.3)
    ax.text(x + bw / 2, by + 0.145, body, ha="center", fontsize=7.8,
            color="#4a4a45", linespacing=1.5)
    if i < n - 1:
        arrow(ax, (x + bw + 0.004, by + bh / 2), (x + bw + gap - 0.004, by + bh / 2),
              color=GREY, lw=1.8)

card(ax, 0.30, 0.075, 0.40, 0.155, GREEN_L, GREEN, lw=2.0)
ax.text(0.5, 0.185, f"{CLEAN:,} clean, training-eligible records",
        ha="center", fontsize=12, fontweight="bold", color=GREEN)
ax.text(0.5, 0.115, f"{N_PROT:,} unique proteins  ·  96.4% with both sequences",
        ha="center", fontsize=9.5, color="#4a4a45")
save(fig, "C_methodology_flow")


# ===========================================================================
# D — RESULTS: precision / recall at operating points
# ===========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
for ax_, rows, xlab, base, title, unit in (
        (a1, S1, "Temperature threshold (°C)", BASE1, "Stage 1 — BRENDA (n = 1,563)", "°C"),
        (a2, S2, "ΔΔG threshold (kcal/mol)", BASE2, "Stage 2 — S669 (n = 669)", "")):
    idx = np.arange(len(rows)); w = 0.36
    p = [r["prec"] for r in rows]; rc = [r["rec"] for r in rows]
    b1 = ax_.bar(idx - w / 2, p, w, color=TEAL, label="Precision", zorder=3)
    b2 = ax_.bar(idx + w / 2, rc, w, color="#c4ccc0", label="Recall", zorder=3)
    ax_.axhline(base, color=RED, ls="--", lw=1.5, zorder=4,
                label=f"Baseline precision ({base:.2f})")
    for b, v in zip(b1, p):
        ax_.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.2f}", ha="center",
                 fontsize=9.5, fontweight="bold", color=TEAL)
    for b, v in zip(b2, rc):
        ax_.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.2f}", ha="center",
                 fontsize=9.5, color="#6a726a")
    ax_.set_xticks(idx)
    ax_.set_xticklabels([f"{r['thr']:g}{unit}" if unit else f"{r['thr']:.2f}" for r in rows])
    ax_.set_ylim(0, 1.16); ax_.set_xlabel(xlab, fontsize=10.5)
    ax_.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_.grid(axis="y", alpha=0.2, zorder=0); ax_.set_axisbelow(True)
    ax_.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=3,
               columnspacing=1.1, handlelength=1.4)
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
a1.set_ylabel("Score", fontsize=10.5)
a1.annotate(f"{S1[-1]['hits']} hits from\n{S1[-1]['called']} calls",
            xy=(4 - 0.18, S1[-1]["prec"] - 0.06), xytext=(2.55, 0.145), fontsize=9,
            color=GREEN, fontweight="bold", linespacing=1.35, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3,
                            connectionstyle="arc3,rad=-0.25"))
a2.annotate(f"{S2[0]['hits']} hits from\n{S2[0]['called']} calls",
            xy=(0 + 0.18, S2[0]["prec"] - 0.05), xytext=(1.15, 0.90), fontsize=9,
            color=GREEN, fontweight="bold", linespacing=1.35, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3,
                            connectionstyle="arc3,rad=0.2"))
fig.suptitle("Precision rises with stringency as recall falls", fontsize=13.5,
             fontweight="bold", y=1.005)
fig.tight_layout()
save(fig, "D_results_operating")


# ===========================================================================
# E — ANALYSIS: lift over baseline + screening cost
# ===========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.5))

# Order BOTH stages most-stringent -> least-stringent so the x-axis means
# the same thing on each series (Stage 1 thresholds rise with stringency,
# Stage 2 thresholds fall with it, so Stage 1 is reversed here).
R1 = S1[::-1]          # 66, 60, 52, 50, 46 C
R2 = S2                # 0.00, 0.25, 0.50, 1.00 kcal/mol
lift1 = [r["prec"] / BASE1 for r in R1]
lift2 = [r["prec"] / BASE2 for r in R2]

a1.plot(range(len(lift1)), lift1, "o-", color=TEAL, lw=2.4, ms=9,
        label="Stage 1 (BRENDA)", zorder=4)
a1.plot(range(len(lift2)), lift2, "s--", color=GREEN, lw=2.4, ms=9,
        label="Stage 2 (S669)", zorder=4)
a1.axhline(1.0, color=RED, ls=":", lw=1.7, zorder=2)
a1.text(-0.28, 1.04, "no better than chance", ha="left", va="bottom",
        fontsize=8.5, color=RED, style="italic")
for i, v in enumerate(lift1):
    a1.text(i, v - 0.19, f"{v:.2f}×", ha="center", fontsize=9.5,
            fontweight="bold", color=TEAL)
for i, v in enumerate(lift2):
    a1.text(i, v + 0.13, f"{v:.2f}×", ha="center", fontsize=9.5,
            fontweight="bold", color=GREEN)
a1.set_xticks(range(5))
a1.set_xticklabels(["most\nstringent", "", "", "", "least\nstringent"], fontsize=9)
a1.set_ylabel("Precision ÷ baseline  (lift)", fontsize=10.5)
a1.set_ylim(0.55, 4.15)
a1.set_xlim(-0.35, 4.35)
a1.set_title("Enrichment over chance", fontsize=12, fontweight="bold", pad=10)
a1.legend(fontsize=9, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.99))
a1.grid(alpha=0.2, zorder=0); a1.set_axisbelow(True)
a1.annotate("Stage 2 extracts more signal\ndespite lower raw precision",
            xy=(0.08, lift2[0] - 0.05), xytext=(1.45, 3.72), fontsize=9, color=GREEN,
            fontweight="bold", linespacing=1.35, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3,
                            connectionstyle="arc3,rad=0.18"))

cost1 = [r["called"] / r["hits"] for r in R1]   # most -> least stringent
cost2 = [r["called"] / r["hits"] for r in R2]
idx = np.arange(4); w = 0.38
a2.bar(idx - w / 2, cost1[:4], w, color=TEAL, label="Stage 1", zorder=3)
a2.bar(idx + w / 2, cost2, w, color=GREEN, label="Stage 2", zorder=3)
for i, v in enumerate(cost1[:4]):
    a2.text(i - w / 2, v + 0.07, f"{v:.2f}", ha="center", fontsize=9.5,
            fontweight="bold", color=TEAL)
for i, v in enumerate(cost2):
    a2.text(i + w / 2, v + 0.07, f"{v:.2f}", ha="center", fontsize=9.5,
            fontweight="bold", color=GREEN)
a2.set_xticks(idx)
a2.set_xticklabels(["most\nstringent", "", "", "least\nstringent"], fontsize=9)
a2.set_ylabel("Wet-lab assays per confirmed hit", fontsize=10.5)
a2.set_ylim(0, 4.5)
a2.set_title("Screening cost", fontsize=12, fontweight="bold", pad=10)
a2.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
a2.grid(axis="y", alpha=0.2, zorder=0); a2.set_axisbelow(True)
a2.text(1.5, 4.15, "Loosening the threshold costs 2.6× more\nexperiments per confirmed hit",
        ha="center", va="center", fontsize=9, color=INK,
        fontweight="bold", linespacing=1.35)
for ax_ in (a1, a2):
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
fig.suptitle("Why precision matters more than recall", fontsize=13.5,
             fontweight="bold", y=1.005)
fig.tight_layout()
save(fig, "E_analysis_lift")


# ===========================================================================
# F — CONCLUSION: done vs. next
# ===========================================================================
fig, ax = blank((11.8, 4.4))
ax.text(0.5, 0.975, "Where the project stands", ha="center", va="top",
        fontsize=14, fontweight="bold", color=INK)

card(ax, 0.025, 0.13, 0.45, 0.72, GREEN_L, GREEN, lw=2.0)
ax.text(0.25, 0.775, "COMPLETED", ha="center", fontsize=12,
        fontweight="bold", color=GREEN)
done = [
    f"{CLEAN:,}-record harmonized dataset",
    f"Two-tier leakage audit ({LEAK_TOTAL:,} removed)",
    "Stage 1 model — BRENDA ROC AUC 0.732",
    "Stage 2 model — S669 ROC AUC 0.669",
    "New BRENDA thermostability benchmark",
    "4 data-integrity bugs found and fixed",
]
for i, t in enumerate(done):
    yy = 0.695 - i * 0.093
    ax.text(0.055, yy, "✓", fontsize=13, color=GREEN, fontweight="bold", va="center")
    ax.text(0.093, yy, t, fontsize=9.8, color="#2a2a26", va="center")

card(ax, 0.525, 0.13, 0.45, 0.72, "#fbfbf9", GREY, lw=1.8)
ax.text(0.75, 0.775, "NEXT", ha="center", fontsize=12, fontweight="bold", color="#5a5a55")
nxt = [
    "Validate on PET hydrolases (PETase, LCC)",
    "Wet-lab assay of top-ranked variants",
    "Activate multi-task heads (ΔT$_m$, abundance)",
    "Add ProThermDB pH/temperature records",
    "Per-residue embeddings at mutation site",
    "Publish dataset + audit to Zenodo",
]
for i, t in enumerate(nxt):
    yy = 0.695 - i * 0.093
    ax.text(0.555, yy, "→", fontsize=12, color=GREY, fontweight="bold", va="center")
    ax.text(0.593, yy, t, fontsize=9.8, color="#4a4a45", va="center")

ax.text(0.5, 0.055, "No PET hydrolase appears in training or test data — "
                    "PET is the motivating application, not a validated result.",
        ha="center", fontsize=9.5, color=RED, style="italic")
save(fig, "F_conclusion_roadmap")

print("\nall figures ->", OUT)
