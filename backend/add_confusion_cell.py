"""Append confusion matrices + regression diagnostics (green theme) to the notebook."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "PETase_Stability_Analysis.ipynb")
nb = nbf.read(NB, as_version=4)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

nb.cells.append(md(r"""
---
# 18 · Confusion matrices at the screening operating points

The confusion matrices in section 11 were taken at each stage's **best-F1**
threshold. That is the wrong operating point for this application: F1 weights
precision and recall equally, but a false positive costs a wet-lab cycle while a
false negative costs almost nothing.

The matrices below are taken at the thresholds actually advocated — Stage 1 at
66 °C and Stage 2 at 0.00 kcal/mol — and each cell is labelled with what it means
experimentally rather than only TP/FP/FN/TN.
"""))

nb.cells.append(code(r'''
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ---- green palette (matches the poster) ----
G_DARK, G_MID, G_LEAF = "#2f4a22", "#4f7a3a", "#7aa860"
G_PALE, G_WASH = "#cfe0c2", "#eef4e9"
GREENS = LinearSegmentedColormap.from_list("poster_greens",
                                           ["#f6faf3", "#cfe0c2", "#7aa860", "#4f7a3a", "#2f4a22"])

def confusion(y, positive_mask):
    tp = int((positive_mask & (y == 1)).sum()); fp = int((positive_mask & (y == 0)).sum())
    fn = int((~positive_mask & (y == 1)).sum()); tn = int((~positive_mask & (y == 0)).sum())
    return tp, fp, fn, tn

S1_THR, S2_THR = 66.0, 0.00
tp1, fp1, fn1, tn1 = confusion(y1, s1 >= S1_THR)
tp2, fp2, fn2, tn2 = confusion(y2, ens2 < S2_THR)

PANELS = [
    dict(title="Stage 1 — BRENDA", sub=f"decision threshold {S1_THR:.0f} °C",
         cm=(tp1, fp1, fn1, tn1), n=len(y1),
         pos="thermostable", neg="not thermostable",
         meaning={"tp": "correctly identified\nas thermostable",
                  "fp": "wrongly advanced —\nwasted campaign",
                  "fn": "missed, but many\nalternatives remain",
                  "tn": "correctly screened out"}),
    dict(title="Stage 2 — S669", sub=f"decision threshold {S2_THR:.2f} kcal/mol",
         cm=(tp2, fp2, fn2, tn2), n=len(y2),
         pos="stabilising", neg="destabilising",
         meaning={"tp": "correctly prioritised\nfor the bench",
                  "fp": "wasted assay",
                  "fn": "missed, but only ~20\ncan be tested anyway",
                  "tn": "correctly rejected"}),
]

fig = plt.figure(figsize=(14.0, 6.6))
fig.suptitle("Confusion matrices at the advocated screening thresholds",
             fontsize=14, fontweight="bold", y=.975, color=G_DARK)

for k, P in enumerate(PANELS):
    tp, fp, fn, tn = P["cm"]
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    vmax = max(P["cm"])
    x0 = .045 + k * .500

    ax = fig.add_axes([x0, .205, .375, .555]); ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # (col, row), value, key, is-on-diagonal
    cells = [((0, 1), tn, "tn", True), ((1, 1), fp, "fp", False),
             ((0, 0), fn, "fn", False), ((1, 0), tp, "tp", True)]
    for (cx, cy), val, key, diag in cells:
        shade = GREENS(0.12 + 0.88 * (val / vmax) ** 0.55)
        lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
        txt = "white" if lum < 0.55 else G_DARK
        ax.add_patch(FancyBboxPatch((cx + .035, cy + .035), .93, .93,
                     boxstyle="round,pad=0.004,rounding_size=.03",
                     fc=shade, ec=G_DARK if diag else G_PALE,
                     lw=2.4 if diag else 1.3, ls="-" if diag else "--"))
        ax.text(cx + .5, cy + .70, f"{val:,}", ha="center", va="center",
                fontsize=27, fontweight="bold", color=txt)
        ax.text(cx + .5, cy + .41, key.upper(), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=txt, alpha=.8)
        ax.text(cx + .5, cy + .20, P["meaning"][key], ha="center", va="center",
                fontsize=7.6, color=txt, alpha=.9, linespacing=1.4)

    ax.text(.5, 2.07, f"predicted\n{P['neg']}", ha="center", va="bottom",
            fontsize=9, color=G_MID, linespacing=1.35)
    ax.text(1.5, 2.07, f"predicted\n{P['pos']}", ha="center", va="bottom",
            fontsize=9, color=G_MID, linespacing=1.35)
    ax.text(-.06, 1.5, f"true\n{P['neg']}", ha="right", va="center",
            fontsize=9, color=G_MID, linespacing=1.35)
    ax.text(-.06, .5, f"true\n{P['pos']}", ha="right", va="center",
            fontsize=9, color=G_MID, linespacing=1.35)
    ax.set_title(f"{P['title']}   (n = {P['n']:,})\n{P['sub']}",
                 fontsize=12, fontweight="bold", pad=56, color=G_DARK)

    axm = fig.add_axes([x0, .058, .375, .108]); axm.axis("off")
    axm.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=.05",
                  fc=G_WASH, ec=G_LEAF, lw=1.4, transform=axm.transAxes))
    for i, (lab, val) in enumerate([("precision", f"{prec:.3f}"), ("recall", f"{rec:.3f}"),
                                    ("of every 100 calls", f"{prec*100:.0f} are right")]):
        axm.text(.17 + i * .33, .63, val, ha="center", va="center",
                 fontsize=13, fontweight="bold", color=G_DARK, transform=axm.transAxes)
        axm.text(.17 + i * .33, .25, lab, ha="center", va="center",
                 fontsize=8, color=G_MID, transform=axm.transAxes)

save(fig, "fig10_confusion_operating"); plt.show()
'''))

nb.cells.append(md(r"""
---
# 19 · Regression diagnostics

The classification view thresholds the prediction. This section looks at the
underlying regression directly: predicted ΔΔG against the experimental value for
all 669 S669 mutations, the residual distribution, and how the six ensemble
members compare individually.
"""))

nb.cells.append(code(r'''
resid = ens2 - y2_ddg
sign_ok = np.sign(ens2) == np.sign(y2_ddg)

fig = plt.figure(figsize=(15.0, 5.3))

# ---------------- (a) predicted vs experimental ----------------
axA = fig.add_axes([.045, .150, .275, .700])
lim = [min(y2_ddg.min(), ens2.min()) - .4, max(y2_ddg.max(), ens2.max()) + .4]
axA.plot(lim, lim, "--", color="#aaa", lw=1.1, zorder=1, label="perfect prediction")
axA.axhline(0, color="#ddd", lw=.8, zorder=1); axA.axvline(0, color="#ddd", lw=.8, zorder=1)
axA.scatter(y2_ddg[sign_ok], ens2[sign_ok], s=14, c=G_MID, alpha=.60,
            edgecolors="none", zorder=3, label=f"sign correct (n={sign_ok.sum()})")
axA.scatter(y2_ddg[~sign_ok], ens2[~sign_ok], s=14, c=G_PALE, alpha=.85,
            edgecolors="#9ab98a", linewidths=.4, zorder=2,
            label=f"sign wrong (n={(~sign_ok).sum()})")
axA.set(xlim=lim, ylim=lim, xlabel="experimental ΔΔG (kcal/mol)",
        ylabel="predicted ΔΔG (kcal/mol)")
axA.set_title("Predicted vs. experimental", fontsize=11.5, fontweight="bold", color=G_DARK)
axA.legend(fontsize=7.6, frameon=False, loc="upper left")
axA.grid(alpha=.2); axA.set_aspect("equal")
axA.text(.97, .04, f"Pearson r = {np.corrcoef(y2_ddg, ens2)[0,1]:.3f}\n"
                   f"RMSE = {np.sqrt((resid**2).mean()):.3f}\n"
                   f"sign agreement = {sign_ok.mean():.1%}",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=8.3,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=G_LEAF, lw=.9))

# ---------------- (b) residuals ----------------
axB = fig.add_axes([.390, .150, .240, .700])
axB.hist(resid, bins=45, color=G_MID, alpha=.85, edgecolor="white", linewidth=.4)
axB.axvline(0, color=G_DARK, ls="--", lw=1.4)
axB.axvline(resid.mean(), color="#8a5a1a", ls=":", lw=1.5,
            label=f"mean {resid.mean():+.2f}")
axB.set(xlabel="predicted − experimental (kcal/mol)", ylabel="count")
axB.set_title("Residual distribution", fontsize=11.5, fontweight="bold", color=G_DARK)
axB.legend(fontsize=8, frameon=False)
axB.grid(axis="y", alpha=.2)
axB.text(.03, .96, "negative mean = the model\nunder-predicts destabilisation",
         transform=axB.transAxes, va="top", fontsize=7.8, color=G_MID,
         linespacing=1.45)

# ---------------- (c) per-model comparison ----------------
axC = fig.add_axes([.700, .150, .270, .700])
pm = per_model[per_model.model != "ENSEMBLE (mean)"].sort_values("pearson_r")
cols = [G_DARK if m == "extra_trees" else G_LEAF for m in pm.model]
axC.barh(pm.model, pm.pearson_r, color=cols, height=.62, zorder=3)
ens_r = float(per_model.loc[per_model.model == "ENSEMBLE (mean)", "pearson_r"].iloc[0])
axC.axvline(ens_r, color="#8a5a1a", ls="--", lw=1.6, zorder=4,
            label=f"ensemble mean ({ens_r:.3f})")
for i, v in enumerate(pm.pearson_r):
    axC.text(v - .012, i, f"{v:.3f}", va="center", ha="right",
             fontsize=8.4, color="white", fontweight="bold")
axC.set(xlabel="Pearson r on S669", xlim=(0, .52))
axC.set_title("Individual ensemble members", fontsize=11.5, fontweight="bold", color=G_DARK)
axC.legend(fontsize=7.8, frameon=True, framealpha=.92, edgecolor="none",
           loc="lower right", bbox_to_anchor=(1.0, .02))
axC.grid(axis="x", alpha=.2); axC.set_axisbelow(True)
axC.tick_params(labelsize=8.5)

for ax in (axA, axB, axC):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle("Stage 2 regression diagnostics on the independent S669 benchmark",
             fontsize=13.5, fontweight="bold", y=.985, color=G_DARK)
save(fig, "fig11_regression_diagnostics"); plt.show()

print(f"sign agreement       {sign_ok.mean():.1%}  ({sign_ok.sum()}/{len(sign_ok)})")
print(f"mean residual        {resid.mean():+.3f} kcal/mol")
print(f"predicted range      {ens2.min():.2f} to {ens2.max():.2f}")
print(f"experimental range   {y2_ddg.min():.2f} to {y2_ddg.max():.2f}")
print("\nThe predicted range is far narrower than the experimental range: the")
print("ensemble regresses toward the mean and rarely commits to a large effect.")
print("That is why sign agreement (76.5%) is much better than the correlation")
print("(0.390) alone would suggest — direction is easier than magnitude.")
'''))

nbf.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
