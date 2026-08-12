"""Slide-ready result figures for the TWO-STAGE paper (not the old 19k model, no pH).
All values are real:
  - Stage 1 scatter from multitask/models_tm/brenda_tm_predictions.csv
  - Stage 2 per-model bars from the confirmed S669 table (guide Table 3A)
  - Internal-vs-external from tm_results.json / the confirmed numbers
Outputs PNG + PDF into paper_figures/ and the project root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "paper_figures")
CSV = os.path.join(ROOT, "multitask", "models_tm", "brenda_tm_predictions.csv")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})
INK = "#1a1a1a"; ACCENT = "#1a6a72"; GREY = "#9aa0a0"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
        fig.savefig(os.path.join(ROOT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ============================================================ Fig 1: Stage 1 scatter
df = pd.read_csv(CSV)
x = df["true_stable_temp_c"].to_numpy(float)
y = df["pred_tm_ensemble"].to_numpy(float)
is_tm = df["basis"].isin(["Tm", "Tm_wt"]).to_numpy()
r_all = np.corrcoef(x, y)[0, 1]
r_tm = np.corrcoef(x[is_tm], y[is_tm])[0, 1]

fig, ax = plt.subplots(figsize=(5.2, 5.0))
lims = [min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3]
ax.plot(lims, lims, "--", color=GREY, lw=1, zorder=1, label="perfect prediction")
ax.scatter(x[~is_tm], y[~is_tm], s=12, c=GREY, alpha=0.45, edgecolors="none", zorder=2,
           label=f"inferred stability ceiling (n={(~is_tm).sum()})")
ax.scatter(x[is_tm], y[is_tm], s=16, c=ACCENT, alpha=0.85, edgecolors="none", zorder=3,
           label=f"measured $T_m$ (n={is_tm.sum()})")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Measured stable temperature (°C)")
ax.set_ylabel("Predicted $T_m$ (°C)")
ax.set_title("Stage 1: predicted vs. measured thermostability (BRENDA)", fontsize=11.5)
ax.text(0.04, 0.96, f"Pearson $r$ = {r_all:.2f} (all)\nPearson $r$ = {r_tm:.2f} (measured $T_m$)",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.7))
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
save(fig, "stage1_tm_scatter")


# ============================================================ Fig 2: Stage 2 per-model
models = ["Extra Trees", "XGBoost", "Random Forest", "LightGBM", "CatBoost", "MLP", "Ensemble"]
pearson = [0.436, 0.382, 0.369, 0.330, 0.324, 0.298, 0.390]
rmse = [1.481, 1.514, 1.523, 1.549, 1.557, 1.612, 1.509]
order = np.argsort(pearson)[::-1]
models = [models[i] for i in order]; pearson = [pearson[i] for i in order]; rmse = [rmse[i] for i in order]
colors = [ACCENT if m == "Ensemble" else GREY for m in models]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
a1.barh(models[::-1], pearson[::-1], color=colors[::-1], height=0.62)
a1.set_xlabel("Pearson $r$ (higher is better)"); a1.set_xlim(0, 0.5)
a1.set_title("S669 regression: correlation", fontsize=10.5)
for i, v in enumerate(pearson[::-1]):
    a1.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5)
a2.barh(models[::-1], rmse[::-1], color=colors[::-1], height=0.62)
a2.set_xlabel("RMSE kcal mol$^{-1}$ (lower is better)"); a2.set_xlim(0, 1.8)
a2.set_title("S669 regression: error", fontsize=10.5)
for i, v in enumerate(rmse[::-1]):
    a2.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=8.5)
a2.set_yticklabels([])
fig.suptitle("Stage 2 per-model performance on the independent S669 benchmark", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "stage2_model_comparison")


# ============================================================ Fig 3: internal vs external
fig, ax = plt.subplots(figsize=(5.6, 4.2))
groups = ["Stage 1\n($T_m$, Pearson $r$)", "Stage 2\n(ΔΔG, Pearson $r$)"]
internal = [0.80, np.nan]   # Stage 1 dev val ~0.80; Stage 2 internal not reported here
external = [0.54, 0.39]
xpos = np.arange(len(groups)); w = 0.36
ax.bar(xpos - w/2, internal, w, color=GREY, label="Internal validation (development)")
ax.bar(xpos + w/2, external, w, color=ACCENT, label="Independent test (BRENDA / S669)")
for i, v in enumerate(internal):
    if not np.isnan(v):
        ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(external):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(groups)
ax.set_ylabel("Pearson correlation"); ax.set_ylim(0, 1.0)
ax.set_title("Honest generalization: internal vs. independent", fontsize=11.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.text(1 + w/2, 0.06, "Stage 2 internal\nnot reported", ha="center", fontsize=7.5, color=GREY)
save(fig, "internal_vs_external")

print("done ->", FIGDIR)
