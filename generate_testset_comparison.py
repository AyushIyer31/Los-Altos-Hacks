"""
Two-test-set comparison figure — uses ONLY the 10-fold CV and S669 benchmarks
(the two evaluations with measured ground-truth ddG). No fresh/unvalidated
predictions. White theme, conference-ready.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "white_graphs"); os.makedirs(OUT, exist_ok=True)
meta = json.loads(open(os.path.join(ROOT, "backend/app/trained_models/model_meta.json")).read())

INK, SUB, GRID, TEAL, NAVY, RED = "#1d2433", "#5a6678", "#e6e9ef", "#0f9e8e", "#1b4965", "#d6504a"
rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
                 "font.family":"DejaVu Sans","text.color":INK,"axes.labelcolor":INK,
                 "xtick.color":SUB,"ytick.color":SUB,"axes.edgecolor":"#c4ccd8"})

# ── Values from the two test sets only ────────────────────────────────────────
CV = {"Accuracy":79.69, "F1":79.72, "Pearson":76.41, "MAE":0.92}    # 10-fold GroupKFold
S6 = {"Accuracy":73.09, "F1":36.62, "Pearson":40.00, "MAE":1.10}    # S669 held-out

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,6),gridspec_kw={"width_ratios":[2.1,1]})

# ── Left: grouped bars (scores as %, MAE handled on right) ────────────────────
ax1.set_facecolor("white"); ax1.grid(axis="y",color=GRID,lw=1,zorder=0); ax1.set_axisbelow(True)
for s in ["top","right"]: ax1.spines[s].set_visible(False)
metrics = ["Accuracy","F1","Pearson"]
x = np.arange(len(metrics)); w = 0.36
b1 = ax1.bar(x-w/2,[CV[m] for m in metrics],w,color=TEAL,zorder=3,label="10-fold Cross-Validation")
b2 = ax1.bar(x+w/2,[S6[m] for m in metrics],w,color=NAVY,zorder=3,label="S669 Independent Test")
for bars in (b1,b2):
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{b.get_height():.1f}",
                 ha="center",fontsize=10,fontweight="bold",color=INK)
ax1.set_xticks(x); ax1.set_xticklabels(["Accuracy","F1 score","Pearson r"],fontsize=11)
ax1.set_ylim(0,95); ax1.set_ylabel("Score (%, Pearson ×100)",fontsize=11)
ax1.set_title("Classification & Correlation",fontsize=12.5,fontweight="bold",color=INK,pad=8)
ax1.legend(loc="upper right",fontsize=10,frameon=True,facecolor="white",edgecolor=GRID)

# ── Right: MAE (different units) ──────────────────────────────────────────────
ax2.set_facecolor("white"); ax2.grid(axis="y",color=GRID,lw=1,zorder=0); ax2.set_axisbelow(True)
for s in ["top","right"]: ax2.spines[s].set_visible(False)
bb = ax2.bar(["CV","S669"],[CV["MAE"],S6["MAE"]],color=[TEAL,NAVY],width=0.5,zorder=3)
for b in bb:
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{b.get_height():.2f}",
             ha="center",fontsize=11,fontweight="bold",color=INK)
ax2.set_ylim(0,1.4); ax2.set_ylabel("MAE (kcal/mol, lower = better)",fontsize=11)
ax2.set_title("Prediction Error",fontsize=12.5,fontweight="bold",color=INK,pad=8)

fig.suptitle("Model Performance on the Two Ground-Truth Benchmarks",
             fontsize=15,fontweight="bold",color=INK,y=1.0)
fig.text(0.5,-0.02,
         "10-fold GroupKFold cross-validation (19,071 training mutations)  vs.  "
         "S669 independent test set (669 mutations, unseen proteins). Both have measured ΔΔG.",
         ha="center",fontsize=9.5,color=SUB,style="italic")
fig.tight_layout(rect=[0,0.02,1,0.97])
out = os.path.join(OUT,"11_two_testset_comparison.png")
fig.savefig(out,dpi=170,bbox_inches="tight",facecolor="white"); plt.close()
print("Saved ->",out)
print(f"CV: {CV}")
print(f"S669: {S6}")
