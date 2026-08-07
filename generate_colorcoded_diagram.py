"""
Color-coded direction diagram for a user-supplied sequence.

Runs single mutations through the v46 model, selects a clear mix of
predicted-stabilizing (green) and predicted-destabilizing (red) mutations,
and renders them on the resolved 3D structure — visualizing the model's
sign/direction accuracy (the abstract's headline strength).
"""

import os, sys, subprocess, time, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)
OUT = os.path.join(ROOT, "white_graphs")
os.makedirs(OUT, exist_ok=True)

SEQ = ("ADTEAATRYPVILVHGLAGTDKFANVVDYWYGIQSDLQSHGAKVYVANLSGFQSDDGPNGRGEQLLAEVKQVLAATGATKVNL"
       "IGHSQGGLTSRYVAAVAPQLVASVTTIGTPHRGSEFADFVQDVLKTDPTGLSSTVIAAFVNVFGTLVSSSHNTDQDALAALRT"
       "LTTAQTATYNRNFPSAGLGAPGSCQTGAATETVGGSQHLLYSWGGTAIQPTSTVLGVTGATDTSTGTLDVANVTDPSTLALLA"
       "TGAVMINRASGQNDGLVSRCSSLFGQVISTSYHWNHLDEINQLLGVRGANAEDPVAVIRTHVNRLKLQGV")
STRUCT_PDB = "/tmp/lcc_struct.pdb"   # 1CVL, resolved + verified (99.4% identity)

# ── 1. Predict single mutations through the model ─────────────────────────────
print("Loading model + scanning mutations...")
from app.services import trained_classifier as tc
tc.train_model()

PANEL = list("GAVLIFEKDRSTNQ")   # representative substitutions
tuples = []
for i, wt in enumerate(SEQ, start=1):
    for mut in PANEL:
        if mut != wt:
            tuples.append((wt, i, mut))

preds = tc.predict_mutations_batch(tuples, sequence=SEQ, protein_id="LCC_USER")
thr = tc.get_optimal_threshold()

rows = []
for (wt, pos, mut), p in zip(tuples, preds):
    rows.append((pos, f"{wt}{pos}{mut}", p["predicted_ddg"], p["predicted_beneficial"]))

# ── 2. Select a clear, well-separated mix ─────────────────────────────────────
# Strongest stabilizing (most negative ddg) and strongest destabilizing,
# one per position, spread along the chain so labels don't collide.
stab = sorted([r for r in rows if r[2] < thr], key=lambda r: r[2])
dest = sorted([r for r in rows if r[2] >= thr], key=lambda r: -r[2])

def pick(cands, k, min_gap=18):
    chosen = []
    for r in cands:
        if all(abs(r[0]-c[0]) >= min_gap for c in chosen):
            chosen.append(r)
        if len(chosen) == k:
            break
    return chosen

green = pick(stab, 5)
red   = pick(dest, 5)
print(f"  threshold={thr:.3f}")
print("  GREEN (stabilizing):", [(r[1], round(r[2],2)) for r in green])
print("  RED   (destabilizing):", [(r[1], round(r[2],2)) for r in red])

# ── 3. Verify each position matches the structure's residue ───────────────────
three2one = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
             'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
             'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
sres = {}
for l in open(STRUCT_PDB):
    if l[:4] == "ATOM":
        rn = int(l[22:26]); sres.setdefault(rn, three2one.get(l[17:20].strip(), 'X'))

def verify(sel):
    ok = []
    for pos, label, ddg, ben in sel:
        wt = label[0]
        if sres.get(pos) == wt:
            ok.append((pos, label, ddg))
        else:
            print(f"  ! {label}: structure has {sres.get(pos)} at {pos}, expected {wt} — skipped")
    return ok
green_ok = verify(green); red_ok = verify(red)

# ── 4. Build 3Dmol render page ────────────────────────────────────────────────
pdb_text = open(STRUCT_PDB).read()
pdb_js = pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
GREEN, RED = "0x10B981", "0xEF4444"

def style_block(positions, color):
    resi = ",".join(str(p) for p, _, _ in positions)
    if not resi:
        return ""
    return (f"viewer.setStyle({{resi:[{resi}]}}, {{cartoon:{{color:'#b9c6d4'}}, "
            f"stick:{{color:'{color}', radius:0.35}}, sphere:{{color:'{color}', radius:0.9}}}});")

def label_block(positions, bg):
    out = []
    for p, label, ddg in positions:
        out.append(f"viewer.addLabel('{label}', {{position:atomFor({p}), backgroundColor:'{bg}', "
                   f"backgroundOpacity:0.95, fontColor:'white', fontSize:15, borderThickness:1, "
                   f"borderColor:'white', inFront:true}});")
    return "\n".join(out)

HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body{{margin:0;background:#fff}} #v{{width:1600px;height:1200px;position:relative}}</style>
</head><body><div id="v"></div><script>
const pdb=`{pdb_js}`;
const viewer=$3Dmol.createViewer(document.getElementById('v'),{{backgroundColor:'white'}});
viewer.addModel(pdb,'pdb');
function atomFor(resi){{const s=viewer.selectedAtoms({{resi:resi,atom:'CA'}});return s.length?{{x:s[0].x,y:s[0].y,z:s[0].z}}:{{x:0,y:0,z:0}};}}
viewer.setStyle({{}},{{cartoon:{{color:'#b9c6d4'}}}});
{style_block(green_ok, GREEN)}
{style_block(red_ok, RED)}
{label_block(green_ok, GREEN)}
{label_block(red_ok, RED)}
viewer.setViewStyle({{style:'outline',color:'black',width:0.05}});
viewer.zoomTo();viewer.zoom(1.15);viewer.render();window.__done=true;
</script></body></html>"""
html_path = "/tmp/lcc_render.html"; open(html_path, "w").write(HTML)

raw_png = "/tmp/lcc_raw.png"
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--no-sandbox", "--hide-scrollbars",
                "--default-background-color=FFFFFFFF", "--window-size=1600,1200",
                "--enable-webgl", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
                "--use-gl=angle", "--use-angle=swiftshader", "--virtual-time-budget=20000",
                f"--screenshot={raw_png}", f"file://{html_path}"], capture_output=True, timeout=90)
time.sleep(1)
print(f"Raw render: {os.path.getsize(raw_png)} bytes")

# ── 5. Composite publication figure ───────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

img = Image.open(raw_png).convert("RGB")
a = np.array(img); m = np.any(a < 245, axis=2); ys, xs = np.where(m); pad = 35
img = img.crop((max(xs.min()-pad,0), max(ys.min()-pad,0), min(xs.max()+pad,img.width), min(ys.max()+pad,img.height)))

INK, SUB = "#1d2433", "#5a6678"; GREENc, REDc = "#10B981", "#EF4444"
fig = plt.figure(figsize=(13, 9), facecolor="white")
ax = fig.add_axes([0.0, 0.0, 0.76, 0.9]); ax.imshow(img); ax.axis("off")
fig.text(0.5, 0.955, "Predicted Stability Effect of Mutations — Color-Coded by Direction",
         ha="center", fontsize=17, fontweight="bold", color=INK)

px = 0.775
fig.text(px, 0.86, "STABILIZING  (green)", fontsize=11.5, fontweight="bold", color=GREENc)
y = 0.815
for p, label, ddg in green_ok:
    fig.text(px, y, f"● {label}", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(px+0.115, y, f"ΔΔG {ddg:+.2f}", fontsize=10, color=SUB, family="monospace"); y -= 0.04
y -= 0.02
fig.text(px, y, "DESTABILIZING  (red)", fontsize=11.5, fontweight="bold", color=REDc); y -= 0.045
for p, label, ddg in red_ok:
    fig.text(px, y, f"● {label}", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(px+0.115, y, f"ΔΔG {ddg:+.2f}", fontsize=10, color=SUB, family="monospace"); y -= 0.04

fig.legend(handles=[Patch(facecolor=GREENc, label="Predicted stabilizing (ΔΔG < 0)"),
                    Patch(facecolor=REDc, label="Predicted destabilizing (ΔΔG > 0)"),
                    Patch(facecolor="#b9c6d4", label="Backbone (cartoon)")],
           loc="lower left", bbox_to_anchor=(px-0.01, 0.12), fontsize=9.5,
           frameon=True, facecolor="white", edgecolor="#e6e9ef")

out = os.path.join(OUT, "10_colorcoded_direction.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved -> {out}")
