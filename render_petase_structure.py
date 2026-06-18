"""
Render a publication-quality 3D image of the mutated IsPETase structure.

Loads the real RCSB 5XJH crystal structure (1.54 A) with the model's top
stabilizing mutations applied, and renders it via 3Dmol.js + headless Chrome:
  - white background, cartoon representation
  - mutated residues as red sticks + labels
  - catalytic triad (Ser160-Asp206-His237) as orange sticks
"""

import os, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PDB_PATH = "/tmp/ispetase_mut.pdb"
OUT_DIR = os.path.join(ROOT, "white_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

pdb_text = open(PDB_PATH).read()
# Escape for embedding in a JS template string
pdb_js = pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

MUTATIONS = [("S122G", 122), ("S187G", 187), ("V281G", 281), ("C239G", 239)]
TRIAD = [("Ser160", 160), ("Asp206", 206), ("His237", 237)]

mut_resi = ",".join(str(p) for _, p in MUTATIONS)
triad_resi = ",".join(str(p) for _, p in TRIAD)
mut_labels_js = "\n".join(
    f"viewer.addLabel('{name}', {{position:atomFor({pos}), backgroundColor:'0xD6504A', "
    f"backgroundOpacity:0.95, fontColor:'white', fontSize:18, borderThickness:1.5, "
    f"borderColor:'white', inFront:true}});"
    for name, pos in MUTATIONS
)

HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body{{margin:0;background:#ffffff}} #v{{width:1600px;height:1200px;position:relative}}</style>
</head><body>
<div id="v"></div>
<script>
const pdb = `{pdb_js}`;
const viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor:'white'}});
viewer.addModel(pdb, 'pdb');

function atomFor(resi) {{
  const sel = viewer.selectedAtoms({{resi:resi, atom:'CA'}});
  if (sel.length) return {{x:sel[0].x, y:sel[0].y, z:sel[0].z}};
  return {{x:0,y:0,z:0}};
}}

// Cartoon — soft blue-grey so the fold reads clearly
viewer.setStyle({{}}, {{cartoon:{{color:'#9fb6c9'}}}});

// Catalytic triad — orange sticks (Ser160-Asp206-His237)
viewer.setStyle({{resi:[{triad_resi}]}}, {{cartoon:{{color:'#9fb6c9'}}, stick:{{colorscheme:'orangeCarbon', radius:0.3}}}});

// Mutated residues — bold red sticks + spheres
viewer.setStyle({{resi:[{mut_resi}]}}, {{cartoon:{{color:'#9fb6c9'}}, stick:{{colorscheme:'redCarbon', radius:0.35}}, sphere:{{colorscheme:'redCarbon', radius:0.85}}}});

{mut_labels_js}

viewer.setViewStyle({{style:'outline', color:'black', width:0.05}});
viewer.zoomTo({{resi:[{mut_resi}]}});
viewer.zoom(0.55);
viewer.rotate(20, 'y');
viewer.render();
window.__done = true;
</script>
</body></html>"""

html_path = "/tmp/petase_render.html"
open(html_path, "w").write(HTML)
print(f"Wrote {html_path} ({len(HTML)} bytes)")

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out_png = os.path.join(OUT_DIR, "9_petase_3d_structure.png")
cmd = [
    chrome, "--headless=new", "--no-sandbox",
    "--hide-scrollbars", "--default-background-color=FFFFFFFF",
    "--window-size=1600,1200",
    # Software WebGL so 3Dmol can render without a GPU
    "--enable-webgl", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
    "--use-gl=angle", "--use-angle=swiftshader",
    "--virtual-time-budget=20000",
    f"--screenshot={out_png}",
    f"file://{html_path}",
]
print("Rendering via headless Chrome...")
subprocess.run(cmd, capture_output=True, timeout=90)
time.sleep(1)
if not os.path.exists(out_png):
    print("ERROR: screenshot not produced")
    raise SystemExit(1)
print(f"Raw render -> {out_png} ({os.path.getsize(out_png)} bytes)")

# ── Composite a publication figure: render + title + legend + DDG annotations ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# Auto-crop the white margins around the structure
img = Image.open(out_png).convert("RGB")
import numpy as np
arr = np.array(img)
mask = np.any(arr < 245, axis=2)
ys, xs = np.where(mask)
pad = 40
img = img.crop((max(xs.min()-pad,0), max(ys.min()-pad,0),
                min(xs.max()+pad, img.width), min(ys.max()+pad, img.height)))

INK, SUB, RED, ORANGE = "#1d2433", "#5a6678", "#d6504a", "#e8833a"
DDG = {"S122G": -0.66, "S187G": -0.60, "V281G": -0.54, "C239G": -0.54}

fig = plt.figure(figsize=(12, 9), facecolor="white")
ax = fig.add_axes([0.0, 0.0, 0.78, 0.92]); ax.imshow(img); ax.axis("off")

fig.text(0.5, 0.965, "IsPETase with Four Predicted Thermostabilizing Mutations",
         ha="center", fontsize=18, fontweight="bold", color=INK)
fig.text(0.5, 0.93,
         "Experimental crystal structure RCSB 5XJH (1.54 Å, X-ray) · mutations applied via NeRF side-chain modeling",
         ha="center", fontsize=10.5, color=SUB, style="italic")

# Right-side annotation panel
px = 0.795
fig.text(px, 0.86, "PREDICTED MUTATIONS", fontsize=11, fontweight="bold", color=RED)
y = 0.81
for m, d in DDG.items():
    fig.text(px, y, f"●  {m}", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(px+0.105, y, f"ΔΔG {d:+.2f}", fontsize=10.5, color=SUB, family="monospace")
    y -= 0.045

fig.text(px, 0.60, "CATALYTIC TRIAD", fontsize=11, fontweight="bold", color=ORANGE)
yy = 0.555
for t in ["Ser160", "Asp206", "His237"]:
    fig.text(px, yy, f"●  {t}", fontsize=11.5, color=INK); yy -= 0.04

leg = [Patch(facecolor=RED, label="Mutated residue"),
       Patch(facecolor=ORANGE, label="Catalytic triad"),
       Patch(facecolor="#9fb6c9", label="Backbone (cartoon)")]
fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(px-0.005, 0.30),
           fontsize=10, frameon=True, facecolor="white", edgecolor="#e6e9ef")

fig.text(px, 0.10,
         "All four substitutions verified\nat the correct residues in the\n1.54 Å crystal structure. Negative\nΔΔG = increased stability.",
         fontsize=8.5, color=SUB, va="top")

comp_png = os.path.join(OUT_DIR, "9_petase_3d_structure.png")
fig.savefig(comp_png, dpi=170, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Composite -> {comp_png}")
