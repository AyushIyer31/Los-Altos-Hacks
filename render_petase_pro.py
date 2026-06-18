"""
Publication-style render of mutated IsPETase — Nature-figure aesthetic.

  - secondary-structure coloring: blue alpha-helices, salmon beta-sheets, white loops
  - two views rotated 90 degrees with a rotation arrow
  - catalytic triad (Ser160-Asp206-His237) as labeled orange sticks
  - predicted stabilizing mutations as labeled green sticks + spheres
  - clean white background
"""
import os, subprocess, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "white_graphs"); os.makedirs(OUT, exist_ok=True)
PDB = "/tmp/ispetase_mut.pdb"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MUT = [("S122G",122),("S187G",187),("V281G",281),("C239G",239)]
TRIAD = [("Ser160",160),("Asp206",206),("His237",237)]
mut_resi = ",".join(str(p) for _,p in MUT)
triad_resi = ",".join(str(p) for _,p in TRIAD)

pdb_text = open(PDB).read()
pdb_js = pdb_text.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")

# Nature-style SS palette
HELIX = "#2f6fb3"   # blue
SHEET = "#e3a6b6"   # salmon/pink
LOOP  = "#e9ecef"   # near-white grey
MUTC  = "0x16a34a"  # green
TRIC  = "0xe8833a"  # orange

def render(rot_y, out_png, with_labels):
    labels = ""
    if with_labels:
        labels = "\n".join(
            f"viewer.addLabel('{n}',{{position:caOf({p}),backgroundColor:'{MUTC}',"
            f"backgroundOpacity:0.95,fontColor:'white',fontSize:15,borderThickness:1,borderColor:'white',inFront:true}});"
            for n,p in MUT)
        labels += "\n" + "\n".join(
            f"viewer.addLabel('{n}',{{position:caOf({p}),backgroundColor:'{TRIC}',"
            f"backgroundOpacity:0.95,fontColor:'white',fontSize:14,borderThickness:1,borderColor:'white',inFront:true}});"
            for n,p in TRIAD)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body{{margin:0;background:#fff}}#v{{width:1500px;height:1400px;position:relative}}</style>
</head><body><div id="v"></div><script>
const pdb=`{pdb_js}`;
const viewer=$3Dmol.createViewer(document.getElementById('v'),{{backgroundColor:'white'}});
viewer.addModel(pdb,'pdb');
function caOf(resi){{const s=viewer.selectedAtoms({{resi:resi,atom:'CA'}});return s.length?{{x:s[0].x,y:s[0].y,z:s[0].z}}:{{x:0,y:0,z:0}};}}
// Secondary-structure coloring (Nature style)
viewer.setStyle({{}},{{cartoon:{{color:'{LOOP}',thickness:0.4,arrows:true}}}});
viewer.setStyle({{ss:'h'}},{{cartoon:{{color:'{HELIX}',thickness:0.5,arrows:true}}}});
viewer.setStyle({{ss:'s'}},{{cartoon:{{color:'{SHEET}',thickness:0.5,arrows:true}}}});
// Catalytic triad — orange sticks
viewer.addStyle({{resi:[{triad_resi}]}},{{stick:{{color:'{TRIC}',radius:0.28}}}});
// Mutations — green sticks + spheres
viewer.addStyle({{resi:[{mut_resi}]}},{{stick:{{color:'{MUTC}',radius:0.32}},sphere:{{color:'{MUTC}',radius:0.75}}}});
{labels}
viewer.setViewStyle({{style:'outline',color:'#222222',width:0.04}});
viewer.zoomTo();
viewer.rotate({rot_y},'y');
viewer.zoom(1.30);
viewer.render();window.__done=true;
</script></body></html>"""
    p = f"/tmp/pro_view_{rot_y}.html"; open(p,"w").write(html)
    subprocess.run([CHROME,"--headless=new","--no-sandbox","--hide-scrollbars",
        "--default-background-color=FFFFFFFF","--window-size=1500,1400",
        "--enable-webgl","--ignore-gpu-blocklist","--enable-unsafe-swiftshader",
        "--use-gl=angle","--use-angle=swiftshader","--virtual-time-budget=20000",
        f"--screenshot={out_png}",f"file://{p}"],capture_output=True,timeout=90)
    time.sleep(1)
    return os.path.getsize(out_png) if os.path.exists(out_png) else 0

print("Rendering view 1 (front)...");  s1 = render(0,  "/tmp/pro1.png", True)
print(f"  {s1} bytes")
print("Rendering view 2 (90 deg)...");  s2 = render(90, "/tmp/pro2.png", False)
print(f"  {s2} bytes")

# ── Composite two-panel figure ────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Ellipse
from PIL import Image

def crop(path):
    im = Image.open(path).convert("RGB"); a = np.array(im)
    m = np.any(a < 245, axis=2); ys,xs = np.where(m); pad=30
    return im.crop((max(xs.min()-pad,0),max(ys.min()-pad,0),min(xs.max()+pad,im.width),min(ys.max()+pad,im.height)))

im1, im2 = crop("/tmp/pro1.png"), crop("/tmp/pro2.png")
INK,SUB = "#1d2433","#5a6678"

fig = plt.figure(figsize=(14,8.5), facecolor="white")
axA = fig.add_axes([0.02,0.06,0.45,0.84]); axA.imshow(im1); axA.axis("off")
axB = fig.add_axes([0.50,0.06,0.45,0.84]); axB.imshow(im2); axB.axis("off")

# rotation arrow between panels
axR = fig.add_axes([0.455,0.45,0.05,0.12]); axR.axis("off"); axR.set_xlim(0,1); axR.set_ylim(0,1)
axR.annotate("", xy=(0.95,0.5), xytext=(0.05,0.5),
             arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.8,connectionstyle="arc3,rad=0.4"))
axR.text(0.5,0.18,"90°",ha="center",fontsize=11,color=INK,fontweight="bold")

fig.text(0.03,0.945,"a",fontsize=20,fontweight="bold",color=INK)
fig.text(0.5,0.965,"IsPETase (RCSB 5XJH, 1.54 Å) — Predicted Thermostabilizing Mutations",
         ha="center",fontsize=16,fontweight="bold",color=INK)

# SS + feature legend
leg = [Patch(facecolor=HELIX,label="α-helix"),
       Patch(facecolor=SHEET,label="β-sheet"),
       Patch(facecolor=LOOP,edgecolor="#c4ccd8",label="loop"),
       Patch(facecolor="#16a34a",label="Stabilizing mutation"),
       Patch(facecolor="#e8833a",label="Catalytic triad")]
fig.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,-0.01),ncol=5,
           fontsize=10.5,frameon=True,facecolor="white",edgecolor="#e6e9ef")

out = os.path.join(OUT,"12_petase_pro_structure.png")
fig.savefig(out,dpi=180,bbox_inches="tight",facecolor="white"); plt.close()
print("Saved ->",out)
