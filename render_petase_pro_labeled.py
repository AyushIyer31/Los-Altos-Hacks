"""
Publication-style IsPETase render with BOTH stabilizing (green) and
destabilizing (red) residues labeled. Nature-figure SS coloring.
"""
import os, sys, subprocess, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
OUT = os.path.join(ROOT, "white_graphs"); os.makedirs(OUT, exist_ok=True)
PDB = "/tmp/5xjh_orig.pdb"          # clean WT crystal structure (chain A, res 30-292)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ISPETASE = ("MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGY"
            "TARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPS"
            "LKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMD"
            "NDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ")

# ── Scan mutations through the model ──────────────────────────────────────────
print("Scanning IsPETase mutations...")
from app.services import trained_classifier as tc
tc.train_model()
thr = tc.get_optimal_threshold()

# Structure residue identities (chain A) — only label positions present + matching WT
three2one = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
             'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
             'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
sres = {}
for l in open(PDB):
    if l[:4] == "ATOM" and l[21] == "A":
        rn = int(l[22:26]); sres.setdefault(rn, three2one.get(l[17:20].strip(), 'X'))

PANEL = list("GAVLIFEKDRST")
CATALYTIC = {160, 206, 237}                # Ser160-Asp206-His237 — never mutate (kills catalysis)
tuples = []
for i, wt in enumerate(ISPETASE, start=1):
    if i in CATALYTIC:
        continue
    if i in sres and sres[i] == wt:        # only positions resolved in the crystal
        for mut in PANEL:
            if mut != wt:
                tuples.append((wt, i, mut))
preds = tc.predict_mutations_batch(tuples, sequence=ISPETASE, protein_id="5XJH")
rows = [(p_pos, f"{wt}{p_pos}{mut}", pr["predicted_ddg"]) for (wt,p_pos,mut),pr in zip(tuples,preds)]

stab = sorted([r for r in rows if r[2] < thr], key=lambda r: r[2])
dest = sorted([r for r in rows if r[2] >= thr], key=lambda r: -r[2])
def pick(c,k,gap=20):
    out=[]
    for r in c:
        if all(abs(r[0]-o[0])>=gap for o in out): out.append(r)
        if len(out)==k: break
    return out
green = pick(stab,4); red = pick(dest,4)
print("  GREEN:", [(r[1],round(r[2],2)) for r in green])
print("  RED:  ", [(r[1],round(r[2],2)) for r in red])

# ── Render (Nature SS style) ──────────────────────────────────────────────────
pdb_js = open(PDB).read().replace("\\","\\\\").replace("`","\\`").replace("$","\\$")
HELIX,SHEET,LOOP = "#2f6fb3","#e3a6b6","#e9ecef"
GREENc,REDc = "0x16a34a","0xdc2626"

def labels_js(sel,color):
    return "\n".join(
        f"viewer.addLabel('{lab}',{{position:caOf({p}),backgroundColor:'{color}',"
        f"backgroundOpacity:0.96,fontColor:'white',fontSize:18,borderThickness:1.5,"
        f"borderColor:'white',inFront:true}});" for p,lab,_ in sel)

green_resi = ",".join(str(p) for p,_,_ in green)
red_resi   = ",".join(str(p) for p,_,_ in red)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body{{margin:0;background:#fff}}#v{{width:1600px;height:1300px;position:relative}}</style>
</head><body><div id="v"></div><script>
const pdb=`{pdb_js}`;
const viewer=$3Dmol.createViewer(document.getElementById('v'),{{backgroundColor:'white'}});
viewer.addModel(pdb,'pdb');
function caOf(resi){{const s=viewer.selectedAtoms({{resi:resi,atom:'CA',chain:'A'}});return s.length?{{x:s[0].x,y:s[0].y,z:s[0].z}}:{{x:0,y:0,z:0}};}}
viewer.setStyle({{}},{{cartoon:{{color:'{LOOP}',arrows:true}}}});
viewer.setStyle({{ss:'h'}},{{cartoon:{{color:'{HELIX}',arrows:true}}}});
viewer.setStyle({{ss:'s'}},{{cartoon:{{color:'{SHEET}',arrows:true}}}});
viewer.addStyle({{resi:[{green_resi}]}},{{stick:{{color:'{GREENc}',radius:0.34}},sphere:{{color:'{GREENc}',radius:0.85}}}});
viewer.addStyle({{resi:[{red_resi}]}},{{stick:{{color:'{REDc}',radius:0.34}},sphere:{{color:'{REDc}',radius:0.85}}}});
{labels_js(green,GREENc)}
{labels_js(red,REDc)}
viewer.setViewStyle({{style:'outline',color:'#222',width:0.045}});
viewer.zoomTo();viewer.zoom(1.25);viewer.render();window.__done=true;
</script></body></html>"""
open("/tmp/pro_lab.html","w").write(html)
subprocess.run([CHROME,"--headless=new","--no-sandbox","--hide-scrollbars",
    "--default-background-color=FFFFFFFF","--window-size=1600,1300",
    "--enable-webgl","--ignore-gpu-blocklist","--enable-unsafe-swiftshader",
    "--use-gl=angle","--use-angle=swiftshader","--virtual-time-budget=20000",
    "--screenshot=/tmp/pro_lab.png","file:///tmp/pro_lab.html"],capture_output=True,timeout=90)
time.sleep(1)
print("Raw render:", os.path.getsize("/tmp/pro_lab.png"), "bytes")

# ── Composite ─────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
im = Image.open("/tmp/pro_lab.png").convert("RGB"); a=np.array(im)
m=np.any(a<245,axis=2); ys,xs=np.where(m); pad=30
im=im.crop((max(xs.min()-pad,0),max(ys.min()-pad,0),min(xs.max()+pad,im.width),min(ys.max()+pad,im.height)))
INK,SUB = "#1d2433","#5a6678"; G,R = "#16a34a","#dc2626"

fig=plt.figure(figsize=(13,9),facecolor="white")
ax=fig.add_axes([0.0,0.04,0.76,0.88]); ax.imshow(im); ax.axis("off")
fig.text(0.5,0.965,"IsPETase — Predicted Stabilizing vs. Destabilizing Residues",
         ha="center",fontsize=15,fontweight="bold",color=INK)
px=0.775
fig.text(px,0.86,"STABILIZING  (green)",fontsize=12,fontweight="bold",color=G); y=0.815
for p,lab,d in green:
    fig.text(px,y,f"● {lab}",fontsize=12,fontweight="bold",color=INK)
    fig.text(px+0.12,y,f"ΔΔG {d:+.2f}",fontsize=10,color=SUB,family="monospace"); y-=0.045
y-=0.02
fig.text(px,y,"DESTABILIZING  (red)",fontsize=12,fontweight="bold",color=R); y-=0.05
for p,lab,d in red:
    fig.text(px,y,f"● {lab}",fontsize=12,fontweight="bold",color=INK)
    fig.text(px+0.12,y,f"ΔΔG {d:+.2f}",fontsize=10,color=SUB,family="monospace"); y-=0.045
fig.legend(handles=[Patch(facecolor=G,label="Stabilizing"),Patch(facecolor=R,label="Destabilizing")],
           loc="lower left",bbox_to_anchor=(px-0.01,0.05),fontsize=10.5,frameon=True,
           facecolor="white",edgecolor="#e6e9ef")
out=os.path.join(OUT,"13_petase_pro_labeled.png")
fig.savefig(out,dpi=180,bbox_inches="tight",facecolor="white"); plt.close()
print("Saved ->",out)
