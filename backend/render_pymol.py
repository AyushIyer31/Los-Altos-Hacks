"""Ray-traced PyMOL renders of the AlphaFold model + the scored mutation.

Runs in the isolated `pymol-render` conda env (PyMOL 3.1), NOT the main project
environment:

    mamba run -n pymol-render python backend/render_pymol.py

IMPORTANT — only the LOV domain is rendered. The ddG for N92Y was measured on an
isolated 132-residue LOV-domain construct (crystallised as 2PR5, residues 21-147).
The AlphaFold model covers full-length YtvA (261 residues: LOV domain + a linker
helix + a STAS domain). Showing the whole protein would imply the measurement
refers to something it does not, so the render is restricted to the construct.

Outputs transparent-background PNGs into paper_figures/pymol/.
"""
import os
import pymol
from pymol import cmd

pymol.finish_launching(["pymol", "-qc"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AF = os.path.join(ROOT, "alphafold_structures", "AF-O34627-F1-v6.pdb")
OUT = os.path.join(ROOT, "paper_figures", "pymol")
os.makedirs(OUT, exist_ok=True)

POS = 107                 # UniProt / PDB numbering of the mutated residue
LOV_LO, LOV_HI = 21, 147  # the crystallised construct (2PR5)
W, H = 2400, 1500

# official AlphaFold pLDDT band colours
cmd.set_color("af_vhigh", [0x00 / 255, 0x53 / 255, 0xD6 / 255])
cmd.set_color("af_conf",  [0x65 / 255, 0xCB / 255, 0xF3 / 255])
cmd.set_color("af_low",   [0xFF / 255, 0xDB / 255, 0x13 / 255])
cmd.set_color("af_vlow",  [0xFF / 255, 0x7D / 255, 0x45 / 255])
cmd.set_color("site_red", [0.75, 0.22, 0.17])

cmd.load(AF, "full")
cmd.remove("solvent")
cmd.create("lov", f"full and resi {LOV_LO}-{LOV_HI}")
cmd.delete("full")
cmd.hide("everything")

cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("specular", 0.22)
cmd.set("ambient", 0.20)
cmd.set("direct", 0.58)
cmd.set("stick_radius", 0.20)
cmd.set("depth_cue", 0)
cmd.set("orthoscopic", 1)
cmd.set("cartoon_side_chain_helper", 1)


def colour_by_plddt(sel="lov"):
    cmd.color("af_vlow",  f"{sel} and b < 50")
    cmd.color("af_low",   f"{sel} and b > 50 and b < 70")
    cmd.color("af_conf",  f"{sel} and b > 70 and b < 90")
    cmd.color("af_vhigh", f"{sel} and b > 90")


# =====================================================================
# 1. LOV domain, cartoon, coloured by pLDDT, mutation as sticks
# =====================================================================
cmd.show("cartoon", "lov")
colour_by_plddt()
cmd.show("sticks", f"lov and resi {POS} and sidechain")
cmd.show("spheres", f"lov and resi {POS} and sidechain")
cmd.color("site_red", f"lov and resi {POS}")
cmd.set("stick_radius", 0.34, f"lov and resi {POS}")
cmd.set("sphere_scale", 0.34, f"lov and resi {POS} and sidechain")

cmd.orient("lov")
cmd.turn("y", 20)
cmd.turn("x", -8)
cmd.zoom("lov", 1.0)
cmd.ray(W, H)
cmd.png(os.path.join(OUT, "af_lov.png"), dpi=300)
print("wrote af_lov.png", flush=True)

# =====================================================================
# 2. same view, context de-emphasised so the site reads immediately
# =====================================================================
cmd.color("grey80", "lov")
cmd.set("cartoon_transparency", 0.62, "lov")
cmd.set("cartoon_transparency", 0.0, f"lov and resi {POS-5}-{POS+5}")
colour_by_plddt(f"lov and resi {POS-5}-{POS+5}")
cmd.color("site_red", f"lov and resi {POS} and sidechain")
cmd.ray(W, H)
cmd.png(os.path.join(OUT, "af_lov_focus.png"), dpi=300)
print("wrote af_lov_focus.png", flush=True)

# =====================================================================
# 3. local environment — the packing around the mutated residue
# =====================================================================
cmd.hide("everything")
cmd.set("cartoon_side_chain_helper", 0)
cmd.set("cartoon_transparency", 0.80, "lov")
cmd.select("pocket", f"byres (lov within 6.5 of (lov and resi {POS} and sidechain))")
cmd.show("cartoon", "pocket")
cmd.color("grey85", "pocket")
cmd.show("sticks", "pocket and sidechain and not hydro")
cmd.set("stick_radius", 0.15, "pocket")
cmd.color("grey60", "pocket and elem C")
cmd.show("sticks", f"lov and resi {POS} and sidechain")
cmd.show("spheres", f"lov and resi {POS} and sidechain")
cmd.color("site_red", f"lov and resi {POS} and elem C")
cmd.set("stick_radius", 0.30, f"lov and resi {POS}")
cmd.set("sphere_scale", 0.28, f"lov and resi {POS} and sidechain")

cmd.orient(f"lov and resi {POS} and sidechain")
cmd.zoom("pocket", 0.6)
cmd.turn("y", 10)
cmd.ray(int(W*0.72), int(H*1.05))
cmd.png(os.path.join(OUT, "af_site.png"), dpi=300)
print("wrote af_site.png", flush=True)

plddt = []
cmd.iterate(f"lov and resi {POS} and name CA", "plddt.append(b)", space={"plddt": plddt})
n_lov = cmd.count_atoms("lov and name CA")
print(f"LOV domain rendered: {n_lov} residues ({LOV_LO}-{LOV_HI})", flush=True)
print(f"site pLDDT = {plddt[0]:.1f}", flush=True)
print("DONE", flush=True)
