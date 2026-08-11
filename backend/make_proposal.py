"""Generate the wet-lab access proposal PDF from the model-designed candidate panel."""
import json, datetime
from fpdf import FPDF
from fpdf.fonts import FontFace

panel = json.load(open("final_panel.json"))
WT = panel[0]["seq"]

INK=(0,0,0); MUT=(0,0,0); ACC=(0,0,0); GREY=(90,90,90); GOOD=(0,0,0)

class PDF(FPDF):
    def header(self):
        return
    def footer(self):
        self.set_y(-12); self.set_font("DejaVu","I",8); self.set_text_color(*GREY)
        self.cell(0,6,f"Page {self.page_no()}",align="C")

def H1(p,t):
    p.ln(2); p.set_font("DejaVu","B",12.5); p.set_text_color(*INK)
    p.cell(0,7,t,new_x="LMARGIN",new_y="NEXT"); p.ln(0.5)
def H2(p,t):
    p.set_font("DejaVu","B",10.5); p.set_text_color(0,0,0)
    p.cell(0,6,t,new_x="LMARGIN",new_y="NEXT")
def body(p,t):
    p.set_font("DejaVu","",10); p.set_text_color(*INK); p.multi_cell(0,5,t); p.ln(1)
def bullet(p,t):
    p.set_font("DejaVu","",10); p.set_text_color(*INK)
    x=p.get_x(); p.cell(5,5,"-"); p.multi_cell(0,5,t); p.set_x(x)

def labeled(p,label,value,lw=38,lab_color=MUT):
    x0=p.l_margin; y0=p.get_y(); avail=p.w-p.r_margin-(x0+lw)
    p.set_xy(x0,y0); p.set_font("DejaVu","B",9.5); p.set_text_color(*lab_color)
    p.multi_cell(lw,5,label,new_x="LMARGIN",new_y="NEXT"); yl=p.get_y()
    p.set_xy(x0+lw,y0); p.set_font("DejaVu","",9.5); p.set_text_color(*INK)
    p.multi_cell(avail,5,value,new_x="LMARGIN",new_y="NEXT"); yv=p.get_y()
    p.set_y(max(yl,yv)); p.ln(0.5)

pdf=PDF(); pdf.set_margins(16,16,16); pdf.set_auto_page_break(True,margin=16)
import matplotlib as _mpl, os as _os
_FD=_os.path.join(_os.path.dirname(_mpl.__file__),"mpl-data","fonts","ttf")
pdf.add_font("DejaVu","",_os.path.join(_FD,"DejaVuSerif.ttf"))
pdf.add_font("DejaVu","B",_os.path.join(_FD,"DejaVuSerif-Bold.ttf"))
pdf.add_font("DejaVu","I",_os.path.join(_FD,"DejaVuSerif-Italic.ttf"))
pdf.add_font("DejaVuMono","",_os.path.join(_FD,"DejaVuSansMono.ttf"))
pdf.add_page()

# ---- Title (centered, professional) ----
pdf.ln(2)
pdf.set_font("DejaVu","B",16); pdf.set_text_color(*INK)
pdf.multi_cell(0,7.5,"Experimental Validation of Machine-Learning-Designed\nThermostable IsPETase Variants",align="C")
pdf.ln(1)
pdf.set_font("DejaVu","",11.5); pdf.set_text_color(*INK)
pdf.cell(0,6,"A Proposal Requesting Wet-Lab Access",align="C",new_x="LMARGIN",new_y="NEXT")
pdf.ln(4)

# ---- Executive summary ----
H1(pdf,"1  Executive Summary")
body(pdf,
"We request access to a molecular-biology / protein-biochemistry wet lab to experimentally "
"validate five computationally designed variants of the plastic-degrading enzyme IsPETase. "
"Enzymatic recycling of PET plastic works best at high temperature, but wild-type IsPETase "
"falls apart near 48 °C, which limits its industrial use.")
body(pdf,
"What we did computationally, in brief: we built an artificial-intelligence tool that has "
"learned, from large public databases of laboratory measurements, how small changes to a "
"protein's sequence affect how much heat it can withstand. We gave this tool the IsPETase "
"sequence; it evaluated every possible single amino-acid change and ranked the ones most likely "
"to make the enzyme more heat-stable. From that ranked list we selected five promising designs.")
body(pdf,
"We now propose to produce these five proteins in the lab, alongside the original enzyme as a "
"control, and measure how much heat each one can take, directly checking whether the tool's "
"predictions hold. The experiment is small (six proteins), inexpensive, and uses standard "
"techniques; its outcome is the critical evidence needed before the tool is used to guide real "
"enzyme-engineering campaigns, for which we have secured downstream deployment interest.")

# ---- Background ----
H1(pdf,"2  Background and Significance")
body(pdf,
"Polyethylene terephthalate (PET) is among the most abundant plastics, and enzymatic recycling, "
"using enzymes to break PET into reusable monomers, is a leading route to a circular plastics "
"economy. PET degrades fastest at elevated temperature, where the polymer softens and becomes "
"accessible. The bottleneck is biological: naturally occurring PET hydrolases such as IsPETase "
"(from Piscinibacter / Ideonella sakaiensis, UniProt A0A0K8P6T7) lose their fold and activity "
"well below the useful temperature range. Engineering enzymes that remain stable and active "
"under heat is therefore the central problem, and improving melting temperature (Tm) is the "
"established, measurable lever for doing so.")
body(pdf,
"Our broader mission is to build a condition-aware enzyme-design pipeline: a system that "
"engineers protein variants able to withstand the many environmental hurdles that prevent "
"enzymes from functioning in real-world settings. Industrial and environmental conditions are "
"rarely ideal: a useful enzyme must tolerate not only high temperature but also extremes of pH, "
"calcium-ion concentration, ionic strength, and related stresses, often at the same time. Our "
"primary focus is therefore to design optimized proteins that remain stable and active across "
"this full range of conditions, rather than under a single idealized setting. The present "
"experiment addresses the first and most fundamental of these axes: it is designed primarily to "
"test the pipeline's thermostability (temperature) predictions, establishing the validation "
"foundation on which the other conditions will build.")
body(pdf,
"Testing mutations experimentally is slow and costly, so exhaustive screening is infeasible. "
"Our pipeline addresses this by ranking candidate mutations computationally, so only a short, "
"high-confidence shortlist reaches the bench. This proposal is that final, essential step: "
"confirming in the lab that the model's top predictions actually stabilize the enzyme.")

# ---- Method ----
H1(pdf,"3  How the Candidates Were Generated")
body(pdf,
"The design pipeline pairs two models built on ESM-2 protein-language-model embeddings: a "
"screening model that predicts an enzyme's Tm from sequence, and a mutation model that predicts "
"the stability change (ΔΔG) of a point mutation. The mutation model was trained on a "
"leakage-audited dataset of over one million stability measurements from seven public sources, "
"with independent benchmarks filtered for sequence-homology overlap so reported accuracy "
"reflects generalization to novel enzymes.")
body(pdf,
"To design the candidates, we scored all ~4,900 allowed single-point mutations of wild-type "
"IsPETase at a 50 °C target temperature. We excluded the signal peptide (residues 1-27), the "
"catalytic triad (Ser160/Asp206/His237), and the structural disulfide cysteines, so that "
"predicted-stabilizing mutations cannot compromise catalysis or the fold. From the ranked "
"output we selected five designs that span distinct positions and mechanisms, plus a combined "
"variant that stacks the three most compatible substitutions.")
pdf.set_font("DejaVu","I",9); pdf.set_text_color(*GREY)
pdf.multi_cell(0,5,"Note on effect sizes: the mutation model is a sequence-based baseline; its "
"predicted per-mutation shifts are modest (~0.5-2 °C) and are reported here as predictions to be "
"tested, not as established facts. Confirming whether these rankings hold, and whether stacking "
"is additive, is precisely the purpose of the requested experiment.")
pdf.ln(1)

# ---- Candidate panel table ----
pdf.add_page()
H1(pdf,"4  Candidate Proteins to Test")
body(pdf,"Table 1 lists the six proteins we propose to test: the wild-type control and five "
"model-designed variants. Positions use full-length UniProt numbering (A0A0K8P6T7). Full-length "
"amino-acid sequences for gene synthesis are provided in the Appendix.")
# caption above table (professional convention)
pdf.set_font("DejaVu","",8.5); pdf.set_text_color(*INK)
pdf.multi_cell(0,4.6,"Table 1. Candidate proteins for wet-lab validation. Predicted values are "
"model outputs at 50 °C; a negative ΔΔG indicates stabilization. IsPETase-3M stacks Y219E, "
"K177I, and R123I, and its predicted ΔTm is the additive estimate.")
pdf.ln(1.5)
# booktabs-style table: horizontal rules only, no fill, centered
hdr=["Construct","Mutation(s)","Type","Pred. ΔTm (°C)","Pred. ΔΔG"]
w=[38,44,30,34,26]; tw=sum(w); x0=(pdf.w-tw)/2.0
def _rule(th):
    yy=pdf.get_y(); pdf.set_draw_color(0,0,0); pdf.set_line_width(th); pdf.line(x0,yy,x0+tw,yy)
_rule(0.5); pdf.ln(1.6)
pdf.set_x(x0); pdf.set_font("DejaVu","B",9); pdf.set_text_color(*INK)
for h,ww in zip(hdr,w): pdf.cell(ww,6,h,align="C")
pdf.ln(6.5); _rule(0.25); pdf.ln(1.6)
pdf.set_font("DejaVu","",9)
for p in panel:
    pdf.set_x(x0)
    muts="+".join(p["muts"]) if p["muts"] else "none (wild-type)"
    row=[p["name"],muts,p["kind"],f'{p["pred_dTm"]:+.2f}',f'{p["pred_ddg"]:+.3f}']
    for v,ww in zip(row,w): pdf.cell(ww,6,v,align="C")
    pdf.ln(6)
pdf.ln(0.5); _rule(0.5); pdf.ln(3)
pdf.ln(1)

# ---- Experimental plan ----
H1(pdf,"5  Experimental Plan (What We Will Do in the Lab)")
H2(pdf,"5.1  Protein production")
bullet(pdf,"Synthesize codon-optimized genes for all six constructs (mature domain, residues "
"28-290) with a C-terminal His6 tag, cloned into a pET-based expression vector.")
bullet(pdf,"Express in E. coli BL21(DE3): IPTG induction, ~18 °C overnight, for soluble protein.")
bullet(pdf,"Purify by Ni-NTA affinity chromatography; assess purity by SDS-PAGE; buffer-exchange "
"into assay buffer. Expected yield is sufficient for all assays from standard shake-flask cultures.")
H2(pdf,"5.2  Thermostability measurement (primary readout)")
bullet(pdf,"Nano-DSF / thermal-shift assay: ramp each purified variant from 25-95 °C and record its "
"melting temperature (Tm). This directly measures each variant's Tm and yields the experimental "
"ΔTm versus wild-type, the quantity our model predicts.")
bullet(pdf,"Thermal challenge at 50 °C: incubate each variant at 50 °C for 30-60 min, cool, then "
"measure residual activity. Wild-type (Tm ~48 °C) is expected to lose activity; stabilized variants "
"should retain it.")
H2(pdf,"5.3  Activity assay")
bullet(pdf,"Esterase activity on a para-nitrophenyl ester substrate (colorimetric, plate-reader) "
"as a fast activity proxy, and/or PET-film/powder hydrolysis quantified by release of degradation "
"products (absorbance or HPLC).")
H2(pdf,"5.4  Success criteria")
bullet(pdf,"Primary: the measured Tm ranking of the variants correlates with the model's predicted "
"ranking; the stacked variant (IsPETase-3M) shows the largest Tm increase.")
bullet(pdf,"Any variant with Tm reproducibly above wild-type validates the pipeline; nano-DSF "
"resolves sub-degree shifts, so even the modest predicted effects are testable.")

# ---- Resources ----
pdf.add_page()
H1(pdf,"6  What We Request From the Host Lab")
rows=[
 ("Bench access","BSL-1 space for routine E. coli work over ~6-8 weeks."),
 ("Molecular biology","Competent E. coli BL21(DE3), expression vector, standard cloning reagents, "
   "media, IPTG, antibiotics. (Genes can be outsourced as synthetic fragments.)"),
 ("Protein purification","Ni-NTA resin / gravity columns or an FPLC/AKTA system; SDS-PAGE apparatus."),
 ("Key instrument","A nano-DSF (e.g. Prometheus) or a qPCR machine for dye-based thermal-shift "
   "assays, the central measurement."),
 ("Plate reader","UV/Vis plate reader for the para-nitrophenyl activity assay."),
 ("Optional","HPLC access for PET degradation-product quantification; PET film/powder substrate."),
 ("Consumables","Plates, cuvettes/capillaries, buffers, centrifuge tubes."),
]
pdf.set_font("DejaVu","",9.5); pdf.set_text_color(*INK)
_bold=FontFace(emphasis="BOLD")
with pdf.table(col_widths=(30,70), text_align="LEFT", line_height=5.2,
               borders_layout="HORIZONTAL_LINES", first_row_as_headings=True,
               headings_style=_bold) as _tbl:
    _r=_tbl.row(); _r.cell("Item"); _r.cell("What we request")
    for k,v in rows:
        _r=_tbl.row(); _r.cell(k, style=_bold); _r.cell(v)
pdf.ln(2)
body(pdf,"We provide the enzyme designs, sequences, analysis, and hands-on labor; we ask the host "
"lab for supervised access, the instruments above, and routine consumables. We are glad to work "
"under the host's safety and IP arrangements.")

# ---- Timeline ----
H1(pdf,"7  Timeline (approximately 6-8 weeks)")
tl=[("Weeks 1-2","Gene synthesis and cloning of the six constructs."),
    ("Weeks 3-4","Expression and Ni-NTA purification; purity check."),
    ("Weeks 5-6","Nano-DSF Tm measurement and 50 °C thermal-challenge / activity assays."),
    ("Weeks 7-8","Data analysis; compare measured vs. predicted Tm; report.")]
for k,v in tl:
    labeled(pdf,k,v,lw=24,lab_color=ACC)
pdf.ln(1)

# ---- Impact ----
H1(pdf,"8  Expected Impact")
body(pdf,"A positive result, measured Tm increases consistent with the predicted ranking, "
"validates that our leakage-audited ML pipeline can propose genuinely stabilizing mutations for "
"a real, industrially relevant enzyme. This is the evidence required before applying the pipeline "
"at scale to engineer heat-tolerant PET-degrading enzymes for deployment. Because the experiment "
"is small, standard, and low-cost, it offers an unusually high ratio of scientific value to "
"bench effort, and provides an excellent, self-contained project for a student collaborator.")

# ---- References ----
H1(pdf,"9  Key References")
pdf.set_font("DejaVu","",8.5); pdf.set_text_color(*INK)
for r in [
 "Yoshida S. et al. (2016). A bacterium that degrades and assimilates poly(ethylene "
 "terephthalate). Science 351:1196-1199.",
 "Son H.F. et al. (2019). Rational protein engineering of thermo-stable PETase. ACS Catal. 9:3519-3526.",
 "Lin Z. et al. (2023). ESM-2: Evolutionary-scale prediction of protein structure. Science 379:1123.",
 "Tsuboyama K. et al. (2023). Mega-scale experimental analysis of protein folding stability. "
 "Nature 620:434-444.",
 "UniProt A0A0K8P6T7 (PETH_PISS1), Poly(ethylene terephthalate) hydrolase.",
]:
    pdf.multi_cell(0,4.6,"-  "+r); pdf.ln(0.3)

# ---- Appendix: sequences ----
pdf.add_page()
H1(pdf,"Appendix: Protein and DNA Sequences (for gene synthesis)")
body(pdf,"Full-length constructs (290 aa). For expression, the mature domain (residues 28-290) is "
"used with a C-terminal His6 tag. Mutated positions are relative to UniProt A0A0K8P6T7. DNA is a "
"codon-optimized E. coli coding sequence (873 bp incl. stop); a commercial optimizer may further "
"tune GC content and remove restriction sites before synthesis.")

def monoblock(seq, width=60, step=3):
    pdf.set_font("DejaVuMono","",7.0); pdf.set_text_color(*INK)
    for i in range(0, len(seq), width):
        pdf.cell(0,3.5,f"{i+1:>4}  "+seq[i:i+width],new_x="LMARGIN",new_y="NEXT")
def caption(t):
    pdf.set_font("DejaVu","I",8); pdf.set_text_color(*GREY)
    pdf.cell(0,4,t,new_x="LMARGIN",new_y="NEXT"); pdf.ln(1.2)

for p in panel:
    if pdf.get_y() > pdf.h - 60: pdf.add_page()
    pdf.set_font("DejaVu","B",9.5); pdf.set_text_color(*MUT)
    lbl=p["name"]+("  ("+"+".join(p["muts"])+")" if p["muts"] else "  (wild-type)")
    pdf.multi_cell(0,5,lbl,new_x="LMARGIN",new_y="NEXT"); pdf.ln(0.8)
    caption("Protein (290 aa):")
    monoblock(p["seq"], 60)
    pdf.ln(3)
    caption("DNA (E. coli-optimized coding sequence, 873 bp, includes stop codon):")
    monoblock(p["dna"], 60)
    pdf.ln(5)

out="/Users/admin/Documents/PET - Lab/IsPETase_WetLab_Proposal.pdf"
pdf.output(out)
print("WROTE",out)
