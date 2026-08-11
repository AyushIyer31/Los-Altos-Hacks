"""Whole-manuscript preparation guide (Scientific Reports): what each section
needs, its owner, format rules, and open confirmations. Temperature-focused
two-stage study; uses only confirmed study values; placeholders for the rest."""
import os
from fpdf import FPDF
from fpdf.fonts import FontFace
import matplotlib as _mpl

INK=(0,0,0); GREY=(90,90,90)
_FD=os.path.join(os.path.dirname(_mpl.__file__),"mpl-data","fonts","ttf")

class PDF(FPDF):
    def header(self): return
    def footer(self):
        self.set_y(-12); self.set_font("S","I",8); self.set_text_color(*GREY)
        self.cell(0,6,f"Page {self.page_no()}",align="C")

pdf=PDF(); pdf.set_margins(16,16,16); pdf.set_auto_page_break(True,margin=16)
pdf.add_font("S","",os.path.join(_FD,"DejaVuSerif.ttf"))
pdf.add_font("S","B",os.path.join(_FD,"DejaVuSerif-Bold.ttf"))
pdf.add_font("S","I",os.path.join(_FD,"DejaVuSerif-Italic.ttf"))
pdf.add_page()
BOLD=FontFace(emphasis="BOLD")
NX=dict(new_x="LMARGIN",new_y="NEXT")

def H1(t):
    pdf.ln(2.5); pdf.set_font("S","B",13); pdf.set_text_color(*INK)
    pdf.multi_cell(0,7,t,**NX); pdf.ln(0.5)
def SEC(t):
    if pdf.get_y()>pdf.h-45: pdf.add_page()
    pdf.ln(1.8); pdf.set_font("S","B",10.6); pdf.set_text_color(*INK)
    pdf.multi_cell(0,5.4,t,**NX); pdf.ln(0.3)
def lab(t):
    pdf.set_font("S","I",8.8); pdf.set_text_color(*GREY); pdf.multi_cell(0,4.2,t,**NX)
def P(t):
    pdf.set_font("S","",9.6); pdf.set_text_color(*INK); pdf.multi_cell(0,4.6,t,**NX); pdf.ln(1)
def b(t,q=False):
    pdf.set_font("S","B" if q else "",9.2 if q else 9.4); pdf.set_text_color(*INK)
    x=pdf.get_x(); pdf.cell(4,4.4,"-"); pdf.multi_cell(0,4.4,t,**NX); pdf.set_x(x)
def block(title, reqs=None, info=None, opens=None):
    SEC(title)
    if reqs:
        lab("Format requirements")
        for r in reqs: b(r)
    if info:
        lab("Information needed")
        for r in info: b(r)
    if opens:
        lab("Open confirmations")
        for r in opens: b(r, q=True)
def table(rows, widths, aligns="LEFT", layout="HORIZONTAL_LINES", fs=8):
    pdf.set_font("S","",fs); pdf.set_text_color(*INK)
    with pdf.table(col_widths=widths, text_align=aligns, line_height=4.6,
                   borders_layout=layout, first_row_as_headings=True,
                   headings_style=BOLD) as t:
        for r in rows:
            row=t.row()
            for c in r: row.cell(str(c))

# ================= TITLE =================
pdf.ln(1); pdf.set_font("S","B",16); pdf.set_text_color(*INK)
pdf.multi_cell(0,7,"Manuscript Preparation Guide",align="C",**NX)
pdf.set_font("S","",11); pdf.set_x(pdf.l_margin)
pdf.cell(0,6,"Section-by-section: owner, requirements and information needed",align="C",**NX)
pdf.set_font("S","I",9.3); pdf.set_text_color(*GREY); pdf.set_x(pdf.l_margin)
pdf.cell(0,5,"Target: Nature Scientific Reports   -   temperature-focused two-stage study",align="C",**NX)
pdf.ln(3)
P("This guide states what each manuscript section requires, who owns it (AB = Abhinav Iyer, "
  "JA = James Ponzio, AY = Ayush Iyer), the Scientific Reports rules that apply, and the items still "
  "to be confirmed. It uses only values confirmed in the study result files; everything unverified is "
  "shown as a bold [AUTHOR TO CONFIRM: ...] item.")

# ================= SCOPE =================
H1("Scope control (read first)")
b("Treat this as a temperature-focused study. The pH-optimum model has no confirmed training run, so "
  "remove all pH results from the Abstract, Results, Methods, Discussion, tables, figures, Data "
  "Availability and Supplementary Information.")
b("EpHod may be mentioned only as possible future work, not as a dataset analysed in the completed study.")
b("Do not describe the training data as PETase-specific; it contains broad protein datasets. Use "
  "phrasing such as 'a general protein-stability framework evaluated for its potential application to "
  "PET-degrading enzyme engineering.'")
b("Do not claim experimental PET degradation, wet-laboratory validation, industrial deployment, "
  "environmental impact or joint temperature-and-pH optimization.")

# ============ VERSION CONTROL ============
H1("Study-version control (do not mix historical and current results)")
P("An older PET-Lab study used 19,071 mutations and different metrics. The current study uses the "
  "1,001,888-row assembled table with two stages. Search the whole manuscript for the older values "
  "below and remove any that were carried over. The current two-stage results (see the Display items section of this guide) are the controlling source for the Results section.")
table([
 ["Metric / statement","Older value","Current value","Required action"],
 ["Training size","19,071 mutations","1,001,888 rows (508,693 ΔΔG for Stage 2)","Report current only"],
 ["Cross-validation accuracy","~79.7%","Not applicable (independent benchmarks used)","Remove if present"],
 ["Pearson correlation","~0.764 (CV)","0.390 (S669 ensemble, independent)","Do not carry old value"],
 ["MAE","~0.92 kcal/mol","Not reported (RMSE 1.509 kcal/mol ensemble)","Do not carry old value"],
 ["S669 accuracy","~70.6-73.2%","Threshold-dependent (Table 3)","Do not carry old value"],
], widths=(26,24,32,26), fs=7.6)
b("[AUTHOR TO CONFIRM: whether any older 19,071-mutation result is intentionally included as a separate "
  "analysis; if not, remove every instance.]", q=True)

# ============ SR COMPLIANCE ============
H1("Scientific Reports compliance checklist (applies across the manuscript)")
for s in [
 "Title preferably <= 20 words. Abstract <= 200 words, unstructured, no references, no subheadings.",
 "<= 6 keywords. Main text preferably ~4,500 words (excluding Abstract, Methods, References, legends).",
 "<= 8 total main display items (figures + tables combined). Current plan: Tables 1-3 + Figure 1 = 4.",
 "Title page: full author names; superscript affiliations with country for each; corresponding author "
 "marked with an asterisk; corresponding-author email.",
 "Methods must allow independent reproduction. Data Availability is mandatory; a separate Code "
 "Availability section is included because custom code is central.",
 "References in numerical Nature style, cited sequentially in square brackets; one source per number.",
 "Author Contributions must list every author; a financial and non-financial competing-interests "
 "statement is required (placed under Additional Information).",
 "Supplementary Information is a separate file. Main tables editable (not images). No footnotes, "
 "graphical abstract, schemes, text boxes or appendices in the main manuscript.",
]:
    b(s)
lab("Recommended final order: Title page; Abstract; Keywords; Introduction; Results; Discussion; "
    "Methods; Data Availability; Code Availability; References; Acknowledgements; Author Contributions; "
    "Additional Information (Competing interests); Figure Legends; Main Tables. Supplementary Information "
    "is a separate file.")

# ================= SECTIONS =================
pdf.add_page()
H1("Section-by-section requirements")

block("1. Title page  [AY / title = AB]",
 reqs=["Title preferably <= 20 words; no undefined abbreviations.",
       "Full author names; superscript-linked affiliations; country for each affiliation.",
       "Corresponding author marked with an asterisk plus email."],
 info=["Final title (<= 20 words).",
       "Authors: Ayush Iyer, James Ponzio, Abhinav Iyer.",
       "Corresponding author: Ayush Iyer, iyer.ayush31@gmail.com."],
 opens=["[AUTHOR TO CONFIRM AFFILIATION - institution and country] for Ayush Iyer.",
        "[AUTHOR TO CONFIRM AFFILIATION - institution and country] for James Ponzio.",
        "[AUTHOR TO CONFIRM AFFILIATION - institution and country] for Abhinav Iyer.",
        "[AUTHOR TO CONFIRM: corresponding-author email before submission.]",
        "ORCID iDs entered via the submission system when requested (do not invent)."])

block("2. Abstract  [AB]",
 reqs=["<= 200 words, unstructured, no citations, no subheadings, no undefined abbreviations."],
 info=["Content order: problem; gap; objective; design; Stage 1 independent result; Stage 2 independent "
       "result; interpretation; scope limitation.",
       "Assembled 1,001,888 records from seven public sources; measurement types routed to different "
       "stages (Stage 1 = whole-protein Tm; Stage 2 = mutation-level ΔΔG).",
       "Leakage auditing used exact and sequence-homology checks.",
       "Stage 1 (BRENDA): ROC AUC 0.732 (95% CI 0.708-0.755).",
       "Stage 2 (S669): ROC AUC 0.669 (95% CI 0.62-0.72); ensemble Pearson 0.390 (95% CI 0.32-0.46).",
       "Prioritize independent evaluation over internal validation; end with a measured interpretation."],
 opens=["[AUTHOR TO CONFIRM: final word count <= 200; internal validation not presented as independent testing.]",
        "Do not describe ROC AUC as precision-recall AUC; do not claim industrial readiness."])

block("3. Keywords  [AB]",
 reqs=["<= 6 keywords or key phrases; no promotional terms."],
 info=["Assess from: protein stability; machine learning; enzyme engineering; thermostability; "
       "mutation-effect prediction; melting temperature; ΔΔG prediction; PET-degrading enzymes.",
       "Choose no more than six; avoid repeating every title word."])

block("4. Introduction  [AB]",
 reqs=["Approximately 4-6 focused paragraphs; every external factual claim carries a citation.",
       "Do not report detailed results here."],
 info=["P1 - plastic/PET context (verify every statistic from an authoritative original source, e.g. "
       "OECD/UNEP/peer-reviewed; do not mix tonnes/tons; state whether values are production, waste, "
       "municipal waste or PET).",
       "P2 - enzymatic PET degradation (PETase, cutinases); thermostability is not the sole determinant "
       "of catalytic performance (activity, binding, conditions, crystallinity, pH also matter).",
       "P3 - computational stability prediction: define Tm and mutation-level ΔΔG; distinguish "
       "whole-protein screening from mutation-level optimization; they are related but not interchangeable.",
       "P4 - leakage problem: exact and sequence-homology leakage can inflate benchmarks; do not claim "
       "all prior models contain leakage.",
       "Final - gap and objective: whole-protein selection and mutation-level optimization are usually "
       "separate; this study evaluates a two-stage screen-then-optimize framework on leakage-audited "
       "independent BRENDA and S669 benchmarks."],
 opens=["[AUTHOR TO CONFIRM: source and exact figures for every plastic-waste statistic.]",
        "[AUTHOR TO CONFIRM: no claims of experimental/wet-lab/field/clinical use unless documented.]"])

block("5. Results  [JA]",
 reqs=["Connected objective prose; four subheadings (dataset/leakage; Stage 1; Stage 2; comparison).",
       "Separate internal validation from independent testing; regression from classification."],
 info=["Cite Table 1 in the dataset subsection, Table 2 in the Stage 1 subsection, and Table 3 in the "
       "Stage 2 subsection; the labeled tables are in the Display items section at the end of this guide "
       "the labeled tables are in the Display items section of this guide.",
       "Dataset/leakage subsection must state: 1,001,888 rows from seven sources; 29,654 Tm records "
       "to Stage 1 and 508,693 ΔΔG records to Stage 2; abundance and ΔTm records not used; exact and "
       "sequence-homology leakage auditing; approximately 132,000 records removed (131,479 by homology).",
       "Stage 1 subsection must state: internal held-out RMSE approximately 6.5 °C and validation "
       "Pearson approximately 0.80 (development only); independent BRENDA ROC AUC 0.732 (95% CI "
       "0.708-0.755); regression Pearson 0.54 (95% CI 0.50-0.57) and 0.60 on the true-Tm subset; the "
       "sample flow (412 removed to 2,034; 471 ambiguous excluded to 1,563; 979 positive, 584 "
       "negative); and the threshold trade-off (precision rises, recall falls with stricter cutoffs).",
       "Stage 2 subsection must state: 669 mutations (168 stabilizing, 501 non-stabilizing); ensemble "
       "Pearson 0.390 (95% CI 0.32-0.46) and ROC AUC 0.669 (95% CI 0.62-0.72) from 2,000 bootstrap "
       "resamples; that Extra Trees gave the strongest single-model regression metrics; and the "
       "high-precision, low-recall behaviour at the strict threshold.",
       "Comparison subsection must state: Stage 1 had a higher ROC AUC than Stage 2; the stages used "
       "different datasets, labels and sample sizes; it is not a controlled head-to-head comparison "
       "and no statistical significance test was performed.",
       "Report exact values; interpret threshold trade-offs neutrally; note Extra Trees (not the "
       "ensemble) gave the strongest listed Stage 2 regression metrics; do not hide Stage 2 low recall.",
       "INSERT Figure 1 here (precision as a function of decision threshold; file "
       "paper_figures/precision_vs_threshold.pdf): cite it in the Stage 1 and Stage 2 subsections, and "
       "present it alongside Tables 2-3, which carry the corresponding recall, F1, accuracy and ROC AUC "
       "so precision is not read as overall reliability."],
 opens=["[AUTHOR TO CONFIRM: original BRENDA size (inferred 2,446); positive/negative/ambiguous class "
        "definitions; true-Tm subset n (for r = 0.60); ΔΔG sign convention; how each Stage 2 decision "
        "threshold is applied; that thresholds were not selected on the benchmarks.]"])

block("6. Discussion  [AB]",
 reqs=["Interpretation, not repetition of Results; ~7 paragraphs; no promotional language."],
 info=["Main findings; meaning of Stage 1 (0.732) and its precision-recall trade-off; meaning of Stage 2 "
       "(0.390 / 0.669) as moderate performance with high-precision/low-recall behaviour; leakage-audit "
       "significance and residual homology risk; fair comparison with prior Tm/ΔΔG models only when "
       "benchmark, sign convention, split and metric are comparable; a full limitations paragraph; "
       "measured future work and conclusion.",
       "State that benchmark precision may not transfer unchanged to PET-degrading enzymes; do not equate "
       "0.93 precision with 93% experimental success."],
 opens=["[AUTHOR TO CONFIRM: which prior models are cited for comparison and that the comparison is fair "
        "(same benchmark/metric/independence).]"])

block("7. Methods  [AY]  (largest information gap)",
 reqs=["Sufficient detail for independent reproduction; no main-text word limit; use clear subsections."],
 info=["Study design (two stages use different targets/subsets; the full table did not train both models).",
       "Source datasets: name, primary publication, repository, version/access date, measurement type, "
       "row count, inclusion/exclusion criteria, license, DOI/accession (Domainome; Tsuboyama mega-scale; "
       "Tsuboyama doubles; Meltome Atlas; ThermoMutDB; FireProtDB; ProDDG/S2648; BRENDA; S669; UniProt).",
       "Harmonization; measurement typing; duplicate/replicate handling; sequence resolution from UniProt "
       "(release, canonical vs isoform, failed mappings).",
       "Leakage auditing operationalized: exact-key and mutation-level overlap; MMseqs2 version; identity "
       "threshold; coverage threshold; e-value; removal logic; counts removed/remaining at each step.",
       "Stage 1 and Stage 2 target construction, including BRENDA class definitions and the ΔΔG "
       "mathematical definition; how the °C and kcal/mol thresholds were applied.",
       "Feature construction verified from current code (do not carry features from the 19,071-mutation "
       "model); if ESM-2 embeddings used, give model/version/layer/pooling/dimensionality/mutation "
       "representation.",
       "Per-model software, package versions, hyperparameters, class weighting, early stopping, feature "
       "scaling, random seeds; ensemble type (averaged/stacked/voted) and how weights were chosen.",
       "Validation/test design (splits, grouping, folds, seed); metric definitions; bootstrap details "
       "(2,000 resamples, resampling unit, CI type, seed); software/environment/hardware.",
       "Ethics: a brief statement only if accurate (computational study, public data). LLM-use "
       "disclosure per Nature Portfolio policy if an LLM was used substantively."],
 opens=["[AUTHOR TO CONFIRM: every dataset version, license and DOI; MMseqs2 version/thresholds/commands "
        "from code; the exact current feature set; each model's package version, hyperparameters and "
        "seed; ensemble type; split/grouping/seed; whether thresholds were selected before seeing "
        "benchmark results; software and library versions; whether/how an LLM was used.]"])

block("8. Data Availability  [JA]",
 reqs=["Mandatory; placed after Methods, before Code Availability.",
       "Do not write 'DOI assigned on publication' (the repository issues the DOI)."],
 info=["Statement: the processed training table, leakage-audited benchmark datasets, saved predictions, "
       "split identifiers, leakage-audit outputs and trained model artifacts generated in this study are "
       "available through Zenodo (https://doi.org/10.5281/zenodo.21257369); source datasets are cited in "
       "Methods and References.",
       "Deposit record: 'PET training set' (Iyer, Ayush), one file, ~2.5 GB."],
 opens=["[AUTHOR TO CONFIRM: the single deposited file contains all listed artifacts (processed table, "
        "benchmark datasets, saved predictions, split identifiers, leakage-audit outputs, trained "
        "models); add a data dictionary/README inside the deposit.]",
        "[AUTHOR TO CONFIRM ACCESS: the record files currently require login (restricted). State whether "
        "access is open or restricted; if restricted, provide the reviewer/reader access mechanism, or "
        "make the files open before submission.]",
        "[AUTHOR TO CONFIRM LICENSE: the deposit is currently licensed GPL-3.0 (a software license); "
        "confirm this is intended for the data and model, or apply a data license such as CC BY 4.0.]",
        "[AUTHOR TO VERIFY: license and redistribution rights for each source dataset; state restrictions.]",
        "[AUTHOR TO CONFIRM: remove EpHod unless the pH analysis is included.]"])

block("9. Code Availability  [JA, technical verification by AY]",
 reqs=["Separate section after Data Availability, before References.",
       "GitHub alone is not a permanent versioned archive."],
 info=["Statement: source code for data assembly, leakage auditing, model training, evaluation, threshold "
       "analysis, bootstrap confidence-interval estimation and figure generation is available at "
       "https://github.com/AyushIyer31/PET-Lab and archived on Zenodo at "
       "https://doi.org/10.5281/zenodo.21519961 (release v1.0.0).",
       "Scripts to verify in the repository: ensemble_eval.py, s669_sweep.py, predict_and_sweep.py, bootstrap_ci.py."],
 opens=["[AUTHOR TO CONFIRM: the archived release v1.0.0 (commit 497c4f5) is the code version that "
        "produced the reported results; if not, tag and archive the correct commit. 10.5281/zenodo.21519961 "
        "is the all-versions (concept) DOI; the version-specific DOI for v1.0.0 is on the Zenodo record if "
        "you prefer to cite the exact version.]",
        "[AUTHOR TO CONFIRM (AY): each named script exists, matches the reported analysis, has no broken "
        "paths or exposed credentials, and runs with documented dependencies.]",
        "[AUTHOR TO CONFIRM: README, license, environment/requirements file, exact package versions, "
        "random seeds, hyperparameters, reproduction commands, figure- and table-generation scripts, "
        "saved predictions and the exact release/commit are present.]"])

block("10. References  [AB]",
 reqs=["Numerical Nature style; sequential; square-bracket in-text; one source per number; every citation "
       "matched and every reference cited; no references in the Abstract."],
 info=["Verify each reference: authors, title, venue, year, volume, pages/article number, DOI, and that "
       "the source supports the attached claim. Use primary sources (dataset papers, PETase discovery/"
       "engineering studies, plastic-waste primary reports).",
       "Include formal dataset citations for all sources and for the Zenodo data and code deposits."],
 opens=["[AUTHOR TO CONFIRM: mark any unverifiable citation as CITATION NOT VERIFIED - REMOVE OR REPLACE; "
        "do not fabricate DOIs.]"])

block("11. Acknowledgements  [AB]",
 reqs=["Brief; no effusive language; no reviewers/editors; no competing interests; not for authors."],
 info=["May include non-author contributors, technical/computational-resource support and verified funding."],
 opens=["[AUTHOR TO CONFIRM: whether Santa Clara University / UC Santa Cruz computing or mentorship support "
        "is to be acknowledged, the exact program/allocation, named individuals and their consent; do not "
        "add institutions automatically or imply endorsement.]"])

block("12. Author Contributions  [AY]",
 reqs=["Every author listed; use CRediT categories; do not infer roles from section assignments.",
       "Use distinct forms for shared initials (Ayush Iyer, Abhinav Iyer, J.P.)."],
 info=["Provisional: 'Ayush Iyer contributed to [CONFIRM ROLES]. James Ponzio contributed to [CONFIRM "
       "ROLES]. Abhinav Iyer contributed to [CONFIRM ROLES]. All authors reviewed and approved the final "
       "manuscript.'"],
 opens=["[AUTHOR TO CONFIRM: actual CRediT roles per author from project/code/analysis records; author "
        "order; that all authors approve the submission and contribution statement.]"])

block("13. Additional Information  [JA]  ->  Competing interests  [AB drafts]",
 reqs=["Heading 'Additional Information' with a single 'Competing interests' subsection.",
       "No publisher boilerplate, reprints, publisher's note, peer-review line, correspondence sentence, "
       "ORCID list or SI URL. Corresponding author belongs on the title page, not here."],
 info=["When all authors confirm none: 'The authors declare no competing interests.' Otherwise, an "
       "explicit author-linked declaration matching the submission system."],
 opens=["[ALL AUTHORS TO CONFIRM: financial and non-financial competing interests, including any interest "
        "connected with PET-Lab. Do not assume 'none'.]"])

block("14. Figure Legends  [AY]",
 reqs=["Complete legend per figure (< 350 words), understandable without the main text; no Results/"
       "Discussion interpretation.",
       "Define: title; each panel; dataset; sample size; model; positive class; axes; symbols/lines/"
       "shading; error bands; resamples; thresholds; abbreviations; units; exclusions."],
 info=["Figure 1 (current plan): precision as a function of the decision threshold, panel a Stage 1 "
       "(BRENDA, n = 1,563) and panel b Stage 2 (S669, n = 669); file paper_figures/"
       "precision_vs_threshold.pdf. The figure and its draft caption are in the Display items section at "
       "the end of this guide (cite as Fig. 1). Present alongside Tables 2-3, which carry recall and ROC "
       "AUC."],
 opens=["[AUTHOR TO CONFIRM: positive class per panel (BRENDA stable; S669 stabilizing); that recall is "
        "reported in the text and Tables 2-3 so precision is not read as overall reliability; whether the "
        "final figure is regenerated from saved per-point predictions.]"])

block("15. Main Tables  [AY]",
 reqs=["Editable (not images); each cited in text; consistent decimals; units in headings; abbreviations "
       "and positive classes defined in notes; sample sizes stated; validation vs independent and "
       "regression vs classification distinguished; do not label 'best' without a stated criterion."],
 info=["Table 1 - dataset composition and measurement-type allocation (Panels A, B).",
       "Table 2 - Stage 1 BRENDA performance (Panel A threshold sweep; Panel B per-model at 60 °C).",
       "Table 3 - Stage 2 S669 performance (Panel A regression; Panel B threshold sweep; Panel C strict "
       "threshold). The labeled tables are shown in the Display items section of this guide."],
 opens=["[AUTHOR TO CONFIRM: table arithmetic and that figures + tables total <= 8 (currently 4).]"])

block("16. Supplementary Information  [AY]  (separate file)",
 reqs=["Separate submission file; first page shows exact title, exact author list and 'Supplementary "
       "Information'; label items Supplementary Table S1, Supplementary Figure S1, Supplementary Methods; "
       "cite each in the main text; supplementary legends live in the SI file."],
 info=["Candidate content: full data-cleaning and per-source inclusion criteria; sequence-resolution "
       "workflow; detailed MMseqs2 commands and leakage-audit thresholds; feature definitions; full "
       "per-model tables if they exceed the main-text limit; hyperparameter tables; sensitivity analyses.",
       "Do not move information essential to the main claims entirely into Supplementary Information."],
 opens=["[AUTHOR TO CONFIRM: which materials are moved to SI and that each SI file actually exists before "
        "it is cited.]"])

# ================= MASTER LIST =================
pdf.add_page()
H1("Master list of open confirmations (consolidated)")
for s in [
 "Scope: confirm the pH-optimum analysis is excluded; remove EpHod from all sections.",
 "Version control: remove any older 19,071-mutation results (accuracy ~79.7%, Pearson ~0.764, MAE ~0.92).",
 "BRENDA: confirm original size (inferred 2,446) and the positive/negative/ambiguous class definitions.",
 "BRENDA: confirm the true-Tm subset sample size for the Pearson correlation of 0.60.",
 "Leakage: confirm the exact number of training records removed (stated ~132,000) and MMseqs2 version, "
 "identity/coverage/e-value thresholds and commands from code.",
 "ΔΔG: define mathematically and confirm that negative ΔΔG denotes stabilization.",
 "Stage 2: confirm how each decision threshold (0.00, +0.25, +0.50, +1.00 kcal/mol) is applied.",
 "Confirm reported AUC values are ROC AUC and the plotted content of Figure 1.",
 "Confirm BRENDA and S669 were independent of all model development, including threshold selection.",
 "Confirm the current feature set, each model's package versions, hyperparameters and seeds, and the "
 "ensemble type from code.",
 "Publish the Zenodo data deposit and insert the data DOI; verify deposit contents.",
 "Archive the exact code release and insert the code DOI; verify repository contents (README, license, "
 "environment, seeds, reproduction commands).",
 "Verify dataset citations, versions, licenses and redistribution rights for every source.",
 "Confirm all author affiliations (institution and country) and the corresponding-author email.",
 "Confirm CRediT author-contribution roles and author order for all three authors.",
 "Confirm competing interests with every author.",
 "Confirm whether and where LLM use must be disclosed under current Scientific Reports policy.",
 "Regenerate Figure 1 from saved predictions as a vector file.",
 "Final author-led numerical and citation verification before submission.",
]:
    b(s, q=True)

# ================= DISPLAY ITEMS =================
def dt(rows,widths,aligns="CENTER"):
    table(rows,widths,aligns,layout="MINIMAL",fs=8.4)
def cap(t):
    pdf.set_font("S","B",9.6); pdf.set_text_color(*INK); pdf.multi_cell(0,4.8,t,**NX); pdf.ln(0.6)
def panel(t):
    pdf.set_font("S","I",8.6); pdf.set_text_color(*GREY); pdf.multi_cell(0,4.2,t,**NX)
def note(t):
    pdf.set_font("S","I",8.4); pdf.set_text_color(*GREY); pdf.multi_cell(0,4.1,t,**NX); pdf.ln(1.5)

pdf.add_page()
H1("Display items (main tables and Figure 1)")
P("The four main display items are shown below with their labels. Insert Tables 1-3 into the manuscript "
  "as editable tables and place Figure 1 as a vector figure. Cross-references: Table 1 - Results dataset "
  "subsection and Section 15; Table 2 - Results Stage 1 subsection; Table 3 - Results Stage 2 subsection; "
  "Figure 1 - Results comparison subsection and Section 14.")

cap("Table 1. Composition of the assembled dataset and allocation by measurement type.")
panel("Panel A - Rows by source dataset")
dt([["Source dataset","Rows"],
 ["Domainome (aPCA abundance)","457,943"],["Tsuboyama et al. 2023 (mega-scale)","357,155"],
 ["Tsuboyama et al. 2023 (double mutants)","138,275"],["Meltome Atlas","27,884"],
 ["ThermoMutDB","11,520"],["FireProtDB","6,549"],["ProDDG/S2648","2,562"],["Total","1,001,888"]],
 widths=(60,25), aligns=("LEFT","RIGHT"))
panel("Panel B - Rows by measurement type and model allocation")
dt([["Measurement type","Rows","Used to train"],
 ["ΔΔG (mutation)","508,693","Stage 2"],["Abundance (proxy)","457,943","Not used in this study"],
 ["Tm (whole protein)","29,654","Stage 1"],["ΔTm","5,598","Not used in this study"],
 ["Total","1,001,888","-"]], widths=(38,22,40), aligns=("LEFT","RIGHT","LEFT"))
note("Note. Exact record counts; both panels sum to 1,001,888. [AUTHOR TO CONFIRM: dataset versions and "
     "redistribution conditions per source.]")

cap("Table 2. Stage 1 melting-temperature model: classification on the independent BRENDA benchmark "
    "(1,563 enzymes; 979 positive, 584 negative).")
panel("Panel A - Decision-threshold analysis")
dt([["Threshold","Precision","Recall","F1","Accuracy"],
 ["46 °C","0.65","0.98","0.78","0.66"],["50 °C","0.71","0.77","0.74","0.66"],
 ["52 °C","0.77","0.63","0.69","0.65"],["60 °C","0.93","0.39","0.55","0.60"],
 ["66 °C","0.98","0.32","0.49","0.58"]], widths=(24,22,18,16,20))
panel("Panel B - Per-model performance at the 60 °C threshold")
dt([["Model","Precision","Recall","F1"],
 ["Extra Trees","0.963","0.369","0.533"],["Random Forest","0.948","0.355","0.517"],
 ["LightGBM","0.933","0.395","0.555"],["CatBoost","0.927","0.400","0.559"],
 ["XGBoost","0.917","0.394","0.551"],["Ensemble","0.945","0.388","0.550"]], widths=(34,24,20,18))
note("Note. ROC AUC = 0.732 (bootstrap 95% CI 0.708 to 0.755). [AUTHOR TO CONFIRM: class definitions; "
     "thresholds not selected on this benchmark.]")

cap("Table 3. Stage 2 mutation-level ΔΔG model: regression and classification on the independent S669 "
    "benchmark (669 mutations; 168 stabilizing, 501 non-stabilizing reference cases).")
panel("Panel A - Regression (predicted vs. experimental ΔΔG)")
dt([["Model","Pearson","Spearman","RMSE (kcal/mol)"],
 ["Extra Trees","0.436","0.461","1.481"],["XGBoost","0.382","0.413","1.514"],
 ["Random Forest","0.369","0.396","1.523"],["LightGBM","0.330","0.364","1.549"],
 ["CatBoost","0.324","0.362","1.557"],["Multilayer perceptron","0.298","0.338","1.612"],
 ["Ensemble","0.390","0.414","1.509"]], widths=(42,20,20,28), aligns=("LEFT","CENTER","CENTER","CENTER"))
panel("Panel B - Ensemble classification across prediction-decision thresholds")
dt([["Threshold","Precision","Recall","F1","Accuracy"],
 ["0.00 kcal/mol","0.81","0.10","0.18","0.77"],["+0.25 kcal/mol","0.51","0.26","0.35","0.75"],
 ["+0.50 kcal/mol","0.38","0.46","0.42","0.67"],["+1.00 kcal/mol","0.31","0.80","0.45","0.51"]],
 widths=(28,20,16,16,20))
panel("Panel C - Per-model performance at the strict (0.00 kcal/mol) threshold")
dt([["Model","Precision","Recall"],
 ["Random Forest","0.900","0.107"],["Extra Trees","0.786","0.065"],["XGBoost","0.735","0.149"],
 ["LightGBM","0.625","0.119"],["CatBoost","0.429","0.107"],["Multilayer perceptron","0.415","0.202"],
 ["Ensemble","0.810","0.101"]], widths=(40,24,24))
note("Note. Ensemble ROC AUC = 0.669 (95% CI 0.62 to 0.72); ensemble regression Pearson = 0.390 "
     "(95% CI 0.32 to 0.46); 2,000 bootstrap resamples. [AUTHOR TO CONFIRM: ΔΔG sign convention; "
     "predicted-stabilizer rule.]")

cap("Figure 1. Precision as a function of the decision threshold for the two-stage framework on "
    "independent benchmarks.")
P("Precision for the Stage 1 whole-protein melting-temperature model on the leakage-audited BRENDA "
  "benchmark (n = 1,563; 979 positive, 584 negative) across melting-temperature thresholds (panel a) and "
  "for the Stage 2 mutation-level ΔΔG model on S669 (n = 669; 168 stabilizing, 501 non-stabilizing "
  "reference cases) across ΔΔG thresholds (panel b). In both panels the horizontal axis is oriented so "
  "that precision increases with more stringent thresholds. Recall, F1, accuracy and ROC AUC for the "
  "same evaluations are reported in Tables 2 and 3. Insert file paper_figures/precision_vs_threshold.pdf.")
_fig="/Users/admin/Documents/PET - Lab/paper_figures/precision_vs_threshold.png"
if pdf.get_y()>pdf.h-70: pdf.add_page()
_w=150; _x=(pdf.w-_w)/2
if os.path.exists(_fig): pdf.image(_fig, x=_x, y=pdf.get_y()+1, w=_w)
pdf.set_y(pdf.get_y()+1+_w*(1200/2520)+2)

# ================= OPTIONAL FIGURES =================
def dbox(x,y,w,h,text,fs=7.4,bold=False):
    pdf.set_draw_color(*GREY); pdf.set_line_width(0.3); pdf.rect(x,y,w,h)
    n=text.count("\n")+1
    pdf.set_xy(x+1.5, y+(h-n*3.3)/2)
    pdf.set_font("S","B" if bold else "",fs); pdf.set_text_color(*INK)
    pdf.multi_cell(w-3,3.3,text,align="C",**NX)
def adown(x,y1,y2):
    pdf.set_draw_color(*GREY); pdf.set_line_width(0.3)
    pdf.line(x,y1,x,y2); pdf.line(x-1.3,y2-1.8,x,y2); pdf.line(x+1.3,y2-1.8,x,y2)
def side(x,y,t):
    pdf.set_xy(x,y); pdf.set_font("S","I",7.4); pdf.set_text_color(*GREY); pdf.multi_cell(44,3.2,t,**NX)
def need(h):
    if pdf.get_y()>pdf.h-16-h: pdf.add_page()
def phbox(h,txt):
    y=pdf.get_y()+1; pdf.set_draw_color(*GREY); pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin,y,pdf.w-2*pdf.l_margin,h)
    pdf.set_xy(pdf.l_margin,y+h/2-4); pdf.set_font("S","I",8.6); pdf.set_text_color(*GREY)
    pdf.multi_cell(0,4.2,txt,align="C",**NX); pdf.set_y(y+h+2)

pdf.add_page()
H1("Optional / candidate figures (in case needed)")
P("These are not part of the current four main display items (Tables 1-3 + Figure 1). Add any of them "
  "only if useful, keeping figures plus tables at eight or fewer. The two schematics use confirmed "
  "values; the scatter plots must be generated from saved per-point predictions and appear here only as "
  "placeholders.")

need(58)
cap("Optional Figure A. Two-stage screen-then-optimize workflow (draft schematic).")
y0=pdf.get_y()+1
dbox(47,y0,100,11,"Assembled protein-stability data\n1,001,888 records; 7 public sources")
sy=y0+20; scL=58; scR=140
dbox(20,sy,76,13,"Stage 1: whole-protein Tm model\n29,654 Tm records")
dbox(102,sy,76,13,"Stage 2: mutation ΔΔG model\n508,693 ΔΔG records")
by=y0+39
dbox(20,by,76,12.5,"Independent BRENDA benchmark\nROC AUC 0.732 (95% CI 0.708-0.755)")
dbox(102,by,76,12.5,"Independent S669 benchmark\nROC AUC 0.669 (95% CI 0.62-0.72)")
midY=y0+16
pdf.set_draw_color(*GREY); pdf.set_line_width(0.3)
pdf.line(97,y0+11,97,midY); pdf.line(scL,midY,scR,midY)
adown(scL,midY,sy); adown(scR,midY,sy)
adown(scL,sy+13,by); adown(scR,sy+13,by)
pdf.set_y(by+12.5+2)
note("Draft schematic using confirmed values. [AUTHOR TO CONFIRM: that this reads as a standard figure; "
     "Scientific Reports discourages decorative schemes and graphical abstracts.]")

need(56)
cap("Optional Figure B. BRENDA benchmark sample flow after leakage auditing.")
y0=pdf.get_y()+1
dbox(42,y0,96,11,"Original BRENDA benchmark: 2,446 enzymes (inferred)")
adown(90,y0+11,y0+18); side(140,y0+12,"minus 412 removed\n(406 by homology)")
dbox(42,y0+18,96,10,"Post-audit set: 2,034 enzymes")
adown(90,y0+28,y0+35); side(140,y0+30,"minus 471 ambiguous")
dbox(30,y0+35,120,11,"Confident binary subset: 1,563\n(979 positive, 584 negative)")
pdf.set_y(y0+46+1)
note("Counts as reported; 2,446 is inferred (2,034 + 412). Training data separately: approximately "
     "132,000 records removed (131,479 by homology). [AUTHOR TO CONFIRM: original benchmark size.]")

need(38)
cap("Optional Figure C. Predicted vs. experimental ΔΔG on the S669 benchmark (regression).")
phbox(30,"[FIGURE PLACEHOLDER - generate from saved per-mutation S669 predictions; ensemble Pearson "
         "0.390, RMSE 1.509 kcal/mol. Do not reconstruct from summary values.]")

need(38)
cap("Optional Figure D. Predicted vs. reported Tm on the BRENDA benchmark (regression).")
phbox(30,"[FIGURE PLACEHOLDER - generate from saved BRENDA predictions; regression Pearson 0.54 "
         "(0.60 on the true-Tm subset). Do not reconstruct from summary values.]")

need(14)
P("Optional Figure E (Supplementary only): per-model precision and recall bar charts would duplicate "
  "Tables 2 and 3; include in Supplementary Information only if wanted, not as a main figure.")

out="/Users/admin/Documents/PET - Lab/Manuscript_Section_Guide.pdf"
pdf.output(out); print("WROTE",out)
