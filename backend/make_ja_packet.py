"""Manuscript-ready packet for JA (James Ponzio): Results, Data Availability,
Code Availability, Additional Information and title-page items, prepared to
Scientific Reports conventions. Uses only values from the source packet; all
missing/ambiguous items are shown as visible [AUTHOR TO CONFIRM: ...] placeholders."""
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
def H2(t):
    pdf.ln(1.5); pdf.set_font("S","B",10.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0,5.5,t,**NX)
def P(t):
    pdf.set_font("S","",9.8); pdf.set_text_color(*INK); pdf.multi_cell(0,4.8,t,**NX); pdf.ln(1.2)
def I(t):
    pdf.set_font("S","I",9); pdf.set_text_color(*GREY); pdf.multi_cell(0,4.4,t,**NX); pdf.ln(1)
def Q(t):
    pdf.set_font("S","B",8.8); pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin+4); pdf.multi_cell(0,4.4,t,**NX); pdf.ln(0.8)
def bullet(t):
    pdf.set_font("S","",9.6); pdf.set_text_color(*INK)
    x=pdf.get_x(); pdf.cell(4,4.6,"-"); pdf.multi_cell(0,4.6,t); pdf.set_x(x)
def dtable(rows, widths, aligns="CENTER", layout="MINIMAL", fs=8.8):
    pdf.set_font("S","",fs); pdf.set_text_color(*INK)
    with pdf.table(col_widths=widths, text_align=aligns, line_height=4.7,
                   borders_layout=layout, first_row_as_headings=True,
                   headings_style=BOLD) as t:
        for r in rows:
            row=t.row()
            for c in r: row.cell(str(c))

# ================= TITLE =================
pdf.ln(1); pdf.set_font("S","B",16); pdf.set_text_color(*INK)
pdf.multi_cell(0,7,"Manuscript-Ready Sections",align="C")
pdf.set_font("S","",11); pdf.set_x(pdf.l_margin)
pdf.cell(0,6,"Prepared for James Ponzio (JA)",align="C",new_x="LMARGIN",new_y="NEXT")
pdf.set_font("S","I",9.3); pdf.set_text_color(*GREY); pdf.set_x(pdf.l_margin)
pdf.cell(0,5,"Results | Data availability | Code availability | Additional information   -   target: Nature Scientific Reports",
         align="C",new_x="LMARGIN",new_y="NEXT")
pdf.ln(3)
P("This document contains manuscript-ready text using only the numerical results confirmed in the "
  "source packet. Missing or ambiguous items are shown as visible [AUTHOR TO CONFIRM: ...] "
  "placeholders and are not silently resolved. All confirmed values are preserved exactly.")

# ================= PART A =================
H1("Part A - Consistency and compliance audit")
dtable([
 ["Location","Issue","Required correction","Confirm?"],
 ["Table 1 (sources)","457,943+357,155+138,275+27,884+11,520+6,549+2,562 = 1,001,888","None; sum matches total","No"],
 ["Table 1 (types)","508,693+457,943+29,654+5,598 = 1,001,888","None; sum matches total","No"],
 ["BRENDA counts","979+584+471=2,034; 979+584=1,563","None; internally consistent","No"],
 ["BRENDA original size","2,034+412=2,446 is inferred, not stated","Present as inferred; placeholder","Yes"],
 ["S669 counts","168+501=669","None; internally consistent","No"],
 ["S669 AUC","Packet uses both 0.67 and 0.669","Use 0.669 in text/tables","No"],
 ["Units","Packet renders C; ddG/dTm/Tm","Use degC, kcal/mol, delta symbols","No"],
 ["Model names","lowercase_underscore forms","Standardize capitalization","No"],
 ["BRENDA r=0.60","Could read as a CI bound","State as separate subset Pearson","Yes"],
 ["Class definitions","Positive/negative/ambiguous not defined","Do not invent; placeholder","Yes"],
 ["delta-delta-G sign","Not defined mathematically","Placeholder","Yes"],
 ["S669 thresholds","How a predicted stabilizer is defined unclear","Placeholder","Yes"],
 ["Figure 1","PR plot described but ROC AUC reported","Do not mislabel; placeholder","Yes"],
 ["Leakage total","~132,000 approximate","Keep approximate; placeholder for exact","Yes"],
 ["Independence","Sweeps computed on benchmarks","Confirm no threshold optimization on them","Yes"],
 ["Licenses/versions","Only BRENDA (CC BY 4.0) verified","Do not invent; verify","Yes"],
 ["DOIs / repo contents","Not verified","Placeholders; do not assert availability","Yes"],
 ["Competing interests","Must not be assumed","Author-confirmation placeholder","Yes"],
 ["Promotional phrasing","'trustworthy/cleanest shortlist', etc.","Removed","No"],
], widths=(20,34,32,10), aligns=("LEFT","LEFT","LEFT","CENTER"), layout="HORIZONTAL_LINES", fs=7.6)

# ================= PART B =================
pdf.add_page()
H1("Part B - Results")
H2("Dataset composition and leakage auditing")
P("The assembled data resource comprised 1,001,888 rows obtained from seven publicly available "
  "sources, with each row assigned a single measurement type (Table 1). The Stage 1 screening model "
  "was trained on the 29,654 whole-protein melting-temperature (Tm) records, and the Stage 2 mutation "
  "model was trained on the 508,693 mutation-level ΔΔG records. The abundance-proxy records (457,943) "
  "and the ΔTm records (5,598) were retained in the assembled resource but were not used to train the "
  "two models reported here.")
P("Both the training data and the benchmark datasets underwent exact-record and sequence-homology "
  "leakage auditing prior to evaluation. Approximately 132,000 potentially overlapping training records "
  "were removed, of which 131,479 were identified through sequence-homology matching rather than "
  "exact-key matching. Removal of homologous records supports the independence of the benchmark "
  "evaluations from the training data.")
Q("[AUTHOR TO CONFIRM: Confirm that abundance-proxy and ΔTm records were not used in any modeling step reported here.]")
Q("[AUTHOR TO CONFIRM: Replace 'approximately 132,000' with the exact number of removed records if available.]")

H2("Stage 1 screening-model performance on the independent BRENDA benchmark")
P("During internal development, the Stage 1 model predicted melting temperature with a held-out "
  "root-mean-square error (RMSE) of approximately 6.5 °C and a validation Pearson correlation of "
  "approximately 0.80. These are internal development results and are distinct from the independent "
  "benchmark results reported below.")
P("On the independent BRENDA benchmark, the confident binary-classification subset comprised 1,563 "
  "enzymes (979 positive and 584 negative). Evaluated as a binary classifier, the model achieved a "
  "receiver-operating-characteristic area under the curve (ROC AUC) of 0.732 (bootstrap 95% confidence "
  "interval 0.708 to 0.755). Evaluated as a regression task, the model achieved a Pearson correlation "
  "of 0.54 (bootstrap 95% confidence interval 0.50 to 0.57) between predicted and reported melting "
  "temperatures; on the subset of enzymes with directly reported Tm values, the Pearson correlation "
  "was 0.60. The regression and classification evaluations are reported separately.")
P("The BRENDA benchmark sample flow was as follows. Following leakage auditing, 412 enzymes were "
  "removed (406 identified through sequence homology rather than exact identifiers), leaving 2,034 "
  "enzymes. Of these, 471 were labeled ambiguous and were excluded from the confident "
  "binary-classification subset, yielding 1,563 enzymes (979 positive and 584 negative).")
P("Classification performance across melting-temperature decision thresholds is summarized in Table 2. "
  "Lower thresholds produced higher recall, and higher thresholds produced higher precision with reduced "
  "recall. The 46 °C threshold produced the highest F1 value (0.78) among the reported thresholds; the "
  "50 °C threshold produced comparable precision and recall (0.71 and 0.77); and the 60 °C and 66 °C "
  "thresholds produced high precision (0.93 and 0.98) with lower recall (0.39 and 0.32). At the 60 °C "
  "threshold, per-model precision ranged from 0.917 to 0.963 and recall from 0.355 to 0.400 across the "
  "individual models and the ensemble (Table 2).")
Q("[AUTHOR TO CONFIRM: The counts imply an original BRENDA benchmark of 2,446 enzymes before 412 were removed. Confirm this starting count.]")
Q("[AUTHOR TO CONFIRM: Provide the definitions of the positive, negative and ambiguous BRENDA classes.]")
Q("[AUTHOR TO CONFIRM: Report the sample size of the true-Tm subset for which the Pearson correlation of 0.60 was obtained.]")
Q("[AUTHOR TO CONFIRM: Confirm the decision thresholds were not selected using the BRENDA benchmark, preserving its independence.]")

H2("Stage 2 mutation-level stability prediction on the independent S669 benchmark")
P("The S669 benchmark comprised 669 single-point mutations. Under the stated classification convention, "
  "168 mutations were designated stabilizing reference cases and 501 non-stabilizing reference cases. "
  "The ensemble model achieved a Pearson correlation of 0.390 (bootstrap 95% confidence interval 0.32 "
  "to 0.46) between predicted and experimental ΔΔG. Treated as a binary classifier for stabilizing "
  "mutations, the ensemble achieved a ROC AUC of 0.669 (bootstrap 95% confidence interval 0.62 to "
  "0.72). Confidence intervals were estimated from 2,000 bootstrap resamples of the saved predictions.")
P("Regression performance for the individual models and the ensemble is reported in Table 3. Extra "
  "Trees produced the highest Pearson (0.436) and Spearman (0.461) correlations and the lowest RMSE "
  "(1.481 kcal/mol) among the reported models. The ensemble produced a Pearson correlation of 0.390 and "
  "an RMSE of 1.509 kcal/mol.")
P("Classification performance across prediction-decision thresholds is reported in Table 3. Increasing "
  "the decision threshold increased recall while reducing precision and accuracy: at the strict 0.00 "
  "kcal/mol threshold, precision was 0.81 and recall 0.10, whereas at the +1.00 kcal/mol threshold, "
  "precision was 0.31 and recall 0.80. These are prediction-decision thresholds and do not alter the "
  "experimental definition of a stabilizing mutation. At the strict threshold, per-model precision "
  "ranged from 0.415 to 0.900 and recall from 0.065 to 0.202 (Table 3); at this threshold the ensemble "
  "favored precision (0.810) over sensitivity (recall 0.101), identifying a small proportion of the "
  "reference stabilizing mutations.")
Q("[AUTHOR TO CONFIRM: Define ΔΔG mathematically (order of subtraction) and confirm negative ΔΔG denotes stabilization. 'Positive class' refers to the label, not a positive numerical ΔΔG value.]")
Q("[AUTHOR TO CONFIRM: Specify precisely how a mutation is classified as a predicted stabilizer at each decision threshold.]")

H2("Comparison of screening and mutation-model operating characteristics")
P("The Stage 1 BRENDA classification produced a higher ROC AUC (0.732) than the Stage 2 S669 "
  "classification (0.669). Across the reported threshold analyses, the Stage 1 model sustained higher "
  "precision over a broader range of recall, whereas the Stage 2 model achieved high precision only at "
  "a strict threshold associated with low recall (Fig. 1). The two stages address different prediction "
  "problems and were evaluated on different datasets, class labels and sample sizes; the reported "
  "metrics therefore do not constitute a controlled head-to-head comparison, and no statistical test of "
  "the difference between the two AUC values was performed.")

# ================= PART C =================
pdf.add_page()
H1("Part C - Publication-ready tables")
H2("Table 1. Composition of the assembled dataset and allocation by measurement type.")
I("Panel A - Rows by source dataset")
dtable([["Source dataset","Rows"],
 ["Domainome (aPCA abundance)","457,943"],["Tsuboyama et al. 2023 (mega-scale)","357,155"],
 ["Tsuboyama et al. 2023 (double mutants)","138,275"],["Meltome Atlas","27,884"],
 ["ThermoMutDB","11,520"],["FireProtDB","6,549"],["ProDDG/S2648","2,562"],
 ["Total","1,001,888"]], widths=(60,25), aligns=("LEFT","RIGHT"))
I("Panel B - Rows by measurement type and model allocation")
dtable([["Measurement type","Rows","Used to train"],
 ["ΔΔG (mutation)","508,693","Stage 2"],["Abundance (proxy)","457,943","Not used in this study"],
 ["Tm (whole protein)","29,654","Stage 1"],["ΔTm","5,598","Not used in this study"],
 ["Total","1,001,888","-"]], widths=(38,22,40), aligns=("LEFT","RIGHT","LEFT"))
I("Table 1 note. Values are exact record counts; both panels sum to 1,001,888. "
  "[AUTHOR TO CONFIRM: dataset versions and redistribution conditions per source.]")

H2("Table 2. Stage 1 melting-temperature model: classification on the independent BRENDA benchmark "
   "(1,563 enzymes; 979 positive, 584 negative).")
I("Panel A - Decision-threshold analysis")
dtable([["Threshold","Precision","Recall","F1","Accuracy"],
 ["46 °C","0.65","0.98","0.78","0.66"],["50 °C","0.71","0.77","0.74","0.66"],
 ["52 °C","0.77","0.63","0.69","0.65"],["60 °C","0.93","0.39","0.55","0.60"],
 ["66 °C","0.98","0.32","0.49","0.58"]], widths=(24,22,18,16,20))
I("Panel B - Per-model performance at the 60 °C threshold")
dtable([["Model","Precision","Recall","F1"],
 ["Extra Trees","0.963","0.369","0.533"],["Random Forest","0.948","0.355","0.517"],
 ["LightGBM","0.933","0.395","0.555"],["CatBoost","0.927","0.400","0.559"],
 ["XGBoost","0.917","0.394","0.551"],["Ensemble","0.945","0.388","0.550"]],
 widths=(34,24,20,18))
I("Table 2 note. ROC AUC = 0.732 (bootstrap 95% CI 0.708 to 0.755). Precision is the proportion of "
  "enzymes predicted positive that were positive; recall is the proportion of positive enzymes "
  "recovered. [AUTHOR TO CONFIRM: class definitions; thresholds not selected on this benchmark.]")

H2("Table 3. Stage 2 mutation-level ΔΔG model: regression and classification on the independent S669 "
   "benchmark (669 mutations; 168 stabilizing, 501 non-stabilizing reference cases).")
I("Panel A - Regression (predicted vs. experimental ΔΔG)")
dtable([["Model","Pearson","Spearman","RMSE (kcal/mol)"],
 ["Extra Trees","0.436","0.461","1.481"],["XGBoost","0.382","0.413","1.514"],
 ["Random Forest","0.369","0.396","1.523"],["LightGBM","0.330","0.364","1.549"],
 ["CatBoost","0.324","0.362","1.557"],["Multilayer perceptron","0.298","0.338","1.612"],
 ["Ensemble","0.390","0.414","1.509"]], widths=(42,20,20,28), aligns=("LEFT","CENTER","CENTER","CENTER"))
I("Panel B - Ensemble classification across prediction-decision thresholds")
dtable([["Threshold","Precision","Recall","F1","Accuracy"],
 ["0.00 kcal/mol","0.81","0.10","0.18","0.77"],["+0.25 kcal/mol","0.51","0.26","0.35","0.75"],
 ["+0.50 kcal/mol","0.38","0.46","0.42","0.67"],["+1.00 kcal/mol","0.31","0.80","0.45","0.51"]],
 widths=(28,20,16,16,20))
I("Panel C - Per-model performance at the strict (0.00 kcal/mol) threshold")
dtable([["Model","Precision","Recall"],
 ["Random Forest","0.900","0.107"],["Extra Trees","0.786","0.065"],["XGBoost","0.735","0.149"],
 ["LightGBM","0.625","0.119"],["CatBoost","0.429","0.107"],["Multilayer perceptron","0.415","0.202"],
 ["Ensemble","0.810","0.101"]], widths=(40,24,24))
I("Table 3 note. Ensemble ROC AUC = 0.669 (bootstrap 95% CI 0.62 to 0.72); ensemble regression Pearson "
  "= 0.390 (bootstrap 95% CI 0.32 to 0.46); 2,000 bootstrap resamples. Thresholds are "
  "prediction-decision thresholds. [AUTHOR TO CONFIRM: ΔΔG sign convention; predicted-stabilizer rule.]")

# ================= PART D =================
pdf.add_page()
H1("Part D - Figure 1 caption")
P("Figure 1. Operating characteristics of the two-stage modeling framework on independent benchmark "
  "datasets. Threshold-dependent precision and recall are shown for the Stage 1 whole-protein "
  "melting-temperature screening model, evaluated on the confident binary subset of the "
  "leakage-audited BRENDA benchmark (n = 1,563; 979 positive, 584 negative), and for the Stage 2 "
  "mutation-level ΔΔG model, evaluated on the S669 benchmark (n = 669; 168 stabilizing, 501 "
  "non-stabilizing reference cases). The Stage 1 classification produced a "
  "receiver-operating-characteristic area under the curve of 0.732, and the Stage 2 stabilizing-mutation "
  "classification produced a receiver-operating-characteristic area under the curve of 0.669. Where "
  "shown, annotated points indicate selected decision thresholds. Curves were generated from saved "
  "out-of-sample predictions.")
H2("To be confirmed before the figure is finalized")
for s in [
 "Whether Figure 1 plots a precision-recall curve, a ROC curve, a threshold plot, or a combination; "
 "label axes and each AUC accordingly. The reported AUC values are ROC AUC and must not be labeled "
 "area under the precision-recall curve.",
 "Which class is treated as positive in each panel.",
 "Whether chance baselines are shown.",
 "Whether bootstrap confidence bands are shown.",
 "Whether the ensemble only, or all models, are plotted.",
 "Whether individual decision thresholds are annotated.",
 "The figure is not generated here; it must be produced from the saved predictions as a vector file "
 "(PDF, EPS or SVG).",
]:
    bullet(s)

# ================= PART E =================
H1("Part E - Data availability")
P("The processed training table, leakage-audited benchmark datasets, saved model predictions, "
  "dataset-split identifiers, leakage-audit outputs and trained model artifacts generated in this study "
  "will be available through Zenodo at [INSERT ZENODO DOI]. Publicly available source datasets analyzed "
  "in this study are identified in the Methods and cited in the reference list.")
Q("[AUTHOR TO CONFIRM: change 'will be available' to 'are available' only after public/reviewer access exists.]")
Q("[AUTHOR TO CONFIRM: verify each listed artifact is actually in the deposit; remove any item not deposited.]")
Q("[AUTHOR TO VERIFY: license, version and redistribution conditions for each source dataset (BRENDA; Tsuboyama mega-scale; Tsuboyama doubles; Domainome; FireProtDB; Meltome Atlas; ThermoMutDB; ProDDG/S2648; S669; UniProt); state any third-party restrictions.]")
Q("[AUTHOR TO CONFIRM: whether the pH-optimum analysis (and EpHod dataset) is part of this submission; if not, remove EpHod entirely.]")

# ================= PART F =================
H1("Part F - Code availability")
P("Source code for dataset assembly, sequence-homology leakage auditing, model training, model "
  "evaluation, threshold analysis, bootstrap confidence-interval estimation and figure generation is "
  "available through the project repository at https://github.com/AyushIyer31/PET-Lab and will be "
  "archived in a DOI-minting repository at [INSERT ARCHIVED CODE DOI].")
Q("[AUTHOR TO CONFIRM: change 'will be archived' to 'is archived' only after the permanent archive exists; GitHub alone is not a permanent versioned archive.]")
Q("[AUTHOR TO CONFIRM: verify these scripts are present in the public repository before listing them: ensemble_eval.py, s669_sweep.py, predict_and_sweep.py, bootstrap_ci.py.]")
Q("[AUTHOR TO CONFIRM: verify the archived release includes README, license, dependency/versions, random seeds, hyperparameters, split identifiers, leakage-audit commands, saved predictions, bootstrap procedure, figure-generation script, and the exact commit/release for this manuscript.]")

# ================= PART G =================
H1("Part G - Additional Information (author-supplied)")
H2("Competing interests")
Q("[ALL AUTHORS TO CONFIRM: whether any relevant financial or non-financial competing interests exist. "
  "If none, use: 'The authors declare no competing interests.' If any exist, provide an explicit "
  "author-specific declaration. The statement must match the submission system.]")

# ================= PART H =================
H1("Part H - Title-page items")
P("Authors:  Ayush Iyer(1,*),  James Ponzio(2),  Abhinav Iyer(3)")
P("(1) [AUTHOR TO CONFIRM AFFILIATION - institution and country]")
P("(2) [AUTHOR TO CONFIRM AFFILIATION - institution and country]")
P("(3) [AUTHOR TO CONFIRM AFFILIATION - institution and country]")
P("* Corresponding author: Ayush Iyer.  Email: iyer.ayush31@gmail.com")
I("Internal submission-system fields (not printed in the manuscript body): "
  "Ayush Iyer / James Ponzio / Abhinav Iyer ORCID iDs - [INSERT ORCID IF REQUESTED BY SUBMISSION SYSTEM]. "
  "Superscripts assume three distinct affiliations; consolidate if authors share an institution.")

# ================= PART I =================
pdf.add_page()
H1("Part I - Internal author action checklist (unresolved only)")
for s in [
 "Confirm the original BRENDA benchmark size (inferred 2,446).",
 "Confirm the exact number of training records removed during leakage auditing (stated ~132,000).",
 "Define the BRENDA positive, negative and ambiguous classes.",
 "Define ΔΔG mathematically and confirm its sign convention.",
 "Specify how each Stage 2 decision threshold classifies a predicted stabilizer.",
 "Confirm whether reported AUC values are ROC AUC or precision-recall AUC, and the plotted content of Figure 1.",
 "Confirm the true-Tm subset sample size (for r = 0.60).",
 "Confirm BRENDA and S669 were independent of all model development (training, feature selection, tuning, threshold selection).",
 "Confirm repository contents.",
 "Publish the Zenodo data deposit and insert the permanent data DOI.",
 "Archive the exact code release and insert the permanent code DOI.",
 "Verify dataset citations, versions, licenses and redistribution rights.",
 "Decide whether the pH-optimum analysis is included; remove EpHod if excluded.",
 "Provide author ORCID iDs if requested by the submission system.",
 "Confirm all author affiliations (institution and country).",
 "Confirm competing interests with every author.",
 "Regenerate Figure 1 from saved predictions as a vector file.",
 "Confirm all tables/figures remain within the eight-display-item limit (plan: Tables 1-3 + Figure 1 = four).",
 "Determine whether use of a large language model for editing/drafting requires disclosure under current Scientific Reports policy; authors remain responsible for verifying every statement, number, citation and interpretation.",
 "Confirm Methods contain the items in the Methods-gap check below.",
 "Conduct a final author-led numerical and citation verification.",
]:
    bullet(s)
I("Methods-gap check (add to Methods; do not invent): inclusion/exclusion criteria per source; "
  "duplicate-removal; measurement-type assignment; Stage 1/2 training targets; BRENDA class "
  "definitions; ΔΔG sign convention; S669 class definition; data splits; MMseqs2 identity/coverage "
  "thresholds and parameters; leakage-removal logic; feature generation; missing-data handling; model "
  "hyperparameters; ensemble construction; threshold-selection procedure; metric definitions; bootstrap "
  "method and 2,000 resamples; CI calculation; random seeds; software packages and versions.")

# ================= PART J =================
H1("Part J - Final verification report")
dtable([
 ["Check","Status","Explanation"],
 ["Dataset totals","Internally consistent","Both count sets sum to 1,001,888."],
 ["BRENDA counts","Internally consistent / confirm","979+584+471=2,034; 979+584=1,563 verified; 2,446 inferred."],
 ["S669 counts","Internally consistent","168+501=669."],
 ["Confidence intervals","Verified from packet","Each CI tied to its metric."],
 ["AUC terminology","Corrected / confirm","0.669 standardized; treated as ROC; Fig 1 content to confirm."],
 ["Regression vs classification","Corrected","Reported as separate evaluations."],
 ["Validation vs test","Corrected","Internal development separated from independent benchmarks."],
 ["Unit consistency","Corrected","degC, kcal/mol and delta symbols standardized."],
 ["Model-name consistency","Corrected","Standard capitalization applied."],
 ["Table/figure references","Corrected","Tables 1-3 and Figure 1 all cited."],
 ["Repository placeholders","Not verifiable","DOIs and contents placeholdered."],
 ["Competing interests","Needs confirmation","Not assumed; placeholder inserted."],
 ["Corresponding-author placement","Corrected","Moved to title page."],
 ["Publisher boilerplate","Corrected","Removed from author-written text."],
 ["Supplementary Information","Corrected","Treated as separate files; none asserted to exist."],
 ["Remaining confirmations","Needs confirmation","See Part I."],
], widths=(28,26,46), aligns=("LEFT","LEFT","LEFT"), layout="HORIZONTAL_LINES", fs=7.8)
I("This manuscript is not fully verified: unresolved placeholders remain (Parts A, E, F, G, H, I).")

out="/Users/admin/Documents/PET - Lab/JA_Paper_Packet.pdf"
pdf.output(out); print("WROTE",out)
