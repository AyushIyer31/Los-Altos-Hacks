"""Assemble the full two-stage (Stage 1 = Tm screen; Stage 2 = ddG optimize) manuscript
into a single readable PDF, following Manuscript_Section_Guide.pdf including its Display
items (Tables 1-3 with all panels + Figure 1) and scope/version-control rules.

- Prose written only from facts in the guide; unresolved items kept as visible
  [AUTHOR TO CONFIRM ...] placeholders (styled), never invented.
- Cautions set apart as labeled WARNING / NOTE callouts in every relevant section.
- Full multi-panel Tables 1-3 and the real Figure 1 image are embedded (Display items).
- Editorial SCOPE banner captures the guide's scope- and version-control rules.
NOTE: this is the NEW two-stage paper; separate from the older technical_paper.md.
"""
import os
from fpdf import FPDF
import matplotlib as _mpl

FD = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "PETase_ML_Manuscript_Full.pdf")
FIG1 = os.path.join(ROOT, "paper_figures", "precision_vs_threshold.png")

INK = (20, 20, 20); GREY = (90, 90, 90)
WARN_BG = (253, 246, 227); WARN_BAR = (176, 122, 0)
NOTE_BG = (233, 241, 242); NOTE_BAR = (26, 94, 102)
SCOPE_BG = (238, 238, 241); SCOPE_BAR = (108, 108, 122)
PH_COL = (160, 45, 45)
HDR_BG = (224, 230, 230); ROW_BG = (245, 247, 247)
BOX = (120, 120, 120)
M = 18.0


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(9); self.set_x(M); self.set_font("S", "I", 7.5); self.set_text_color(*GREY)
        self.cell(0, 4, "PETase-ML — two-stage leakage-audited stability framework (working draft)")
        self.set_y(M)

    def footer(self):
        self.set_y(-13); self.set_font("S", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 4, "Page %s" % self.page_no(), align="C")


pdf = PDF(format="letter")
pdf.set_margins(M, M, M); pdf.set_auto_page_break(True, 16)
pdf.add_font("S", "", os.path.join(FD, "DejaVuSerif.ttf"))
pdf.add_font("S", "B", os.path.join(FD, "DejaVuSerif-Bold.ttf"))
pdf.add_font("S", "I", os.path.join(FD, "DejaVuSerif-Italic.ttf"))
PW = pdf.w; CW = PW - 2 * M; PB = pdf.h - 16
NX = dict(new_x="LMARGIN", new_y="NEXT")


def ensure(space):
    if pdf.get_y() + space > PB:
        pdf.add_page()


def count_lines(text, width, size):
    pdf.set_font("S", "", size)
    plain = text.replace("**", "").replace("*", "")
    words = plain.split(); line = ""; n = 1
    for w in words:
        t = (line + " " + w).strip()
        if pdf.get_string_width(t) <= width - 1:
            line = t
        else:
            n += 1; line = w
    return n


def H1(t):
    pdf.ln(2.5); pdf.set_x(M); pdf.set_font("S", "B", 14); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 7.2, t, **NX); pdf.ln(1.2)


def H2(t):
    pdf.ln(1.6); pdf.set_x(M); pdf.set_font("S", "B", 10.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.2, t, **NX); pdf.ln(0.8)


def P(t):
    pdf.set_x(M); pdf.set_font("S", "", 9.7); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.9, t, markdown=True, **NX); pdf.ln(1.8)


def PH(t):
    pdf.set_x(M); pdf.set_font("S", "I", 8.7); pdf.set_text_color(*PH_COL)
    pdf.multi_cell(0, 4.4, "» " + t, **NX); pdf.set_text_color(*INK); pdf.ln(1.2)


def callout(label, text, bg, bar):
    body = "**%s —** %s" % (label, text)
    lh = 4.6
    n = count_lines(body, CW - 12, 9.2)
    h = n * lh + 6
    ensure(h + 3)
    y = pdf.get_y()
    pdf.set_fill_color(*bg); pdf.rect(M, y, CW, h, "F")
    pdf.set_fill_color(*bar); pdf.rect(M, y, 2.2, h, "F")
    pdf.set_xy(M + 7, y + 3)
    old = pdf.l_margin; pdf.set_left_margin(M + 7)
    pdf.set_font("S", "", 9.2); pdf.set_text_color(*INK)
    pdf.multi_cell(CW - 12, lh, body, markdown=True, **NX)
    pdf.set_left_margin(old)
    pdf.set_y(y + h); pdf.ln(3)


def WARN(t): callout("WARNING", t, WARN_BG, WARN_BAR)
def NOTE(t): callout("NOTE", t, NOTE_BG, NOTE_BAR)
def SCOPE(t): callout("EDITORIAL SCOPE (not for submission)", t, SCOPE_BG, SCOPE_BAR)


def caption(t):
    ensure(10)
    pdf.set_x(M); pdf.set_font("S", "B", 9.3); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.8, t, **NX); pdf.ln(0.4)


def tnote(t):
    pdf.set_x(M); pdf.set_font("S", "I", 7.6); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 3.9, t, **NX); pdf.set_text_color(*INK); pdf.ln(1.4)


def panel(label, headers, rows, widths, first_left=True):
    ensure(5.2 * (len(rows) + 1) + (5 if label else 0) + 4)
    if label:
        pdf.set_x(M); pdf.set_font("S", "I", 8); pdf.set_text_color(*GREY)
        pdf.multi_cell(0, 4.2, label, **NX); pdf.set_text_color(*INK)
    h = 5.4; y = pdf.get_y(); x = M
    pdf.set_font("S", "B", 8); pdf.set_fill_color(*HDR_BG)
    for i, ht in enumerate(headers):
        al = "L" if (i == 0 and first_left) else "C"
        pdf.set_xy(x, y); pdf.cell(widths[i], h, (" " if al == "L" else "") + ht, fill=True, align=al)
        x += widths[i]
    pdf.set_y(y + h)
    pdf.set_font("S", "", 8)
    for r, row in enumerate(rows):
        y = pdf.get_y(); x = M; fill = r % 2 == 0
        if fill:
            pdf.set_fill_color(*ROW_BG)
        for i, c in enumerate(row):
            al = "L" if (i == 0 and first_left) else "C"
            pdf.set_xy(x, y); pdf.cell(widths[i], 5.0, (" " if al == "L" else "") + c, fill=fill, align=al)
            x += widths[i]
        pdf.set_y(y + 5.0)
    pdf.ln(1.2)


def flowbox(x, y, w, h, text, size=7.6):
    pdf.set_draw_color(*BOX); pdf.set_line_width(0.3); pdf.rect(x, y, w, h)
    pdf.set_xy(x, y + (h - count_lines(text, w - 4, size) * 3.6) / 2 - 0.4)
    pdf.set_font("S", "", size); pdf.set_text_color(*INK)
    old = pdf.l_margin; pdf.set_left_margin(x + 2); pdf.set_x(x + 2)
    pdf.multi_cell(w - 4, 3.6, text, align="C", **NX)
    pdf.set_left_margin(old)


def connect(x, y0, y1):
    pdf.set_draw_color(*BOX); pdf.set_line_width(0.3); pdf.line(x, y0, x, y1)


# ============================================================ 1. TITLE PAGE
pdf.add_page()
pdf.ln(12)
pdf.set_font("S", "B", 16); pdf.set_text_color(*INK)
pdf.multi_cell(0, 8, "A Two-Stage, Leakage-Audited Machine-Learning Framework for "
                     "Temperature- and Stability-Aware Screening of PET-Degrading Enzymes",
               align="C", **NX)
pdf.ln(2)
PH("[AUTHOR TO CONFIRM: final title, <= 20 words, no undefined abbreviations.]")
pdf.ln(2)
pdf.set_font("S", "", 11); pdf.set_text_color(*INK)
pdf.multi_cell(0, 5.5, "Ayush Iyer*, James Ponzio, Abhinav Iyer", align="C", **NX)
pdf.ln(1)
pdf.set_font("S", "I", 9); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.6, "*Corresponding author: Ayush Iyer — iyer.ayush31@gmail.com", align="C", **NX)
pdf.ln(3)
pdf.set_text_color(*INK)
PH("[AUTHOR TO CONFIRM AFFILIATION — institution and country] for Ayush Iyer.")
PH("[AUTHOR TO CONFIRM AFFILIATION — institution and country] for James Ponzio.")
PH("[AUTHOR TO CONFIRM AFFILIATION — institution and country] for Abhinav Iyer.")
PH("[AUTHOR TO CONFIRM: corresponding-author email before submission; ORCID iDs entered via the "
   "submission system when requested (do not invent).]")
pdf.ln(3)
SCOPE("Temperature-focused study. The pH-optimum analysis is excluded (no confirmed training run); "
      "EpHod appears only as possible future work. The training data are broad protein-stability "
      "datasets, not PETase-specific — described as a general protein-stability framework evaluated "
      "for potential application to PET-degrading enzyme engineering. No claims of experimental PET "
      "degradation, wet-lab validation, industrial deployment, environmental impact, or joint "
      "temperature-and-pH optimization are made. Older 19,071-mutation results (accuracy ~79.7%, "
      "Pearson ~0.764, MAE ~0.92) are superseded by the current 1,001,888-row two-stage results and "
      "must not appear anywhere in the manuscript.")

# ============================================================ 2. ABSTRACT
pdf.add_page()
H1("Abstract")
P("Plastic pollution is a defining environmental challenge, and enzymatic depolymerization of "
  "polyethylene terephthalate (PET) offers a biological route to recycling; however, candidate "
  "enzymes must remain stable under demanding conditions, and computational screening is frequently "
  "inflated by data leakage. We built a **general protein-stability framework** and evaluated it for "
  "potential application to PET-degrading enzyme engineering. We assembled **1,001,888** mutation and "
  "stability records from seven public sources and routed them by measurement type into a two-stage "
  "design: **Stage 1** screens whole-protein thermostability (melting temperature, Tm) and **Stage 2** "
  "optimizes mutation-level folding stability (ΔΔG). Training records were audited for exact and "
  "sequence-homology leakage against the independent benchmarks. On the independent BRENDA benchmark, "
  "Stage 1 reached a **ROC AUC of 0.732 (95% CI 0.708–0.755)**. On the independent S669 benchmark, "
  "Stage 2 reached a **ROC AUC of 0.669 (95% CI 0.62–0.72)** and an ensemble **Pearson correlation of "
  "0.390 (95% CI 0.32–0.46)**. Both stages showed a precision–recall trade-off in which precision rose "
  "and recall fell under stricter thresholds. These results describe a moderate, leakage-controlled "
  "screen-then-optimize framework suited to prioritizing candidates for experimental testing, not a "
  "validated industrial solution.")
NOTE("\"ROC AUC\" throughout denotes receiver-operating-characteristic AUC, not precision–recall AUC. "
     "No claim of industrial, wet-lab, field, or clinical readiness is made.")
PH("[AUTHOR TO CONFIRM: final word count <= 200; internal validation not presented as independent "
   "testing.]")

# ============================================================ 3. KEYWORDS
H1("Keywords")
P("protein stability; machine learning; enzyme engineering; thermostability (Tm); "
  "mutation-effect (ΔΔG) prediction; PET-degrading enzymes")

# ============================================================ 4. INTRODUCTION
pdf.add_page()
H1("1.  Introduction")
P("Polyethylene terephthalate (PET) is among the most heavily produced synthetic polymers, and its "
  "accumulation as waste is a major contributor to global plastic pollution. Biological recycling, "
  "in which enzymes depolymerize PET into recoverable monomers, is an attractive complement to "
  "mechanical and chemical recycling.")
PH("[AUTHOR TO CONFIRM: source and exact figures for every plastic-waste statistic (OECD/UNEP/"
   "peer-reviewed primary source); do not mix tonnes/tons; state whether values are production, "
   "waste, municipal waste, or PET-specific.]")
P("Enzymatic PET degradation has advanced through the discovery and engineering of PET hydrolases "
  "such as PETase and related cutinases. Thermostability matters because reactions run faster near "
  "PET's glass-transition temperature, but it is **not** the sole determinant of catalytic "
  "performance: activity, substrate binding, reaction conditions, polymer crystallinity, and pH also "
  "matter.")
P("Computational stability prediction addresses two related but non-interchangeable questions. "
  "Whole-protein screening asks whether a given enzyme is thermostable, summarized by its melting "
  "temperature (Tm). Mutation-level optimization asks how a specific substitution changes folding "
  "stability, expressed as the change in unfolding free energy (ΔΔG). Conflating the two leads to "
  "misleading claims about what a model can do.")
P("A recurring problem in this literature is data leakage. Exact-match leakage (identical sequences "
  "or mutations shared between training and test) and, more subtly, sequence-homology leakage (shared "
  "homologous background) can inflate benchmark performance. We do not claim that all prior models "
  "contain leakage; rather, we control for it explicitly and report the effect.")
P("Because whole-protein selection and mutation-level optimization are usually treated separately, "
  "this study evaluates a single, integrated **two-stage screen-then-optimize framework** — a general "
  "protein-stability method assessed for potential application to PET-degrading enzyme engineering — "
  "on leakage-audited, independent BRENDA (temperature) and S669 (ΔΔG) benchmarks, and reports its "
  "performance without overstating readiness.")
WARN("This manuscript makes no claim of experimental, wet-lab, field, or clinical use, and the "
     "training data are not PETase-specific. Every external factual claim (including all plastic-waste "
     "statistics) must carry a citation to a verified primary source before submission.")

# ============================================================ 5. RESULTS
pdf.add_page()
H1("2.  Results")

H2("2.1  Dataset composition and leakage auditing")
P("After harmonization, the multi-source staging table comprised **1,001,888 mutation and stability "
  "records** drawn from seven public sources (**Table 1**, Panel A). Because the framework treats "
  "temperature stability and folding stability as distinct problems, records were routed by "
  "measurement type (**Table 1**, Panel B): **29,654 melting-temperature (Tm) records** were assigned "
  "to Stage 1 and **508,693 folding free-energy change (ΔΔG) records** to Stage 2. Abundance-proxy "
  "(457,943) and ΔTm (5,598) records were retained in the table but were **not used** for model "
  "training.")
P("To prevent optimistic bias, every candidate training record was audited against the independent "
  "test proteins at two levels: exact-match (identical sequence and mutation) and sequence-homology "
  "(shared homologous background). Auditing removed **approximately 132,000 records**, of which "
  "**131,479 were flagged by homology alone** and would have been invisible to exact-match filtering. "
  "This indicates that homology-level leakage, rather than duplicate records, is the dominant "
  "contamination risk in this setting.")

H2("2.2  Stage 1 — temperature-stability performance (BRENDA)")
P("**Internal validation (development only).**  During development, the Stage 1 temperature model "
  "achieved an internal held-out RMSE of **approximately 6.5 °C** and a validation Pearson correlation "
  "of **approximately 0.80**. These figures describe performance on data drawn from the training "
  "distribution and are reported solely to characterize model fitting; they are not evidence of "
  "generalization and are kept separate from the independent results below.")
P("**Independent testing (BRENDA benchmark).**  Independent evaluation used a curated BRENDA "
  "whole-protein temperature-stability benchmark. During curation, **412 records were removed** to "
  "yield **2,034 proteins**; a further **471 ambiguous cases were excluded**, leaving **1,563 proteins "
  "(979 positive, 584 negative)**. On this independent set the classifier reached a **ROC AUC of 0.732 "
  "(95% CI 0.708–0.755)**. Evaluated as a regressor against measured Tm, it obtained a **Pearson "
  "correlation of 0.54 (95% CI 0.50–0.57)**, rising to **0.60 on the subset with directly measured "
  "true-Tm values** (**Table 2**).")
P("**Threshold behaviour.**  Classification performance depended strongly on the decision threshold "
  "(**Fig. 1**, Panel a). As the temperature cutoff was made more stringent, precision increased while "
  "recall fell — from 0.65 (recall 0.98) at the most permissive cutoff to 0.93 (recall 0.39) and 0.98 "
  "(recall 0.32) at the strictest cutoffs (**Table 2**, Panel A). At a fixed 60 °C threshold the "
  "individual models behaved similarly (**Table 2**, Panel B). The strict-threshold regime therefore "
  "identifies a small, high-confidence set of stabilizing candidates at the cost of missing most true "
  "positives.")
WARN("A precision value measured at a strict threshold is not an estimate of experimental success. "
     "The 0.93 precision is obtained at **39% recall**; **do not equate 0.93 precision with 93% "
     "experimental success**, and do not read it as overall model reliability. The corresponding "
     "recall, F1, accuracy, and ROC AUC in Table 2 give the complete picture.")

H2("2.3  Stage 2 — ΔΔG stabilization performance (S669)")
P("Independent folding-stability performance was assessed on the S669 benchmark of **669 single "
  "mutations (168 stabilizing, 501 non-stabilizing reference cases)**. Across **2,000 bootstrap "
  "resamples**, the gradient-boosted ensemble achieved an ensemble **Pearson correlation of 0.390 "
  "(95% CI 0.32–0.46)**, **RMSE 1.509 kcal/mol**, and a **ROC AUC of 0.669 (95% CI 0.62–0.72)** for "
  "stabilizing/non-stabilizing classification (**Table 3**, Panels A–B). Among individual models, "
  "**Extra Trees — not the ensemble — produced the strongest single-model regression metrics** "
  "(Pearson 0.436; **Table 3**, Panel A).")
P("As in Stage 1, precision and recall traded off with threshold stringency (**Fig. 1**, Panel b; "
  "**Table 3**, Panels B–C). The model operates as a high-precision, low-recall filter at strict "
  "cutoffs: the top-ranked candidates are enriched for true stabilizers, but the majority of "
  "stabilizing mutations are not recovered.")
NOTE("The Stage 2 low recall is a reported property of the model, not an omission. The headline "
     "**0.81 precision corresponds to only 10% recall**; it indicates that a small set of top-ranked "
     "mutations is enriched for stabilizers and is intended to support prioritization of candidates "
     "for experimental testing, not exhaustive recovery of every stabilizing mutation.")

H2("2.4  Cross-stage comparison")
P("Stage 1 achieved a higher independent ROC AUC (**0.732**) than Stage 2 (**0.669**). However, the "
  "two stages were trained and evaluated on **different datasets, prediction targets (temperature "
  "stability vs. folding ΔΔG), class labels, and sample sizes (1,563 vs. 669 records)**. The "
  "comparison is therefore descriptive only: it is **not a controlled head-to-head evaluation, and no "
  "statistical significance test was performed** between the two stages. Any difference in the "
  "reported metrics should be interpreted in light of these differences rather than as evidence that "
  "one stage is intrinsically superior.")
PH("[AUTHOR TO CONFIRM: original BRENDA size (inferred 2,446); positive/negative/ambiguous class "
   "definitions; true-Tm subset n (for r = 0.60); ΔΔG sign convention; how each Stage 2 decision "
   "threshold is applied; that thresholds were not selected on the benchmarks.]")

# ============================================================ 6. DISCUSSION
pdf.add_page()
H1("3.  Discussion")
P("The two-stage framework provides a moderate, leakage-controlled route from whole-protein screening "
  "to mutation-level optimization. Stage 1's independent ROC AUC of 0.732 indicates useful but "
  "imperfect discrimination of thermostable proteins, and its precision–recall trade-off means the "
  "operating point should be chosen to match the intended use (broad recall for discovery vs. high "
  "precision for shortlisting).")
P("Stage 2's independent performance (Pearson 0.390; RMSE 1.509 kcal/mol; ROC AUC 0.669) is best "
  "described as moderate, with pronounced high-precision/low-recall behaviour at strict ΔΔG "
  "thresholds. This makes the model a candidate-prioritization filter rather than a comprehensive "
  "predictor of every stabilizing mutation.")
P("The central methodological contribution is explicit leakage auditing. Removing ~132,000 records — "
  "the great majority (131,479) detectable only through sequence homology — shows that homology-level "
  "contamination, not duplicate records, is the dominant risk to honest benchmarking here. A residual "
  "homology risk remains and is acknowledged rather than dismissed.")
P("Comparisons with prior Tm or ΔΔG models are only meaningful when the benchmark, sign convention, "
  "data split, and metric are matched. We therefore restrict quantitative comparison to genuinely "
  "comparable settings and avoid implying superiority across incomparable evaluations.")
P("Limitations include the moderate correlation on S669, the low recall at high-precision operating "
  "points, the sequence-based (rather than structure-based) scope, residual homology risk, and the "
  "fact that both benchmarks are general stability sets rather than PET-hydrolase-specific assays. "
  "Future work should include PET-enzyme-specific evaluation, the pH-optimum analysis once a training "
  "run is confirmed, and, ultimately, experimental validation before any deployment claim.")
WARN("Benchmark precision may not transfer unchanged to PET-degrading enzymes, which are "
     "under-represented in these general stability sets. In particular, **do not equate 0.93 precision "
     "with 93% experimental success**: benchmark precision is measured at a chosen threshold on "
     "curated data, not in the laboratory.")
PH("[AUTHOR TO CONFIRM: which prior models are cited for comparison and that each comparison is fair "
   "(same benchmark, metric, and independence).]")

# ============================================================ 7. METHODS
pdf.add_page()
H1("4.  Methods")
NOTE("This section is the largest information gap. Every dataset version, threshold, feature, "
     "hyperparameter, and seed below must be verified against the **current** code before submission; "
     "do not carry parameters or features from the earlier 19,071-mutation model.")
H2("4.1  Study design")
P("The framework comprises two stages that use different prediction targets and different data "
  "subsets: Stage 1 (whole-protein Tm) and Stage 2 (mutation-level ΔΔG). The full 1,001,888-record "
  "table did not train both models; records were routed by measurement type (Table 1, Panel B).")
H2("4.2  Source datasets")
P("Records were drawn from seven public sources spanning both measurement types (Table 1, Panel A), "
  "together with the independent BRENDA and S669 benchmarks and UniProt for sequence resolution.")
PH("[AUTHOR TO CONFIRM: for each of Domainome; Tsuboyama mega-scale; Tsuboyama doubles; Meltome Atlas; "
   "ThermoMutDB; FireProtDB; ProDDG/S2648; BRENDA; S669; UniProt — name, primary publication, "
   "repository, version/access date, measurement type, row count, criteria, license, DOI/accession.]")
H2("4.3  Harmonization and sequence resolution")
P("Measurement typing, duplicate/replicate handling, and canonical-sequence resolution from UniProt "
  "were applied to build the staging table.")
PH("[AUTHOR TO CONFIRM: UniProt release; canonical vs isoform handling; number and treatment of "
   "failed mappings; duplicate/replicate policy.]")
H2("4.4  Leakage auditing")
P("Auditing combined an exact-key/mutation-level overlap check with a sequence-homology search; "
  "records matching test proteins were removed, with counts recorded at each step.")
PH("[AUTHOR TO CONFIRM: MMseqs2 version; identity threshold; coverage threshold; e-value; exact "
   "removal logic; counts removed/remaining at each step (total ~132,000; 131,479 by homology).]")
H2("4.5  Targets, features, models, and evaluation")
P("Stage 1 and Stage 2 targets were constructed from the BRENDA class definitions and the ΔΔG "
  "mathematical definition, respectively, with °C and kcal/mol thresholds applied as reported in the "
  "Results. Features, per-model software and hyperparameters, the ensemble construction, the "
  "validation/test design, and the bootstrap procedure (2,000 resamples) must be documented from the "
  "current code.")
PH("[AUTHOR TO CONFIRM: exact current feature set (and, if ESM-2 is used, model/version/layer/pooling/"
   "dimensionality/mutation representation); each model's package version, hyperparameters, class "
   "weighting, early stopping, scaling, and seed; ensemble type and weighting; split/grouping/fold/"
   "seed; bootstrap resampling unit, CI type, and seed; software/environment/hardware; ethics and any "
   "substantive LLM-use disclosure per Nature Portfolio policy.]")

# ============================================================ 8. DATA AVAILABILITY
pdf.add_page()
H1("5.  Data Availability")
P("The processed training table, leakage-audited benchmark datasets, saved predictions, split "
  "identifiers, leakage-audit outputs, and trained model artifacts generated in this study are "
  "available through Zenodo (https://doi.org/10.5281/zenodo.21257369). Source datasets are cited in "
  "Methods and References. Deposit record: \"PET training set\" (Iyer, Ayush), one file, ~2.5 GB.")
WARN("Before submission: the Zenodo record files currently **require login (restricted)** and the "
     "deposit is licensed **GPL-3.0 (a software license)**. Make the files openly accessible (or "
     "provide a reviewer access mechanism) and apply a data license such as **CC BY 4.0**; do not "
     "write \"DOI assigned on publication\" (the repository issues the DOI).")
PH("[AUTHOR TO CONFIRM: the single deposited file contains all listed artifacts and includes a data "
   "dictionary/README; license and redistribution rights for each source dataset (state "
   "restrictions); remove EpHod unless the pH analysis is included.]")

# ============================================================ 9. CODE AVAILABILITY
H1("6.  Code Availability")
P("Source code for data assembly, leakage auditing, model training, evaluation, threshold analysis, "
  "bootstrap confidence-interval estimation, and figure generation is available at "
  "https://github.com/AyushIyer31/PET-Lab and archived on Zenodo at "
  "https://doi.org/10.5281/zenodo.21519961 (release v1.0.0). Key scripts include ensemble_eval.py, "
  "s669_sweep.py, predict_and_sweep.py, and bootstrap_ci.py.")
NOTE("Confirm that archived release v1.0.0 (commit 497c4f5) is the exact code version that produced "
     "the reported results; if not, tag and archive the correct commit. The cited DOI is the "
     "all-versions (concept) DOI; a version-specific DOI is available on the Zenodo record.")
PH("[AUTHOR TO CONFIRM (AY): each named script exists, matches the reported analysis, has no broken "
   "paths or exposed credentials, and runs with documented dependencies; README, license, "
   "environment/requirements file, exact package versions, seeds, hyperparameters, reproduction "
   "commands, and saved predictions are present.]")

# ============================================================ 10-13 back matter
pdf.add_page()
H1("7.  References")
P("Numerical Nature style; sequential; square-bracket in-text citations; one source per number; every "
  "citation matched and every reference cited; no references in the Abstract. Include formal dataset "
  "citations for all seven sources and for the Zenodo data and code deposits.")
PH("[AUTHOR TO CONFIRM: verify authors, title, venue, year, volume, pages/article number, and DOI for "
   "every reference, and that each source supports its attached claim; mark any unverifiable citation "
   "as CITATION NOT VERIFIED — REMOVE OR REPLACE; do not fabricate DOIs.]")

H1("8.  Acknowledgements")
P("Brief acknowledgement of non-author contributors, technical/computational-resource support, and "
  "verified funding, if any.")
PH("[AUTHOR TO CONFIRM: whether Santa Clara University / UC Santa Cruz computing (e.g., the Nautilus "
   "cluster) or mentorship support is acknowledged — exact program/allocation, named individuals, and "
   "their consent; do not add institutions automatically or imply endorsement.]")

H1("9.  Author Contributions")
P("Ayush Iyer contributed to [CONFIRM CRediT ROLES]. James Ponzio contributed to [CONFIRM CRediT "
  "ROLES]. Abhinav Iyer contributed to [CONFIRM CRediT ROLES]. All authors reviewed and approved the "
  "final manuscript.")
PH("[AUTHOR TO CONFIRM: actual CRediT roles per author from project/code/analysis records; author "
   "order; that all authors approve the submission and the contribution statement.]")

H1("10.  Additional Information")
H2("Competing interests")
P("The authors declare no competing interests.")
PH("[ALL AUTHORS TO CONFIRM: financial and non-financial competing interests, including any interest "
   "connected with PET-Lab; do not assume \"none\".]")

# ============================================================ 14. FIGURE LEGENDS
pdf.add_page()
H1("11.  Figure Legends")
P("**Figure 1. Precision as a function of the decision threshold for the two-stage framework on "
  "independent benchmarks.** Precision for the Stage 1 whole-protein melting-temperature model on the "
  "leakage-audited BRENDA benchmark (n = 1,563; 979 positive, 584 negative) across melting-temperature "
  "thresholds (panel a) and for the Stage 2 mutation-level ΔΔG model on S669 (n = 669; 168 stabilizing, "
  "501 non-stabilizing reference cases) across ΔΔG thresholds (panel b). In both panels the horizontal "
  "axis is oriented so that precision increases with more stringent thresholds. Recall, F1, accuracy, "
  "and ROC AUC for the same evaluations are reported in Tables 2–3.")
PH("[AUTHOR TO CONFIRM: positive class per panel (BRENDA stable; S669 stabilizing); that the final "
   "figure is regenerated from saved per-point predictions as a vector file; legend < 350 words.]")

# ============================================================ DISPLAY ITEMS
pdf.add_page()
H1("Display items (Tables 1–3 and Figure 1)")

caption("Table 1.  Composition of the assembled dataset and allocation by measurement type.")
panel("Panel A — Rows by source dataset",
      ["Source dataset", "Rows"],
      [["Domainome (aPCA abundance)", "457,943"],
       ["Tsuboyama et al. 2023 (mega-scale)", "357,155"],
       ["Tsuboyama et al. 2023 (double mutants)", "138,275"],
       ["Meltome Atlas", "27,884"],
       ["ThermoMutDB", "11,520"],
       ["FireProtDB", "6,549"],
       ["ProDDG/S2648", "2,562"],
       ["Total", "1,001,888"]],
      [130, CW - 130])
panel("Panel B — Rows by measurement type and model allocation",
      ["Measurement type", "Rows", "Used to train"],
      [["ΔΔG (mutation)", "508,693", "Stage 2"],
       ["Abundance (proxy)", "457,943", "Not used in this study"],
       ["Tm (whole protein)", "29,654", "Stage 1"],
       ["ΔTm", "5,598", "Not used in this study"],
       ["Total", "1,001,888", "—"]],
      [70, 40, CW - 110])
tnote("Exact record counts; both panels sum to 1,001,888. [AUTHOR TO CONFIRM: dataset versions and "
      "redistribution conditions per source.]")

caption("Table 2.  Stage 1 melting-temperature model — classification on the independent BRENDA "
        "benchmark (1,563 enzymes; 979 positive, 584 negative).")
panel("Panel A — Decision-threshold analysis",
      ["Threshold", "Precision", "Recall", "F1", "Accuracy"],
      [["46 °C", "0.65", "0.98", "0.78", "0.66"],
       ["50 °C", "0.71", "0.77", "0.74", "0.66"],
       ["52 °C", "0.77", "0.63", "0.69", "0.65"],
       ["60 °C", "0.93", "0.39", "0.55", "0.60"],
       ["66 °C", "0.98", "0.32", "0.49", "0.58"]],
      [44, 34, 34, 34, CW - 146])
panel("Panel B — Per-model performance at the 60 °C threshold",
      ["Model", "Precision", "Recall", "F1"],
      [["Extra Trees", "0.963", "0.369", "0.533"],
       ["Random Forest", "0.948", "0.355", "0.517"],
       ["LightGBM", "0.933", "0.395", "0.555"],
       ["CatBoost", "0.927", "0.400", "0.559"],
       ["XGBoost", "0.917", "0.394", "0.551"],
       ["Ensemble", "0.945", "0.388", "0.550"]],
      [60, 40, 40, CW - 140])
tnote("ROC AUC = 0.732 (bootstrap 95% CI 0.708 to 0.755). Precision is reported with recall, F1, and "
      "accuracy so it is not read as overall reliability. [AUTHOR TO CONFIRM: class definitions; "
      "thresholds not selected on this benchmark.]")

caption("Table 3.  Stage 2 mutation-level ΔΔG model — regression and classification on the "
        "independent S669 benchmark (669 mutations; 168 stabilizing, 501 non-stabilizing).")
panel("Panel A — Regression (predicted vs. experimental ΔΔG)",
      ["Model", "Pearson", "Spearman", "RMSE (kcal/mol)"],
      [["Extra Trees", "0.436", "0.461", "1.481"],
       ["XGBoost", "0.382", "0.413", "1.514"],
       ["Random Forest", "0.369", "0.396", "1.523"],
       ["LightGBM", "0.330", "0.364", "1.549"],
       ["CatBoost", "0.324", "0.362", "1.557"],
       ["Multilayer perceptron", "0.298", "0.338", "1.612"],
       ["Ensemble", "0.390", "0.414", "1.509"]],
      [60, 36, 40, CW - 136])
panel("Panel B — Ensemble classification across prediction-decision thresholds",
      ["Threshold", "Precision", "Recall", "F1", "Accuracy"],
      [["0.00 kcal/mol", "0.81", "0.10", "0.18", "0.77"],
       ["+0.25 kcal/mol", "0.51", "0.26", "0.35", "0.75"],
       ["+0.50 kcal/mol", "0.38", "0.46", "0.42", "0.67"],
       ["+1.00 kcal/mol", "0.31", "0.80", "0.45", "0.51"]],
      [50, 34, 34, 34, CW - 152])
panel("Panel C — Per-model performance at the strict (0.00 kcal/mol) threshold",
      ["Model", "Precision", "Recall"],
      [["Random Forest", "0.900", "0.107"],
       ["Extra Trees", "0.786", "0.065"],
       ["XGBoost", "0.735", "0.149"],
       ["LightGBM", "0.625", "0.119"],
       ["CatBoost", "0.429", "0.107"],
       ["Multilayer perceptron", "0.415", "0.202"],
       ["Ensemble", "0.810", "0.101"]],
      [70, 45, CW - 115])
tnote("Ensemble ROC AUC = 0.669 (95% CI 0.62 to 0.72); ensemble regression Pearson = 0.390 "
      "(95% CI 0.32 to 0.46); 2,000 bootstrap resamples. [AUTHOR TO CONFIRM: ΔΔG sign convention; "
      "predicted-stabilizer rule.]")

# --- Figure 1 (embedded image) ---
ensure(CW * 0.476 + 20)
caption("Figure 1.  Precision as a function of the decision threshold (panel a: Stage 1 / BRENDA; "
        "panel b: Stage 2 / S669).")
pdf.image(FIG1, x=M, w=CW)
tnote("Recall, F1, accuracy, and ROC AUC for the same evaluations are in Tables 2–3. Regenerate as a "
      "vector file from saved per-point predictions before submission.")

# ============================================================ 15. MAIN TABLES / 16. SI
pdf.add_page()
H1("12.  Main Tables")
P("Tables 1–3 (with all panels) and Figure 1 are laid out in the Display items section above; four "
  "main display items in total (Fig. 1 + Tables 1–3), within the <= 8 display-item limit. Tables must "
  "be submitted as editable objects (not images), each cited in the text, with consistent decimals, "
  "units in headings, positive classes and abbreviations defined in notes, and validation vs. "
  "independent / regression vs. classification distinguished.")
PH("[AUTHOR TO CONFIRM: table arithmetic; that the per-source counts sum to 1,001,888; do not label "
   "any model 'best' without a stated criterion.]")

H1("13.  Supplementary Information (separate file)")
P("Candidate content: full data-cleaning and per-source inclusion criteria; sequence-resolution "
  "workflow; detailed MMseqs2 commands and leakage-audit thresholds; feature definitions; full "
  "per-model tables; hyperparameter tables; sensitivity analyses. Information essential to the main "
  "claims must remain in the main text.")
PH("[AUTHOR TO CONFIRM: which materials move to SI and that each SI file exists before it is cited; "
   "SI file shows exact title, exact author list, and 'Supplementary Information', with items labeled "
   "Supplementary Table S1 / Supplementary Figure S1 / Supplementary Methods.]")

# ============================================================ OPTIONAL DISPLAY ITEMS
pdf.add_page()
H1("Optional display items (not part of the four main items)")
P("These are optional; add only if useful, keeping figures + tables at eight or fewer. Scientific "
  "Reports discourages decorative schematics and graphical abstracts.")

caption("Optional Figure A.  Two-stage screen-then-optimize workflow (draft schematic).")
y0 = pdf.get_y() + 1
flowbox(M + 35, y0, CW - 70, 11, "Assembled protein-stability data — 1,001,888 records; 7 public sources")
connect(M + CW * 0.28, y0 + 11, y0 + 18); connect(M + CW * 0.72, y0 + 11, y0 + 18)
flowbox(M, y0 + 18, CW / 2 - 4, 11, "Stage 1: whole-protein Tm model — 29,654 Tm records")
flowbox(M + CW / 2 + 4, y0 + 18, CW / 2 - 4, 11, "Stage 2: mutation ΔΔG model — 508,693 ΔΔG records")
connect(M + CW / 4 - 2, y0 + 29, y0 + 36); connect(M + 3 * CW / 4 + 2, y0 + 29, y0 + 36)
flowbox(M, y0 + 36, CW / 2 - 4, 11, "Independent BRENDA benchmark — ROC AUC 0.732 (95% CI 0.708–0.755)")
flowbox(M + CW / 2 + 4, y0 + 36, CW / 2 - 4, 11, "Independent S669 benchmark — ROC AUC 0.669 (95% CI 0.62–0.72)")
pdf.set_y(y0 + 47 + 1)
tnote("Draft schematic using confirmed values. [AUTHOR TO CONFIRM: that this reads as a standard "
      "figure.]")

caption("Optional Figure B.  BRENDA benchmark sample flow after leakage auditing.")
y0 = pdf.get_y() + 1
flowbox(M + 30, y0, CW - 60, 10, "Original BRENDA benchmark: 2,446 enzymes (inferred)")
connect(M + CW / 2, y0 + 10, y0 + 16)
flowbox(M + 30, y0 + 16, CW - 60, 10, "Post-audit set: 2,034 enzymes  (−412 removed; 406 by homology)")
connect(M + CW / 2, y0 + 26, y0 + 32)
flowbox(M + 30, y0 + 32, CW - 60, 10, "Confident binary subset: 1,563  (979 positive, 584 negative; −471 ambiguous)")
pdf.set_y(y0 + 43)
tnote("Counts as reported; 2,446 is inferred (2,034 + 412). Training data separately: ~132,000 "
      "records removed (131,479 by homology). [AUTHOR TO CONFIRM: original benchmark size.]")

caption("Optional Figures C–D.  Predicted vs. experimental scatter plots (regression).")
PH("[FIGURE PLACEHOLDERS — generate C (S669 ΔΔG; ensemble Pearson 0.390, RMSE 1.509) and D (BRENDA "
   "Tm; Pearson 0.54, 0.60 on the true-Tm subset) from saved per-point predictions; do not "
   "reconstruct from summary values.]")
P("Optional Figure E (Supplementary only): per-model precision/recall bar charts would duplicate "
  "Tables 2–3; include in Supplementary Information only if wanted, not as a main figure.")

pdf.output(OUT)
print("wrote", OUT, "pages:", pdf.page_no())
