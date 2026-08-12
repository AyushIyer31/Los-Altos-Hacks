"""Detailed, section-by-section Nature Scientific Reports requirements for the
PETase-ML two-stage manuscript. For each section: required CONTENT, Scientific
Reports FORMAT/rules, and common AVOID / rejection triggers.

Grounded in the project's Manuscript Section Guide + Scientific Reports author
conventions. Journal rules change: verify all limits against the current
Scientific Reports 'For authors' / submission-guidelines page before submission.
"""
import os
from fpdf import FPDF
import matplotlib as _mpl

FD = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "PETase_ML_ScientificReports_Section_Requirements.pdf")

INK = (20, 20, 20); GREY = (95, 95, 95)
TEAL = (26, 94, 102)
CBG = (233, 241, 242); CBAR = (26, 94, 102)
CONTENT = (26, 94, 102); FORMAT = (60, 90, 150); AVOID = (176, 60, 40)
M = 18.0


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(9); self.set_x(M); self.set_font("S", "I", 7.5); self.set_text_color(*GREY)
        self.cell(0, 4, "Scientific Reports — section-by-section requirements (PETase-ML)")
        self.set_y(M)

    def footer(self):
        self.set_y(-13); self.set_font("S", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 4, "Page %s" % self.page_no(), align="C")


pdf = PDF(format="letter")
pdf.set_margins(M, M, M); pdf.set_auto_page_break(True, 16)
pdf.add_font("S", "", os.path.join(FD, "DejaVuSerif.ttf"))
pdf.add_font("S", "B", os.path.join(FD, "DejaVuSerif-Bold.ttf"))
pdf.add_font("S", "I", os.path.join(FD, "DejaVuSerif-Italic.ttf"))
PW = pdf.w; CW = PW - 2 * M
NX = dict(new_x="LMARGIN", new_y="NEXT")


def H1(t):
    pdf.ln(2.2); pdf.set_x(M); pdf.set_font("S", "B", 14); pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 7.2, t, **NX); pdf.ln(1.0)


def section(num_title, owner):
    pdf.ln(1.6); pdf.set_x(M); pdf.set_font("S", "B", 11.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.6, num_title, **NX)
    pdf.set_x(M); pdf.set_font("S", "I", 8.2); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.2, owner, **NX); pdf.set_text_color(*INK); pdf.ln(0.6)


def label(t, color):
    pdf.ln(0.4); pdf.set_x(M); pdf.set_font("S", "B", 8.8); pdf.set_text_color(*color)
    pdf.multi_cell(0, 4.4, t, **NX); pdf.set_text_color(*INK)


def bullets(items):
    for it in items:
        pdf.set_x(M); pdf.set_font("S", "", 9.0); pdf.set_text_color(*INK)
        pdf.cell(4.5, 4.4, "•")
        old = pdf.l_margin; pdf.set_left_margin(M + 5.5); pdf.set_x(M + 5.5)
        pdf.multi_cell(CW - 5.5, 4.4, it, markdown=True, **NX)
        pdf.set_left_margin(old)
    pdf.ln(0.6)


def P(t, size=9.3):
    pdf.set_x(M); pdf.set_font("S", "", size); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.7, t, markdown=True, **NX); pdf.ln(1.4)


def callout(text):
    lh = 4.5
    plain = text.replace("**", "")
    pdf.set_font("S", "", 9.0)
    words = plain.split(); line = ""; n = 1
    for w in words:
        t = (line + " " + w).strip()
        if pdf.get_string_width(t) <= CW - 12 - 1:
            line = t
        else:
            n += 1; line = w
    h = n * lh + 6
    if pdf.get_y() + h > pdf.h - 16:
        pdf.add_page()
    y = pdf.get_y()
    pdf.set_fill_color(*CBG); pdf.rect(M, y, CW, h, "F")
    pdf.set_fill_color(*CBAR); pdf.rect(M, y, 2.2, h, "F")
    pdf.set_xy(M + 7, y + 3)
    old = pdf.l_margin; pdf.set_left_margin(M + 7)
    pdf.set_font("S", "", 9.0); pdf.set_text_color(*INK)
    pdf.multi_cell(CW - 12, lh, text, markdown=True, **NX)
    pdf.set_left_margin(old); pdf.set_y(y + h); pdf.ln(3)


# ============================================================ COVER
pdf.add_page()
pdf.ln(16)
pdf.set_font("S", "B", 18); pdf.set_text_color(*TEAL)
pdf.multi_cell(0, 9, "Nature Scientific Reports", align="C", **NX)
pdf.multi_cell(0, 9, "Section-by-Section Requirements", align="C", **NX)
pdf.ln(2)
pdf.set_font("S", "", 11); pdf.set_text_color(*INK)
pdf.multi_cell(0, 5.6, "For the PETase-ML two-stage, leakage-audited stability manuscript", align="C", **NX)
pdf.ln(3)
pdf.set_font("S", "I", 9); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.6, "Owners: AB = Abhinav Iyer, JA = James Ponzio, AY = Ayush Iyer", align="C", **NX)
pdf.ln(6)
pdf.set_text_color(*INK)
callout("How to read this: each section lists (1) CONTENT that must be present, (2) FORMAT and "
        "Scientific Reports rules, and (3) AVOID / common rejection triggers. Journal rules change — "
        "verify every numeric limit against the current Scientific Reports 'For authors' page before "
        "submission.")

# ============================================================ GENERAL
H1("General compliance (applies across the manuscript)")
label("Article structure & order", CONTENT)
bullets([
    "Research Article. Section order: Title page; Abstract; (Keywords); Introduction; Results; "
    "Discussion; Methods; Data Availability; Code Availability; References; Acknowledgements; Author "
    "Contributions; Additional Information (Competing interests); Figure Legends; Main Tables.",
    "Methods appear **after** the Discussion, not near the front.",
    "Results and Discussion may be combined, but this manuscript keeps them separate.",
    "Supplementary Information is a **separate file**, not part of the main PDF."])
label("Length & display items", FORMAT)
bullets([
    "Title <= 20 words. Abstract <= 200 words, unstructured, no citations. Up to 6 keywords.",
    "No hard main-text word limit, but concision is expected (project target ~4,500 words, excluding "
    "Abstract, Methods, References, and legends).",
    "Keep figures + tables few; current plan is 4 (Tables 1-3 + Figure 1). Treat 8 as an upper bound.",
    "Main tables must be **editable objects, not images**; figures as high-resolution vector where "
    "possible."])
label("Mandatory declarations", FORMAT)
bullets([
    "Data Availability (mandatory); Code Availability (custom code is central here); Author "
    "Contributions; Competing Interests (under Additional Information); Acknowledgements.",
    "Ethics/consent statements only if applicable (this is a computational study on public data).",
    "Disclose substantive LLM use per Nature Portfolio policy."])
label("Avoid (house-style rejection triggers)", AVOID)
bullets([
    "No footnotes, graphical abstract, schemes, text boxes, or appendices in the main manuscript.",
    "No references in the Abstract. No undefined abbreviations in Title/Abstract.",
    "Do not mix the older 19,071-mutation results with the current two-stage results anywhere."])

# ============================================================ 1 TITLE
H1("1.  Title page")
section("Title page", "Owner: AY  (title = AB)")
label("Content — must include", CONTENT)
bullets([
    "Final title (<= 20 words), no undefined abbreviations.",
    "Full author names: Ayush Iyer, James Ponzio, Abhinav Iyer.",
    "Superscript-linked affiliation for each author, **including country**.",
    "Corresponding author marked with an asterisk + email (Ayush Iyer, iyer.ayush31@gmail.com)."])
label("Format & SR rules", FORMAT)
bullets([
    "Affiliations numbered and superscript-linked to author names.",
    "ORCID iDs are entered via the submission system when requested (not typed into the title page)."])
label("Avoid", AVOID)
bullets([
    "Do not invent ORCID iDs or affiliations. Do not exceed 20 words or use jargon abbreviations.",
    "Do not describe the data as PETase-specific in the title; it is a general stability framework."])
callout("Open: affiliation (institution + country) for all three authors; confirm corresponding-"
        "author email before submission.")

# ============================================================ 2 ABSTRACT
H1("2.  Abstract")
section("Abstract", "Owner: AB")
label("Content — must include (in this order)", CONTENT)
bullets([
    "Problem; gap; objective; design; **Stage 1 independent result**; **Stage 2 independent result**; "
    "interpretation; scope limitation.",
    "1,001,888 records from seven public sources; measurement types routed to two stages (Stage 1 = "
    "whole-protein Tm; Stage 2 = mutation-level ΔΔG).",
    "Leakage auditing (exact + sequence-homology).",
    "Stage 1 (BRENDA) ROC AUC 0.732 (95% CI 0.708-0.755); Stage 2 (S669) ROC AUC 0.669 (95% CI "
    "0.62-0.72), ensemble Pearson 0.390 (95% CI 0.32-0.46).",
    "End with a measured interpretation (prioritization tool, not validated solution)."])
label("Format & SR rules", FORMAT)
bullets([
    "<= 200 words; unstructured (no subheadings); **no citations**; no undefined abbreviations.",
    "Prioritize independent evaluation over internal validation."])
label("Avoid", AVOID)
bullets([
    "Do not describe ROC AUC as precision-recall AUC. Do not claim industrial/wet-lab readiness.",
    "Do not present internal validation (RMSE ~6.5 °C, r ~0.80) as if it were independent testing."])

# ============================================================ 3 KEYWORDS
H1("3.  Keywords")
section("Keywords", "Owner: AB")
label("Content", CONTENT)
bullets([
    "Choose <= 6 from: protein stability; machine learning; enzyme engineering; thermostability; "
    "mutation-effect prediction; melting temperature; ΔΔG prediction; PET-degrading enzymes."])
label("Format & SR rules", FORMAT)
bullets(["No promotional terms; avoid repeating every word already in the title."])

# ============================================================ 4 INTRODUCTION
H1("4.  Introduction")
section("Introduction", "Owner: AB")
label("Content — must include (~4-6 paragraphs)", CONTENT)
bullets([
    "P1 — plastic/PET context (every statistic from an authoritative primary source; state whether "
    "production/waste/municipal/PET; do not mix tonnes/tons).",
    "P2 — enzymatic PET degradation (PETase, cutinases); thermostability is **not** the sole "
    "determinant of catalytic performance.",
    "P3 — define Tm and mutation-level ΔΔG; distinguish whole-protein screening from mutation-level "
    "optimization (related, not interchangeable).",
    "P4 — the leakage problem (exact + homology); do not claim all prior models leak.",
    "Final — gap + objective: a general, leakage-audited two-stage screen-then-optimize framework "
    "evaluated on independent BRENDA and S669."])
label("Format & SR rules", FORMAT)
bullets([
    "Every external factual claim carries a citation. Do not report detailed results here.",
    "Numbered references in order of first appearance."])
label("Avoid", AVOID)
bullets([
    "No experimental/wet-lab/field/clinical claims. No PETase-specific description of the training data.",
    "No unsourced statistics."])

# ============================================================ 5 RESULTS
H1("5.  Results")
section("Results", "Owner: JA")
label("Content — must include (four subheadings)", CONTENT)
bullets([
    "Dataset/leakage: 1,001,888 rows / 7 sources; 29,654 Tm to Stage 1; 508,693 ΔΔG to Stage 2; "
    "abundance + ΔTm not used; ~132,000 removed (131,479 by homology). Cite **Table 1**.",
    "Stage 1: internal RMSE ~6.5 °C and validation r ~0.80 (development only); independent BRENDA ROC "
    "AUC 0.732; regression r 0.54 (0.60 on true-Tm subset); sample flow (2,034 -> 1,563; 979/584); "
    "threshold trade-off. Cite **Table 2** and **Fig. 1**.",
    "Stage 2: 669 mutations (168/501); ensemble Pearson 0.390, ROC AUC 0.669 (2,000 bootstraps); Extra "
    "Trees strongest single-model regression; high-precision/low-recall at strict threshold. Cite "
    "**Table 3** and **Fig. 1**.",
    "Comparison: Stage 1 ROC AUC > Stage 2, but different datasets/labels/sizes; descriptive only, no "
    "significance test."])
label("Format & SR rules", FORMAT)
bullets([
    "Connected objective prose. **Separate internal validation from independent testing**, and "
    "**regression from classification**.",
    "Report exact values; present precision alongside recall/F1/accuracy/ROC AUC.",
    "Every table and figure cited in the text in order."])
label("Avoid", AVOID)
bullets([
    "Do not hide Stage 2 low recall. Do not equate 0.93 precision with 93% experimental success.",
    "Do not label a model 'best' without a stated criterion; note Extra Trees (not the ensemble) gives "
    "the strongest single-model regression metrics."])
callout("Open: original BRENDA size (inferred 2,446); class definitions; true-Tm subset n (for "
        "r = 0.60); ΔΔG sign convention; how each Stage 2 threshold is applied; that thresholds were "
        "not selected on the benchmarks.")

# ============================================================ 6 DISCUSSION
H1("6.  Discussion")
section("Discussion", "Owner: AB")
label("Content — must include (~7 paragraphs)", CONTENT)
bullets([
    "Main findings; meaning of Stage 1 (0.732) and its precision-recall trade-off; meaning of Stage 2 "
    "(0.390 / 0.669) as moderate with high-precision/low-recall behaviour.",
    "Significance of leakage auditing and residual homology risk.",
    "Fair comparison with prior Tm/ΔΔG models only when benchmark, sign convention, split, and metric "
    "match.",
    "A full limitations paragraph; measured future work (incl. pH once trained; PET-specific "
    "evaluation; experimental validation) and conclusion."])
label("Format & SR rules", FORMAT)
bullets(["Interpretation, not repetition of Results. No promotional language."])
label("Avoid", AVOID)
bullets([
    "Do not equate 0.93 precision with 93% experimental success; state that benchmark precision may "
    "not transfer to PET-degrading enzymes.",
    "No claims beyond the evidence (deployment, environmental impact)."])

# ============================================================ 7 METHODS
H1("7.  Methods")
section("Methods  (largest information gap)", "Owner: AY")
label("Content — must include (reproducible detail)", CONTENT)
bullets([
    "Study design (two stages, different targets/subsets; full table did not train both models).",
    "Each source dataset: name, primary publication, repository, version/access date, measurement "
    "type, row count, inclusion/exclusion criteria, license, DOI/accession.",
    "Harmonization; measurement typing; duplicate/replicate handling; UniProt sequence resolution "
    "(release, canonical vs isoform, failed mappings).",
    "Leakage auditing operationalized: exact-key + mutation overlap; MMseqs2 version; identity, "
    "coverage, e-value thresholds; removal logic; counts at each step.",
    "Target construction (BRENDA class definitions; ΔΔG definition; how °C and kcal/mol thresholds "
    "applied).",
    "Features from current code (if ESM-2: model/version/layer/pooling/dimensionality/mutation "
    "representation).",
    "Per-model software, package versions, hyperparameters, class weighting, early stopping, scaling, "
    "seeds; ensemble type and weighting.",
    "Validation/test design (splits, grouping, folds, seed); metric definitions; bootstrap details "
    "(2,000 resamples, unit, CI type, seed); software/environment/hardware.",
    "Ethics statement (computational, public data) and LLM-use disclosure if used substantively."])
label("Format & SR rules", FORMAT)
bullets([
    "Sufficient detail for **independent reproduction**; no main-text word limit; clear subsections.",
    "Methods placed after Discussion."])
label("Avoid", AVOID)
bullets(["Do not carry features or parameters from the 19,071-mutation model; verify everything from "
         "the current code."])

# ============================================================ 8 DATA AVAILABILITY
H1("8.  Data Availability")
section("Data Availability", "Owner: JA")
label("Content — must include", CONTENT)
bullets([
    "Statement that the processed training table, leakage-audited benchmarks, saved predictions, split "
    "identifiers, leakage-audit outputs, and trained models are available via Zenodo "
    "(10.5281/zenodo.21257369); source datasets cited in Methods/References.",
    "Deposit: 'PET training set' (Iyer, Ayush), one file, ~2.5 GB."])
label("Format & SR rules", FORMAT)
bullets([
    "Mandatory; placed after Methods, before Code Availability.",
    "The repository issues the DOI — do not write 'DOI assigned on publication'."])
label("Avoid / fix before submission", AVOID)
bullets([
    "Deposit currently **login-restricted** — make open or provide a reviewer access mechanism.",
    "Deposit currently **GPL-3.0** (a software license) — apply a data license (e.g., CC BY 4.0).",
    "Verify license and redistribution rights for each source dataset; remove EpHod unless pH included."])

# ============================================================ 9 CODE AVAILABILITY
H1("9.  Code Availability")
section("Code Availability", "Owner: JA (technical verification AY)")
label("Content — must include", CONTENT)
bullets([
    "Statement: code for data assembly, leakage auditing, training, evaluation, threshold analysis, "
    "bootstrap CIs, and figure generation is at github.com/AyushIyer31/PET-Lab and archived on Zenodo "
    "(10.5281/zenodo.21519961, release v1.0.0).",
    "Name verifiable scripts (ensemble_eval.py, s669_sweep.py, predict_and_sweep.py, bootstrap_ci.py)."])
label("Format & SR rules", FORMAT)
bullets([
    "Separate section after Data Availability, before References.",
    "GitHub alone is not a permanent archive — a versioned Zenodo (or equivalent) archive is required."])
label("Avoid", AVOID)
bullets([
    "Confirm the archived release/commit is the exact code that produced the results.",
    "No broken paths or exposed credentials; include README, license, environment file, seeds, and "
    "reproduction commands."])

# ============================================================ 10 REFERENCES
H1("10.  References")
section("References", "Owner: AB")
label("Content & format", CONTENT)
bullets([
    "Numbered, cited sequentially; one source per number; every citation matched and every reference "
    "cited; no references in the Abstract.",
    "Verify each: authors, title, venue, year, volume, pages/article number, DOI, and that the source "
    "supports the claim. Prefer primary sources.",
    "Include formal dataset citations for all seven sources and for the Zenodo data + code deposits."])
label("Avoid", AVOID)
bullets(["Mark any unverifiable citation 'CITATION NOT VERIFIED — REMOVE OR REPLACE'; do not fabricate "
         "DOIs. (Confirm SR in-text style: superscript vs square-bracket, per current guidelines.)"])

# ============================================================ 11 ACK
H1("11.  Acknowledgements")
section("Acknowledgements", "Owner: AB")
label("Content", CONTENT)
bullets(["Non-author contributors, technical/computational-resource support, and verified funding."])
label("Avoid", AVOID)
bullets([
    "Brief; no effusive language; not for authors, reviewers, editors, or competing interests.",
    "Only name Santa Clara University / UC Santa Cruz computing (e.g., Nautilus) or mentors with the "
    "exact program/allocation, named individuals, and their consent; do not imply endorsement."])

# ============================================================ 12 AUTHOR CONTRIBUTIONS
H1("12.  Author Contributions")
section("Author Contributions", "Owner: AY")
label("Content — must include", CONTENT)
bullets([
    "Every author listed with roles using the **CRediT** taxonomy.",
    "Use distinct forms for shared initials (Ayush Iyer / Abhinav Iyer / J.P.).",
    "Close with: 'All authors reviewed and approved the final manuscript.'"])
label("CRediT roles to assign per author (choose all that apply)", FORMAT)
bullets([
    "Conceptualization · Data curation · Formal analysis · Funding acquisition · Investigation · "
    "Methodology · Project administration · Resources · Software · Supervision · Validation · "
    "Visualization · Writing – original draft · Writing – review & editing."])
label("Avoid", AVOID)
bullets(["Do not infer roles from who owns which manuscript section; use the actual project record.",
         "Confirm author order and that all authors approve the statement."])
callout("ACTION: paste your role assignments for Ayush Iyer, James Ponzio, and Abhinav Iyer and they "
        "will be written into the manuscript's Author Contributions section verbatim.")

# ============================================================ 13 ADDITIONAL INFO
H1("13.  Additional Information (Competing interests)")
section("Additional Information", "Owner: JA  (Competing interests drafted by AB)")
label("Content", CONTENT)
bullets([
    "Heading 'Additional Information' with a single 'Competing interests' subsection.",
    "If none: 'The authors declare no competing interests.' Otherwise an explicit, author-linked "
    "declaration matching the submission system."])
label("Avoid", AVOID)
bullets([
    "No publisher boilerplate, reprints line, publisher's note, peer-review line, correspondence "
    "sentence, ORCID list, or SI URL here (the corresponding author belongs on the title page).",
    "Do not assume 'none' — every author must confirm financial and non-financial interests, including "
    "any interest connected with PET-Lab."])

# ============================================================ 14 FIGURE LEGENDS
H1("14.  Figure Legends")
section("Figure Legends", "Owner: AY")
label("Content — Figure 1 legend must define", CONTENT)
bullets([
    "Title; each panel (a = Stage 1 BRENDA, n = 1,563; b = Stage 2 S669, n = 669); dataset; sample "
    "size; model; positive class (BRENDA stable; S669 stabilizing); axes; units; symbols/lines; error "
    "bands; resamples; thresholds; abbreviations; exclusions.",
    "State that recall, F1, accuracy, and ROC AUC are in Tables 2-3."])
label("Format & SR rules", FORMAT)
bullets(["Complete legend understandable without the main text; < 350 words; no Results/Discussion "
         "interpretation."])
label("Avoid", AVOID)
bullets(["Regenerate the final figure as a vector file from saved per-point predictions, not from "
         "summary values."])

# ============================================================ 15 MAIN TABLES
H1("15.  Main Tables")
section("Main Tables", "Owner: AY")
label("Content", CONTENT)
bullets([
    "Table 1 — dataset composition + measurement-type allocation (Panels A, B).",
    "Table 2 — Stage 1 BRENDA (Panel A threshold sweep; Panel B per-model at 60 °C).",
    "Table 3 — Stage 2 S669 (Panel A regression; Panel B threshold sweep; Panel C strict threshold)."])
label("Format & SR rules", FORMAT)
bullets([
    "Editable objects (not images); each cited in text; consistent decimals; units in headings; "
    "abbreviations + positive classes defined in notes; sample sizes stated.",
    "Distinguish validation vs independent and regression vs classification."])
label("Avoid", AVOID)
bullets(["Do not label 'best' without a stated criterion; confirm arithmetic and that figures + tables "
         "total <= 8 (currently 4)."])

# ============================================================ 16 SI
H1("16.  Supplementary Information (separate file)")
section("Supplementary Information", "Owner: AY")
label("Content — candidates", CONTENT)
bullets([
    "Full data-cleaning + per-source inclusion criteria; sequence-resolution workflow; detailed "
    "MMseqs2 commands and leakage-audit thresholds; feature definitions; full per-model tables; "
    "hyperparameter tables; sensitivity analyses."])
label("Format & SR rules", FORMAT)
bullets([
    "Separate submission file; first page shows exact title, exact author list, and 'Supplementary "
    "Information'.",
    "Label items Supplementary Table S1 / Supplementary Figure S1 / Supplementary Methods; cite each "
    "in the main text; SI legends live in the SI file."])
label("Avoid", AVOID)
bullets(["Do not move information essential to the main claims entirely into SI; ensure each SI file "
         "exists before it is cited."])

pdf.output(OUT)
print("wrote", OUT, "pages:", pdf.page_no())
