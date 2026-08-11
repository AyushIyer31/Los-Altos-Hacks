"""Render Section 5 (Results) to a manuscript-style PDF with the caution text set
apart as labeled Warning/Note callout boxes. Content mirrors
manuscript_section5_results.md; exact values from backend/make_perf_figures.py."""
import os
from fpdf import FPDF
import matplotlib as _mpl

FD = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "manuscript_section5_results.pdf")

INK = (20, 20, 20)
GREY = (90, 90, 90)
WARN_BG = (253, 246, 227); WARN_BAR = (176, 122, 0)
NOTE_BG = (233, 241, 242); NOTE_BAR = (26, 94, 102)
M = 18.0


class PDF(FPDF):
    def footer(self):
        self.set_y(-13); self.set_font("S", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 4, "PETase-ML manuscript — Section 5. Results   |   Page %s" % self.page_no(), align="C")


pdf = PDF(format="letter")
pdf.set_margins(M, M, M); pdf.set_auto_page_break(True, 16)
pdf.add_font("S", "", os.path.join(FD, "DejaVuSerif.ttf"))
pdf.add_font("S", "B", os.path.join(FD, "DejaVuSerif-Bold.ttf"))
pdf.add_font("S", "I", os.path.join(FD, "DejaVuSerif-Italic.ttf"))
PW = pdf.w
CW = PW - 2 * M
NX = dict(new_x="LMARGIN", new_y="NEXT")


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
    pdf.ln(1); pdf.set_x(M); pdf.set_font("S", "B", 16); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8.5, t, **NX); pdf.ln(1.5)


def H2(t):
    pdf.ln(2.2); pdf.set_x(M); pdf.set_font("S", "B", 11.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.6, t, **NX); pdf.ln(1.2)


def P(t):
    pdf.set_x(M); pdf.set_font("S", "", 10); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.0, t, markdown=True, **NX); pdf.ln(2.0)


def callout(text, bg, bar):
    lh = 4.8
    n = count_lines(text, CW - 12, 9.5)
    h = n * lh + 6
    if pdf.get_y() + h > pdf.h - 18:
        pdf.add_page()
    y = pdf.get_y()
    pdf.set_fill_color(*bg); pdf.rect(M, y, CW, h, "F")
    pdf.set_fill_color(*bar); pdf.rect(M, y, 2.2, h, "F")
    pdf.set_xy(M + 7, y + 3)
    old = pdf.l_margin; pdf.set_left_margin(M + 7)
    pdf.set_font("S", "", 9.5); pdf.set_text_color(*INK)
    pdf.multi_cell(CW - 12, lh, text, markdown=True, **NX)
    pdf.set_left_margin(old)
    pdf.set_y(y + h); pdf.ln(3)


pdf.add_page()
H1("5. Results")

H2("5.1  Dataset composition and leakage auditing")
P("After harmonization, the multi-source staging table comprised **1,001,888 mutation records** "
  "drawn from seven experimental sources (**Table 1**). Because the framework treats temperature "
  "stability and folding stability as distinct problems, records were routed by measurement type: "
  "**29,654 melting-temperature (Tm) records** were assigned to Stage 1 and **508,693 folding "
  "free-energy change (ΔΔG) records** to Stage 2. Abundance-proxy and ΔTm records were retained in "
  "the table but were **not used** for model training.")
P("To prevent optimistic bias, every candidate training record was audited against the independent "
  "test proteins at two levels: exact-match (identical sequence and mutation) and sequence-homology "
  "(shared homologous background). Auditing removed **approximately 132,000 records**, of which "
  "**131,479 were flagged by homology alone** and would have been invisible to exact-match filtering. "
  "This indicates that homology-level leakage, rather than duplicate records, is the dominant "
  "contamination risk in this setting.")

H2("5.2  Stage 1 — temperature-stability performance (BRENDA)")
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
  "(**Fig. 1**). As the temperature cutoff was made more stringent, precision increased while recall "
  "fell: precision rose from **0.65 (recall 0.98)** at the most permissive cutoff to **0.93 (recall "
  "0.39)** and **0.98 (recall 0.32)** at the strictest cutoffs (**Table 2**). The strict-threshold "
  "regime therefore identifies a small, high-confidence set of stabilizing candidates at the cost of "
  "missing most true positives.")
callout("**WARNING —** A precision value measured at a strict threshold is not an estimate of "
        "experimental success. The 0.93 precision is obtained at **39% recall**; **do not equate 0.93 "
        "precision with 93% experimental success**, and do not read it as overall model reliability. "
        "The corresponding recall, F1, accuracy, and ROC AUC in Table 2 give the complete picture.",
        WARN_BG, WARN_BAR)

H2("5.3  Stage 2 — ΔΔG stabilization performance (S669)")
P("Independent folding-stability performance was assessed on the S669 benchmark of **669 single "
  "mutations (168 stabilizing, 501 non-stabilizing)**. Across **2,000 bootstrap resamples**, the "
  "gradient-boosted ensemble achieved a **Pearson correlation of 0.390 (95% CI 0.32–0.46)** and a "
  "**ROC AUC of 0.669 (95% CI 0.62–0.72)** for stabilizing/non-stabilizing classification (**Table 3**). "
  "Among individual models, **Extra Trees — not the ensemble — produced the strongest single-model "
  "regression metrics**.")
P("As in Stage 1, precision and recall traded off with threshold stringency (**Fig. 1**). At the strict "
  "ΔΔG threshold, precision reached **0.81 at a recall of 0.10**; relaxing the threshold lowered "
  "precision to **0.51 (recall 0.26)**, **0.38 (recall 0.46)**, and **0.31 (recall 0.80)** (**Table 3**). "
  "The model thus operates as a high-precision, low-recall filter at strict cutoffs: the top-ranked "
  "candidates are enriched for true stabilizers, but the majority of stabilizing mutations are not "
  "recovered.")
callout("**NOTE —** The Stage 2 low recall is a reported property of the model, not an omission. The "
        "headline **0.81 precision corresponds to only 10% recall**; it indicates that a small set of "
        "top-ranked mutations is enriched for stabilizers and is intended to support prioritization of "
        "candidates for experimental testing, not exhaustive recovery of every stabilizing mutation.",
        NOTE_BG, NOTE_BAR)

H2("5.4  Cross-stage comparison")
P("Stage 1 achieved a higher independent ROC AUC (**0.732**) than Stage 2 (**0.669**). However, the two "
  "stages were trained and evaluated on **different datasets, prediction targets (temperature stability "
  "vs. folding ΔΔG), class labels, and sample sizes (1,563 vs. 669 records)**. The comparison is "
  "therefore descriptive only: it is **not a controlled head-to-head evaluation, and no statistical "
  "significance test was performed** between the two stages. Any difference in the reported metrics "
  "should be interpreted in light of these differences rather than as evidence that one stage is "
  "intrinsically superior.")

pdf.output(OUT)
print("wrote", OUT, "pages:", pdf.page_no())
