"""Generate an Overleaf-ready LaTeX proposal (black & white, no byline)."""
import json
panel = json.load(open("final_panel.json"))

def seqblock(seq):
    lines=[]
    for i in range(0,len(seq),60):
        lines.append(f"{i+1:>3}  {seq[i:i+60]}")
    return "\n".join(lines)

rows=""
for p in panel:
    muts="+".join(p["muts"]) if p["muts"] else "none (wild-type)"
    dtm = f"${p['pred_dTm']:+.2f}$"
    ddg = f"${p['pred_ddg']:+.3f}$"
    rows += f"{p['name']} & {muts} & {p['kind']} & {dtm} & {ddg} \\\\\n"

def dnablock(seq):
    return "\n".join(f"{i+1:>4}  {seq[i:i+60]}" for i in range(0,len(seq),60))

seqs=""
for p in panel:
    lbl=p["name"]+("  ("+"+".join(p["muts"])+")" if p["muts"] else "  (wild-type)")
    seqs += ("\\paragraph{"+lbl+"}\n"
             "\\noindent\\textit{Protein (290 aa):}\n\\begin{verbatim}\n"+seqblock(p["seq"])+"\n\\end{verbatim}\n"
             "\\medskip\n\\noindent\\textit{DNA -- E.\\,coli-optimized coding sequence (873 bp, incl. stop):}\n"
             "\\begin{verbatim}\n"+dnablock(p["dna"])+"\n\\end{verbatim}\n\\bigskip\n\n")

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[T1]{fontenc}
\pagestyle{plain}

\title{Experimental Validation of Machine-Learning-Designed\\
Thermostable IsPETase Variants\\[0.4em]
{\large A Proposal Requesting Wet-Lab Access}}
\author{}
\date{}

\begin{document}
\maketitle

\section{Executive Summary}
We request access to a molecular-biology / protein-biochemistry wet lab to experimentally
validate five computationally designed variants of the plastic-degrading enzyme IsPETase.
Enzymatic recycling of PET plastic works best at high temperature, but wild-type IsPETase
falls apart near $48\,^{\circ}$C, which limits its industrial use.

\emph{What we did computationally, in brief:} we built an artificial-intelligence tool that has
learned, from large public databases of laboratory measurements, how small changes to a
protein's sequence affect how much heat it can withstand. We gave this tool the IsPETase
sequence; it evaluated every possible single amino-acid change and ranked the ones most likely
to make the enzyme more heat-stable. From that ranked list we selected five promising designs.

We now propose to produce these five proteins in the lab, alongside the original enzyme as a
control, and measure how much heat each one can take, directly checking whether the tool's
predictions hold. The experiment is small (six proteins), inexpensive, and uses standard
techniques; its outcome is the critical evidence needed before the tool is used to guide real
enzyme-engineering campaigns, for which we have secured downstream deployment interest.

\section{Background and Significance}
Polyethylene terephthalate (PET) is among the most abundant plastics, and enzymatic recycling,
using enzymes to break PET into reusable monomers, is a leading route to a circular
plastics economy. PET degrades fastest at elevated temperature, where the polymer softens and
becomes accessible. The bottleneck is biological: naturally occurring PET hydrolases such as
IsPETase (from \emph{Piscinibacter} / \emph{Ideonella sakaiensis}, UniProt A0A0K8P6T7) lose
their fold and activity well below the useful temperature range. Engineering enzymes that
remain stable and active under heat is therefore the central problem, and improving melting
temperature ($T_\mathrm{m}$) is the established, measurable lever for doing so.

Our broader mission is to build a condition-aware enzyme-design pipeline: a system that
engineers protein variants able to withstand the many environmental hurdles that prevent enzymes
from functioning in real-world settings. Industrial and environmental conditions are rarely
ideal: a useful enzyme must tolerate not only high temperature but also extremes of pH,
calcium-ion concentration, ionic strength, and related stresses, often at the same time. Our
primary focus is therefore to design optimized proteins that remain stable and active across this
full range of conditions, rather than under a single idealized setting. The present experiment
addresses the first and most fundamental of these axes: it is designed primarily to test the
pipeline's thermostability (temperature) predictions, establishing the validation foundation on
which the other conditions will build.

Testing mutations experimentally is slow and costly, so exhaustive screening is infeasible.
Our pipeline addresses this by ranking candidate mutations computationally, so only a short,
high-confidence shortlist reaches the bench. This proposal is that final, essential step:
confirming in the lab that the model's top predictions actually stabilize the enzyme.

\section{How the Candidates Were Generated}
The design pipeline pairs two models built on ESM-2 protein-language-model embeddings: a
screening model that predicts an enzyme's $T_\mathrm{m}$ from sequence, and a mutation model
that predicts the stability change ($\Delta\Delta G$) of a point mutation. The mutation model
was trained on a leakage-audited dataset of over one million stability measurements from seven
public sources, with independent benchmarks filtered for sequence-homology overlap so reported
accuracy reflects generalization to novel enzymes.

To design the candidates, we scored all $\sim$4{,}900 allowed single-point mutations of
wild-type IsPETase at a $50\,^{\circ}$C target temperature. We excluded the signal peptide
(residues 1--27), the catalytic triad (Ser160/Asp206/His237), and the structural disulfide
cysteines, so that predicted-stabilizing mutations cannot compromise catalysis or the fold.
From the ranked output we selected five designs that span distinct positions and mechanisms,
plus a combined variant that stacks the three most compatible substitutions.

\emph{Note on effect sizes: the mutation model is a sequence-based baseline; its predicted
per-mutation shifts are modest ($\sim$0.5--2$\,^{\circ}$C) and are reported here as predictions
to be tested, not as established facts. Confirming whether these rankings hold, and whether
stacking is additive, is precisely the purpose of the requested experiment.}

\section{Candidate Proteins to Test}
Table~\ref{tab:candidates} lists the six proteins we propose to test: the wild-type control
and five model-designed variants. Positions use full-length UniProt numbering (A0A0K8P6T7).
Full-length amino-acid sequences for gene synthesis are provided in the Appendix.

\begin{table}[ht]
\centering
\caption{Candidate proteins for wet-lab validation. Predicted values are model outputs at
$50\,^{\circ}$C; a negative $\Delta\Delta G$ indicates stabilization. IsPETase-3M stacks
Y219E, K177I, and R123I, and its predicted $\Delta T_\mathrm{m}$ is the additive estimate.}
\label{tab:candidates}
\begin{tabular}{llccc}
\toprule
Construct & Mutation(s) & Type & Pred.\ $\Delta T_\mathrm{m}$ ($^{\circ}$C) & Pred.\ $\Delta\Delta G$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}

\section{Experimental Plan (What We Will Do in the Lab)}
\subsection{Protein production}
\begin{itemize}
\item Synthesize codon-optimized genes for all six constructs (mature domain, residues
28--290) with a C-terminal His$_6$ tag, cloned into a pET-based expression vector.
\item Express in \emph{E.\ coli} BL21(DE3): IPTG induction, $\sim$18$\,^{\circ}$C overnight,
for soluble protein.
\item Purify by Ni-NTA affinity chromatography; assess purity by SDS-PAGE; buffer-exchange
into assay buffer. Expected yield is sufficient for all assays from standard shake-flask
cultures.
\end{itemize}
\subsection{Thermostability measurement (primary readout)}
\begin{itemize}
\item Nano-DSF / thermal-shift assay: ramp each purified variant from 25--95$\,^{\circ}$C and
record its melting temperature ($T_\mathrm{m}$). This directly measures each variant's
$T_\mathrm{m}$ and yields the experimental $\Delta T_\mathrm{m}$ versus wild-type, the
quantity our model predicts.
\item Thermal challenge at $50\,^{\circ}$C: incubate each variant at $50\,^{\circ}$C for
30--60 min, cool, then measure residual activity. Wild-type ($T_\mathrm{m}\approx48\,^{\circ}$C)
is expected to lose activity; stabilized variants should retain it.
\end{itemize}
\subsection{Activity assay}
\begin{itemize}
\item Esterase activity on a \emph{para}-nitrophenyl ester substrate (colorimetric,
plate-reader) as a fast activity proxy, and/or PET-film/powder hydrolysis quantified by
release of degradation products (absorbance or HPLC).
\end{itemize}
\subsection{Success criteria}
\begin{itemize}
\item Primary: the measured $T_\mathrm{m}$ ranking of the variants correlates with the model's
predicted ranking; the stacked variant (IsPETase-3M) shows the largest $T_\mathrm{m}$ increase.
\item Any variant with $T_\mathrm{m}$ reproducibly above wild-type validates the pipeline;
nano-DSF resolves sub-degree shifts, so even the modest predicted effects are testable.
\end{itemize}

\section{What We Request From the Host Lab}
\begin{table}[ht]
\centering
\caption{Resources requested from the host laboratory.}
\label{tab:resources}
\begin{tabular}{@{}l p{0.66\textwidth}@{}}
\toprule
Item & What we request \\
\midrule
Bench access & BSL-1 space for routine \emph{E.\ coli} work over $\sim$6--8 weeks. \\
\addlinespace
Molecular biology & Competent \emph{E.\ coli} BL21(DE3), an expression vector, standard cloning
reagents, media, IPTG, and antibiotics. Genes can be outsourced as synthetic fragments. \\
\addlinespace
Protein purification & Ni-NTA resin / gravity columns or an FPLC/AKTA system; SDS-PAGE apparatus. \\
\addlinespace
Key instrument & A nano-DSF (e.g.\ Prometheus) or a qPCR machine for dye-based thermal-shift
assays, the central measurement. \\
\addlinespace
Plate reader & UV/Vis plate reader for the \emph{para}-nitrophenyl activity assay. \\
\addlinespace
Optional & HPLC access for PET degradation-product quantification; PET film/powder substrate. \\
\addlinespace
Consumables & Plates, cuvettes/capillaries, buffers, centrifuge tubes. \\
\bottomrule
\end{tabular}
\end{table}
We provide the enzyme designs, sequences, analysis, and hands-on labor; we ask the host lab for
supervised access, the instruments above, and routine consumables. We are glad to work under
the host's safety and IP arrangements.

\section{Timeline (approximately 6--8 weeks)}
\begin{itemize}
\item \textbf{Weeks 1--2:} gene synthesis and cloning of the six constructs.
\item \textbf{Weeks 3--4:} expression and Ni-NTA purification; purity check.
\item \textbf{Weeks 5--6:} nano-DSF $T_\mathrm{m}$ measurement and $50\,^{\circ}$C
thermal-challenge / activity assays.
\item \textbf{Weeks 7--8:} data analysis; compare measured vs.\ predicted $T_\mathrm{m}$; report.
\end{itemize}

\section{Expected Impact}
A positive result, measured $T_\mathrm{m}$ increases consistent with the predicted ranking,
validates that our leakage-audited ML pipeline can propose genuinely stabilizing mutations for
a real, industrially relevant enzyme. This is the evidence required before applying the pipeline
at scale to engineer heat-tolerant PET-degrading enzymes for deployment. Because the experiment
is small, standard, and low-cost, it offers an unusually high ratio of scientific value to bench
effort, and provides an excellent, self-contained project for a student collaborator.

\section{Key References}
\begin{enumerate}
\item Yoshida S.\ et al.\ (2016). A bacterium that degrades and assimilates poly(ethylene
terephthalate). \emph{Science} 351:1196--1199.
\item Son H.F.\ et al.\ (2019). Rational protein engineering of thermo-stable PETase.
\emph{ACS Catal.} 9:3519--3526.
\item Lin Z.\ et al.\ (2023). Evolutionary-scale prediction of atomic-level protein structure
(ESM-2). \emph{Science} 379:1123--1130.
\item Tsuboyama K.\ et al.\ (2023). Mega-scale experimental analysis of protein folding
stability. \emph{Nature} 620:434--444.
\item UniProt A0A0K8P6T7 (PETH\_PISS1), Poly(ethylene terephthalate) hydrolase.
\end{enumerate}

\appendix
\section{Protein and DNA Sequences (for gene synthesis)}
Full-length constructs (290 aa). For expression, the mature domain (residues 28--290) is used
with a C-terminal His$_6$ tag. Mutated positions are relative to UniProt A0A0K8P6T7. DNA is a
codon-optimized \emph{E.\,coli} coding sequence (873 bp incl.\ stop); a commercial optimizer may
further tune GC content and remove restriction sites before synthesis.

\small
""" + seqs + r"""\normalsize

\end{document}
"""

out="/Users/admin/Documents/PET - Lab/IsPETase_WetLab_Proposal.tex"
open(out,"w").write(tex)
print("WROTE", out, f"({len(tex)} chars)")
