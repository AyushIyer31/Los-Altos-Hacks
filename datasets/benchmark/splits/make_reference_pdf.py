"""Generate an APA-7 reference page (PDF) for the dataset sources, split by
dataset partition: Training, In-Distribution Test, Independent Test.

Uses a custom word-wrapper so words never break mid-token and inline italics
(journal title + volume) wrap cleanly with a proper hanging indent."""
from fpdf import FPDF

OUT = "datasets/benchmark/splits/data_source_references_APA.pdf"
LH = 6.4          # line height
HANG = 9.0        # hanging indent (mm)
FS = 11           # body font size

# each reference = list of (text, style) segments; style '' normal, 'I' italic
PMBD = [
    ("Gan, Z., & Zhang, H. (2019). PMBD: A comprehensive plastics microbial "
     "biodegradation database. ", ""),
    ("Database, 2019", "I"),
    (", baz119. https://doi.org/10.1093/database/baz119", ""),
]
ZRIMEC = [
    ("Zrimec, J., Kokina, M., Jonasson, S., Zorrilla, F., & Zelezniak, A. (2021). "
     "Plastic-degrading potential across the global microbiome correlates with "
     "recent pollution trends. ", ""),
    ("mBio, 12", "I"),
    ("(5), e02155-21. https://doi.org/10.1128/mbio.02155-21", ""),
]
UNIPROT = [
    ("The UniProt Consortium. (2025). UniProt: The Universal Protein Knowledgebase "
     "in 2025. ", ""),
    ("Nucleic Acids Research, 53", "I"),
    ("(D1), D609-D617. https://doi.org/10.1093/nar/gkae1010", ""),
]
SERGEJB = [
    ("SergejB-BF curated plastic-degrading enzyme dataset ", "I"),
    ("[Unpublished raw dataset compiling ~330 primary sources]. (n.d.). Provided in "
     "datasets/benchmark/extra/sergej_db.xlsx.", ""),
]
PLASTICENZ = [
    ("Krzynowek, A., Snoeks, J., & Faust, K. (2026). PlasticEnz: An integrated "
     "database and screening tool combining homology and machine learning to "
     "identify plastic-degrading enzymes in meta-omics datasets. ", ""),
    ("PLOS Computational Biology, 22", "I"),
    ("(1), e1013892. https://doi.org/10.1371/journal.pcbi.1013892", ""),
]

def note(n):
    return [(f"  [n = {n} sequences in this split]", "")]

SECTIONS = [
    ("Training Set", "28,263 sequences", [
        (PMBD, "8,120"), (SERGEJB, "85"), (UNIPROT, "20,008"), (ZRIMEC, "50"),
    ]),
    ("In-Distribution Test Set", "9,839 sequences", [
        (PMBD, "2,157"), (SERGEJB, "15"), (UNIPROT, "4,378"), (ZRIMEC, "3,289"),
    ]),
    ("Independent Test Set", "709 sequences", [
        (PLASTICENZ, "709"),
    ]),
]


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 9)
        self.set_text_color(120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def tokenize(segments):
    """-> list of tokens; each token is a list of (text, style) pieces (no spaces)."""
    chars = [(ch, st) for text, st in segments for ch in text]
    tokens, cur = [], []
    def flush():
        if cur:
            pieces = []
            for ch, st in cur:
                if pieces and pieces[-1][1] == st:
                    pieces[-1][0] += ch
                else:
                    pieces.append([ch, st])
            tokens.append([(t, s) for t, s in pieces])
            cur.clear()
    for ch, st in chars:
        if ch == " ":
            flush()
        else:
            cur.append((ch, st))
    flush()
    return tokens


def tok_w(pdf, token):
    w = 0
    for txt, st in token:
        pdf.set_font("Times", st, FS)
        w += pdf.get_string_width(txt)
    return w


def space_w(pdf):
    pdf.set_font("Times", "", FS)
    return pdf.get_string_width(" ")


def layout(pdf, tokens):
    sw = space_w(pdf)
    lines, line, w, first = [], [], 0, True
    for tk in tokens:
        tw = tok_w(pdf, tk)
        allowed = pdf.epw if first else pdf.epw - HANG
        add = tw if not line else sw + tw
        if line and w + add > allowed:
            lines.append(line)
            line, w, first = [tk], tw, False
        else:
            line.append(tk)
            w += add
    if line:
        lines.append(line)
    return lines


def reference(pdf, segments):
    lines = layout(pdf, tokenize(segments))
    if pdf.get_y() + len(lines) * LH + 2.4 > pdf.h - pdf.b_margin:
        pdf.add_page()
    left, sw = pdf.l_margin, space_w(pdf)
    for i, line in enumerate(lines):
        pdf.set_x(left + (0 if i == 0 else HANG))
        for j, tk in enumerate(line):
            if j > 0:
                pdf.set_font("Times", "", FS)
                pdf.cell(sw, LH, " ")
            for txt, st in tk:
                pdf.set_font("Times", st, FS)
                pdf.cell(pdf.get_string_width(txt), LH, txt)
        pdf.ln(LH)
    pdf.ln(2.4)


pdf = PDF(format="A4")
pdf.set_margins(25.4, 25.4, 25.4)
pdf.set_auto_page_break(True, 20)
pdf.add_page()

pdf.set_font("Times", "B", 15)
pdf.cell(0, 9, "Data Source References", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Times", "I", 11)
pdf.cell(0, 7, "Plastic-Degrader Classifier Dataset  -  APA 7th edition",
         new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(4)
pdf.set_font("Times", "", 10)
pdf.multi_cell(0, 5.2,
    "References are grouped by the dataset partition each source contributes to. "
    "Sources marked UniProt cover both reviewed (UniProtKB) and look-alike "
    "hard-negative entries. The same source may appear in more than one split; "
    "n indicates the number of sequences it contributes to that split.")
pdf.ln(4)

for heading, total, refs in SECTIONS:
    if pdf.get_y() + 30 > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.set_font("Times", "B", 12.5)
    pdf.cell(0, 8, f"{heading}  ({total})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    for segs, count in refs:
        reference(pdf, segs + note(count))
    pdf.ln(2)

pdf.output(OUT)
print("wrote", OUT)
