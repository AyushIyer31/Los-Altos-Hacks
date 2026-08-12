"""Append the harmonised-schema example table to PETase_Stability_Analysis.ipynb."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "PETase_Stability_Analysis.ipynb")
nb = nbf.read(NB, as_version=4)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

nb.cells.append(md(r"""
---
# 15 · What harmonisation actually produces

Seven databases record protein stability in incompatible ways: some measure a
mutation, some a whole protein; some report assay conditions, some none; some
measure free energy, others a cell-abundance proxy. Harmonisation maps all of them
onto **one 24-column schema** without discarding the differences or inventing
values to fill gaps.

The three records below are pulled live from `staging_clean.csv` — one per source,
chosen to show three different data shapes.
"""))

nb.cells.append(code(r'''
SCHEMA_COLS = ["protein_name", "uniprot_id", "pdb_id", "chain",
               "wt_sequence", "mut_sequence", "mutation", "position",
               "wt_aa", "mut_aa", "assay_temperature_c", "ph", "denaturant",
               "condition_quality", "measurement_type", "measured_value",
               "source_dataset", "pmid"]

_st = pd.read_csv(os.path.join(ROOT, "datasets", "staging", "staging_clean.csv"),
                  usecols=SCHEMA_COLS, low_memory=False)

def exemplar(source, mtype, name_contains, mut):
    """One named record, requiring BOTH sequences present so the row is
    actually usable for sequence-based training (ThermoMutDB rows with
    PDB-based numbering are deliberately left sequence-less, not fabricated)."""
    s = _st[(_st.source_dataset == source) & (_st.measurement_type == mtype)
            & (_st.mutation == mut)
            & _st.protein_name.str.contains(name_contains, case=False, na=False)
            & _st.wt_sequence.notna() & _st.mut_sequence.notna()]
    return s.iloc[0]

# Three named proteins, three measurement types, assay pH spanning 2.0 -> 8.2.
EX = [exemplar("ThermoMutDB", "ddG", "Endolysin", "V131A"),
      exemplar("ThermoMutDB", "dTm", "ribonuclease T1", "Y68F"),
      exemplar("FireProtDB", "Tm", "Dihydrofolate reductase", "I115A")]

pd.DataFrame({f"record {i+1}": r for i, r in enumerate(EX)}).fillna("—")
'''))

nb.cells.append(md(r"""
### 15.1 · The same three records as a poster figure
"""))

nb.cells.append(code(r'''
from matplotlib.patches import FancyBboxPatch, Rectangle

def fmt(v, seq=False, n=22):
    if pd.isna(v):
        return "—"
    if seq:
        return f"{str(v)[:n]}…  ({len(str(v))} aa)"
    if isinstance(v, float):
        return f"{v:g}"
    s = str(v)
    return s if len(s) <= 34 else s[:32] + "…"

GROUPS = [
    ("IDENTITY", "#8a8a85", [
        ("protein_name",        lambda r: fmt(r.protein_name)),
        ("uniprot_id",          lambda r: fmt(r.uniprot_id)),
        ("pdb_id · chain",      lambda r: fmt(r.pdb_id) + ("" if pd.isna(r.chain) else f" · {r.chain}")),
        ("wt_sequence",         lambda r: fmt(r.wt_sequence, seq=True)),
    ]),
    ("MUTATION", "#5a6a72", [
        ("mutation",            lambda r: fmt(r.mutation)),
        ("wt_aa → mut_aa @ pos", lambda r: "—" if pd.isna(r.mutation)
                                 else f"{r.wt_aa} → {r.mut_aa}  @ {int(r.position)}"),
        ("mut_sequence",        lambda r: fmt(r.mut_sequence, seq=True)),
    ]),
    ("CONDITIONS", "#4a3aa7", [
        ("assay_temperature_c", lambda r: fmt(r.assay_temperature_c)),
        ("ph",                  lambda r: fmt(r.ph)),
        ("denaturant / assay",  lambda r: fmt(r.denaturant)),
        ("condition_quality",   lambda r: fmt(r.condition_quality)),
    ]),
    ("MEASURE", "#1a6a72", [
        ("measurement_type",    lambda r: fmt(r.measurement_type)),
        ("measured_value",      lambda r: fmt(round(float(r.measured_value), 4))),
    ]),
    ("SOURCE", "#8a8a85", [
        ("source_dataset",      lambda r: fmt(r.source_dataset)),
        ("pmid",                lambda r: "—" if pd.isna(r.pmid) else str(int(r.pmid))),
    ]),
]

HEADERS = [("record 1", "ΔΔG  ·  strongly acidic assay"),
           ("record 2", "ΔT$_m$  ·  neutral assay"),
           ("record 3", "T$_m$  ·  alkaline assay")]

n_rows = sum(len(g[2]) for g in GROUPS)
fig, ax = plt.subplots(figsize=(15.0, 8.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

LX, LW, CW, GAP = .012, .175, .258, .008
XS = [LX + LW + GAP + i * (CW + GAP) for i in range(3)]
TOP, RH = .845, .0455
BOT = TOP - n_rows * RH

ax.text(.5, .985, "One schema for seven heterogeneous databases",
        ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
ax.text(.5, .948, "three real records from the harmonised staging table "
                  "(1,001,888 rows · 24 columns)",
        ha="center", va="top", fontsize=10.5, color="#555", style="italic")

# column headers
for x, (h, sub), r in zip(XS, HEADERS, EX):
    ax.add_patch(FancyBboxPatch((x, TOP + .012), CW, .062,
                 boxstyle="round,pad=0.003,rounding_size=.008",
                 fc="#f2f2f0", ec="#bbb", lw=1.1))
    ax.text(x + CW/2, TOP + .058, f"{h}  ·  {r.source_dataset}", ha="center",
            va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(x + CW/2, TOP + .030, sub, ha="center", va="center",
            fontsize=8.2, color="#666", style="italic", linespacing=1.3)

# rows, grouped
y = TOP
for gname, gcol, fields in GROUPS:
    gtop = y
    for fname, getter in fields:
        y -= RH
        if int((TOP - y) / RH) % 2 == 1:
            ax.add_patch(Rectangle((LX, y), LW + GAP + 3*(CW+GAP), RH,
                                   fc="#fafaf9", ec="none", zorder=0))
        ax.text(LX + .030, y + RH/2, fname, ha="left", va="center",
                fontsize=8.4, color="#3a3a36", family="monospace")
        for x, r in zip(XS, EX):
            val = getter(r)
            miss = val == "—"
            ax.text(x + .012, y + RH/2, val, ha="left", va="center",
                    fontsize=8.5, color="#c0392b" if miss else INK,
                    fontweight="bold" if (miss or fname in
                        ("measurement_type", "measured_value")) else "normal",
                    family="monospace" if fname == "wt_sequence" or
                        fname == "mut_sequence" else "sans-serif")
        ax.plot([LX, LX + LW + GAP + 3*(CW+GAP)], [y, y], color="#e2e2df",
                lw=.7, zorder=1)
    # group label bar
    ax.add_patch(Rectangle((LX, y), .0165, gtop - y, fc=gcol, ec="none", zorder=3))
    # shrink the rotated label so it can never overrun a short group's bar
    ax.text(LX + .0082, (gtop + y)/2, gname, ha="center", va="center",
            fontsize=min(7.6, 190 * (gtop - y) / max(len(gname), 1)),
            fontweight="bold", color="white", rotation=90, zorder=4)

ax.plot([LX, LX + LW + GAP + 3*(CW+GAP)], [TOP, TOP], color="#555", lw=1.5, zorder=2)

# footer
ax.add_patch(FancyBboxPatch((LX, BOT - .122), LW + GAP + 3*(CW+GAP), .100,
             boxstyle="round,pad=0.004,rounding_size=.008",
             fc="#fdf6f4", ec="#c0392b", lw=1.3))
ax.text(.5, BOT - .072,
        f"Three measurement types, three assay pH values ({EX[0].ph} · {EX[1].ph} · {EX[2].ph}), one schema. "
        "Conditions travel with every record instead of being averaged away,\n"
        "and any field the source did not report is kept NULL (—) rather than imputed — "
        "so a measured pH is never confused with an assumed one.",
        ha="center", va="center", fontsize=9.2, color=INK, linespacing=1.6)

save(fig, "fig7_schema_table"); plt.show()
'''))

nbf.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
