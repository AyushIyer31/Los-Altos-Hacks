"""Convert the Meltome Atlas (FLIP cross-species 'mixed_split') into staging.

Source: FLIP benchmark mirror, mixed_split.fasta
  header: '>SequenceN TARGET=<Tm> SET=<split> VALIDATION=<bool>'
Measurement: Tm (absolute melting temperature, deg C) of WILD-TYPE proteins
(thermal proteome profiling). No mutations -> trains a sequence->Tm head.
"""
import csv
import re

SRC = "datasets/downloads/meltome_mixed.fasta"
OUT = "datasets/downloads/meltome_tm.csv"
COLS = ["seq_id", "tm", "split", "sequence", "source"]
HDR = re.compile(r">(\S+)\s+TARGET=([-\d.]+)\s+SET=(\S+)")


def main():
    n = 0
    with open(SRC) as f, open(OUT, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=COLS)
        w.writeheader()
        sid = tm = split = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                m = HDR.match(line)
                sid, tm, split = (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)
            elif line and sid is not None:
                w.writerow({"seq_id": sid, "tm": tm, "split": split,
                            "sequence": line.strip(), "source": "Meltome"})
                n += 1
                sid = None
    print(f"Written: {n} -> {OUT}")


if __name__ == "__main__":
    main()
