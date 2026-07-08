"""Stream the FireProtDB Postgres dump once and reconstruct the ProTherm
(dataset_id=1) condition-varied records: mutation + ddG/dTm/Tm + pH + temperature.

Reads the SQL on stdin (via `unzip -p ... | python3 extract_fireprot_protherm.py`).
EAV model: experiment(dataset_id) groups measurement rows (type in
{ddG,pH,temperature,dTm,Tm,...}); mutant->substitution gives the mutation and
mutant.source_id -> sequence gives the WT sequence.
Writes a flat CSV to datasets/downloads/fireprotdb_protherm.csv.
"""
import sys, csv, os

TARGET_DATASET = "1"  # ProTherm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fireprotdb_protherm.csv")

def colmap(header):
    # header like: COPY public.measurement (id, experiment_id, ...) FROM stdin;
    cols = header[header.index("(")+1:header.index(")")].split(",")
    return {c.strip().strip('"'): i for i, c in enumerate(cols)}

exp1 = set()                 # experiment ids in ProTherm
exp = {}                     # exp_id -> {type: value}, plus _mutant/_seq
ann = {}                     # exp_id -> {annotation type: value} (PH, EXP_TEMPERATURE, ...)
need_mut = set()             # mutant ids referenced
mut_src = {}                 # mutant_id -> source(WT) sequence id
mut_subs = {}                # mutant_id -> list of (pos, src_aa, tgt_aa)
need_seq = set()             # sequence ids referenced (WT)
seqs = {}                    # sequence id -> sequence

cur = None; cm = None
for line in sys.stdin:
    if cur is None:
        if line.startswith("COPY public."):
            t = line.split("public.",1)[1].split(" ",1)[0]
            if t in ("experiment","experiment_annotation","measurement",
                     "mutant","substitution","sequence"):
                cur = t; cm = colmap(line)
        continue
    if line.startswith("\\."):
        cur = None; continue
    f = line.rstrip("\n").split("\t")
    if cur == "experiment":
        if f[cm["dataset_id"]] == TARGET_DATASET:
            exp1.add(f[cm["id"]])
    elif cur == "experiment_annotation":
        eid = f[cm["experiment_id"]]
        if eid in exp1:
            v = f[cm["num_value"]]
            ann.setdefault(eid, {})[f[cm["type"]]] = (
                v if v != "\\N" else f[cm["str_value"]])
    elif cur == "measurement":
        eid = f[cm["experiment_id"]]
        if eid in exp1:
            d = exp.setdefault(eid, {})
            typ = f[cm["type"]]
            val = f[cm["num_value"]]
            d[typ] = val if val != "\\N" else f[cm["str_value"]]
            mid = f[cm["mutant_id"]]
            if mid != "\\N":
                d["_mut"] = mid; need_mut.add(mid)
    elif cur == "mutant":
        mid = f[cm["id"]]
        if mid in need_mut:
            sid = f[cm["source_id"]]; mut_src[mid] = sid; need_seq.add(sid)
    elif cur == "sequence":
        sid = f[cm["id"]]
        if sid in need_seq:
            seqs[sid] = f[cm["sequence"]]
    elif cur == "substitution":
        mid = f[cm["mutant_id"]]
        if mid in need_mut:
            mut_subs.setdefault(mid, []).append(
                (f[cm["position"]], f[cm["source_aa"]], f[cm["target_aa"]]))

# join + write
n=0; nsingle=0
with open(OUT,"w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["wt_sequence","mutation","n_subs","position","wt_aa","mut_aa",
                "ddG","dTm","Tm","dG","temperature_c","pH","buffer","ion_conc",
                "method","measure","exp_id"])
    for eid,d in exp.items():
        mid=d.get("_mut")
        if not mid: continue
        subs=mut_subs.get(mid,[])
        wtseq=seqs.get(mut_src.get(mid,""),"")
        if len(subs)==1:
            pos,wa,ma=subs[0]; mut=f"{wa}{pos}{ma}"; nsingle+=1
        else:
            pos=wa=ma=""; mut=";".join(f"{a}{p}{b}" for p,a,b in subs)
        a=ann.get(eid,{})
        def g(src,k):
            v=src.get(k,""); return "" if v in ("\\N",None) else v
        w.writerow([wtseq,mut,len(subs),pos,wa,ma,
                    g(d,"DDG"),g(d,"DTM"),g(d,"TM"),g(d,"DG"),
                    g(a,"EXP_TEMPERATURE"),g(a,"PH"),g(a,"BUFFER"),
                    g(a,"ION_CONC"),g(a,"METHOD"),g(a,"MEASURE"),eid])
        n+=1
print(f"wrote {n} ProTherm records ({nsingle} single-substitution) -> {OUT}", file=sys.stderr)
# report measurement type vocabulary seen
from collections import Counter
tc=Counter()
for d in exp.values():
    for k in d:
        if not k.startswith("_"): tc[k]+=1
print("measurement types seen:", dict(tc.most_common(15)), file=sys.stderr)
