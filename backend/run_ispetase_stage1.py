"""Run the Stage 1 (sequence -> Tm) screener on IsPETase, UniProt A0A0K8P6T7.

Reproduces the Stage 1 feature pipeline from multitask/build_features_tm.py exactly:
    ESM-2 esm2_t30_150M_UR50D, layer 30, mean-pooled over residues (640-d)
    + log1p(sequence length)                                     = 641 features
then averages the three saved ensemble members (LightGBM, XGBoost, CatBoost).

NOTE: the ESM forward pass runs in a SUBPROCESS. Loading torch and the gradient
boosters into one process segfaults on macOS (duplicate OpenMP runtimes), so the
embedding is computed separately and passed via a .npy file.

    /usr/local/bin/python3 backend/run_ispetase_stage1.py
"""
import os
import sys
import json
import subprocess
import tempfile
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, "multitask", "models_tm")
UNIPROT = "A0A0K8P6T7"
MODELS = ["tm_lightgbm", "tm_xgboost", "tm_catboost"]

EMBED_SRC = '''
import sys, numpy as np, torch, esm
seq, out = sys.argv[1], sys.argv[2]
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.eval()
_, _, tok = alphabet.get_batch_converter()([("q", seq[:1022])])
with torch.no_grad():
    reps = model(tok, repr_layers=[30])["representations"][30]
np.save(out, reps[0, 1:len(seq[:1022]) + 1].mean(0).float().numpy())
'''


def fetch_sequence(acc):
    txt = urllib.request.urlopen(
        f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30).read().decode()
    return txt.split("\n")[0], "".join(txt.split("\n")[1:]).strip()


def main():
    header, seq = fetch_sequence(UNIPROT)
    print(header)
    print(f"length: {len(seq)} residues\n")

    with tempfile.TemporaryDirectory() as td:
        script, npy = os.path.join(td, "e.py"), os.path.join(td, "e.npy")
        open(script, "w").write(EMBED_SRC)
        print("embedding with esm2_t30_150M_UR50D (layer 30, mean-pooled) ...")
        subprocess.run([sys.executable, script, seq, npy], check=True)
        emb = np.load(npy)

    x = np.concatenate([emb, [np.log1p(len(seq))]]).astype(np.float32)[None, :]
    print(f"feature vector: {x.shape[1]} dims  (640 embedding + 1 log-length)\n")

    import joblib
    preds = {}
    for name in MODELS:
        mdl = joblib.load(os.path.join(MDIR, f"{name}.joblib"))
        preds[name] = float(np.asarray(mdl.predict(x)).ravel()[0])
    ens = float(np.mean(list(preds.values())))

    print("STAGE 1 — predicted melting temperature")
    print("-" * 46)
    for k, v in preds.items():
        print(f"  {k:14s} {v:6.2f} °C")
    print(f"  {'ENSEMBLE':14s} {ens:6.2f} °C   <- 3-model mean")

    print("\nAgainst the Stage 1 operating points:")
    for thr, prec in [(46, .650), (50, .713), (52, .766), (60, .928), (66, .978)]:
        call = "THERMOSTABLE" if ens >= thr else "not thermostable"
        print(f"  threshold {thr:>2} °C (precision {prec:.3f})  ->  {call}")

    out = os.path.join(ROOT, "paper_figures", "ispetase_stage1.json")
    json.dump({"uniprot": UNIPROT, "header": header, "length": len(seq),
               "sequence": seq, "per_model": preds, "ensemble_tm_c": round(ens, 2)},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
