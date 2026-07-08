"""Combine every trained model into one ensemble and score it on S669.

Loads each model's saved S669 predictions (written by the GPU + CPU jobs), averages
them, and reports per-model vs ensemble Pearson / Spearman / RMSE. Run after both
jobs finish. CPU only, needs just numpy.
"""
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
FEATURES = os.path.join(HERE, "features.npz")


def metrics(p, y):
    pear = float(np.corrcoef(p, y)[0, 1])
    rank = float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(y)))[0, 1])
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return pear, rank, rmse


def main():
    y = np.load(FEATURES, allow_pickle=True)["y_s669"]
    preds = {}
    for f in sorted(glob.glob(os.path.join(MODELS, "*_s669_pred.npy"))):
        name = os.path.basename(f).replace("_s669_pred.npy", "")
        p = np.load(f)
        if len(p) == len(y):
            preds[name] = p

    if not preds:
        print("No model predictions found in", MODELS)
        return

    print(f"S669 n={len(y)}\n{'model':16s} {'Pearson':>8} {'Spearman':>9} {'RMSE':>7}")
    for name, p in preds.items():
        pe, sp, rm = metrics(p, y)
        print(f"{name:16s} {pe:8.3f} {sp:9.3f} {rm:7.3f}")

    ens = np.mean(np.stack(list(preds.values())), axis=0)
    pe, sp, rm = metrics(ens, y)
    print("-" * 44)
    print(f"{'ENSEMBLE':16s} {pe:8.3f} {sp:9.3f} {rm:7.3f}")
    print(f"\n(ensemble = mean of {len(preds)} models: {', '.join(preds)})")


if __name__ == "__main__":
    main()
