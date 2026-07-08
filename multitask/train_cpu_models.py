"""CPU job: train RandomForest + ExtraTrees + MLP (no GPU needed).

Loads the SAME features.npz as the GPU job and trains the models that get no
benefit from a GPU, so they run on a cheap CPU node IN PARALLEL with the A100 job.
Forests on 500K+ rows are memory-hungry — this is why the CPU job requests lots of
RAM and the features are PCA-reduced in build_features.py.

Run on a CPU node. Needs: scikit-learn.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(HERE, "features.npz")
OUT = os.path.join(HERE, "models")


def score(model, X, y):
    p = model.predict(X)
    pear = float(np.corrcoef(p, y)[0, 1])
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return p, pear, rmse


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(FEATURES, allow_pickle=True)
    Xtr, ytr, wtr = d["X_tr"], d["y_tr"], d["w_tr"]
    Xva, yva = d["X_va"], d["y_va"]
    Xs, ys = d["X_s669"], d["y_s669"]
    print(f"train {Xtr.shape}  val {Xva.shape}  s669 {Xs.shape}")

    import joblib
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor

    models = {
        # max_depth capped (was None): unbounded trees produced multi-GB model
        # files that overran the PVC, and overfit (val ~0.75 but S669 ~0.18).
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=25, n_jobs=-1, random_state=42),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=25, n_jobs=-1, random_state=42),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(256, 128), early_stopping=True,
            max_iter=100, random_state=42),
    }

    for name, m in models.items():
        print(f"\n[{name}] training on CPU ...")
        # RF/ExtraTrees support sample_weight; MLP does not.
        if name == "mlp":
            m.fit(Xtr, ytr)
        else:
            m.fit(Xtr, ytr, sample_weight=wtr)
        _, vp, vr = score(m, Xva, yva)
        sp_pred, sp, sr = score(m, Xs, ys)
        print(f"  val Pearson={vp:.3f} RMSE={vr:.3f} | S669 Pearson={sp:.3f} RMSE={sr:.3f}")
        joblib.dump(m, os.path.join(OUT, f"{name}.joblib"), compress=3)
        np.save(os.path.join(OUT, f"{name}_s669_pred.npy"), sp_pred)

    print(f"\nsaved CPU models -> {OUT}")


if __name__ == "__main__":
    main()
