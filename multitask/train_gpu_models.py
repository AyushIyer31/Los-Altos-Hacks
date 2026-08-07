"""GPU job: train XGBoost + LightGBM + CatBoost on the A100 (GPU mode).

Loads the shared features.npz, trains the three gradient-boosting models with GPU
acceleration ON, saves each model + its S669 score. These are the ONLY classical
models that actually use the GPU — RandomForest/ExtraTrees/MLP go in the CPU job.

Run on a node with 1 A100. Needs: xgboost, lightgbm, catboost.
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
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    models = {
        # GPU mode is the key change vs the old CPU pipeline:
        "xgboost": XGBRegressor(
            n_estimators=600, max_depth=9, learning_rate=0.02, subsample=0.8,
            tree_method="hist", device="cuda", random_state=42, verbosity=0),
        # LightGBM GPU mode needs an OpenCL runtime that the pytorch image
        # lacks ("No OpenCL device found"). OpenCL gives little speedup here and
        # is fragile, so run it on CPU; XGBoost + CatBoost still use the GPU.
        "lightgbm": LGBMRegressor(
            n_estimators=1200, max_depth=10, learning_rate=0.02,
            device="cpu", n_jobs=-1, random_state=42, verbosity=-1),
        "catboost": CatBoostRegressor(
            iterations=1500, depth=8, learning_rate=0.03,
            task_type="GPU", random_seed=42, verbose=False),
    }

    for name, m in models.items():
        print(f"\n[{name}] training ...")
        try:
            m.fit(Xtr, ytr, sample_weight=wtr)
            _, vp, vr = score(m, Xva, yva)
            sp_pred, sp, sr = score(m, Xs, ys)
            print(f"  val Pearson={vp:.3f} RMSE={vr:.3f} | S669 Pearson={sp:.3f} RMSE={sr:.3f}")
            joblib.dump(m, os.path.join(OUT, f"{name}.joblib"))
            np.save(os.path.join(OUT, f"{name}_s669_pred.npy"), sp_pred)
        except Exception as e:
            # one model failing must not discard the others already trained
            print(f"  [{name}] FAILED: {e}")

    print(f"\nsaved GPU models -> {OUT}")


if __name__ == "__main__":
    main()
