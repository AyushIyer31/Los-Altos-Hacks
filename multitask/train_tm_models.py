"""Train whole-protein sequence->Tm regressors on features_tm.npz, then evaluate
on the leakage-audited BRENDA temperature-stability test set.

Two evaluations on BRENDA:
  - REGRESSION: predicted Tm vs BRENDA stable_temp_c (overall, and on the cleaner
    basis='Tm*' subset where BRENDA's value is a real melting temperature).
  - CLASSIFICATION: threshold predicted Tm at 60C -> stable/not, scored vs BRENDA's
    positive/negative labels (precision/recall/F1/accuracy + AUC).

CPU-only (the Tm set is small). Output: models_tm/ + tm_results.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "features_tm.npz")
OUT = os.path.join(HERE, "models_tm")
THRESH = 60.0   # predicted Tm >= 60C -> "stable" (matches BRENDA positive label)


def reg(p, y):
    return (float(np.corrcoef(p, y)[0, 1]), float(np.sqrt(((p - y) ** 2).mean())))


def clf(pred_temp, label):
    keep = np.isin(label, ["positive", "negative"])
    yt = (label[keep] == "positive").astype(int)
    pp = (pred_temp[keep] >= THRESH).astype(int)
    tp = int(((pp == 1) & (yt == 1)).sum()); fp = int(((pp == 1) & (yt == 0)).sum())
    fn = int(((pp == 0) & (yt == 1)).sum()); tn = int(((pp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(yt)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yt, pred_temp[keep]))
    except Exception:
        auc = float("nan")
    return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                accuracy=round(acc, 3), auc=round(auc, 3),
                tp=tp, fp=fp, fn=fn, tn=tn, n=int(len(yt)))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(F, allow_pickle=True)
    Xtr, ytr, Xva, yva = d["X_tr"], d["y_tr"], d["X_va"], d["y_va"]
    Xb, bt, bl, bb = d["X_brenda"], d["brenda_temp"].astype(float), d["brenda_label"], d["brenda_basis"]
    tm_mask = np.array([str(x).startswith("Tm") for x in bb])
    print(f"train {Xtr.shape}  val {Xva.shape}  brenda {Xb.shape}  (Tm-basis rows {int(tm_mask.sum())})")

    import joblib
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    models = {
        "extra_trees": ExtraTreesRegressor(n_estimators=400, max_depth=25, n_jobs=-1, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=400, max_depth=25, n_jobs=-1, random_state=42),
    }
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    except Exception as e:
        print("no xgboost:", e)
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03,
                                               subsample=0.8, device="cpu", verbose=-1)
    except Exception as e:
        print("no lightgbm:", e)
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(iterations=800, depth=8, learning_rate=0.03, verbose=0)
    except Exception as e:
        print("no catboost:", e)

    results, bpreds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        vpear, vrmse = reg(m.predict(Xva), yva)
        pb = m.predict(Xb)
        bpreds[name] = pb
        bpear, brmse = reg(pb, bt)
        tmpear, tmrmse = reg(pb[tm_mask], bt[tm_mask]) if tm_mask.sum() > 2 else (float("nan"),) * 2
        cm = clf(pb, bl)
        results[name] = dict(val_pearson=round(vpear, 3), val_rmse=round(vrmse, 3),
                             brenda_pearson=round(bpear, 3), brenda_rmse=round(brmse, 3),
                             brenda_pearson_Tmbasis=round(tmpear, 3), clf=cm)
        print(f"\n[{name}] val P={vpear:.3f} RMSE={vrmse:.2f} | "
              f"BRENDA P={bpear:.3f} (Tm-basis {tmpear:.3f}) | "
              f"clf F1={cm['f1']} acc={cm['accuracy']} AUC={cm['auc']}")
        joblib.dump(m, os.path.join(OUT, f"tm_{name}.joblib"), compress=3)

    ens = np.mean(np.stack(list(bpreds.values())), 0)
    bpear, brmse = reg(ens, bt)
    cm = clf(ens, bl)
    results["ensemble"] = dict(brenda_pearson=round(bpear, 3), brenda_rmse=round(brmse, 3), clf=cm)
    print(f"\n[ENSEMBLE] BRENDA P={bpear:.3f} | clf F1={cm['f1']} acc={cm['accuracy']} AUC={cm['auc']}")
    print(f"  confusion: tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']} (n={cm['n']})")

    json.dump(results, open(os.path.join(OUT, "tm_results.json"), "w"), indent=2)
    print(f"\nsaved models + tm_results.json -> {OUT}")


if __name__ == "__main__":
    main()
