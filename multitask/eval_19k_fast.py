"""Fast S669 eval of the 19K model: load the already-built features_19k.npz and
train only the fast models (Extra-Trees + XGBoost + LightGBM), skipping the slow
CatBoost/RandomForest that hung the full job. Unbuffered output."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "features_19k.npz"), allow_pickle=True)
Xtr, ytr, wtr = d["X_tr"], d["y_tr"], d["w_tr"]
Xva, yva = d["X_va"], d["y_va"]
Xs, ys = d["X_s669"], d["y_s669"]
print(f"19K model: train {Xtr.shape}  val {Xva.shape}  s669 {Xs.shape}", flush=True)


def reg(p, y):
    pear = float(np.corrcoef(p, y)[0, 1])
    rank = float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(y)))[0, 1])
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return pear, rank, rmse


from sklearn.ensemble import ExtraTreesRegressor
models = {"extra_trees": ExtraTreesRegressor(n_estimators=200, max_depth=25, n_jobs=-1, random_state=42)}
try:
    import xgboost as xgb
    models["xgboost"] = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
except Exception as e:
    print("no xgboost", e, flush=True)
try:
    import lightgbm as lgb
    models["lightgbm"] = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, device="cpu", verbose=-1)
except Exception as e:
    print("no lightgbm", e, flush=True)

print(f"\n{'model':14s} {'valP':>6} {'S669_P':>7} {'Spear':>6} {'RMSE':>6}", flush=True)
spreds = {}
for name, m in models.items():
    if name == "extra_trees":
        m.fit(Xtr, ytr, sample_weight=wtr)
    else:
        m.fit(Xtr, ytr)
    vp, _, _ = reg(m.predict(Xva), yva)
    ps = m.predict(Xs); spreds[name] = ps
    p, sp, rm = reg(ps, ys)
    print(f"{name:14s} {vp:6.3f} {p:7.3f} {sp:6.3f} {rm:6.2f}", flush=True)

ens = np.mean(np.stack(list(spreds.values())), 0)
p, sp, rm = reg(ens, ys)
print(f"{'ENSEMBLE':14s} {'':>6} {p:7.3f} {sp:6.3f} {rm:6.2f}", flush=True)

# S669 stabilizing-classifier threshold sweep (positive = ddG<0)
yt = (ys < 0).astype(int)
from sklearn.metrics import roc_auc_score
auc = float(roc_auc_score(yt, -ens))
P, N = int(yt.sum()), int((1 - yt).sum())
print(f"\nS669 stabilizing/destabilizing = {P}/{N} | ensemble AUC = {auc:.3f}", flush=True)
print(f"{'thr':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}   (tp,fp,fn,tn)", flush=True)
for thr in np.arange(-1.0, 1.01, 0.25):
    pp = (ens < thr).astype(int)
    tp = int(((pp == 1) & (yt == 1)).sum()); fp = int(((pp == 1) & (yt == 0)).sum())
    fn = int(((pp == 0) & (yt == 1)).sum()); tn = int(((pp == 0) & (yt == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = (tp + tn) / len(yt)
    print(f"{thr:6.2f} {pr:6.3f} {rc:7.3f} {f1:6.3f} {acc:6.3f}   ({tp},{fp},{fn},{tn})", flush=True)
print("\ndone (19K fast).", flush=True)
