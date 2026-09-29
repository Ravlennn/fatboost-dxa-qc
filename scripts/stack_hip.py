"""Стекинг для бедра: OOF-вероятности CNN + геометрические признаки → LR (вложенный OOF по тем же фолдам).
Запуск: .venv/bin/python scripts/stack_hip.py --tag convnext384
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dxaqc.metrics import bootstrap_ci, format_table, best_threshold, binary_metrics
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="convnext384"); ap.add_argument("--tags", default=None, help="несколько тегов через запятую → усреднение CNN")
args = ap.parse_args()
tags = args.tags.split(",") if args.tags else [args.tag]
il = pd.read_csv(ROOT / "data/interim/image_labels.csv")
hp = il[il.region != "spine"].reset_index(drop=True)
geo = pd.read_csv(ROOT / "outputs/eda/hip_features2.csv"); geo1 = pd.read_csv(ROOT / "outputs/eda/hip_features.csv")
hp = hp.merge(geo.drop(columns=["v_rot", "v_roi"]), left_index=True, right_on="i").merge(geo1[["i", "lat_margin", "shaft_w", "bone_frac"]], on="i")
oofs = [pd.read_csv(ROOT / "outputs/hip_cnn" / t / "oof.csv") for t in tags]
cnn = oofs[0][["image_uid", "fold"]].copy()
for c in ["p_quality", "p_v_rotation", "p_v_roi"]:
    cnn[c] = np.mean([o[c].values for o in oofs], axis=0)
df = hp.merge(cnn, on="image_uid", how="inner")
print("images", len(df), "tags", tags)
GEO = ["lt_res12", "lt_resarea8", "lt_curv", "mono_up", "gap_lat", "bottom", "lat_margin", "shaft_w", "bone_frac", "gap_top", "top_b"]
def oof_lr(X, y, folds, C=0.3):
    p = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k; m = ~np.isnan(y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr & m], y[tr & m].astype(int))
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p
rep = [f"# Стекинг бедра: CNN {tags} + геометрия (OOF)\n"]
summary = {}
for target, pc in [("quality", "p_quality"), ("v_rotation", "p_v_rotation"), ("v_roi", "p_v_roi")]:
    y = df[target].values.astype(float); m = ~np.isnan(y); g = df.study.values; f = df.fold.values
    Xg = df[GEO].fillna(df[GEO].median()).values.astype(float)
    variants = {"CNN": df[pc].values, "геометрия LR": oof_lr(Xg, y, f),
                "CNN + геометрия LR": oof_lr(np.column_stack([np.log(df[pc].values / (1 - df[pc].values + 1e-6) + 1e-6), Xg]), y, f)}
    if target == "quality":
        # для бинарного класса: добавить логиты по типам
        X2 = np.column_stack([np.log(df[c].values / (1 - df[c].values + 1e-6) + 1e-6) for c in ["p_quality", "p_v_rotation", "p_v_roi"]] + [Xg])
        variants["CNN(3 выхода) + геометрия LR"] = oof_lr(X2, y, f)
    res = {}
    for name, p in variants.items():
        t = best_threshold(y[m].astype(int), p[m]); res[f"{name} @best={t:.2f}"] = bootstrap_ci(y[m].astype(int), p[m], g[m], thr=t, n_boot=500)
        summary[f"{target}/{name}"] = {"roc_auc": round(res[f'{name} @best={t:.2f}']["roc_auc"]["value"], 3), "f1_best": round(res[f'{name} @best={t:.2f}']["f1"]["value"], 3)}
    rep += [f"## {target} (pos {int(np.nansum(y))} / {int(m.sum())})\n", format_table(res), ""]
out = ROOT / "outputs/hip_stack"; out.mkdir(exist_ok=True)
(out / f"metrics_{'+'.join(tags)}.md").write_text("\n".join(rep), encoding="utf-8"); (out / f"summary_{'+'.join(tags)}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("\n".join(rep))
