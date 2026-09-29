"""Обучает LR на convnext-эмбеддингах бедра (все 150 размеченных) + пороги из OOF бейзлайна → models/hip_convnext_lr.joblib"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dxaqc.metrics import best_threshold

il = pd.read_csv(ROOT / "data/interim/image_labels.csv")
X = np.load(ROOT / "outputs/baseline/emb_convnext_tiny.npy")
oof = pd.read_csv(ROOT / "outputs/baseline/oof_convnext_tiny.csv")
hip = (il.region != "spine").values
bundle = {}
for key, col, pcol in [("quality", "quality", "p_quality"), ("hip_positioning", "v_rotation", "p_v_rotation"), ("hip_roi_margins", "v_roi", "p_v_roi")]:
    y = il.loc[hip, col].values.astype(float); m = ~np.isnan(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)).fit(X[hip][m], y[m].astype(int))
    o = oof[oof.region != "spine"]; mo = o[col].notna() & o[pcol].notna()
    thr = best_threshold(o.loc[mo, col].values.astype(int), o.loc[mo, pcol].values)
    bundle[key] = {"model": clf, "thr": float(thr)}
    print(key, "n", int(m.sum()), "pos", int(y[m].sum()), "thr(OOF)", round(thr, 2))
bundle["backbone"] = "convnext_tiny_IMAGENET1K_V1"; bundle["input"] = "letterbox 224, left hip mirrored"
joblib.dump(bundle, ROOT / "models/hip_convnext_lr.joblib"); print("saved")
