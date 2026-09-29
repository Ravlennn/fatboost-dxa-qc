"""Копирует веса фолдов и конфиг из outputs/hip_cnn/<tag> в models/hip_cnn/<tag>, считает пороги из OOF (макс. F1)."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dxaqc.metrics import best_threshold
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="convnext384"); a = ap.parse_args()
src = ROOT / "outputs/hip_cnn" / a.tag; dst = ROOT / "models/hip_cnn" / a.tag; dst.mkdir(parents=True, exist_ok=True)
for f in list(src.glob("fold*.pt")) + [src / "config.json"]: shutil.copy(f, dst / f.name)
oof = pd.read_csv(src / "oof.csv"); thr = {}
for t in ["quality", "v_rotation", "v_roi"]:
    m = oof[t].notna() & oof["p_" + t].notna(); thr[t] = best_threshold(oof.loc[m, t].astype(int).values, oof.loc[m, "p_" + t].values)
(dst / "thresholds.json").write_text(json.dumps(thr, indent=1)); print("exported to", dst, thr)
