"""OOF-оценка MVP: позвоночник — правила (без обучения, все 99), бедро — OOF LR из бейзлайна. → outputs/mvp/metrics.md"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, pydicom
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dxaqc.metrics import bootstrap_ci, format_table, binary_metrics
from dxaqc.spine_rules import measure_spine
OUT = ROOT / "outputs/mvp"; OUT.mkdir(parents=True, exist_ok=True)

il = pd.read_csv(ROOT / "data/interim/image_labels.csv")
import json
ens = json.loads((ROOT / "models/hip_cnn/ensemble.json").read_text())
_oofs = [pd.read_csv(ROOT / "outputs/hip_cnn" / t / "oof.csv") for t in ens["tags"]]
oof = _oofs[0][["image_uid"]].copy()
for c in ["p_quality", "p_v_rotation", "p_v_roi"]:
    oof[c] = np.mean([o[c].values for o in _oofs], axis=0)
oof = oof.set_index("image_uid")
b = {"quality": {"thr": ens["thresholds"]["quality"]}, "hip_positioning": {"thr": ens["thresholds"]["v_rotation"]}, "hip_roi_margins": {"thr": ens["thresholds"]["v_roi"]}}
# Вложенные пороги: для фолда k порог подбирается только на OOF остальных фолдов (честная F1)
from dxaqc.metrics import best_threshold as _bt
_fold = _oofs[0].set_index("image_uid")["fold"]
_lab = _oofs[0].set_index("image_uid")
NESTED = {}
for k in sorted(_fold.unique()):
    tr = _fold != k
    NESTED[k] = {}
    for key, col, pcol in [("quality", "quality", "p_quality"), ("hip_positioning", "v_rotation", "p_v_rotation"), ("hip_roi_margins", "v_roi", "p_v_roi")]:
        m = tr & _lab[col].notna()
        NESTED[k][key] = _bt(_lab.loc[m, col].astype(int).values, oof.loc[m.index[m], pcol].values)
# Позвоночник: измерения по всем снимкам, затем вложенный подбор порогов правил по сетке
from dxaqc.spine_rules import AXIS_DEG, AXIS_MAX_CURV, ILIAC_RATIO_MIN, ARTIFACT_PX
folds = pd.read_csv(ROOT / "data/interim/folds.csv").set_index("image_uid")["fold"]
sp_rows = []
for _, r in il[il.region == "spine"].iterrows():
    a = pydicom.dcmread(str(ROOT / "data/interim/train/Исследования" / r.path)).pixel_array
    m = measure_spine(a)
    sp_rows.append(dict(image_uid=r.image_uid, fold=int(folds.loc[r.image_uid]), q=int(r.quality), angle=m.axis_angle_deg, curv=m.curvature_deg, iliac=m.iliac_ratio, art=m.artifact_px))
SP = pd.DataFrame(sp_rows).set_index("image_uid")
GRID = [(T, C, R, K) for T in (3, 3.5, 4, 5) for C in (3, 4, 5, 99) for R in (1.5, 1.7, 2.0) for K in (40, 60, 80, 120)]
def _rule(d, T, C, R, K):
    return {"spine_axis_tilt": (d.angle.abs() >= T) & (d.curv.fillna(0) <= C), "spine_scan_range": d.iliac < R, "spine_artifact": d.art >= K}
def _f1(d, g):
    from sklearn.metrics import f1_score
    f = _rule(d, *g); return f1_score(d.q, (f["spine_axis_tilt"] | f["spine_scan_range"] | f["spine_artifact"]).astype(int))
SP_PARAMS = {k: max(GRID, key=lambda g: _f1(SP[SP.fold != k], g)) for k in sorted(SP.fold.unique())}
CODE_PARAMS = (AXIS_DEG, AXIS_MAX_CURV, ILIAC_RATIO_MIN, ARTIFACT_PX)

rows = []
for _, r in il.iterrows():
    a = pydicom.dcmread(str(ROOT / "data/interim/train/Исследования" / r.path)).pixel_array
    d = dict(study=r.study, image_uid=r.image_uid, region=r.region, quality=r.quality)
    if r.region == "spine":
        m = measure_spine(a); s = m.scores()
        g = SP_PARAMS[int(folds.loc[r.image_uid])]          # пороги, подобранные без этого фолда
        fl = _rule(SP.loc[[r.image_uid]], *g); f = {c: bool(v.iloc[0]) for c, v in fl.items()}
        d.update(p_quality=max(s.values()), yhat=int(any(f.values())), angle=m.axis_angle_deg, curv=m.curvature_deg, iliac=m.iliac_ratio, art=m.artifact_px,
                 yhat_code_thr=int(any(m.flags().values())))
        for code, col in [("spine_scan_range", "v_layout"), ("spine_axis_tilt", "v_axis"), ("spine_artifact", "v_artifact")]:
            d[f"y_{code}"] = r[col]; d[f"p_{code}"] = s[code]; d[f"f_{code}"] = int(f[code])
    else:
        if r.image_uid not in oof.index:   # неразмеченные (эндопротезы) не входят в OOF
            continue
        o = oof.loc[r.image_uid]; nb = NESTED[int(_fold.loc[r.image_uid])]
        fpos = int(o.p_v_rotation >= nb["hip_positioning"]); froi = int(o.p_v_roi >= nb["hip_roi_margins"])
        d.update(p_quality=o.p_quality, yhat=int(o.p_quality >= nb["quality"] or fpos or froi))
        d.update(y_hip_positioning=r.v_rotation, p_hip_positioning=o.p_v_rotation, f_hip_positioning=fpos, y_hip_roi_margins=r.v_roi, p_hip_roi_margins=o.p_v_roi, f_hip_roi_margins=froi)
    rows.append(d)
df = pd.DataFrame(rows); df.to_csv(OUT / "oof_mvp.csv", index=False)

rep = [f"# MVP: позвоночник — правила, бедро — ансамбль CNN {ens['tags']} (OOF, вложенные пороги)\n",
       "Бинарный класс: `yhat` = хотя бы одно нарушение. **Все пороги (CNN и правил) подобраны вложенно: для фолда k только на OOF остальных фолдов.** Бутстрап 95% ДИ по исследованиям. "
       "Остаточный оптимизм: выбор архитектур/состава ансамбля и вид признаков правил делались с просмотром всех данных.\n",
       f"Пороги правил по фолдам (T°, кривизна, iliac, artifact px): {SP_PARAMS}; в коде зашиты {CODE_PARAMS}.\n"]
res = {}
for name, mask in [("все", np.ones(len(df), bool)), ("позвоночник (правила)", (df.region == "spine").values), ("бедро (CNN OOF)", (df.region != "spine").values)]:
    mm = mask & df.quality.notna().values
    res[name] = bootstrap_ci(df.quality[mm].astype(int).values, df.yhat[mm].values.astype(float), df.study[mm].values, thr=0.5)
rep += ["## Бинарный класс качества (по бинарным решениям yhat)\n", format_table(res), ""]
res2 = {}
for name, mask in [("все", np.ones(len(df), bool)), ("позвоночник", (df.region == "spine").values), ("бедро", (df.region != "spine").values)]:
    mm = mask & df.quality.notna().values
    res2[name] = bootstrap_ci(df.quality[mm].astype(int).values, df.p_quality[mm].values, df.study[mm].values, thr=0.5)
rep += ["## ROC/PR-AUC по непрерывному скору p_quality (порог 0.5 здесь не осмыслен, смотреть только AUC)\n", format_table(res2), ""]
lines = ["| код | n | pos | sens | spec | F1 | ROC-AUC |", "|---|---|---|---|---|---|---|"]; f1s = []
for code in ["spine_scan_range", "spine_axis_tilt", "spine_artifact", "hip_positioning", "hip_roi_margins"]:
    m = df[f"y_{code}"].notna()
    y = df.loc[m, f"y_{code}"].astype(int).values; yh = df.loc[m, f"f_{code}"].values.astype(float); p = df.loc[m, f"p_{code}"].values
    bm = binary_metrics(y, yh, 0.5); auc = binary_metrics(y, p, 0.5)["roc_auc"]; f1s.append(bm["f1"])
    lines.append(f"| {code} | {bm['n']} | {bm['pos']} | {bm['sensitivity']:.2f} | {bm['specificity']:.2f} | {bm['f1']:.2f} | {auc:.3f} |")
lines.append(f"| **macro-F1** | | | | | **{np.mean(f1s):.3f}** | |")
rep += ["## Типы нарушений\n", *lines, ""]
(OUT / "metrics.md").write_text("\n".join(rep), encoding="utf-8"); print("\n".join(rep))
