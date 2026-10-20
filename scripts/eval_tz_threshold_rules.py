"""TZ thresholds taken literally as rules, on top of the release logic (README «Эксперименты»).

Each rule forces a defect and its cause code where a landmark measurement breaks the TZ threshold
(tilt > 5°, less than half a vertebra above L1 = half of Th12, ROI margins 30/20 mm). Scored on the
published isolated decisions replayed with the release rules (scripts/per_type_metrics.py), paired
bootstrap over studies, Bonferroni over the rules tried.

  python scripts/eval_tz_threshold_rules.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/scripts')
import per_type_metrics as M
D = ROOT + '/outputs/'
t0 = pd.read_csv(D + 'final_validation_isolated/decisions_combined.csv')
pred = {r['image_uid']: r['measurements'] for r in json.load(open(D + 'landmarks_v1/ours/predictions.json'))}
mg = pd.Series([pred.get(u) for u in t0.image_uid], index=t0.index)
base = M.replay(t0, True, True, mg).reset_index(drop=True)
m = [pred.get(u) or {} for u in base.image_uid]
spine = base.region.eq('spine').values; hip = ~spine
g = lambda k: np.array([x.get(k, np.nan) for x in m], float)
CAUSES = ['spine_scan_range', 'spine_axis_tilt', 'spine_artifact', 'hip_positioning', 'hip_roi_margins']

def force(t, mask, code):
    t = t.copy(); mask = mask & np.isfinite(mask.astype(float))
    t.loc[mask, 'quality_class'] = 1
    t.loc[mask, 'score_quality'] = np.maximum(t.loc[mask, 'score_quality'], 1.0)
    t.loc[mask, 'reported_' + code] = True
    return t

rules = {
  'R1 |наклон, калибр.| > 5°': ('spine_axis_tilt', spine & (np.abs(g('spine_tilt_deg_calibrated')) > 5)),
  'R2 |наклон, сырой| > 5°': ('spine_axis_tilt', spine & (np.abs(g('spine_tilt_deg')) > 5)),
  'R3 над L1 < ½ позвонка (½ Th12)': ('spine_scan_range', spine & (g('top_margin_vert') < 0.5)),
  'R4 поля ROI < 30/20 мм (буква ТЗ)': ('hip_roi_margins', hip & ((g('roi_top_mm') < 30) | (g('roi_bottom_mm') < 30) |
                                                  (g('roi_left_mm') < 20) | (g('roi_right_mm') < 20))),
}
def mets(t, idx=None):
    d = t if idx is None else t.iloc[idx]
    y = d.y_quality.astype(int)
    f = f1_score(y, d.quality_class.astype(int)); a = roc_auc_score(y, d.score_quality)
    cf = []
    for c in CAUSES:
        dd = d[d['y_' + c].notna()]
        cf.append(f1_score(dd['y_' + c].astype(int), dd['reported_' + c].astype(int), zero_division=0))
    return f, a, np.mean(cf), dict(zip(CAUSES, cf))
st = base.study.values; studies = np.unique(st); pos = {s: np.where(st == s)[0] for s in studies}
rng = np.random.default_rng(0); boots = [np.concatenate([pos[s] for s in rng.choice(studies, len(studies))]) for _ in range(2000)]
f0, a0, mf0, c0 = mets(base)
print(f'release: F1 {f0:.3f} AUC {a0:.3f} macroF1 {mf0:.3f}')
alpha = 0.05 / len(rules)
for name, (code, mask) in rules.items():
    t = force(base, mask, code)
    f, a, mf, c = mets(t)
    y = base['y_' + code]; fired = mask.sum(); tp = int((mask & (y == 1)).sum()); new_def = int((mask & (base.quality_class.values == 0)).sum())
    d = np.array([[*mets(t, b)[:3]] for b in boots]) - np.array([[*mets(base, b)[:3]] for b in boots])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    print(f'{name}: срабатываний {fired}, из них нарушение {code} по разметке {tp}, новых браков {new_def} | '
          f'F1 {f0:.3f}->{f:.3f} [{lo[0]:+.3f};{hi[0]:+.3f}]  AUC {a0:.3f}->{a:.3f} [{lo[1]:+.3f};{hi[1]:+.3f}]  '
          f'macroF1 {mf0:.3f}->{mf:.3f}  F1 {code} {c0[code]:.3f}->{c[code]:.3f}')
