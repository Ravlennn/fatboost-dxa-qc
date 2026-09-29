"""Геометрические правила для поясничного отдела (ТЗ п. 2.3.1).

Измерения по пикселям (DICOM-теги пустые):
- axis_angle_deg: наклон оси столба к вертикали (по границам столба, робастная регрессия Тейла–Сена);
- curvature_deg: разница углов верхней и нижней половины (сколиоз даёт большую кривизну при малом общем наклоне);
- iliac_ratio: яркость латерально от столба в нижних 15% кадра относительно середины кадра
  (гребни подвздошных костей видны → ratio высокий);
- artifact_px: число пикселей top-hat (мелкие очень контрастные объекты) вне крупных костей.
Пороги подобраны на 99 размеченных снимках, см. docs/EDA.md и outputs/eda/spine_rules_eval.csv.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import theilslopes

# Пороги правил (см. подбор в outputs/eda/spine_rules_eval.csv)
AXIS_DEG = 4.0          # ТЗ: допустимый наклон до 5°, эксперты фактически метят с ~3–4°
AXIS_MAX_CURV = 4.0     # больше → искривление (сколиоз), а не наклон укладки
ILIAC_RATIO_MIN = 1.7
ARTIFACT_PX = 80


@dataclass
class SpineMeasurements:
    axis_angle_deg: float
    curvature_deg: float
    axis_coverage: float      # доля строк, где столб найден
    iliac_ratio: float
    iliac_lateral_frac: float
    artifact_px: int
    artifact_max: float
    bone_thr: float
    left: np.ndarray          # левая граница столба по строкам (nan = не найдено)
    right: np.ndarray

    def flags(self) -> dict[str, bool]:
        axis = (not np.isnan(self.axis_angle_deg)) and abs(self.axis_angle_deg) >= AXIS_DEG \
            and (np.isnan(self.curvature_deg) or self.curvature_deg <= AXIS_MAX_CURV)
        return {
            "spine_axis_tilt": bool(axis),
            "spine_scan_range": bool(self.iliac_ratio < ILIAC_RATIO_MIN),
            "spine_artifact": bool(self.artifact_px >= ARTIFACT_PX),
        }

    def scores(self) -> dict[str, float]:
        """Монотонные скоры 0..1 для ROC/PR (не вероятности)."""
        ang = 0.0 if np.isnan(self.axis_angle_deg) else abs(self.axis_angle_deg)
        return {
            "spine_axis_tilt": float(1 / (1 + np.exp(-(ang - AXIS_DEG) * 1.2))),
            "spine_scan_range": float(1 / (1 + np.exp((self.iliac_ratio - ILIAC_RATIO_MIN) * 2.0))),
            "spine_artifact": float(1 / (1 + np.exp(-(self.artifact_px - ARTIFACT_PX) / 40.0))),
        }


def _otsu(v: np.ndarray) -> float:
    v = v[v > 8]
    if v.size < 100:
        return 128.0
    t, _ = cv2.threshold(np.clip(v, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def column_borders(a: np.ndarray, sigma: float = 4.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Границы столба по строкам: сильное размытие → Otsu → трекинг run'а от центра вверх и вниз."""
    h, w = a.shape
    sm = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
    t = _otsu(sm)
    bone = (sm > t).astype(np.uint8)
    mid = bone[int(h * .3):int(h * .7)].sum(0).astype(np.float32)
    mid = cv2.GaussianBlur(mid.reshape(1, -1), (0, 0), 8).ravel()
    lo, hi = int(w * .2), int(w * .8)
    start = lo + int(np.argmax(mid[lo:hi]))
    L = np.full(h, np.nan); R = np.full(h, np.nan)

    def runs(row):
        d = np.diff(np.concatenate([[0], row, [0]]))
        return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

    def track(order, prev):
        for y in order:
            rr = [r for r in runs(bone[y]) if 6 <= r[1] - r[0] <= w * 0.45]
            if not rr:
                continue
            best = min(rr, key=lambda r: abs((r[0] + r[1]) / 2 - prev))
            c = (best[0] + best[1]) / 2
            if abs(c - prev) > 10:
                continue
            L[y], R[y] = best; prev = c

    y0 = int(h * .5)
    track(range(y0, h), start); track(range(y0 - 1, -1, -1), start)
    return L, R, t


def segment_angle(L: np.ndarray, R: np.ndarray, lo: float, hi: float) -> float:
    h = len(L); ys = np.arange(h)
    sel = (ys >= h * lo) & (ys < h * hi) & ~np.isnan(L)
    if sel.sum() < 12:
        return float("nan")
    y = ys[sel].astype(float)
    s = (theilslopes(L[sel], y)[0] + theilslopes(R[sel], y)[0]) / 2
    return float(-np.degrees(np.arctan(s)))  # + = верх столба правее низа


def iliac_ratio(a: np.ndarray, L: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    h, w = a.shape
    sm = cv2.GaussianBlur(a.astype(np.float32), (5, 5), 1.5)
    c = np.nanmedian((L + R) / 2); wm = np.nanmedian(R - L)
    if np.isnan(c):
        c, wm = w / 2, 40
    x0, x1 = int(max(0, c - wm * 0.75 - 10)), int(min(w, c + wm * 0.75 + 10))

    def lat_mean(y0, y1):
        band = sm[int(h * y0):int(h * y1)]
        m = np.ones(band.shape, bool); m[:, x0:x1] = False
        return float(band[m].mean()) if m.any() else 0.0

    ratio = lat_mean(.85, 1.0) / (lat_mean(.35, .65) + 1e-6)
    t = _otsu(sm) * 0.75
    band = sm[int(h * .8):]; m = np.ones(band.shape, bool); m[:, x0:x1] = False
    b = (band > t) & m
    frac = min(b[:, :int(c)].mean() if int(c) > 0 else 0.0, b[:, int(c):].mean() if int(c) < w else 0.0)
    return float(ratio), float(frac)


def artifact_pixels(a: np.ndarray, thr_tophat: float = 40.0) -> tuple[int, float]:
    h, w = a.shape
    sm = cv2.GaussianBlur(a.astype(np.float32), (5, 5), 1.5)
    t = _otsu(sm)
    bone = (sm > t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bone, connectivity=8)
    big = np.zeros_like(bone)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 0.004 * h * w:
            big[lab == k] = 1
    big = cv2.dilate(big, np.ones((7, 7), np.uint8))
    th = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    outside = (big == 0) & (a > 8)
    px = int(((th > thr_tophat) & outside).sum())
    mx = float(th[outside].max()) if outside.any() else 0.0
    return px, mx


def measure_spine(pixels: np.ndarray) -> SpineMeasurements:
    a = pixels.astype(np.float32)
    L, R, t = column_borders(a)
    ang = segment_angle(L, R, .05, .85)
    top, bot = segment_angle(L, R, .05, .45), segment_angle(L, R, .45, .85)
    curv = float("nan") if (np.isnan(top) or np.isnan(bot)) else abs(top - bot)
    ratio, frac = iliac_ratio(a, L, R)
    px, mx = artifact_pixels(a)
    return SpineMeasurements(ang, curv, float(np.mean(~np.isnan(L))), ratio, frac, px, mx, t, L, R)
