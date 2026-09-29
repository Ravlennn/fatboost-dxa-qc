"""Lesser trochanter (LT) detector on the medial femoral contour; no learned weights.

Anchored on landmark-model points (neck centre and neck axis give the medial side).
In DXA the LT is less dense than the cortex, so it appears as medium-density bone
protruding medially beyond the bright cortical edge, ~20-60 mm below the neck centre.
TZ: correct positioning = contour slightly deformed by the LT; over-rotation = smooth
contour; under-rotation = LT too large. See docs/METHODS_AND_MODELS.md.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MM = 0.6  # GE Lunar Prodigy render


def median5(a: np.ndarray) -> np.ndarray:
    """Running median, window 5, edges replicated (== scipy.ndimage.median_filter mode='nearest')."""
    p = np.pad(np.asarray(a, float), 2, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 5), axis=1)


def trace_medial_edge(img, y_top, y_bot, x_start):
    """Medial (right, bright->dark) edge x per row from y_bot up to y_top."""
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    xs = {}
    x = x_start
    for y in range(y_bot, y_top - 1, -1):
        lo, hi = int(max(x - 4, 1)), int(min(x + 7, img.shape[1] - 2))
        if hi <= lo:
            break
        seg = gx[y, lo:hi + 1]
        k = int(np.argmin(seg))  # strongest bright->dark transition
        if seg[k] > -4:  # no edge: keep previous estimate
            xs[y] = float(x)
            continue
        x = lo + k
        xs[y] = float(x)
    return xs


def find_shaft_edge(img, y):
    """Medial edge of the shaft at row y: rightmost strong falling edge of the widest bright run."""
    row = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)[y]
    thr = (np.percentile(row, 95) + np.percentile(row, 20)) / 2
    bright = row > thr
    runs, start = [], None
    for i, b in enumerate(np.append(bright, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append((i - start, start, i - 1))
            start = None
    if not runs:
        return None
    _, a, b = max(runs)
    return b, a


def detect(img, pts, mm: float | None = None):
    mm = float(mm or MM)
    h, w = img.shape
    head, troch, c = (np.array(pts[k], float) for k in ('neck_head_side', 'neck_troch_side', 'neck_center'))
    mirror = head[0] < troch[0]
    if mirror:  # canonical: medial side to the right
        img = img[:, ::-1].copy()
        c = np.array([w - 1 - c[0], c[1]])
    y_top = int(c[1] + 8)
    y_bot = int(min(h - 4, c[1] + 0.85 * (h - c[1])))
    if y_bot - y_top < 40:
        return None, 'too little shaft below the neck'
    edge = find_shaft_edge(img, y_bot)
    if edge is None:
        return None, 'shaft not found'
    xs = trace_medial_edge(img, y_top, y_bot, edge[0])
    ys = np.array(sorted(xs))
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    # Intensity scale: bright cortex vs soft tissue just medial of the shaft (lower shaft rows).
    low = ys[ys >= y_bot - 0.3 * (y_bot - y_top)]
    cortex = np.median([f[y, max(int(xs[y]) - 3, 0)] for y in low])
    tissue = np.median([f[y, min(int(xs[y]) + 15, w - 1)] for y in low])
    t_bone = tissue + 0.30 * (cortex - tissue)
    # Outer medial contour = cortical edge + medium-density bone protruding beyond it
    # (a faint LT lies outside the cortex; a dense LT bends the cortical edge itself).
    ext = {}
    for y in ys:
        x0 = int(xs[y])
        run = 0
        for x in range(x0 + 1, min(x0 + 30, w)):
            if f[y, x] > t_bone:
                run += 1
            else:
                break
        ext[y] = run if run < 29 else 0  # run hitting the window edge = merged with ischium
    # Anatomical window below the neck centre (~0.6 mm/px): the LT lies ~20-60 mm below it;
    # nearer the neck the grey neck tissue touches the cortex and mimics a protrusion.
    fit_y = ys[(ys >= c[1] + 35) & (ys <= min(c[1] + 95, y_bot))]
    if len(fit_y) < 20:
        return None, 'too little shaft below the neck'
    res = median5(np.array([ext[y] for y in fit_y], float))
    k1 = int(np.argmax(res))
    peak = float(res[k1])
    on = res >= max(2.0, 0.4 * peak)
    lo_i, hi_i = k1, k1
    while lo_i > 0 and on[lo_i - 1]:
        lo_i -= 1
    while hi_i < len(res) - 1 and on[hi_i + 1]:
        hi_i += 1
    area = float(res[lo_i:hi_i + 1].sum())
    visible = peak >= 5.0 and (hi_i - lo_i + 1) >= 4  # >= 3 mm, fixed after visual QA of sheet 0
    # Dense LT bends the cortical edge itself: bump of the edge above a robust line (extra feature).
    xe = np.array([xs[y] for y in fit_y], float)
    keep = np.ones(len(fit_y), bool)
    for _ in range(3):
        coef = np.polyfit(fit_y[keep], xe[keep], 1)
        r_e = xe - np.polyval(coef, fit_y)
        keep = r_e < max(0.5, np.percentile(r_e, 50))
    edge_bump = float(median5(r_e).max())
    x_out = xe + res
    y1, y2, y3 = int(fit_y[k1]), int(fit_y[lo_i]), int(fit_y[hi_i])
    apex_dx = float(x_out[k1] - xs[y1])

    def P(y, dx=0.0):
        x = xs[y] + dx
        return [float(w - 1 - x if mirror else x), float(y)]
    return {'visible': bool(visible), 'prom_px': peak, 'y_apex': y1,
            'points': {'lt_apex': P(y1, apex_dx), 'lt_base_top': P(y2), 'lt_base_bottom': P(y3)},
            'edge': [P(int(y)) for y in ys[::3]], 'mirror': bool(mirror),
            'lt_prominence_mm': peak * MM, 'lt_area_mm2': area * MM * MM,
            'lt_length_mm': (y3 - y2 + 1) * MM, 'lt_edge_bump_mm': edge_bump * MM}, None



def head_features(res: dict | None) -> dict:
    """Features used by the hip fusion heads (scripts/export_lt_bundle.py)."""
    if res is None:
        return {'lt_prom': 0.0, 'lt_prom_sq': 0.0, 'lt_area': 0.0, 'lt_edge_bump': 0.0}
    return {'lt_prom': res['lt_prominence_mm'], 'lt_prom_sq': res['lt_prominence_mm'] ** 2,
            'lt_area': math.log1p(res['lt_area_mm2']), 'lt_edge_bump': res['lt_edge_bump_mm']}
