"""Геометрия проксимального отдела бедра (ТЗ п. 2.3.2). Все измерения в «правой» ориентации:
диафиз слева, таз справа (левое бедро зеркалится вызывающим кодом).

- shaft_*: границы диафиза, найденные трекингом снизу вверх;
- lt_bump_px / lt_bump_mm: выступ малого вертела над линией медиального контура диафиза;
- lateral_margin_px: расстояние от самой латеральной кости (большой вертел) до латерального края;
- top_lateral_bone: доля кости в верхних 8% строк латеральной половины (головка срезана верхом кадра);
- bottom_margin_px: от уровня малого вертела до нижнего края.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HipMeasurements:
    shaft_rows: tuple[int, int]        # (y_top, y_bottom) отслеженного диафиза
    shaft_width_px: float
    lt_bump_px: float                  # высота выступа малого вертела (px)
    lt_bump_area_px: float             # площадь выступа
    lt_row: int                        # строка максимума выступа
    lateral_margin_px: float
    shaft_touches_lateral: bool
    top_lateral_bone: float
    bottom_margin_px: float
    bone_frac: float
    lateral: np.ndarray                # x латеральной границы по строкам (nan)
    medial: np.ndarray
    medial_base: np.ndarray            # базовая линия медиального контура


def _otsu(v: np.ndarray) -> float:
    v = v[v > 8]
    if v.size < 100:
        return 128.0
    t, _ = cv2.threshold(np.clip(v, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _runs(row: np.ndarray):
    d = np.diff(np.concatenate([[0], row, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def measure_hip(pixels: np.ndarray, spacing_mm: float = 0.6) -> HipMeasurements:
    a = pixels.astype(np.float32); h, w = a.shape
    sm = cv2.GaussianBlur(a, (0, 0), 2.0)
    t = _otsu(sm)
    bone = (sm > t).astype(np.uint8)
    bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lat = np.full(h, np.nan); med = np.full(h, np.nan)

    # старт: нижние 5% строк, run в левых 65% ширины с макс. шириной
    y_start = None; prev = None
    for y in range(h - 1, int(h * 0.9), -1):
        rr = [r for r in _runs(bone[y]) if (r[0] + r[1]) / 2 < w * 0.65 and 8 <= r[1] - r[0] <= w * 0.5]
        if rr:
            best = max(rr, key=lambda r: r[1] - r[0]); prev = best; y_start = y; break
    widths = []
    if y_start is not None:
        lat[y_start], med[y_start] = prev; widths.append(prev[1] - prev[0])
        for y in range(y_start - 1, -1, -1):
            rr = _runs(bone[y])
            cand = [r for r in rr if r[0] < prev[1] + 6 and r[1] > prev[0] - 6]  # перекрывается с предыдущим
            if not cand:
                break
            r = max(cand, key=lambda r: min(r[1], prev[1]) - max(r[0], prev[0]))
            wmed = np.median(widths[-15:]) if widths else r[1] - r[0]
            if (r[1] - r[0]) > 2.6 * wmed or y < h * 0.15:   # слились с тазом / вышли в область шейки
                break
            lat[y], med[y] = r; widths.append(r[1] - r[0]); prev = r
    ys = np.where(~np.isnan(lat))[0]
    if len(ys) < 20:
        return HipMeasurements((0, 0), float("nan"), float("nan"), float("nan"), -1, float("nan"), False,
                               float("nan"), float("nan"), float(bone.mean()), lat, med, np.full(h, np.nan))
    y_top, y_bot = int(ys.min()), int(ys.max())
    shaft_w = float(np.median(widths[: max(10, len(widths) // 3)]))
    # базовая линия медиального контура: линейная подгонка по нижней трети отслеженных строк
    n = len(ys); base_rows = ys[-max(10, n // 3):]
    p = np.polyfit(base_rows.astype(float), med[base_rows], 1)
    base = np.full(h, np.nan); base[ys] = np.polyval(p, ys.astype(float))
    dev = med[ys] - base[ys]                       # + = медиальный контур выступает вправо (к тазу)
    upper = ys < (y_bot - n // 3)                  # выступ ищем выше базовой зоны
    if upper.any():
        dv = np.where(upper, dev, -np.inf); i = int(np.argmax(dv))
        bump = float(max(0.0, dv[i])); lt_row = int(ys[i])
        area = float(np.clip(dev[upper], 0, None).sum())
    else:
        bump, lt_row, area = 0.0, -1, 0.0
    # латеральное поле: самая левая кость в верхних 70% строк (большой вертел / диафиз)
    cols_any = np.where(bone[: int(h * 0.7)].any(axis=0))[0]
    lateral_margin = float(cols_any.min()) if len(cols_any) else float("nan")
    touches = bool(bone[:, :2].any())
    # верх латерально: доля кости в верхних 8% строк и левых 45% столбцов
    top_lat = float(bone[: max(3, int(h * 0.08)), : int(w * 0.45)].mean())
    bottom_margin = float(h - lt_row) if lt_row >= 0 else float("nan")
    return HipMeasurements((y_top, y_bot), shaft_w, bump, area, lt_row, lateral_margin, touches, top_lat, bottom_margin,
                           float(bone.mean()), lat, med, base)



def flare_row(m: HipMeasurements, thr_px: float = 5.0, run: int = 5) -> int | None:
    """Строка начала «расширения» медиального контура (уровень малого вертела).
    Берём непрерывную зону отклонения > thr_px, которая доходит до верха отслеженного сегмента,
    и возвращаем её нижнюю границу. Изгибы диафиза внизу так не ловятся."""
    ys = np.where(~np.isnan(m.medial) & ~np.isnan(m.medial_base))[0]
    if len(ys) < 20:
        return None
    dev = m.medial[ys] - m.medial_base[ys]
    above = dev > thr_px
    if above[:run].sum() < run - 1:   # у верхнего конца отклонения нет → расширение не достигнуто
        return None
    i = 0
    while i < len(ys) and above[i]:
        i += 1
    return int(ys[max(0, i - 1)])


def lt_crop(pixels: np.ndarray, m: HipMeasurements | None = None, size: int = 144) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Кроп зоны малого вертела в нативном разрешении (ориентация «правое бедро»).
    Центр: строка начала «расширения» медиального контура (flare_row), сдвиг вверх на 0.2·size и медиально на 0.1·size,
    чтобы выступ вертела и переход в шейку попали в окно. Если контур не найден — верх отслеженного диафиза,
    если и его нет — точка (0.45w, 0.45h)."""
    h, w = pixels.shape
    if m is None:
        m = measure_hip(pixels)
    ys = np.where(~np.isnan(m.medial))[0]
    fr = flare_row(m)
    if fr is not None:
        cx, cy = int(m.medial[fr]) + int(size * 0.1), fr - int(size * 0.2)
    elif len(ys) >= 20:
        y_ref = int(ys.min()); cx, cy = int(m.medial[y_ref]) + int(size * 0.1), y_ref - int(size * 0.25)
    else:
        cx, cy = int(w * 0.45), int(h * 0.45)
    x0, y0 = cx - size // 2, cy - size // 2
    pad = size
    big = np.zeros((h + 2 * pad, w + 2 * pad), pixels.dtype); big[pad:pad + h, pad:pad + w] = pixels
    crop = big[y0 + pad:y0 + pad + size, x0 + pad:x0 + pad + size]
    return crop, (x0, y0, x0 + size, y0 + size)
