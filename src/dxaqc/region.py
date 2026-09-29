"""Определение анатомической области и стороны по пикселям (DICOM-теги пустые)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPINE = "lumbar_spine"
HIP_RIGHT = "hip_right"
HIP_LEFT = "hip_left"


@dataclass
class RegionResult:
    region: str          # lumbar_spine | hip_right | hip_left
    spine_score: float   # зеркальная симметрия: позвоночник ~0.5–0.9, бедро ~ -0.3–0.35
    side_score: float    # >0 правое бедро, <0 левое; 0 для позвоночника
    method: str


def mirror_symmetry(a: np.ndarray) -> float:
    a = a.astype(np.float64)
    a = a - a.mean()
    f = a[:, ::-1]
    denom = np.sqrt((a * a).sum() * (f * f).sum())
    return float((a * f).sum() / denom) if denom > 0 else 0.0


def hip_side_score(a: np.ndarray) -> float:
    """Центроид кости по X в верхних 30% минус в нижних 30%.
    Таз (медиально) в верхней части, диафиз (латерально) внизу.
    >0: медиальная сторона справа → правое бедро (рентгеновская конвенция)."""
    a = a.astype(np.float64)
    h, w = a.shape
    thr = np.percentile(a, 85)

    def cx(band: np.ndarray) -> float:
        m = band > thr
        return float(np.where(m)[1].mean() / w) if m.any() else 0.5

    return cx(a[: int(h * 0.3)]) - cx(a[int(h * 0.7):])


def classify_region(pixels: np.ndarray) -> RegionResult:
    h, w = pixels.shape
    sym = mirror_symmetry(pixels)
    # Основной признак — ширина кадра GE Lunar (300 позвоночник, 280/248 бедро),
    # проверено на 252 изображениях. Запасной — симметрия, если ширина нестандартная.
    if w >= 295:
        is_spine, method = True, "width"
    elif w <= 290:
        is_spine, method = False, "width"
    else:
        is_spine, method = sym > 0.4, "symmetry"
    # Защита от противоречия: ширина говорит одно, симметрия резко другое
    if method == "width" and ((is_spine and sym < 0.0) or (not is_spine and sym > 0.6)):
        is_spine, method = sym > 0.4, "symmetry_override"
    if is_spine:
        return RegionResult(SPINE, sym, 0.0, method)
    side = hip_side_score(pixels)
    return RegionResult(HIP_RIGHT if side > 0 else HIP_LEFT, sym, side, method)
