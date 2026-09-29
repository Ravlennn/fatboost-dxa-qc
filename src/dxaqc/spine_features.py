"""Numerical spine descriptors for learned heads; no quality-rule decisions."""
from __future__ import annotations

import warnings

import cv2
import numpy as np

from .spine_rules import measure_spine

FEATURE_VERSION = "spine_geometry_v1_height400"
VALUE_NAMES = ["axis_abs_deg", "curvature_deg", "axis_coverage", "log1p_iliac_ratio", "iliac_lateral_frac",
               "artifact_fraction", "artifact_max_255", "bone_threshold_255", "column_width_fraction",
               "column_center_fraction", "aspect_ratio", "foreground_fraction", "intensity_mean", "intensity_std",
               "mass_center_x", "mass_center_y", "mass_std_x", "mass_std_y", "mass_cov_xy",
               "top_intensity_fraction", "bottom_intensity_fraction"]
FEATURE_NAMES = VALUE_NAMES + [name + "_missing" for name in VALUE_NAMES]


def spine_features(pixels: np.ndarray) -> np.ndarray:
    """Return 42 finite descriptors, including explicit missing-value indicators.

    The existing extractor uses pixel-sized morphology kernels. Isotropic resize
    to height400 makes those kernels comparable while preserving image angles.
    These are image descriptors, not anatomically verified measurements or mm.
    """
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or min(pixels.shape) < 16:
        raise ValueError("Spine descriptors expect a nonempty grayscale uint8 image")
    original_h, original_w = pixels.shape
    width = max(16, int(round(original_w * 400 / original_h)))
    if width > 1600:
        raise ValueError("Unsupported aspect ratio for spine descriptors")
    a = cv2.resize(pixels, (width, 400), interpolation=cv2.INTER_LINEAR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = measure_spine(a)
    h, w = a.shape
    valid = np.isfinite(m.left) & np.isfinite(m.right)
    col_width = float(np.median((m.right-m.left)[valid]) / w) if valid.any() else np.nan
    col_center = float(np.median((m.right+m.left)[valid]/2) / w) if valid.any() else np.nan
    weight = a.astype(np.float64) / 255.
    total = float(weight.sum())
    yy, xx = np.indices(a.shape, dtype=np.float64)
    xx, yy = xx / max(w-1, 1), yy / max(h-1, 1)
    if total > 0:
        cx, cy = float((weight*xx).sum()/total), float((weight*yy).sum()/total)
        sx, sy = float(np.sqrt((weight*(xx-cx)**2).sum()/total)), float(np.sqrt((weight*(yy-cy)**2).sum()/total))
        covariance = float((weight*(xx-cx)*(yy-cy)).sum()/total)
        top, bottom = float(weight[:h//5].sum()/total), float(weight[-h//5:].sum()/total)
    else:
        cx = cy = sx = sy = covariance = top = bottom = np.nan
    values = np.asarray([abs(m.axis_angle_deg), m.curvature_deg, m.axis_coverage,
                         np.log1p(np.maximum(0., m.iliac_ratio)), m.iliac_lateral_frac, m.artifact_px/(h*w),
                         m.artifact_max/255., m.bone_thr/255., col_width, col_center, original_w/original_h,
                         float((a > 8).mean()), float(weight.mean()), float(weight.std()), cx, cy, sx, sy,
                         covariance, top, bottom], dtype=np.float64)
    missing = ~np.isfinite(values)
    result = np.concatenate([np.where(missing, 0., values), missing.astype(np.float64)])
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid spine descriptors")
    return result
