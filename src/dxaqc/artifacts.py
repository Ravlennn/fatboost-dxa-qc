"""Foreign-object (artifact) detector for AP lumbar spine DXA; no learned weights.

Anchored on the landmark model (L1-L4 column axis and vertebra height). Two channels:
  * wire: thin bright ridges (bra underwires, cables) — strong Hessian ridge response at
    a small scale, grouped into long, elongated, mostly horizontal components outside the
    vertebral column (ribs are broader, broken and oblique); axis-aligned
    straight lines (frame/padding edges) are ignored;
  * metal: compact near-saturated blobs (hooks, clips, buttons, hardware) outside the
    column; on the column only if far brighter than its neighbourhood (normalised renders
    saturate dense bone as well).
Designed on unlabelled image sheets (quality labels hidden), see docs/METHODS_AND_MODELS.md.
"""
from __future__ import annotations

import cv2
import numpy as np

SEQ = ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')


def _ridge(f: np.ndarray, sigma: float) -> np.ndarray:
    """Bright-ridge strength: -(largest-magnitude negative Hessian eigenvalue) * sigma^2."""
    g = cv2.GaussianBlur(f, (0, 0), sigma)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    l2 = (dxx + dyy - tmp) / 2  # most negative eigenvalue
    return np.clip(-l2, 0, None) * sigma ** 2


def column_mask(shape, points: dict, half_width_vert: float = 0.85) -> tuple[np.ndarray, float]:
    """Mask of the vertebral column band around the L1-L4 axis, extended to the frame."""
    h, w = shape
    pts = np.array([points[k] for k in SEQ], float)
    vh = float(max(np.median(np.diff(pts[:, 1])), 1.0))
    fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
    ys = np.arange(h)
    xc = np.polyval(fit, ys)
    xx = np.arange(w)[None, :]
    return np.abs(xx - xc[:, None]) <= half_width_vert * vh, vh


def detect(pixels: np.ndarray, points: dict, mm: float = 0.6) -> dict:
    h, w = pixels.shape
    f = pixels.astype(np.float32)
    col, vh = column_mask(pixels.shape, points)
    border = np.zeros((h, w), bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    # Padding edges (constant-zero regions from the scanner render) create straight steps.
    pad = cv2.dilate((pixels <= 2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    # ---- wire channel
    r_thin = _ridge(f, 1.2)
    bg = cv2.medianBlur(pixels, 15).astype(np.float32)
    contrast = f - bg
    # Continuous strong thin ridges; wires are the longest ones outside the column (ribs give
    # weaker, broken thin-scale responses and run obliquely).
    thin = (r_thin > 40) & (contrast > 10) & ~col & ~border & ~pad
    thin = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, 8)
    wires, wire_len, wire_mask = [], 0.0, np.zeros((h, w), np.uint8)
    for k in range(1, n):
        x, y, bw, bh, area = st[k]
        length = float(np.hypot(bw, bh))
        if length < 0.45 * vh or area < 12:
            continue
        if bw >= 0.9 * max(bw, bh) and bh <= 2 or bh >= 0.9 * max(bw, bh) and bw <= 2:
            continue  # perfectly straight axis-aligned: frame edge
        ys_, xs_ = np.nonzero(lab == k)
        if len(xs_) < 10:
            continue
        # Orientation from principal axis; wires across the chest/abdomen are mostly horizontal.
        cov = np.cov(np.stack([xs_, ys_]))
        ev, evec = np.linalg.eigh(cov)
        ang = abs(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
        ang = min(ang, 180 - ang)
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        if ang > 40 or elong < 4:
            continue
        wires.append({'bbox': [int(x), int(y), int(bw), int(bh)], 'length_px': length, 'angle': float(ang),
                      'contrast': float(contrast[lab == k].mean())})
        wire_len += length
        wire_mask[lab == k] = 1

    # ---- metal channel
    # Normalised renders saturate dense bone too, so metal = near-saturated AND much brighter
    # than its 15x15 neighbourhood (a clip on a vertebra) or saturated off the column.
    # Off the column local contrast still matters: saturated iliac crests are part of a large
    # bright bone, a hook or button is brighter than the soft tissue around it.
    sat = (pixels >= 245) & np.where(col, contrast > 90, contrast > 60) & ~border
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sat.astype(np.uint8), 8)
    metal_area, blobs = 0, []
    for k in range(1, n2):
        x, y, bw, bh, area = st2[k]
        on_col = col[lab2 == k].mean() > 0.5
        if area < (6 if on_col else 4) or area > 0.05 * h * w:
            continue
        blobs.append({'bbox': [int(x), int(y), int(bw), int(bh)], 'area_px': int(area), 'on_column': bool(on_col)})
        metal_area += area
    mask = (wire_mask > 0) | np.isin(lab2, [k for k in range(1, n2)
                                              if any(b['bbox'] == [int(v) for v in st2[k, :4]] for b in blobs)])
    return {'wire_len_vert': wire_len / vh, 'n_wires': len(wires), 'metal_area_mm2': metal_area * mm * mm,
            'n_metal': len(blobs), 'wires': wires, 'metal': blobs, 'mask': mask.astype(np.uint8)}


def head_features(res: dict | None) -> dict:
    if res is None:
        return {'art_wire': 0.0, 'art_metal': 0.0, 'art_any': 0.0}
    wire, metal = np.log1p(res['wire_len_vert']), np.log1p(res['metal_area_mm2'])
    return {'art_wire': float(wire), 'art_metal': float(metal), 'art_any': float(max(wire, metal))}
