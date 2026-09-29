"""Anatomical landmark heatmap model (hip ROI / femoral neck, lumbar L1-L4).

Trained on weak labels extracted from GE Lunar overlays (Arak, see
docs/METHODS_AND_MODELS.md). Images use our DICOM convention: bone bright, uint8.
Coordinates are (x, y) in the original image pixels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HIP_POINTS = ['roi_top', 'roi_bottom', 'roi_left', 'roi_right', 'neck_center', 'neck_head_side', 'neck_troch_side']
SPINE_POINTS = ['L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom']
POINTS = HIP_POINTS + SPINE_POINTS
HIP_CH = list(range(len(HIP_POINTS)))
SPINE_CH = list(range(len(HIP_POINTS), len(POINTS)))
INPUT = 256
STRIDE = 4
NECK_HALF = 30.0  # px along the neck axis for the two axis points
# v2 points (manual labels on our DICOMs): lesser trochanter, Th12, iliac crests.
NEW_HIP_POINTS = ['lt_apex', 'lt_base_top', 'lt_base_bottom']
NEW_SPINE_POINTS = ['th12_mid', 'iliac_left_top', 'iliac_right_top']
POINTS_V2 = POINTS + NEW_HIP_POINTS + NEW_SPINE_POINTS
_MIRROR = {'roi_left': 'roi_right', 'roi_right': 'roi_left',
           'iliac_left_top': 'iliac_right_top', 'iliac_right_top': 'iliac_left_top'}


def flip_perm(points: list[str]) -> list[int]:
    """Channel permutation under horizontal flip (left/right-named points swap)."""
    return [points.index(_MIRROR.get(n, n)) for n in points]


def region_names(points: list[str], region: str) -> list[str]:
    spine = region.startswith('spine') or region == 'lumbar_spine'
    allowed = SPINE_POINTS + NEW_SPINE_POINTS if spine else HIP_POINTS + NEW_HIP_POINTS
    return [n for n in points if n in allowed]


FLIP_PERM = flip_perm(POINTS)


def region_channels(region: str) -> list[int]:
    return SPINE_CH if region.startswith('spine') or region == 'lumbar_spine' else HIP_CH


class LandmarkNet(nn.Module):
    """ConvNeXt-Tiny encoder + small FPN decoder -> heatmaps at stride 4."""

    def __init__(self, n_points: int = len(POINTS), pretrained: bool = True, width: int = 128):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = torchvision.models.convnext_tiny(weights=weights).features
        chans = [96, 192, 384, 768]
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in chans)
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
        )
        self.head = nn.Conv2d(width, n_points, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):  # x: (B,1,H,W) in [0,1]
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (1, 3, 5, 7):
                feats.append(x)
        p = self.lateral[3](feats[3])
        for k in (2, 1, 0):
            p = F.interpolate(p, size=feats[k].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[k](feats[k])
        return self.head(self.smooth(p))


def letterbox(img: np.ndarray, size: int = INPUT):
    """Resize keeping aspect, pad bottom/right with 0. Returns canvas, scale."""
    import cv2
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.zeros((size, size), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s


def decode(heat: torch.Tensor, win: int = 3):
    """Heatmaps (B,K,h,w) -> coords (B,K,2) in input pixels, peak values (B,K).
    Argmax, then soft-argmax refinement in a (2*win+1)^2 window."""
    B, K, h, w = heat.shape
    flat = heat.reshape(B, K, -1)
    peak, idx = flat.max(-1)
    py, px = (idx // w).float(), (idx % w).float()
    ys = torch.arange(h, device=heat.device).float().view(1, 1, h, 1)
    xs = torch.arange(w, device=heat.device).float().view(1, 1, 1, w)
    m = ((ys - py[..., None, None]).abs() <= win) & ((xs - px[..., None, None]).abs() <= win)
    wgt = heat.clamp(min=0) * m
    tot = wgt.sum((-1, -2)).clamp(min=1e-6)
    cx = (wgt * xs).sum((-1, -2)) / tot
    cy = (wgt * ys).sum((-1, -2)) / tot
    coords = torch.stack([cx, cy], -1) * STRIDE + (STRIDE - 1) / 2
    return coords, peak


class LandmarkPredictor:
    def __init__(self, checkpoint: str, device: str = 'cpu'):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.points = list(ck.get('points', POINTS))
        self.perm = flip_perm(self.points)
        self.model = LandmarkNet(n_points=len(self.points), pretrained=False)
        self.model.load_state_dict(ck['model'])
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, img: np.ndarray, region: str) -> dict:
        canvas, s = letterbox(img)
        x = torch.from_numpy(canvas).float().div(255)[None, None].to(self.device)
        heat = self.model(x)
        # Test-time flip averaging; ROI left/right channels swap under flip.
        heat_f = torch.flip(self.model(torch.flip(x, [-1])), [-1])[:, self.perm]
        heat = (heat + heat_f) / 2
        coords, peak = decode(heat)
        names = region_names(self.points, region)
        ch = [self.points.index(n) for n in names]
        pts = (coords[0, ch] / s).cpu().numpy()
        return {'points': {n: [float(p[0]), float(p[1])] for n, p in zip(names, pts)},
                'confidence': {n: float(v) for n, v in zip(names, peak[0, ch].cpu().numpy())}}


# ---------------------------------------------------------------- measurements
MM_PER_PX = 0.6  # GE Lunar Prodigy render, docs/EDA.md
SPINE_SEQ = ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')
TILT_SLOPE = 0.921  # measured vs applied rotation, scripts/calibrate_geometry.py (label-free)


def measurements(points: dict, region: str, shape, pixels: np.ndarray | None = None,
                 mm_per_px: float | None = None) -> dict:
    """TZ-style geometric measurements from landmark points (image px).
    With pixels, spine measurements also include iliac crest visibility."""
    import math
    h, w = shape
    mm = float(mm_per_px or MM_PER_PX)
    if pixels is not None and region in ('spine', 'lumbar_spine'):
        return {**measurements(points, region, shape, mm_per_px=mm), **iliac_features(pixels, points)}
    p = {k: np.asarray(v, float) for k, v in points.items()}
    if region in ('spine', 'lumbar_spine'):
        seq = [p[k] for k in SPINE_SEQ]
        d = seq[-1] - seq[0]
        xs, ys = np.array([q[0] for q in seq]), np.array([q[1] for q in seq])
        fit = np.polyfit(ys, xs, 1)
        vh = float(max(np.median(np.diff(ys)), 1.0))
        tilt = math.degrees(math.atan2(d[0], d[1]))
        return {'spine_tilt_deg': tilt,
                # Display/DICOM SR value: synthetic-rotation calibration shows an 8% under-
                # estimation (docs/METHODS_AND_MODELS.md). Heads keep using the raw value.
                'spine_tilt_deg_calibrated': tilt / TILT_SLOPE,
                'spine_tilt_fit_deg': math.degrees(math.atan(fit[0])),
                'spine_lateral_dev_px': float(np.abs(xs - np.polyval(fit, ys)).max()),
                'vertebra_height_px': vh,
                'top_margin_vert': float(seq[0][1] / vh), 'bottom_margin_vert': float((h - seq[-1][1]) / vh),
                'top_margin_mm': float(seq[0][1] * mm), 'bottom_margin_mm': float((h - seq[-1][1]) * mm),
                'mm_per_px': mm}
    d = p['neck_troch_side'] - p['neck_head_side']
    return {'roi_top_mm': float(p['roi_top'][1] * mm), 'roi_bottom_mm': float((h - p['roi_bottom'][1]) * mm),
            'roi_left_mm': float(p['roi_left'][0] * mm), 'roi_right_mm': float((w - p['roi_right'][0]) * mm),
            'mm_per_px': mm,
            'neck_axis_deg': math.degrees(math.atan2(d[1], abs(d[0]))),
            'neck_center_x_rel': float(p['neck_center'][0] / w), 'neck_center_y_rel': float(p['neck_center'][1] / h)}


def head_features(m: dict, region: str) -> dict:
    """Named features used by landmark reason heads (scripts/run_landmark_fusion.py)."""
    if region in ('spine', 'lumbar_spine'):
        return {'abs_tilt': abs(m['spine_tilt_deg']), 'abs_tilt_fit': abs(m['spine_tilt_fit_deg']),
                'lateral_dev': m['spine_lateral_dev_px'] / m['vertebra_height_px'],
                'top_margin': m['top_margin_vert'], 'bottom_margin': m['bottom_margin_vert'],
                'min_margin': min(m['top_margin_vert'], m['bottom_margin_vert']),
                **({'iliac_max': m['iliac_frac_max'], 'iliac_min': m['iliac_frac_min'],
                    'iliac_top_rel': m['iliac_top_rel'], 'frame_height_vert': m['frame_height_vert'],
                    'above_L4_vert': m['above_L4_vert']} if 'iliac_frac_max' in m else {})}
    right = region == 'hip_right'
    lat = m['roi_left_mm'] if right else m['roi_right_mm']
    med = m['roi_right_mm'] if right else m['roi_left_mm']
    return {'roi_top': m['roi_top_mm'], 'roi_bottom': m['roi_bottom_mm'], 'roi_lateral': lat, 'roi_medial': med,
            'roi_min': min(m['roi_top_mm'], m['roi_bottom_mm'], lat, med),
            'neck_angle': m['neck_axis_deg'], 'neck_y': m['neck_center_y_rel']}


def apply_head(head: dict, feats: dict) -> float:
    """Standardized logistic head exported as JSON.

    'features' is either a list of named features or a string tag for a raw vector
    (the view gate runs on the 1024-dim frozen DenseNet embedding, keys f0..f1023)."""
    names = head['features'] if isinstance(head['features'], list) else [f'f{i}' for i in range(len(head['mean']))]
    x = (np.array([feats[f] for f in names]) - np.array(head['mean'])) / np.array(head['scale'])
    z = float(x @ np.array(head['coef']) + head['intercept'])
    return float(1 / (1 + np.exp(-z)))


# --------------------------------------------------------------------- overlay
def draw_overlay(pixels: np.ndarray, region: str, points: dict, m: dict, violations: list[str],
                 quality_class: int, scale: int = 3, lt: dict | None = None,
                 art_mask: np.ndarray | None = None) -> np.ndarray:
    """RGB visualisation of landmarks, measurements and verdict (TZ 2.6)."""
    import cv2
    img = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    P = {k: (int(round(x * scale)), int(round(y * scale))) for k, (x, y) in points.items()}
    bad_c, ok_c, info_c = (255, 60, 60), (60, 220, 90), (255, 200, 0)
    lines = []
    if region in ('spine', 'lumbar_spine'):
        seq = [P[k] for k in SPINE_SEQ]
        axis_c = bad_c if 'spine_axis_tilt' in violations else ok_c
        cv2.line(v, seq[0], seq[-1], axis_c, 2)
        top = (seq[0][0], 0)
        cv2.line(v, top, (seq[0][0], v.shape[0] - 1), (160, 160, 160), 1)  # vertical reference
        rng_c = bad_c if 'spine_scan_range' in violations else info_c
        for name, q in zip(SPINE_SEQ, seq):
            cv2.line(v, (q[0] - 25 * scale, q[1]), (q[0] + 25 * scale, q[1]), rng_c, 1)
        for lab, a, b in zip(('L1', 'L2', 'L3', 'L4'), seq[:-1], seq[1:]):
            cv2.putText(v, lab, (a[0] + 27 * scale, (a[1] + b[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_c, 1)
        lines += [f"tilt {m.get('spine_tilt_deg_calibrated', m['spine_tilt_deg']):+.1f} deg "
                  f"(limit 5, measurement error ~0.7 deg)",
                  f"above L1 {m['top_margin_vert']:.2f} vert, below L4 {m['bottom_margin_vert']:.2f} vert"]
        if art_mask is not None and art_mask.any():
            big = cv2.resize(art_mask, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            v[big] = (255, 60, 60) if 'spine_artifact' in violations else (255, 200, 0)
            lines.append('foreign object found (wire/metal)')
        if 'iliac_frac_max' in m:
            lines.append(f"iliac crests in frame: {m['iliac_frac_max']:.2f} (0 = not visible)")
            vh = max(np.median(np.diff([q[1] for q in seq])), 1)
            y0 = int(seq[-1][1] - vh)
            for a, b in ((seq[-1][0] - 3.2 * vh, seq[-1][0] - 1.0 * vh), (seq[-1][0] + 1.0 * vh, seq[-1][0] + 3.2 * vh)):
                cv2.rectangle(v, (int(a), y0), (int(b), v.shape[0] - 1), rng_c, 1)
    else:
        roi_c = bad_c if 'hip_roi_margins' in violations else info_c
        cv2.rectangle(v, (P['roi_left'][0], P['roi_top'][1]), (P['roi_right'][0], P['roi_bottom'][1]), roi_c, 2)
        cv2.line(v, P['neck_head_side'], P['neck_troch_side'], bad_c if 'hip_positioning' in violations else ok_c, 2)
        cv2.circle(v, P['neck_center'], 4, ok_c, -1)
        lines += [f"ROI margins mm: top {m['roi_top_mm']:.0f} bottom {m['roi_bottom_mm']:.0f}",
                  f"left {m['roi_left_mm']:.0f} right {m['roi_right_mm']:.0f} (need 30/30/20)"]
        if lt:
            pc = bad_c if 'hip_positioning' in violations else ok_c
            q = [(int(round(a * scale)), int(round(b * scale))) for a, b in
                 (lt['points'][k] for k in ('lt_base_top', 'lt_apex', 'lt_base_bottom'))]
            cv2.polylines(v, [np.array(q, np.int32)], False, pc, 2)
            cv2.circle(v, q[1], 4, pc, -1)
            lines.append(f"lesser trochanter {lt['lt_prominence_mm']:.1f} mm "
                         f"({'visible' if lt['visible'] else 'not visible'}; normal: small bump)")
    verdict = 'OK' if not quality_class else 'DEFECT: ' + ', '.join(violations or ['unspecified'])
    lines = [verdict] + lines
    pad = 18 * len(lines) + 8
    out = np.zeros((v.shape[0] + pad, v.shape[1], 3), np.uint8)
    out[:v.shape[0]] = v
    for k, t in enumerate(lines):
        cv2.putText(out, t, (6, v.shape[0] + 18 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (bad_c if k == 0 and quality_class else (ok_c if k == 0 else (230, 230, 230))), 1, cv2.LINE_AA)
    return out


def iliac_features(pixels: np.ndarray, points: dict) -> dict:
    """Iliac crest visibility (TZ: lower scan edge must show the iliac crests).

    Anchored on predicted L4 bottom and vertebra height: two lateral windows from
    one vertebra above L4 bottom to the image bottom. Bone threshold is relative
    to the L3-L4 vertebral body brightness, so it is exposure-independent."""
    h, w = pixels.shape
    ys = [points[k][1] for k in SPINE_SEQ]
    vh = float(max(np.median(np.diff(ys)), 1.0))
    x4, y4 = points['L4_bottom']
    x3, y3 = points['L3L4']
    body = pixels[int(max(y3, 0)):int(max(y4, y3 + 1)), int(max(x3 - 0.4 * vh, 0)):int(min(x3 + 0.4 * vh, w))]
    ref = float(np.median(body)) if body.size else float(np.percentile(pixels, 90))
    thr = 0.5 * max(ref, 1.0)
    y0 = int(np.clip(y4 - 1.0 * vh, 0, h - 1))
    fr = []
    for a, b in ((x4 - 3.2 * vh, x4 - 1.0 * vh), (x4 + 1.0 * vh, x4 + 3.2 * vh)):
        a, b = int(np.clip(a, 0, w)), int(np.clip(b, 0, w))
        win = pixels[y0:, a:b]
        # A window cut off by the frame border counts only for the part inside the image.
        fr.append(float((win > thr).mean()) if win.size else 0.0)
    # Top of the lateral bone (iliac crest) relative to the L4 bottom, in vertebra heights.
    # Anchored at the BOTTOM: unlike the margin above L1, this does not move when the frame
    # is cropped at the top (see docs/METHODS_AND_MODELS.md).
    tops = []
    for a, b in ((x4 - 3.2 * vh, x4 - 1.0 * vh), (x4 + 1.0 * vh, x4 + 3.2 * vh)):
        a, b = int(np.clip(a, 0, w)), int(np.clip(b, 0, w))
        win = pixels[y0:, a:b]
        if win.size:
            rows = (win > thr).mean(1)
            hit = np.where(rows > 0.25)[0]
            tops.append((y0 + hit[0] - y4) / vh if len(hit) else 3.0)
        else:
            tops.append(3.0)
    return {'iliac_frac_min': min(fr), 'iliac_frac_max': max(fr), 'iliac_frac_mean': float(np.mean(fr)),
            'below_L4_vert': float((h - y4) / vh),
            'iliac_top_rel': float(np.clip(min(tops), -3, 3)),
            'frame_height_vert': float(h / vh),
            'above_L4_vert': float(y4 / vh)}
