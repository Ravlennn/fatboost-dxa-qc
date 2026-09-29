"""Extract GE Lunar analysis overlays from the Arak DXA archive.

Arak PNGs are Lunar report renders: bone is dark, and the analysis geometry
(ROI box, femoral neck box and axis, L1-L4 boundaries) is drawn with the exact
grey value 64. This script turns those drawings into weak anatomical labels and
writes overlay-free images inverted to our DICOM convention (bone bright).

Outputs:
  data/external/arak_dxa/clean/<stem>.png       inpainted, inverted image
  outputs/arak_overlays_v1/annotations.jsonl    one record per archive image
  outputs/arak_overlays_v1/summary.json         counts per view / parse status
  outputs/arak_overlays_v1/audit_{hip,spine,failed}.jpg  visual audit sheets

Labels are what the Lunar software/operator drew, not verified anatomy and not
quality labels.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LINE_VALUE = 64
ROOT = Path(__file__).resolve().parents[1]


def hough(mask: np.ndarray, min_len: int, gap: int = 4) -> np.ndarray:
    """Segments as rows (x1, y1, x2, y2, length, angle_deg in [0, 180))."""
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, 15, minLineLength=min_len, maxLineGap=gap)
    if lines is None:
        return np.zeros((0, 6))
    s = lines.reshape(-1, 4).astype(float)
    length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    ang = np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180
    return np.column_stack([s, length, ang])


def ang_dist(a, b):
    d = np.abs(np.asarray(a) - b) % 180
    return np.minimum(d, 180 - d)


def find_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned analysis box from row/column occupancy of line pixels."""
    h, w = mask.shape
    m = mask > 0
    cols = m.sum(0)
    vx = np.where(cols > 0.35 * h)[0]
    if len(vx) < 2:
        return None
    x0, x1 = int(vx.min()), int(vx.max())
    if x1 - x0 < 0.3 * w:
        return None
    rows = m[:, x0:x1 + 1].sum(1)
    hy = np.where(rows > 0.5 * (x1 - x0))[0]
    if len(hy) < 2:
        return None
    y0, y1 = int(hy.min()), int(hy.max())
    if y1 - y0 < 0.3 * h:
        return None
    return x0, y0, x1, y1


def box_line_mask(shape, box, t=1):
    bm = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = box
    cv2.rectangle(bm, (x0, y0), (x1, y1), 255, 1 + 2 * t)
    return bm


def parse_hip(mask, box):
    """Neck box (two long parallel sides) and neck axis (perpendicular)."""
    inner = mask.copy()
    inner[box_line_mask(mask.shape, box) > 0] = 0
    segs = hough(inner, 20)
    diag = segs[(ang_dist(segs[:, 5], 0) > 8) & (ang_dist(segs[:, 5], 90) > 8)] if len(segs) else segs
    if len(diag) < 3:
        return None, 'hip: too few oblique segments'
    # Two orientation clusters ~90 deg apart; the neck-box long sides are the
    # longest pair of parallel segments with a clear gap between them.
    best = None
    for i in range(len(diag)):
        for j in range(i + 1, len(diag)):
            a, b = diag[i], diag[j]
            if ang_dist(a[5], b[5]) > 6 or min(a[4], b[4]) < 40:
                continue
            ma, mb = (a[:2] + a[2:4]) / 2, (b[:2] + b[2:4]) / 2
            th = np.radians(a[5])
            normal = np.array([-np.sin(th), np.cos(th)])
            gap = abs(np.dot(mb - ma, normal))
            along = abs(np.dot(mb - ma, [np.cos(th), np.sin(th)]))
            if not 8 <= gap <= 60 or along > 0.5 * min(a[4], b[4]):
                continue
            score = min(a[4], b[4])
            if best is None or score > best[0]:
                best = (score, a, b, gap)
    if best is None:
        return None, 'hip: neck box sides not found'
    _, a, b, gap = best
    side_ang = float((a[5] + b[5]) / 2) if ang_dist(a[5], b[5]) == abs(a[5] - b[5]) else float(a[5])
    center = ((a[:2] + a[2:4]) / 2 + (b[:2] + b[2:4]) / 2) / 2
    axis_ang = (side_ang + 90) % 180
    # Longest segment close to the neck-axis direction confirms orientation.
    ax = diag[ang_dist(diag[:, 5], axis_ang) < 10]
    axis_len = float(ax[:, 4].max()) if len(ax) else 0.0
    if len(ax):
        axis_ang = float(ax[np.argmax(ax[:, 4]), 5])
    # Neck axis descends from the femoral head (medial, toward pelvis) to the
    # trochanter. In image coordinates (y down) angle < 90 means it descends to
    # the right, i.e. the head/pelvis is on the left of the image.
    pelvis_side_image = 'left' if axis_ang < 90 else 'right'
    return {
        'neck_center': [round(float(center[0]), 1), round(float(center[1]), 1)],
        'neck_axis_angle_deg': round(axis_ang, 2),
        'neck_box_side_angle_deg': round(side_ang, 2),
        'neck_box_width_px': round(float(gap), 1),
        'neck_box_length_px': round(float(min(a[4], b[4])), 1),
        'neck_axis_segment_px': round(axis_len, 1),
        'pelvis_side_image': pelvis_side_image,
        # Our DICOMs: right hip = shaft left, pelvis right on screen.
        'side_our_convention': 'hip_right' if pelvis_side_image == 'right' else 'hip_left',
    }, None


def parse_spine(mask, box):
    x0, y0, x1, y1 = box
    bw = x1 - x0
    inner = mask.copy()
    inner[:, max(x0 - 1, 0):x0 + 2] = 0
    inner[:, x1 - 1:x1 + 2] = 0
    segs = hough(inner, int(0.4 * bw), gap=6)
    hor = segs[ang_dist(segs[:, 5], 0) < 25] if len(segs) else segs
    if len(hor) < 4:
        return None, 'spine: fewer than 4 boundary lines'
    # Cluster by y at the box centre.
    cx = (x0 + x1) / 2
    ys = []
    for s in hor:
        x1s, y1s, x2s, y2s = s[:4]
        t = (cx - x1s) / (x2s - x1s) if x2s != x1s else 0.5
        ys.append(y1s + t * (y2s - y1s))
    keep = [k for k in range(len(hor)) if y0 - 6 <= ys[k] <= y1 + 6 and ang_dist(hor[k, 5], 0) <= 20]
    hor, ys = hor[keep], [ys[k] for k in keep]
    order = np.argsort(ys)
    clusters: list[list[int]] = []
    for k in order:
        if clusters and ys[k] - ys[clusters[-1][-1]] < 8:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    lines = []
    for c in clusters:
        segc = hor[c]
        wts = segc[:, 4]
        ang = np.degrees(np.arctan2(np.sin(np.radians(2 * segc[:, 5])) @ wts,
                                    np.cos(np.radians(2 * segc[:, 5])) @ wts)) / 2
        lines.append({'y_center': round(float(np.average(np.array(ys)[c], weights=wts)), 1),
                      'angle_deg': round(float(ang), 2),
                      'length_px': round(float(wts.max()), 1)})
    # L1 top and L4 bottom coincide with the analysis box edges; Hough often
    # misses them where the bone contour interrupts the stroke.
    for edge in (y0, y1):
        if all(abs(l['y_center'] - edge) > 8 for l in lines):
            lines.append({'y_center': float(edge), 'angle_deg': 0.0, 'length_px': float(bw), 'from_box_edge': True})
    lines.sort(key=lambda l: l['y_center'])
    if len(lines) != 5:
        return None, f'spine: {len(lines)} boundary lines (expected 5: L1 top .. L4 bottom)'
    heights = np.diff([l['y_center'] for l in lines])
    if heights.min() < 0.5 * np.median(heights):
        return None, 'spine: irregular vertebra spacing'
    return {
        'boundaries': lines,
        'n_boundaries': len(lines),
        'vertebra_heights_px': [round(float(h), 1) for h in heights],
        'top_margin_px': round(lines[0]['y_center'], 1),
        'bottom_margin_px': round(mask.shape[0] - lines[-1]['y_center'], 1),
    }, None


def label_boxes(a):
    """Lunar region labels: dark filled rectangles with white text.
    Spine uses 16x16 squares (L1..L4); forearm uses wider ones (UD, MID, 1/3)."""
    n, _, st, _ = cv2.connectedComponentsWithStats((a <= 3).astype(np.uint8), 8)
    return [tuple(int(v) for v in s[:4]) for s in st[1:]
            if 12 <= s[2] <= 30 and 12 <= s[3] <= 20 and s[4] > 0.6 * s[2] * s[3]]


def classify_view(a, mask, box):
    h, w = a.shape
    if h > 800:
        return 'lateral_vfa'
    labels = label_boxes(a)
    if any(bw >= 18 for _, _, bw, _ in labels):
        return 'forearm'
    if sum(1 for _, _, bw, bh in labels if bw <= 17 and bh <= 17) >= 3 and box is not None:
        return 'spine_ap'
    if box is None:
        return 'unknown'
    segs = hough(mask, 20)
    if not len(segs):
        return 'unknown'
    oblique = segs[(ang_dist(segs[:, 5], 0) > 12) & (ang_dist(segs[:, 5], 90) > 12) & (segs[:, 4] > 40)]
    x0, _, x1, _ = box
    long_h = segs[(ang_dist(segs[:, 5], 0) < 25) & (segs[:, 4] > 0.6 * (x1 - x0))]
    if len(oblique) >= 2:
        return 'hip'
    if len(long_h) >= 4:
        return 'spine_ap'
    return 'unknown'


def clean_image(a, mask, box=None, seed=0):
    """Inpaint overlay strokes and invert to bone-bright convention.

    Telea inpainting leaves smooth streaks that a network could learn as a
    shortcut (our DICOMs have no such streaks), so inpainted pixels receive
    noise matched to the local texture of the untouched neighbourhood."""
    lines = np.zeros_like(mask)
    for s in hough(mask, 10, gap=6):
        cv2.line(lines, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 255, 2)
    # Dashes of the neck axis are short: take small 64-valued components that
    # Hough missed only when they lie next to detected strokes.
    # Short marker strokes (Ward's triangle, neck-box ticks) escape Hough and
    # would leak the neck position, so every 64-valued pixel is inpainted; the
    # few genuine bone pixels with that value are refilled with local texture.
    lines |= mask
    # Lunar bone-edge contour is drawn in pure white over tissue.
    # Lunar's bone-edge contour is a 1 px ridge ~+10..20 brighter than its
    # neighbours (not a fixed value). Scan noise also makes short horizontal
    # ridges, so keep only vertically extended curves and pieces on the box
    # top/bottom rows where the contour runs along the ROI edge.
    ridge = ((a.astype(np.int16) - cv2.medianBlur(a, 5).astype(np.int16)) >= 8).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ridge, 8)
    good = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h = st[k, :4]
        if h >= 10 and max(w, h) >= 15:
            good[k] = True
        elif box is not None and w >= 10 and h <= 3 and min(abs(y - box[1]), abs(y + h - 1 - box[3])) <= 2:
            good[k] = True
    contour = (good[lab] & (ridge > 0)).astype(np.uint8) * 255
    # Ward's triangle / neck markers are drawn near black.
    contour |= ((a <= 10).astype(np.uint8) * 255)
    for x, y, bw, bh in label_boxes(a):
        cv2.rectangle(lines, (x - 1, y - 1), (x + bw, y + bh), 255, -1)
    inpaint = cv2.dilate(lines | contour, np.ones((2, 2), np.uint8))
    out = cv2.inpaint(a, inpaint, 3, cv2.INPAINT_TELEA)
    m = inpaint > 0
    resid = a.astype(np.float32) - cv2.medianBlur(a, 5).astype(np.float32)
    keep = (~cv2.dilate(inpaint, np.ones((3, 3), np.uint8)).astype(bool)).astype(np.float32)
    num = cv2.boxFilter(resid ** 2 * keep, -1, (15, 15), normalize=False)
    den = cv2.boxFilter(keep, -1, (15, 15), normalize=False)
    std = np.sqrt(num / np.maximum(den, 1))
    noise = np.random.default_rng(seed).standard_normal(a.shape).astype(np.float32) * std
    out = out.astype(np.float32)
    out[m] += noise[m]
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Lunar renders bone dark on a bright field; a few exports are already
    # bone-bright. Normalise both to our DICOM convention.
    inverted = np.median(a) > 127
    return (255 - out if inverted else out), int((inpaint > 0).sum())


def draw_audit(clean, rec):
    im = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    if rec.get('box'):
        x0, y0, x1, y1 = rec['box']
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 200, 255), 1)
    hip = rec.get('hip')
    if hip:
        cx, cy = hip['neck_center']
        th = np.radians(hip['neck_axis_angle_deg'])
        d = np.array([np.cos(th), np.sin(th)]) * 70
        cv2.line(im, (int(cx - d[0]), int(cy - d[1])), (int(cx + d[0]), int(cy + d[1])), (0, 0, 255), 1)
        ts = np.radians(hip['neck_box_side_angle_deg'])
        rect = ((cx, cy), (hip['neck_box_length_px'], hip['neck_box_width_px']), float(np.degrees(ts)))
        cv2.drawContours(im, [cv2.boxPoints(rect).astype(np.int32)], 0, (0, 255, 0), 1)
        cv2.putText(im, hip['side_our_convention'][4:], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    sp = rec.get('spine')
    if sp and rec.get('box'):
        x0, _, x1, _ = rec['box']
        cx = (x0 + x1) / 2
        for l in sp['boundaries']:
            t = np.tan(np.radians(l['angle_deg']))
            cv2.line(im, (x0, int(l['y_center'] + (x0 - cx) * t)), (x1, int(l['y_center'] + (x1 - cx) * t)), (0, 0, 255), 1)
    return im


def sheet(tiles, path, cols=8, size=(200, 240)):
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    W = np.zeros((rows * size[1], cols * size[0], 3), np.uint8)
    for i, (name, t) in enumerate(tiles):
        t = cv2.resize(t, size)
        cv2.putText(t, name[:14], (2, size[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        W[(i // cols) * size[1]:(i // cols + 1) * size[1], (i % cols) * size[0]:(i % cols + 1) * size[0]] = t
    cv2.imwrite(str(path), W, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default=str(ROOT / 'data/external/arak_dxa/source.zip'))
    ap.add_argument('--clean-dir', default=str(ROOT / 'data/external/arak_dxa/clean'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/arak_overlays_v1'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--audit-n', type=int, default=48)
    args = ap.parse_args()

    out, clean_dir = Path(args.out), Path(args.clean_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.archive)
    names = sorted(n for n in z.namelist() if n.lower().endswith('.png'))
    if args.limit:
        names = names[:args.limit]

    rng = np.random.default_rng(0)
    audit_idx = set(rng.choice(len(names), min(len(names), 400), replace=False).tolist())
    tiles = collections.defaultdict(list)
    counts = collections.Counter()
    seen = {}
    with open(out / 'annotations.jsonl', 'w') as f:
        for i, n in enumerate(names):
            stem = Path(n).stem
            a = np.array(Image.open(io.BytesIO(z.read(n))).convert('L'))
            mask = (a == LINE_VALUE).astype(np.uint8) * 255
            box = find_box(mask) if a.shape[0] <= 800 else None
            view = classify_view(a, mask, box)
            rec = {'archive_path': n, 'stem': stem, 'patient_key': stem.rsplit('-', 1)[0].lstrip('0'),
                   'height': int(a.shape[0]), 'width': int(a.shape[1]), 'view': view,
                   'box': list(box) if box else None, 'line_pixels': int((mask > 0).sum())}
            h = hash(a.tobytes())
            rec['duplicate_of'] = seen.get(h)
            seen.setdefault(h, stem)
            err = None
            if view == 'hip':
                rec['hip'], err = parse_hip(mask, box)
            elif view == 'spine_ap':
                rec['spine'], err = parse_spine(mask, box)
            elif view == 'unknown':
                err = 'view not recognised'
            rec['parse_ok'] = err is None and view in ('hip', 'spine_ap')
            rec['parse_error'] = err
            if view != 'lateral_vfa':
                clean, npx = clean_image(a, mask, box, seed=i)
                rec['inpainted_px'] = npx
                cv2.imwrite(str(clean_dir / f'{stem}.png'), clean)
                p = clean_dir / f'{stem}.png'
                rec['clean_path'] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                if i in audit_idx:
                    key = view if view == 'forearm' else ('failed' if not rec['parse_ok'] else ('hip' if view == 'hip' else 'spine'))
                    if len(tiles[key]) < args.audit_n:
                        tiles[key].append((stem, draw_audit(clean, rec)))
            counts[(view, 'ok' if rec['parse_ok'] else (err or 'n/a'))] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(names)}', flush=True)

    for k, t in tiles.items():
        sheet(t, out / f'audit_{k}.jpg')
    summary = {'images': len(names),
               'by_view_status': {f'{v} | {s}': c for (v, s), c in sorted(counts.items())},
               'line_value': LINE_VALUE}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
