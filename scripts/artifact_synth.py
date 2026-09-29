"""Procedural DXA artifacts (metal / dense foreign objects) with pixel masks.

In our convention (bone bright) dense metal renders as near-saturated structures
with sharp edges; bra underwires are usually faint double lines over soft tissue. Types follow the TZ example (bra
underwire) and what we see in labelled spine artifacts: underwires, hooks and
clasps, buttons/rivets, zippers, rings, spinal hardware (screws + rods).

Backgrounds must come from external data only (cleaned Arak), so evaluation
on our labelled spine images stays zero-shot.

  python scripts/artifact_synth.py --preview outputs/artifact_synth/preview.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

TYPES = ('underwire', 'hooks', 'buttons', 'zipper', 'ring', 'hardware')


def _stroke(mask, pts, width):
    cv2.polylines(mask, [np.round(pts).astype(np.int32)], False, 1.0, int(max(1, round(width))), cv2.LINE_AA)


def underwire(h, w, rng):
    """Pair of U-shaped wires in the upper part, often cut by the frame edge.
    Real underwires in our scans are faint double lines (wire in a casing)."""
    m = np.zeros((h, w), np.float32)
    cy = rng.uniform(-0.1, 0.3) * h
    span = rng.uniform(0.3, 0.6) * w
    depth = rng.uniform(0.06, 0.2) * h
    width = rng.uniform(1.0, 2.5)
    gap = rng.uniform(0.0, 0.2) * w
    double = rng.random() < 0.6
    for side in (-1, 1):
        if rng.random() < 0.2:  # sometimes only one wire in the field of view
            continue
        cx = w / 2 + side * (gap / 2 + span / 2)
        t = np.linspace(-1, 1, 80)
        x = cx + t * span / 2
        y = cy + depth * (1 - t ** 2) * rng.uniform(0.7, 1.0) + rng.normal(0, 0.5, t.size).cumsum() * 0.05
        _stroke(m, np.stack([x, y], 1), width)
        if double:
            _stroke(m, np.stack([x, y + rng.uniform(3, 6)], 1), width)
    return m


def hooks(h, w, rng):
    m = np.zeros((h, w), np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.05, 0.9) * h
    for _ in range(rng.integers(2, 5)):
        dx, dy = rng.normal(0, 6, 2)
        a, b = rng.uniform(3, 8), rng.uniform(2, 5)
        cv2.rectangle(m, (int(cx + dx - a), int(cy + dy - b)), (int(cx + dx + a), int(cy + dy + b)), 1.0, -1)
    return m


def buttons(h, w, rng):
    m = np.zeros((h, w), np.float32)
    x0, y0 = rng.uniform(0.15, 0.85) * w, rng.uniform(0.1, 0.9) * h
    for k in range(rng.integers(1, 4)):
        r = rng.uniform(3, 9)
        c = (int(x0 + rng.normal(0, 3)), int(y0 + k * rng.uniform(15, 35)))
        cv2.circle(m, c, int(r), 1.0, -1 if rng.random() < 0.7 else 2, cv2.LINE_AA)
    return m


def zipper(h, w, rng):
    m = np.zeros((h, w), np.float32)
    x, y0 = rng.uniform(0.2, 0.8) * w, rng.uniform(0.3, 0.8) * h
    for k in range(rng.integers(8, 25)):
        y = y0 + k * 4
        if y >= h:
            break
        cv2.rectangle(m, (int(x - 3), int(y)), (int(x + 3), int(y + 2)), 1.0, -1)
    cv2.rectangle(m, (int(x - 5), int(y0 - 8)), (int(x + 5), int(y0)), 1.0, -1)
    return m


def ring(h, w, rng):
    m = np.zeros((h, w), np.float32)
    c = (int(rng.uniform(0.3, 0.7) * w), int(rng.uniform(0.3, 0.8) * h))
    cv2.circle(m, c, int(rng.uniform(4, 10)), 1.0, int(rng.uniform(1, 3)), cv2.LINE_AA)
    return m


def hardware(h, w, rng, spine_x=None):
    """Pedicle screws in pairs along the column plus optional rods."""
    m = np.zeros((h, w), np.float32)
    cx = spine_x if spine_x is not None else w / 2 + rng.normal(0, 0.05 * w)
    off = rng.uniform(0.07, 0.14) * w
    y = rng.uniform(0.2, 0.6) * h
    levels = rng.integers(2, 5)
    step = rng.uniform(0.12, 0.2) * h
    heads = []
    for k in range(levels):
        yy = y + k * step
        if yy >= h - 5:
            break
        for side in (-1, 1):
            x = cx + side * off
            ang = np.radians(rng.uniform(-20, 20) + (90 - side * 20))
            L, W = rng.uniform(15, 30), rng.uniform(4, 7)
            p = np.array([[x, yy], [x + L * np.cos(ang) * -side * 0.5, yy + L * np.sin(ang) * 0.2]])
            _stroke(m, p, W)
            cv2.circle(m, (int(x), int(yy)), int(W), 1.0, -1)
            heads.append((x, yy, side))
    if rng.random() < 0.8 and len(heads) >= 4:
        for side in (-1, 1):
            pts = np.array([(x, yy) for x, yy, s in heads if s == side])
            _stroke(m, pts, rng.uniform(3, 5))
    return m


GEN = {'underwire': underwire, 'hooks': hooks, 'buttons': buttons, 'zipper': zipper, 'ring': ring, 'hardware': hardware}
WEIGHTS = np.array([0.35, 0.15, 0.15, 0.1, 0.05, 0.2])


def apply_artifacts(img: np.ndarray, rng, n_max=2, types=TYPES):
    """Return (image uint8, mask uint8 {0,1}, list of types).

    Each artifact is rendered in its own mode: dense metal = max(bg, value >= densest bone);
    underwires are usually faint (additive contrast over tissue), as in our labelled scans."""
    h, w = img.shape
    out = img.astype(np.float32)
    total = np.zeros((h, w), np.float32)
    used = []
    wts = WEIGHTS[[TYPES.index(t) for t in types]]
    for _ in range(rng.integers(1, n_max + 1)):
        t = str(rng.choice(types, p=wts / wts.sum()))
        m = GEN[t](h, w, rng)
        soft = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.4, 1.0))
        if t == 'underwire' and rng.random() < 0.75:
            out = out + soft * (rng.uniform(25, 110) + rng.normal(0, 5, img.shape))
        else:
            val = rng.uniform(max(np.percentile(img, 99.5), 200), 255)
            out = out * (1 - soft) + np.maximum(out, val + rng.normal(0, 4, img.shape)) * soft
        total = np.maximum(total, m)
        used.append(t)
    return np.clip(out, 0, 255).astype(np.uint8), (total > 0.3).astype(np.uint8), used


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='outputs/ssl_v1/manifest.csv')
    ap.add_argument('--preview', default='outputs/artifact_synth/preview.jpg')
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.manifest)
    bg = df[df.source.eq('arak') & df.view.eq('spine_ap')].path.tolist()  # Pakistan PNGs carry drawn overlays
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(args.n):
        img = cv2.imread(str(root / bg[rng.integers(len(bg))]), 0)
        out, m, used = apply_artifacts(img, rng)
        v = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        side = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(side, cnts, -1, (0, 0, 255), 1)
        t = cv2.resize(np.hstack([v, side]), (320, 200))
        cv2.putText(t, ','.join(used), (3, 12), 0, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    Path(root / args.preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / args.preview), np.vstack(rows))
    print('preview ->', args.preview)


if __name__ == '__main__':
    main()
