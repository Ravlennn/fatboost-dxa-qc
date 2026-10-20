"""C — CNN on a crop around the lesser trochanter, placed by the landmark model (docs/PREREG_NEW_MODELS.md).

In September the crop came from hand rules before the landmark network existed (5a7df5b); the
only representation that later moved the rotation ranking (RAD-DINO patch tokens) looked at this
zone. Here the crop is centred on the trochanter apex found by the label-free detector on the
landmark model's neck points (outputs/landmarks_v1/ours/predictions.json); without a detection,
60 px below the neck centre on the medial side. Every crop is turned to one side (medial to the
right, as in the detector), CROP_MM square, then resized. Outputs: hip quality and rotation (the ROI
margins cannot be judged from a crop). Brightness/contrast only: the trochanter outline is the label.

  python scripts/train_hip_ltcrop_cnn.py --tag ltcrop_base --device mps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import qc_cnn_common as C  # noqa: E402

TARGETS = ['quality', 'v_rotation']
CROP_MM = 60.0
MM = 0.6


def crop(pixels: np.ndarray, pts: dict) -> Image.Image:
    from dxaqc.lesser_trochanter import detect
    lt, _ = detect(pixels, pts, mm=MM)
    head, troch, c = (np.array(pts[k], float) for k in ('neck_head_side', 'neck_troch_side', 'neck_center'))
    medial = 1.0 if head[0] > troch[0] else -1.0
    centre = np.array(lt['points']['lt_apex']) if lt is not None else c + np.array([20.0 * medial, 60.0])
    half = int(round(CROP_MM / MM / 2))
    h, w = pixels.shape
    pad = np.zeros((h + 4 * half, w + 4 * half), np.uint8)
    pad[2 * half:2 * half + h, 2 * half:2 * half + w] = pixels
    x, y = int(round(centre[0])) + 2 * half, int(round(centre[1])) + 2 * half
    out = pad[y - half:y + half, x - half:x + half]
    if medial < 0:   # canonical: medial side to the right
        out = out[:, ::-1]
    return Image.fromarray(np.ascontiguousarray(out))


def main():
    args = C.parser(__doc__.splitlines()[0], res=224).parse_args()
    out, dev = C.setup(args, C.ROOT / 'outputs/ltcrop_cnn')
    from dxaqc.dicom_io import read_dicom
    rows = C.labelled()
    rows = rows[rows.region != 'spine'].reset_index(drop=True)
    pts = {r['image_uid']: r['points'] for r in json.load(open(C.ROOT / 'outputs/landmarks_v1/ours/predictions.json'))}
    images = [crop(read_dicom(C.DATA / r.path).pixels, pts[r.image_uid]) for r in rows.itertuples()]
    print('hips', len(rows), 'crop', images[0].size, 'device', dev, flush=True)
    train_tf, test_tf = C.photometric(args.res), C.plain(args.res)

    def build():
        from dxaqc.hip_backbones import build_hip_model
        return build_hip_model('convnext_tiny', outputs=len(TARGETS), pretrained=True)

    C.run(args, out, dev, rows, rows[TARGETS].values, TARGETS, images, train_tf, test_tf, build)


if __name__ == '__main__':
    main()
