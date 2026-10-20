"""A — fine-tuned CNN for the lumbar spine (docs/PREREG_NEW_MODELS.md).

The spine release has no fine-tuned network: a frozen DenseNet121 with linear heads, landmark
geometry and the metal detector. On the hip the step from a frozen encoder to a fine-tuned
ConvNeXt was the largest gain of the project; this arm makes the same step for the spine.

ConvNeXt-Tiny 384 px, outputs: quality, scan range (v_layout), axis tilt (v_axis), foreign objects
(v_artifact). Only brightness/contrast are jittered: a rotation would change the tilt label and a
shift or crop the scan-range label.

  python scripts/train_spine_cnn.py --tag spine_base --device mps
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import qc_cnn_common as C  # noqa: E402

TARGETS = ['quality', 'v_layout', 'v_axis', 'v_artifact']


def main():
    args = C.parser(__doc__.splitlines()[0]).parse_args()
    out, dev = C.setup(args, C.ROOT / 'outputs/spine_cnn')
    rows = C.labelled()
    rows = rows[rows.region == 'spine'].reset_index(drop=True)
    images = [C.square(pydicom.dcmread(str(C.DATA / r.path)).pixel_array) for r in rows.itertuples()]
    print('spines', len(rows), 'device', dev, 'res', args.res, flush=True)
    train_tf, test_tf = C.photometric(args.res), C.plain(args.res)

    def build():
        from dxaqc.hip_backbones import build_hip_model
        return build_hip_model('convnext_tiny', outputs=len(TARGETS), pretrained=True)

    C.run(args, out, dev, rows, rows[TARGETS].values, TARGETS, images, train_tf, test_tf, build)


if __name__ == '__main__':
    main()
