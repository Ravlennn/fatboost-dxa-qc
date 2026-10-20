"""B — one network for the spine and the hip, separate outputs per region (docs/PREREG_NEW_MODELS.md).

The shared encoder learns from all 249 labelled frames instead of 99 or 150. Seven outputs:
spine quality, scan range, axis tilt, foreign objects; hip quality, rotation, ROI margins. The loss
of a frame only covers the outputs of its own region (the others are unknown, NaN, and masked).
Each region keeps its own augmentation: the current hip recipe for hips, brightness/contrast only
for spines. Left hips are mirrored as in train_hip_cnn.py.

oof.csv carries the columns the two evaluators read: p_quality (from the frame's own region),
p_v_layout, p_v_axis, p_v_artifact for spines and p_v_rotation, p_v_roi for hips.

  python scripts/train_joint_cnn.py --tag joint_base --device mps
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import qc_cnn_common as C  # noqa: E402

SPINE = ['quality', 'v_layout', 'v_axis', 'v_artifact']
HIP = ['quality', 'v_rotation', 'v_roi']
TARGETS = ['q_spine', 'v_layout', 'v_axis', 'v_artifact', 'q_hip', 'v_rotation', 'v_roi']


def main():
    args = C.parser(__doc__.splitlines()[0]).parse_args()
    out, dev = C.setup(args, C.ROOT / 'outputs/joint_cnn')
    rows = C.labelled()
    is_spine = (rows.region == 'spine').values
    Y = np.full((len(rows), len(TARGETS)), np.nan, np.float32)
    Y[is_spine, 0:4] = rows.loc[is_spine, SPINE].values
    Y[~is_spine, 4:7] = rows.loc[~is_spine, HIP].values
    images = []
    for r in rows.itertuples():
        a = pydicom.dcmread(str(C.DATA / r.path)).pixel_array
        if r.region == 'hip_left':
            a = np.ascontiguousarray(a[:, ::-1])
        images.append(C.square(a))
    print('frames', len(rows), f'(spine {is_spine.sum()}, hip {(~is_spine).sum()})', 'device', dev, flush=True)
    spine_tf, hip_tf, test_tf = C.photometric(args.res), C.hip_full(args.res), C.plain(args.res)

    def build():
        from dxaqc.hip_backbones import build_hip_model
        return build_hip_model('convnext_tiny', outputs=len(TARGETS), pretrained=True)

    def finalize(res: pd.DataFrame) -> pd.DataFrame:
        sp = res.region.eq('spine')
        res['p_quality'] = np.where(sp, res.p_q_spine, res.p_q_hip)
        for c in ('p_v_layout', 'p_v_axis', 'p_v_artifact'):
            res.loc[~sp, c] = np.nan
        for c in ('p_v_rotation', 'p_v_roi'):
            res.loc[sp, c] = np.nan
        return res

    C.run(args, out, dev, rows, Y, TARGETS, images, lambda j: spine_tf if is_spine[j] else hip_tf, test_tf, build, finalize)


if __name__ == '__main__':
    main()
