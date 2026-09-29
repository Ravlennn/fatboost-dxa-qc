import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dxaqc.dicom_io import DicomImage  # noqa: E402
from dxaqc.landmarks import (FLIP_PERM, POINTS, STRIDE, apply_head, decode, draw_overlay,  # noqa: E402
                             head_features, letterbox, measurements)
from dxaqc.overlay import write_overlay  # noqa: E402
from dxaqc.predictor import Prediction  # noqa: E402

SPINE = {'L1_top': [100, 40], 'L1L2': [100, 100], 'L2L3': [100, 160], 'L3L4': [100, 220], 'L4_bottom': [100, 280]}
HIP = {'roi_top': [140, 60], 'roi_bottom': [140, 230], 'roi_left': [40, 145], 'roi_right': [240, 145],
       'neck_center': [150, 120], 'neck_head_side': [175, 100], 'neck_troch_side': [125, 140]}


class Measurements(unittest.TestCase):
    def test_vertical_spine_has_zero_tilt_and_margins_in_vertebrae(self):
        m = measurements(SPINE, 'lumbar_spine', (300, 200))
        self.assertAlmostEqual(m['spine_tilt_deg'], 0.0)
        self.assertAlmostEqual(m['vertebra_height_px'], 60.0)
        self.assertAlmostEqual(m['top_margin_vert'], 40 / 60)
        self.assertAlmostEqual(m['bottom_margin_vert'], 20 / 60)

    def test_tilt_sign_and_magnitude(self):
        pts = {k: [100 + (v[1] - 40) * np.tan(np.radians(6)), v[1]] for k, v in SPINE.items()}
        self.assertAlmostEqual(measurements(pts, 'spine', (300, 200))['spine_tilt_deg'], 6.0, places=4)

    def test_hip_margins_mm_and_lateral_side(self):
        m = measurements(HIP, 'hip_right', (260, 280))
        self.assertAlmostEqual(m['roi_top_mm'], 60 * 0.6)
        self.assertAlmostEqual(m['roi_right_mm'], 40 * 0.6)
        right, left = head_features(m, 'hip_right'), head_features(m, 'hip_left')
        self.assertEqual(right['roi_lateral'], m['roi_left_mm'])
        self.assertEqual(left['roi_lateral'], m['roi_right_mm'])
        self.assertEqual(right['roi_min'], min(m['roi_top_mm'], m['roi_bottom_mm'], m['roi_left_mm'], m['roi_right_mm']))

    def test_apply_head_is_standardized_logistic(self):
        head = {'features': ['abs_tilt'], 'mean': [2.0], 'scale': [1.0], 'coef': [1.0], 'intercept': 0.0}
        self.assertAlmostEqual(apply_head(head, {'abs_tilt': 2.0}), 0.5)
        self.assertGreater(apply_head(head, {'abs_tilt': 8.0}), 0.99)


class Geometry(unittest.TestCase):
    def test_flip_perm_swaps_only_roi_left_right(self):
        swapped = [(POINTS[i], POINTS[j]) for i, j in enumerate(FLIP_PERM) if i != j]
        self.assertEqual(sorted(swapped), [('roi_left', 'roi_right'), ('roi_right', 'roi_left')])

    def test_decode_recovers_gaussian_peak(self):
        h = torch.zeros(1, 1, 64, 64)
        ys, xs = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing='ij')
        h[0, 0] = torch.exp(-((xs - 20.3) ** 2 + (ys - 31.6) ** 2) / 4.5)
        c, _ = decode(h)
        self.assertAlmostEqual(c[0, 0, 0].item(), 20.3 * STRIDE + (STRIDE - 1) / 2, delta=0.5)
        self.assertAlmostEqual(c[0, 0, 1].item(), 31.6 * STRIDE + (STRIDE - 1) / 2, delta=0.5)

    def test_letterbox_keeps_aspect(self):
        canvas, s = letterbox(np.full((300, 150), 7, np.uint8))
        self.assertEqual(canvas.shape, (256, 256))
        self.assertAlmostEqual(s, 256 / 300)
        self.assertEqual(canvas[:, 200:].max(), 0)


class Overlay(unittest.TestCase):
    def test_overlay_png_and_secondary_capture(self):
        pixels = np.random.default_rng(0).integers(0, 255, (300, 200), dtype=np.uint8)
        m = measurements(SPINE, 'lumbar_spine', pixels.shape)
        pred = Prediction(1, ['spine_axis_tilt'], {}, {'landmarks': {'points': SPINE, 'measurements': m}})
        rgb = draw_overlay(pixels, 'lumbar_spine', SPINE, m, pred.violations, 1)
        self.assertEqual(rgb.ndim, 3)
        img = DicomImage(Path('x.dcm'), '1.2.3', '1.2.3.4', '1.2.3.4.5', pixels)
        with tempfile.TemporaryDirectory() as d:
            a = write_overlay(Path(d), 'Иссл/CR000000_ПОП.dcm', img, 'lumbar_spine', pred)
            b = write_overlay(Path(d), 'Иссл/CR000000_ПОП.dcm', img, 'lumbar_spine', pred)
            self.assertEqual(len({p.name for p in a + b}), 4, 'second write must not overwrite the first')
            ds = pydicom.dcmread(a[1])
            self.assertEqual(ds.SOPClassUID, '1.2.840.10008.5.1.4.1.1.7')
            self.assertEqual(ds.StudyInstanceUID, '1.2.3')
            self.assertIn('spine_axis_tilt', ds.ImageComments)

    def test_no_landmarks_no_files(self):
        img = DicomImage(Path('x.dcm'), '1', '2', '3', np.zeros((10, 10), np.uint8))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(write_overlay(Path(d), 'a.dcm', img, 'hip_left', Prediction(0)), [])



class BundleHeads(unittest.TestCase):
    def test_exported_heads_get_all_their_features(self):
        import json
        heads_path = Path(__file__).resolve().parents[1] / 'models/release/landmarks/heads.json'
        if not heads_path.is_file():
            self.skipTest('landmark bundle not present')
        pixels = np.random.default_rng(1).integers(0, 255, (300, 200), dtype=np.uint8)
        feats = head_features(measurements(SPINE, 'lumbar_spine', pixels.shape, pixels), 'lumbar_spine')
        for code, head in json.loads(heads_path.read_text())['heads'].items():
            missing = set(head['features']) - feats.keys()
            self.assertFalse(missing, f'{code}: features not produced at inference: {missing}')
            self.assertTrue(0 <= apply_head(head, feats) <= 1)


if __name__ == '__main__':
    unittest.main()
