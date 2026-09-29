import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dxaqc.dicom_io import estimate_pixel_spacing  # noqa: E402
from dxaqc.landmarks import apply_head, measurements  # noqa: E402

SPINE = {'L1_top': [100, 40], 'L1L2': [100, 100], 'L2L3': [100, 160], 'L3L4': [100, 220], 'L4_bottom': [100, 280]}
HIP = {'roi_top': [140, 60], 'roi_bottom': [140, 230], 'roi_left': [40, 145], 'roi_right': [240, 145],
       'neck_center': [150, 120], 'neck_head_side': [175, 100], 'neck_troch_side': [125, 140]}


class PixelSpacing(unittest.TestCase):
    def test_pixel_spacing_tag_wins(self):
        mm, src = estimate_pixel_spacing({'PixelSpacing': [0.45, 0.45], 'Rows': 300, 'Columns': 300})
        self.assertAlmostEqual(mm, 0.45)
        self.assertEqual(src, 'PixelSpacing')

    def test_exposed_area_other_vendor_is_accepted(self):
        # Not GE's ~0.6 mm/px: both axes agree, so the value must be used, not the default.
        mm, src = estimate_pixel_spacing({'ExposedArea': [120, 150], 'Rows': 375, 'Columns': 300})
        self.assertAlmostEqual(mm, 0.4)
        self.assertEqual(src, 'ExposedArea')

    def test_inconsistent_exposed_area_falls_back(self):
        mm, src = estimate_pixel_spacing({'ExposedArea': [180, 90], 'Rows': 300, 'Columns': 300})
        self.assertEqual(src, 'default')
        self.assertAlmostEqual(mm, 0.6)

    def test_measurements_scale_with_spacing(self):
        a = measurements(HIP, 'hip_right', (260, 280), mm_per_px=0.6)
        b = measurements(HIP, 'hip_right', (260, 280), mm_per_px=0.3)
        self.assertAlmostEqual(a['roi_top_mm'], 2 * b['roi_top_mm'])
        self.assertAlmostEqual(b['mm_per_px'], 0.3)

    def test_spine_tilt_is_spacing_invariant(self):
        a = measurements(SPINE, 'spine', (300, 200), mm_per_px=0.6)
        b = measurements(SPINE, 'spine', (300, 200), mm_per_px=0.35)
        self.assertAlmostEqual(a['spine_tilt_deg'], b['spine_tilt_deg'])


class ViewGate(unittest.TestCase):
    def test_gate_head_is_in_the_bundle_and_applies_to_a_raw_vector(self):
        manifest_path = ROOT / 'models/release/manifest.json'
        if not manifest_path.is_file():
            self.skipTest('release bundle not present')
        manifest = json.loads(manifest_path.read_text())
        if 'view_gate' not in manifest:
            self.skipTest('view gate not exported')
        head = json.loads((ROOT / 'models/release' / manifest['view_gate']).read_text())
        self.assertEqual(len(head['mean']), len(head['coef']))
        self.assertIn(manifest['view_gate'], manifest['files'])
        feats = {f'f{i}': float(v) for i, v in enumerate(np.zeros(len(head['mean'])))}
        self.assertTrue(0 <= apply_head(head, feats) <= 1)
        self.assertTrue(0 < head['threshold'] < 1)

    def test_predictor_exposes_the_switch(self):
        from dxaqc.release_predictor import ReleasePredictor
        self.assertTrue(hasattr(ReleasePredictor, 'predict_image'))


if __name__ == '__main__':
    unittest.main()
