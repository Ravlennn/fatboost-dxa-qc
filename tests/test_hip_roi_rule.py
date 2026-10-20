"""Hip ROI margins: a box cut off by the image edge is a TZ violation and forces a defect."""
import os
import unittest
from pathlib import Path

from dxaqc.predictor import HIP_ROI
from dxaqc.release_predictor import roi_edge_score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get('DXAQC_MODEL_DIR', ROOT / 'models/release'))
HIP_DICOM = Path(os.environ.get('DXAQC_HIP_DICOM', ROOT / 'data/Для теста/CR000000_ППОБ.dcm'))


def margins(top=30., bottom=30., left=20., right=20.):
    return {'roi_top_mm': top, 'roi_bottom_mm': bottom, 'roi_left_mm': left, 'roi_right_mm': right}


class RoiEdgeScoreTests(unittest.TestCase):
    def test_threshold_is_the_image_edge(self):
        self.assertEqual(roi_edge_score(margins(bottom=0.))[1], 0.5)
        self.assertGreater(roi_edge_score(margins(bottom=-1.))[1], 0.5)
        self.assertLess(roi_edge_score(margins(bottom=0.2))[1], 0.5)

    def test_smallest_margin_decides(self):
        self.assertEqual(roi_edge_score(margins(left=-3.))[0], -3.)
        self.assertEqual(roi_edge_score(margins(top=500., right=-500.))[1], 1.0)


@unittest.skipUnless((BUNDLE / 'manifest.json').is_file() and HIP_DICOM.is_file(), 'needs release weights and a hip DICOM')
class RoiRuleOnReleaseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dxaqc.dicom_io import read_dicom
        from dxaqc.release_predictor import ReleasePredictor
        cls.predictor = ReleasePredictor(BUNDLE)
        img = read_dicom(HIP_DICOM)
        cls.pixels, cls.mm = img.pixels, img.meta.get('spacing_mm')
        cls.region, cls.base = cls.predictor.predict_image(cls.pixels, mm_per_px=cls.mm)

    def cut_off(self, rule=True, beyond_px=20):
        """Box leaves the image at the bottom; the quality head and the positioning head are made to say
        'normal' (threshold above any score), so only the ROI rule can produce the defect."""
        points = {k: list(v) for k, v in self.base.details['landmarks']['points'].items()}
        points['roi_bottom'][1] = self.pixels.shape[0] + beyond_px
        saved = self.predictor.hip_heads
        self.predictor.hip_heads = {k: {**h, 'threshold': 1.01} for k, h in saved.items()}
        self.predictor.hip_roi_rule = rule
        try:
            return self.predictor.predict(self.pixels, self.region, mm_per_px=self.mm, points=points)
        finally:
            self.predictor.hip_heads, self.predictor.hip_roi_rule = saved, True

    def test_box_inside_image_is_not_a_violation(self):
        self.assertTrue(self.region.startswith('hip'))
        self.assertGreater(self.base.details['landmarks']['roi_min_margin_mm'], 0)
        self.assertLess(self.base.scores[HIP_ROI], 0.5)
        self.assertNotIn(HIP_ROI, self.base.violations)

    def test_box_outside_image_forces_defect_with_roi_code(self):
        pred = self.cut_off()
        self.assertLess(pred.details['landmarks']['roi_min_margin_mm'], 0)
        self.assertGreater(pred.scores[HIP_ROI], 0.5)
        self.assertEqual(pred.quality_class, 1)
        self.assertIn(HIP_ROI, pred.violations)

    def test_box_one_pixel_beyond_the_edge_is_already_a_violation(self):
        pred = self.cut_off(beyond_px=1)
        self.assertLess(pred.details['landmarks']['roi_min_margin_mm'], 0)
        self.assertIn(HIP_ROI, pred.violations)

    def test_rule_can_be_switched_off(self):
        pred = self.cut_off(rule=False)
        self.assertNotIn('roi_min_margin_mm', pred.details['landmarks'])
        self.assertEqual(pred.quality_class, 0)  # the same image without the rule: heads say normal


if __name__ == '__main__':
    unittest.main()
