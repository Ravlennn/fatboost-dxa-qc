"""prob_<code>: one probability per violation type, threshold at 0.5, consistent with violation_type."""
import csv
import tempfile
import unittest
from pathlib import Path

from dxaqc.api import analyze
from dxaqc.predictor import Prediction
from dxaqc.report import TYPE_PROB_COLUMNS, check_report, make_row, type_probs, write_report
from test_release_cli import synthetic_dicom


class HipPredictor:
    """Hip defect by positioning; the ROI cause stays below its own threshold."""
    name = 'type_probs_test'

    def predict_image(self, pixels):
        return 'hip_left', Prediction(1, ['hip_positioning'],
                                      {'quality': 0.8, 'hip_positioning': 0.6, 'hip_roi_margins': 0.1},
                                      {'quality_threshold': 0.4,
                                       'reason_thresholds': {'hip_positioning': 0.3, 'hip_roi_margins': 0.2}})


class TypeProbTests(unittest.TestCase):
    def test_threshold_maps_to_half_and_codes_agree(self):
        p = type_probs({'hip_positioning': 0.3, 'hip_roi_margins': 0.1},
                       {'hip_positioning': 0.3, 'hip_roi_margins': 0.2}, ['hip_positioning'], 'hip_right')
        self.assertEqual(p, {'prob_hip_positioning': 0.5, 'prob_hip_roi_margins': 0.25})

    def test_reported_code_is_at_least_half_even_below_its_threshold(self):
        # 'closest cause' rule: a defect without a cause above threshold still reports one code
        p = type_probs({'hip_positioning': 0.2}, {'hip_positioning': 0.3}, ['hip_positioning'], 'hip_left')
        self.assertGreaterEqual(p['prob_hip_positioning'], 0.5)

    def test_other_region_codes_are_absent(self):
        p = type_probs({'hip_positioning': 0.9, 'spine_axis_tilt': 0.9},
                       {'hip_positioning': 0.5, 'spine_axis_tilt': 0.5}, [], 'lumbar_spine')
        self.assertEqual(set(p), {'prob_spine_axis_tilt'})

    def test_check_report_rejects_prob_that_contradicts_codes(self):
        row = make_row(path_to_study='a.dcm', study_uid='1', image_uid='2', anatomical_region='hip_left',
                       quality_class=1, violation_type='hip_positioning',
                       scores={'quality_prob': 0.7, 'prob_hip_positioning': 0.4})
        self.assertTrue(any('prob_hip_positioning' in e for e in check_report([row])))

    def test_columns_on_request_and_default_report_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_dicom(root / 'in' / 'a.dcm')
            rows, _ = analyze(root / 'in', predictor=HipPredictor())
            self.assertEqual(check_report(rows), [])
            out = write_report(rows, root / 'r.csv', strict_columns=True, type_probs=True)
            with out.open(encoding='utf-8') as f:
                row = next(csv.DictReader(f))
            self.assertEqual(list(row)[-len(TYPE_PROB_COLUMNS):], TYPE_PROB_COLUMNS)
            self.assertGreaterEqual(float(row['prob_hip_positioning']), 0.5)
            self.assertLess(float(row['prob_hip_roi_margins']), 0.5)
            self.assertEqual(row['prob_spine_axis_tilt'], '')
            out = write_report(rows, root / 'r2.csv', strict_columns=True)
            self.assertNotIn('prob_hip_positioning', out.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
