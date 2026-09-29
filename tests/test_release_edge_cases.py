"""Regressions for per-file isolation and bounded square preprocessing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pydicom.errors import InvalidDicomError
from dxaqc.api import analyze
from dxaqc.dicom_io import read_dicom
from dxaqc.predictor import Prediction
from dxaqc.release_features import image_tensor
from dxaqc.release_predictor import hip_tensor
from test_release_cli import synthetic_dicom


class EdgeCaseTests(unittest.TestCase):
    def test_nan_prediction_is_failure_without_discarding_other_file(self):
        class Predictor:
            name = 'test'
            calls = 0
            def predict_image(self, pixels):
                self.calls += 1
                return 'hip_right', Prediction(0, scores={'quality': float('nan') if self.calls == 1 else .2})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_dicom(root / 'a.dcm')
            synthetic_dicom(root / 'b.dcm', pixels=np.arange(36*40, dtype=np.uint16).reshape(36,40) % 253)
            rows, summary = analyze(root, predictor=Predictor())
            self.assertEqual([row['processing_status'] for row in rows], ['Failure', 'Success'])
            self.assertEqual((summary['success'], summary['failure']), (1,1))

    def test_extreme_aspect_ratio_rejected_before_pixel_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = synthetic_dicom(Path(directory) / 'wide.dcm', pixels=np.arange(16*5000, dtype=np.uint16).reshape(16,5000))
            with self.assertRaisesRegex(InvalidDicomError, 'padding limit'):
                read_dicom(path)

    def test_tensor_helpers_reject_oversized_square_before_allocation(self):
        pixels = np.empty((16,5000), dtype=np.uint8)
        with patch('numpy.zeros', side_effect=AssertionError('Must not allocate padded canvas')):
            with self.assertRaises(ValueError):
                image_tensor(pixels)
            with self.assertRaises(ValueError):
                hip_tensor(pixels, 'hip_right')

if __name__ == '__main__': unittest.main()
