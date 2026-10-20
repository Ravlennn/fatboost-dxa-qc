"""DICOM SR: one Basic Text SR per analysed image, in the source study, readable back with Cyrillic."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_release_cli import synthetic_dicom  # noqa: E402
from dxaqc.api import analyze  # noqa: E402
from dxaqc.predictor import Prediction  # noqa: E402
from dxaqc.sr import BASIC_TEXT_SR  # noqa: E402


class BadHipPredictor:
    name, model_id = 'sr_test', 'synthetic-sr'

    def predict_image(self, pixels):
        return 'hip_left', Prediction(1, ['hip_positioning'], {'quality': 0.9, 'quality_threshold': 0.5},
                                      {'quality_threshold': 0.5, 'warnings': ['проверка'],
                                       'landmarks': {'measurements': {'roi_top_mm': 24.4, 'neck_axis_deg': 12.0},
                                                     'lesser_trochanter': {'visible': True, 'lt_prominence_mm': 10.2}}})


class DicomSR(unittest.TestCase):
    def test_sr_per_image_references_source_and_carries_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_dicom(tmp / 'in' / 'a.dcm', image_id=1)
            synthetic_dicom(tmp / 'in' / 'b.dcm', image_id=2)
            rows, stats = analyze(tmp / 'in', predictor=BadHipPredictor(), sr_dir=tmp / 'sr')
            self.assertEqual((stats['sr'], stats['sr_failures']), (2, 0))
            files = sorted((tmp / 'sr').glob('*_SR.dcm'))
            self.assertEqual(len(files), 2)
            ds = pydicom.dcmread(files[0])
            self.assertEqual((ds.SOPClassUID, ds.Modality, ds.VerificationFlag), (BASIC_TEXT_SR, 'SR', 'UNVERIFIED'))
            self.assertEqual(ds.StudyInstanceUID, '2.25.100')
            ref = ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].ReferencedSOPSequence[0]
            self.assertEqual(ref.ReferencedSOPInstanceUID, '2.25.1001')
            content = {i.ConceptNameCodeSequence[0].CodeValue: i for i in ds.ContentSequence}
            self.assertEqual(content['REGION'].TextValue, 'Проксимальный отдел бедра')
            self.assertEqual(content['VERDICT'].TextValue, 'Есть нарушение качества')
            self.assertIn('укладка', content['VIOLATIONS'].TextValue.lower())
            self.assertEqual(content['LT_PROMINENCE'].MeasuredValueSequence[0].NumericValue, '10.2')
            self.assertIn('QUALITY_PROB', content)
            # both reports of one study land in one SR series; re-running gives the same UIDs
            self.assertEqual(ds.SeriesInstanceUID, pydicom.dcmread(files[1]).SeriesInstanceUID)
            analyze(tmp / 'in', predictor=BadHipPredictor(), sr_dir=tmp / 'sr2')
            self.assertEqual(ds.SOPInstanceUID, pydicom.dcmread(sorted((tmp / 'sr2').glob('*.dcm'))[0]).SOPInstanceUID)


if __name__ == '__main__':
    unittest.main()
