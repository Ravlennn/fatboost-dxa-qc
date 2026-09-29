"""Acceptance tests for the offline CLI; all DICOMs are synthetic."""
from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import logging
import stat
import tempfile
import threading
import unittest
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
from openpyxl import load_workbook
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from dxaqc.api import analyze
from dxaqc.bundle import verify_bundle
from dxaqc.cli import main
from dxaqc.dicom_io import read_dicom
from dxaqc.inputs import input_directory
from dxaqc.predictor import Prediction
from dxaqc.report import COLUMNS, SCORE_COLUMNS, check_report, make_row, write_report
from dxaqc.server import handler_for


def synthetic_dicom(path: Path, *, pixels=None, mono="MONOCHROME2", uids=True, image_id=1):
    """No patient metadata; deterministic UIDs and nonconstant 32×40 pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels is None:
        pixels = (np.arange(32 * 40, dtype=np.uint16).reshape(32, 40) % 253).astype(np.uint8)
    pixels = np.ascontiguousarray(pixels)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = f"2.25.{1000 + image_id}"
    meta.ImplementationClassUID = "2.25.999"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    if uids:
        ds.StudyInstanceUID = "2.25.100"
        ds.SOPInstanceUID = f"2.25.{1000 + image_id}"
    ds.SeriesInstanceUID = "2.25.200"
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = mono
    ds.BitsAllocated = pixels.dtype.itemsize * 8
    ds.BitsStored = ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 0
    if pixels.ndim == 3:
        ds.NumberOfFrames = pixels.shape[0]
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


class SyntheticPredictor:
    name = "synthetic_test"
    model_id = "synthetic-no-weights"

    def __init__(self):
        self.calls = 0

    def predict_image(self, pixels):
        self.calls += 1
        return "hip_right", Prediction(0, scores={"quality": 0.2})


def zip_bytes(entries):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return out.getvalue()


class ReleaseCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.saved_handlers = list(logging.root.handlers)
        self.saved_log_level = logging.root.level
        logging.root.handlers = [logging.NullHandler()]

    def tearDown(self):
        # main() installs handlers whose files live in the temporary directory.
        for handler in logging.root.handlers:
            handler.close()
        logging.root.handlers = self.saved_handlers
        logging.root.setLevel(self.saved_log_level)
        self.temporary.cleanup()

    def cli(self, *arguments):
        out, error = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            code = main(list(map(str, arguments)))
        return code, out.getvalue(), error.getvalue()

    def test_monochrome1_and_monochrome2_normalize_identically(self):
        native = (np.arange(32 * 40).reshape(32, 40) % 253).astype(np.uint8)
        a = synthetic_dicom(self.root / "normal.dcm", pixels=native)
        b = synthetic_dicom(self.root / "inverse.dcm", pixels=252 - native, mono="MONOCHROME1")
        np.testing.assert_array_equal(read_dicom(a).pixels, read_dicom(b).pixels)
        self.assertEqual(read_dicom(a).pixels.dtype, np.uint8)

    def test_release_hip_preprocessing_exactly_matches_training_transform(self):
        import torch
        from PIL import Image
        from torchvision import transforms as T
        from dxaqc.release_predictor import hip_tensor

        torch.set_num_threads(2)
        for shape in ((37, 53), (53, 37)):
            native = (np.arange(np.prod(shape)).reshape(shape) % 253).astype(np.uint8)
            path = synthetic_dicom(self.root / "native.dcm", pixels=native)
            normalized = read_dicom(path).pixels
            for region in ("hip_left", "hip_right"):
                data = native[:, ::-1] if region == "hip_left" else native
                height, width = data.shape
                side = max(height, width)
                canvas = np.zeros((side, side), dtype=np.uint8)
                canvas[(side-height)//2:(side-height)//2+height, (side-width)//2:(side-width)//2+width] = data
                for resolution in (64, 384):
                    transform = T.Compose([T.Resize((resolution, resolution)), T.Grayscale(3), T.ToTensor(),
                                           T.Normalize([.485, .456, .406], [.229, .224, .225])])
                    expected = transform(Image.fromarray(canvas))[None]
                    with self.subTest(shape=shape, region=region, resolution=resolution):
                        torch.testing.assert_close(hip_tensor(normalized, region, resolution), expected, rtol=0, atol=0)

    def test_missing_uid_is_stable_across_zip_extractions(self):
        path = synthetic_dicom(self.root / "source.dcm", uids=False)
        archive = self.root / "batch.zip"
        archive.write_bytes(zip_bytes([("study/image.dcm", path.read_bytes())]))
        first, _ = analyze(archive, predictor=SyntheticPredictor())
        second, _ = analyze(archive, predictor=SyntheticPredictor())
        for name in ("path_to_study", "study_uid", "image_uid", "quality_class", "violation_type"):
            self.assertEqual(first[0][name], second[0][name])
        self.assertTrue(first[0]["study_uid"].startswith("2.25."))
        self.assertEqual(first[0]["path_to_study"], "study/image.dcm")

    def test_corruption_does_not_stop_batch_and_duplicate_files_keep_rows(self):
        batch = self.root / "input"
        first = synthetic_dicom(batch / "a.dcm")
        synthetic_dicom(batch / "b.dcm", image_id=2)
        (batch / "broken.dcm").write_bytes(b"broken")
        predictor = SyntheticPredictor()
        rows, stats = analyze(batch, predictor=predictor)
        self.assertEqual(len(rows), 3)
        self.assertEqual(stats["success"], 2)
        self.assertEqual(stats["failure"], 1)
        self.assertEqual(stats["duplicates"], 1)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(len({row["image_uid"] for row in rows if row["processing_status"] == "Success"}), 2)
        self.assertEqual(check_report(rows), [])
        self.assertTrue(first.is_file())

    def test_directory_path_mode_allows_several_images_per_study(self):
        synthetic_dicom(self.root / "input/study/a.dcm")
        synthetic_dicom(self.root / "input/study/b.dcm", image_id=2)
        rows, _ = analyze(self.root / "input", predictor=SyntheticPredictor(), path_mode="dir")
        self.assertEqual([row["path_to_study"] for row in rows], ["study", "study"])
        self.assertEqual(check_report(rows, path_mode="dir"), [])
        self.assertTrue(check_report(rows, path_mode="file"))

    def test_single_file_path_is_filename(self):
        path = synthetic_dicom(self.root / "one.dcm")
        rows, _ = analyze(path, predictor=SyntheticPredictor())
        self.assertEqual(rows[0]["path_to_study"], "one.dcm")

    def test_constant_pixels_fail_and_multiframe_has_explicit_warning(self):
        synthetic_dicom(self.root / "constant.dcm", pixels=np.zeros((32, 40), np.uint8))
        native = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
        path = synthetic_dicom(self.root / "frames.dcm", pixels=np.stack([native, native[::-1]]))
        rows, stats = analyze(self.root, predictor=SyntheticPredictor())
        self.assertEqual((stats["success"], stats["failure"]), (1, 1))
        self.assertTrue(any("multiframe" in row["error_message"] for row in rows))
        self.assertEqual(read_dicom(path).shape, (32, 40))

    def test_failed_prediction_does_not_inflate_success_counter(self):
        synthetic_dicom(self.root / "a.dcm")

        class InvalidScorePredictor(SyntheticPredictor):
            def predict_image(self, pixels):
                return "hip_right", Prediction(0, scores={"quality": "invalid"})

        rows, stats = analyze(self.root, predictor=InvalidScorePredictor())
        self.assertEqual(rows[0]["processing_status"], "Failure")
        self.assertEqual(stats["success"], 0)
        self.assertEqual(stats["failure"], 1)

    def test_zip_rejects_traversal_absolute_backslash_and_symlink(self):
        for name in ("../escape.dcm", "/absolute.dcm", "C:/absolute.dcm", "folder\\escape.dcm"):
            with self.subTest(name=name):
                archive = self.root / "unsafe.zip"
                archive.write_bytes(zip_bytes([(name, b"payload")]))
                with self.assertRaises(ValueError), input_directory(archive):
                    pass
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.write_bytes(zip_bytes([(link, b"../outside")]))
        with self.assertRaises(ValueError), input_directory(archive):
            pass

    def test_zip_rejects_duplicate_casefolded_entries_and_limits(self):
        archive = self.root / "unsafe.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.write_bytes(zip_bytes([("a.dcm", b"x"), ("A.dcm", b"x")]))
        with self.assertRaisesRegex(ValueError, "Duplicate"), input_directory(archive):
            pass
        archive.write_bytes(zip_bytes([("a.dcm", b"12345")]))
        for limit in ("MAX_FILES", "MAX_ARCHIVE_BYTES", "MAX_FILE_BYTES"):
            with self.subTest(limit=limit), patch("dxaqc.inputs." + limit, 0):
                with self.assertRaises(ValueError), input_directory(archive):
                    pass

    def test_zip_nested_input_is_processed_and_temporary_directory_removed(self):
        path = synthetic_dicom(self.root / "source.dcm")
        archive = self.root / "batch.zip"
        archive.write_bytes(zip_bytes([("one/two/image.dcm", path.read_bytes())]))
        with input_directory(archive) as extracted:
            self.assertTrue((extracted / "one/two/image.dcm").is_file())
        self.assertFalse(extracted.exists())
        rows, stats = analyze(archive, predictor=SyntheticPredictor())
        self.assertEqual(stats["success"], 1)
        self.assertEqual(rows[0]["path_to_study"], "one/two/image.dcm")

    def test_strict_csv_xlsx_have_eight_columns_and_xlsx_strings_not_formulas(self):
        row = make_row(path_to_study="=SUM(1,2)", study_uid="2.25.1", image_uid="2.25.2",
                       anatomical_region="hip_left", quality_class=0)
        csv_path, xlsx_path = self.root / "report.csv", self.root / "report.xlsx"
        write_report([row], csv_path, strict_columns=True)
        write_report([row], xlsx_path, strict_columns=True)
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, COLUMNS)
            self.assertEqual(next(reader)["path_to_study"], "=SUM(1,2)")
        workbook = load_workbook(xlsx_path, data_only=False)
        self.assertEqual([cell.value for cell in workbook.active[1]], COLUMNS)
        self.assertEqual(workbook.active["A2"].data_type, "s")
        self.assertEqual(workbook.active["A2"].value, "=SUM(1,2)")
        workbook.close()

    def test_report_contract_rejects_nonfinite_time_and_scores(self):
        base = make_row(path_to_study="a", study_uid="2.25.1", image_uid="2.25.2", anatomical_region="hip_left", quality_class=0)
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value, column="time"):
                self.assertTrue(check_report([{**base, "time_of_processing": value}]))
        for column in SCORE_COLUMNS:
            for value in (float("nan"), float("inf"), -0.1, 1.1):
                with self.subTest(column=column, value=value):
                    self.assertTrue(check_report([{**base, column: value}]))

    def test_cli_requires_models_and_never_silently_uses_dummy(self):
        path = synthetic_dicom(self.root / "input.dcm")
        with patch("dxaqc.predictor.load_predictor") as legacy:
            code, _, error = self.cli("-i", path, "-o", self.root / "report.csv", "--model-dir", self.root / "missing")
        self.assertEqual(code, 2, error)
        self.assertIn("Model bundle missing", error)
        legacy.assert_not_called()
        self.assertFalse((self.root / "report.csv").exists())

    def test_cli_empty_batch_exit_four_and_header_only_output(self):
        folder = self.root / "empty"
        folder.mkdir()
        report = self.root / "report.csv"
        code, _, error = self.cli("-i", folder, "-o", report, "--predictor", "dummy")
        self.assertEqual(code, 4, error)
        with report.open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, [COLUMNS])

    def test_cli_fail_on_error_keeps_complete_report(self):
        batch = self.root / "input"
        synthetic_dicom(batch / "a.dcm")
        (batch / "broken.dcm").write_bytes(b"broken")
        report = self.root / "report.csv"
        code, _, error = self.cli("-i", batch, "-o", report, "--predictor", "dummy", "--fail-on-error")
        self.assertEqual(code, 3, error)
        with report.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["processing_status"] for row in rows}, {"Success", "Failure"})
        summary = json.loads(report.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["status"], "partial_failure")

    def test_cli_rejects_input_output_or_log_collision_without_modification(self):
        path = synthetic_dicom(self.root / "input.csv")
        before = path.read_bytes()
        code, _, _ = self.cli("-i", path, "-o", path, "--predictor", "dummy")
        self.assertEqual(code, 2)
        self.assertEqual(path.read_bytes(), before)
        code, _, _ = self.cli("-i", path, "-o", self.root / "report.csv", "--log", path, "--predictor", "dummy")
        self.assertEqual(code, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_outputs_inside_input_are_excluded_on_first_and_repeat_run(self):
        batch = self.root / "input"
        synthetic_dicom(batch / "a.dcm")
        report = batch / "report.csv"
        for _ in range(2):
            code, _, error = self.cli("-i", batch, "-o", report, "--predictor", "dummy")
            self.assertEqual(code, 0, error)
            summary = json.loads(report.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["files"], 1)

    def test_cli_output_inside_input_cannot_overwrite_existing_dicom(self):
        batch = self.root / "input"
        path = synthetic_dicom(batch / "scan.csv")
        before = path.read_bytes()
        code, _, _ = self.cli("-i", batch, "-o", path, "--predictor", "dummy")
        self.assertEqual(path.read_bytes(), before, "Existing input DICOM was overwritten by output")
        self.assertEqual(code, 2)
        code, _, _ = self.cli("-i", batch, "-o", self.root / "report.csv", "--log", path, "--predictor", "dummy")
        self.assertEqual(path.read_bytes(), before, "Existing input DICOM was modified by logging")
        self.assertEqual(code, 2)

    def test_bundle_rejects_hash_mismatch_before_deserialization(self):
        from dxaqc.release_predictor import ReleasePredictor

        bundle = self.root / "models"
        bundle.mkdir()
        (bundle / "weights.pt").write_bytes(b"corrupt weights")
        (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"weights.pt": "0" * 64}}))
        with patch("torch.load", side_effect=AssertionError("must not deserialize")) as load:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_bundle(bundle)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ReleasePredictor(bundle)
        load.assert_not_called()

    def test_bundle_rejects_path_outside_root(self):
        outside = self.root / "outside.pt"
        outside.write_bytes(b"weights")
        bundle = self.root / "models"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"../outside.pt": hashlib.sha256(b"weights").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "Invalid bundle path"):
            verify_bundle(bundle)

    def test_http_handler_validates_transport_and_processes_zip(self):
        source = synthetic_dicom(self.root / "source.dcm")
        body = zip_bytes([("a.dcm", source.read_bytes())])
        Handler = handler_for(SyntheticPredictor())

        def request(payload=body, content_type="application/zip", length=None):
            handler = Handler.__new__(Handler)
            handler.path = "/v1/batch"
            handler.headers = Message()
            handler.headers["Content-Type"] = content_type
            handler.headers["Content-Length"] = str(len(payload) if length is None else length)
            handler.rfile = io.BytesIO(payload)
            response = []
            handler.reply = lambda status, data: response.append((status, data))
            handler.do_POST()
            return response[0]

        status, data = request()
        self.assertEqual(status, 200)
        self.assertEqual(data["summary"]["success"], 1)
        self.assertEqual(request(content_type="text/plain")[0], 415)
        self.assertEqual(request(length=0)[0], 413)
        self.assertEqual(request(length=len(body) + 10)[0], 400)
        self.assertEqual(request(payload=b"invalid ZIP")[0], 400)

    def test_http_localhost_health_and_post(self):
        try:
            server = HTTPServer(("127.0.0.1", 0), handler_for(SyntheticPredictor()))
        except PermissionError as error:
            self.skipTest(f"Localhost bind is blocked by sandbox: {error}")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["model_id"], "synthetic-no-weights")
            source = synthetic_dicom(self.root / "source.dcm")
            body = zip_bytes([("a.dcm", source.read_bytes())])
            connection.request("POST", "/v1/batch", body=body, headers={"Content-Type": "application/zip"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["summary"]["success"], 1)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
