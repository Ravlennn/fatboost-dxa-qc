"""Visualisation series (TZ 2.6): PNG + DICOM Secondary Capture per image."""
from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .dicom_io import DicomImage
from .landmarks import draw_overlay
from .predictor import Prediction

SECONDARY_CAPTURE = '1.2.840.10008.5.1.4.1.1.7'
_series_uids: dict[str, str] = {}


def _safe(name: str) -> str:
    return re.sub(r'[^\w.-]+', '_', name).strip('_')[-120:] or 'image'


def secondary_capture(rgb: np.ndarray, img: DicomImage, description: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
    now = datetime.datetime.now()
    ds.SOPClassUID = SECONDARY_CAPTURE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = img.study_uid
    # One derived series per source series, as the TZ asks for an additional series.
    ds.SeriesInstanceUID = _series_uids.setdefault(img.series_uid or img.study_uid, generate_uid())
    ds.Modality = 'OT'
    ds.ConversionType = 'WSD'
    ds.SeriesDescription = 'DXA QC overlay (AI)'
    ds.ImageComments = description[:10240]
    ds.DerivationDescription = f'dxaqc landmarks overlay of SOP {img.sop_uid}'
    ds.ContentDate, ds.ContentTime = now.strftime('%Y%m%d'), now.strftime('%H%M%S')
    ds.PatientID, ds.PatientName = 'Anonymized', 'Anonymized'
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, 'RGB', 0
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return ds


def write_overlay(out_dir: Path, rel_name: str, img: DicomImage, region: str, pred: Prediction) -> list[Path]:
    lm = pred.details.get('landmarks')
    if not lm:
        return []
    rgb = draw_overlay(img.pixels, region, lm['points'], lm['measurements'], pred.violations, pred.quality_class,
                       lt=lm.get('lesser_trochanter'), art_mask=lm.get('artifact_mask'))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe(Path(rel_name).with_suffix('').as_posix())
    if (out_dir / f'{stem}.png').exists():  # distinct inputs must never overwrite each other
        stem = f'{stem}_{img.sop_uid[-12:]}'
    png, dcm = out_dir / f'{stem}.png', out_dir / f'{stem}.dcm'
    Image.fromarray(rgb).save(png)
    text = f"region={region}; quality_class={pred.quality_class}; violations={';'.join(pred.violations)}; " + \
        '; '.join(f'{k}={v:.2f}' for k, v in lm['measurements'].items())
    if lm.get('lesser_trochanter'):
        text += f"; lesser_trochanter_mm={lm['lesser_trochanter']['lt_prominence_mm']:.2f}"
    secondary_capture(rgb, img, text).save_as(dcm, enforce_file_format=True)
    return [png, dcm]
