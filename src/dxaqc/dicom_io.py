"""Чтение DICOM: обход папки, нормализация пикселей, извлечение идентификаторов."""
from __future__ import annotations

import hashlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

SKIP_NAMES = {".DS_Store", "DICOMDIR", "Thumbs.db"}


@dataclass
class DicomImage:
    path: Path
    study_uid: str
    series_uid: str
    sop_uid: str
    pixels: np.ndarray  # uint8, 2D (H, W)
    meta: dict = field(default_factory=dict)

    @property
    def md5(self) -> str:
        """Хэш пикселей с учётом формы: одинаковые байты разной формы — разные изображения."""
        h = hashlib.md5(str(self.pixels.shape).encode()); h.update(self.pixels.tobytes())
        return h.hexdigest()

    @property
    def warnings(self) -> list[str]:
        return self.meta.get("warnings", [])

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape


def find_dicom_files(root: Path) -> list[Path]:
    """Все файлы под root, отсортированные, кроме служебных. Проверка на DICOM — при чтении."""
    root = Path(root)
    if root.is_file():
        return [root]
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name not in SKIP_NAMES and not p.name.startswith("._")]
    return files


def _to_uint8(ds: pydicom.Dataset, warn: list[str] | None = None) -> np.ndarray:
    arr = ds.pixel_array
    nframes = int(ds.get("NumberOfFrames", 1) or 1)
    if nframes > 1 and warn is not None:
        warn.append(f"multiframe: {nframes} кадров, взят первый")
    if arr.ndim == 3:
        # многокадровый (F,H,W) → первый кадр; RGB (H,W,3) → серый
        if arr.shape[-1] in (3, 4) and arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) > 1:
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0, ..., :3].mean(axis=-1)
    arr = arr.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if not np.isfinite(arr).all():
        raise InvalidDicomError('non-finite pixel values or rescale parameters')
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


DEFAULT_SPACING_MM = 0.6  # GE Lunar Prodigy: 300 px = 180 мм, 280 px = 170 мм (см. docs/EDA.md)


def estimate_pixel_spacing(meta: dict) -> tuple[float, str]:
    """Размер пикселя в мм. PixelSpacing → ExposedArea/матрица (если согласуются) → константа 0.6."""
    ps = meta.get("PixelSpacing")
    if ps and all(0.05 < float(v) < 5 for v in ps):
        return float(np.mean(ps)), "PixelSpacing"
    ea = meta.get("ExposedArea")
    rows, cols = meta.get("Rows"), meta.get("Columns")
    if ea and len(ea) == 2 and rows and cols:
        sw, sh = ea[0] / cols, ea[1] / rows
        # Any vendor: accept when both axes agree within 10% and the value is physically sane.
        if 0.15 <= sw <= 1.5 and 0.15 <= sh <= 1.5 and abs(sw - sh) <= 0.1 * max(sw, sh):
            return float((sw + sh) / 2), "ExposedArea"
    return DEFAULT_SPACING_MM, "default"


def read_dicom(path: Path, *, source_id: str | None = None) -> DicomImage:
    """Читает один файл. Бросает InvalidDicomError, если это не DICOM с пикселями."""
    path = Path(path)
    if path.stat().st_size > 512 * 1024 ** 2:
        raise InvalidDicomError('DICOM exceeds 512 MiB input limit')
    try:
        ds = pydicom.dcmread(str(path), force=True)
    except Exception as e:  # noqa: BLE001
        raise InvalidDicomError(f"не удалось прочитать как DICOM: {e}") from e
    if "PixelData" not in ds:
        raise InvalidDicomError("нет PixelData")
    count = int(ds.get('Rows', 0)) * int(ds.get('Columns', 0))
    count *= int(ds.get('NumberOfFrames', 1) or 1) * int(ds.get('SamplesPerPixel', 1))
    if count <= 0 or count > 25_000_000:
        raise InvalidDicomError('invalid dimensions or decoded pixel limit exceeded')
    if max(int(ds.get('Rows', 0)), int(ds.get('Columns', 0))) > 4096:
        raise InvalidDicomError('image side exceeds 4096 pixel padding limit')
    warn: list[str] = []
    pixels = _to_uint8(ds, warn)
    if pixels.ndim != 2 or min(pixels.shape) < 16 or max(pixels.shape) > 4096:
        raise InvalidDicomError(f"неожиданная форма изображения {pixels.shape}")
    if pixels.max() == pixels.min():
        raise InvalidDicomError("пустой кадр: изображение константное")
    study_uid = str(ds.get("StudyInstanceUID", "") or "")
    sop_uid = str(ds.get("SOPInstanceUID", "") or "")
    if not sop_uid:
        sop_uid = "2.25." + str(int.from_bytes(hashlib.md5(str(pixels.shape).encode() + pixels.tobytes()).digest(), 'big'))
        warn.append("нет SOPInstanceUID, image_uid сгенерирован из пикселей")
    if not study_uid:
        parent = str(Path(source_id).parent) if source_id is not None else str(path.parent.resolve())
        study_uid = "2.25." + str(int.from_bytes(hashlib.md5(parent.encode()).digest(), 'big'))
        warn.append("нет StudyInstanceUID, study_uid сгенерирован из папки")
    def numbers(key, cast=float):
        value = ds.get(key)
        if value is None or value == '':
            return None
        try:
            return [cast(x) for x in value]
        except (ValueError, TypeError):
            warn.append(f'invalid optional metadata: {key}')
            return None

    meta = {
        "Modality": str(ds.get("Modality", "")),
        "Manufacturer": str(ds.get("Manufacturer", "")),
        "Model": str(ds.get("ManufacturerModelName", "")),
        "BodyPartExamined": str(ds.get("BodyPartExamined", "")),
        "Laterality": str(ds.get("ImageLaterality", "") or ds.get("Laterality", "")),
        "SeriesDescription": str(ds.get("SeriesDescription", "")),
        "PixelSpacing": numbers('PixelSpacing'),
        "ExposedArea": numbers('ExposedArea', int),
        "PatientOrientation": [str(v) for v in ds.PatientOrientation] if "PatientOrientation" in ds else None,
        "Rows": int(pixels.shape[0]),
        "Columns": int(pixels.shape[1]),
        "warnings": warn,
    }
    meta["spacing_mm"], meta["spacing_source"] = estimate_pixel_spacing(meta)
    return DicomImage(
        path=path,
        study_uid=study_uid,
        series_uid=str(ds.get("SeriesInstanceUID", "") or ""),
        sop_uid=sop_uid,
        pixels=pixels,
        meta=meta,
    )


def study_key(img: DicomImage, root: Path) -> str:
    """Ключ исследования: StudyInstanceUID, при его отсутствии — папка файла относительно root."""
    if img.study_uid:
        return img.study_uid
    try:
        return str(img.path.parent.relative_to(root))
    except ValueError:
        return str(img.path.parent)
