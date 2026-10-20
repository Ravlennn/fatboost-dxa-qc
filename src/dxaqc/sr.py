"""DICOM SR: the verdict and its measurements as a structured text report (TZ 2.6).

One Basic Text SR per analysed image, in the source study, in its own series "DXA QC report (AI)".
It carries the organizer's wording (same texts as the CSV), the measurements in mm and degrees,
the probability, the warnings and the model version, and references the source image, so a PACS
files it next to the study and a reader sees why the image was flagged.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from .report import REGION_TEXT, VIOLATION_TEXT

BASIC_TEXT_SR = '1.2.840.10008.5.1.4.1.1.88.11'
SCHEME = 'DXAQC'                       # private coding scheme for the concept names below
_series: dict[str, str] = {}

PATIENT_STUDY = ('PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex', 'StudyDate', 'StudyTime',
                 'AccessionNumber', 'StudyID', 'ReferringPhysicianName', 'StudyDescription')


def _uid(*parts: str) -> str:
    """Deterministic 2.25 UID: the same image always gets the same report UID."""
    return '2.25.' + str(int(hashlib.sha256('|'.join(parts).encode()).hexdigest()[:30], 16))


def _code(value: str, meaning: str, scheme: str = SCHEME) -> Dataset:
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def _text(code: str, name: str, value: str) -> Dataset:
    item = Dataset()
    item.RelationshipType, item.ValueType = 'CONTAINS', 'TEXT'
    item.ConceptNameCodeSequence = [_code(code, name)]
    item.TextValue = value
    return item


def _num(code: str, name: str, value: float, unit: str, unit_meaning: str) -> Dataset:
    item = Dataset()
    item.RelationshipType, item.ValueType = 'CONTAINS', 'NUM'
    item.ConceptNameCodeSequence = [_code(code, name)]
    mv = Dataset()
    mv.NumericValue = f'{value:.1f}'
    mv.MeasurementUnitsCodeSequence = [_code(unit, unit_meaning, 'UCUM')]
    item.MeasuredValueSequence = [mv]
    return item


def _measurements(region: str, details: dict) -> list[Dataset]:
    lm = details.get('landmarks') or {}
    m = lm.get('measurements') or {}
    out = []
    if region == 'lumbar_spine' and m:
        tilt = m.get('spine_tilt_deg_calibrated', m.get('spine_tilt_deg'))
        if tilt is not None:
            out.append(_num('TILT', 'Наклон оси L1–L4', abs(float(tilt)), 'deg', 'градус'))
        for key, name in (('top_margin_mm', 'Поле над L1'), ('bottom_margin_mm', 'Поле под L4')):
            if key in m:
                out.append(_num(key.upper()[:16], name, float(m[key]), 'mm', 'миллиметр'))
    elif region.startswith('hip') and m:
        for key, name in (('roi_top_mm', 'Поле ROI сверху'), ('roi_bottom_mm', 'Поле ROI снизу'),
                          ('roi_left_mm', 'Поле ROI слева'), ('roi_right_mm', 'Поле ROI справа')):
            if key in m:
                out.append(_num(key.upper()[:16], name, float(m[key]), 'mm', 'миллиметр'))
        if 'neck_axis_deg' in m:
            out.append(_num('NECK_AXIS', 'Ось шейки бедра', float(m['neck_axis_deg']), 'deg', 'градус'))
        lt = lm.get('lesser_trochanter')
        if lt:
            if lt.get('visible'):
                out.append(_num('LT_PROMINENCE', 'Выступ малого вертела', float(lt['lt_prominence_mm']), 'mm', 'миллиметр'))
            else:
                out.append(_text('LT_PROMINENCE', 'Выступ малого вертела', 'малый вертел за контуром не виден'))
    return out


def build_sr(img, region: str, pred, quality_prob: float | None, model_id: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = BASIC_TEXT_SR
    meta.MediaStorageSOPInstanceUID = _uid('dxaqc-sr', img.sop_uid, model_id)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SpecificCharacterSet = 'ISO_IR 192'              # UTF-8: the report is in Russian
    try:                                                 # patient/study attributes so PACS files it with the study
        src = pydicom.dcmread(img.path, stop_before_pixels=True, force=True)
        for key in PATIENT_STUDY:
            if key in src:
                setattr(ds, key, src.data_element(key).value)
    except Exception:                                    # noqa: BLE001 — a report without them is still valid
        pass
    now = datetime.datetime.now()
    ds.SOPClassUID, ds.SOPInstanceUID = BASIC_TEXT_SR, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = img.study_uid
    ds.SeriesInstanceUID = _series.setdefault(img.study_uid, _uid('dxaqc-sr-series', img.study_uid, model_id))
    ds.Modality, ds.SeriesNumber, ds.InstanceNumber = 'SR', 9901, 1
    ds.SeriesDescription = 'DXA QC report (AI)'
    ds.Manufacturer = 'DXA QC (AI)'
    ds.ContentDate, ds.ContentTime = now.strftime('%Y%m%d'), now.strftime('%H%M%S')
    ds.CompletionFlag, ds.VerificationFlag = 'COMPLETE', 'UNVERIFIED'   # AI output, not signed by a physician
    ds.ValueType, ds.ContinuityOfContent = 'CONTAINER', 'SEPARATE'
    ds.ConceptNameCodeSequence = [_code('DXAQC_REPORT', 'Контроль качества денситометрии')]
    ref = Dataset()
    ref.StudyInstanceUID = img.study_uid
    series = Dataset()
    series.SeriesInstanceUID = img.series_uid or img.study_uid
    inst = Dataset()
    inst.ReferencedSOPClassUID = img.meta.get('sop_class_uid') or '1.2.840.10008.5.1.4.1.1.1'
    inst.ReferencedSOPInstanceUID = img.sop_uid
    series.ReferencedSOPSequence = [inst]
    ref.ReferencedSeriesSequence = [series]
    ds.CurrentRequestedProcedureEvidenceSequence = [ref]

    texts = [VIOLATION_TEXT.get(c, c) for c in pred.violations]
    items = [
        _text('REGION', 'Анатомическая область', REGION_TEXT.get(region, region)),
        _text('VERDICT', 'Заключение', 'Есть нарушение качества' if pred.quality_class else 'Исследование качественное'),
        _text('VIOLATIONS', 'Нарушения', '; '.join(dict.fromkeys(texts)) if texts else 'Нет'),
    ]
    if quality_prob is not None:
        items.append(_num('QUALITY_PROB', 'Вероятность нарушения', 100 * float(quality_prob), '%', 'процент'))
    items += _measurements(region, pred.details)
    notes = [w for w in pred.details.get('warnings', []) if isinstance(w, str)]
    if notes:
        items.append(_text('WARNINGS', 'Предупреждения', '; '.join(notes)))
    items.append(_text('DISCLAIMER', 'Примечание',
                        'Автоматический анализ качества укладки; требует подтверждения врачом.'))
    items.append(_text('MODEL', 'Модель', model_id))
    ds.ContentSequence = items
    return ds


def write_sr(out_dir: Path, rel_name: str, img, region: str, pred, quality_prob, model_id: str) -> Path:
    from .overlay import _safe
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe(Path(rel_name).with_suffix('').as_posix())
    target = out_dir / f'{stem}_SR.dcm'
    if target.exists():
        target = out_dir / f'{stem}_{img.sop_uid[-12:]}_SR.dcm'
    build_sr(img, region, pred, quality_prob, model_id).save_as(target, enforce_file_format=True)
    return target


def reviewed_sr(path: Path, quality: int, violations: list[str], quality_prob: float | None,
                corrected_points: bool, at: str) -> FileDataset:
    """The model's SR with the specialist's confirmed decision: new instance, VERIFIED, verdict replaced.
    The model's own verdict stays in the report as a separate item so the change is traceable."""
    ds = pydicom.dcmread(path)
    items = list(ds.ContentSequence)
    by_code = {i.ConceptNameCodeSequence[0].CodeValue: i for i in items}
    model_verdict = f"{by_code['VERDICT'].TextValue}; {by_code['VIOLATIONS'].TextValue}"
    texts = [VIOLATION_TEXT.get(c, c) for c in violations]
    by_code['VERDICT'].TextValue = 'Есть нарушение качества' if quality else 'Исследование качественное'
    by_code['VIOLATIONS'].TextValue = '; '.join(dict.fromkeys(texts)) if texts else 'Нет'
    if 'QUALITY_PROB' in by_code and quality_prob is not None:
        by_code['QUALITY_PROB'].MeasuredValueSequence[0].NumericValue = f'{100 * float(quality_prob):.1f}'
    if 'DISCLAIMER' in by_code:
        by_code['DISCLAIMER'].TextValue = 'Заключение подтверждено врачом в интерфейсе DXA QC.'
    pos = items.index(by_code['VIOLATIONS']) + 1
    items[pos:pos] = [_text('MODEL_VERDICT', 'Заключение модели', model_verdict),
                      _text('REVIEW', 'Проверка врачом',
                            'Точки разметки исправлены врачом' if corrected_points else 'Разметка модели принята')]
    ds.ContentSequence = items
    uid = _uid('dxaqc-sr-reviewed', ds.SOPInstanceUID, at)
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = uid
    ds.InstanceNumber = 2
    stamp = at.replace('-', '').replace(':', '').replace('T', '')
    obs = Dataset()
    obs.VerifyingObserverName = 'Врач'
    obs.VerifyingOrganization = 'DXA QC'
    obs.VerificationDateTime = stamp
    obs.VerifyingObserverIdentificationCodeSequence = []
    ds.VerifyingObserverSequence = [obs]
    ds.VerificationFlag = 'VERIFIED'
    ds.ContentDate, ds.ContentTime = stamp[:8], stamp[8:14]
    return ds
