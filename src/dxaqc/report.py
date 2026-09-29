"""Итоговая таблица в формате ТЗ (п. 2.5)."""
from __future__ import annotations

from pathlib import Path
import math
import os
import tempfile

import pandas as pd

COLUMNS = [
    "path_to_study", "study_uid", "image_uid", "anatomical_region",
    "quality_class", "violation_type", "processing_status", "time_of_processing",
]
EXTRA_COLUMNS = ["error_message"]  # не из ТЗ; отключается флагом --strict-columns
CODES = ["spine_scan_range", "spine_axis_tilt", "spine_artifact", "hip_positioning", "hip_roi_margins"]
UNSPECIFIED = 'quality_unspecified'
SCORE_COLUMNS = ["quality_score"] + ["score_" + c for c in CODES]  # непрерывные скоры для ROC-AUC; флаг --scores
REGIONS = {"lumbar_spine": CODES[:3], "hip_right": CODES[3:], "hip_left": CODES[3:]}


def make_row(*, path_to_study: str, study_uid: str = "", image_uid: str = "", anatomical_region: str = "",
             quality_class: int | None = None, violation_type: str = "", status: str = "Success",
             seconds: float = 0.0, error: str = "", scores: dict | None = None) -> dict:
    scores = scores or {}
    return {
        "path_to_study": path_to_study,
        "study_uid": study_uid,
        "image_uid": image_uid,
        "anatomical_region": anatomical_region,
        "quality_class": quality_class if quality_class is not None else "",
        "violation_type": violation_type,
        "processing_status": status,
        "time_of_processing": round(float(seconds), 4),
        "error_message": error,
        **{c: (round(float(scores[c]), 4) if c in scores else "") for c in SCORE_COLUMNS},
    }


def check_report(rows: list[dict], *, path_mode: str = 'file') -> list[str]:
    """Самопроверка контракта: дубли путей, согласованность quality_class и кодов, коды по области, Failure без решения."""
    errs, seen = [], set()
    for i, r in enumerate(rows, 1):
        p = r.get("path_to_study", "")
        if not p or (p in seen and path_mode == 'file'):
            errs.append(f"строка {i}: пустой или повторный path_to_study")
        seen.add(p)
        codes = [c for c in str(r.get("violation_type") or "").split(";") if c]
        if r.get("processing_status") == "Success":
            q = r.get("quality_class")
            if q not in (0, 1):
                errs.append(f"строка {i}: quality_class={q!r}")
            elif int(q) != int(bool(codes)):
                errs.append(f"строка {i}: quality_class={q} при violation_type={codes}")
            reg = r.get("anatomical_region")
            if reg not in REGIONS:
                errs.append(f"строка {i}: область {reg!r}")
            elif set(codes) - set(REGIONS[reg]) - {UNSPECIFIED}:
                errs.append(f"строка {i}: коды не для области {reg}")
            if not r.get("study_uid") or not r.get("image_uid"):
                errs.append(f"строка {i}: пустой uid")
        elif r.get("processing_status") == "Failure":
            if r.get("quality_class") not in ("", None) or codes:
                errs.append(f"строка {i}: Failure с решением о качестве")
        else:
            errs.append(f"строка {i}: processing_status={r.get('processing_status')!r}")
        t = r.get("time_of_processing")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            errs.append(f"строка {i}: time_of_processing={t!r}")
        for column in SCORE_COLUMNS:
            value = r.get(column, '')
            if value not in ('', None) and (not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1):
                errs.append(f'строка {i}: invalid {column}')
    return errs


def write_report(rows: list[dict], output: Path, strict_columns: bool = False, scores: bool = False) -> Path:
    output = Path(output)
    if output.suffix.lower() not in ('.csv', '.xlsx'):
        raise ValueError('Output extension must be .csv or .xlsx')
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = COLUMNS if strict_columns else COLUMNS + EXTRA_COLUMNS
    if scores:
        cols = cols + SCORE_COLUMNS
    df = pd.DataFrame(rows, columns=cols)
    fd, tmp = tempfile.mkstemp(prefix='.dxaqc-', suffix=output.suffix, dir=output.parent)
    os.close(fd)
    try:
        if output.suffix.lower() == '.xlsx':
            from openpyxl import Workbook
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = 'DXA QC'
            sheet.append(cols)
            for row in df.itertuples(index=False, name=None):
                sheet.append(row)
                for cell in sheet[sheet.max_row]:
                    if isinstance(cell.value, str):
                        cell.data_type = 's'
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            workbook.save(tmp)
        else:
            df.to_csv(tmp, index=False, encoding='utf-8')
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return output
