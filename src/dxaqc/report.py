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
# Вероятность нарушения: организатор разрешил колонку quality_prob в [0; 1] для ROC-AUC (ответ на вопрос 8).
PROB_COLUMN = "quality_prob"
EXTRA_COLUMNS = ["error_message"]  # не из ТЗ; отключается флагом --strict-columns
CODES = ["spine_scan_range", "spine_axis_tilt", "spine_artifact", "hip_positioning", "hip_roi_margins"]
UNSPECIFIED = 'quality_unspecified'
SCORE_COLUMNS = ["quality_score"] + ["score_" + c for c in CODES]  # непрерывные скоры для ROC-AUC; флаг --scores
REGIONS = {"lumbar_spine": CODES[:3], "hip_right": CODES[3:], "hip_left": CODES[3:]}
# Вероятность каждого типа нарушения в той же шкале, что quality_prob (0.5 = порог этой причины):
# по ним организатор может считать ROC-AUC отдельно для каждого типа. Пусто, если тип не относится к области.
TYPE_PROB_COLUMNS = ["prob_" + c for c in CODES]

# Закрытые списки значений, которые организатор проверяет в отчёте (ответы на вопросы 6 и 15).
# Внутри сервиса остаются коды; перевод делается только при записи CSV/XLSX.
REGION_TEXT = {"lumbar_spine": "Поясничный отдел позвоночника",
               "hip_left": "Проксимальный отдел бедра", "hip_right": "Проксимальный отдел бедра"}
VIOLATION_TEXT = {"spine_scan_range": "Некорректная укладка",
                  "spine_axis_tilt": "Не выровнена ось позвоночника",
                  "spine_artifact": "Присутствуют посторонние предметы",
                  "hip_positioning": "Некорректная укладка",
                  "hip_roi_margins": "Некорректная область интереса"}


def quality_prob(score: float, threshold: float, quality_class: int) -> float:
    """Map a region's quality score onto [0, 1] with its own decision threshold at exactly 0.5.

    Spine and hip heads live on different scales; aligning each threshold to 0.5 keeps the ranking
    inside a region and makes one ROC-AUC over both regions meaningful. The result always agrees
    with quality_class (>= 0.5 iff 1), including rows where a rule overrode the binary head.
    """
    s, t = float(min(max(score, 0.0), 1.0)), float(min(max(threshold, 1e-6), 1 - 1e-6))
    p = 0.5 * s / t if s < t else 0.5 + 0.5 * (s - t) / (1 - t)
    if quality_class == 1:
        p = max(p, 0.5)
    elif quality_class == 0:
        p = min(p, 0.4999)
    return round(float(p), 4)


def type_probs(scores: dict, thresholds: dict, violations: list[str], region: str) -> dict:
    """prob_<code> for the region's codes; >= 0.5 iff the code is reported in violation_type."""
    out = {}
    for code in REGIONS.get(region, ()):
        if code in scores and code in thresholds:
            out["prob_" + code] = quality_prob(scores[code], thresholds[code], int(code in violations))
    return out


def report_text(row: dict) -> dict:
    """A row as the organizer reads it: Russian region and violation texts, ';'-separated, deduplicated."""
    out = dict(row)
    out["anatomical_region"] = REGION_TEXT.get(row.get("anatomical_region") or "", row.get("anatomical_region") or "")
    texts = []
    for code in str(row.get("violation_type") or "").split(";"):
        if code and VIOLATION_TEXT.get(code, code) not in texts:
            texts.append(VIOLATION_TEXT.get(code, code))
    out["violation_type"] = ";".join(texts)
    return out


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
        PROB_COLUMN: round(float(scores[PROB_COLUMN]), 4) if PROB_COLUMN in scores else "",
        **{c: (round(float(scores[c]), 4) if c in scores else "") for c in TYPE_PROB_COLUMNS + SCORE_COLUMNS},
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
        prob = r.get(PROB_COLUMN, "")
        if prob not in ("", None):
            if not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
                errs.append(f"строка {i}: quality_prob={prob!r}")
            elif r.get("processing_status") == "Success" and r.get("quality_class") in (0, 1) and (prob >= 0.5) != bool(r["quality_class"]):
                errs.append(f"строка {i}: quality_prob={prob} не согласуется с quality_class={r['quality_class']}")
        for code in CODES:
            p = r.get("prob_" + code, "")
            if p in ("", None):
                continue
            if not isinstance(p, (int, float)) or not 0 <= p <= 1:
                errs.append(f"строка {i}: prob_{code}={p!r}")
            elif r.get("processing_status") == "Success" and (p >= 0.5) != (code in codes):
                errs.append(f"строка {i}: prob_{code}={p} не согласуется с violation_type={codes}")
        t = r.get("time_of_processing")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            errs.append(f"строка {i}: time_of_processing={t!r}")
        for column in SCORE_COLUMNS:
            value = r.get(column, '')
            if value not in ('', None) and (not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1):
                errs.append(f'строка {i}: invalid {column}')
    return errs


def write_report(rows: list[dict], output: Path, strict_columns: bool = False, scores: bool = False,
                 type_probs: bool = False) -> Path:
    output = Path(output)
    if output.suffix.lower() not in ('.csv', '.xlsx'):
        raise ValueError('Output extension must be .csv or .xlsx')
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = COLUMNS + [PROB_COLUMN] + (TYPE_PROB_COLUMNS if type_probs else []) + ([] if strict_columns else EXTRA_COLUMNS)
    if scores:
        cols = cols + SCORE_COLUMNS
    df = pd.DataFrame([report_text(r) for r in rows], columns=cols)
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
