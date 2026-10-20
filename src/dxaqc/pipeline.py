"""Пакетная обработка: папка с DICOM → строки отчёта. Ни одно исключение не выходит наружу."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .dicom_io import DicomImage, find_dicom_files, read_dicom, study_key
from .predictor import BasePredictor, Prediction
from .region import classify_region
from .report import CODES, make_row, check_report, quality_prob, type_probs

log = logging.getLogger(__name__)

NON_DICOM_SUFFIXES = {'.xlsx', '.xls', '.xlsm', '.csv', '.tsv', '.txt', '.md', '.pdf', '.doc', '.docx', '.rtf',
                      '.json', '.xml', '.html', '.htm', '.log', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif',
                      '.tiff', '.zip', '.rar', '.7z', '.py', '.sh', '.yaml', '.yml', '.ini'}


def _rel(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def process_image(img: DicomImage, predictor: BasePredictor, cache: dict[str, tuple[str, Prediction]]) -> tuple[str, Prediction, bool]:
    """Область + предсказание. Дубликаты по пикселям считаются один раз (cache по md5)."""
    key = img.md5
    if key in cache:
        region, pred = cache[key]
        return region, pred, True
    if hasattr(predictor, 'predict_image'):
        spacing = img.meta.get('spacing_mm')
        try:
            region, pred = predictor.predict_image(img.pixels, mm_per_px=spacing)
        except TypeError:  # legacy predictors without the spacing argument
            region, pred = predictor.predict_image(img.pixels)
    else:
        rr = classify_region(img.pixels)
        region = rr.region
        pred = predictor.predict(img.pixels, region)
        pred.details.update({"spine_score": rr.spine_score, "side_score": rr.side_score, "region_method": rr.method})
    cache[key] = (region, pred)
    return region, pred, False


def process_folder(root: Path, predictor: BasePredictor, *, path_mode: str = "file", exclude: tuple[Path, ...] = (),
                   overlay_dir: Path | None = None, on_result=None, on_start=None,
                   sr_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Возвращает строки отчёта и сводку. path_mode: 'file' — путь к файлу, 'dir' — папка исследования.

    on_start(paths) получает относительные пути всех файлов до обработки;
    on_result(index, total, row, img, region, pred) вызывается после каждого файла (для прогресса в
    веб-интерфейсе); img/region/pred — None, если файл не прочитан. Ошибка колбэка не влияет на отчёт.
    """
    root = Path(root)
    excluded = {p.resolve() for p in exclude}
    found = [p for p in find_dicom_files(root) if p.resolve() not in excluded]
    # documents and pictures that ship next to the studies (e.g. a label spreadsheet) are not
    # images: they are logged and counted, not reported as failed rows
    files = [p for p in found if p.suffix.lower() not in NON_DICOM_SUFFIXES]
    non_dicom = [p for p in found if p.suffix.lower() in NON_DICOM_SUFFIXES]
    for p in non_dicom:
        log.info("skip non-DICOM file %s", _rel(p, root))
    if not files:
        log.warning("входная папка пуста: %s", root)
    rows: list[dict] = []
    cache: dict[str, tuple[str, Prediction]] = {}
    stats = {"files": len(files), "success": 0, "failure": 0, "skipped_non_dicom": len(non_dicom), "duplicates": 0, "studies": set(),
             "overlays": 0, "overlay_failures": 0, "sr": 0, "sr_failures": 0}
    t_all = time.perf_counter()
    if on_start is not None:
        try:
            on_start([_rel(f, root) for f in files])
        except Exception:  # noqa: BLE001
            log.exception("start callback failed")

    def notify(i, img=None, region=None, pred=None):
        if on_result is None:
            return
        try:
            on_result(i, len(files), rows[-1], img, region, pred)
        except Exception:  # noqa: BLE001 — a progress listener must never break the batch
            log.exception("progress callback failed")

    for i, f in enumerate(files):
        t0 = time.perf_counter()
        rel_file = _rel(f, root)
        try:
            img = read_dicom(f, source_id=rel_file)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            log.warning("FAIL read %s: %s", rel_file, msg)
            stats["failure"] += 1
            rows.append(make_row(path_to_study=rel_file, status="Failure", seconds=time.perf_counter() - t0, error=msg))
            notify(i)
            continue
        skey = study_key(img, root)
        stats["studies"].add(skey)
        p2s = _rel(f.parent, root) if path_mode == "dir" else rel_file
        try:
            region, pred, dup = process_image(img, predictor, cache)
            if dup:
                stats["duplicates"] += 1
            if img.warnings:
                log.warning("%s: %s", rel_file, "; ".join(img.warnings))
            sc = {"score_" + c: v for c, v in pred.scores.items() if c in CODES}
            sc["quality_score"] = pred.scores.get("quality", pred.scores.get("hip_quality", max(sc.values()) if sc else float(pred.quality_class)))
            sc["quality_prob"] = quality_prob(sc["quality_score"], pred.details.get("quality_threshold", 0.5), int(pred.quality_class))
            sc.update(type_probs(pred.scores, pred.details.get("reason_thresholds", {}), list(pred.violations), region))
            row = make_row(
                path_to_study=p2s, study_uid=img.study_uid, image_uid=img.sop_uid, anatomical_region=region,
                quality_class=int(pred.quality_class), violation_type=";".join(pred.violations),
                status="Success", seconds=time.perf_counter() - t0,
                error="; ".join(img.warnings + pred.details.get('warnings', [])), scores=sc,
            )
            errors = check_report([row], path_mode=path_mode)
            if errors:
                raise ValueError('Invalid model output: ' + '; '.join(errors))
            if overlay_dir is not None:
                try:
                    from .overlay import write_overlay
                    stats["overlays"] += bool(write_overlay(Path(overlay_dir), rel_file, img, region, pred))
                except Exception as oe:  # noqa: BLE001 — visualisation must never fail the row
                    stats["overlay_failures"] += 1
                    log.warning("overlay failed %s: %s", rel_file, oe)
            if sr_dir is not None:
                try:
                    from .sr import write_sr
                    row["_sr_file"] = write_sr(Path(sr_dir), rel_file, img, region, pred, sc.get("quality_prob"),
                                               getattr(predictor, 'model_id', None) or getattr(predictor, 'name', 'dxaqc')).name
                    stats["sr"] += 1
                except Exception as se:  # noqa: BLE001 — the report must never fail the row
                    stats["sr_failures"] += 1
                    log.warning("SR failed %s: %s", rel_file, se)
            rows.append(row)
            stats["success"] += 1
            notify(i, img, region, pred)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            log.exception("FAIL predict %s", rel_file)
            stats["failure"] += 1
            rows.append(make_row(path_to_study=p2s, study_uid=img.study_uid, image_uid=img.sop_uid,
                                 status="Failure", seconds=time.perf_counter() - t0, error=msg))
            notify(i, img)
    stats["studies"] = len(stats["studies"])
    stats["total_seconds"] = round(time.perf_counter() - t_all, 3)
    stats['success_rate'] = stats['success'] / len(files) if files else 0.0
    durations = {}
    for row in rows:
        if row['study_uid']:
            durations[row['study_uid']] = durations.get(row['study_uid'], 0) + row['time_of_processing']
    stats['max_study_seconds'] = round(max(durations.values(), default=0), 4)
    return rows, stats
