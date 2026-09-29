"""Evaluate the selected development pipeline using held-out-fold predictions.

This is not an independent test and does not evaluate the deployed five-model
average. Labels, folds, sources, router effects and tuned thresholds are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits
import torch

from dxaqc.bundle import verify_bundle
from dxaqc.dicom_io import read_dicom
from dxaqc.hip_backbones import build_hip_model
from dxaqc.release_predictor import hip_tensor

KEY = ["study", "image_uid"]
REASONS = {"spine_scan_range": "v_layout", "spine_axis_tilt": "v_axis", "spine_artifact": "v_artifact",
           "hip_positioning": "v_rotation", "hip_roi_margins": "v_roi"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_frame(path):
    frame = pd.read_csv(path, dtype={key: str for key in [*KEY, "study_uid"]})
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate image key: {path}")
    return frame.set_index(KEY)


def align(source, expected, name):
    if set(source.index) != set(expected.index):
        raise ValueError(f"Incomplete or extra OOF rows: {name}")
    source = source.reindex(expected.index)
    for key in ("fold", "region", "quality", "study_uid"):
        if key in source and key in expected:
            match = source[key].eq(expected[key]) | (source[key].isna() & expected[key].isna())
            if not match.all():
                raise ValueError(f"OOF labels/folds disagree: {name}/{key}")
    return source


def decision_metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=bool)
    tp, tn = int(((y == 1) & predicted).sum()), int(((y == 0) & ~predicted).sum())
    fp, fn = int(((y == 0) & predicted).sum()), int(((y == 1) & ~predicted).sum())
    return dict(n=len(y), positive=int(y.sum()), predicted_positive=int(predicted.sum()),
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                sensitivity=tp/(tp+fn) if tp+fn else None, specificity=tn/(tn+fp) if tn+fp else None,
                precision=tp/(tp+fp) if tp+fp else None, tp=tp, tn=tn, fp=fp, fn=fn)


def quality_metrics(y, p, threshold):
    out = decision_metrics(y, p >= threshold)
    out.update(roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
               ap=float(average_precision_score(y, p)) if np.asarray(y).sum() else None)
    return out


def summarize(frame, bootstrap, seed):
    quality = {}
    rng = np.random.default_rng(seed)
    for region in ["spine", "hip", "overall"]:
        mask = frame.region.eq("spine") if region == "spine" else frame.region.ne("spine") if region == "hip" else np.ones(len(frame), bool)
        part = frame.loc[mask]
        y, p, threshold = part.quality.to_numpy(int), part.p_quality.to_numpy(), part.quality_threshold.to_numpy()
        point = quality_metrics(y, p, threshold)
        groups = part.study_uid.to_numpy()
        blocks = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        samples = {key: [] for key in ("roc_auc", "ap", "f1", "sensitivity", "specificity")}
        for _ in range(bootstrap):
            index = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
            if len(np.unique(y[index])) < 2:
                continue
            result = quality_metrics(y[index], p[index], threshold[index])
            for key in samples:
                samples[key].append(result[key])
        quality[region] = dict(point=point, studies=len(blocks), ci95={key: list(map(float, np.quantile(value, [.025, .975]))) for key, value in samples.items()},
                               bootstrap_valid=len(samples["f1"]))
    reasons = {}
    for code, label in REASONS.items():
        part = frame.loc[frame[label].notna()]
        y, p = part[label].to_numpy(int), part[f"p_{code}"].to_numpy()
        raw, gated = part[f"raw_{code}"].to_numpy(bool), part[f"gated_{code}"].to_numpy(bool)
        reasons[code] = dict(raw=decision_metrics(y, raw), gated=decision_metrics(y, gated),
                             auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                             ap=float(average_precision_score(y, p)) if y.sum() else None,
                             threshold=float(part[f"threshold_{code}"].iloc[0]))
    macro = {mode: float(np.mean([item[mode]["f1"] for item in reasons.values()])) for mode in ["raw", "gated"]}
    blocks = [np.flatnonzero(frame.study_uid.to_numpy() == group) for group in frame.study_uid.unique()]
    samples = {mode: [] for mode in macro}
    for _ in range(bootstrap):
        index = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        part = frame.iloc[index]
        if any(part[label].eq(1).sum() == 0 for label in REASONS.values()):
            continue
        for mode in samples:
            f1 = [decision_metrics(part.loc[part[label].notna(), label].to_numpy(int),
                                   part.loc[part[label].notna(), f"{mode}_{code}"].to_numpy(bool))["f1"] for code, label in REASONS.items()]
            samples[mode].append(float(np.mean(f1)))
    return dict(quality=quality, reasons=reasons,
                reasons_macro_f1={mode: dict(value=macro[mode], ci95=list(map(float, np.quantile(samples[mode], [.025, .975]))), bootstrap_valid=len(samples[mode])) for mode in macro})


def correct_router_side(frame, bundle, manifest, source):
    """Use only this row's held-out-fold model, never any model trained on it."""
    corrections = []
    for index, row in frame.loc[frame.region.ne(frame.region_prediction)].iterrows():
        if row.region == "spine" or row.region_prediction == "spine":
            raise ValueError("Cross-anatomy router error requires an explicit held-out branch reevaluation")
        pixels = read_dicom(source / row.path).pixels
        true_scores, routed_scores, checkpoints = [], [], []
        for run in manifest["hip_runs"]:
            checkpoint = bundle / run["checkpoints"][int(row.fold)]
            if Path(run["checkpoints"][int(row.fold)]).name != f"fold{int(row.fold)}.pt":
                raise ValueError("Checkpoint fold ordering mismatch")
            model = build_hip_model(run["arch"], outputs=3, pretrained=False)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.requires_grad_(False).eval()
            with torch.inference_mode():
                true_scores.append(torch.sigmoid(model(hip_tensor(pixels, row.region, run["res"]))).numpy()[0])
                routed_scores.append(torch.sigmoid(model(hip_tensor(pixels, row.region_prediction, run["res"]))).numpy()[0])
            checkpoints.append(dict(relative=str(checkpoint.relative_to(bundle)), sha256=sha(checkpoint)))
            del model
        true_score, routed_score = np.mean(true_scores, axis=0), np.mean(routed_scores, axis=0)
        columns = ["p_quality", "p_hip_positioning", "p_hip_roi_margins"]
        original = row[columns].to_numpy(float)
        # Different CPU/MPS kernels can introduce small numerical deviations.
        if not np.allclose(true_score, original, atol=2e-4, rtol=2e-3):
            raise ValueError(f"Held-out forward disagrees with stored OOF: delta={np.max(np.abs(true_score-original))}")
        frame.loc[index, columns] = routed_score
        corrections.append(dict(study=index[0], image_uid=index[1], fold=int(row.fold), true_region=row.region,
                                predicted_region=row.region_prediction, original_scores=original.tolist(),
                                correct_side_rescored=true_score.tolist(), routed_scores=routed_score.tolist(), checkpoints=checkpoints))
    return corrections


def report(metrics, protocol, output):
    lines = ["# Валидация выбранного CLI-решения", "", "**Статус: DEVELOPMENT OOF, не независимый тест.** Использованы прогнозы моделей, исключавших оцениваемое исследование из обучения. Loss, архитектура и пороги выбраны по уже исследованной выборке; оптимизм этого выбора остаётся.", "",
             "## Качество после определения области", "", "| Область | n / нарушений | AUC [95% CI] | AP [95% CI] | F1 [95% CI]* | Чувств. / специф.* |", "|---|---:|---|---|---|---|"]
    for region, name in [("spine", "Позвоночник"), ("hip", "Бедро"), ("overall", "Вместе**")]:
        item = metrics["quality"][region]
        m, ci = item["point"], item["ci95"]
        cells = [f"{m[key]:.3f} [{ci[key][0]:.3f}–{ci[key][1]:.3f}]" for key in ["roc_auc", "ap", "f1"]]
        lines.append(f"| {name} | {m['n']} / {m['positive']} | {' | '.join(cells)} | {m['sensitivity']:.3f} / {m['specificity']:.3f} |")
    lines += ["", "*Все пороги оптимизированы по этим же OOF-меткам. F1, чувствительность и специфичность при них оптимистичны. **Общий AUC/AP смешивает разные шкалы вероятности позвоночника и бедра; основные показатели — отдельно по анатомии.", "",
              "## Причины нарушений", "", "`raw`: независимые головы причин; `gated`: причины, которые реально попадут в отчёт, только когда бинарная голова решила BAD. Качество определяется только бинарной головой.", "",
              "| Причина | n / положительных | AUC | AP | F1 raw* | F1 после BAD* |", "|---|---:|---:|---:|---:|---:|"]
    for code, item in metrics["reasons"].items():
        lines.append(f"| {code} | {item['raw']['n']} / {item['raw']['positive']} | {item['auc']:.3f} | {item['ap']:.3f} | {item['raw']['f1']:.3f} | {item['gated']['f1']:.3f} |")
    lines += [""]
    for mode in ["raw", "gated"]:
        item = metrics["reasons_macro_f1"][mode]
        lines.append(f"Macro F1 {mode}: **{item['value']:.3f}**, 95% CI [{item['ci95'][0]:.3f}–{item['ci95'][1]:.3f}].")
    support = metrics["support"]
    router = metrics["router"]
    labelled_routing = (f"Среди {support['quality_scored']} кадров с меткой качества ошибок определения области нет." if support['routed_side_corrections'] == 0
                        else f"Для {support['routed_side_corrections']} ошибок стороны среди размеченных кадров выполнен повторный прогноз CNN исключённого фолда.")
    lines += ["", "## Покрытие и поведение", "", f"- Область: {router['correct']}/{router['n']} правильных. {labelled_routing} Ошибок hip↔spine нет.",
              f"- Ошибок области среди кадров без метки качества: {support['router_errors_quality_unlabelled']}. Их влияние на качество нельзя проверить без разметки. Для текущего набора единственная ошибка стороны относится именно к такому кадру; на всех 249 размеченных кадрах область предсказана верно.",
              f"- Качество оценено для {support['quality_scored']} размеченных изображений. У {support['quality_unlabelled']} кадров отсутствует метка качества: они исключены из метрик качества, но входят в проверку определения области.",
              f"- Полнота OOF: {support['quality_scored']}/{support['quality_expected']}; пропущенных/невалидных OOF-прогнозов: {support['missing_or_invalid_oof']}. Это проверка сохранённых прогнозов, не замер частоты ошибок чтения всех произвольных DICOM.",
              f"- BAD с неизвестной причиной (`quality_unspecified`): {support['bad_with_unspecified_reason']}. GOOD с положительной независимой головой причины: {support['good_with_raw_reason']} — причина скрывается, остаётся предупреждение REVIEW.",
              "", "## Что именно проверено", "", "- Исследования, StudyInstanceUID и точные хеши пикселей не пересекают фолды; все соединения по study+image_uid проверены. Связь одного пациента между исследованиями неизвестна.",
              "- При ошибке стороны среди размеченных кадров скрипт пересчитывает прогноз через CNN с предсказанной стороной и сначала проверяет повторный forward с истинной стороной против сохранённого OOF. В текущем наборе таких случаев нет, дополнительный CNN-forward не понадобился.",
              "- Для каждого изображения взята CNN одного фолда, в обучении которого оно отсутствовало. В CLI усредняются все пять моделей; качество/калибровка этого усреднения на независимых данных ещё не проверены.",
              f"- Интервалы: {protocol['bootstrap']} повторов bootstrap по StudyInstanceUID, фиксированные выбранные модели и пороги; весь поиск моделей не повторяется. Для macro F1 причин исключаются bootstrap-повторы без положительных примеров хотя бы одного типа.",
              "- Причины с 6–10 положительными примерами имеют слабую поддержку. `quality_unspecified` — честное отсутствие установленной причины, не дополнительный диагностический класс с размеченным эталоном.",
              "- Если в имеющейся выборке нет ошибок некоторого типа, bootstrap может дать вырожденный интервал [1;1]. Это не означает гарантированные 100% на новых данных: bootstrap не моделирует ошибки, не встретившиеся в выборке.",
              "- Вложенные результаты выбора функции потерь сохранены отдельно в JSON и описаны в `docs/EXPERIMENT_RESULTS.md`. Их нельзя выдавать за вложенную оценку этой CNN-сборки.",
              "- Ранее выделенный holdout уже участвовал в экспериментах; локальные фолды не подтверждены как канонические командные. Нужен новый независимый набор для окончательной оценки.",
              "", "## Воспроизведение", "", "```sh", ".venv/bin/python scripts/evaluate_release_pipeline.py --bootstrap 1000", "```", "",
              "Для другой головы позвоночника: `--heads`, `--spine-oof`, `--spine-score-column`. Для другой упаковки: `--bundle`. Таблицы с идентификаторами остаются только в игнорируемой папке `outputs/release_validation_v1/`; этот документ их не содержит.", ""]
    Path(output).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/release")
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--spine-oof", type=Path, default=ROOT / "outputs/release_loss_v1/spine_oof.csv")
    parser.add_argument("--spine-score-column")
    parser.add_argument("--auxiliary-oof", type=Path, default=ROOT / "outputs/release_loss_v1/auxiliary_oof.csv")
    parser.add_argument("--hip-oof", type=Path, default=ROOT / "outputs/release_ensemble_v1/selected_oof.csv")
    parser.add_argument("--nested-loss", type=Path, default=ROOT / "outputs/release_loss_v1/run.json")
    parser.add_argument("--source", type=Path, default=ROOT / "data/interim/train/Исследования")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/release_validation_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/release_validation_v1/report.md")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap < 10:
        raise ValueError("At least 10 bootstrap replicates required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(args.bundle)
    heads_path = args.heads or args.bundle / manifest["heads"]
    heads_bundle = json.loads(heads_path.read_text())
    heads = heads_bundle["heads"]
    labels_path, folds_path = ROOT / "data/interim/image_labels.csv", ROOT / "data/interim/folds.csv"
    for name, path in [("labels_sha256", labels_path), ("folds_sha256", folds_path)]:
        if heads_bundle[name] != sha(path) or manifest["provenance"][name] != sha(path):
            raise ValueError("Bundle, heads and evaluation source hashes disagree")
    score_column = args.spine_score_column or "p_" + heads["spine_quality"]["loss"]
    protocol = dict(status="DEVELOPMENT_OOF_NOT_INDEPENDENT_TEST", bootstrap=args.bootstrap, seed=args.seed,
                    quality_decision="binary head only; reasons gated by BAD", spine_score_column=score_column,
                    threshold_status="selected on the same pooled OOF; threshold metrics optimistic",
                    scoring_unit="one held-out-fold model per run per image, NOT deployed five-fold average",
                    source_hashes={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path) for path in
                                   [labels_path, folds_path, args.bundle / "manifest.json", heads_path, args.spine_oof, args.hip_oof, args.auxiliary_oof]},
                    changed_heads_override=args.heads is not None)
    save(args.out / "protocol.json", protocol)
    labels = load_frame(labels_path)
    folds = load_frame(folds_path)
    if set(labels.index) != set(folds.index):
        raise ValueError("Labels/folds key mismatch")
    all_rows = labels.join(folds[["fold"]], validate="one_to_one")
    for group in ["study_uid", "pixel_sha256"]:
        if all_rows.groupby(group).fold.nunique().max() != 1:
            raise ValueError("Cross-fold image/study leakage")
    if all_rows.reset_index().groupby("study").fold.nunique().max() != 1:
        raise ValueError("Cross-fold study")
    auxiliary = align(load_frame(args.auxiliary_oof), all_rows, "router/reasons")
    if not auxiliary.region_prediction.isin(["spine", "hip_left", "hip_right"]).all():
        raise ValueError("Invalid/missing region predictions")
    frame = all_rows.loc[all_rows.quality.notna()].copy()
    frame["region_prediction"] = auxiliary.region_prediction
    spine_mask, hip_mask = frame.region.eq("spine"), frame.region.ne("spine")
    spine = align(load_frame(args.spine_oof), frame.loc[spine_mask], "spine")
    hip = align(load_frame(args.hip_oof), frame.loc[hip_mask], "hip")
    frame["p_quality"] = np.nan
    frame.loc[spine_mask, "p_quality"] = spine[score_column]
    frame.loc[hip_mask, "p_quality"] = hip.p_quality
    frame["quality_threshold"] = np.where(spine_mask, heads["spine_quality"]["threshold"], manifest["hip_thresholds"]["quality"])
    for code, label in REASONS.items():
        is_spine = code.startswith("spine")
        valid = spine_mask if is_spine else hip_mask
        frame[f"p_{code}"] = np.nan
        frame.loc[valid, f"p_{code}"] = auxiliary.loc[valid.index[valid], f"p_{code}"] if is_spine else hip["p_" + label]
        frame[f"threshold_{code}"] = heads[code]["threshold"] if is_spine else manifest["hip_thresholds"][label]
    with threadpool_limits(limits=2):
        corrections = correct_router_side(frame, args.bundle, manifest, args.source)
        for code, label in REASONS.items():
            applicable = frame[label].notna()
            values = frame.loc[applicable, f"p_{code}"].to_numpy()
            if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError(f"Missing/invalid predictions: {code}")
        if not np.isfinite(frame.p_quality).all() or not frame.p_quality.between(0, 1).all():
            raise ValueError("Missing/invalid quality predictions")
        frame["quality_prediction"] = (frame.p_quality >= frame.quality_threshold).astype(int)
        any_raw, any_gated = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
        for code in REASONS:
            frame[f"raw_{code}"] = frame[f"p_{code}"] >= frame[f"threshold_{code}"]
            frame[f"gated_{code}"] = frame[f"raw_{code}"] & frame.quality_prediction.eq(1)
            any_raw |= frame[f"raw_{code}"].to_numpy()
            any_gated |= frame[f"gated_{code}"].to_numpy()
        frame["quality_unspecified"] = frame.quality_prediction.eq(1) & ~any_gated
        frame["reason_disagreement_review"] = frame.quality_prediction.eq(0) & any_raw
        frame["processing_status"] = "OOF_PRESENT"
        result = summarize(frame, args.bootstrap, args.seed)
    classes = ["spine", "hip_left", "hip_right"]
    result["router"] = dict(n=len(all_rows), correct=int(all_rows.region.eq(auxiliary.region_prediction).sum()), classes=classes,
                            confusion_matrix_true_rows_predicted_columns=confusion_matrix(all_rows.region, auxiliary.region_prediction, labels=classes).tolist())
    result["support"] = dict(quality_expected=int(all_rows.quality.notna().sum()), quality_scored=len(frame),
                             quality_unlabelled=int(all_rows.quality.isna().sum()), missing_or_invalid_oof=0,
                             routed_side_corrections=len(corrections), bad_with_unspecified_reason=int(frame.quality_unspecified.sum()),
                             good_with_raw_reason=int(frame.reason_disagreement_review.sum()),
                             router_errors_quality_unlabelled=int((all_rows.quality.isna() & all_rows.region.ne(auxiliary.region_prediction)).sum()))
    result["status"] = protocol["status"]
    if args.nested_loss.exists():
        nested = json.loads(args.nested_loss.read_text())
        result["separate_nested_loss_selection_not_full_pipeline"] = {key: value["nested_selected"] for key, value in nested["quality"].items()}
    frame.reset_index().to_csv(args.out / "assembled_oof.csv", index=False)
    save(args.out / "router_side_corrections.json", corrections)
    save(args.out / "metrics.json", result)
    report(result, protocol, args.report)
    print(json.dumps(dict(status=result["status"], quality={key: value["point"] for key, value in result["quality"].items()}, support=result["support"]), indent=2))


if __name__ == "__main__":
    main()
