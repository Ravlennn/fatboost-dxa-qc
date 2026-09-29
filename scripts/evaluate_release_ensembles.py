"""Audit complete hip OOF predictions and select a small, predefined ensemble.

No model training, checkpoint loading, weight search, or independent-test claim.
Run: .venv/bin/python scripts/evaluate_release_ensembles.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TAGS = (
    "local_convnext384_s0_lowload_v2",
    "local_densenet384_s0_lowload_v2",
    "cgmh_control_full_s0_v1",
)
TARGETS = ("quality", "v_rotation", "v_roi")
KEY = ["study", "image_uid"]
PROTOCOL = {
    "version": 1,
    "candidate_rule": "Three complete singles, all three equal-weight pairs, one equal-weight triple",
    "primary_metric": "unweighted mean of five held-out-fold quality average precision values",
    "selection_rule": "Within 0.01 of the maximum mean fold AP, prefer fewer runs; then higher mean fold AP",
    "complexity_tolerance_ap": 0.01,
    "threshold_rule": "Maximize pooled OOF F1 independently for each target; ties prefer the larger threshold",
    "threshold_status": "Deployment tuning on the same OOF labels; optimized F1 is optimistic",
    "bootstrap": "Paired StudyInstanceUID cluster bootstrap; fixed folds; percentile 95% intervals",
    "exclusions": ["partial CGMH ROI run", "old holdout comparisons with different populations"],
    "independent_test": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_source(root: Path):
    base = root / "data/interim"
    labels = pd.read_csv(base / "image_labels.csv", dtype={"study": str, "image_uid": str, "study_uid": str})
    folds = pd.read_csv(base / "folds.csv", dtype={"study": str, "image_uid": str})
    if labels.duplicated(KEY).any() or folds.duplicated(KEY).any():
        raise ValueError("Duplicate image key in labels/folds")
    merged = labels.merge(folds, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (merged._merge != "both").any():
        raise ValueError("Labels and folds contain different image keys")
    merged = merged.drop(columns="_merge")
    for group in ("study", "study_uid", "pixel_sha256"):
        if merged[group].isna().any() or merged.groupby(group).fold.nunique().max() != 1:
            raise ValueError(f"Missing or cross-fold group: {group}")
    if set(merged.fold) != set(range(5)):
        raise ValueError("Expected five folds")
    for target in (*TARGETS, "v_layout", "v_axis", "v_artifact"):
        if not merged[target].dropna().isin((0, 1)).all():
            raise ValueError(f"Nonbinary labels: {target}")
    # Folder study IDs precede DICOM anonymization; require a bijection, not text equality.
    if merged.groupby("study").study_uid.nunique().max() != 1 or merged.groupby("study_uid").study.nunique().max() != 1:
        raise ValueError("Training study identifier does not map one-to-one to StudyInstanceUID")
    hashes = {name: sha256(base / name) for name in ("image_labels.csv", "folds.csv")}
    local = json.loads((base / "local_provenance.json").read_text())
    if any(local["tables"][name] != digest for name, digest in hashes.items()):
        raise ValueError("Local provenance table hashes differ")
    hip = merged.loc[merged.region.isin(("hip_left", "hip_right")) & merged.quality.notna()].copy()
    hip = hip.sort_values(KEY).reset_index(drop=True)
    runs, metadata = {}, {}
    for tag in TAGS:
        directory = root / "outputs/hip_cnn" / tag
        provenance = json.loads((directory / "provenance.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        if provenance["tables"] != hashes or provenance["split_status"] != local["status"]:
            raise ValueError(f"Source provenance mismatch for {tag}")
        if config.get("crop", 0) or config.get("roi_file") or config["res"] != 384:
            raise ValueError(f"Unexpected preprocessing for {tag}")
        raw = pd.read_csv(directory / "oof.csv", dtype={"study": str, "image_uid": str})
        if raw.duplicated(KEY).any():
            raise ValueError(f"Duplicate OOF key in {tag}")
        joined = hip.merge(raw, on=KEY, how="outer", validate="one_to_one", suffixes=("", "_oof"), indicator=True)
        if (joined._merge != "both").any():
            raise ValueError(f"OOF does not cover exactly the labelled hips for {tag}")
        for name in ("fold", "region", *TARGETS):
            same = (joined[name] == joined[name + "_oof"]) | (joined[name].isna() & joined[name + "_oof"].isna())
            if not same.all():
                raise ValueError(f"OOF labels/folds differ: {tag}/{name}")
        probabilities = joined[["p_" + target for target in TARGETS]].to_numpy(float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Invalid/incomplete OOF probabilities: {tag}")
        history = json.loads((directory / "history.json").read_text())
        weights = {}
        for fold in range(5):
            checkpoint = directory / f"fold{fold}.pt"
            if not checkpoint.is_file() or history[str(fold)][-1]["epoch"] != config["epochs"]:
                raise ValueError(f"Incomplete training {tag}/fold{fold}")
            weights[f"fold{fold}.pt"] = sha256(checkpoint)
        runs[tag] = probabilities
        metadata[tag] = {
            "config": config, "provenance": provenance,
            "oof_sha256": sha256(directory / "oof.csv"), "weights_sha256": weights,
        }
    contradictions = {}
    for region, frame in merged.groupby(merged.region.eq("spine").map({True: "spine", False: "hip"})):
        reasons = ["v_layout", "v_axis", "v_artifact"] if region == "spine" else ["v_rotation", "v_roi"]
        any_bad = frame[reasons].eq(1).any(axis=1)
        all_good = frame[reasons].eq(0).all(axis=1)
        contradictions[region] = {
            "rows": len(frame), "quality_missing": int(frame.quality.isna().sum()),
            "good_quality_with_positive_reason": int((frame.quality.eq(0) & any_bad).sum()),
            "bad_quality_with_all_reasons_negative": int((frame.quality.eq(1) & all_good).sum()),
            "reason_missing": {name: int(frame[name].isna().sum()) for name in reasons},
        }
    overlap = {}
    for first, second in itertools.combinations(TAGS, 2):
        overlap[first + " + " + second] = {
            "identical_prediction_folds": [fold for fold in range(5) if np.array_equal(runs[first][hip.fold == fold], runs[second][hip.fold == fold])],
            "identical_checkpoint_folds": [fold for fold in range(5) if metadata[first]["weights_sha256"][f"fold{fold}.pt"] == metadata[second]["weights_sha256"][f"fold{fold}.pt"]],
        }
    audit = {
        "n_all_images": len(merged), "n_hips": len(hip), "n_hip_studies": int(hip.study_uid.nunique()),
        "table_hashes": hashes, "source_status": local["status"], "source_limitations": local["limitations"],
        "group_and_pixel_leakage_check": "PASS: no study or exact pixel hash crosses folds",
        "label_audit": contradictions, "run_overlap": overlap, "source_runs": metadata,
    }
    return hip, runs, audit


def metrics(y, p, folds, threshold=0.5):
    valid = np.isfinite(y)
    y, p, folds = np.asarray(y)[valid].astype(int), np.asarray(p)[valid], np.asarray(folds)[valid]
    positive = p >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    tn = int(np.sum(~positive & (y == 0)))
    fold_ap = [average_precision_score(y[folds == fold], p[folds == fold]) for fold in np.unique(folds) if len(np.unique(y[folds == fold])) == 2]
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "positive": int(y.sum()), "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "ap": float(average_precision_score(y, p)) if both else None,
        "mean_fold_ap": float(np.mean(fold_ap)) if len(fold_ap) == 5 else None, "n_evaluable_folds": len(fold_ap),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "sensitivity": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def tune_threshold(y, p):
    valid = np.isfinite(y)
    y, p = np.asarray(y)[valid].astype(int), np.asarray(p)[valid]
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))[-1]])


def evaluate(root: Path, output: Path, n_boot: int, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    # Persist the criterion before reading predictions or calculating results.
    save_json(output / "protocol.json", {**PROTOCOL, "bootstrap_replicates": n_boot, "bootstrap_seed": seed})
    hip, runs, audit = validate_source(root)
    save_json(output / "audit.json", audit)
    candidates = {}
    for size in range(1, 4):
        for tags in itertools.combinations(TAGS, size):
            candidates[" + ".join(tags)] = {"tags": list(tags), "p": np.mean([runs[tag] for tag in tags], axis=0)}
    y = hip.quality.to_numpy(float)
    folds = hip.fold.to_numpy(int)
    for item in candidates.values():
        item["point"] = metrics(y, item["p"][:, 0], folds)
    best_ap = max(item["point"]["mean_fold_ap"] for item in candidates.values())
    eligible = [name for name, item in candidates.items() if item["point"]["mean_fold_ap"] >= best_ap - PROTOCOL["complexity_tolerance_ap"]]
    selected = min(eligible, key=lambda name: (len(candidates[name]["tags"]), -candidates[name]["point"]["mean_fold_ap"], name))
    best_single = max(TAGS, key=lambda name: candidates[name]["point"]["mean_fold_ap"])
    keys = ("roc_auc", "ap", "mean_fold_ap", "f1", "sensitivity", "specificity")
    samples = {name: {key: [] for key in keys} for name in candidates}
    groups = hip.study_uid.to_numpy(str)
    unique = np.unique(groups)
    grouped_indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = np.concatenate([grouped_indices[index] for index in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) < 2:
            continue
        for name, item in candidates.items():
            result = metrics(y[idx], item["p"][idx, 0], folds[idx])
            for key in keys:
                samples[name][key].append(result[key])
    results = []
    for name, item in candidates.items():
        intervals, deltas = {}, {}
        for key in keys:
            values = np.asarray(samples[name][key], dtype=float)
            intervals[key] = list(map(float, np.percentile(values[np.isfinite(values)], [2.5, 97.5])))
            difference = values - np.asarray(samples[best_single][key], dtype=float)
            deltas[key] = {
                "value": item["point"][key] - candidates[best_single]["point"][key],
                "ci95": list(map(float, np.percentile(difference[np.isfinite(difference)], [2.5, 97.5]))),
                "valid_replicates": int(np.isfinite(difference).sum()),
            }
        results.append({"name": name, "tags": item["tags"], "quality_at_0_5": item["point"], "ci95": intervals,
                        "paired_difference_vs_best_single": deltas,
                        "fold_quality_ap": {str(fold): float(average_precision_score(y[folds == fold], item["p"][folds == fold, 0])) for fold in range(5)}})
    results.sort(key=lambda item: -item["quality_at_0_5"]["mean_fold_ap"])
    chosen = candidates[selected]
    target_results, thresholds = {}, {}
    for j, target in enumerate(TARGETS):
        target_y = hip[target].to_numpy(float)
        thresholds[target] = tune_threshold(target_y, chosen["p"][:, j])
        target_results[target] = {
            "at_0_5": metrics(target_y, chosen["p"][:, j], folds),
            "at_deployment_threshold_optimistic": metrics(target_y, chosen["p"][:, j], folds, thresholds[target]),
        }
    selection = {
        "schema_version": 1, "tags": chosen["tags"], "weights": [1 / len(chosen["tags"])] * len(chosen["tags"]),
        "folds": list(range(5)), "targets": list(TARGETS), "thresholds": thresholds,
        "quality_decision": "quality head only; reasons must not silently override the binary quality threshold",
        "threshold_status": PROTOCOL["threshold_status"], "selection_rule": PROTOCOL["selection_rule"],
        "best_single_reference": best_single, "selected_mean_fold_ap": chosen["point"]["mean_fold_ap"],
        "maximum_candidate_mean_fold_ap": best_ap, "metrics": target_results,
        "source_status": audit["source_status"], "table_hashes": audit["table_hashes"],
        "deployment_limitation": "OOF uses one held-out model per run; deployment averages all five folds. Threshold/calibration transfer is not independently tested.",
        "independent_test": False,
    }
    stable = {}
    for j, target in enumerate(TARGETS):
        values = chosen['p'][:, j]
        below = values[values < thresholds[target]]
        stable[target] = float((below.max() + thresholds[target]) / 2) if len(below) else thresholds[target]
        np.testing.assert_array_equal(values >= thresholds[target], values >= stable[target])
    selection['stable_thresholds'] = stable
    selection['stable_threshold_policy'] = 'Midpoint of adjacent OOF scores bounding the selected decision partition; same OOF classifications, larger CPU/MPS numerical margin'
    save_json(output / "selection.json", selection)
    save_json(output / "comparison.json", {"bootstrap_replicates": n_boot, "reference": best_single, "results": results})
    selected_oof = hip[[*KEY, "study_uid", "region", "fold", *TARGETS]].copy()
    for j, target in enumerate(TARGETS):
        selected_oof["p_" + target] = chosen["p"][:, j]
    selected_oof.to_csv(output / "selected_oof.csv", index=False)
    rows = [
        "# Выбор ансамбля для CLI", "",
        "Сравнение сохранённых полных OOF-прогнозов: 150 изображений бедра. До вычислений зафиксированы семь вариантов и критерий в `protocol.json`.", "",
        "**Критерий:** среднее AP по пяти фолдам; в пределах 0,01 от максимума выбираем вариант с меньшим числом запусков. Все смеси равновесные. AP — average precision, а не трапецеидальная PR-AUC.", "",
        "| Вариант | Mean fold AP | AUC [95% ДИ] | AP | F1 @ 0,5 |", "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        point = result["quality_at_0_5"]
        lo, hi = result["ci95"]["roc_auc"]
        rows.append(f"| {result['name']} | {point['mean_fold_ap']:.3f} | {point['roc_auc']:.3f} [{lo:.3f}; {hi:.3f}] | {point['ap']:.3f} | {point['f1']:.3f} |")
    top = results[0]
    delta_ap = top["paired_difference_vs_best_single"]["mean_fold_ap"]
    delta_auc = top["paired_difference_vs_best_single"]["roc_auc"]
    rows += ["", f"**Выбран:** `{selected}`. {len(chosen['tags']) * 5} весов для усреднения фолдов при инференсе.", "",
             f"Лучший по точечной оценке ансамбль даёт относительно лучшей одиночной модели Δ mean fold AP = {delta_ap['value']:+.3f} [95% ДИ {delta_ap['ci95'][0]:+.3f}; {delta_ap['ci95'][1]:+.3f}], Δ AUC = {delta_auc['value']:+.3f} [{delta_auc['ci95'][0]:+.3f}; {delta_auc['ci95'][1]:+.3f}]. Интервалы включают ноль; подтверждённого прироста нет.", "",
             "| Голова | n / положительных | AUC | AP | Порог для развёртывания | F1 при этом пороге* | Чувств. / специф.* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for target, values in target_results.items():
        point = values["at_0_5"]
        tuned = values["at_deployment_threshold_optimistic"]
        rows.append(f"| {target} | {point['n']} / {point['positive']} | {point['roc_auc']:.3f} | {point['ap']:.3f} | {thresholds[target]:.6f} | {tuned['f1']:.3f} | {tuned['sensitivity']:.3f} / {tuned['specificity']:.3f} |")
    rows += ["", "*Пороги оптимизированы на тех же OOF-метках. Эти F1/чувствительность/специфичность оптимистичны и не являются оценкой на независимом тесте.", "",
             "## Ограничения и аудит", "",
             f"- Разбиение `{audit['source_status']}`: {audit['n_hips']} бёдер в {audit['n_hip_studies']} исследованиях. Состав фолдов с командой не подтверждён.",
             "- Соединение выполнено по `(study, image_uid)`, с проверкой меток, фолдов, SHA-256 исходных таблиц, полного покрытия OOF, завершённой последней эпохи и наличия всех пяти весов.",
             "- Исследования и точные хеши пикселей не пересекают фолды. Связать разные исследования одного пациента по доступным таблицам невозможно.",
             f"- Парный кластерный bootstrap по StudyInstanceUID, {n_boot} повторов; интервалы условны на имеющихся обученных моделях и не учитывают весь поиск архитектур/гиперпараметров.",
             "- Для интервала mean fold AP оставлены только повторы, где в каждом из пяти фолдов есть оба класса. Число пригодных повторов сохранено в парных сравнениях.",
             "- Прежний holdout использован в текущей CV. Выбор из семи кандидатов добавляет оптимизм; независимый тест отсутствует.",
             "- OOF содержит прогноз одного исключавшего пример фолда, а при развёртывании усредняются пять моделей. Калибровка и пороги такого усреднения отдельно не проверены.",
             "- Неполный эксперимент CGMH ROI исключён. Сегментация на внешних изображениях не доказала пользу для наших снимков.",
             "- Итоговое качество определяется отдельной бинарной головой. Принудительное OR с причинами меняет классификатор и потребовало бы отдельной оценки."]
    for pair, overlap in audit["run_overlap"].items():
        if overlap["identical_prediction_folds"]:
            rows.append(f"- `{pair}`: идентичные OOF-прогнозы на фолдах {overlap['identical_prediction_folds']}; это не независимые повторения обучения.")
    rows += ["", "### Согласованность разметки", ""]
    for region, counts in audit["label_audit"].items():
        rows.append(f"- {region}: отсутствует quality у {counts['quality_missing']}; quality=0 с положительной причиной у {counts['good_quality_with_positive_reason']}; quality=1 со всеми причинами=0 у {counts['bad_quality_with_all_reasons_negative']}.")
    rows += ["", "Полные результаты, парные разности и контрольные суммы: `outputs/release_ensemble_v1/{comparison,audit,selection,protocol}.json`. Идентификаторы и прогнозы остаются в игнорируемом `outputs/`.", "",
             "Воспроизведение: `.venv/bin/python scripts/evaluate_release_ensembles.py --bootstrap 1000 --seed 1729`. CPU-пулы ограничены двумя потоками; обучение не выполняется.", ""]
    report = "\n".join(rows)
    (output / "report.md").write_text(report)
    print(json.dumps({"selected": selected, "thresholds": thresholds, "mean_fold_ap": selection["selected_mean_fold_ap"], "maximum_mean_fold_ap": best_ap, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/release_ensemble_v1")
    args = parser.parse_args()
    if args.bootstrap < 500:
        parser.error("Use at least 500 bootstrap replicates")
    with threadpool_limits(limits=2):
        evaluate(ROOT, args.output, args.bootstrap, args.seed)
