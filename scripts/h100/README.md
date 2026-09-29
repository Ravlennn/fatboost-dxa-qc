# Запуск на 2×H100

Всё, что можно было сделать без GPU, уже сделано и проверено короткими прогонами на Mac (M4 Pro, MPS). Здесь — только тяжёлые серии. Каждый скрипт можно перезапускать: готовые прогоны (`oof.csv`, чекпойнты) пропускаются. Логи — `outputs/h100_logs/`.

## 1. Что скопировать на сервер (не в git)

```bash
# с Mac, из корня проекта
rsync -av --relative \
  data/interim/image_labels.csv data/interim/folds.csv data/interim/dicom_index.csv \
  "data/interim/train/Исследования" \
  data/external/arak_dxa/source.zip data/external/pakistan_dxa_v1 \
  outputs/arak_overlays_v1 outputs/landmarks_v1 outputs/release_validation_v1 outputs/release_ensemble_v1 \
  outputs/landmark_fusion_v1 outputs/landmark_labeling \
  user@h100:/path/to/dxaqc/
```

`outputs/landmarks_v1` можно не копировать: `00_setup.sh` переобучит модель ориентиров (~5 мин на H100). Ручная разметка v2 необязательна: без неё используются автоматические точки малого вертела (`scripts/auto_label_lt.py`). Если разметка есть — положите `landmarks_manual_v2.json` в `outputs/landmark_labeling/`.

## 2. Запуск

```bash
git clone https://github.com/w4std/-_-_-.git dxaqc && cd dxaqc
# скопировать данные (п. 1), затем:
bash scripts/h100/run_all.sh          # всё сразу, ~10–14 ч
# или по частям:
bash scripts/h100/00_setup.sh         # uv sync --frozen (CUDA-колёса torch из uv.lock), ImageNet-веса, подготовка данных
bash scripts/h100/10_hip_ensemble.sh 0 1   # ConvNeXt-S (GPU0) и -B (GPU1), 512 px, 5 seeds × 5 folds
bash scripts/h100/20_ssl.sh 0              # DINO на 4.1k DXA → probe-gate → дообучение бедра с DXA-энкодером
bash scripts/h100/30_artifacts.sh 1        # сегментатор артефактов на синтетике (INIT=... — старт с DXA-энкодера)
bash scripts/h100/40_landmarks_v2.sh 0     # ручная разметка v2, если есть, иначе автоматический малый вертел v2
```

Переменные: `EPOCHS` (бедро/артефакты), `SSL_EPOCHS` (DINO, по умолчанию 200), `ARCH` (`convnext_small`|`convnext_base`).

## 3. Что вернуть

```bash
rsync -av user@h100:/path/to/dxaqc/outputs/{hip_cnn/h100_*,hip_ensemble_eval*,ssl_v1,ssl_probe,artifact_seg_h100,landmarks_v2,h100_logs} ./outputs/
```

Решение о замене модели в релизе принимается только по заранее заданному критерию: парный ΔAUC/ΔAP с 95% ДИ против релиза на тех же фолдах (таблицы `table.md`). Если ДИ включает 0 — выигрыша нет, меньшая модель остаётся.

## 4. Ожидаемое время (оценка)

| Серия | GPU | Время |
|---|---|---|
| Бедро ConvNeXt-S 512, 25 моделей | 1 | 2–3 ч |
| Бедро ConvNeXt-B 512, 25 моделей | 1 | 3–4 ч |
| DINO ConvNeXt-S, 200 эпох × 4.1k | 1 | 4–6 ч |
| Бедро с DXA-энкодером, 25 моделей | 1 | 2–3 ч |
| Артефакты, 60 × 4000 синтетических | 1 | 1–2 ч |
| Ориентиры v2, 6 моделей | 1 | < 1 ч |

## Замечания

- Python 3.12: DataLoader-воркеры используют `fork`, глобальные массивы изображений не копируются на диск. На 3.14+ (forkserver по умолчанию) ставьте `--workers 0`.
- Инференс финальной сборки остаётся CPU-совместимым; веса новых моделей добавляются в бандл через `scripts/export_*`, `dxaqc doctor` проверяет SHA-256.
