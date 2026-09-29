# Комплект для обучения на 2×H100 (DXA QC)

Всё, что нужно, чтобы запустить тяжёлые эксперименты на сервере **без участия разработчика**: шаги, скрипты, список данных, критерии и что прислать обратно. Скрипты обучения лежат в `scripts/` и `scripts/h100/`, здесь — обвязка и инструкция.

Репозиторий: `https://github.com/w4std/-_-_-` (приватный), актуальная ветка — `main`.

---

## 0. Что и зачем обучаем

Текущая сборка работает на CPU и уже сдаётся. H100 нужны, чтобы проверить четыре улучшения. Каждое сравнивается с текущей сборкой на **тех же 5 фолдах**, парным bootstrap по исследованиям. В релиз попадает только то, у чего 95% ДИ прироста выше нуля.

| # | Серия | Скрипт | Что должно улучшиться | Сейчас (см. `docs/FINAL_VALIDATION.md`) | Время |
|---|---|---|---|---|---|
| 1 | Ансамбль бедра: ConvNeXt-S и ConvNeXt-B, 512 px, 5 seeds × 5 фолдов | `scripts/h100/10_hip_ensemble.sh` | стабильный порог ротации и качества бедра | ротация AUC 0.829 / F1 0.575, качество бедра AUC 0.851 / F1 0.634 | 3–4 ч |
| 2 | Самообучение DINO на ~4.1 тыс. DXA → дообучение бедра с DXA-энкодером | `scripts/h100/20_ssl.sh` | качество бедра, признаки для артефактов | те же | 6–9 ч |
| 3 | Сегментатор артефактов на синтетике (два варианта фонов) | `scripts/h100/30_artifacts.sh` | маска артефакта; причина `spine_artifact` | F1 0.774 (детектор без обучения + CNN) | 1–2 ч |
| 4 | Ориентиры v2: малый вертел, Th12, гребни | `scripts/h100/40_landmarks_v2.sh` | ротация через точки вертела | см. `docs/METHODS_AND_MODELS.md` | < 1 ч |

`scripts/h100/run_all.sh` запускает всё. GPU0: 2 → 4, GPU1: 1 → 3. Итого **≈10–14 ч**. Любой скрипт можно перезапускать: завершённые прогоны (`oof.csv`, чекпойнты) пропускаются.

---

## 1. Требования к машине

- Linux x86_64, 2× NVIDIA H100 80 ГБ, драйвер с поддержкой CUDA 12.x (`nvidia-smi` работает).
- Свободно на диске: **≥ 60 ГБ** (чекпойнты ансамблей ~25 ГБ, DINO ~5 ГБ).
- Интернет **на время `00_setup.sh`**: скачать Python-пакеты (`uv sync`) и ImageNet-веса torchvision. Дальше обучение идёт офлайн.
- `git`, `curl`, `tmux` или `screen` (чтобы обучение не прервалось при отключении SSH).
- Python ставить не нужно: `uv` сам поставит 3.12 из `uv.lock`. Установка uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

---

## 2. Шаги

### Вариант без git (рекомендуется, если на сервере нет доступа к репозиторию)

На Mac: `bash h100_kit/build_portable_kit.sh`. Получится `~/Downloads/dxaqc_h100_kit/`: код, данные, веса и `START_HERE.md`. Папку целиком переносим на сервер и дальше идём по `START_HERE.md`. Шаги 2.1–2.2 ниже тогда не нужны.

### 2.1 На Mac (делаем мы, один раз)

```bash
cd "<repo>"                         # актуальная ветка main
bash h100_kit/make_data_archive.sh  # -> h100_kit/dist/dxaqc_h100_data.tar (~400 МБ) + .sha256
```

Передать `dxaqc_h100_data.tar` на сервер: `scp`, `rsync` или диск. **Медицинские данные не кладём в git и во внешние облака.** Список содержимого — `h100_kit/data_files.txt`.

### 2.2 На сервере

```bash
git clone https://github.com/w4std/-_-_-.git dxaqc      # нужен доступ к приватному репо
cd dxaqc
tar -xf /path/to/dxaqc_h100_data.tar                     # распаковать В КОРЕНЬ репозитория
bash h100_kit/verify_data.sh                             # SHA-256 всех файлов, 499 DICOM
tmux new -s dxa                                          # чтобы пережить разрыв SSH
bash scripts/h100/run_all.sh 2>&1 | tee outputs/run_all.log
```

Отсоединиться от tmux — `Ctrl-b d`, вернуться — `tmux attach -t dxa`.

**По частям** (например, если одна карта занята):

```bash
bash scripts/h100/00_setup.sh              # окружение, веса, подготовка данных (~15 мин)
bash scripts/h100/10_hip_ensemble.sh 0 1   # ConvNeXt-S на GPU0, ConvNeXt-B на GPU1
bash scripts/h100/20_ssl.sh 0              # DINO → проверка энкодера → бедро с DXA-энкодером
bash scripts/h100/30_artifacts.sh 1        # внешние фоны
BG_OURS=1 OUT=outputs/artifact_seg_h100_bgours bash scripts/h100/30_artifacts.sh 1
bash scripts/h100/40_landmarks_v2.sh 0     # ручная разметка v2, если есть, иначе автоматические точки
```

Переменные: `EPOCHS` (бедро/артефакты, по умолчанию 40/60), `SSL_EPOCHS` (200), `ARCH` (`convnext_small` | `convnext_base`).

### 2.3 Как следить

- `nvidia-smi` — загрузка обеих карт.
- `tail -f outputs/h100_logs/gpu0.log outputs/h100_logs/gpu1.log` — общий ход.
- `ls outputs/h100_logs/` — отдельный лог на каждый прогон; строки `FAILED` в `gpu*.log` показывают, что упало.
- Бедро: в логе прогона строки `fold k ep N ... val {...}` каждые 5 эпох. DINO: `{"epoch": N, "loss": ...}`, loss должен плавно падать от ~9.

### 2.4 Вернуть результаты

```bash
bash h100_kit/collect_results.sh            # таблицы, OOF, логи, финальный DINO-энкодер (без чекпойнтов)
bash h100_kit/collect_results.sh --weights  # + все *.pt, если решим ставить модель в релиз
```

Прислать получившийся `h100_results_<дата>.tar.gz`. На Mac распаковать в корне репозитория.

---

### 2.5 Замеры без обучения

`bash h100_kit/measure.sh` воспроизводит главную таблицу метрик, проверки переобучения и прогоняет CLI по всем 499 DICOM на GPU и CPU: время на исследование и доля Success. Результаты — в `outputs/measure_<дата>/`.

## 3. Где результаты и как решать

| Серия | Главная таблица | Решение «берём в релиз», если |
|---|---|---|
| Ансамбль бедра | `outputs/hip_ensemble_eval/table.md` | ΔAUC или ΔAP против `release` с ДИ > 0 для `quality` / `v_rotation` |
| DINO | `outputs/ssl_probe/table.md` — быстрая проверка энкодера; `outputs/hip_ensemble_eval_ssl/table.md` — дообучение | энкодер лучше ImageNet на проверке **и** ΔAUC дообучения > 0 с ДИ > 0 |
| Артефакты | `outputs/artifact_seg_h100*/metrics.json` | после возврата: fusion с текущим детектором через `scripts/eval_artifacts_lm.py` — ΔAUC > 0 с ДИ > 0 |
| Ориентиры v2 | `outputs/landmarks_v2/eval/table.md` | `hip_positioning_lt` fusion лучше CNN с ДИ > 0 |

Итоговую таблицу сборки после любых замен пересчитывает `scripts/evaluate_final.py` на Mac. Цифры сравнимы только с ней.

---

## 4. Если что-то пошло не так

| Симптом | Что делать |
|---|---|
| `CUDA not available` в `00_setup.sh` | проверить `nvidia-smi`; `uv sync --frozen` ставит колёса torch с CUDA 12 — нужен драйвер ≥ 525 |
| `MISSING data/...` | архив распакован не в корень репозитория или не полностью; повторить `tar -xf` и `verify_data.sh` |
| `CUDA out of memory` | уменьшить батч: в `10_hip_ensemble.sh` поменять `16`/`12` на `8`; для DINO `--batch 64` в `20_ssl.sh` |
| Нет интернета на сервере | на машине с интернетом: `uv sync --frozen` и `00_setup.sh` (кэш весов в `models/torch_home`), затем скопировать `.venv` и `models/torch_home` |
| Python ≥ 3.14 и зависание DataLoader | в скриптах `--workers 0` (форк-воркеры рассчитаны на 3.12, который ставит uv) |
| Прогон упал посередине | просто запустить тот же скрипт ещё раз — готовое пропускается |
| `Run already exists` в бедре | прогон с таким тегом уже начат; удалить папку `outputs/hip_cnn/<tag>` незавершённого прогона и перезапустить |

---

## 5. Что нужно от нас

См. `h100_kit/CHECKLIST.md`.
