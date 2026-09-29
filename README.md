# DXA QC — локальный контроль качества DICOM

Исследовательский CLI и локальный HTTP API для проверки качества DXA-снимков
поясничного отдела позвоночника и проксимального отдела бедра. Система принимает
один DICOM, каталог или ZIP и формирует CSV/XLSX с одной строкой на входной
файл. Инференс выполняется локально, без отправки изображений во внешние
сервисы.

> **Статус:** development prototype для хакатона. Метрики получены на OOF
> development-выборке; независимого закрытого и межцентрового теста нет. Проект
> не является медицинским изделием.

> 🔴 **Документация:**
> [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) — единое
> описание архитектуры, технологического стека, установки, развёртывания,
> HTTP API, воспроизводимости и ограничений проекта.

## Возможности

- вход: DICOM, рекурсивный каталог или ZIP;
- области: `lumbar_spine`, `hip_right`, `hip_left`;
- бинарный класс: `0` — нарушение не обнаружено, `1` — обнаружено;
- причины: `spine_scan_range`, `spine_axis_tilt`, `spine_artifact`,
  `hip_positioning`, `hip_roi_margins`;
- безопасная изоляция повреждённых файлов: ошибка одного DICOM не останавливает
  пакет;
- CSV/XLSX, JSON-summary, лог и опциональные PNG/DICOM overlays;
- локальный batch API (`GET /health`, `POST /v1/batch`);
- проверяемый model bundle с SHA-256 и полностью офлайн-инференс;
- CPU по умолчанию, опционально CUDA/MPS.

## Результаты development-валидации

Канонический протокол — пять внешних folds по исследованиям, внутренний выбор
порогов и 95% bootstrap CI по исследованиям.

| Срез | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Качество, все области | **0.667** | **0.833** | 0.661 |
| Качество, позвоночник | 0.708 | 0.814 | 0.723 |
| Качество, бедро | 0.634 | 0.851 | 0.665 |

Macro-F1 пяти причин — **0.552**. Полная таблица, интервалы, support и проверка
на переобучение: [`docs/FINAL_VALIDATION.md`](docs/FINAL_VALIDATION.md).

Эти значения нельзя трактовать как результат независимого клинического теста.
Редкие причины содержат всего 6–17 положительных кадров.

## Что нужно для воспроизведения

В Git намеренно не хранятся медицинские данные, release-веса и тяжёлые
результаты обучения. Поэтому различаются четыре сценария:

| Задача | Достаточно чистого Git-клона | Что требуется дополнительно |
|---|---|---|
| Установить проект и запустить source-only тесты | Да | — |
| Запустить реальный инференс | Нет | `models/release/` |
| Повторить финальные метрики | Нет | приватные DICOM, labels/folds, OOF и измерения |
| Повторить обучение с нуля | Нет | приватные и внешние данные, официальные checkpoints |

Release bundle должен поставляться отдельно и содержать
`models/release/manifest.json` со всеми перечисленными в нём файлами и SHA-256.
Чистый клон без bundle не должен молча переключаться на фиктивную модель.

## 1. Получение проекта

```sh
git clone https://github.com/Ravlennn/fatboost-dxa-qc.git
cd fatboost-dxa-qc
```

Требования:

- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- 4 GiB RAM для обычного CPU-инференса;
- Docker — только для контейнерного сценария.

## 2. Окружение

Каноническое исследовательское окружение закреплено в `uv.lock`:

```sh
uv sync --frozen
```

Команда устанавливает editable-пакет и создаёт `.venv`. На Linux полный lock
может включать крупные CUDA-компоненты PyTorch даже при последующем CPU-запуске.
Для минимального CPU runtime рекомендуется готовый Docker-образ или сборка по
`requirements-runtime.txt`, описанная в
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

Все дальнейшие команды используют `uv run`, поэтому одинаково работают в
PowerShell, Git Bash, Linux и macOS.

## 3. Проверка исходного кода

```sh
uv run python -m unittest discover -s tests -v
```

В чистом клоне тесты, которым нужен `models/release/` или официальные
ImageNet-веса, будут помечены `skipped`. Остальные тесты используют
синтетические DICOM и проверяют ZIP, UID, MONOCHROME1, отчёты, геометрию,
изоляцию ошибок и безопасность путей.

Статическая проверка Python и shell-скриптов:

```sh
uv run python -m compileall -q src scripts tests
sh -n scripts/build.sh scripts/run.sh scripts/run-copy.sh
```

## 4. Установка model bundle

Скопируйте отдельно полученный комплект моделей в:

```text
models/release/
  manifest.json
  encoder/**
  hip/**
  landmarks/**
  heads.json
  view_gate.json
```

Фактический состав задаётся `manifest.json`; некоторые опциональные файлы могут
отличаться. Проверка целостности:

```sh
uv run dxaqc doctor --model-dir models/release
```

Ожидается JSON со `status: "ok"`, `model_id` и количеством проверенных файлов.
Если bundle отсутствует или SHA-256 не совпадает, запуск завершается ошибкой.

## 5. Запуск CLI

Один DICOM:

```sh
uv run dxaqc -i /path/image.dcm -o outputs/result.csv \
  --model-dir models/release
```

Каталог или ZIP с подробностями и scores:

```sh
uv run dxaqc -i /path/studies.zip -o outputs/result.xlsx \
  --model-dir models/release --details --scores --fail-on-error
```

Overlays с ориентирами и измерениями:

```sh
uv run dxaqc -i /path/studies -o outputs/result.csv \
  --model-dir models/release --overlays outputs/overlays.zip
```

Рядом с отчётом создаются:

```text
result.log
result.summary.json
```

Полезные параметры:

| Параметр | Назначение |
|---|---|
| `--device cpu|cuda|mps|auto` | Устройство, по умолчанию CPU |
| `--threads 1..8` | Число CPU threads, по умолчанию 2 |
| `--details` | Добавить `error_message` |
| `--scores` | Добавить непрерывные оценки |
| `--strict-columns` | Оставить ровно 8 колонок ТЗ |
| `--fail-on-error` | Вернуть код 3 при наличии Failure |
| `--no-view-gate` | Отключить фильтр неподдержанных проекций |

Коды завершения:

- `0` — отчёт записан;
- `2` — ошибка конфигурации, аргументов или bundle;
- `3` — в отчёте есть `Failure` и указан `--fail-on-error`;
- `4` — вход не содержит файлов;
- `130` — прерывание пользователем.

## 6. Формат результата

Строгий отчёт содержит:

```text
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing
```

`quality_unspecified` означает, что бинарная модель обнаружила нарушение, но
ни одна конкретная причина не прошла свой порог. `Success` означает успешную
техническую обработку, а не высокую уверенность клинического заключения.

Одинаковые пиксели вычисляются один раз, но каждый входной файл получает
собственную строку. Повреждённый или неподдержанный DICOM получает `Failure`,
остальной пакет продолжает обрабатываться.

## 7. Docker

Docker image включает исходный код и `models/release/`, поэтому перед сборкой
bundle должен быть установлен:

```sh
sh scripts/build.sh
sh scripts/run.sh /absolute/studies.zip /absolute/results/result.csv
```

На Windows `.sh`-скрипты запускаются из Git Bash или WSL. Для Docker-сред без
работающих bind mounts:

```sh
sh scripts/run-copy.sh /absolute/studies.zip /absolute/results/new-result.xlsx
```

Контейнер инференса запускается с `--network none`, двумя CPU, лимитом памяти,
read-only filesystem в основном сценарии и временной директорией. Подробности:
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Передача готового Docker-образа

Если целевая Linux x86_64 машина не имеет доступа к интернету или исходному
репозиторию, образ можно собрать заранее и передать вместе с SHA-256:

```sh
DXAQC_IMAGE=dxaqc:release-amd64 sh scripts/build.sh --platform linux/amd64
docker save dxaqc:release-amd64 -o dxaqc-release-amd64.tar
sha256sum dxaqc-release-amd64.tar > dxaqc-release-amd64.tar.sha256
```

Пример инструкции для запуска переданного образа:

```bash
sha256sum dxaqc-release-amd64.tar
docker load -i dxaqc-release-amd64.tar
docker run --rm -p 127.0.0.1:8080:8080 \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=3g \
  --cpus=2 --memory=4g \
  dxaqc:release-amd64 serve --host 0.0.0.0 --port 8080
```

HTTP API публикуется только на `127.0.0.1:8080`, поэтому он недоступен извне
без явного изменения настройки порта.

## 8. HTTP API

```sh
uv run dxaqc serve --model-dir models/release \
  --host 127.0.0.1 --port 8080 --threads 2
```

Проверка готовности и пакетный запрос:

```sh
curl http://127.0.0.1:8080/health
curl -H "Content-Type: application/zip" \
  --data-binary @studies.zip http://127.0.0.1:8080/v1/batch
```

API локальный, без аутентификации и обрабатывает пакеты последовательно. Не
публикуйте его через `0.0.0.0` без reverse proxy, аутентификации и сетевых
ограничений.

## 9. Восстановление данных для обучения

Приватные данные не копируются в Git. Ожидаемая структура:

```text
data/interim/train/Исследования/**
data/interim/test/Для теста/**
data/interim/image_labels.csv
data/interim/folds.csv
data/interim/dicom_index.csv
data/interim/local_provenance.json
```

Для восстановления таблиц требуется отдельно переданный, уже проверенный
annotation bundle с файлами `labels_candidate.jsonl` и `quarantine.jsonl`:

```sh
uv run python scripts/prepare_local_team_data.py \
  --source /path/to/reviewed_annotation_bundle
```

Скрипт проверяет Study/SOP UID, pixel SHA-256, уникальность изображений и
группировку folds. Он отказывается перезаписывать существующие labels/folds.

Важно: репозиторий не содержит процедуру создания этого annotation bundle из
исходной Excel-разметки. Для действительно полного повтора bundle необходимо
архивировать отдельно вместе с контрольной суммой и условиями доступа.

## 10. Обучение и повтор метрик

Основные этапы:

1. подготовить приватные таблицы и DICOM;
2. разместить официальные ImageNet checkpoints в
   `models/torch_home/hub/checkpoints/`;
3. обучить полные пятифолдовые hip-модели через `train_hip_cnn.py`;
4. выбрать ансамбль через `evaluate_release_ensembles.py`;
5. построить frozen DenseNet головы через `run_release_loss_suite.py`;
6. собрать bundle через `prepare_release.py` и export-скрипты;
7. выполнить parity, unit-тесты и `dxaqc doctor`;
8. пересчитать итог через `evaluate_final.py` и `check_overfit.py`.

Подробный протокол: [`docs/TRAINING.md`](docs/TRAINING.md). Полная матрица
необходимых артефактов и ограничений:
[`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

Точный повтор опубликованных чисел дополнительно требует игнорируемых Git
артефактов из `outputs/`: OOF hip-моделей, release loss/ensemble, landmarks,
lesser-trochanter и artifact measurements. Если они не сохранены, обучение
нужно повторять, а побитовое совпадение на другом устройстве не гарантируется.

## 11. Структура проекта

```text
src/dxaqc/             runtime, CLI, API и модели
scripts/               подготовка, обучение, оценка и упаковка
scripts/h100/          тяжёлые GPU-серии
tests/                 unit и acceptance-тесты
docs/                  документация и отчёты экспериментов
h100_kit/              переносимый комплект для H100
models/                локальные веса; release bundle не хранится в Git
data/                  приватные/внешние данные; не хранится в Git
outputs/               результаты и OOF; в основном не хранится в Git
```

## 12. Документация

- [Техническая документация](docs/PROJECT_DOCUMENTATION.md)
- [Развёртывание и API](docs/DEPLOYMENT.md)
- [Обучение](docs/TRAINING.md)
- [Итоговая валидация](docs/FINAL_VALIDATION.md)
- [Методы и модели](docs/METHODS_AND_MODELS.md)
- [Сводка экспериментов](docs/EXPERIMENT_RESULTS.md)
- [Источники исследований и датасеты](docs/RESEARCH_SOURCES.md)
- [Соответствие ТЗ и ограничения](docs/RELEASE.md)
- [Метрики по исследованиям](docs/STUDY_LEVEL.md)
- [Устойчивость и domain shift](docs/ROBUSTNESS.md)
- [Сторонние компоненты и данные](docs/THIRD_PARTY.md)
- [Сценарий демонстрации](docs/DEMO.md)
- [Хронологический журнал экспериментов](docs/EXPERIMENTS.md)
- [Техническое задание](docs/ТЗ-ДепЗдрав.pdf)

## 13. Приватность и ограничения

- Не публикуйте DICOM, приватную разметку и OOF с идентификаторами.
- Не загружайте данные организаторов в Kaggle/Colab/облачные сервисы без
  письменного разрешения.
- Перед распространением производных весов проверьте лицензии внешних наборов,
  особенно Arak.
- В репозитории пока нет корневого `LICENSE`; до его добавления нельзя считать
  код и release bundle разрешёнными для свободного перераспространения.
- Не сравнивайте метрики из старых holdout/разбиений с финальным вложенным OOF
  как одну таблицу лидеров.
- Не используйте систему для клинических решений без независимой проверки,
  экспертного контроля и выполнения регуляторных требований.
