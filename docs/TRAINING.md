# Воспроизведение обучения и экспериментов

## Данные

Таблицы `data/interim/image_labels.csv`, `folds.csv` и `local_provenance.json` задают неизменяемые метки и пять локальных фолдов. DICOM ожидаются под `data/interim/train/Исследования`. Это 252 уникальных изображения, 249 с quality-меткой, включая 99 позвоночника и 150 бедра. Пропуски не превращать в нули. Три противоречия quality/reasons сохранены как исходная разметка.

Фолды локально восстановлены и не подтверждены как командные. Проверяются study, StudyInstanceUID, SOPInstanceUID и нормализованный pixel hash. Связь разных исследований одного пациента неизвестна. Ранее отложенные 49 validation и 49 test включены в последующие CV и больше не являются независимым тестом. Три файла организатора «Для теста» используются только для технического smoke без меток.

## Дешёвый loss-цикл

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_release_loss_suite.py --help
```

Frozen DenseNet121 ImageNet; checkpoint SHA-256 проверяется. Признаки извлекаются один раз; scaler, class weights и линейные головы fit только на train. Пять фиксированных вариантов: balanced logistic, BCE, balanced BCE, focal γ2, label smoothing .05. Внешние пять фолдов с внутренними тремя отделяют выбор loss/порога от оценки. Итоговые головы refit на всей development-выборке и экспортированы как JSON. Итог — в README, раздел «Эксперименты».

## CNN бедра

Рецепт существующего победителя: ConvNeXt-Tiny ImageNet, 384px, 30 эпох, AdamW lr1e-4 weight_decay.05, пять study-фолдов, masked weighted BCE для quality/rotation/ROI. Непрерывные scores сохранены в OOF, каждый файл оценивает модель, которая не видела его исследование.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_HOME=models/torch_home \
  .venv/bin/python scripts/train_hip_cnn.py --tag new_unique_run --arch convnext_tiny \
  --res 384 --epochs 30 --cpu-threads 2 --batch-pause 1
```

Команда запускает тяжёлое обучение; она не нужна для инференса готового комплекта. Всегда новый `--tag`; частичные запуски не выдавать за полный OOF. `local_convnext384_s0_lowload_v2` переносит два завершённых фолда из ранней серии, остальные перезапускались с seed+fold; побитовая воспроизводимость старой RNG-траектории не заявляется.

В старом рецепте были affine/crop/erasing аугментации. Для DXA они потенциально меняют метку качества. Не объявлять их оптимальными без нового контролируемого опыта. Исторический CNN обучался на native uint8 [0,252]; runtime восстанавливает этот диапазон после общей DICOM-нормализации. Проверять parity предобработки и scores перед выпуском.

## Дообучение по исправлениям специалиста

Кнопка «Исправления» в интерфейсе (`GET /api/jobs/<id>/corrections.json`) выгружает `{"batch": …, "reviews": [...]}`, по элементу на каждый проверенный снимок:

```json
{"path": "…/CR000000.dcm", "study_uid": "…", "image_uid": "…", "region": "hip_right", "state": "confirmed", "at": "2026-09-21T22:01:22",
 "points": {"roi_top": [x, y], "neck_head_side": [x, y], …},
 "corrected_points": true, "overrides": {"hip_positioning": true},
 "model": {"quality": 1, "violations": ["hip_roi_margins"], "points": {…}},
 "final": {"quality": 1, "violations": ["hip_positioning"]}}
```

Координаты — в пикселях исходного кадра. `final` — метки качества и причин, `points` — ориентиры для обучения `scripts/train_landmarks.py`; `model` сохраняет исходный ответ для анализа ошибок.

Порядок дообучения:

1. Добавить `final` в таблицу меток как новые строки (по `image_uid`), не перезаписывая исходную экспертную разметку. Новые исследования распределить по фолдам по `study_uid`.
2. Точки специалиста — точные метки ориентиров; они дополняют слабые метки Arak.
3. Переобучить изменённый уровень (ориентиры или CNN бедра) с новым `--tag` и оценить протоколом полной изоляции (`scripts/run_hip_nested_local.sh`, `scripts/evaluate_final_isolated.py`).
4. Выпускать новый комплект только если 95% ДИ прироста выше нуля на тех же фолдах; затем `prepare_release.py` и `dxaqc doctor`.

Импорт выгрузки в таблицы делается вручную: исправлений пока нет, и формат таблиц меток фиксируется только после первых реальных данных.

## Выбор и выпуск

```sh
.venv/bin/python scripts/evaluate_release_ensembles.py --help
.venv/bin/python scripts/prepare_release.py --output models/release_candidate
.venv/bin/dxaqc doctor --model-dir models/release_candidate
```

Сравнение 3 одиночных, 3 равновесных пар и тройки фиксируется заранее. Выбор: mean fold AP; в пределах .01 AP предпочтение меньшему числу моделей. Никаких непроверенных незавершённых фолдов, silent fallback или скачиваний в runtime.

Пороги причины не должны логическим OR менять бинарный класс: тогда опубликованная метрика binary-головы перестаёт описывать итог. При BAD без причины выше порога выдаётся причина с наибольшим отношением скора к порогу и warning `review: … reported the closest one`; при конфликте GOOD/причина — warning `review:`. Это правило отдельно проверяется в итоговой валидации.

Новый loss или backbone — гипотеза. Сводка экспериментов — README, раздел «Эксперименты»; для каждого опыта фиксировались гипотеза, фиксированный протокол, SHA данных и кода, status, seed, runtime, OOF, метрики с group bootstrap, решение о включении. Повторное использование внешних фолдов для следующих решений остаётся разработкой; финальное подтверждение — только новые закрытые исследования.
