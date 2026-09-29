# Воспроизведение обучения и экспериментов

## Данные

Таблицы `data/interim/image_labels.csv`, `folds.csv` и `local_provenance.json` задают неизменяемые метки и пять локальных фолдов. DICOM ожидаются под `data/interim/train/Исследования`. Это 252 уникальных изображения, 249 с quality-меткой, включая 99 позвоночника и 150 бедра. Пропуски не превращать в нули. Три противоречия quality/reasons сохранены как исходная разметка.

Фолды локально восстановлены и не подтверждены как командные. Проверяются study, StudyInstanceUID, SOPInstanceUID и нормализованный pixel hash. Связь разных исследований одного пациента неизвестна. Ранее отложенные 49 validation и 49 test включены в последующие CV и больше не являются независимым тестом. Три файла организатора «Для теста» используются только для технического smoke без меток.

## Дешёвый loss-цикл

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_release_loss_suite.py --help
```

Frozen DenseNet121 ImageNet; checkpoint SHA-256 проверяется. Признаки извлекаются один раз; scaler, class weights и линейные головы fit только на train. Пять фиксированных вариантов: balanced logistic, BCE, balanced BCE, focal γ2, label smoothing .05. Внешние пять фолдов с внутренними тремя отделяют выбор loss/порога от оценки. Итоговые головы refit на всей development-выборке и экспортированы как JSON. Полный протокол и числа — `docs/EXPERIMENT_RESULTS.md`.

## CNN бедра

Рецепт существующего победителя: ConvNeXt-Tiny ImageNet, 384px, 30 эпох, AdamW lr1e-4 weight_decay.05, пять study-фолдов, masked weighted BCE для quality/rotation/ROI. Непрерывные scores сохранены в OOF, каждый файл оценивает модель, которая не видела его исследование.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_HOME=models/torch_home \
  .venv/bin/python scripts/train_hip_cnn.py --tag new_unique_run --arch convnext_tiny \
  --res 384 --epochs 30 --cpu-threads 2 --batch-pause 1
```

Команда запускает тяжёлое обучение; она не нужна для инференса готового комплекта. Всегда новый `--tag`; частичные запуски не выдавать за полный OOF. `local_convnext384_s0_lowload_v2` переносит два завершённых фолда из ранней серии, остальные перезапускались с seed+fold; побитовая воспроизводимость старой RNG-траектории не заявляется.

В старом рецепте были affine/crop/erasing аугментации. Для DXA они потенциально меняют метку качества. Не объявлять их оптимальными без нового контролируемого опыта. Исторический CNN обучался на native uint8 [0,252]; runtime восстанавливает этот диапазон после общей DICOM-нормализации. Проверять parity предобработки и scores перед выпуском.

## Выбор и выпуск

```sh
.venv/bin/python scripts/evaluate_release_ensembles.py --help
.venv/bin/python scripts/prepare_release.py --output models/release_candidate
.venv/bin/dxaqc doctor --model-dir models/release_candidate
```

Сравнение 3 одиночных, 3 равновесных пар и тройки фиксируется заранее. Выбор: mean fold AP; в пределах .01 AP предпочтение меньшему числу моделей. Никаких непроверенных незавершённых фолдов, silent fallback или скачиваний в runtime.

Пороги причины не должны логическим OR менять бинарный класс: тогда опубликованная метрика binary-головы перестаёт описывать итог. При BAD без установленной причины выдаётся `quality_unspecified`; при конфликте GOOD/причина — warning `review:`. Это правило отдельно проверяется в итоговой валидации.

Новый loss или backbone — гипотеза. Основной журнал `docs/EXPERIMENTS.md`: гипотеза, фиксированный протокол, SHA данных и кода, status, seed, runtime, OOF, метрики с group bootstrap, решение о включении. Повторное использование внешних фолдов для следующих решений остаётся разработкой; финальное подтверждение — только новые закрытые исследования.
