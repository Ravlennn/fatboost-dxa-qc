# DXA QC — автономный комплект для сервера с 2×H100

В этой папке есть всё: код (`dxaqc/`), данные, веса текущей модели, ImageNet-веса, скрипты обучения и замеров. Git и доступ к репозиторию не нужны. Подробная инструкция — `dxaqc/h100_kit/README.md`, чек-лист — `dxaqc/h100_kit/CHECKLIST.md`.

## 0. Перенести папку на сервер

Скопировать ВСЮ папку `dxaqc_h100_kit` (≈1.4 ГБ) по scp, rsync или на диске. Не через облачные хранилища: внутри медицинские снимки.

```bash
rsync -a --info=progress2 dxaqc_h100_kit user@server:~/
```

## 1. Проверить целостность и данные (1 мин)

```bash
cd ~/dxaqc_h100_kit
sha256sum -c --quiet KIT_SHA256.txt && echo "kit OK"      # все файлы комплекта
cd dxaqc && bash h100_kit/verify_data.sh                      # данные: 688 файлов, 499 DICOM
```

## 2. Окружение (~10–15 мин, нужен интернет)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env   # если uv ещё нет
bash scripts/h100/00_setup.sh      # Python 3.12 + пакеты из uv.lock (torch с CUDA), веса ConvNeXt-S/B, подготовка данных
```

Если интернета на сервере нет, напишите нам: соберём офлайн-вариант с колёсами пакетов (~4 ГБ).

## 3. Замеры текущей модели (~20–30 мин, без обучения)

```bash
bash h100_kit/measure.sh
```

Скрипт показывает и сохраняет в `outputs/measure_<дата>/`:
- исправность весов (SHA-256) и результаты тестов;
- **главную таблицу метрик** — должна совпасть с `docs/FINAL_VALIDATION.md` (качество: F1 0.667, AUC 0.833);
- **проверку переобучения:** AUC с перемешанными метками должен быть ≈ 0.5;
- **время и долю Success** на всех 499 DICOM, на GPU и на CPU.

## 4. Обучение (≈10–14 ч, в tmux)

```bash
tmux new -s dxa
bash scripts/h100/run_all.sh 2>&1 | tee outputs/run_all.log
# отсоединиться: Ctrl-b d; вернуться: tmux attach -t dxa
```

Ход: `tail -f outputs/h100_logs/gpu0.log outputs/h100_logs/gpu1.log`, `nvidia-smi`.

## 5. Вернуть результаты

```bash
bash h100_kit/collect_results.sh     # -> h100_results_<дата>.tar.gz (таблицы, OOF, логи, энкодер DINO)
tar -czf measure.tar.gz outputs/measure_*
```

Прислать оба архива. Что считается успехом каждой серии — `dxaqc/h100_kit/README.md`, раздел 3.
