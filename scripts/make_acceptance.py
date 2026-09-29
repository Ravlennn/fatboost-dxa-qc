"""Генерирует приёмочный набор синтетических DICOM на базе трёх файлов «Для теста».
Запуск: .venv/bin/python scripts/make_acceptance.py [--src "data/interim/test/Для теста"] [--out tests/fixtures/acceptance]
Ожидаемые статусы пишутся в expected.csv (path_to_study, expected_status, note)."""
from __future__ import annotations
import argparse, copy, csv, io, shutil, zipfile
from pathlib import Path
import numpy as np, pydicom
from pydicom.uid import generate_uid

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data/interim/test/Для теста"); ap.add_argument("--out", default="tests/fixtures/acceptance")
a = ap.parse_args(); src = Path(a.src); out = Path(a.out)
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
files = sorted(src.glob("*.dcm"))
if len(files) < 3: raise SystemExit(f"нужны 3 файла в {src}")
spine = next(f for f in files if "ПОП" in f.name); hip = next(f for f in files if "ППОБ" in f.name); hip_l = next(f for f in files if "ЛПОБ" in f.name)
ds = pydicom.dcmread(spine); px = ds.pixel_array
rows = []
def save(name, d, status, note): d.save_as(out / name); rows.append((name, status, note))
save("01_spine.dcm", ds, "Success", "позвоночник как есть")
save("02_hip_right.dcm", pydicom.dcmread(hip), "Success", "правое бедро")
save("03_hip_left.dcm", pydicom.dcmread(hip_l), "Success", "левое бедро")
d = copy.deepcopy(ds); d.SOPInstanceUID = generate_uid(); save("04_exact_duplicate.dcm", d, "Success", "те же пиксели, другой SOP UID: своя строка, duplicates+1")
(out / "05_garbage.dcm").write_bytes(b"not a dicom at all"); rows.append(("05_garbage.dcm", "Failure", "не DICOM"))
(out / "06_empty.dcm").write_bytes(b""); rows.append(("06_empty.dcm", "Failure", "пустой файл"))
d = copy.deepcopy(ds); del d.StudyInstanceUID; del d.SOPInstanceUID; save("07_no_uid.dcm", d, "Success", "UID сгенерированы, предупреждение в error_message")
d = copy.deepcopy(ds); d.PixelData = np.zeros_like(px).tobytes(); save("08_constant.dcm", d, "Failure", "константный кадр")
d = copy.deepcopy(ds); d.NumberOfFrames = 2; d.PixelData = np.stack([px, np.zeros_like(px)]).tobytes(); d.SOPInstanceUID = generate_uid(); save("09_multiframe.dcm", d, "Success", "взят первый кадр, предупреждение")
d = copy.deepcopy(ds); d.PhotometricInterpretation = "MONOCHROME1"; d.PixelData = (255 - px).tobytes(); d.SOPInstanceUID = generate_uid(); save("10_mono1.dcm", d, "Success", "инверсия, результат как у 01")
d = copy.deepcopy(ds); del d.PixelData; save("11_no_pixeldata.dcm", d, "Failure", "нет PixelData")
nested = out / "12_nested" / "study_x" / "series_1"; nested.mkdir(parents=True); shutil.copy(hip, nested / "CR000000.dcm"); rows.append(("12_nested/study_x/series_1/CR000000.dcm", "Success", "вложенные папки"))
with zipfile.ZipFile(out / "13_archive.zip", "w") as z:
    z.write(spine, "inner/a/spine.dcm"); z.write(hip_l, "inner/b/hip.dcm")
rows.append(("13_archive.zip", "n/a", "zip с вложенными папками: поддержка архива пока не реализована, ожидается 2 строки после реализации"))
(out / "14_empty_dir").mkdir(); rows.append(("14_empty_dir/", "n/a", "пустая папка: предупреждение, строк нет"))
with (out / "expected.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path_to_study", "expected_status", "note"]); w.writerows(rows)
print(f"{len(rows)} записей → {out}")
