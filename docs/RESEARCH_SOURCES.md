# Источники исследований и внешние данные

Дата проверки: 2026-09-20. Документ объединяет клинические источники,
литературный обзор и аудит внешних наборов. Факт наличия ссылки не означает, что
данные разрешено скачивать, передавать в облако или включать в релиз. Точные
лицензионные замечания к использованным компонентам находятся в
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## 1. Клинические критерии

- [ISCD DXA Atlas: Hip Rotation](https://iscd.org/dxaatlas/hip-rotation/) —
  профиль малого вертела как признак ротации бедра.
- [Official Adult Positions ISCD 2023](https://iscd.org/official-positions-2023/)
  — официальные позиции по получению и анализу DXA.
- [Improving DXA Quality by Avoiding Common Technical and Diagnostic Pitfalls,
  Part 1](https://tech.snmjournals.org/content/51/3/167) — укладка, Th12,
  гребни подвздошных костей и ошибки анализа.
- [Operator-Related Errors and Pitfalls in
  DXA](https://www.sciencedirect.com/science/article/abs/pii/S1076633220304517)
  — частота ошибок укладки; ротация бедра около 20% исследований.
- [Incorrect hip analysis: clinical
  case](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530240/) — разбор
  ошибки анализа бедра.
- [Pitfalls and sources of error in DXA
  reporting](https://pubmed.ncbi.nlm.nih.gov/41740278/) — артефакты,
  дегенеративные изменения и эндопротезы.
- [Evaluation of patient positioning during DXA in daily
  practice](https://pubmed.ncbi.nlm.nih.gov/17965906/) — ошибки укладки в
  рутинной практике.

## 2. Автоматизация и перенос методов

- [CNN contour detection of hip/proximal femur on
  DXA](https://www.tandfonline.com/doi/full/10.1080/21681163.2023.2296626) —
  U-Net-сегментация и проблема наложения таза и бедра.
- [Automated femur landmarking in
  DXA](https://pubmed.ncbi.nlm.nih.gov/42453281/) — U-Net + геометрические
  ориентиры.
- [Transferable CNN data mining on spine
  DXA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12797949/) — перенос
  ImageNet-моделей на DXA.
- [Deep Learning in DXA Image
  Segmentation](https://www.sciencedirect.com/org/science/article/pii/S1546221820000673)
  — обзор сегментации DXA.
- [VerteNet](https://arxiv.org/html/2502.02097v1) и
  [код](https://github.com/zaidilyas89/VerteNet) — T12–L5 на боковых DXA;
  полезен как обоснование явной точки Th12, но веса не переносятся напрямую на
  AP-проекцию.
- [Automated QA for digital
  radiography](https://pmc.ncbi.nlm.nih.gov/articles/PMC13361558/) —
  attention U-Net, геометрия и классический ML.
- [Lesser trochanter profile and femoral
  rotation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11415034/) — высокая
  воспроизводимость профиля малого вертела; источник проверенной гипотезы о
  сравнении сторон.
- [Lumbar radiography QC with enhanced
  U-Net](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9087032/) —
  сегментация + правила укладки.
- [Mammography positioning
  QC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10151341/) и
  [QC-Automator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6987246/) —
  близкие задачи контроля позиционирования и артефактов.

Вывод обзора: гибрид «локализация анатомии → измерения → простой
классификатор» соответствует современной практике и лучше подходит малой
выборке, чем большой black-box без новых меток.

## 3. Реально использованные внешние наборы

### Arak Bone Densitometry Center

- [статья](https://www.medrxiv.org/content/10.1101/2025.01.25.24319689v1.full);
- [зеркало Kaggle](https://www.kaggle.com/datasets/tommyngx/arak-bone-densitometry-center);
- проверенный архив: 229 362 899 bytes;
- SHA-256: `fc30415a0c0566227266c51b2d0b06c3a094ff85bbcc6da0285b652d56e88389`;
- содержимое: 4020 PNG и `FinallDATA/TableFinal-OSTEO.xlsx` на 3643 строки;
- 4019 уникальных grayscale-изображений после декодирования, один точный дубль;
- отдельных QC-меток, масок, README и LICENSE внутри архива нет.

Набор использован для слабых анатомических меток и исследовательского
предобучения. BMD/T-score нельзя превращать в GOOD/BAD. Идентификаторы PNG не
дают автоматически 4019 независимых пациентов; разбиение строится после
восстановления patient groups. Kaggle указывает CC BY-NC 4.0, но загрузчик
зеркала не совпадает с авторами статьи и разрешение на зеркало не приложено.
Изображения не входят в проект или release bundle.

### Pakistan DXA

- [Mendeley Data](https://data.mendeley.com/datasets/9c2tz2xfyv/1);
- 173 уникальных исходных DXA после исключения готовых аугментаций и дублей;
- карточка набора — CC BY 4.0;
- классы остеопороза не являются метками качества укладки;
- patient IDs не подтверждены.

Набор участвовал в исторических переносах и не является новым независимым
источником для следующего сравнения.

### CGMH-PelvisSeg

- [Kaggle](https://www.kaggle.com/datasets/tommyngx/cgmh-pelvisseg);
- [код PELE](https://github.com/ECNUACRush/PELEscores);
- [статья PELE](https://openreview.net/pdf?id=TJ5INXQMvE).

Проверены 400 image/mask пар; сегментатор достиг Dice 0.9496 на своём validation,
но перенос crop на целевые DXA дал fallback для 107 из 153 изображений. Это
рентген, не DXA. Лицензионные сведения источников расходятся (CC0 в API,
research/education subtitle, CC-BY-NC-SA в статье), поэтому набор нельзя
считать безусловно разрешённым для коммерческого использования.

## 4. Проверенные кандидаты, не включённые в обучение релиза

| Набор | Что доступно | Решение |
|---|---|---|
| [BUU-LSPINE](https://services.informatics.buu.ac.th/spine/) | AP/LA рентген позвоночника, координаты, собственный EULA | резерв для ориентиров; не DXA и не QC |
| [DeepFluoro](https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi%3A10.7281%2FT1%2FIFSXNV) | 366 проекций, 6 кадаверов, маски/ориентиры | слишком мало независимых объектов; CC BY-NC 4.0 |
| [Vertebrae for Scoliosis v2](https://data.mendeley.com/datasets/4kby36n3ng/2) | 737 PA-рентгенограмм и полигоны | проверить происхождение, дубли и группы |
| [PelviXNet / PXR150](https://figshare.com/articles/dataset/Pelvic_X-ray_images_for_PelviXNet_model/17185814) | 150 тазовых рентгенограмм | возможное анатомическое предобучение; проверить дубли с CGMH |
| [MTDDH](https://doi.org/10.57760/sciencedb.24372) | детские тазовые рентгенограммы, сегментация и ориентиры | большой возрастной/domain shift |
| [AASCE 2019](https://aasce19.github.io/) | 609 AP-рентгенограмм, 68 ориентиров | лицензия/скачивание не подтверждены |
| [UK Biobank DXA](https://biobank.ctsu.ox.ac.uk/ukb/field.cgi?id=20158) | закрытый доступ к DXA | не открытый набор; не использовать как публичный источник |
| [Hip-35](https://www.nature.com/articles/s41597-026-07375-0) | 35 000 синтетических рентгенограмм | синтетика/pathology не заменяет реальные QC-метки |

Не подходят как источник изображений DXA QC: табличный
[Indian femur DEXA](https://data.mendeley.com/datasets/kys6x6wykj/1), knee
X-ray osteoporosis datasets, а также репозитории без доступного массива.

## 5. Правила использования внешних данных

1. Проверить лицензию именно изображений, а не только статьи или карточки.
2. Сохранить URL, версию, размер, SHA-256 архива и дату получения.
3. Проверить декодирование, дубли, возможные производные одного снимка и
   пациентские/исследовательские группы до разбиения.
4. Не смешивать внешние изображения с целевой validation.
5. Не превращать диагноз, BMD или патологию в метку качества укладки.
6. Не загружать приватные DICOM организаторов в Kaggle, Colab или другой
   внешний сервис без письменного разрешения.
7. Не включать исходные внешние данные в Git, Docker или release archive.
8. Для кандидата фиксировать гипотезу и gate до просмотра целевых OOF.

## 6. Следующий обоснованный цикл

Широкий перебор архитектур на прежних 249 размеченных кадрах исчерпан. Полезный
следующий цикл требует:

- нового закрытого набора другого центра/аппарата;
- ручной проверки точек малого вертела, Th12 и гребней;
- экспертной разметки редких причин;
- повторного внешнего протокола без выбора гипотез по его результатам.

До появления новых меток допустимы только воспроизводимые проверки реализации,
калибровки и производительности; они не являются новой оценкой качества.
