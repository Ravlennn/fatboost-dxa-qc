"""Landmark labeling package for physicians: files to send, an instruction, and a checker for what comes back.

Physicians place anatomical points only (vertebrae, iliac crests, femoral head and neck, lesser trochanter).
They never see or give a quality verdict: the target labels stay the organizer's.

  .venv/bin/python scripts/make_labeling_package.py prepare   # model points for every image -> _work/
  .venv/bin/python scripts/make_labeling_package.py build     # HTML files per physician, PDF, zips
  .venv/bin/python scripts/make_labeling_package.py check FILE.json ...   # validate returned files

Output: outputs/labeling_package/ (git-ignored: it holds the organizer's images).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

DATA = ROOT / 'data/interim/train/Исследования'
OUT = ROOT / 'outputs/labeling_package'
WORK = OUT / '_work'
PACKAGE = 'dxaqc-landmarks-2026-10'
DOCTORS = 3
SHARED = {'spine': 8, 'hip': 12}       # images every physician labels (inter-rater agreement)
SEED = 42

# Points the physician places. key -> (short label, name, hint). Order = order of work on the image.
SPINE = {
    'th12_mid': ('Th12', 'Th12 — центр тела',
                 'Центр тела XII грудного позвонка (несёт XII ребро, лежит над L1). '
                 'Если центр тела Th12 за верхним краем кадра — «Нет на снимке».'),
    'L1_top': ('L1 верх', 'L1 — верхняя замыкательная пластинка',
               'Середина верхней замыкательной пластинки L1 (первый позвонок без рёбер).'),
    'L1L2': ('L1/L2', 'Промежуток L1/L2', 'Середина межпозвонкового промежутка между L1 и L2.'),
    'L2L3': ('L2/L3', 'Промежуток L2/L3', 'Середина межпозвонкового промежутка между L2 и L3.'),
    'L3L4': ('L3/L4', 'Промежуток L3/L4', 'Середина межпозвонкового промежутка между L3 и L4.'),
    'L4_bottom': ('L4 низ', 'L4 — нижняя замыкательная пластинка',
                  'Середина нижней замыкательной пластинки L4.'),
    'iliac_left_top': ('Гребень Л', 'Гребень подвздошной кости слева на снимке',
                       'Самая высокая точка гребня подвздошной кости, который на снимке слева. '
                       'Если гребень не виден (за нижним или боковым краем кадра) — «Нет на снимке».'),
    'iliac_right_top': ('Гребень П', 'Гребень подвздошной кости справа на снимке',
                        'Самая высокая точка гребня подвздошной кости, который на снимке справа. '
                        'Если гребень не виден — «Нет на снимке».'),
}
HIP = {
    'head_center': ('Головка', 'Центр головки бедренной кости',
                    'Центр окружности головки бедренной кости.'),
    'neck_center': ('Шейка', 'Середина шейки бедра',
                    'Середина шейки в самом узком месте — посередине между верхним и нижним контуром шейки.'),
    'lt_base_top': ('МВ верх', 'Малый вертел — верхнее основание',
                    'Где контур малого вертела отходит от кортикального слоя диафиза (верхний край выступа). '
                    'Если малый вертел не выступает за контур диафиза — «Нет на снимке» для всех трёх точек вертела.'),
    'lt_apex': ('МВ', 'Малый вертел — вершина',
                'Самая медиальная (выступающая внутрь) точка контура малого вертела.'),
    'lt_base_bottom': ('МВ низ', 'Малый вертел — нижнее основание',
                       'Где контур малого вертела возвращается к кортикальному слою диафиза (нижний край выступа).'),
}
POINT_SETS = {'spine': SPINE, 'hip': HIP}


def _png_data_url(pixels: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _initial_points(region: str, pixels: np.ndarray, predictor) -> dict:
    """Model points as a starting position. Physicians confirm or move every one of them."""
    from dxaqc.lesser_trochanter import detect
    h, w = pixels.shape
    model_region = 'lumbar_spine' if region == 'spine' else region
    p = predictor.landmarks.predict(pixels, model_region)['points']
    clip = lambda xy: [float(np.clip(xy[0], 1, w - 2)), float(np.clip(xy[1], 1, h - 2))]
    if region == 'spine':
        out = {k: p[k] for k in ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')}
        top, bot = np.array(p['L1_top']), np.array(p['L4_bottom'])
        vh = float(np.median(np.diff([p[k][1] for k in ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')])))
        out['th12_mid'] = [top[0], top[1] - 0.5 * vh]
        out['iliac_left_top'] = [bot[0] - 2.0 * vh, bot[1] + 0.3 * vh]
        out['iliac_right_top'] = [bot[0] + 2.0 * vh, bot[1] + 0.3 * vh]
        return {k: clip(v) for k, v in out.items()}
    c, head_side = np.array(p['neck_center']), np.array(p['neck_head_side'])
    out = {'neck_center': list(c), 'head_center': list(c + 1.3 * (head_side - c))}
    lt, _ = detect(pixels, p, mm=0.6)
    if lt is not None:
        out.update({k: lt['points'][k] for k in ('lt_base_top', 'lt_apex', 'lt_base_bottom')})
    else:   # detector found no shaft: a rough guess below the neck on the medial side
        t = np.array(p['neck_troch_side'])
        med = -1.0 if t[0] > c[0] else 1.0
        base = c + np.array([0.0, 60.0])
        out.update({'lt_apex': [base[0] + med * 25, base[1]], 'lt_base_top': [base[0] + med * 12, base[1] - 15],
                    'lt_base_bottom': [base[0] + med * 12, base[1] + 15]})
    # A flat or absent trochanter makes the detector stack its three points: spread them so each can be grabbed.
    apex = np.array(out['lt_apex'])
    for key, dy in (('lt_base_top', -12.0), ('lt_base_bottom', 12.0)):
        if np.hypot(*(np.array(out[key]) - apex)) < 8:
            out[key] = [apex[0], apex[1] + dy]
    return {k: clip(v) for k, v in out.items()}


def prepare() -> None:
    from dxaqc.api import create_predictor
    from dxaqc.dicom_io import read_dicom
    predictor = create_predictor(ROOT / 'models/release', threads=4)
    labels = pd.read_csv(ROOT / 'data/interim/image_labels.csv')
    rng = random.Random(SEED)
    rows = list(labels.itertuples())
    rng.shuffle(rows)                   # neutral ids: numbering does not follow studies
    (WORK / 'img').mkdir(parents=True, exist_ok=True)
    items, counters = [], {'spine': 0, 'hip': 0}
    for r in rows:
        kind = 'spine' if r.region == 'spine' else 'hip'
        counters[kind] += 1
        item_id = f"{'П' if kind == 'spine' else 'Б'}-{counters[kind]:03d}"
        img = read_dicom(DATA / r.path)
        pixels = img.pixels
        points = _initial_points(r.region, pixels, predictor)
        Image.fromarray(pixels).save(WORK / 'img' / f'{item_id}.png')
        items.append({'id': item_id, 'kind': kind, 'region': r.region, 'image_uid': r.image_uid,
                      'study_uid': r.study_uid, 'path': r.path, 'width': int(pixels.shape[1]),
                      'height': int(pixels.shape[0]), 'init': points})
        print(item_id, r.region, pixels.shape, flush=True)
    (WORK / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(items), 'items ->', WORK / 'items.json')


COLORS = {
    'th12_mid': '#b48cff', 'L1_top': '#5cc8ff', 'L1L2': '#4fd6c4', 'L2L3': '#7ee08a', 'L3L4': '#c8e070',
    'L4_bottom': '#f0b85a', 'iliac_left_top': '#ff8a65', 'iliac_right_top': '#ff6f91',
    'head_center': '#ff6f91', 'neck_center': '#5cc8ff', 'lt_base_top': '#c8e070', 'lt_apex': '#f2c14e',
    'lt_base_bottom': '#f0905a',
}
KIND_RU = {'spine': 'позвоночник', 'hip': 'бедро'}
REGION_LABEL = {'spine': 'Поясничный отдел позвоночника', 'hip_left': 'Бедро, левое', 'hip_right': 'Бедро, правое'}


def assign(items: list[dict]) -> dict[int, dict[str, list[dict]]]:
    """Every physician gets the shared images plus a third of the rest, spine and hip in separate files."""
    rng = random.Random(SEED)
    out = {d: {'spine': [], 'hip': []} for d in range(1, DOCTORS + 1)}
    for kind in ('spine', 'hip'):
        pool = [it for it in items if it['kind'] == kind]
        if kind == 'hip':   # shared hips: half left, half right
            shared = (rng.sample([it for it in pool if it['region'] == 'hip_left'], SHARED['hip'] // 2)
                      + rng.sample([it for it in pool if it['region'] == 'hip_right'], SHARED['hip'] - SHARED['hip'] // 2))
        else:
            shared = rng.sample(pool, SHARED['spine'])
        ids = {it['id'] for it in shared}
        rest = [it for it in pool if it['id'] not in ids]
        rng.shuffle(rest)
        for j, it in enumerate(rest):
            out[j % DOCTORS + 1][kind].append(it)
        for d in out:
            own = out[d][kind] + shared
            random.Random(SEED + d).shuffle(own)   # shared images sit among the others, unmarked
            out[d][kind] = own
        for it in shared:
            it['shared'] = True
    return out


def _start_points(it: dict) -> dict:
    """Starting points as shown: a Th12 above the frame is parked beside L1 so that the two can be told apart."""
    pts = {k: list(v) for k, v in it['init'].items()}
    if it['kind'] == 'spine' and pts['th12_mid'][1] < 4:
        pts['th12_mid'] = [max(2.0, pts['L1_top'][0] - 30), 3.0]
    return pts


def build() -> None:
    items = json.loads((WORK / 'items.json').read_text(encoding='utf-8'))
    template = (ROOT / 'scripts/labeling_template.html').read_text(encoding='utf-8')
    plan = assign(items)
    internal = OUT / '_internal'
    internal.mkdir(parents=True, exist_ok=True)
    for d, kinds in plan.items():
        folder = OUT / f'Врач_{d}'
        folder.mkdir(parents=True, exist_ok=True)
        for kind, its in kinds.items():
            points = [{'key': k, 'short': v[0], 'name': v[1], 'hint': v[2], 'color': COLORS[k]}
                      for k, v in POINT_SETS[kind].items()]
            data = {
                'package': PACKAGE, 'part': f'врач{d}_{KIND_RU[kind]}', 'kind': kind,
                'title': f'Разметка ориентиров DXA · {KIND_RU[kind].capitalize()} · Врач {d}',
                'welcome': (f'В этом файле {len(its)} снимков ({KIND_RU[kind]}). На каждом нужно проверить '
                            f'{len(points)} анатомических точек. Оценивать качество снимка не нужно.'),
                'points': points,
                'items': [{'id': it['id'], 'label': REGION_LABEL[it['region']], 'width': it['width'],
                           'height': it['height'], 'init': _start_points(it),
                           'img': _png_data_url(np.array(Image.open(WORK / 'img' / f"{it['id']}.png")))}
                          for it in its],
            }
            html = template.replace('/*DATA*/null', json.dumps(data, ensure_ascii=False))
            path = folder / f'Разметка_{KIND_RU[kind]}.html'
            path.write_text(html, encoding='utf-8')
            print(f'{path.relative_to(OUT)}: {len(its)} снимков, {path.stat().st_size / 1e6:.1f} МБ')
    # Ours only, never sent: which neutral id is which image, and who labels what.
    (internal / 'assignment.json').write_text(json.dumps({
        'package': PACKAGE, 'seed': SEED, 'shared': SHARED,
        'items': {it['id']: {k: it[k] for k in ('kind', 'region', 'image_uid', 'study_uid', 'path', 'width', 'height', 'init')}
                  | {'shared': bool(it.get('shared'))} for it in items},
        'doctors': {d: {k: [it['id'] for it in v] for k, v in kinds.items()} for d, kinds in plan.items()},
    }, ensure_ascii=False, indent=1), encoding='utf-8')


def check(files: list[str]) -> None:
    """Validate returned files and report progress and agreement on the shared images (mm at 0.6 mm/px)."""
    meta = json.loads((OUT / '_internal/assignment.json').read_text(encoding='utf-8'))['items']
    by_image: dict[str, list[tuple[str, dict]]] = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding='utf-8'))
        if d.get('format') != 'dxaqc-landmarks-v1' or d.get('package') != PACKAGE:
            print(f'{f}: не файл этого пакета'); continue
        done = sum(r['complete'] for r in d['items'])
        problems = []
        for r in d['items']:
            m = meta.get(r['id'])
            if m is None:
                problems.append(f"{r['id']}: неизвестный снимок"); continue
            for k, p in r['points'].items():
                if p['status'] == 'ok' and not (0 <= p['x'] < m['width'] and 0 <= p['y'] < m['height']):
                    problems.append(f"{r['id']}.{k}: вне снимка")
            if r['complete'] and not r['unusable']:
                by_image.setdefault(r['id'], []).append((d['doctor'], r['points']))
        print(f"{Path(f).name}: {d['doctor']}, {d['part']}: готово {done} из {len(d['items'])}, "
              f"сомнений {sum(r['doubt'] for r in d['items'])}, непригодных {sum(r['unusable'] for r in d['items'])}, "
              f"медиана {np.median([r['seconds'] for r in d['items'] if r['complete']] or [0]):.0f} с на снимок")
        for p in problems[:10]:
            print('   ', p)
    pairs: dict[str, list[float]] = {}
    for rows in by_image.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                for k, a in rows[i][1].items():
                    b = rows[j][1].get(k)
                    if b and a['status'] == b['status'] == 'ok':
                        pairs.setdefault(k, []).append(0.6 * float(np.hypot(a['x'] - b['x'], a['y'] - b['y'])))
    if pairs:
        print('Расхождение врачей на общих снимках, мм (медиана / 90-й перцентиль):')
        for k, v in pairs.items():
            print(f'   {k:16s} {np.median(v):5.1f} / {np.percentile(v, 90):5.1f}   (пар: {len(v)})')


CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
INSTR = WORK / 'instr'   # screenshots of the labeler and the example crops (organizer images: not in git)


def _jpeg_data_url(path: Path, width: int = 1600) -> str:
    im = Image.open(path).convert('RGB')
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=86, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def instruction(deadline: str, send_to: str, contact: str) -> None:
    """One PDF per physician (their own image counts), the cover letter, and a zip per physician."""
    import re
    import subprocess
    import zipfile
    plan = json.loads((OUT / '_internal/assignment.json').read_text(encoding='utf-8'))['doctors']
    schemes = (ROOT / 'scripts/labeling_schemes.html').read_text(encoding='utf-8')
    svg = {k: re.search(rf'<div class="fig" id="{k}">(<svg.*?</svg>)</div>', schemes, re.S).group(1) for k in ('spine', 'hip')}
    base = (ROOT / 'scripts/labeling_instruction.html').read_text(encoding='utf-8')
    fills = {'{{SVG_SPINE}}': svg['spine'], '{{SVG_HIP}}': svg['hip'],
             '{{IMG_UI_HIP}}': _jpeg_data_url(INSTR / 'ui_hip.png'), '{{IMG_UI_SPINE}}': _jpeg_data_url(INSTR / 'ui_spine.png'),
             '{{IMG_EX_LT}}': _jpeg_data_url(INSTR / 'ex_lt.png', 900), '{{IMG_EX_CRESTS}}': _jpeg_data_url(INSTR / 'ex_П-088.png', 900),
             '{{IMG_EX_NOCRESTS}}': _jpeg_data_url(INSTR / 'ex_П-047.png', 900),
             '{{DEADLINE}}': deadline, '{{SEND_TO}}': send_to, '{{CONTACT}}': contact}
    for d, kinds in plan.items():
        html = base
        for k, v in {**fills, '{{N_TOTAL}}': str(len(kinds['spine']) + len(kinds['hip'])),
                     '{{N_SPINE}}': str(len(kinds['spine'])), '{{N_HIP}}': str(len(kinds['hip']))}.items():
            html = html.replace(k, v)
        folder = OUT / f'Врач_{d}'
        src = WORK / f'instruction_{d}.html'
        src.write_text(html, encoding='utf-8')
        pdf = folder / 'Инструкция.pdf'
        subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        f'--print-to-pdf={pdf}', src.as_uri()], check=True, capture_output=True)
        with zipfile.ZipFile(OUT / f'Врач_{d}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in ('Инструкция.pdf', 'Разметка_позвоночник.html', 'Разметка_бедро.html'):
                zf.write(folder / f, f'Врач_{d}/{f}')
        print(f'Врач_{d}.zip: {(OUT / f"Врач_{d}.zip").stat().st_size / 1e6:.1f} МБ, PDF {pdf.stat().st_size / 1e6:.1f} МБ')
    (OUT / 'Письмо_врачам.txt').write_text(LETTER.format(deadline=deadline, send_to=send_to, contact=contact), encoding='utf-8')


LETTER = """Тема: Разметка анатомических ориентиров на снимках денситометрии

Здравствуйте!

Мы разрабатываем систему автоматического контроля качества укладки при денситометрии (DXA)
и просим вашей помощи с разметкой. Нужно отметить анатомические точки на снимках позвоночника
и бедра: позвонки L1–L4, Th12, гребни подвздошных костей, головку и шейку бедра, малый вертел.
Оценивать качество снимков и ставить заключения не нужно — только точки.

Во вложении архив: инструкция (PDF, читается за 7 минут) и два файла разметки. Они открываются
двойным щелчком в браузере, без интернета и установки. Точки уже расставлены программой
как черновик — их нужно проверить и поправить. Около 1 минуты на снимок, всего 1,5–2 часа,
можно в несколько подходов: работа сохраняется автоматически.

Когда закончите, нажмите в каждом файле «Сохранить результат» и пришлите два .json-файла
из папки «Загрузки»: {send_to}, до {deadline}.

Снимки обезличены. Пожалуйста, не передавайте файлы третьим лицам и удалите их после сдачи.

Вопросы: {contact}

Спасибо!
"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'build', 'instruction', 'check'])
    ap.add_argument('files', nargs='*')
    ap.add_argument('--deadline', default='[срок]')
    ap.add_argument('--send-to', default='[кому и куда прислать]')
    ap.add_argument('--contact', default='[контакт для вопросов]')
    args = ap.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'build':
        build()
    elif args.stage == 'instruction':
        instruction(args.deadline, args.send_to, args.contact)
    else:
        check(args.files)
