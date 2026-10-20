"""Decision cards: how the model reached its verdict, on cases picked mechanically from the isolated validation.

For every violation type: the confirmed defect with the highest score (TP), the missed defect with the
lowest score (FN) and the false alarm with the highest score (FP); for each region the clearest correct
normal (TN). Verdict and scores come from the isolated validation (scripts/per_type_metrics.py,
decisions_release_rules.csv): the model never saw the image. Landmarks, margins, lesser trochanter and
artefacts are drawn by the release model; those detectors were not trained on quality labels. The release
verdict is shown separately: the release model did see the image during training.

The output contains organizer images: keep it local (outputs/ is not in git) and never upload it.

  python scripts/make_case_cards.py            # -> outputs/case_cards/cases.html
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.api import default_model_dir  # noqa: E402
from dxaqc.dicom_io import read_dicom  # noqa: E402
from dxaqc.landmarks import draw_overlay  # noqa: E402
from dxaqc.release_predictor import ReleasePredictor  # noqa: E402

DECISIONS = ROOT / 'outputs/final_validation_isolated/decisions_release_rules.csv'
DATA = ROOT / 'data/interim'
OUT = ROOT / 'outputs/case_cards'
R = ['spine_scan_range', 'spine_axis_tilt', 'spine_artifact', 'hip_positioning', 'hip_roi_margins']
NAME = {'spine_scan_range': 'Границы сканирования', 'spine_axis_tilt': 'Наклон оси позвоночника',
        'spine_artifact': 'Посторонние предметы', 'hip_positioning': 'Позиционирование (ротация) бедра',
        'hip_roi_margins': 'Поля области интереса бедра', 'spine': 'Позвоночник', 'hip': 'Бедро'}
KIND = {'TP': ('Верно найден брак', '#2e7d32'), 'FN': ('Пропуск', '#b26a00'),
        'FP': ('Ложная тревога', '#b3261e'), 'TN': ('Верная норма', '#2e7d32')}


def pick(t: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    out = []
    for c in R:
        u = t[t['score_' + c].notna() & t['y_' + c].notna()]
        y, d = u['y_' + c].eq(1), u['reported_' + c].astype(bool)
        for kind, mask, ascending in (('TP', y & d, False), ('FN', y & ~d, True), ('FP', ~y & d, False)):
            if mask.any():
                out.append((kind, c, u[mask].sort_values('score_' + c, ascending=ascending).iloc[0]))
    for region, mask in (('spine', t.region.eq('spine')), ('hip', t.region.str.startswith('hip'))):
        u = t[mask & t.y_quality.eq(0) & t.quality_class.eq(0)]
        if len(u):
            out.append(('TN', region, u.sort_values('score_quality').iloc[0]))
    return out


def evidence(region: str, lm: dict) -> list[str]:
    """Measurements in the words of the TZ criteria."""
    m = lm.get('measurements', {})
    if region == 'lumbar_spine':
        art = lm.get('artifacts', {})
        return [f"Наклон оси L1–L4: {m.get('spine_tilt_deg', float('nan')):+.1f}° (ТЗ: нарушение при отклонении более 5°)",
                f"Поле над L1: {m.get('top_margin_vert', float('nan')):.2f}, под L4: {m.get('bottom_margin_vert', float('nan')):.2f} высоты позвонка",
                f"Детектор артефактов: проводов {len(art.get('wires', []))}, металла {len(art.get('metal', []))}"]
    lt = lm.get('lesser_trochanter') or {}
    lines = [f"Поля рамки, мм: сверху {m.get('roi_top_mm', 0):.0f}, снизу {m.get('roi_bottom_mm', 0):.0f}, "
             f"слева {m.get('roi_left_mm', 0):.0f}, справа {m.get('roi_right_mm', 0):.0f} (ТЗ: 30 сверху и снизу, 20 по бокам)"]
    if 'roi_min_margin_mm' in lm:
        mm = lm['roi_min_margin_mm']
        lines.append(f"Наименьшее поле {mm:.1f} мм: " + ('рамка выходит за край снимка — нарушение полей' if mm <= 0 else 'рамка внутри снимка'))
    lines.append(f"Выступ малого вертела: {lt['lt_prominence_mm']:.1f} мм (ТЗ: при правильной ротации вертел слегка деформирует контур)"
                 if lt else 'Малый вертел не измерен')
    return lines


def card(kind: str, focus: str, r: pd.Series, region: str, pred, png: bytes) -> str:
    label, color = KIND[kind]
    lm = pred.details.get('landmarks') or {}
    truth = [NAME[c] for c in R if r['y_' + c] == 1] or ['нет']
    found = [NAME[c] for c in R if bool(r['reported_' + c])] or ['нет']
    scores = ''.join(f"<tr><td>{NAME[c]}</td><td style='{'color:#b3261e;font-weight:600' if r['score_' + c] >= 1 else ''}'>"
                     f"{r['score_' + c]:.2f}</td></tr>" for c in R if pd.notna(r['score_' + c]))
    final = ', '.join(NAME.get(c, c) for c in pred.violations) or 'нет'
    comment = r.get('comment')
    return f"""<section class='card'><p><span class='tag' style='background:{color}'>{label}</span> <b>{NAME[focus]}</b>
<span class='mut'>исследование …{str(r.study)[-8:]}, {r.region}</span></p>
<div class='row'><img alt='снимок с ориентирами' src='data:image/png;base64,{base64.b64encode(png).decode()}'><div class='info'>
<p><b>Эксперт:</b> {'брак' if r.y_quality else 'норма'}; причины: {html.escape(', '.join(truth))}{'; комментарий: «' + html.escape(str(comment).strip()) + '»' if isinstance(comment, str) and comment.strip() else ''}</p>
<p><b>Модель (изолированная валидация, снимок не видела):</b> {'брак' if r.quality_class else 'норма'}, качество {r.score_quality:.2f} от порога; причины: {html.escape(', '.join(found))}</p>
<table><tr><th>Причина</th><th>Оценка / порог</th></tr>{scores}</table>
<p><b>На чём основано решение:</b></p><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in evidence(region, lm))}</ul>
<p class='mut'>Финальная модель (видела снимок при обучении): {'брак' if pred.quality_class else 'норма'}; {html.escape(final)}</p>
</div></div></section>"""


def main() -> None:
    t = pd.read_csv(DECISIONS)
    labels = pd.read_csv(DATA / 'image_labels.csv')[['image_uid', 'path', 'comment']]
    t = t.merge(labels, on='image_uid', validate='1:1')
    predictor = ReleasePredictor(default_model_dir())
    cards, index = [], []
    for kind, focus, r in pick(t):
        img = read_dicom(DATA / 'train/Исследования' / r.path)
        region, pred = predictor.predict_image(img.pixels, mm_per_px=img.meta.get('spacing_mm'))
        lm = pred.details.get('landmarks') or {}
        rgb = draw_overlay(img.pixels, region, lm['points'], lm['measurements'], [c for c in R if bool(r['reported_' + c])],
                           int(r.quality_class), lt=lm.get('lesser_trochanter'), art_mask=lm.get('artifact_mask'))
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, 'PNG')
        cards.append(card(kind, focus, r, region, pred, buf.getvalue()))
        index.append({'kind': kind, 'focus': focus, 'image_uid': r.image_uid, 'study': r.study})
        print(kind, focus, r.region, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    (OUT / 'cases.html').write_text(f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Разбор решений DXA QC</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px;background:#fff;color:#14171b}}
.card{{border:1px solid #e3e5e8;border-radius:8px;padding:12px;margin:14px 0;background:#fafafa}}.row{{display:flex;gap:16px;flex-wrap:wrap}}
img{{max-width:420px;width:100%;image-rendering:pixelated}}.info{{flex:1;min-width:280px}}.mut{{color:#5a6069}}
.tag{{color:#fff;padding:2px 8px;border-radius:10px;font-size:13px}}table{{border-collapse:collapse}}td,th{{padding:2px 8px;border-bottom:1px solid #e3e5e8;text-align:left}}
</style></head><body><h1>Разбор решений: верные, пропуски, ложные тревоги</h1>
<p>Отбор механический (см. docstring скрипта). Оценка — доля от порога причины, 1.0 = порог. Только для локального просмотра: внутри снимки организатора.</p>
{''.join(cards)}</body></html>""")
    print(len(cards), 'карточек →', OUT / 'cases.html')


if __name__ == '__main__':
    main()
