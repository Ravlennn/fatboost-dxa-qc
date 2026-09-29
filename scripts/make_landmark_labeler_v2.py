"""Labeler v2: new landmarks for rotation and scan range, on all our images.

New points (initialised at heuristic positions from the v1 landmarks, status 'todo'):
  hip   : lt_apex (most medial tip of the lesser trochanter),
          lt_base_top / lt_base_bottom (where the lesser trochanter leaves / rejoins the shaft cortex)
  spine : th12_mid (middle of the Th12 body), iliac_left_top / iliac_right_top (highest crest point)
Each point: 'todo' (not reviewed), 'set' (placed/confirmed) or 'invisible' (not in image / not visible).
'invisible' is itself a label: an over-rotated hip hides the lesser trochanter.
v1 points are shown and may be corrected (status 'set' when moved).

  .venv/bin/python scripts/make_landmark_labeler_v2.py
  -> outputs/landmark_labeling/labeler_v2.html  (export: landmarks_manual_v2.json)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'
NEW_HIP = ['lt_apex', 'lt_base_top', 'lt_base_bottom']
NEW_SPINE = ['th12_mid', 'iliac_left_top', 'iliac_right_top']


def init_new(r):
    p = r['points']
    if r['region'] == 'spine':
        top, bot = np.array(p['L1_top']), np.array(p['L4_bottom'])
        vh = float(np.median(np.diff([p[k][1] for k in ('L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom')])))
        return {'th12_mid': [top[0], max(top[1] - 0.5 * vh, 1)],
                'iliac_left_top': [bot[0] - 2.0 * vh, min(bot[1] + 0.3 * vh, r['height'] - 2)],
                'iliac_right_top': [bot[0] + 2.0 * vh, min(bot[1] + 0.3 * vh, r['height'] - 2)]}
    c, t = np.array(p['neck_center']), np.array(p['neck_troch_side'])
    # Lesser trochanter: below the neck on the medial side of the shaft (medial = away from trochanter).
    med = -1.0 if t[0] > c[0] else 1.0
    base = c + np.array([0.0, 60.0])
    return {'lt_apex': [base[0] + med * 25, base[1]], 'lt_base_top': [base[0] + med * 12, base[1] - 15],
            'lt_base_bottom': [base[0] + med * 12, base[1] + 15]}


HTML = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Разметка v2: малый вертел, Th12, гребни</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--acc:#4af}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;gap:16px;padding:12px}
#side{width:370px;flex:none} canvas{background:#000;cursor:crosshair;border:1px solid #333}
button{background:#222;color:var(--fg);border:1px solid #444;padding:5px 9px;border-radius:4px;cursor:pointer;margin:2px}
button:hover{border-color:var(--acc)} .pt{display:flex;align-items:center;gap:6px;margin:2px 0;padding:2px 4px;border-radius:3px;cursor:pointer}
.pt.sel{background:#243} .sw{width:12px;height:12px;border-radius:50%;flex:none} .mut{color:var(--mut);font-size:12px;line-height:1.45}
.st{font-size:11px;padding:1px 5px;border-radius:3px;margin-left:auto} .todo{background:#553} .set{background:#253} .invisible{background:#533}
#prog{height:6px;background:#333;border-radius:3px;margin:8px 0} #prog div{height:100%;background:var(--acc);border-radius:3px}
select{background:#222;color:#eee;border:1px solid #444;padding:3px}
</style></head><body>
<div><canvas id="c"></canvas></div>
<div id="side">
<h3 style="margin:4px 0" id="title"></h3><div class="mut" id="meta"></div>
<div>Показывать: <select id="flt"><option value="all">все</option><option value="hip">бёдра</option><option value="spine">позвоночник</option><option value="todo">неразмеченные</option></select></div>
<div id="prog"><div></div></div><div class="mut" id="cnt"></div>
<div id="pts"></div>
<div><button id="inv">Не видна (I)</button><button id="ok">Подтвердить (Enter)</button></div>
<div><button id="prev">← Пред.</button><button id="next">След. →</button><button id="export">Скачать JSON</button></div>
<div class="mut" style="margin-top:10px">
<b>Порядок:</b> выберите точку (1–9 или клик в списке), кликните/перетащите на снимке — статус станет «set». Если структура не видна или вне кадра — «Не видна» (I). Серые точки v1 можно поправить, но это необязательно. Снимок готов, когда у новых точек нет статуса todo.<br><br>
<b>Малый вертел</b> (бедро): <i>lt_apex</i> — самая медиальная вершина выступа малого вертела; <i>lt_base_top/bottom</i> — где его контур отходит от кортикального слоя диафиза и возвращается к нему. Если контур диафиза гладкий и выступа нет — все три «не видна».<br>
<b>Th12</b>: середина тела Th12 по высоте, по оси столба (если Th12 не попал в кадр — «не видна»).<br>
<b>Гребни</b>: самая верхняя точка гребня подвздошной кости слева/справа на экране; если гребня нет в кадре — «не видна».<br>
Прогресс хранится в браузере. В конце — «Скачать JSON».
</div></div>
<script>
const DATA=__DATA__;
const NEW={hip:__NEW_HIP__,spine:__NEW_SPINE__};
const COLORS=['#f44','#4f4','#48f','#ff4','#f4f','#4ff','#fa4','#aaf','#faf','#afa'];
const KEY='dxa_landmarks_v2';
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let order=[...DATA.keys()], oi=0, sel=0, drag=false, S=2.6; const imgs={};
const c=document.getElementById('c'), g=c.getContext('2d');
const kind=d=>d.region==='spine'?'spine':'hip';
function cur(){const d=DATA[order[oi]]; if(!st[d.id]){const pts={},status={};
  NEW[kind(d)].forEach(n=>{pts[n]=d.init[n].slice();status[n]='todo'});
  Object.keys(d.pred).forEach(n=>{pts[n]=d.pred[n].slice();status[n]='pred'});
  st[d.id]={points:pts,status:status}} return [d,st[d.id]]}
function names(d){return [...NEW[kind(d)],...Object.keys(d.pred)]}
function done(d){const s=st[d.id]; return s && NEW[kind(d)].every(n=>s.status[n]!=='todo')}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function applyFilter(){const f=document.getElementById('flt').value;
  order=[...DATA.keys()].filter(i=>f==='all'||(f==='todo'?!done(DATA[i]):kind(DATA[i])===f)); if(!order.length)order=[...DATA.keys()]; oi=0; sel=0; render()}
function render(){const [d,s]=cur(); const im=imgs[d.id]; if(!im.complete){im.onload=render;return}
 S=Math.max(1.2,Math.min(3,(innerHeight-30)/d.h,(innerWidth-410)/d.w)); c.width=d.w*S; c.height=d.h*S; g.drawImage(im,0,0,c.width,c.height);
 const ns=names(d); const P=s.points;
 if(kind(d)==='hip'&&s.status.lt_base_top!=='invisible'){g.strokeStyle='#f84';g.lineWidth=1.5;g.beginPath();g.moveTo(P.lt_base_top[0]*S,P.lt_base_top[1]*S);g.lineTo(P.lt_apex[0]*S,P.lt_apex[1]*S);g.lineTo(P.lt_base_bottom[0]*S,P.lt_base_bottom[1]*S);g.stroke()}
 ns.forEach((n,k)=>{const stt=s.status[n]; if(stt==='invisible')return; const p=P[n]; const isNew=k<NEW[kind(d)].length;
  g.globalAlpha=isNew?1:0.55; g.fillStyle=isNew?COLORS[k%10]:'#bbb'; g.beginPath(); g.arc(p[0]*S,p[1]*S,k===sel?7:(isNew?5:3.5),0,7); g.fill();
  g.strokeStyle=stt==='todo'?'#fff':'#000'; g.setLineDash(stt==='todo'?[3,2]:[]); g.stroke(); g.setLineDash([]); g.globalAlpha=1});
 document.getElementById('title').textContent=`${order[oi]+1}/${DATA.length} · ${d.region}`;
 document.getElementById('meta').textContent=`quality=${d.quality} ${d.viol.join(',')}`;
 document.getElementById('pts').innerHTML=ns.map((n,k)=>{const isNew=k<NEW[kind(d)].length;const stt=s.status[n];
  return `<div class="pt ${k===sel?'sel':''}" data-k="${k}"><span class="sw" style="background:${isNew?COLORS[k%10]:'#777'}"></span>${k+1}. ${n}${isNew?'':' <span class=mut>(v1)</span>'}<span class="st ${stt==='pred'?'':stt}">${stt}</span></div>`}).join('');
 const nd=DATA.filter(done).length; document.querySelector('#prog div').style.width=(100*nd/DATA.length)+'%';
 document.getElementById('cnt').textContent=`готово ${nd}/${DATA.length} · бёдер ${DATA.filter(x=>kind(x)==='hip'&&done(x)).length} · позвоночников ${DATA.filter(x=>kind(x)==='spine'&&done(x)).length}`}
function place(e){const [d,s]=cur();const r=c.getBoundingClientRect();const n=names(d)[sel];s.points[n]=[(e.clientX-r.left)/S,(e.clientY-r.top)/S];s.status[n]='set';save();render()}
c.onmousedown=e=>{const [d,s]=cur();const r=c.getBoundingClientRect();const x=(e.clientX-r.left)/S,y=(e.clientY-r.top)/S;
 let best=-1,bd=1e9;names(d).forEach((n,k)=>{if(s.status[n]==='invisible')return;const p=s.points[n];const dd=Math.hypot(p[0]-x,p[1]-y);if(dd<bd){bd=dd;best=k}});
 if(bd<7){sel=best;drag=true;render()}else place(e)};
c.onmousemove=e=>{if(drag)place(e)}; onmouseup=()=>drag=false;
document.getElementById('pts').onclick=e=>{const el=e.target.closest('.pt');if(el){sel=+el.dataset.k;render()}};
function setStatus(v){const [d,s]=cur();const n=names(d)[sel];s.status[n]=v;save();
 const ns=NEW[kind(d)]; const nx=ns.findIndex(m=>s.status[m]==='todo'); if(nx>=0)sel=nx; render()}
document.getElementById('inv').onclick=()=>setStatus('invisible');
document.getElementById('ok').onclick=()=>setStatus('set');
document.getElementById('prev').onclick=()=>{oi=(oi-1+order.length)%order.length;sel=0;render()};
document.getElementById('next').onclick=()=>{oi=(oi+1)%order.length;sel=0;render()};
document.getElementById('flt').onchange=applyFilter;
document.getElementById('export').onclick=()=>{const out=DATA.filter(d=>st[d.id]).map(d=>({id:d.id,study:d.study,region:d.region,w:d.w,h:d.h,...st[d.id]}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='landmarks_manual_v2.json';a.click()};
onkeydown=e=>{if(e.key==='ArrowRight')document.getElementById('next').click();else if(e.key==='ArrowLeft')document.getElementById('prev').click();
 else if(e.key==='i'||e.key==='ш')setStatus('invisible');else if(e.key==='Enter')setStatus('set');
 else if(/^[1-9]$/.test(e.key)){sel=Math.min(+e.key-1,names(cur()[0]).length-1);render()}};
DATA.forEach(d=>{const im=new Image();im.src=d.png;imgs[d.id]=im}); onresize=render; applyFilter();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=str(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/landmark_labeling/labeler_v2.html'))
    ap.add_argument('--init-from', type=Path, help='start new points from auto labels (scripts/auto_label_lt.py)')
    args = ap.parse_args()
    rows = json.load(open(args.pred))
    auto = {r['id']: r['points'] for r in json.load(open(args.init_from))} if args.init_from else {}
    # Hips first (rotation is the priority), violations interleaved so both classes get labels early.
    rows.sort(key=lambda r: (r['region'] == 'spine', hashlib.sha1(r['image_uid'].encode()).hexdigest()))
    data = []
    for r in rows:
        img = read_dicom(DATA / r['path']).pixels
        _, buf = cv2.imencode('.png', img)
        data.append({'id': r['image_uid'], 'study': r['study'], 'region': r['region'], 'w': r['width'], 'h': r['height'],
                     'quality': r['quality'], 'viol': r['violations'], 'pred': r['points'],
                     'init': {k: [float(v[0]), float(v[1])] for k, v in
                              {**init_new(r), **{n: q for n, q in auto.get(r['image_uid'], {}).items()
                                                 if n in NEW_HIP + NEW_SPINE}}.items()},
                     'png': 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace('__DATA__', json.dumps(data)).replace('__NEW_HIP__', json.dumps(NEW_HIP)).replace('__NEW_SPINE__', json.dumps(NEW_SPINE))
    out.write_text(html, encoding='utf-8')
    print(f'{len(data)} images -> {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
