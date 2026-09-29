"""Build a self-contained HTML tool to correct predicted landmarks on our DICOMs.

Points are prefilled from the model (predictions.json); the annotator drags
them into place and exports JSON. Used only to measure transfer Arak -> ours.

  .venv/bin/python scripts/make_landmark_labeler.py --n 30
  open outputs/landmark_labeling/labeler.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.dicom_io import read_dicom  # noqa: E402

DATA = ROOT / 'data/interim/train/Исследования'

HTML = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Разметка ориентиров DXA</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--acc:#4af}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;gap:16px;padding:12px}
#side{width:340px;flex:none} canvas{background:#000;cursor:crosshair;image-rendering:pixelated;border:1px solid #333}
button{background:#222;color:var(--fg);border:1px solid #444;padding:6px 10px;border-radius:4px;cursor:pointer;margin:2px}
button:hover{border-color:var(--acc)} .pt{display:flex;align-items:center;gap:6px;margin:2px 0;cursor:pointer;padding:2px 4px;border-radius:3px}
.pt.sel{background:#243} .sw{width:12px;height:12px;border-radius:50%} .mut{color:var(--mut);font-size:12px;line-height:1.4}
#prog{height:6px;background:#333;border-radius:3px;margin:8px 0} #prog div{height:100%;background:var(--acc);border-radius:3px}
</style></head><body>
<div><canvas id="c"></canvas></div>
<div id="side">
<h3 style="margin:4px 0" id="title"></h3><div class="mut" id="meta"></div>
<div id="prog"><div></div></div>
<div id="pts"></div>
<p><label><input type="checkbox" id="bad"> Снимок нельзя разметить</label></p>
<p><label><input type="checkbox" id="done"> Проверено</label></p>
<div><button id="prev">← Пред.</button><button id="next">След. →</button><button id="reset">Сбросить к модели</button></div>
<div><button id="export">Скачать JSON</button></div>
<div class="mut" style="margin-top:12px">
<b>Как работать.</b> Точки уже поставлены моделью: перетащите неверные. Клик по пустому месту двигает выбранную точку (выбор — в списке или клавишами 1–7). Отметьте «Проверено» и нажмите → (клавиши ←/→, D — проверено). Всё сохраняется в браузере; в конце — «Скачать JSON».<br><br>
<b>Позвоночник:</b> пять точек на границах тел позвонков по центру столба: верх L1, L1/L2, L2/L3, L3/L4, низ L4.<br>
<b>Бедро:</b> центр шейки — середина самой узкой части шейки; две точки оси шейки — по оси шейки к головке и к большому вертелу (~18 мм от центра). ROI (верх/низ/лево/право) — края, где стандартно ставится рамка анализа total hip; если не уверены — не трогайте их и отметьте в списке «не уверен».
</div></div>
<script>
const DATA=__DATA__;
const COLORS=['#f44','#4f4','#48f','#ff4','#f4f','#4ff','#fa4'];
const KEY='dxa_landmarks_v1';
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0, sel=0, drag=false, S=2.6;
const c=document.getElementById('c'), g=c.getContext('2d'); const imgs={};
function cur(){const d=DATA[i]; if(!st[d.id]) st[d.id]={points:JSON.parse(JSON.stringify(d.pred)),unsure:{},bad:false,done:false}; return [d,st[d.id]]}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function render(){const [d,s]=cur(); const im=imgs[d.id]; if(!im.complete){im.onload=render;return}
 S=Math.max(1.2,Math.min(3,(window.innerHeight-30)/d.h,(window.innerWidth-380)/d.w)); c.width=d.w*S;c.height=d.h*S; g.imageSmoothingEnabled=false; g.drawImage(im,0,0,c.width,c.height);
 const names=Object.keys(s.points); g.lineWidth=1.5;
 if(d.region==='spine'){g.strokeStyle='#ff0a';g.beginPath();names.forEach((n,k)=>{const p=s.points[n];k?g.lineTo(p[0]*S,p[1]*S):g.moveTo(p[0]*S,p[1]*S)});g.stroke()}
 else{const P=s.points;g.strokeStyle='#fa08';g.strokeRect(P.roi_left[0]*S,P.roi_top[1]*S,(P.roi_right[0]-P.roi_left[0])*S,(P.roi_bottom[1]-P.roi_top[1])*S);
  g.strokeStyle='#f44';g.beginPath();g.moveTo(P.neck_head_side[0]*S,P.neck_head_side[1]*S);g.lineTo(P.neck_troch_side[0]*S,P.neck_troch_side[1]*S);g.stroke()}
 names.forEach((n,k)=>{const p=s.points[n];g.fillStyle=COLORS[k%7];g.beginPath();g.arc(p[0]*S,p[1]*S,k===sel?7:5,0,7);g.fill();g.strokeStyle='#000';g.stroke()});
 document.getElementById('title').textContent=`${i+1}/${DATA.length} · ${d.region}`;
 document.getElementById('meta').textContent=`${d.id.slice(-24)} · quality=${d.quality} ${d.viol.join(',')}`;
 document.getElementById('pts').innerHTML=names.map((n,k)=>`<div class="pt ${k===sel?'sel':''}" data-k="${k}"><span class="sw" style="background:${COLORS[k%7]}"></span>${k+1}. ${n}${n.startsWith('roi')?` <label class="mut"><input type="checkbox" data-u="${n}" ${s.unsure[n]?'checked':''}> не уверен</label>`:''}</div>`).join('');
 document.getElementById('bad').checked=s.bad; document.getElementById('done').checked=s.done;
 const nd=DATA.filter(x=>st[x.id]&&st[x.id].done).length; document.querySelector('#prog div').style.width=(100*nd/DATA.length)+'%';
}
function setPt(e){const [d,s]=cur();const r=c.getBoundingClientRect();const n=Object.keys(s.points)[sel];s.points[n]=[(e.clientX-r.left)/S,(e.clientY-r.top)/S];save();render()}
c.onmousedown=e=>{const [d,s]=cur();const r=c.getBoundingClientRect();const x=(e.clientX-r.left)/S,y=(e.clientY-r.top)/S;
 let best=-1,bd=1e9;Object.values(s.points).forEach((p,k)=>{const dd=Math.hypot(p[0]-x,p[1]-y);if(dd<bd){bd=dd;best=k}});
 if(bd<8){sel=best;drag=true;render()}else setPt(e)};
c.onmousemove=e=>{if(drag)setPt(e)}; window.onmouseup=()=>drag=false;
document.getElementById('pts').onclick=e=>{const u=e.target.dataset.u;if(u){const[d,s]=cur();s.unsure[u]=e.target.checked;save();return}
 const el=e.target.closest('.pt');if(el){sel=+el.dataset.k;render()}};
document.getElementById('bad').onchange=e=>{cur()[1].bad=e.target.checked;save()};
document.getElementById('done').onchange=e=>{cur()[1].done=e.target.checked;save();render()};
document.getElementById('prev').onclick=()=>{i=(i-1+DATA.length)%DATA.length;sel=0;render()};
document.getElementById('next').onclick=()=>{i=(i+1)%DATA.length;sel=0;render()};
document.getElementById('reset').onclick=()=>{delete st[DATA[i].id];save();render()};
document.getElementById('export').onclick=()=>{const out=DATA.filter(d=>st[d.id]).map(d=>({id:d.id,region:d.region,w:d.w,h:d.h,pred:d.pred,...st[d.id]}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='landmarks_manual.json';a.click()};
window.onkeydown=e=>{if(e.key==='ArrowRight')document.getElementById('next').click();else if(e.key==='ArrowLeft')document.getElementById('prev').click();
 else if(e.key==='d'||e.key==='в'){const s=cur()[1];s.done=!s.done;save();render()} else if(/^[1-7]$/.test(e.key)){sel=Math.min(+e.key-1,Object.keys(cur()[1].points).length-1);render()}};
DATA.forEach(d=>{const im=new Image();im.src=d.png;imgs[d.id]=im}); window.onresize=render; render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=str(ROOT / 'outputs/landmarks_v1/ours/predictions.json'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/landmark_labeling/labeler.html'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rows = json.load(open(args.pred))
    rng = np.random.default_rng(args.seed)
    # Stratified: equal thirds spine / right hip / left hip, half of each with a violation.
    pick, per = [], args.n // 3
    for reg in ('spine', 'hip_right', 'hip_left'):
        for q, k in ((1, per // 2), (0, per - per // 2)):
            pool = [r for r in rows if r['region'] == reg and r['quality'] == q]
            pick += [pool[j] for j in rng.choice(len(pool), min(k, len(pool)), replace=False)]
    data = []
    for r in pick:
        img = read_dicom(DATA / r['path']).pixels
        ok, buf = cv2.imencode('.png', img)
        data.append({'id': r['image_uid'], 'region': r['region'], 'w': r['width'], 'h': r['height'],
                     'quality': r['quality'], 'viol': r['violations'], 'pred': r['points'],
                     'png': 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace('__DATA__', json.dumps(data)), encoding='utf-8')
    (out.parent / 'selection.json').write_text(json.dumps([d['id'] for d in data], indent=1))
    print(f'{len(data)} images -> {out}')


if __name__ == '__main__':
    main()
