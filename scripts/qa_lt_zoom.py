"""Zoomed QA sheet of automatic lesser-trochanter detections (no quality labels shown).

  python scripts/qa_lt_zoom.py START COUNT out.jpg
"""
import json, numpy as np, cv2, sys
from pathlib import Path
root=str(Path(__file__).resolve().parents[1])
sys.path.insert(0,root+'/src'); sys.path.insert(0,root+'/scripts')
from dxaqc.dicom_io import read_dicom
import auto_label_lt as A
R=[r for r in json.load(open(root+'/outputs/landmarks_v1/ours/predictions.json')) if r['region']!='spine']
start,n=int(sys.argv[1]),int(sys.argv[2])
tiles=[]
for i in range(start,min(start+n,len(R))):
    r=R[i]; img=read_dicom(root+'/data/interim/train/Исследования/'+r['path']).pixels
    res,_=A.detect(img,r['points']); W=img.shape[1]
    if res['mirror']: img=img[:,::-1].copy()
    c=np.array(r['points']['neck_center']); cx=W-1-c[0] if res['mirror'] else c[0]
    x0=int(max(cx-40,0)); y0=int(c[1]); crop=img[y0:y0+110,x0:x0+90]
    v=cv2.cvtColor(cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC),cv2.COLOR_GRAY2BGR)
    col=(0,255,0) if res['visible'] else (0,0,255)
    for k in ('lt_base_top','lt_apex','lt_base_bottom'):
        q=res['points'][k]; qx=(W-1-q[0] if res['mirror'] else q[0])-x0
        cv2.circle(v,(int(qx*3),int((q[1]-y0)*3)),5,col,2)
    cv2.putText(v,f"#{i} {'LT' if res['visible'] else 'none'} {res['lt_prominence_mm']:.1f}",(4,18),0,0.6,(0,255,255),2)
    tiles.append(v)
while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[j:j+8]) for j in range(0,len(tiles),8)]))
