"""Audit original 400 CGMH image/mask pairs, deduplicate, freeze external split."""
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from dxaqc.anatomy_roi import letterbox


def main():
    os.nice(10)
    folder = ROOT/'data/external/cgmh'; out=folder/'prepared'
    if out.exists(): raise ValueError('Prepared data already exists; refusing overwrite')
    with zipfile.ZipFile(folder/'dataset.zip') as z:
        names=z.namelist(); images=sorted(n for n in names if '/CGMH_PelvisSegment/Image/' in n and n.endswith('.png'))
        masks={n for n in names if '/CGMH_PelvisSegment/Label/' in n and n.endswith('.png')}
        if len(images)!=400 or len(masks)!=400: raise ValueError('Unexpected original collection size')
        xs=[]; ys=[]; rows=[]; seen={}; duplicates=[]; previews=[]
        for name in images:
            label=name.replace('/Image/','/Label/')
            if label not in masks: raise ValueError(f'Missing mask: {name}')
            im=Image.open(io.BytesIO(z.read(name))); mask=Image.open(io.BytesIO(z.read(label)))
            original=np.array(im)
            if original.ndim==2 and original.dtype!=np.uint8:
                lo,hi=float(original.min()),float(original.max())
                if hi<=lo: raise ValueError('Constant image')
                a=np.round((original.astype(np.float32)-lo)/(hi-lo)*255).astype(np.uint8)
            else: a=np.array(im.convert('L'))
            m=np.array(mask)
            if a.shape != m.shape or not set(np.unique(m)) <= {0,255}: raise ValueError('Invalid mask geometry/values')
            if not (m>0).any(): raise ValueError('Empty ground truth')
            digest=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
            md=hashlib.sha256(m.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest]!=md: raise ValueError('Duplicate image has conflicting masks')
                duplicates.append(name); continue
            seen[digest]=md
            small,_=letterbox(a); target,_=letterbox(m,mask=True)
            xs.append(small); ys.append(target)
            rows.append(dict(name=name,sha256=digest,mask_sha256=md,shape=list(a.shape),area=float((m>0).mean())))
            if len(previews)<12:
                rgb=np.stack([small]*3,-1); rgb[target>0,0]=255
                previews.append(Image.fromarray(rgb))
        order=np.random.default_rng(42).permutation(len(rows)); val=set(order[:max(1,round(len(rows)*.2))].tolist())
        for i,r in enumerate(rows): r['split']='validation' if i in val else 'train'
        out.mkdir()
        np.savez_compressed(out/'arrays.npz',images=np.stack(xs),masks=np.stack(ys))
        (out/'manifest.json').write_text(json.dumps(rows,indent=2))
        archive_hash=hashlib.file_digest((folder/'dataset.zip').open('rb'),'sha256').hexdigest()
        summary=dict(archive_sha256=archive_hash,pairs=len(images),unique=len(rows),duplicates=duplicates,
                     split=dict(Counter(r['split'] for r in rows)),seed=42,
                     limitations=['Image-level split; patient IDs unavailable','Only original CGMH 400; extra 140 excluded',
                                  'License metadata conflict: research/education use only pending clarification',
                                  'External radiograph segmentation validation is not DXA QC validation'])
        (out/'audit.json').write_text(json.dumps(summary,indent=2))
        montage=Image.new('RGB',(4*256,3*256))
        for i,p in enumerate(previews): montage.paste(p,(i%4*256,i//4*256))
        montage.save(out/'mask_audit.jpg')
        print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
