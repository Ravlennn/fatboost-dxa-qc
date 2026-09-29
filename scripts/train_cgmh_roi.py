"""External-only femur segmentation, followed by fixed DXA ROI inference. Low load."""
import json
import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from dxaqc.anatomy_roi import FemurUNet, letterbox, roi_box


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42); np.random.seed(42)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    data=ROOT/'data/external/cgmh/prepared'; out=ROOT/'outputs/cgmh_roi_v1'
    if (out/'config.json').exists(): raise ValueError('Run already exists')
    out.mkdir(parents=True,exist_ok=True)
    config=dict(seed=42,epochs=30,batch_size=12,res=256,device=device,cpu_threads=2,batch_pause=1,
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(config,indent=2))
    records=json.loads((data/'manifest.json').read_text()); arrays=np.load(data/'arrays.npz')
    x,y=arrays['images'],arrays['masks']
    class DS(Dataset):
        def __init__(self,split): self.ids=[i for i,r in enumerate(records) if r['split']==split]; self.train=split=='train'
        def __len__(self): return len(self.ids)
        def __getitem__(self,i):
            j=self.ids[i]; a=x[j]; b=y[j]
            # Single-side views bridge bilateral radiographs and unilateral DXA.
            if self.train and torch.rand(())<.5:
                start=0 if torch.rand(())<.5 else 128
                a,_=letterbox(a[:,start:start+128]); b,_=letterbox(b[:,start:start+128],mask=True)
            a=torch.from_numpy(a.copy()).float()[None]/255; b=torch.from_numpy(b.copy()).float()[None]/255
            if self.train:
                if torch.rand(())<.5: a=a.flip(-1); b=b.flip(-1)
                angle=float(torch.rand(())*16-8)
                a=TF.rotate(a,angle); b=TF.rotate(b,angle)
                a=(a*(.8+.4*torch.rand(()))).clamp(0,1)
            return a,b
    tr=DataLoader(DS('train'),batch_size=12,shuffle=True,num_workers=0)
    va=DataLoader(DS('validation'),batch_size=12,num_workers=0)
    model=FemurUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    def rest():
        if device=='mps': torch.mps.synchronize()
        time.sleep(1)
    history=[]; best=-1
    for ep in range(30):
        model.train(); total=0
        for a,b in tr:
            a,b=a.to(device),b.to(device); logits=model(a); p=logits.sigmoid()
            dice=(2*(p*b).sum((1,2,3))+1)/(p.sum((1,2,3))+b.sum((1,2,3))+1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b)+(1-dice).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(a); rest()
        model.eval(); scores=[]
        with torch.no_grad():
            for a,b in va:
                p=(model(a.to(device)).sigmoid().cpu()>=.5).float()
                scores.extend(((2*(p*b).sum((1,2,3))+1)/(p.sum((1,2,3))+b.sum((1,2,3))+1)).tolist()); rest()
        score=float(np.mean(scores)); row=dict(epoch=ep+1,train_loss=total/len(tr.dataset),validation_dice=score)
        history.append(row); print(row,flush=True)
        if score>best:
            best=score; torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
    report=f"# CGMH femur segmentation\n\nCommit: {config['commit']}; seed=42.\nBest validation Dice: {best:.4f}.\nImage-level external validation used for checkpoint selection, not an independent test.\nNot a DXA quality metric; patient IDs unavailable.\n"
    (out/'metrics.md').write_text(report)
    with (ROOT/'docs/EXPERIMENTS.md').open('a') as f: f.write('\n'+report)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)); model.eval()
    labels=pd.read_csv(ROOT/'data/interim/image_labels.csv'); rows=[]; previews=[]
    # No DXA quality labels or folds are read by segmenter fitting/selection.
    for r in labels[labels.region!='spine'].itertuples():
        a=pydicom.dcmread(ROOT/'data/interim/train/Исследования'/r.path).pixel_array
        if r.region=='hip_left': a=np.ascontiguousarray(a[:,::-1])
        small,(top,left,nh,nw)=letterbox(a)
        with torch.no_grad(): p=model(torch.tensor(small,dtype=torch.float32,device=device)[None,None]/255).sigmoid()[0,0].cpu().numpy()
        p=np.array(Image.fromarray(p[top:top+nh,left:left+nw]).resize((a.shape[1],a.shape[0]),Image.Resampling.BILINEAR))
        box,status=roi_box(p)
        rows.append(dict(image_uid=r.image_uid,box=list(map(int,box)),status=status))
        if len(previews)<36:
            im=Image.fromarray(a).convert('RGB'); ImageDraw.Draw(im).rectangle(box,outline='red',width=3)
            im.thumbnail((192,192)); previews.append(im)
        rest()
    (out/'dxa_rois.json').write_text(json.dumps(rows,indent=2))
    canvas=Image.new('RGB',(192*6,192*6))
    for i,im in enumerate(previews): canvas.paste(im,(i%6*192,i//6*192))
    canvas.save(out/'dxa_roi_review.jpg')
    print('DXA ROI candidates ready for visual review',flush=True)

if __name__=='__main__': main()
