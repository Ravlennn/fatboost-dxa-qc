"""Experimental external-data femur segmenter; not a quality classifier."""
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn


class FemurUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU(),
                                 nn.Conv2d(b,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU())
        self.enc1 = block(1,16); self.enc2 = block(16,32); self.enc3 = block(32,64)
        self.mid = block(64,128); self.dec3 = block(192,64)
        self.dec2 = block(96,32); self.dec1 = block(48,16); self.head = nn.Conv2d(16,1,1)

    def forward(self, x):
        a = self.enc1(x); b = self.enc2(nn.functional.max_pool2d(a,2))
        c = self.enc3(nn.functional.max_pool2d(b,2)); d = self.mid(nn.functional.max_pool2d(c,2))
        def up(x, skip):
            return torch.cat([nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False),skip],1)
        return self.head(self.dec1(up(self.dec2(up(self.dec3(up(d,c)),b)),a)))


def letterbox(a, size=256, mask=False):
    h,w = a.shape; scale = size/max(h,w)
    nh,nw = max(1,round(h*scale)),max(1,round(w*scale))
    top,left = (size-nh)//2,(size-nw)//2
    resized = np.array(Image.fromarray(a).resize((nw,nh), Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR))
    out = np.zeros((size,size), dtype=a.dtype); out[top:top+nh,left:left+nw] = resized
    return out, (top,left,nh,nw)


def roi_box(prob, margin=0.15):
    """Largest component; conservative full-image fallback. No QC labels used."""
    h,w = prob.shape
    labels,n = ndimage.label(prob >= 0.5)
    if n == 0: return (0,0,w,h), 'empty_mask'
    counts = np.bincount(labels.ravel()); counts[0] = 0
    component = labels == counts.argmax(); area = component.mean()
    if area < .015 or area > .8: return (0,0,w,h), 'implausible_area'
    yy,xx = np.where(component); x0,x1,y0,y1 = xx.min(),xx.max()+1,yy.min(),yy.max()+1
    dx,dy = max(1,round((x1-x0)*margin)),max(1,round((y1-y0)*margin))
    return (max(0,x0-dx),max(0,y0-dy),min(w,x1+dx),min(h,y1+dy)), 'mask_roi'
