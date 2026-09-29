import unittest
import numpy as np
import torch
from dxaqc.anatomy_roi import FemurUNet, letterbox, roi_box


class AnatomyROITest(unittest.TestCase):
    def test_empty_or_implausible_fallback(self):
        self.assertEqual(roi_box(np.zeros((50,80))),((0,0,80,50),'empty_mask'))
        self.assertEqual(roi_box(np.ones((50,80)))[1],'implausible_area')

    def test_largest_component_margin_and_bounds(self):
        p=np.zeros((100,100)); p[20:70,30:60]=1; p[0:2,0:2]=1
        box,status=roi_box(p,.1)
        self.assertEqual(box,(27,15,63,75)); self.assertEqual(status,'mask_roi')

    def test_mask_geometry(self):
        a=np.zeros((120,60),np.uint8); a[20:80,10:40]=255
        b,(_,_,h,w)=letterbox(a,256,True)
        self.assertEqual((h,w),(256,128)); self.assertEqual(set(np.unique(b)),{0,255})

    def test_model_backward(self):
        torch.set_num_threads(2)
        m=FemurUNet(); x=torch.rand(2,1,64,64); y=m(x)
        self.assertEqual(y.shape,x.shape); y.square().mean().backward()
        self.assertTrue(torch.isfinite(m.enc1[0].weight.grad).all())

if __name__=='__main__': unittest.main()
