"""CUDA support/index guards; no real model or filesystem writes."""
from types import SimpleNamespace
import math
import unittest
import torch
from joint_tv_model import select_members,build_support_index,rays_for_pixels,candidate_depths


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')

    def model(self):
        t=lambda x:torch.tensor(x,device='cuda',dtype=torch.float64)
        return SimpleNamespace(_xyz=t([[.3,-.1,2.],[-.2,.15,-2.],[0,0,.01],[-1,.3,4.]]),
            _scaling=t([[.1,.2,.15],[.2,.1,.4],[.4,.2,.2],[.3,.4,.2]]).log(),
            _rotation=t([[.9,.3,.4,.2]]*4),_opacity=t([[1.5],[1.2],[2.],[.8]]),filter_3D=t([[.001]]*4))

    def camera(self):
        return SimpleNamespace(camera_center=torch.zeros(3,device='cuda'),
            R=torch.diag(torch.tensor([1.00001,.99999,1.],device='cuda')),
            Fx=400.,Fy=500.,Cx=319.5,Cy=239.5)

    def test_conservative_front_back_and_zcrossing(self):
        g=self.model();cam=self.camera()
        xx,yy=torch.meshgrid(torch.tensor([0.,159.75,319.5,479.25,639.],device='cuda'),
                            torch.tensor([0.,119.75,239.5,359.25,479.],device='cuda'),indexing='xy')
        xy=torch.stack((xx.flatten(),yy.flatten()),-1);o,d=rays_for_pixels(cam,xy)
        for epsilon in (1e-6,1e-8):
            index=build_support_index(g,cam,epsilon)
            reference,_=select_members(g,o,d,epsilon=epsilon)
            for i,(u,v) in enumerate(xy.tolist()):
                ids=index.query(u,u,v,v)
                actual,_=select_members(g,o[i:i+1],d[i:i+1],epsilon=epsilon,candidate_ids=ids)
                self.assertTrue(torch.equal(reference[i],actual[0]))
            self.assertIn(1,reference[12].tolist()) # Behind camera must not be culled.
            self.assertTrue(bool(torch.isinf(index.xmin[2])&torch.isinf(index.xmax[2])))

    def test_overflow_does_not_understate_empty_support_bound(self):
        g=self.model();cam=self.camera();o,d=rays_for_pixels(cam,[[319.5,239.5]])
        full,normal=select_members(g,o,d,epsilon=1e-6,max_members=1024)
        empty,overflow=select_members(g,o,d,epsilon=1e-6,max_members=1)
        self.assertGreater(len(full[0]),1);self.assertEqual(len(empty[0]),0)
        self.assertTrue(bool(overflow['overflow'][0]))
        self.assertGreater(float(overflow['omitted_TV_upper_bound'][0]),float(normal['omitted_TV_upper_bound'][0]))

    def test_zero_quaternion_rejected_before_support_culling(self):
        g=self.model();g._rotation[0]=0;cam=self.camera();o,d=rays_for_pixels(cam,[[319.5,239.5]])
        with self.assertRaises(ValueError):select_members(g,o,d)
        with self.assertRaises(ValueError):build_support_index(g,cam)

    def test_empty_support_saturated_opacity_has_finite_zero_gradients(self):
        for sign in (1.,-1.):
            g=self.model();g._opacity[0]=sign*float('inf')
            raw=tuple(getattr(g,n).requires_grad_() for n in ('_xyz','_scaling','_rotation','_opacity'))
            cam=self.camera();o,d=rays_for_pixels(cam,[[319.5,239.5]])
            empty=torch.empty(0,device='cuda',dtype=torch.long)
            depth,valid,diag=candidate_depths(g,o,d,candidate_ids=empty)
            self.assertEqual(float(depth),0.);self.assertFalse(bool(valid))
            self.assertTrue(diag['convergence'][0])
            for grad in torch.autograd.grad(depth.sum(),raw):
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertEqual(int(torch.count_nonzero(grad)),0)


if __name__=='__main__':unittest.main(verbosity=2)
