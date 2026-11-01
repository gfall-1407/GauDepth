"""Numerical controls for fixed smooth-branch photo derivative and missing mass."""
import unittest

import numpy as np
import torch

from local_plane_direction_photo_cpu import (DT, PATCH, bilinear_np, photo_cost_torch,
    photo_projection_np, planes_torch)
from run_local_plane_direction_photo_cpu import frozen_photo_rows


CAM = dict(R=np.eye(3), t=np.zeros(3), center=np.zeros(3),
           fx=1000., fy=1000., cx=500., cy=500.)


class Controls(unittest.TestCase):
    def test_nonzero_smooth_photo_fd(self):
        rng=np.random.default_rng(20261101)
        row=dict(index=0,axis=2,xyz=np.array([0.,0.,2.]),quat=np.array([1.,0.,0.,0.]),
                 dmu=np.array([0.,.01,.03]),dq=np.array([0.,.01,0.,0.]),
                 uv=np.array([500.,500.])+PATCH,ix=np.full(25,500),iy=np.full(25,500))
        src=rng.random(25)
        corner=rng.random((25,4))
        dst=dict(CAM,t=np.array([.1,0.,0.]))
        eps=torch.zeros(1,dtype=DT,requires_grad=True)
        val=photo_cost_torch(eps,[row],CAM,dst,[src],[corner])[0]
        ad=float(torch.autograd.grad(val,eps)[0][0])
        with torch.no_grad():
            plus=float(photo_cost_torch(torch.tensor([1e-6],dtype=DT),[row],CAM,dst,[src],[corner])[0])
            minus=float(photo_cost_torch(torch.tensor([-1e-6],dtype=DT),[row],CAM,dst,[src],[corner])[0])
        fd=(plus-minus)/2e-6
        self.assertGreater(abs(ad),1e-8)
        self.assertLessEqual(abs(ad-fd),1e-7+1e-4*max(abs(ad),abs(fd)))

    def test_missing_mass_kept(self):
        occ=[dict(p=p,gid=7,index=0,axis=2,m=1/32,center=np.array([500.,500.])) for p in range(32)]
        direction=dict(occ=occ,xyz=np.array([[0.,0.,2.]]),quat=np.array([[1.,0.,0.,0.]]),
                       dmu=np.zeros((1,3)),dq=np.zeros((1,4)))
        blank=np.zeros((1162,1554),dtype=np.float64)
        rows,eligible,_,_=frozen_photo_rows(direction,CAM,CAM,blank,blank)
        self.assertEqual(len(rows),32)
        self.assertFalse(eligible)
        self.assertAlmostEqual(sum(x["mass"] for x in rows),1.)
        self.assertEqual({x["reason"] for x in rows},{"low_source_texture"})
        rng=np.random.default_rng(42)
        textured=rng.random((1162,1554))
        far=dict(CAM,t=np.array([100.,0.,0.]))
        rows,eligible,_,_=frozen_photo_rows(direction,CAM,far,textured,textured)
        self.assertFalse(eligible)
        self.assertEqual({x["reason"] for x in rows},{"out_of_bounds"})
        self.assertAlmostEqual(sum(x["mass"] for x in rows),1.)

    def test_plane_projection_and_boundary(self):
        pi=planes_torch(torch.tensor([[0.,0.,2.]],dtype=DT),
            torch.tensor([[1.,0.,0.,0.]],dtype=DT),np.array([2]),CAM["R"],CAM["t"])
        self.assertTrue(torch.allclose(pi,torch.tensor([[0.,0.,.5]],dtype=DT)))
        dst=dict(CAM,t=np.array([.1,0.,0.]))
        xy=photo_projection_np(pi.numpy()[0],CAM,dst,np.array([[500.,500.]]))
        self.assertAlmostEqual(xy[0,0],550.)
        with self.assertRaises(ValueError):
            bilinear_np(np.zeros((3,3)),np.array([[2.,1.]]))


if __name__=="__main__":unittest.main()
