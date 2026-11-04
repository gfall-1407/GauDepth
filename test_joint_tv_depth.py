"""Run with CUDA on the experiment host: python -m unittest test_joint_tv_depth -v."""
import math
import unittest
import torch
from joint_tv_depth import (joint_depth, continuous_sep_depth, ray_profiles,
                            sep_optical_depth, profile_knots, potential)


class JointTVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Run numerical tests on the authorized GPU server")
        cls.device = torch.device("cuda")

    def p(self, m, s, a, grad=False):
        return tuple(torch.tensor([x], device=self.device, dtype=torch.float64,
                                  requires_grad=grad) for x in (m,s,a))

    def solve(self, p, n=8):
        return joint_depth(*p, near=-10, far=10, subdivisions=n)

    def test_single_and_aligned_equal_sep(self):
        for p in (self.p([0.], [1.], [.9]), self.p([0.,0.], [1.,1.], [.6,.4])):
            r = self.solve(p)
            s,v = continuous_sep_depth(*p,near=-10,far=10)
            self.assertTrue(bool(r.valid.all() & v.all()))
            self.assertLess(float((r.depth-s).abs().max()), 8e-4)

    def test_recovery_after_first_peak(self):
        p = self.p([-.3,.3], [1.,1.], [.53,.53])
        r = self.solve(p)
        s,_ = continuous_sep_depth(*p,near=-10,far=10)
        self.assertTrue(bool(r.valid.all()))
        self.assertTrue(-.3 < float(r.depth) < 0)
        self.assertGreater(float(r.depth), float(s))
        h = potential(r.depth[:,None],*p)
        self.assertLess(float((h-math.log(2)).abs().max()), 2e-4)

    def test_true_no_root_not_solver_failure(self):
        p = self.p([-.3,.3], [1.,1.], [.30,.30])
        r = self.solve(p)
        _,sv = continuous_sep_depth(*p,near=-10,far=10)
        self.assertTrue(bool(sv.all()))
        self.assertFalse(bool(r.has_sampled_root.any()))
        self.assertEqual(float(r.depth), 0.)

    def test_front_layer_history_retained(self):
        p = self.p([-3.,3.], [.2,.2], [.4,.9])
        r = self.solve(p)
        isolated = self.solve(self.p([3.],[.2],[.9]))
        self.assertTrue(bool(r.valid.all()))
        self.assertLess(float(r.depth), float(isolated.depth))
        # With separated layers cancellation is negligible, joint equals sep.
        sd,_ = continuous_sep_depth(*p,near=-10,far=10)
        self.assertLess(float((r.depth-sd).abs()), 3e-4)

    def test_refinement_and_total_inequality(self):
        p = self.p([-.5,.1,1.2],[.7,.4,.6],[.5,.7,.6])
        r1,r2,r4 = (self.solve(p,n) for n in (1,2,4))
        sep = sep_optical_depth(p[0].new_tensor([10.]),*p,near=-10)
        self.assertTrue(bool((r1.total <= r2.total+1e-12).all()))
        self.assertTrue(bool((r2.total <= r4.total+1e-12).all()))
        self.assertTrue(bool((r4.total <= sep+1e-12).all()))
        self.assertLess(float((r4.depth-r2.depth).abs()), .003)

    def test_gradcheck_actual_adaptive_grid(self):
        # Includes differentiable Gaussian-dependent knots, sorting, and root.
        p = self.p([-.31,.27],[.91,1.03],[.57,.52],grad=True)
        fn = lambda *x: self.solve(x,2).depth
        self.assertTrue(bool(self.solve(p,2).valid.all()))
        self.assertGreater(float(torch.autograd.grad(fn(*p).sum(),p[0])[0].abs().sum()), .1)
        self.assertTrue(torch.autograd.gradcheck(fn,p,eps=1e-6,atol=2e-5,rtol=2e-4))
        r32=self.solve(tuple(x.detach().float() for x in p),2)
        self.assertTrue(bool(r32.valid.all()))
        self.assertLess(float((r32.depth.double()-fn(*p)).abs()), 3e-6)

    def test_full_covariance_chain_gradient(self):
        p = [torch.tensor(x, device=self.device,dtype=torch.float64,requires_grad=True)
             for x in ([[0.1,0.0,2.]], [[.3,.5,.2]], [[.97,.1,.04,.12]], [[.9]])]
        o = torch.zeros(1,3,device=self.device,dtype=torch.float64)
        d = o.new_tensor([[0.,0.,1.]])
        fn = lambda *x: joint_depth(*ray_profiles(o,d,*x),near=.1,far=4.,subdivisions=2).depth
        self.assertTrue(bool(joint_depth(*ray_profiles(o,d,*p),near=.1,far=4.,subdivisions=2).valid.all()))
        self.assertGreater(float(torch.autograd.grad(fn(*p).sum(),p[0])[0].abs().sum()), .1)
        self.assertTrue(torch.autograd.gradcheck(fn,tuple(p),eps=1e-6,atol=3e-5,rtol=3e-4))

    def test_invalid_near_and_gradient_guard(self):
        p = self.p([0.],[1.],[.9],grad=True)
        near = joint_depth(*p,near=-.1,far=10.)
        self.assertFalse(bool(near.valid.any()))
        guard = joint_depth(*p,near=-10,far=10,min_slope=100.)
        self.assertTrue(bool(guard.has_sampled_root.all()))
        self.assertFalse(bool(guard.valid.any()))
        guard.depth.sum().backward()
        for x in p:
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertEqual(float(x.grad.abs().sum()),0.)

    def test_continuous_turning_root_rejected(self):
        r = self.solve(self.p([0.],[1.],[.75]))
        self.assertTrue(bool(r.has_sampled_root.all()))
        self.assertFalse(bool(r.valid.any()))
        self.assertLess(float(r.analytic_slope), 1e-10)

    def test_empty_profile_and_amplitude_boundary(self):
        r = self.solve(self.p([0.],[1.],[0.]))
        self.assertFalse(bool(r.valid.any()))
        with self.assertRaises(ValueError):
            self.solve(self.p([0.],[1.],[1.]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
