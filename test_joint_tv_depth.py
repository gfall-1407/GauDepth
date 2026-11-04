"""Run with CUDA on the experiment host: python -m unittest test_joint_tv_depth -v."""
import math
import os
import unittest
import torch
from joint_tv_depth import (joint_depth, continuous_sep_depth, ray_profiles,
                            sep_optical_depth, profile_knots, potential)
from joint_tv_convergence import gradient_delta, refinement_comparison


class JointTVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CPU is explicit and only for this tiny synthetic suite during review.
        # It is never a fallback for the full-scene GPU probe.
        device = os.environ.get('JOINT_TEST_DEVICE', 'cuda')
        if device not in ('cpu', 'cuda'):
            raise ValueError('JOINT_TEST_DEVICE must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("Run numerical tests on the authorized GPU server")
        cls.device = torch.device(device)
        if device == 'cpu':
            torch.set_num_threads(1)

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
        r8,r = self.solve(p,8),self.solve(p,16)
        isolated = self.solve(self.p([3.],[.2],[.9]),16)
        self.assertTrue(bool(r.valid.all()))
        self.assertLess(float(r.depth), float(isolated.depth))
        # With separated layers cancellation is negligible, joint equals sep.
        sd,_ = continuous_sep_depth(*p,near=-10,far=10)
        self.assertLess(float((r.depth-sd).abs()), float((r8.depth-sd).abs()))
        self.assertLess(float((r.depth-sd).abs()), 3e-4)

    def test_front_layer_continuous_gradient_convergence(self):
        # Well-separated layers have an analytic root. Their overlap is <1e-48.
        # This checks accuracy against a continuous reference, not just gradcheck
        # of the same interpolated algorithm. Preserve the failing low-res case.
        p = self.p([-3.,3.],[.2,.2],[.4,.9],grad=True)
        m,s,a = p
        back_g = 1 - (0.5/(1-a[:,0])).square()
        exact = m[:,1]-s[:,1]*torch.sqrt(2*torch.log(a[:,1]/back_g))
        exact_grad = torch.autograd.grad(exact.sum(),p)
        opacity_errors=[]
        depth_errors=[]
        for n in (2,4,8,16):
            x=tuple(v.detach().clone().requires_grad_() for v in p)
            r=self.solve(x,n)
            self.assertTrue(bool(r.valid.all()))
            grad=torch.autograd.grad(r.depth.sum(),x)
            opacity_errors.append(float((grad[2]-exact_grad[2]).abs().max()))
            depth_errors.append(float((r.depth-exact).abs()))
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grad))
        self.assertGreater(opacity_errors[0], .25*float(exact_grad[2].abs().max()))
        self.assertTrue(all(b<a for a,b in zip(opacity_errors,opacity_errors[1:])))
        self.assertTrue(all(b<a for a,b in zip(depth_errors,depth_errors[1:])))
        # Strengthened 16-level accuracy; the previous 3e-4 depth limit is kept.
        self.assertLess(depth_errors[-1],3e-4)
        self.assertLess(opacity_errors[-1],.003)

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

    def test_gradient_comparison_uses_vectors_not_norms(self):
        a=torch.tensor([1.,0.],device=self.device)
        b=torch.tensor([0.,1.],device=self.device)
        self.assertEqual(float(a.norm()),float(b.norm()))
        self.assertFalse(gradient_delta(a,b)['passed'])

    def test_invalid_roots_do_not_pass_convergence(self):
        z=torch.zeros(2,device=self.device)
        a=dict(valid=False,has_sampled_root=False,depth=0.,total=.1,subdivisions=2)
        b=dict(a,subdivisions=4)
        comparison=refinement_comparison(a,b,{'xyz':z},{'xyz':z})
        self.assertFalse(comparison['observed_pair_passed'])
        self.assertIsNone(comparison['depth_abs_change'])

    def test_nonfinite_gradient_report_is_serializable(self):
        import json
        a=torch.zeros(2,device=self.device)
        b=torch.tensor([float('nan'),float('inf')],device=self.device)
        result=gradient_delta(a,b)
        self.assertFalse(result['passed'])
        self.assertFalse(result['finite'])
        json.dumps(result,allow_nan=False)

    def test_finite_float32_gradient_difference_does_not_overflow(self):
        import json
        a=torch.tensor([3e38],device=self.device,dtype=torch.float32)
        result=gradient_delta(a,-a)
        self.assertTrue(result['finite'])
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['relative_l2'],2.)
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
