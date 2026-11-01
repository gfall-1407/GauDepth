import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_group_reference import one_ray_reference


class SurfaceGroupReferenceTest(unittest.TestCase):
    def inputs(self, z=(1., 1.), x=(-.01, .01), alpha=(.4, .4), scale=.04):
        center = torch.tensor([[x[0], 0., z[0]], [x[1], 0., z[1]]], dtype=torch.float64,
                              requires_grad=True)
        quat = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64,
                            requires_grad=True)
        scales = torch.tensor([[scale, scale, .01], [scale, scale, .01]],
                              dtype=torch.float64, requires_grad=True)
        opacity = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
        ray = torch.tensor([0., 0., 1.], dtype=torch.float64, requires_grad=True)
        baseline = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        return center, quat, scales, opacity, ray, baseline

    def test_group_and_detached_branch_gradients(self):
        m, _, s, a, v, old = self.inputs()
        theta = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        q = torch.stack([torch.stack([one, zero, zero, zero]),
                         torch.stack([torch.cos(theta/2), zero, torch.sin(theta/2), zero])])
        z, info = one_ray_reference(m, q, s, a, v, 1000., 1000., old)
        self.assertTrue(info['candidate_valid'])
        self.assertEqual(info['selected'], [0, 1])
        grads = torch.autograd.grad(z, (m, theta, s, a, v, old), allow_unused=True)
        self.assertIsNotNone(grads[0])
        self.assertGreater(abs(grads[1].item()), 1e-5)
        self.assertIsNone(grads[2])
        self.assertIsNone(grads[3])
        self.assertIsNotNone(grads[4])
        self.assertIsNone(grads[5])
        eps = 1e-6
        with torch.no_grad():
            def at(angle):
                qq = torch.stack([torch.tensor([1., 0., 0., 0.], dtype=torch.float64),
                                  torch.stack([torch.cos(angle/2), zero, torch.sin(angle/2), zero])])
                return one_ray_reference(m.detach(), qq, s.detach(), a.detach(), v.detach(),
                                         1000., 1000., old.detach())[0]
            finite_difference = (at(theta.detach()+eps)-at(theta.detach()-eps))/(2*eps)
        self.assertAlmostEqual(grads[1].item(), finite_difference.item(), places=7)

    def test_front_group_failure_does_not_select_behind(self):
        m, q, s, a, v, old = self.inputs(z=(1., 1.02))
        z, info = one_ray_reference(m, q, s, a, v, 1000., 1000., old)
        self.assertIs(z, old)
        self.assertEqual(info['reason'], 'insufficient_distinct_geometry')
        self.assertEqual(info['selected'], [0])

    def test_low_alpha_duplicate_and_capacity_fallback(self):
        m, q, s, a, v, old = self.inputs(alpha=(.1, .1))
        z, info = one_ray_reference(m, q, s, a, v, 1000., 1000., old)
        self.assertIs(z, old)
        self.assertEqual(info['reason'], 'low_group_alpha')
        duplicate, info = one_ray_reference(m[:1].repeat(2, 1), q, s, a.new_tensor([.8, .8]),
                                            v, 1000., 1000., old)
        self.assertIs(duplicate, old)
        self.assertEqual(info['reason'], 'insufficient_distinct_geometry')
        many, info = one_ray_reference(m[:1].repeat(65, 1), q[:1].repeat(65, 1),
                                       s[:1].repeat(65, 1), a[:1].repeat(65),
                                       v, 1000., 1000., old, max_accepted=64)
        self.assertIs(many, old)
        self.assertEqual(info['reason'], 'overflow')

    def test_bridge_exposes_ambiguous_complete_link(self):
        m, q, s, a, v, old = self.inputs()
        m = torch.tensor([[0., 0., 1.], [0., 0., 1.005], [0., 0., 1.010]], dtype=torch.float64)
        q = q[:1].repeat(3, 1)
        s = s[:1].repeat(3, 1)
        a = torch.full((3,), .8, dtype=torch.float64)
        depth, info = one_ray_reference(m, q, s, a, v, 400., 400., old)
        self.assertEqual(info['edges'], [[0, 1], [1, 2]])
        self.assertTrue(info['order_ambiguous'])
        self.assertEqual(info['p3_witness'], [0, 1, 2])
        self.assertEqual(info['groups'], [[0, 1], [2]])
        self.assertTrue(info['candidate_valid'])
        self.assertAlmostEqual(depth.item(), 1.0025)


if __name__ == '__main__':
    unittest.main()
