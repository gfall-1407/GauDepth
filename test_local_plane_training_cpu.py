"""Small fixed numerical controls for local_plane_training_cpu_v1."""
import unittest

import numpy as np
import torch

from local_plane_training_cpu import (ANCHOR_OFFSETS, InvalidWindow, fit,
    frozen_objective_torch, geometry, member_plane, project)


K = np.array([[1000., 0, 500], [0, 1000., 500], [0, 0, 1.]])
GEOM = geometry(K, K, np.eye(3), np.array([.1, 0, 0]), np.array([500., 500.])+ANCHOR_OFFSETS)
R = np.eye(3)
T = np.zeros(3)
AXES = np.array([2, 2])


def raw(zs=(2., 2.), quat_delta=0.):
    xyz = np.array([[0., 0., zs[0]], [.1, .1, zs[1]]])
    q = np.array([[1., 0, 0, 0], [1., quat_delta, 0, 0]])
    return xyz, q


def planes(xyz, quat):
    return np.stack([member_plane(m, q, 2, R, T)[0] for m, q in zip(xyz, quat)])


class Controls(unittest.TestCase):
    def test_coplanar_and_perturbation_gradient(self):
        xyz, quat = raw()
        best, _ = fit(planes(xyz, quat), np.array([.5, .5]), GEOM)
        self.assertEqual(best["M"], 1)
        self.assertLess(best["data_residual"], 1e-20)
        xyz[1, 2] += .001
        quat[1, 1] += .001
        best, _ = fit(planes(xyz, quat), np.array([.5, .5]), GEOM)
        x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        q = torch.tensor(quat, dtype=torch.float64, requires_grad=True)
        loss = frozen_objective_torch(x, q, AXES, [.5, .5], best["latent"],
                                      best["assignment"], R, T, GEOM)
        gx, gq = torch.autograd.grad(loss, (x, q))
        self.assertGreater(float(gx.abs().max()), 1e-9)
        self.assertGreater(float(gq.abs().max()), 1e-9)
        with torch.no_grad():
            improved = frozen_objective_torch(x-1e-6*gx, q-1e-6*gq, AXES,
                [.5, .5], best["latent"], best["assignment"], R, T, GEOM)
        self.assertLess(float(improved), float(loss))

    def test_two_layer_and_minority(self):
        p = np.array([[0., 0., .5], [0., 0., .47]])
        equal, models = fit(p, np.array([.5, .5]), GEOM)
        self.assertEqual(equal["M"], 2)
        self.assertAlmostEqual(models[0]["objective"], 2.25, places=10)
        self.assertAlmostEqual(models[1]["objective"], 1., places=10)
        minority, models = fit(p, np.array([.9, .1]), GEOM)
        self.assertEqual(minority["M"], 1)
        self.assertAlmostEqual(models[0]["objective"], .81, places=10)

    def test_clone_and_order(self):
        p = np.array([[0., 0., .5], [0., 0., .47]])
        baseline, _ = fit(p, np.array([.5, .5]), GEOM)
        clones = np.concatenate((np.repeat(p[:1], 9, axis=0), p[1:]))
        conserve, _ = fit(clones, np.array([.5/9]*9+[.5]), GEOM)
        self.assertEqual(conserve["M"], baseline["M"])
        self.assertAlmostEqual(conserve["objective"], baseline["objective"], places=12)
        self.assertTrue(np.allclose(conserve["latent"], baseline["latent"], atol=1e-12))
        wrong, _ = fit(clones, np.array([.5]*10), GEOM)
        self.assertEqual(wrong["M"], 1)
        for order in (np.arange(10)[::-1], np.random.default_rng(20261101).permutation(10)):
            alt, _ = fit(clones[order], np.array([.5/9]*9+[.5])[order], GEOM)
            self.assertEqual(alt["M"], conserve["M"])
            self.assertAlmostEqual(alt["objective"], conserve["objective"], places=12)

    def test_clone_gradient_sum(self):
        xyz, quat = raw((2., 100/47), .001)
        original, _ = fit(planes(xyz, quat), np.array([.5, .5]), GEOM)
        def grad(xyz, quat, mass, latent, assignment):
            x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
            q = torch.tensor(quat, dtype=torch.float64, requires_grad=True)
            value = frozen_objective_torch(x, q, np.full(len(xyz), 2),
                mass/sum(mass), latent, assignment, R, T, GEOM)
            return torch.autograd.grad(value, (x, q))
        a = grad(xyz, quat, np.array([.5, .5]), original["latent"], original["assignment"])
        clone_x = np.concatenate((np.repeat(xyz[:1], 9, axis=0), xyz[1:]))
        clone_q = np.concatenate((np.repeat(quat[:1], 9, axis=0), quat[1:]))
        cp = planes(clone_x, clone_q)
        cb, _ = fit(cp, np.array([.5/9]*9+[.5]), GEOM)
        b = grad(clone_x, clone_q, np.array([.5/9]*9+[.5]), cb["latent"], cb["assignment"])
        for left, right in zip(a, b):
            self.assertTrue(torch.allclose(left[0], right[:9].sum(0), atol=1e-11))
            self.assertTrue(torch.allclose(left[1], right[9], atol=1e-11))

    def test_pole(self):
        with self.assertRaises(InvalidWindow):
            project(np.array([1e12, 0, 0]), GEOM)

    def test_zero_mass_kept_out_of_fit(self):
        p = np.array([[0., 0., .5], [np.nan, 0., 0.]])
        best, _ = fit(p, np.array([1., 0.]), GEOM)
        self.assertEqual(best["assignment"], [0, -1])
        self.assertEqual(best["member_D"][1], 0.)
        self.assertEqual(best["M"], 1)


if __name__ == "__main__":
    unittest.main()
