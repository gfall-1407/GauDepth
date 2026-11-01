import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_depth_plugin import (add_step_metrics, fixed_group_short_axis_ls_z,
                                   parameter_metrics, reset_warm_start_radii,
                                   route_depth, sample_native_median_map, should_log_parameters,
                                   validate_ab_output_dir)


class DepthPluginWiringTest(unittest.TestCase):
    def test_continuous_median_query_keeps_map_and_point_gradients(self):
        camera = SimpleNamespace(
            image_height=3, image_width=3, Fx=1.0, Fy=1.0, Cx=1.0, Cy=1.0,
            world_view_transform=torch.eye(4),
        )
        point = torch.tensor([[[0.2, 0.3, 1.0]]], requires_grad=True)
        median = torch.arange(1.0, 10.0).reshape(1, 3, 3).requires_grad_()
        alpha = torch.ones((1, 3, 3))
        result = sample_native_median_map(point, camera, median, alpha)
        self.assertTrue(result['inside'].item())
        self.assertAlmostEqual(result['sampled_depth'][0, 0, 2].item(), 6.1, places=5)
        result['sampled_depth'].sum().backward()
        self.assertIsNotNone(point.grad)
        self.assertGreater(point.grad.abs().sum().item(), 0)
        self.assertIsNotNone(median.grad)
        self.assertGreater(median.grad.abs().sum().item(), 0)

    def test_ab_destination_preflight_and_warm_start_shape(self):
        with patch('training_depth_plugin.Path') as mock_path:
            target = mock_path.return_value
            target.exists.return_value = True
            target.is_dir.return_value = True
            target.iterdir.return_value = iter([object()])
            with self.assertRaises(FileExistsError):
                validate_ab_output_dir('occupied_output')
            target.iterdir.return_value = iter([])
            validate_ab_output_dir('empty_output')
            target.exists.return_value = False
            validate_ab_output_dir('new_output')
        model = SimpleNamespace(get_xyz=torch.zeros((5, 3)), max_radii2D=torch.ones(2))
        reset_warm_start_radii(model)
        self.assertEqual(tuple(model.max_radii2D.shape), (5,))
        self.assertTrue(torch.equal(model.max_radii2D, torch.zeros(5)))
        self.assertTrue(should_log_parameters(7001, 7001, 100))
        self.assertFalse(should_log_parameters(7002, 7001, 100))
        self.assertTrue(should_log_parameters(7100, 7001, 100))

    def test_disabled_is_same_object_and_gradient(self):
        x = torch.tensor([[[2., 3.]]], requires_grad=True)
        pkg = {'expected_depth': x, 'median_depth': x * 2, 'render': x + 1}
        self.assertIs(route_depth(pkg), pkg)
        identity = route_depth(pkg, 'identity')
        self.assertIs(identity['expected_depth'], pkg['expected_depth'])
        self.assertIs(identity['median_depth'], pkg['median_depth'])
        (identity['expected_depth'].sum() + identity['median_depth'].sum()).backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 3.)))
        with self.assertRaises(NotImplementedError):
            route_depth(pkg, 'candidate')

    def test_step_log_uses_actual_parameter_update(self):
        param = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.Adam([{'params': [param], 'lr': .01, 'name': 'xyz'}])
        (param**2).sum().backward()
        rows, before = parameter_metrics(optimizer)
        self.assertGreater(rows['xyz']['grad_rms'], 0)
        optimizer.step()
        rows = add_step_metrics(rows, before, optimizer, True)
        self.assertGreater(rows['xyz']['step_rms'], 0)

    def test_fixed_group_ls_camera_z_autograd_and_finite_difference(self):
        centers = torch.tensor([[.1, 0., 1.], [-.15, .2, 1.03]],
                               dtype=torch.float64, requires_grad=True)
        theta = torch.tensor(.22, dtype=torch.float64, requires_grad=True)
        ray = torch.tensor([.2, -.1, 1.], dtype=torch.float64)

        def value(mu, angle):
            zero = torch.zeros_like(angle)
            one = torch.ones_like(angle)
            normals = torch.stack([torch.stack([zero, zero, one]),
                                   torch.stack([torch.sin(angle), zero, torch.cos(angle)])])
            return fixed_group_short_axis_ls_z(mu, normals, ray, [0, 1])

        z = value(centers, theta)
        grad_mu, grad_theta = torch.autograd.grad(z, (centers, theta))
        eps = 1e-6
        with torch.no_grad():
            plus = centers.clone(); plus[1, 2] += eps
            minus = centers.clone(); minus[1, 2] -= eps
            fd_mu = (value(plus, theta) - value(minus, theta)) / (2 * eps)
            fd_theta = (value(centers, theta + eps) - value(centers, theta - eps)) / (2 * eps)
        self.assertAlmostEqual(grad_mu[1, 2].item(), fd_mu.item(), places=8)
        self.assertAlmostEqual(grad_theta.item(), fd_theta.item(), places=8)
        self.assertGreater(abs(grad_theta.item()), 1e-4)
        normals = torch.tensor([[0., 0., 1.], [.2, 0., .9797958971132712]], dtype=torch.float64)
        original = fixed_group_short_axis_ls_z(centers.detach(), normals, ray, [0, 1])
        flipped = fixed_group_short_axis_ls_z(centers.detach(),
                                               torch.stack([-normals[0], normals[1]]), ray, [0, 1])
        self.assertAlmostEqual(original.item(), flipped.item(), places=12)
        self.assertAlmostEqual(fixed_group_short_axis_ls_z(
            torch.tensor([[0., 0., 1.]], dtype=torch.float64),
            torch.tensor([[0., 0., 1.]], dtype=torch.float64),
            torch.tensor([.2, 0., 1.], dtype=torch.float64), [0]).item(), 1., places=12)

    def test_fixed_group_rejects_grazing_denominator(self):
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            fixed_group_short_axis_ls_z(torch.tensor([[0., 0., 1.]]),
                                        torch.tensor([[1., 0., 0.]]),
                                        torch.tensor([0., 0., 1.]), [0])


if __name__ == '__main__':
    unittest.main()
