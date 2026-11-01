"""Small CPU checks; the real selected-root solver is a separately reviewed dependency."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


source = next((Path(__file__).resolve().parents[1] / '实验').glob('10_*/raw_param_bridge.py'))
spec = importlib.util.spec_from_file_location('raw_param_bridge', source)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class RawParamBridgeTest(unittest.TestCase):
    def test_root_file_bytes_are_bound_before_parsing(self):
        with patch.object(bridge, 'sha256', side_effect=[bridge.EXPECTED_CAPTURE_SHA, 'wrong']):
            with self.assertRaisesRegex(ValueError, 'wrong_roots_sha256'):
                bridge.load_fixed_capture('capture.npz', 'roots.json', 'metadata.json')

    def test_host_activation_and_infinite_logit(self):
        xyz = torch.tensor([[0., 0., 1.], [0., 0., 2.]], dtype=torch.float64)
        log_scale = torch.tensor([[0., 0., 0.], [0., 0., 0.]],
                                 dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[2., 0., 0., 0.], [1., 0., 0., 0.]],
                            dtype=torch.float64, requires_grad=True)
        logit = torch.tensor([[float('inf')], [0.]], dtype=torch.float64,
                             requires_grad=True)
        saved_filter = torch.tensor([[1.], [0.]], dtype=torch.float64)
        result = bridge.activate(xyz, log_scale, quat, logit, saved_filter)
        self.assertAlmostEqual(float(result['scale_render'][0, 0]), 2**.5)
        self.assertAlmostEqual(float(result['opacity_render'][0, 0]), 2**-1.5)
        self.assertAlmostEqual(float(result['opacity_render'][1, 0]), .5)
        self.assertTrue(torch.allclose(result['rotation'],
                                       torch.eye(3, dtype=torch.float64).expand(2, 3, 3)))
        gradients = torch.autograd.grad(result['opacity_render'].sum(), (log_scale, logit))
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))
        self.assertEqual(float(gradients[1][0, 0]), 0.)

    def test_camera_z_ray_keeps_recorded_float32_axis_error(self):
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[0, 0] = 1+2e-8
        matrix[2, 2] = 1+2e-8
        origin, ray, forward = bridge.world_camera_ray(
            matrix, torch.tensor([100., 200., 0., 0.], dtype=torch.float64),
            torch.tensor([10., 20.], dtype=torch.float64))
        self.assertTrue(torch.equal(origin, torch.zeros(3, dtype=torch.float64)))
        self.assertAlmostEqual(float(ray[0]), .1*(1+2e-8))
        self.assertAlmostEqual(float(ray[2]), 1+2e-8)
        self.assertAlmostEqual(float(ray @ forward), 1+4e-8)

    def test_two_queries_accumulate_to_one_shared_id(self):
        arrays = dict(center=np.array([[0., 0., 2.]], dtype=np.float32),
                      raw_scale=np.zeros((1, 3), dtype=np.float32),
                      raw_quaternion=np.array([[1., 0., 0., 0.]], dtype=np.float32),
                      opacity_logit=np.zeros((1, 1), dtype=np.float32),
                      filter_3D=np.zeros((1, 1), dtype=np.float32),
                      xy=np.zeros((2, 2), dtype=np.float32),
                      world_view_transform=np.eye(4, dtype=np.float32),
                      intrinsics=np.array([100., 100., 0., 0.]))
        raw = bridge.shared_raw_leaves(arrays)
        def toy_root(mu, scales, omega, amplitude, rotation, origin, ray, bracket):
            depth = mu[:, 2].sum() + scales.sum()*0 + amplitude.sum()*0 + rotation.sum()*0
            return depth, mu.new_tensor([0., 0., -1.]), {'root_value': float(depth.detach())}
        row0 = dict(query=0, indices=[0], bracket=(1., 3.))
        row1 = dict(query=1, indices=[0], bracket=(1., 3.))
        z0, _, _, _ = bridge.root_from_shared(arrays, row0, raw, toy_root)
        z1, _, _, _ = bridge.root_from_shared(arrays, row1, raw, toy_root)
        gradient = torch.autograd.grad(z0+z1, raw['xyz'])[0]
        self.assertEqual(float(gradient[0, 2]), 2.)
        self.assertEqual(float(gradient[0, 0]), 0.)


if __name__ == '__main__':
    unittest.main()
