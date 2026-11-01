import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(next((root / '实验').glob('10_*/raw_param_bridge.py')).parent))
path = next((root / '实验').glob('11_*/local_patch_root.py'))
spec = importlib.util.spec_from_file_location('local_patch_root', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from raw_param_bridge import shared_raw_leaves


class LocalPatchTest(unittest.TestCase):
    def test_planar_root_traces_four_substeps_with_finite_gradient(self):
        xy = np.array([(x, y) for y in (199., 200., 201.)
                       for x in (99., 100., 101.)], dtype=np.float32)
        arrays = dict(xy=xy, center=np.array([[0., 0., 1.]], dtype=np.float32),
                      raw_scale=np.zeros((1, 3), dtype=np.float32),
                      raw_quaternion=np.array([[1., 0., 0., 0.]], dtype=np.float32),
                      opacity_logit=np.zeros((1, 1), dtype=np.float32),
                      filter_3D=np.zeros((1, 1), dtype=np.float32),
                      world_view_transform=np.eye(4, dtype=np.float32),
                      intrinsics=np.array([100., 100., 100., 200.]))
        raw = shared_raw_leaves(arrays)
        row = dict(query=4, ids=[7], indices=[0], bracket=(.5, 1.5))
        def plane_solver(mu, scales, omega, amplitude, rotation, origin, ray, bracket):
            z = mu[0, 2]+0*(scales.sum()+amplitude.sum()+rotation.sum())
            return z, z.new_tensor([0., 0., -1.]), {'root_value': float(z.detach())}
        depth, normal, trace = module.patch_root_normal(arrays, row, raw, plane_solver)
        self.assertAlmostEqual(float(depth), 1.)
        torch.testing.assert_close(normal, torch.tensor([0., 0., -1.], dtype=torch.float64))
        self.assertEqual(len(trace['steps']), 9)
        self.assertTrue(all(len(value) == (1 if j == 4 else 4)
                            for j, value in trace['steps'].items()))
        grad = torch.autograd.grad(depth+normal.sum(), raw['xyz'])[0]
        self.assertAlmostEqual(float(grad[0, 2]), 1.)


if __name__ == '__main__':
    unittest.main()
