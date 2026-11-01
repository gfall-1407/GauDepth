"""Bounded CUDA versus original Torch v2 selection; run after rebuilding the extension."""

import unittest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_group_batch import _chunk, _chunk_cuda


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class GroupSelectCUDA(unittest.TestCase):
    def test_first_qualified_and_audit(self):
        device = 'cuda'
        centers = torch.tensor([[0., 0., 1.], [-.01, 0., 1.02],
                                [.01, 0., 1.02], [0., 0., 2.]], device=device)
        rotations = torch.eye(3, device=device).expand(4, 3, 3).contiguous()
        scales = torch.tensor([[.04, .04, .01]] * 4, device=device)
        xy = torch.zeros((4, 2), device=device)
        ids = torch.tensor([[0, 1, 2, -1], [1, 2, -1, -1],
                            [0, 1, 2, -1], [-1, -1, -1, -1]],
                           dtype=torch.int32, device=device)
        alpha = torch.tensor([[.8, .4, .4, 0], [.4, .4, 0, 0],
                              [.8, .4, .4, 0], [0, 0, 0, 0]], device=device)
        count = torch.tensor([3, 2, 5, 0], dtype=torch.int32, device=device)
        overflow = torch.tensor([False, False, True, False], device=device)
        native = torch.full((4,), 1.3, device=device)
        primitive = torch.ones((4,), dtype=torch.bool, device=device)
        args = (centers, rotations, scales, xy, ids, alpha, count, overflow,
                native, 1000., 1000., 0., 0., primitive, 'first_qualified')
        expected, audit = _chunk(*args)
        actual, gpu_audit = _chunk_cuda(*args)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for key in audit:
            self.assertIn(key, gpu_audit)
            if audit[key].dtype.is_floating_point:
                torch.testing.assert_close(gpu_audit[key], audit[key], rtol=2e-4, atol=2e-6,
                                           msg=key)
            else:
                torch.testing.assert_close(gpu_audit[key], audit[key], rtol=0, atol=0,
                                           msg=key)


if __name__ == '__main__':
    unittest.main()
