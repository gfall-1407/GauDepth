import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_group_batch import grouped_depth_from_support
path = Path(__file__).resolve().parents[1] / '实验' / '07_训练读出接入' / 'audit_captured_support.py'
spec = importlib.util.spec_from_file_location('captured_audit_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CapturedAuditTest(unittest.TestCase):
    def test_main_runs_cpu_v1_parity_on_frozen_capture(self):
        temporary = Path(__file__).resolve().parents[1] / 'output_depth_plugin' / 'audit_fixture'
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            capture, output = temporary / 'support.npz', temporary / 'audit'
            np.savez_compressed(capture,
                center=np.array([[-.01, 0., 1.], [.01, 0., 1.]], dtype=np.float32),
                quaternion=np.array([[1., 0., 0., 0.]] * 2, dtype=np.float32),
                scale_unaa=np.array([[.04, .04, .01]] * 2, dtype=np.float32),
                world_view_transform=np.eye(4, dtype=np.float32),
                focal=np.array([1000., 1000., 0., 0.], dtype=np.float32),
                ids=np.array([[10, 20, -1]], dtype=np.int32),
                alpha=np.array([[.4, .4, 0.]], dtype=np.float32),
                count=np.array([2], dtype=np.int32), overflow=np.array([False]),
                unique_id=np.array([10, 20], dtype=np.int64),
                flat_pixel=np.array([5]), xy=np.zeros((1, 2), dtype=np.float32),
                native_expected=np.array([1.3], dtype=np.float32))
            with patch.object(sys, 'argv', ['audit', '--capture', str(capture),
                                            '--output', str(output)]):
                module.main()
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['cpu_v1_parity']['ok'])
            self.assertEqual(summary['reason_count'], {'accepted': 1})
        finally:
            shutil.rmtree(temporary)

    def test_first_group_and_overflow_are_separate(self):
        center = np.array([[-.01, 0., 1.], [.01, 0., 1.], [0., 0., 1.02]], dtype=np.float32)
        rotation = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
        scale = np.array([[.04, .04, .01]] * 3, dtype=np.float32)
        prepared = dict(center=center, rotation=rotation, scale=scale,
                        axis=np.array([2, 2, 2]), normal=rotation[:, :, 2],
                        primitive_valid=np.ones(3, dtype=bool))
        capture = dict(ids=np.array([[10, 20, 30], [10, 20, 30]], dtype=np.int32),
                       alpha=np.array([[.4, .4, .8], [.4, .4, .8]], dtype=np.float32),
                       count=np.array([3, 4], dtype=np.int32),
                       overflow=np.array([False, True]),
                       unique_id=np.array([10, 20, 30]),
                       flat_pixel=np.array([5, 8]),
                       xy=np.zeros((2, 2), dtype=np.float32),
                       focal=np.array([1000., 1000., 0., 0.], dtype=np.float32))
        first = module.audit_ray(0, capture, prepared)
        self.assertEqual(first['reason'], 'accepted')
        self.assertEqual([len(g['ids']) for g in first['groups']], [2, 1])
        self.assertEqual(first['groups'][0]['ids'], [10, 20])
        self.assertEqual(first['first_boundary'], {'angle_fail': False, 'distance_fail': True})
        self.assertEqual(first['later_qualified_groups'], 0)
        pc = SimpleNamespace(get_xyz=torch.from_numpy(center),
                             get_rotation=torch.tensor([[1., 0., 0., 0.]] * 3),
                             get_scaling=torch.from_numpy(scale))
        camera = SimpleNamespace(world_view_transform=torch.eye(4),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        depth, audit = grouped_depth_from_support(pc, camera,
            torch.from_numpy(capture['xy'][:1]), torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.from_numpy(capture['alpha'][:1]), torch.tensor([3]),
            torch.tensor([False]), torch.tensor([1.3]))
        self.assertTrue(audit['candidate'][0])
        self.assertEqual(int(audit['first_group_size'][0]), len(first['groups'][0]['ids']))
        self.assertAlmostEqual(float(depth[0]), first['groups'][0]['depth'])
        second = module.audit_ray(1, capture, prepared)
        self.assertEqual(second['reason'], 'overflow_unresolved')
        self.assertEqual(second['groups'], [])


if __name__ == '__main__':
    unittest.main()
