import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_depth_plugin import add_step_metrics, parameter_metrics, route_depth


class DepthPluginWiringTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
