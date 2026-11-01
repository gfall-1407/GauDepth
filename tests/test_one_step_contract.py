import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


source = next((Path(__file__).resolve().parents[1] / '实验').glob('11_*/one_step_contract.py'))
spec = importlib.util.spec_from_file_location('one_step_contract', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContractTest(unittest.TestCase):
    def test_bad_ply_and_invalid_local_patch_block_step(self):
        with self.assertRaisesRegex(ValueError, 'wrong_or_missing'):
            module.require_full_ply('no_such_model.ply')
        report = dict(rays={'17': {'status': 'invalid_local_patch'},
                            '24': {'status': 'invalid_local_patch'}})
        with self.assertRaisesRegex(ValueError, 'fixed_local_patch_not_valid'):
            module.require_valid_patch_report(report)

    def test_appearance_initialized_once_and_frozen_but_rgb_gradient_flows(self):
        class Fake:
            calls = 0
            def create_app_model(self, count, model):
                Fake.calls += 1
                self.app_model = SimpleNamespace(name='PGSR')
                self._appearance_embeddings = torch.nn.Parameter(torch.zeros((count, 2)))
        first, second = Fake(), Fake()
        mapping = {'0012': 0, '0049': 1, '0050': 2, '0058': 3}
        token = module.initialize_shared_fresh_appearance(
            first, second, mapping, 'fresh_seed0_pgsr_frozen')
        self.assertEqual(Fake.calls, 1)
        self.assertEqual(token, module.appearance_digest(second._appearance_embeddings, mapping))
        self.assertFalse(first._appearance_embeddings.requires_grad)
        self.assertFalse(second._appearance_embeddings.requires_grad)
        render = torch.tensor([.3], requires_grad=True)
        (torch.exp(first._appearance_embeddings[1, 0])*render+
         first._appearance_embeddings[1, 1]).sum().backward()
        self.assertEqual(float(render.grad), 1.)

    def test_fresh_adam_only_moves_union_rows(self):
        names = module.RAW_GROUPS
        schedule = SimpleNamespace(xyz_scheduler_args=lambda step: .02 if step == 7001 else 99.)
        opt = SimpleNamespace(scaling_lr=.03, rotation_lr=.04, opacity_lr=.05)
        self.assertEqual(module.host_7001_group_lr(schedule, opt),
                         dict(xyz=.02, scaling=.03, rotation=.04, opacity=.05))
        params = {name: torch.nn.Parameter(torch.ones((5, 2))) for name in names}
        optimizer = module.fresh_one_step_optimizer(params, {name: .01 for name in names})
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [.001]*4)
        loss = sum(value.sum() for value in params.values())
        loss.backward()
        ids = torch.tensor([1, 3], dtype=torch.long)
        module.mask_raw_grads(params, ids, 5)
        optimizer.step()
        for value in params.values():
            torch.testing.assert_close(value.detach()[[0, 2, 4]], torch.ones((3, 2)))
            self.assertTrue(bool((value.detach()[[1, 3]] < 1).all()))


if __name__ == '__main__':
    unittest.main()
