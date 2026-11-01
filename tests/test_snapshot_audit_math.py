"""CPU checks for the pure audit algebra; native GPU parity is checked at runtime."""

import ast
import math
import unittest
from pathlib import Path

import torch


source = Path(__file__).resolve().parents[1] / '实验' / '09_冻结监督审计' / 'snapshot_supervision.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'masked_mean', 'numeric', 'gradient', 'gradient_summary',
         'cosine', 'assert_native_pm_parity'}
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)


class SnapshotMathTest(unittest.TestCase):
    def test_pm_formula_matches_existing_masked_means(self):
        audit = dict(pm_mask=torch.tensor([True, False, True]),
                     ncc_mask=torch.tensor([True, False, False]),
                     weights=torch.tensor([.8, 0., .5]),
                     pixel_noise=torch.tensor([.1, .5, .2], requires_grad=True),
                     ncc_raw=torch.tensor([.2, 0., 0.], requires_grad=True))
        geo = (audit['weights'] * audit['pixel_noise'])[audit['pm_mask']].mean()
        ncc = (audit['weights'] * audit['ncc_raw'])[audit['ncc_mask']].mean()
        namespace['assert_native_pm_parity'](geo, ncc, audit)
        with self.assertRaisesRegex(AssertionError, 'geometric'):
            namespace['assert_native_pm_parity'](geo + .01, ncc, audit)
        with self.assertRaisesRegex(AssertionError, 'NCC'):
            namespace['assert_native_pm_parity'](geo, ncc + .01, audit)

    def test_fixed_masks_and_gradients_do_not_accumulate(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        mask = torch.tensor([True, False])
        weight = torch.tensor([.8, 0.]).detach()
        loss = namespace['masked_mean'](weight * (x - 3).square(), mask)
        grads = namespace['gradient'](loss, {'xyz': x})
        self.assertIsNone(x.grad)
        self.assertAlmostEqual(float(grads['xyz'][0]), -3.2, places=6)
        self.assertEqual(float(grads['xyz'][1]), 0.)
        row = namespace['gradient_summary'](grads)['xyz']
        self.assertTrue(row['finite'])
        self.assertAlmostEqual(row['rms'], 3.2 / math.sqrt(2), places=6)
        self.assertAlmostEqual(namespace['cosine'](grads['xyz'], grads['xyz']), 1.)
        self.assertIsNone(namespace['masked_mean'](x, torch.zeros_like(mask)))


if __name__ == '__main__':
    unittest.main()
