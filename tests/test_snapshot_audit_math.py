"""CPU checks for the pure audit algebra; native GPU parity is checked at runtime."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


source = Path(__file__).resolve().parents[1] / '实验' / '09_冻结监督审计' / 'snapshot_supervision.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'masked_mean', 'numeric', 'gradient', 'gradient_summary',
         'cosine', 'assert_native_pm_parity'}
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)


class SnapshotMathTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'synthetic PM instrumentation needs local CUDA')
    def test_instrumented_pm_method_matches_uninstrumented_outputs(self):
        pm_source = Path(__file__).resolve().parents[1] / 'utils' / 'loss_utils.py'
        pm_tree = ast.parse(pm_source.read_text(encoding='utf-8'))
        pm_class = next(node for node in pm_tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'PatchMatch')
        def fake_sample(points, *args):
            return dict(sampled_depth=points,
                        inside=torch.ones(points.shape[:2], dtype=torch.bool, device='cuda'))
        def fake_ncc(depth_select, *args):
            return .2 + depth_select * 0, torch.ones_like(depth_select, dtype=torch.bool)
        env = dict(torch=torch, GaussianModel=object, Camera=object,
                   sample_depth=fake_sample,
                   warp_patch_ncc=SimpleNamespace(warp_patch_ncc=fake_ncc))
        exec(compile(ast.Module(body=[pm_class], type_ignores=[]), str(pm_source), 'exec'), env)
        pm = env['PatchMatch'](3, 1., 0., SimpleNamespace(), debug=False)
        camera = SimpleNamespace(image_height=2, image_width=2, Cx=.5, Cy=.5,
                                 Fx=2., Fy=2., R=torch.eye(3, device='cuda'),
                                 T=torch.zeros(3, device='cuda'),
                                 world_view_transform=torch.eye(4, device='cuda'),
                                 gray_image=torch.zeros((2, 2), device='cuda'))
        depth = torch.ones((1, 2, 2), device='cuda', requires_grad=True)
        pkg = dict(selected_depth=depth,
                   normal=torch.zeros((3, 2, 2), device='cuda'))
        audited = {}
        ncc, geo = pm(None, pkg, camera, camera, 7001, audit=audited)
        ncc_plain, geo_plain = pm(None, pkg, camera, camera, 7001)
        torch.testing.assert_close(ncc, ncc_plain)
        torch.testing.assert_close(geo, geo_plain)
        namespace['assert_native_pm_parity'](geo, ncc, audited)
        self.assertEqual(int(audited['pm_mask'].sum()), 4)
        self.assertEqual(int(audited['ncc_mask'].sum()), 4)
        self.assertFalse(audited['weights'].requires_grad)

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
