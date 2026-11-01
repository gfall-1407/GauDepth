"""CPU checks for the pure audit algebra; native GPU parity is checked at runtime."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


source = Path(__file__).resolve().parents[1] / '实验' / '09_冻结监督审计' / 'snapshot_supervision.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'scaled_filter_focals', 'masked_mean', 'numeric', 'require_finite', 'gradient', 'gradient_summary',
         'cosine', 'assert_native_pm_parity'}
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)


class SnapshotMathTest(unittest.TestCase):
    def test_filter_focals_preserve_distinct_axes(self):
        camera = SimpleNamespace(width=1600, height=1200, model='PINHOLE',
                                 params=[1000., 800., 800., 600.])
        self.assertEqual(namespace['scaled_filter_focals'](camera, 2), (800, 600, 500., 400.))
        camera.model = 'SIMPLE_PINHOLE'
        self.assertEqual(namespace['scaled_filter_focals'](camera, 2), (800, 600, 500., 500.))

    def test_empty_masks_and_nonfinite_numbers(self):
        empty = torch.zeros(3, dtype=torch.bool)
        audit = dict(pm_mask=empty, ncc_mask=empty, weights=torch.zeros(3),
                     pixel_noise=torch.zeros(3), ncc_raw=None)
        namespace['assert_native_pm_parity'](torch.tensor([0.]), torch.tensor([0.]), audit)
        self.assertIsNone(namespace['masked_mean'](None, empty))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            namespace['numeric'](torch.tensor(float('nan')))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            namespace['cosine'](torch.tensor([float('inf')]), torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            namespace['gradient_summary']({'xyz': torch.tensor([float('nan')])})

    @unittest.skipUnless(torch.cuda.is_available(), 'synthetic PM instrumentation needs local CUDA')
    def test_audit_rejects_nonfinite_before_masks(self):
        pm_source = Path(__file__).resolve().parents[1] / 'utils' / 'loss_utils.py'
        pm_tree = ast.parse(pm_source.read_text(encoding='utf-8'))
        pm_class = next(node for node in pm_tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'PatchMatch')
        state = {'sample': None, 'ncc': None}
        def fake_sample(points, *args):
            value = (torch.ones_like(points) if state['sample'] == 'finite' else
                     points if state['sample'] is None else points * state['sample'])
            return dict(sampled_depth=value,
                        inside=torch.ones(points.shape[:2], dtype=torch.bool, device='cuda'))
        def fake_ncc(depth_select, *args):
            value = .2 + depth_select * 0
            if state['ncc'] is not None:
                value = value + state['ncc']
            return value, torch.ones_like(depth_select, dtype=torch.bool)
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
        depth = torch.ones((1, 2, 2), device='cuda')
        pkg = dict(selected_depth=depth, normal=torch.zeros((3, 2, 2), device='cuda'))
        for which in ('selected depth', 'sampled depth', 'pixel noise', 'NCC input', 'NCC output'):
            state.update(sample=None, ncc=None)
            pkg['selected_depth'] = depth.clone()
            pkg['normal'] = torch.zeros((3, 2, 2), device='cuda')
            camera.Fx = 2.
            if which == 'selected depth':
                pkg['selected_depth'][0, 0, 0] = float('nan')
            elif which == 'sampled depth':
                state['sample'] = float('inf')
            elif which == 'pixel noise':
                camera.Fx = float('nan')
                state['sample'] = 'finite'
            elif which == 'NCC input':
                pkg['normal'][0, 0, 0] = float('nan')
            else:
                state['ncc'] = float('nan')
            with self.subTest(which=which), self.assertRaisesRegex(ValueError, 'nonfinite audit ' + which):
                pm(None, pkg, camera, camera, 7001, audit={})

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
