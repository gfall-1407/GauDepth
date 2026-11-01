import sys
import unittest
import importlib.util
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_group_batch import compose_sample_points, grouped_depth_from_support, native_sample_xy
from surface_group_reference import one_ray_reference
from training_depth_plugin import (mesh_depth_channel, record_depth_route_on_dataset,
                                   route_depth, summarize_candidate_audit)


class BatchDepthTest(unittest.TestCase):
    def test_saved_cfg_args_reloads_candidate_mesh_route(self):
        from arguments import ModelParams, get_combined_args
        path = Path(__file__).resolve().parents[1] / 'output_depth_plugin' / 'cfg_route_fixture'
        path.mkdir(parents=True, exist_ok=False)
        try:
            dataset = SimpleNamespace(model_path=str(path), depth_ratio=0.)
            record_depth_route_on_dataset(dataset, 'candidate', 'expected', 'first_qualified')
            (path / 'cfg_args').write_text(str(Namespace(**vars(dataset))), encoding='utf-8')
            parser = ArgumentParser()
            ModelParams(parser, sentinel=True)
            parser.add_argument('--depth_route', default=None)
            parser.add_argument('--depth_target', default=None)
            parser.add_argument('--candidate_selection', default=None)
            with patch.object(sys, 'argv', ['mesh_extract.py', '-m', str(path)]):
                loaded = get_combined_args(parser)
            self.assertEqual((loaded.depth_route, loaded.depth_target,
                              loaded.candidate_selection),
                             ('candidate', 'expected', 'first_qualified'))
        finally:
            shutil.rmtree(path)

    def test_mesh_requires_candidate_channel_to_match_training_target(self):
        self.assertEqual(mesh_depth_channel(0., 'candidate', 'expected'), 'expected_depth')
        self.assertEqual(mesh_depth_channel(.6, 'candidate', 'median'), 'median_depth')
        self.assertEqual(mesh_depth_channel(0., 'baseline', 'median'), 'expected_depth')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            mesh_depth_channel(0., 'candidate', 'median')

    def test_first_qualified_selects_later_group_and_exposes_thin_foreground_risk(self):
        xyz = torch.tensor([[0., 0., 1.], [-.01, 0., 1.02], [.01, 0., 1.02]],
                           requires_grad=True)
        quaternion = torch.tensor([[1., 0., 0., 0.]] * 3, requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz, _rotation=quaternion,
                             get_scaling=torch.tensor([[.04, .04, .01]] * 3))
        camera = SimpleNamespace(world_view_transform=torch.eye(4),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        xy = torch.zeros((1, 2), requires_grad=True)
        native = torch.tensor([1.3], requires_grad=True)
        common = (pc, camera, xy, torch.tensor([[0, 1, 2]], dtype=torch.int32),
                  torch.tensor([[.8, .4, .4]]), torch.tensor([3]),
                  torch.tensor([False]), native)
        old, old_audit = grouped_depth_from_support(*common)
        new, audit = grouped_depth_from_support(*common, selection='first_qualified')
        self.assertFalse(old_audit['candidate'][0])
        torch.testing.assert_close(old, native)
        self.assertTrue(audit['candidate'][0])
        self.assertEqual(audit['chosen_group_index'].tolist(), [1])
        self.assertEqual(audit['skipped_prefix_groups'].tolist(), [1])
        self.assertEqual(audit['skipped_prefix_members'].tolist(), [1])
        self.assertAlmostEqual(float(audit['skipped_prefix_alpha'][0]), .8)
        self.assertAlmostEqual(float(new[0]), 1.02, places=5)
        # This foreground is a real synthetic surface at z=1.0. Passing the
        # later group is a readout, not evidence that it is first visible.
        direct_grad = torch.autograd.grad(new.sum(), xyz, retain_graph=True)[0]
        detached_surrogate = new.detach() + native - native.detach()
        detached_grad = torch.autograd.grad(detached_surrogate.sum(), xyz,
                                           retain_graph=True, allow_unused=True)[0]
        self.assertGreater(float(direct_grad.abs().sum()), 0.)
        self.assertIsNone(detached_grad)
        new.sum().backward()
        self.assertEqual(float(xyz.grad[0].abs().sum()), 0.)
        self.assertGreater(float(xyz.grad[1:].abs().sum()), 0.)
        self.assertEqual(float(native.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(quaternion.grad).all())
        self.assertTrue(torch.isfinite(xy.grad).all())

    def test_first_qualified_keeps_first_and_overflow_fallback(self):
        xyz = torch.tensor([[-.01, 0., 1.], [.01, 0., 1.]], requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz,
                             _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                             get_scaling=torch.tensor([[.04, .04, .01]] * 2))
        camera = SimpleNamespace(world_view_transform=torch.eye(4),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        common = (pc, camera, torch.zeros((2, 2)),
                  torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
                  torch.tensor([[.4, .4], [.4, .4]]), torch.tensor([2, 3]),
                  torch.tensor([False, True]), torch.tensor([1.3, 1.3]))
        old, _ = grouped_depth_from_support(*common)
        new, audit = grouped_depth_from_support(*common, selection='first_qualified')
        torch.testing.assert_close(old, new)
        self.assertEqual(audit['candidate'].tolist(), [True, False])
        self.assertEqual(audit['chosen_group_index'].tolist(), [0, -1])

    def test_full_sample_depth_wrapper_fallback_backprop(self):
        from utils.graphics_utils import getProjectionMatrix
        import types
        scene_module = types.ModuleType('scene')
        model_module = types.ModuleType('scene.gaussian_model')
        model_module.GaussianModel = object
        diff_module = types.ModuleType('diff_gaussian_rasterization')
        diff_module.GaussianRasterizationSettings = lambda **kwargs: SimpleNamespace(**kwargs)
        native_points = torch.tensor([[[.1, .2, 1.3], [.3, .4, 1.4]]], requires_grad=True)

        class FakeRasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings

            def sample_depth(self, **kwargs):
                self.xy = kwargs['support_xy']
                return (native_points, torch.tensor([[False, False]]),
                        torch.full((2, 2), -1, dtype=torch.int32),
                        torch.zeros((2, 2)), torch.zeros(2, dtype=torch.int32),
                        torch.zeros(2, dtype=torch.bool))

        diff_module.GaussianRasterizer = FakeRasterizer
        renderer_path = Path(__file__).resolve().parents[1] / 'gaussian_renderer' / '__init__.py'
        spec = importlib.util.spec_from_file_location('renderer_wrapper_test', renderer_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'scene': scene_module,
                                      'scene.gaussian_model': model_module,
                                      'diff_gaussian_rasterization': diff_module}):
            spec.loader.exec_module(module)
        pc = SimpleNamespace(get_xyz=torch.tensor([[-.01, 0., 1.], [.01, 0., 1.]]),
                             get_rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                             get_scaling=torch.tensor([[.04, .04, .01]] * 2),
                             get_scaling_with_3D_filter=torch.tensor([[.04, .04, .01]] * 2),
                             get_opacity_with_3D_filter=torch.full((2, 1), .4),
                             active_sh_degree=0)
        camera = SimpleNamespace(FoVx=1., FoVy=1., image_height=1, image_width=2,
                                 world_view_transform=torch.eye(4),
                                 full_proj_transform=getProjectionMatrix(.01, 100., 1., 1.).T,
                                 camera_center=torch.zeros(3), Fx=1., Fy=1., Cx=.5, Cy=0.)
        world = torch.tensor([[[0., 0., -1e-7], [float('nan'), 0., 1.]]],
                             requires_grad=True)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False)
        with patch.object(torch.cuda, 'synchronize', lambda *args: None), \
             patch.object(torch.cuda, 'max_memory_allocated', lambda *args: 0):
            result = module.sample_depth(world, camera, pc, pipe, 1., depth_support=True)
        torch.testing.assert_close(result['sampled_depth'], native_points)
        self.assertFalse(result['candidate_audit']['candidate'].any())
        result['sampled_depth'].sum().backward()
        self.assertTrue(torch.isfinite(world.grad).all())
        self.assertEqual(world.grad.tolist(), [[[0.] * 3] * 2])
        self.assertEqual(native_points.grad.tolist(), [[[1.] * 3] * 2])

    def test_sample_wrapper_invalid_projection_and_fallback_backprop(self):
        from utils.graphics_utils import getProjectionMatrix
        projection = getProjectionMatrix(.01, 100., 1., 1.).T
        camera = SimpleNamespace(full_proj_transform=projection,
                                 image_width=32, image_height=32,
                                 Fx=16., Fy=16., Cx=15.5, Cy=15.5)
        world = torch.tensor([[0., 0., -1e-7], [float('nan'), 0., 1.]],
                             requires_grad=True)
        xy = native_sample_xy(world, camera)
        self.assertFalse(torch.isfinite(xy).all())
        native_points = torch.tensor([[[.1, .2, 1.3], [.3, .4, 1.4]]], requires_grad=True)
        candidate_z = native_points[..., 2].reshape(-1)
        output = compose_sample_points(candidate_z, xy, torch.tensor([False, False]),
                                       native_points, camera)
        torch.testing.assert_close(output, native_points)
        output.sum().backward()
        self.assertTrue(torch.isfinite(world.grad).all())
        self.assertEqual(world.grad.tolist(), [[0.] * 3] * 2)
        self.assertTrue(torch.isfinite(native_points.grad).all())
        self.assertEqual(native_points.grad.tolist(), [[[1.] * 3] * 2])

    def test_empty_schema_and_fallback_gradient(self):
        pc = SimpleNamespace(get_xyz=torch.empty((0, 3)),
                             get_rotation=torch.empty((0, 4)), get_scaling=torch.empty((0, 3)))
        camera = SimpleNamespace(world_view_transform=torch.eye(4),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        native = torch.tensor([1.3], requires_grad=True)
        result, audit = grouped_depth_from_support(pc, camera, torch.zeros((1, 2)),
            torch.full((1, 2), -1, dtype=torch.int32), torch.zeros((1, 2)),
            torch.tensor([0]), torch.tensor([False]), native)
        self.assertIs(result, native)
        self.assertEqual(audit['candidate'].tolist(), [False])
        self.assertEqual(audit['no_valid'].tolist(), [True])
        self.assertIn('overflow', audit)
        summary = summarize_candidate_audit(audit, .5, 42.)
        self.assertEqual(summary['queries'], 1)
        self.assertEqual(summary['selected'], 0)
        self.assertEqual(summary['gpu_peak_allocated_mb'], 42.)
        result.sum().backward()
        self.assertEqual(native.grad.tolist(), [1.])
        zero, zero_audit = grouped_depth_from_support(pc, camera, torch.empty((0, 2)),
            torch.empty((0, 2), dtype=torch.int32), torch.empty((0, 2)),
            torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bool), torch.empty(0))
        self.assertEqual(zero.numel(), 0)
        self.assertEqual(zero_audit['candidate'].numel(), 0)

    def test_invalid_query_and_quaternion_do_not_pollute_gradients(self):
        xyz = torch.tensor([[-.01, 0., 1.], [.01, 0., 1.],
                            [float('nan'), 0., 1.], [0., 0., 1.],
                            [0., 0., 1.], [0., 0., 1.]],
                           dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.],
                             [1., 0., 0., 0.], [float('nan'), 0., 0., 0.],
                             [0., 0., 0., 0.], [1., 0., 0., 0.]],
                            dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([[.04, .04, .01]] * 5 + [[float('inf'), .04, .01]],
                             dtype=torch.float64, requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz,
                             get_rotation=torch.nn.functional.normalize(quat, dim=-1),
                             _rotation=quat, get_scaling=scale)
        camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        xy = torch.tensor([[0., 0.], [float('nan'), 0.], [float('inf'), 0.]],
                          dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]] * 3, dtype=torch.int32)
        alpha = torch.full((3, 6), .4, dtype=torch.float64)
        native = torch.tensor([1.3, 1.4, 1.5], dtype=torch.float64, requires_grad=True)
        depth, audit = grouped_depth_from_support(pc, camera, xy, ids, alpha,
                            torch.full((3,), 6), torch.zeros(3, dtype=torch.bool), native)
        self.assertEqual(audit['candidate'].tolist(), [True, False, False])
        self.assertEqual(audit['invalid_count'].tolist(), [4, 6, 6])
        partition = (audit['invalid_basic'] + audit['invalid_short_axis'] +
                     audit['invalid_grazing'] + audit['invalid_tau'])
        torch.testing.assert_close(partition, audit['invalid_count'])
        self.assertEqual(audit['first_group_size'].tolist(), [2, 0, 0])
        summary = summarize_candidate_audit(audit, 1.)
        self.assertEqual(summary['first_ge2'], 1)
        self.assertEqual(summary['first_alpha_pass'], 1)
        self.assertEqual(summary['selected'], 1)
        depth.sum().backward()
        for grad in (xyz.grad, quat.grad, xy.grad, native.grad):
            self.assertTrue(torch.isfinite(grad).all(), grad)
        self.assertIsNone(scale.grad)
        self.assertEqual(native.grad.tolist(), [0., 1., 1.])

    def test_sample_projection_uses_native_homogeneous_w(self):
        from utils.graphics_utils import getProjectionMatrix
        width, height = 801, 601
        projection = getProjectionMatrix(.01, 100., 1.1, .9).T.double()
        camera = SimpleNamespace(full_proj_transform=projection,
                                 image_width=width, image_height=height)
        points = torch.tensor([[.1, -.2, 2.], [.0, .0, 1.]],
                              dtype=torch.float64, requires_grad=True)
        pixels = native_sample_xy(points, camera)
        hom = points @ projection[:3, :] + projection[3, :]
        expected = ((hom[:, :2] / (hom[:, 3:4] + 1e-7) + 1) *
                    torch.tensor([width, height]) - 1) / 2
        torch.testing.assert_close(pixels, expected)
        self.assertAlmostEqual(pixels[1, 0].item(), (width-1)/2)
        self.assertAlmostEqual(pixels[1, 1].item(), (height-1)/2)
        pixels.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_fractional_tile_boundary_matches_native_not_fov_shortcut(self):
        import math
        from utils.graphics_utils import getProjectionMatrix
        width, height, fx, fy, z = 1600, 1200, 1000., 1000., .21
        cx = (width - 1) / 2
        target_fov_x = 15.4999
        world_x = (target_fov_x - cx) * z / fx
        fov_x = 2 * math.atan(width / (2 * fx))
        fov_y = 2 * math.atan(height / (2 * fy))
        camera = SimpleNamespace(full_proj_transform=getProjectionMatrix(
            .01, 100., fov_x, fov_y).T, image_width=width, image_height=height)
        xy = native_sample_xy(torch.tensor([[world_x, 0., z]], dtype=torch.float32), camera)
        self.assertLess(target_fov_x, 15.5)
        self.assertGreater(xy[0, 0].item(), 15.5)
        self.assertEqual(int((target_fov_x + .5) / 16), 0)
        self.assertEqual(int((xy[0, 0].item() + .5) / 16), 1)

    def test_zero_quaternion_is_excluded_like_reference(self):
        xyz = torch.tensor([[-.01, 0., 1.], [.01, 0., 1.], [0., 0., .9]],
                           dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]],
                            dtype=torch.float64, requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz, get_rotation=quat,
                             get_scaling=torch.tensor([[.04, .04, .01]] * 3, dtype=torch.float64))
        camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        xy = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        alpha = torch.full((1, 3), .4, dtype=torch.float64)
        native = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
        output, audit = grouped_depth_from_support(pc, camera, xy, ids, alpha,
                            torch.tensor([3]), torch.tensor([False]), native)
        ref, info = one_ray_reference(xyz, quat, pc.get_scaling, alpha[0],
                                      torch.tensor([0., 0., 1.], dtype=torch.float64),
                                      1000., 1000., native[0], gaussian_ids=ids[0])
        torch.testing.assert_close(output[0], ref)
        self.assertEqual(audit['invalid_count'].item(), info['invalid_count'])
        output.sum().backward()
        for grad in (xyz.grad, quat.grad, xy.grad, native.grad):
            self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(quat.grad[2].tolist(), [0.] * 4)

    def test_random_small_queries_match_one_ray_reference(self):
        generator = torch.Generator().manual_seed(641)
        n, k, queries = 12, 6, 20
        xyz = torch.rand((n, 3), generator=generator, dtype=torch.float64) * .04
        xyz[:, 2] += 1.
        quat = torch.randn((n, 4), generator=generator, dtype=torch.float64)
        quat[:, 1:] *= .04
        quat[:, 0] += 1.
        scale = torch.full((n, 3), .04, dtype=torch.float64)
        scale[:, 2] = .01
        pc = SimpleNamespace(get_xyz=xyz,
                             get_rotation=torch.nn.functional.normalize(quat, dim=-1),
                             _rotation=quat, get_scaling=scale)
        camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64),
                                 Fx=400., Fy=410., Cx=0., Cy=0.)
        xy = (torch.rand((queries, 2), generator=generator, dtype=torch.float64) - .5) * 4
        ids = torch.stack([torch.randperm(n, generator=generator)[:k] for _ in range(queries)]).int()
        alpha = torch.rand((queries, k), generator=generator, dtype=torch.float64) * .8
        count = torch.randint(1, k+1, (queries,), generator=generator, dtype=torch.int32)
        native = torch.full((queries,), 1.3, dtype=torch.float64)
        batch, audit = grouped_depth_from_support(pc, camera, xy, ids, alpha,
                                  count, torch.zeros(queries, dtype=torch.bool), native, chunk_size=7)
        for row in range(queries):
            selected = ids[row, :count[row]].long()
            ray = torch.tensor([xy[row, 0]/400., xy[row, 1]/410., 1.], dtype=torch.float64)
            single, info = one_ray_reference(xyz[selected], quat[selected], scale[selected],
                alpha[row, :count[row]], ray, 400., 410., native[row], gaussian_ids=selected)
            torch.testing.assert_close(batch[row], single, atol=1e-12, rtol=1e-12)
            self.assertEqual(bool(audit['candidate'][row]), info['candidate_valid'])
            self.assertEqual(int(audit['valid_count'][row]), info['valid_count'])

    def test_route_preserves_each_native_fallback(self):
        expected = torch.tensor([[[1., 2.]]], requires_grad=True)
        median = torch.tensor([[[3., 4.]]], requires_grad=True)
        candidate = torch.tensor([[[5., 6.]]], requires_grad=True)
        pkg = dict(expected_depth=expected, median_depth=median,
                   candidate_depth=candidate,
                   candidate_audit={'candidate': torch.tensor([True, False])})
        output = route_depth(pkg, 'candidate', 'both')
        torch.testing.assert_close(output['expected_depth'], torch.tensor([[[5., 2.]]]))
        torch.testing.assert_close(output['median_depth'], torch.tensor([[[5., 4.]]]))
        (output['expected_depth'] + output['median_depth']).sum().backward()
        self.assertEqual(candidate.grad.tolist(), [[[2., 0.]]])
        self.assertEqual(expected.grad.tolist(), [[[0., 1.]]])
        self.assertEqual(median.grad.tolist(), [[[0., 1.]]])

    def test_batch_matches_one_ray_and_keeps_gradients(self):
        xyz = torch.tensor([[-.01, 0., 1.], [.01, 0., 1.],
                            [0., 0., 1.02]], dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[1., 0., 0., 0.]] * 3, dtype=torch.float64,
                            requires_grad=True)
        scale = torch.tensor([[.04, .04, .01]] * 3, dtype=torch.float64,
                             requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz, get_rotation=quat, get_scaling=scale)
        camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        xy = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 1, -1], [2, -1, -1], [0, 1, 2]], dtype=torch.int32)
        alpha = torch.tensor([[.4, .4, 0.], [.8, 0., 0.], [.4, .4, .8]], dtype=torch.float64)
        count = torch.tensor([2, 1, 3], dtype=torch.int32)
        overflow = torch.tensor([False, False, True])
        native = torch.tensor([1.3, 1.4, 1.5], dtype=torch.float64, requires_grad=True)
        depth, audit = grouped_depth_from_support(pc, camera, xy, ids, alpha,
                                                  count, overflow, native, chunk_size=2)
        ref, _ = one_ray_reference(xyz[:2], quat[:2], scale[:2], alpha[0, :2],
                                   torch.tensor([0., 0., 1.], dtype=torch.float64),
                                   1000., 1000., native[0], gaussian_ids=ids[0, :2])
        torch.testing.assert_close(depth[0], ref)
        torch.testing.assert_close(depth[1:], native[1:])
        self.assertEqual(audit['candidate'].tolist(), [True, False, False])
        depth.sum().backward()
        self.assertIsNotNone(xyz.grad)
        self.assertGreater(xyz.grad[:2].abs().sum().item(), 0)
        self.assertEqual(native.grad.tolist(), [0., 1., 1.])

    def test_excluded_nan_member_has_finite_gradient(self):
        xyz = torch.tensor([[-.01, 0., 1.], [.01, 0., 1.],
                            [float('nan'), 0., 1.]], dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[1., 0., 0., 0.]] * 3, dtype=torch.float64,
                            requires_grad=True)
        scale = torch.tensor([[.04, .04, .01]] * 3, dtype=torch.float64)
        pc = SimpleNamespace(get_xyz=xyz, get_rotation=quat, get_scaling=scale)
        camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64),
                                 Fx=1000., Fy=1000., Cx=0., Cy=0.)
        xy = torch.zeros((1, 2), dtype=torch.float64)
        ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        alpha = torch.full((1, 3), .4, dtype=torch.float64)
        native = torch.tensor([1.3], dtype=torch.float64)
        depth, audit = grouped_depth_from_support(pc, camera, xy, ids, alpha,
                            torch.tensor([3]), torch.tensor([False]), native)
        self.assertTrue(audit['candidate'].item())
        self.assertEqual(audit['invalid_count'].item(), 1)
        depth.sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad[:2]).all())


if __name__ == '__main__':
    unittest.main()
