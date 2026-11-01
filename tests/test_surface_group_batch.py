import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_group_batch import grouped_depth_from_support, native_sample_xy
from surface_group_reference import one_ray_reference
from training_depth_plugin import route_depth, summarize_candidate_audit


class BatchDepthTest(unittest.TestCase):
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
        pc = SimpleNamespace(get_xyz=xyz, get_rotation=quat, get_scaling=scale)
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
        pc = SimpleNamespace(get_xyz=xyz, get_rotation=quat, get_scaling=scale)
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
