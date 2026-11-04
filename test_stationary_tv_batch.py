"""CUDA-only parity checks for batched stationary-TV; no scene/model writes."""
import unittest
import torch

from joint_tv_depth import ray_profiles
from stationary_tv_depth import stationary_depth
from stationary_tv_batch import stationary_depth_batch, GUARD_NAMES


class BatchStationaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; these tests must not silently run on CPU')

    def parameters(self, mean, sigma, amplitude, dtype=torch.float64):
        return tuple(torch.tensor(x, device='cuda', dtype=dtype, requires_grad=True)
                     for x in (mean, sigma, amplitude))

    def mixed(self):
        return self.parameters(
            [[-3., 3., 5., 6.], [-.31, .27, 5., 6.], [-.3, .3, 5., 6.],
             [0., 5., 6., 7.], [1., 5., 6., 7.], [0., 1., 2., 3.]],
            [[.2, .2, 1., 1.], [.91, 1.03, 1., 1.], [1., 1., 1., 1.],
             [1., 1., 1., 1.], [.005, 1., 1., 1.], [1., 1., 1., 1.]],
            [[.4, .9, 0., 0.], [.57, .52, 0., 0.], [.3, .3, 0., 0.],
             [.75, 0., 0., 0.], [.9, 0., 0., 0.], [0., 0., 0., 0.]])

    def assert_close(self, a, b, atol=1e-9, rtol=1e-8):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def test_mixed_forward_and_weighted_gradients_match_reference(self):
        p = self.mixed()
        batch = stationary_depth_batch(*p, near=-10, far=10, subdivisions=4)
        reference = [stationary_depth(*(x[i:i + 1] for x in p),
                     near=-10, far=10, subdivisions=4) for i in range(len(p[0]))]
        expected_depth = torch.cat([r.depth for r in reference])
        self.assertEqual(batch.valid.tolist(), [True, True, False, False, True, False])
        for name in ('depth', 'valid', 'has_root', 'total', 'near_potential', 'slope'):
            expected = torch.cat([getattr(r, name) for r in reference])
            if expected.dtype == torch.bool:
                self.assertTrue(torch.equal(getattr(batch, name), expected))
            else:
                self.assert_close(getattr(batch, name), expected)
        weights = p[0].new_tensor([.2, -.7, 1.3, 2., -.5, 3.])
        actual = torch.autograd.grad((batch.depth * weights).sum(), p, retain_graph=True)
        expected = torch.autograd.grad((expected_depth * weights).sum(), p)
        for a, b in zip(actual, expected): self.assert_close(a, b, atol=2e-8, rtol=2e-7)
        for a in actual:
            self.assertTrue(bool((a[~batch.valid] == 0).all()))
        for i, ref in enumerate(reference):
            count = int(batch.critical_count[i])
            self.assertEqual(count, len(ref.critical_points))
            self.assertEqual(int(batch.prefix_count[i]), ref.prefix_count)
            self.assert_close(batch.critical_points[i, :count], ref.critical_points)
            self.assert_close(batch.monotonic_directions[i, :count + 1], ref.monotonic_directions)

    def test_independent_forward_gradcheck_with_nonzero_near(self):
        p = self.parameters([[.2], [.3]], [[1.], [1.05]], [[.9], [.85]])
        fn = lambda *x: stationary_depth_batch(*x, near=-3, far=10,
                                               max_near_potential=.01).depth
        out = stationary_depth_batch(*p, near=-3, far=10, max_near_potential=.01)
        self.assertTrue(bool(out.valid.all()))
        self.assertTrue(bool((out.near_potential > .001).all()))
        self.assertTrue(torch.autograd.gradcheck(fn, p, eps=1e-6, atol=2e-6, rtol=2e-5))

    def test_zero_padding_preserves_active_parameters(self):
        p = self.parameters([[1.], [1.2], [.8]], [[.005], [.06], [.08]], [[.9], [.95], [.85]])
        zeros = p[0].new_zeros((3, 3))
        padded = (torch.cat((p[0], zeros + 4), -1),
                  torch.cat((p[1], zeros + 1), -1),
                  torch.cat((p[2], zeros), -1))
        member_mask = torch.zeros((3, 4), device='cuda', dtype=torch.bool)
        member_mask[:, 0] = True
        plain = stationary_depth_batch(*p, near=.2001, far=8, subdivisions=4)
        extra = stationary_depth_batch(*padded, near=.2001, far=8, subdivisions=4,
                                       member_mask=member_mask)
        self.assertTrue(bool(plain.valid.all() & extra.valid.all()))
        self.assert_close(plain.depth, extra.depth)
        ga = torch.autograd.grad(plain.depth.sum(), p, retain_graph=True)
        gb = torch.autograd.grad(extra.depth.sum(), p)
        for a, b in zip(ga, gb): self.assert_close(a, b)

    def test_no_critical_points_and_invalid_metadata(self):
        p = self.parameters([[1., 2.], [3., 4.]], [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]])
        r = stationary_depth_batch(*p, near=.2001, far=8)
        self.assertEqual(r.critical_points.shape, (2, 0))
        self.assertFalse(bool(r.valid.any() | r.has_root.any()))
        self.assertTrue(bool((r.depth == 0).all()))
        gradients = torch.autograd.grad(r.depth.sum(), p)
        self.assertTrue(all(bool((x == 0).all()) for x in gradients))
        for row in r.metadata():
            self.assertIsNone(row['root_residual'])
            self.assertIsNone(row['root_bracket_width'])
            self.assertIn('no_detected_root', row['guard_failures'])
            self.assertFalse(row['continuous_error_certified'])
        self.assertEqual(GUARD_NAMES, ('no_detected_root', 'near_potential', 'root_slope',
                                      'nonfinite_root_or_TV', 'monotonic_direction', 'root_residual'))

    def test_full_covariance_parameter_chain_and_nonuniform_weights(self):
        xyz = torch.tensor([[0., 0., 1.], [.1, .05, 1.4], [-.15, -.05, 1.7]],
                           device='cuda', dtype=torch.float64, requires_grad=True)
        log_scale = torch.tensor([[.15, .08, .12], [.18, .13, .1], [.15, .1, .14]],
                                 device='cuda', dtype=torch.float64).log().requires_grad_()
        q = torch.tensor([[1., .1, .2, .05], [1., -.2, .1, .05], [1., .05, -.1, .2]],
                         device='cuda', dtype=torch.float64, requires_grad=True)
        logits = torch.tensor([[3.], [2.], [2.5]], device='cuda', dtype=torch.float64, requires_grad=True)
        raw = (xyz, log_scale, q, logits)
        origin = xyz.new_zeros((3, 3))
        direction = xyz.new_tensor([[0., 0., 1.], [.01, 0., 1.], [-.01, .01, 1.]])
        p = ray_profiles(origin, direction, xyz, log_scale.exp(), q, logits.sigmoid() * (1 - 1e-6))
        batch = stationary_depth_batch(*p, near=.2001, far=8., subdivisions=4)
        reference = torch.cat([stationary_depth(*(x[i:i + 1] for x in p),
                              near=.2001, far=8., subdivisions=4).depth for i in range(3)])
        self.assertTrue(bool(batch.valid.all()))
        self.assert_close(batch.depth, reference)
        weights = xyz.new_tensor([.2, -.7, 1.3])
        gb = torch.autograd.grad((batch.depth * weights).sum(), raw, retain_graph=True)
        gr = torch.autograd.grad((reference * weights).sum(), raw)
        for a, b in zip(gb, gr): self.assert_close(a, b, atol=2e-8, rtol=2e-7)

    def test_chunk_size_and_float32_parameter_cast(self):
        p = self.parameters([[-.31, .27], [-.5, .4]], [[.91, 1.03], [.8, 1.]],
                            [[.57, .52], [.6, .55]], dtype=torch.float32)
        a = stationary_depth_batch(*p, near=-10, far=10, subdivisions=4, sample_chunk=17)
        b = stationary_depth_batch(*p, near=-10, far=10, subdivisions=4, sample_chunk=128)
        self.assertEqual(a.depth.dtype, torch.float64)
        self.assertTrue(bool(a.valid.all() & b.valid.all()))
        self.assert_close(a.depth, b.depth)
        ga = torch.autograd.grad(a.depth.sum(), p)
        gb = torch.autograd.grad(b.depth.sum(), p)
        for x, y in zip(ga, gb):
            self.assertEqual(x.dtype, torch.float32)
            self.assert_close(x, y, atol=1e-6, rtol=1e-6)

    def test_mask_padding_width_and_batch_composition_near_merging_extrema(self):
        # The second row has a nearly merging symmetric pair and an overlapping
        # distant layer; the others have different numbers of critical points.
        means = [[-.5, .1, .8, 1.4], [-.738, .738, 3., 3.0005], [.1, .11, 2., 4.]]
        sigmas = [[.7, .4, .45, .6], [1., 1., .15, .15001], [.005, .006, .1, .2]]
        amplitudes = [[.5, .7, .45, .6], [.6, .6, .24, .25], [.85, .82, .4, .9]]
        weights = [.3, .7, -1.1]
        for refinement in (4, 8, 16):
            reference = []
            for m, s, a in zip(means, sigmas, amplitudes):
                p = self.parameters([m], [s], [a])
                r = stationary_depth(*p, near=-10, far=10, subdivisions=refinement)
                self.assertTrue(bool(r.valid))
                reference.append((r, torch.autograd.grad(r.depth.sum(), p)))
            for ids in ([0], [1], [2], [2, 0, 1]):
                for padding in (0, 3, 9):
                    # Nonzero masked amplitudes verify explicit mask semantics;
                    # they would materially change H if accidentally included.
                    m = [means[i] + [(-2. + k / 7) for k in range(padding)] for i in ids]
                    s = [sigmas[i] + [(.08 + k / 13) for k in range(padding)] for i in ids]
                    a = [amplitudes[i] + [.8] * padding for i in ids]
                    p = self.parameters(m, s, a)
                    member_mask = torch.zeros_like(p[0], dtype=torch.bool)
                    member_mask[:, :4] = True
                    result = stationary_depth_batch(*p, near=-10, far=10,
                        subdivisions=refinement, member_mask=member_mask)
                    w = p[0].new_tensor([weights[i] for i in ids])
                    gradients = torch.autograd.grad((result.depth * w).sum(), p)
                    for row, ref_id in enumerate(ids):
                        expected, expected_grad = reference[ref_id]
                        count = len(expected.critical_points)
                        self.assertEqual(bool(result.valid[row]), bool(expected.valid))
                        self.assertEqual(bool(result.has_root[row]), bool(expected.has_root))
                        self.assertEqual(int(result.critical_count[row]), count)
                        self.assertEqual(int(result.prefix_count[row]), expected.prefix_count)
                        self.assertEqual(int(result.grid_samples[row]), expected.grid_samples)
                        self.assert_close(result.depth[row:row + 1], expected.depth)
                        self.assert_close(result.total[row:row + 1], expected.total)
                        self.assert_close(result.critical_points[row, :count], expected.critical_points)
                        self.assert_close(result.monotonic_directions[row, :count + 1],
                                          expected.monotonic_directions)
                        for actual, desired in zip(gradients, expected_grad):
                            self.assert_close(actual[row, :4], desired[0] * weights[ref_id],
                                              atol=2e-8, rtol=2e-7)
                            self.assertTrue(bool((actual[row, 4:] == 0).all()))

    def test_invalid_subnormal_sigma_has_finite_zero_gradient(self):
        # Physical-domain standardized coordinates remain finite; replacing the
        # missing root by zero makes (0-1)/1e-309 overflow. Backward must mask the
        # invalid row before evaluating that sentinel, not multiply NaNs by 0.
        p = self.parameters([[1.]], [[1e-309]], [[.2]])
        r = stationary_depth_batch(*p, near=1., far=1. + 1e-12, subdivisions=4)
        self.assertFalse(bool(r.valid.any() | r.has_root.any()))
        self.assertTrue(bool(torch.isfinite(r.depth).all()))
        for g in torch.autograd.grad(r.depth.sum(), p):
            self.assertTrue(bool(torch.isfinite(g).all()))
            self.assertTrue(bool((g == 0).all()))

    def test_member_mask_validation_and_real_zero_amplitude(self):
        p = self.parameters([[1., 2.]], [[.1, .2]], [[.9, 0.]])
        for mask in (torch.ones((1, 2), device='cuda', dtype=torch.int64), [[True, False]]):
            with self.assertRaises(TypeError):
                stationary_depth_batch(*p, near=.2001, far=8, member_mask=mask)
        for mask in (torch.ones((1, 1), device='cuda', dtype=torch.bool),
                     torch.ones((1, 2), device='cpu', dtype=torch.bool)):
            with self.assertRaises(ValueError):
                stationary_depth_batch(*p, near=.2001, far=8, member_mask=mask)
        unmasked = stationary_depth_batch(*p, near=.2001, far=8, subdivisions=4)
        explicit = stationary_depth_batch(*p, near=.2001, far=8, subdivisions=4,
                                          member_mask=torch.ones_like(p[0], dtype=torch.bool))
        self.assertTrue(torch.equal(unmasked.grid_samples, explicit.grid_samples))
        self.assert_close(unmasked.depth, explicit.depth)
        a = torch.autograd.grad(unmasked.depth.sum(), p)
        b = torch.autograd.grad(explicit.depth.sum(), p)
        for x, y in zip(a, b): self.assert_close(x, y)
        self.assertGreater(float(a[2][0, 1].abs()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
