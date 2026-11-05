"""Small, deterministic tests; no dataset or training run required.

Run: python -m unittest discover -s tests -p 'test_unimodal_regularizer.py' -v
CPU mathematical tests require only the standard library. Torch reference and
compiled-renderer tests are skipped explicitly when their dependencies are absent.
"""
import math
import random
import unittest

try:
    import torch
except ImportError:
    torch = None


def sign(x):
    return float(x > 0) - float(x < 0)


def loss_weights(weights, min_mass=0.5, epsilon=1e-6):
    if not weights or sum(weights) < min_mass:
        return 0.0, False, [0.0] * len(weights)
    padded = [0.0] + list(weights) + [0.0]
    variation = sum(abs(b - a) for a, b in zip(padded, padded[1:]))
    peak = max(weights)
    excess = max(0.0, variation - 2 * peak)
    denominator = 2 * peak + epsilon
    grads = [0.0] * len(weights)
    if excess > 0:
        peak_index = weights.index(peak)  # earliest maximum, not equal splitting
        for i, weight in enumerate(weights):
            dv = sign(weight - padded[i]) + sign(weight - padded[i + 2])
            dm = float(i == peak_index)
            grads[i] = (dv - 2 * dm) / denominator - 2 * excess * dm / denominator**2
    return excess / denominator, True, grads


def accepted_ray(raw_alphas, powers=None):
    """Raw alphas are conic_opacity.w * exp(power), BEFORE the .99 clip."""
    powers = [0.0] * len(raw_alphas) if powers is None else powers
    transmittance = 1.0
    accepted = []
    for index, (raw, power) in enumerate(zip(raw_alphas, powers)):
        if power > 0:
            continue
        alpha = min(0.99, raw)
        if alpha < 1 / 255:
            continue
        next_t = transmittance * (1 - alpha)
        if next_t < 1e-4:
            break  # terminating Gaussian is NOT included in this renderer
        accepted.append((index, alpha, transmittance, alpha * transmittance))
        transmittance = next_t
    return accepted


def loss_raw_alphas(raw_alphas, powers=None, min_mass=0.5, epsilon=1e-6):
    accepted = accepted_ray(raw_alphas, powers)
    value, valid, gw = loss_weights([r[3] for r in accepted], min_mass, epsilon)
    gradients = [0.0] * len(raw_alphas)
    suffix = 0.0
    for (index, alpha, transmittance, weight), grad_weight in reversed(list(zip(accepted, gw))):
        grad_alpha = transmittance * grad_weight - suffix / (1 - alpha)
        suffix += grad_weight * weight
        # Saturation and its boundary use derivative zero for this new loss.
        if raw_alphas[index] < 0.99:
            gradients[index] = grad_alpha
    return value, valid, gradients



def replay_backward(raw_alphas, powers=None, min_mass=0.5, epsilon=1e-6):
    """Two forward pending-neighbor traversals, preserving exact weight ties."""
    accepted = accepted_ray(raw_alphas, powers)
    gradients = [0.0] * len(raw_alphas)
    loss, valid, _ = loss_weights([row[3] for row in accepted], min_mass, epsilon)
    if not valid or loss <= 0:
        return gradients
    peak = max(row[3] for row in accepted)
    argmax = next(row[0] for row in accepted if row[3] == peak)
    inv = 1 / (2 * peak + epsilon)
    max_coefficient = 2 * inv + 2 * loss * inv
    last_index = accepted[-1][0]
    powers = [0.0] * len(raw_alphas) if powers is None else powers

    def weight_gradient(record, right):
        index, alpha, before, weight, left = record
        g = (sign(weight - left) + sign(weight - right)) * inv
        return g - (max_coefficient if index == argmax else 0)

    total = 0.0
    for pass_index in range(2):
        transmittance, left, prefix, pending = 1.0, 0.0, 0.0, None
        # An extra sentinel finishes the last accepted contribution.
        for index in range(last_index + 2):
            if index <= last_index:
                alpha = min(.99, raw_alphas[index])
                if powers[index] > 0 or alpha < 1 / 255:
                    continue
                weight = transmittance * alpha
            else:
                weight = 0.0
            if pending is not None:
                pi, pa, pt, pw, _ = pending
                g = weight_gradient(pending, weight)
                if pass_index == 0:
                    total += g * pw
                else:
                    prefix += g * pw
                    da = pt * g - (total - prefix) / (1 - pa)
                    gradients[pi] = da if raw_alphas[pi] < .99 else 0.0
            if index <= last_index:
                pending = (index, alpha, transmittance, weight, left)
                transmittance *= 1 - alpha
                left = weight
    return gradients


def alphas_from_weights(weights):
    transmittance = 1.0
    alphas = []
    for weight in weights:
        alphas.append(weight / transmittance)
        transmittance -= weight
    return alphas


def finite_difference(function, values, step=1e-6):
    result = []
    for i in range(len(values)):
        left, right = list(values), list(values)
        left[i] -= step
        right[i] += step
        result.append((function(right) - function(left)) / (2 * step))
    return result


class MathematicalTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, tolerance=2e-7):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=tolerance)

    def test_single_peak_and_plateau_zero(self):
        for weights in ([0.6], [0.1, 0.2, 0.3, 0.1], [0.1, 0.2, 0.2, 0.1], [0.3, 0.2, 0.1]):
            loss, valid, grad = loss_weights(weights)
            self.assertTrue(valid)
            self.assertAlmostEqual(loss, 0, delta=1e-15)
            self.assertVectorClose(grad, [0] * len(weights))

    def test_two_peaks_and_valley_depth(self):
        deep = loss_weights([0.3, 0.05, 0.25])[0]
        shallow = loss_weights([0.3, 0.15, 0.25])[0]
        self.assertGreater(deep, shallow)
        self.assertAlmostEqual(deep, 0.4 / 0.600001)

    def test_gate_empty_and_low_mass(self):
        for weights in ([], [0.01, 0.02, 0.01], [0.2, 0.01, 0.2]):
            loss, valid, grad = loss_weights(weights)
            self.assertEqual(loss, 0)
            self.assertFalse(valid)
            self.assertEqual(grad, [0] * len(weights))
        self.assertTrue(loss_weights([0.3, 0.05, 0.15])[1])

    def test_weight_gradient_finite_difference(self):
        weights = [0.26, 0.08, 0.32, 0.04]
        self.assertVectorClose(loss_weights(weights)[2], finite_difference(lambda w: loss_weights(w)[0], weights))

    def test_alpha_gradient_includes_later_transmittance(self):
        alphas = alphas_from_weights([0.26, 0.08, 0.32, 0.04])
        _, _, gradient = loss_raw_alphas(alphas)
        numerical = finite_difference(lambda a: loss_raw_alphas(a)[0], alphas)
        self.assertVectorClose(gradient, numerical)
        # A local-only derivative would omit a nonzero suffix contribution.
        accepted = accepted_ray(alphas)
        gw = loss_weights([r[3] for r in accepted])[2]
        self.assertGreater(abs(gradient[0] - accepted[0][2] * gw[0]), 0.1)

    def test_random_smooth_alpha_gradients(self):
        rng = random.Random(812)
        for _ in range(100):
            alphas = [rng.uniform(0.08, 0.5) for _ in range(7)]
            self.assertVectorClose(loss_raw_alphas(alphas)[2], finite_difference(lambda a: loss_raw_alphas(a)[0], alphas), 3e-6)

    def test_two_forward_replays_match_full_derivative(self):
        rng = random.Random(814)
        for _ in range(100):
            raw = [rng.uniform(.0001, 1.2) for _ in range(30)]
            powers = [1.0 if rng.random() < .15 else 0.0 for _ in raw]
            expected = loss_raw_alphas(raw, powers)[2]
            self.assertVectorClose(replay_backward(raw, powers), expected, 3e-8)

    def test_tie_uses_first_peak_and_abs_zero(self):
        weights = [0.25, 0.05, 0.25]
        loss, valid, gradient = loss_weights(weights)
        self.assertTrue(valid)
        self.assertGreater(loss, 0)
        self.assertLess(gradient[0], gradient[2])
        self.assertEqual(sign(0), 0)
        tied_raw = alphas_from_weights(weights)
        self.assertVectorClose(replay_backward(tied_raw), loss_raw_alphas(tied_raw)[2])
        # The equal adjacent values contribute zero through their shared edge.
        weights = [0.3, 0.1, 0.1, 0.2]
        gradient = loss_weights(weights)[2]
        self.assertAlmostEqual(gradient[1], gradient[2])

    def test_skip_clip_and_early_termination(self):
        raw = [0.25, 0.001, 0.9, 0.08, 1.2, 0.8, 0.9, 0.3]
        powers = [0, 0, 1, 0, 0, 0, 0, 0]
        accepted = accepted_ray(raw, powers)
        self.assertEqual([r[0] for r in accepted], [0, 3, 4, 5, 6])
        self.assertEqual(accepted[2][1], 0.99)
        gradient = loss_raw_alphas(raw, powers)[2]
        for index in [1, 2, 4, 7]:
            self.assertEqual(gradient[index], 0)
        self.assertVectorClose(gradient, finite_difference(lambda a: loss_raw_alphas(a, powers)[0], raw), 3e-6)


@unittest.skipIf(torch is None, 'PyTorch not installed')
class TorchReferenceTests(unittest.TestCase):
    def test_autograd_agrees_with_full_alpha_derivative(self):
        raw = alphas_from_weights([0.26, 0.08, 0.32, 0.04])
        alpha = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        before = torch.cat([alpha.new_ones(1), (1 - alpha).cumprod(0)[:-1]])
        weights = alpha * before
        padded = torch.cat([weights.new_zeros(1), weights, weights.new_zeros(1)])
        variation = (padded[1:] - padded[:-1]).abs().sum()
        peak = weights[weights.detach().argmax()]
        loss = (variation - 2 * peak).clamp(min=0) / (2 * peak + 1e-6)
        loss.backward()
        for actual, expected in zip(alpha.grad.tolist(), loss_raw_alphas(raw)[2]):
            self.assertAlmostEqual(actual, expected, delta=1e-10)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), 'CUDA PyTorch unavailable')
class CudaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except ImportError as exc:
            raise unittest.SkipTest('compiled rasterizer unavailable: ' + str(exc))
        cls.Settings = GaussianRasterizationSettings
        cls.Rasterizer = GaussianRasterizer

    def render(self, enabled, opacities, require_depth=False, means=None, scales=None):
        device = 'cuda'
        settings = self.Settings(
            image_height=1, image_width=1, tanfovx=1., tanfovy=1., kernel_size=0.,
            bg=torch.zeros(3, device=device), scale_modifier=1.,
            viewmatrix=torch.eye(4, device=device), projmatrix=torch.eye(4, device=device),
            sh_degree=0, campos=torch.zeros(3, device=device), prefiltered=False,
            require_depth=require_depth, debug=True, unimodal_enabled=enabled,
            unimodal_min_mass=.5, unimodal_epsilon=1e-6)
        count = opacities.shape[0]
        if means is None:
            means = torch.tensor([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.], [0., 0., 4.]], device=device)[:count]
        if scales is None:
            scales = torch.tensor([[.4, .35, .06]] * count, device=device).reshape(count, 3)
        return self.Rasterizer(settings)(
            means3D=means, means2D=torch.zeros_like(means), opacities=opacities,
            colors_precomp=torch.ones((count, 3), device=device) * .3,
            scales=scales,
            rotations=torch.tensor([[1., 0., 0., 0.]] * count, device=device).reshape(count, 4))

    def test_enabled_disabled_and_cuda_opacity_gradient(self):
        raw = alphas_from_weights([0.26, 0.08, 0.32, 0.04])
        opacity = torch.tensor(raw, device='cuda', dtype=torch.float32).reshape(-1, 1).requires_grad_()
        disabled = self.render(False, opacity)
        enabled = self.render(True, opacity)
        self.assertEqual(len(disabled), 6)
        self.assertEqual(len(enabled), 8)
        for a, b in zip(disabled, enabled[:6]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(enabled[6].shape, (1, 1, 1))
        self.assertTrue(bool(enabled[7].item()))
        self.assertGreater(enabled[6].item(), 0)
        enabled[6].sum().backward()
        actual = opacity.grad.flatten().tolist()
        numerical = finite_difference(lambda v: self.render(True, torch.tensor(v, device='cuda').reshape(-1, 1))[6].sum().item(), raw, step=2e-4)
        for a, b in zip(actual, numerical):
            self.assertAlmostEqual(a, b, delta=3e-3)
        # Geometry enabled follows the same visibility sequence and loss.
        torch.testing.assert_close(enabled[6], self.render(True, opacity, require_depth=True)[6])


    def test_regularizer_only_geometry_gradients(self):
        raw = alphas_from_weights([0.26, 0.08, 0.32, 0.04])
        opacity = torch.tensor(raw, device='cuda').reshape(-1, 1)
        # Distinct scales avoid eigenvector degeneracies; tiny off-axis centers
        # make mean and covariance derivatives nontrivial without changing order.
        means = torch.tensor([[.025, -.03, 1.], [-.035, .02, 2.],
                              [.02, .035, 3.], [-.03, -.025, 4.]],
                             device='cuda', requires_grad=True)
        scales = torch.tensor([[.4, .35, .06]] * 4, device='cuda', requires_grad=True)
        result = self.render(True, opacity, require_depth=False, means=means, scales=scales)
        self.assertTrue(bool(result[7].item()))
        self.assertGreater(result[6].item(), 0)
        result[6].sum().backward()
        for parameter, name in ((means, 'means3D'), (scales, 'scales')):
            actual = parameter.grad.detach().flatten()
            self.assertTrue(bool(torch.isfinite(actual).all()), name)
            self.assertGreater(actual.abs().max().item(), 1e-5, name)
            raw_values = parameter.detach().flatten().tolist()

            def evaluate(values):
                changed = torch.tensor(values, device='cuda').reshape(4, 3)
                m = changed if name == 'means3D' else means.detach()
                scale = changed if name == 'scales' else scales.detach()
                return self.render(True, opacity, require_depth=False, means=m, scales=scale)[6].sum().item()

            numerical = torch.tensor(finite_difference(evaluate, raw_values, step=5e-4), device='cuda')
            torch.testing.assert_close(actual, numerical, rtol=3e-2, atol=4e-3,
                                       msg=name + ' regularizer-only central finite differences')

    def test_empty_and_insufficient_mass_maps(self):
        for count, alpha in ((0, .0), (4, .01)):
            opacity = torch.full((count, 1), alpha, device='cuda', requires_grad=True)
            result = self.render(True, opacity, require_depth=False)
            self.assertEqual(result[6].shape, (1, 1, 1))
            self.assertEqual(result[6].item(), 0)
            self.assertEqual(result[7].item(), 0)
            self.assertFalse(result[7].requires_grad)
            result[6].sum().backward()
            self.assertTrue(bool(torch.isfinite(opacity.grad).all()))
            self.assertEqual(opacity.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
