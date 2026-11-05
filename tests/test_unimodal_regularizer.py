"""Small, deterministic tests; no dataset or training run required.

Run: python -m unittest discover -s tests -p 'test_unimodal_regularizer.py' -v
CPU mathematical tests require only the standard library. Torch reference and
compiled-renderer tests are skipped explicitly when their dependencies are absent.
For a required GPU run, set GAUDEPTH_REQUIRE_CUDA_TESTS=1 (missing CUDA/build
then fails). Build this repository extension with `python setup.py build_ext`
from submodules/diff-gaussian-rasterization using the same Python as the tests.
The loader uses repository Python sources and an in-place or build/lib* binary;
it never falls back to site-packages. GAUDEPTH_RASTERIZER_BUILD may select a
specific directory containing _C when multiple compatible builds exist.
Geometry objectives use a fixed 4x4 interior of a 64x64 perspective image.
The float64 ray reference checks smooth branches and matches GPU forward maps;
finite differences assert the same branches at both perturbation endpoints.
Exact peak ties are tested separately against the selected first-peak subgradient.
"""
import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import sys
import random
import unittest

try:
    import torch
except ImportError:
    torch = None


if os.environ.get('GAUDEPTH_REQUIRE_CUDA_TESTS') == '1' and (torch is None or not torch.cuda.is_available()):
    raise RuntimeError('CUDA tests required, but CUDA PyTorch is unavailable')


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
        # Never silently import a different renderer from site-packages.
        root = Path(__file__).resolve().parents[1] / 'submodules' / 'diff-gaussian-rasterization'
        package = root / 'diff_gaussian_rasterization'
        override = os.environ.get('GAUDEPTH_RASTERIZER_BUILD')
        directories = [Path(override)] if override else [package] + list((root / 'build').glob('lib*/diff_gaussian_rasterization'))
        candidates = [d for d in directories if any((d / ('_C' + suffix)).is_file()
                      for suffix in importlib.machinery.EXTENSION_SUFFIXES)]
        if not candidates:
            message = 'Build this repository rasterizer first (python setup.py build_ext); no compatible local _C extension found'
            if os.environ.get('GAUDEPTH_REQUIRE_CUDA_TESTS') == '1':
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        if package in candidates:
            artifact = package
        elif len(candidates) == 1:
            artifact = candidates[0]
        else:
            raise RuntimeError('Multiple local builds: set GAUDEPTH_RASTERIZER_BUILD to the chosen package directory')
        name = 'diff_gaussian_rasterization'
        if name in sys.modules:
            raise RuntimeError('Renderer already imported; run this test in a fresh Python process')
        spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                    submodule_search_locations=[str(package), str(artifact)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise  # ABI/build errors must fail, rather than turn into skips.
        print('Repository rasterizer: ' + str(module.__file__), file=sys.stderr)
        print('CUDA extension: ' + str(module._C.__file__), file=sys.stderr)
        cls.Settings = module.GaussianRasterizationSettings
        cls.Rasterizer = module.GaussianRasterizer

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


    def perspective_scene(self):
        means = torch.tensor([[-.020, .015, 1.20], [.012, -.010, 1.90],
                              [-.015, -.020, 2.60], [.020, .015, 3.40]], device='cuda')
        scales = torch.tensor([[.12, .10, .05], [.14, .11, .06],
                               [.16, .12, .07], [.18, .13, .08]], device='cuda')
        opacity = torch.tensor(alphas_from_weights([.26, .08, .32, .04]), device='cuda').reshape(4, 1)
        return means, scales, opacity

    def perspective_render(self, enabled, means, scales, opacity):
        # Same projection convention as Camera/getProjectionMatrix, including transpose.
        tangent = math.tan(math.radians(60) / 2)
        near, far = .01, 100.
        projection = torch.zeros((4, 4), device='cuda')
        projection[0, 0] = projection[1, 1] = 1 / tangent
        projection[3, 2] = 1
        projection[2, 2] = far / (far - near)
        projection[2, 3] = -far * near / (far - near)
        settings = self.Settings(
            image_height=64, image_width=64, tanfovx=tangent, tanfovy=tangent, kernel_size=0.,
            bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
            viewmatrix=torch.eye(4, device='cuda'), projmatrix=projection.T.contiguous(),
            sh_degree=0, campos=torch.zeros(3, device='cuda'), prefiltered=False,
            require_depth=False, debug=False, unimodal_enabled=enabled,
            unimodal_min_mass=.5, unimodal_epsilon=1e-6)
        return self.Rasterizer(settings)(
            means3D=means, means2D=torch.zeros_like(means), opacities=opacity,
            colors_precomp=torch.full((4, 3), .3, device='cuda'), scales=scales,
            rotations=torch.tensor([[1., 0., 0., 0.]] * 4, device='cuda'))

    def ray_reference(self, means, scales, opacity):
        # Independent float64 pinhole covariance reference, specialized to this
        # identity-view, identity-rotation, zero-filter fixture. Only the fixed
        # interior pixels participate; all four alphas are safely above cutoff.
        m, s = means.detach().cpu().double(), scales.detach().cpu().double()
        focal = 32 / math.tan(math.radians(60) / 2)
        pixels = torch.tensor([[x, y] for y in range(30, 34) for x in range(30, 34)], dtype=torch.float64)
        alphas = []
        for center, scale, op in zip(m, s, opacity.detach().cpu().double().flatten()):
            x, y, z = center.tolist()
            jacobian = torch.tensor([[focal/z, 0., -focal*x/z**2],
                                     [0., focal/z, -focal*y/z**2]], dtype=torch.float64)
            covariance = jacobian @ torch.diag(scale.square()) @ jacobian.T
            delta = pixels - (31.5 + focal * center[:2] / z)
            power = -.5 * torch.einsum('ni,ij,nj->n', delta, torch.linalg.inv(covariance), delta)
            alphas.append(op * power.exp())
        alpha = torch.stack(alphas, dim=1)
        self.assertGreater(alpha.min().item(), 1/255 + .01)
        self.assertLess(alpha.max().item(), .99 - .01)
        transmittance = torch.cat([torch.ones((16, 1), dtype=torch.float64),
                                  (1-alpha).cumprod(1)[:, :-1]], dim=1)
        weights = transmittance * alpha
        peaks, indices = weights.max(1)
        sorted_weights = weights.sort(1, descending=True).values
        self.assertGreater((sorted_weights[:, 0]-sorted_weights[:, 1]).min().item(), 1e-3)
        # TV abs signs, max branch, cutoff, early-stop and mass gate must be smooth.
        self.assertGreater((weights[:, 1:]-weights[:, :-1]).abs().min().item(), .01)
        mass = 1-(1-alpha).prod(1)
        self.assertGreater(mass.min().item(), .55)
        self.assertGreater((1-alpha).prod(1).min().item(), .0001)
        padded = torch.cat([torch.zeros((16, 1), dtype=torch.float64), weights,
                            torch.zeros((16, 1), dtype=torch.float64)], dim=1)
        excess = (padded[:, 1:]-padded[:, :-1]).abs().sum(1)-2*peaks
        self.assertGreater(excess.min().item(), .01)
        loss = excess / (2*peaks + 1e-6)
        return m.norm(dim=1).argsort(), indices, mass, loss, (weights[:, 1:]-weights[:, :-1]).sign()

    def check_perspective_geometry(self, enabled):
        means, scales, opacity = self.perspective_scene()
        means.requires_grad_(); scales.requires_grad_()
        base = self.perspective_render(enabled, means, scales, opacity)
        # Probe the same gate even when the differentiated objective is legacy RGB.
        probe = base if enabled else self.perspective_render(True, means.detach(), scales.detach(), opacity)
        reference = self.ray_reference(means, scales, opacity)
        torch.testing.assert_close(probe[4][0, 30:34, 30:34].flatten().cpu().double(), reference[2], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(probe[6][0, 30:34, 30:34].flatten().cpu().double(), reference[3], rtol=2e-5, atol=2e-6)
        self.assertTrue(bool(probe[7][0, 30:34, 30:34].all()))
        self.assertTrue(bool((base[1] > 0).all()))
        def objective(output):
            return (output[6] if enabled else output[0])[:, 30:34, 30:34].sum()
        objective(base).backward()
        step = 5e-4
        for parameter, name in ((means, 'means3D'), (scales, 'scales')):
            actual = parameter.grad.detach().flatten()
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertGreater(actual.abs().max().item(), 1e-5)
            def evaluate(values):
                changed = torch.tensor(values, device='cuda').reshape(4, 3)
                m = changed if name == 'means3D' else means.detach()
                s = changed if name == 'scales' else scales.detach()
                state = self.ray_reference(m, s, opacity)
                self.assertTrue(torch.equal(state[0], reference[0]), 'depth sorting changed')
                self.assertTrue(torch.equal(state[1], reference[1]), 'max contribution branch changed')
                self.assertTrue(torch.equal(state[4], reference[4]), 'TV absolute-value branch changed')
                output = self.perspective_render(enabled, m, s, opacity)
                self.assertTrue(torch.equal(output[1], base[1]), 'radii/visibility changed')
                gate = output if enabled else self.perspective_render(True, m, s, opacity)
                self.assertTrue(torch.equal(gate[7][:, 30:34, 30:34], probe[7][:, 30:34, 30:34]), 'gate changed on objective support')
                torch.testing.assert_close(gate[4][0, 30:34, 30:34].flatten().cpu().double(), state[2], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(gate[6][0, 30:34, 30:34].flatten().cpu().double(), state[3], rtol=2e-5, atol=2e-6)
                return objective(output).item()
            numerical = torch.tensor(finite_difference(evaluate, parameter.detach().flatten().tolist(), step), device='cuda')
            # Preserve the original tolerance; branch stability is asserted above.
            torch.testing.assert_close(actual, numerical, rtol=3e-2, atol=4e-3)
            print('Perspective {} {} max absolute error: {:.6g}'.format('regularizer' if enabled else 'legacy RGB', name, (actual-numerical).abs().max().item()), file=sys.stderr)

    def test_perspective_legacy_geometry_gradients(self):
        self.check_perspective_geometry(False)

    def test_perspective_regularizer_geometry_gradients(self):
        self.check_perspective_geometry(True)

    def test_cuda_tied_peak_uses_first_subgradient(self):
        # Exactly representable float32 weights after the transmittance recurrence.
        raw = [.25, 1/6, .4]
        opacity = torch.tensor(raw, device='cuda').reshape(3, 1).requires_grad_()
        alpha = opacity.detach().flatten()
        weights = alpha * torch.cat([alpha.new_ones(1), (1-alpha).cumprod(0)[:-1]])
        self.assertEqual(weights[0].item(), weights[2].item())
        output = self.render(True, opacity)
        output[6].sum().backward()
        expected = loss_raw_alphas(raw)[2]
        torch.testing.assert_close(opacity.grad.flatten(), torch.tensor(expected, device='cuda'), rtol=0, atol=3e-3)
        # Exact ties have a selected subgradient; central differences are not its oracle.
        self.assertLess(loss_weights([.25, .125, .25])[2][0], loss_weights([.25, .125, .25])[2][2])

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
