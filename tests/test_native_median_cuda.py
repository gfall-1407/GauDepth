"""Small native CUDA contract check for continuous median sampling."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submodules" / "diff-gaussian-rasterization"))
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    GaussianRasterizer = None


@unittest.skipUnless(torch.cuda.is_available() and GaussianRasterizer is not None, "local CUDA extension unavailable")
class NativeMedianSampleTest(unittest.TestCase):
    def test_same_query_sequence_and_selected_geometry_gradient(self):
        device = "cuda"
        projection = torch.zeros((4, 4), device=device)
        projection[0, 0] = projection[1, 1] = 1.0
        projection[2, 2] = 100.0 / 99.9
        projection[2, 3] = -10.0 / 99.9
        projection[3, 2] = 1.0
        settings = GaussianRasterizationSettings(
            image_height=65, image_width=65, tanfovx=1.0, tanfovy=1.0,
            kernel_size=0.0, bg=torch.zeros(3, device=device), scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device), projmatrix=projection.T.contiguous(),
            sh_degree=0, campos=torch.zeros(3, device=device),
            prefiltered=False, require_depth=True, debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=device, requires_grad=True)
        opacity = torch.tensor([[0.2], [0.2]], device=device, requires_grad=True)
        scales = torch.full((2, 3), 0.2, device=device, requires_grad=True)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device, requires_grad=True)
        query = torch.tensor([[[0.0, 0.0, 2.5]]], device=device, requires_grad=True)
        kwargs = dict(means3D=means, opacities=opacity, scales=scales, rotations=rotations)
        expected, expected_inside = rasterizer.sample_depth(query, use_median=False, **kwargs)
        median, median_inside = rasterizer.sample_depth(query, use_median=True, **kwargs)
        rendered = rasterizer(
            means3D=means, means2D=torch.zeros_like(means), opacities=opacity,
            colors_precomp=torch.ones((2, 3), device=device),
            scales=scales, rotations=rotations,
        )
        self.assertTrue(expected_inside.item() and median_inside.item())
        self.assertGreater(median[0, 0, 2].item(), expected[0, 0, 2].item())
        self.assertAlmostEqual(median[0, 0, 2].item(), 3.0, places=4)
        self.assertAlmostEqual(expected[0, 0, 2].item(), rendered[2][0, 32, 32].item(), places=4)
        self.assertAlmostEqual(median[0, 0, 2].item(), rendered[3][0, 32, 32].item(), places=4)
        grad_means, grad_opacity, grad_query = torch.autograd.grad(
            median[..., 2].sum(), (means, opacity, query), allow_unused=True)
        self.assertAlmostEqual(grad_means[0].abs().sum().item(), 0.0, places=5)
        self.assertGreater(grad_means[1].abs().sum().item(), 0.0)
        self.assertAlmostEqual(grad_opacity.abs().sum().item(), 0.0, places=5)
        self.assertIsNotNone(grad_query)

        offcenter = torch.tensor([[[0.02, 0.01, 2.5]]], device=device, requires_grad=True)
        off_output, _ = rasterizer.sample_depth(offcenter, use_median=True, **kwargs)
        analytic = torch.autograd.grad(off_output[..., 0].sum(), offcenter)[0][0, 0, 0].item()
        step = 1e-3
        with torch.no_grad():
            plus = offcenter.clone()
            minus = offcenter.clone()
            plus[..., 0] += step
            minus[..., 0] -= step
            plus_value = rasterizer.sample_depth(plus, use_median=True, **kwargs)[0][..., 0].item()
            minus_value = rasterizer.sample_depth(minus, use_median=True, **kwargs)[0][..., 0].item()
        finite_difference = (plus_value - minus_value) / (2 * step)
        self.assertAlmostEqual(analytic, finite_difference, delta=2e-2)

        front_opacity = torch.tensor([[0.8], [0.8]], device=device, requires_grad=True)
        front_kwargs = dict(kwargs, opacities=front_opacity)
        front_median, _ = rasterizer.sample_depth(query, use_median=True, **front_kwargs)
        self.assertAlmostEqual(front_median[0, 0, 2].item(), 2.0, places=4)
        front_grad = torch.autograd.grad(front_median[..., 2].sum(), means)[0]
        self.assertGreater(front_grad[0].abs().sum().item(), 0.0)
        self.assertAlmostEqual(front_grad[1].abs().sum().item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
