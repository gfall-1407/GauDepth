"""Small CUDA smoke/parity check for the optional accepted-member query.

Run after building this exact extension: python tests/gpu_support_smoke.py
This is a synthetic renderer test; it does not run training or load DTU data.
"""

import math
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer,
)


H = W = 48
TAN_FOV = 0.5


def point_at(x, y, z=2.0):
    return [((x + 0.5) * 2 / W - 1) * z * TAN_FOV,
            ((y + 0.5) * 2 / H - 1) * z * TAN_FOV, z]


def make_rasterizer():
    projection = torch.zeros((4, 4), device="cuda")
    projection[0, 0] = projection[1, 1] = 1 / TAN_FOV
    projection[2, 2] = 10 / (10 - 0.2)
    projection[2, 3] = -10 * 0.2 / (10 - 0.2)
    projection[3, 2] = 1
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=TAN_FOV, tanfovy=TAN_FOV,
        kernel_size=0.0, bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=projection.T.contiguous(), sh_degree=0,
        campos=torch.zeros(3, device="cuda"), prefiltered=False,
        require_depth=True, debug=False,
    )
    return GaussianRasterizer(settings)


def make_gaussians():
    # Enough weak overlapping members to exercise K=128 overflow before T stop.
    centers = [point_at(16, 24)] * 300 + [point_at(32, 24)] * 4
    means = torch.tensor(centers, device="cuda", dtype=torch.float32)
    p = means.shape[0]
    colors = torch.full((p, 3), 0.4, device="cuda")
    scales = torch.tensor([[0.2, 0.2, 0.02]], device="cuda").repeat(p, 1)
    rotations = torch.zeros((p, 4), device="cuda")
    rotations[:, 0] = 1.0
    return means, colors, scales, rotations


def render(raster, means, colors, scales, rotations, opacity, xy=None, k=128):
    return raster(means3D=means, means2D=torch.zeros_like(means),
                  colors_precomp=colors, opacities=opacity,
                  scales=scales, rotations=rotations,
                  support_xy=xy, support_k=k)


def sample(raster, points, means, scales, rotations, opacity, xy=None, k=128):
    return raster.sample_depth(points, means, opacity, scales=scales,
                               rotations=rotations, support_xy=xy, support_k=k)


def check_members(ids, alpha, count, overflow, k):
    assert ids.shape == alpha.shape == (count.numel(), k)
    assert overflow.shape == count.shape
    valid = torch.arange(k, device="cuda")[None, :] < count[:, None].clamp(max=k)
    assert torch.all(ids[~valid] == -1)
    assert torch.all(alpha[~valid] == 0)
    assert torch.all((ids[valid] >= 0) & (ids[valid] < 304))
    assert torch.all((alpha[valid] >= 1 / 255) & (alpha[valid] <= 0.99))
    assert torch.equal(overflow, count > k)


def main():
    assert torch.cuda.is_available(), "CUDA/PyTorch extension required"
    assert torch.cuda.current_stream().cuda_stream == torch.cuda.default_stream().cuda_stream
    raster = make_rasterizer()
    means, colors, scales, rotations = make_gaussians()
    xy = torch.tensor([
        [15, 24], [16, 24], [31, 24], [32, 24],
        [15.5, 24], [15.75, 24], [31.5, 24],
        [-1, 24], [W, 24], [math.nan, 24],
    ], device="cuda", dtype=torch.float32)
    points = torch.tensor([point_at(float(x), 24) for x in
                           (15, 16, 31, 32, 15.5, 15.75, 31.5)], device="cuda")

    opacity0 = torch.full((means.shape[0], 1), 0.02, device="cuda", requires_grad=True)
    opacity1 = opacity0.detach().clone().requires_grad_()
    base = render(raster, means, colors, scales, rotations, opacity0)
    queried = render(raster, means, colors, scales, rotations, opacity1, xy)
    for a, b in zip(base, queried[:6]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for result in (base, queried):
        (result[0].mean() + result[2].mean() + result[4].mean()).backward()
    torch.testing.assert_close(opacity0.grad, opacity1.grad, rtol=0, atol=0)

    rid, ra, rc, ro = queried[6:]
    check_members(rid, ra, rc, ro, 128)
    assert rc[0] > 128 and ro[0], "synthetic overlap should exercise full count"
    assert torch.all(rc[-3:] == 0) and torch.all(rid[-3:] == -1)
    empty = render(raster, means, colors, scales, rotations, opacity0.detach(), xy[:0])
    assert empty[6].shape == (0, 128) and empty[8].shape == (0,)

    opacity2 = opacity0.detach().clone().requires_grad_()
    opacity3 = opacity0.detach().clone().requires_grad_()
    sbase = sample(raster, points, means, scales, rotations, opacity2)
    squery = sample(raster, points, means, scales, rotations, opacity3, xy)
    for a, b in zip(sbase, squery[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sbase[0].mean().backward()
    squery[0].mean().backward()
    torch.testing.assert_close(opacity2.grad, opacity3.grad, rtol=0, atol=0)
    sid, sa, sc, so = squery[2:]
    check_members(sid, sa, sc, so, 128)
    assert torch.equal(rc[:4], sc[:4])  # Integer pixels use the same tile.
    assert torch.equal(rid[:4], sid[:4])
    torch.testing.assert_close(ra[:4], sa[:4], rtol=0, atol=0)
    assert torch.all(sc[-3:] == 0) and torch.all(sid[-3:] == -1)
    sempty = sample(raster, points, means, scales, rotations, opacity2.detach(), xy[:0])
    assert sempty[2].shape == (0, 128) and sempty[4].shape == (0,)
    # Fractional samples are evaluated at x=15.5/15.75/31.5 without rounding.
    assert torch.all(sc[4:7] > 0)
    torch.cuda.synchronize()
    print("PASS: native outputs/backward, integer parity, fractional samples, overflow, invalid queries")


if __name__ == "__main__":
    main()
