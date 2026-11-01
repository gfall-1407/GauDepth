"""Small CUDA smoke/parity check for the optional accepted-member query.

Run after building this exact extension: python tests/gpu_support_smoke.py
This is a synthetic renderer test; it does not run training or load DTU data.
"""

import math
import torch
from diff_gaussian_rasterization import _C

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
    centers = [point_at(16, 24)] * 300 + [point_at(32, 24)] * 4 + [point_at(30, 24)]
    means = torch.tensor(centers, device="cuda", dtype=torch.float32)
    p = means.shape[0]
    colors = torch.full((p, 3), 0.4, device="cuda")
    scales = torch.tensor([[0.2, 0.2, 0.02]], device="cuda").repeat(p, 1)
    scales[-1, :2] = 0.19
    rotations = torch.zeros((p, 4), device="cuda")
    rotations[:, 0] = 1.0
    opacity = torch.full((p, 1), 0.02, device="cuda")
    opacity[-1] = 0.9  # Accepted at 15.75 from sample tile 1, absent in render tile 0.
    return means, colors, scales, rotations, opacity


def render(raster, means, colors, scales, rotations, opacity, xy=None, k=128):
    return raster(means3D=means, means2D=torch.zeros_like(means),
                  colors_precomp=colors, opacities=opacity,
                  scales=scales, rotations=rotations,
                  support_xy=xy, support_k=k)


def sample(raster, points, means, scales, rotations, opacity, xy=None, k=128):
    return raster.sample_depth(points, means, opacity, scales=scales,
                               rotations=rotations, support_xy=xy, support_k=k)


def check_members(ids, alpha, count, overflow, k, p):
    assert ids.shape == alpha.shape == (count.numel(), k)
    assert overflow.shape == count.shape
    valid = torch.arange(k, device="cuda")[None, :] < count[:, None].clamp(max=k)
    assert torch.all(ids[~valid] == -1)
    assert torch.all(alpha[~valid] == 0)
    assert torch.all((ids[valid] >= 0) & (ids[valid] < p))
    assert torch.all((alpha[valid] >= 1 / 255) & (alpha[valid] <= 0.99))
    assert torch.equal(overflow, count > k)


def assert_grad_close(label, left, right):
    assert left.grad is not None and right.grad is not None, label
    delta = (left.grad - right.grad).abs().max().item()
    print(f"{label} gradient maxdiff={delta:.3g}")
    torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=2e-5)


def native_sample_lists(raster, points, means, opacity, scales, rotations):
    """Read original native sample binning buffers, not the support query output."""
    s = raster.raster_settings
    raw = _C.sample_rasterized_depth(
        points.detach(), means.detach(), opacity.detach(), scales.detach(),
        rotations.detach(), s.scale_modifier, torch.empty(0), s.viewmatrix,
        s.projmatrix, s.tanfovx, s.tanfovy, 0.0, H, W, s.campos,
        s.prefiltered, s.debug,
    )
    rendered, binning, tiles = raw[0], raw[5], raw[8]
    tile_count = ((W + 15) // 16) * ((H + 15) // 16)
    assert binning.data_ptr() % 128 == tiles.data_ptr() % 128 == 0
    sorted_ids = binning[:rendered * 4].view(torch.int32).cpu().tolist()
    ranges = tiles[:tile_count * 8].view(torch.int32).reshape(-1, 2).cpu().tolist()
    return sorted_ids, ranges


def independent_alpha(xy, mean, scale, opacity):
    """Projected diagonal 3D covariance for this identity-camera/quaternion scene."""
    px, py = xy
    mx, my, z = mean
    sx, sy, sz = scale
    f = W / (2 * TAN_FOV)
    center_x = (mx / (z * TAN_FOV) + 1) * W / 2 - 0.5
    center_y = (my / (z * TAN_FOV) + 1) * H / 2 - 0.5
    jxz, jyz = -f * mx / (z * z), -f * my / (z * z)
    cxx = (f * sx / z) ** 2 + (jxz * sz) ** 2
    cyy = (f * sy / z) ** 2 + (jyz * sz) ** 2
    cxy = jxz * jyz * sz * sz
    det = cxx * cyy - cxy * cxy
    dx, dy = center_x - px, center_y - py
    power = -0.5 * ((cyy * dx * dx - 2 * cxy * dx * dy + cxx * dy * dy) / det)
    return min(0.99, opacity * math.exp(power))


def oracle_sample(xy, sorted_ids, ranges, means, scales, opacity):
    px, py = xy
    tile = min((H - 1) // 16, int((py + 0.5) / 16)) * ((W + 15) // 16) + \
           min((W - 1) // 16, int((px + 0.5) / 16))
    begin, end = ranges[tile]
    ids, alphas, transmittance = [], [], 1.0
    for gid in sorted_ids[begin:end]:
        a = independent_alpha(xy, means[gid], scales[gid], opacity[gid])
        if a < 1 / 255:
            continue
        next_t = transmittance * (1 - a)
        if next_t < 1e-4:
            break
        ids.append(gid)
        alphas.append(a)
        transmittance = next_t
    return ids, alphas


def main():
    assert torch.cuda.is_available(), "CUDA/PyTorch extension required"
    assert torch.cuda.current_stream().cuda_stream == torch.cuda.default_stream().cuda_stream
    raster = make_rasterizer()
    means, colors, scales, rotations, opacity = make_gaussians()
    xy = torch.tensor([
        [15, 24], [16, 24], [31, 24], [32, 24],
        [15.5, 24], [15.75, 24], [31.5, 24],
        [-1, 24], [W, 24], [math.nan, 24],
    ], device="cuda", dtype=torch.float32)
    points = torch.tensor([point_at(float(x), 24) for x in
                           (15, 16, 31, 32, 15.5, 15.75, 31.5)], device="cuda")

    m0, m1 = [means.clone().requires_grad_() for _ in range(2)]
    s0, s1 = [scales.clone().requires_grad_() for _ in range(2)]
    q0, q1 = [rotations.clone().requires_grad_() for _ in range(2)]
    o0, o1 = [opacity.clone().requires_grad_() for _ in range(2)]
    base = render(raster, m0, colors, s0, q0, o0)
    queried = render(raster, m1, colors, s1, q1, o1, xy)
    for a, b in zip(base, queried[:6]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for result in (base, queried):
        (result[0].mean() + result[2].mean() + result[4].mean()).backward()
    for name, a, b in (("render.xyz", m0, m1), ("render.scale", s0, s1),
                       ("render.quat", q0, q1), ("render.opacity", o0, o1)):
        assert_grad_close(name, a, b)

    rid, ra, rc, ro = queried[6:]
    check_members(rid, ra, rc, ro, 128, means.shape[0])
    assert rc[0].item() == 300 and ro[0].item(), "left tile must count all 300 members"
    assert torch.all(rc[-3:] == 0) and torch.all(rid[-3:] == -1)
    full = render(raster, means, colors, scales, rotations, opacity, xy[:1], k=320)
    assert full[8][0].item() == 300 and not full[9][0].item()
    assert set(full[6][0, :300].cpu().tolist()) == set(range(300))
    empty = render(raster, means, colors, scales, rotations, opacity, xy[:0])
    assert empty[6].shape == (0, 128) and empty[8].shape == (0,)

    m2, m3 = [means.clone().requires_grad_() for _ in range(2)]
    s2, s3 = [scales.clone().requires_grad_() for _ in range(2)]
    q2, q3 = [rotations.clone().requires_grad_() for _ in range(2)]
    o2, o3 = [opacity.clone().requires_grad_() for _ in range(2)]
    p2, p3 = [points.clone().requires_grad_() for _ in range(2)]
    sbase = sample(raster, p2, m2, s2, q2, o2)
    squery = sample(raster, p3, m3, s3, q3, o3, xy)
    for a, b in zip(sbase, squery[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sbase[0].mean().backward()
    squery[0].mean().backward()
    for name, a, b in (("sample.xyz", m2, m3), ("sample.scale", s2, s3),
                       ("sample.quat", q2, q3), ("sample.opacity", o2, o3),
                       ("sample.query_xyz", p2, p3)):
        assert_grad_close(name, a, b)
    sid, sa, sc, so = squery[2:]
    check_members(sid, sa, sc, so, 128, means.shape[0])
    assert torch.equal(rc[:4], sc[:4])  # Integer pixels use the same tile.
    assert torch.equal(rid[:4], sid[:4])
    torch.testing.assert_close(ra[:4], sa[:4], rtol=0, atol=0)
    assert torch.all(sc[-3:] == 0) and torch.all(sid[-3:] == -1)
    sempty = sample(raster, points, means, scales, rotations, opacity, xy[:0])
    assert sempty[2].shape == (0, 128) and sempty[4].shape == (0,)
    # Independent CPU formula reads native sorted tile buffers, then computes
    # acceptance from projected 3D covariance rather than the new query kernel.
    sorted_ids, ranges = native_sample_lists(raster, points, means, opacity, scales, rotations)
    host_means = means.cpu().tolist()
    host_scales = scales.cpu().tolist()
    host_opacity = opacity[:, 0].cpu().tolist()
    for i in (4, 5, 6):
        coordinate = xy[i].cpu().tolist()
        expected_ids, expected_alpha = oracle_sample(
            coordinate, sorted_ids, ranges, host_means, host_scales, host_opacity)
        assert sc[i].item() == len(expected_ids), (i, sc[i].item(), len(expected_ids))
        assert sid[i, :len(expected_ids)].cpu().tolist() == expected_ids, i
        torch.testing.assert_close(sa[i, :len(expected_ids)].cpu(),
                                   torch.tensor(expected_alpha), rtol=2e-3, atol=5e-5)
    assert sc[5].item() == rc[5].item() + 1  # Native +0.5 sample tile admits ID 304.
    assert 304 in sid[5, :sc[5]].cpu().tolist() and 304 not in rid[5, :rc[5]].cpu().tolist()
    torch.cuda.synchronize()
    print("PASS: bounded CUDA smoke and synthetic CPU oracle; real-scene native parity remains untested")


if __name__ == "__main__":
    main()
