"""Synthetic native forward/backward checks; requires this branch's built extension."""

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, _C
from utils.contribution_band_reference import band_depth_and_grad


W = H = 49


def make_raster(band):
    proj = torch.zeros((4, 4), device="cuda")
    proj[0, 0] = proj[1, 1] = 2
    proj[2, 2] = 10 / 9.8
    proj[2, 3] = -2 / 9.8
    proj[3, 2] = 1
    return GaussianRasterizer(GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=.5, tanfovy=.5,
        kernel_size=0., bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=proj.T.contiguous(),
        sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False,
        require_depth=True, debug=False, band_depth=band,
    ))


def scene():
    means = torch.tensor([[0., 0., 1.], [0., 0., 1.5], [0., 0., 2.]],
                         device="cuda", requires_grad=True)
    opacity = torch.tensor([[.25], [.4], [.5]], device="cuda", requires_grad=True)
    scales = torch.tensor([[.15, .15, .01]] * 3, device="cuda")
    rotations = torch.tensor([[1., 0., 0., 0.]] * 3, device="cuda")
    colors = torch.tensor([[.4, .5, .6]] * 3, device="cuda")
    return means, opacity, scales, rotations, colors


def check_close(x, y, atol=2e-3):
    assert abs(x - y) < atol, (x, y)


def main():
    baseline, band = make_raster(False), make_raster(True)
    means, opacity, scales, rotations, colors = scene()
    inputs = dict(means3D=means, means2D=torch.zeros_like(means),
                  colors_precomp=colors, opacities=opacity, scales=scales,
                  rotations=rotations)
    base_out = baseline(**inputs)
    band_out = band(**inputs)
    for i in (0, 1, 4, 5):  # RGB, radii, alpha, normal
        torch.testing.assert_close(base_out[i], band_out[i])
    expected, depth_grad, alpha_grad = band_depth_and_grad(
        [.25, .4, .5], [1., 1.5, 2.])
    base_expected = (.25 * 1. + .3 * 1.5 + .225 * 2.) / .775
    check_close(base_out[2][0, 24, 24].item(), base_expected)
    observed = band_out[2][0, 24, 24]
    check_close(observed.item(), expected)
    observed.backward()
    for i in range(3):
        check_close(opacity.grad[i, 0].item(), alpha_grad[i], atol=1e-2)
        check_close(means.grad[i, 2].item(), depth_grad[i], atol=1e-2)

    origin = torch.tensor([[[0., 0., 1.]]], device="cuda")
    sampled_origin, inside_origin = band.sample_depth(origin, means, opacity, scales=scales, rotations=rotations)
    sampled_base, inside_base = baseline.sample_depth(origin, means, opacity, scales=scales, rotations=rotations)
    assert inside_base.item()
    check_close(sampled_base[0, 0, 2].item(), base_expected)
    assert inside_origin.item()
    check_close(sampled_origin[0, 0, 2].item(), expected)
    opacity.grad.zero_()
    sampled_origin[0, 0, 2].backward()
    for i in range(3):
        check_close(opacity.grad[i, 0].item(), alpha_grad[i], atol=1e-2)

    # Continuous query must use the same depth, with a query-position gradient.
    query = torch.tensor([[[.01, .005, 1.]]], device="cuda", requires_grad=True)
    sample, inside = band.sample_depth(query, means, opacity, scales=scales, rotations=rotations)
    assert inside.item()
    assert torch.isfinite(sample).all()
    sample[0, 0, 2].backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    eps = 1e-3
    for axis in (0, 1):
        plus, minus = query.detach().clone(), query.detach().clone()
        plus[0, 0, axis] += eps
        minus[0, 0, axis] -= eps
        with torch.no_grad():
            fp = band.sample_depth(plus, means, opacity, scales=scales, rotations=rotations)[0][0, 0, 2]
            fm = band.sample_depth(minus, means, opacity, scales=scales, rotations=rotations)[0][0, 0, 2]
        check_close(query.grad[0, 0, axis].item(), ((fp - fm) / (2 * eps)).item(), atol=3e-2)

    # Exact normalized prefixes at .25 and .75 use the right-side owner.
    edge_means, edge_opacity, edge_scales, edge_rotations, edge_colors = scene()
    with torch.no_grad():
        edge_opacity[:, 0] = torch.tensor([.2, .5, .5], device="cuda")
    edge = band(means3D=edge_means, means2D=torch.zeros_like(edge_means),
                colors_precomp=edge_colors, opacities=edge_opacity,
                scales=edge_scales, rotations=edge_rotations)[2][0, 24, 24]
    edge.backward()
    _, _, edge_grad = band_depth_and_grad([.2, .5, .5], [1., 1.5, 2.])
    for i in range(3):
        check_close(edge_opacity.grad[i, 0].item(), edge_grad[i], atol=1e-2)

    near_means, near_opacity, near_scales, near_rotations, near_colors = scene()
    near_alpha = [.2000004, .5, .4999997499998752]
    with torch.no_grad():
        near_opacity[:, 0] = torch.tensor(near_alpha, device="cuda")
    near_depth = band(means3D=near_means, means2D=torch.zeros_like(near_means),
                      colors_precomp=near_colors, opacities=near_opacity,
                      scales=near_scales, rotations=near_rotations)[2][0, 24, 24]
    near_depth.backward()
    near_expected, _, near_grad = band_depth_and_grad(near_alpha, [1., 1.5, 2.])
    check_close(near_depth.item(), near_expected)
    for i in range(3):
        check_close(near_opacity.grad[i, 0].item(), near_grad[i], atol=1e-2)

    single_means = means.detach()[:1].clone().requires_grad_()
    single_opacity = opacity.detach()[:1].clone().requires_grad_()
    single = band(means3D=single_means, means2D=torch.zeros_like(single_means),
                  colors_precomp=colors[:1], opacities=single_opacity,
                  scales=scales[:1], rotations=rotations[:1])[2][0, 24, 24]
    check_close(single.item(), 1.)
    single.backward()
    check_close(single_opacity.grad.item(), 0., atol=1e-4)
    check_close(single_means.grad[0, 2].item(), 1., atol=1e-3)

    weak_opacity = torch.full_like(single_opacity, .001, requires_grad=True)
    weak_input = dict(means3D=single_means, means2D=torch.zeros_like(single_means),
                      colors_precomp=colors[:1], opacities=weak_opacity,
                      scales=scales[:1], rotations=rotations[:1])
    weak_base = baseline(**weak_input)[2][0, 24, 24]
    weak_band = band(**weak_input)[2][0, 24, 24]
    check_close(weak_band.item(), weak_base.item(), atol=1e-6)

    # Native sample binning uses int((x + .5) / BLOCK_X), not int(x) / BLOCK_X.
    def at_pixel(x, z):
        return [((x + .5) * 2 / W - 1) * z * .5, 0., z]

    tile_means = torch.tensor([at_pixel(8, 1.), at_pixel(29.05, 2.)],
                              device="cuda")
    tile_scales = torch.tensor([[4 / 49, 4 / 49, .01],
                                [8.4 / 49, 8.4 / 49, .01]], device="cuda")
    tile_rot = rotations[:2]
    tile_opacity = torch.tensor([[.05], [.99]], device="cuda", requires_grad=True)
    tile_points = torch.tensor([[at_pixel(x, 1.) for x in
                                 (15.49, 15.5, 15.75, 31.5)]], device="cuda")
    s = band.raster_settings
    raw = _C.sample_rasterized_depth(
        tile_points, tile_means, tile_opacity, tile_scales, tile_rot,
        1., torch.empty(0), s.viewmatrix, s.projmatrix, .5, .5, 0.,
        H, W, s.campos, False, True, False)
    rendered = raw[0]
    sorted_ids = raw[5][:rendered * 4].view(torch.int32).cpu().tolist()
    tile_count = ((W + 15) // 16) * ((H + 15) // 16)
    ranges = raw[8][:tile_count * 8].view(torch.int32).reshape(-1, 2).cpu().tolist()
    members = [sorted_ids[ranges[i][0]:ranges[i][1]] for i in (4, 5, 6)]
    assert members == [[0], [0, 1], [1]], members
    tile_depth, tile_inside = band.sample_depth(
        tile_points, tile_means, tile_opacity, scales=tile_scales, rotations=tile_rot)
    assert tile_inside.all()
    tile_z = tile_depth[0, :, 2].detach().cpu().tolist()
    assert .9 < tile_z[0] < 1.1 and 1.1 < tile_z[1] < 1.5
    assert 1.2 < tile_z[2] < 1.7 and 1.9 < tile_z[3] < 2.1, tile_z

    eps = 1e-4
    for index in (1, 2, 3):
        q = tile_points[:, index:index + 1].detach().clone().requires_grad_()
        tile_opacity.grad = None
        z = band.sample_depth(q, tile_means, tile_opacity,
                              scales=tile_scales, rotations=tile_rot)[0][0, 0, 2]
        z.backward()
        plus = q.detach().clone()
        plus[0, 0, 0] += eps
        with torch.no_grad():
            zp = band.sample_depth(plus, tile_means, tile_opacity,
                                   scales=tile_scales, rotations=tile_rot)[0][0, 0, 2]
        # Boundary points 15.5 and 31.5 use the right-side tile derivative.
        check_close(q.grad[0, 0, 0].item(), ((zp - z.detach()) / eps).item(), atol=.1)
        if index == 2:
            plus_opacity = tile_opacity.detach().clone()
            minus_opacity = tile_opacity.detach().clone()
            plus_opacity[1, 0] += eps
            minus_opacity[1, 0] -= eps
            with torch.no_grad():
                fp = band.sample_depth(q.detach(), tile_means, plus_opacity,
                                       scales=tile_scales, rotations=tile_rot)[0][0, 0, 2]
                fm = band.sample_depth(q.detach(), tile_means, minus_opacity,
                                       scales=tile_scales, rotations=tile_rot)[0][0, 0, 2]
            check_close(tile_opacity.grad[1, 0].item(), ((fp - fm) / (2 * eps)).item(), atol=.02)
            assert abs(tile_opacity.grad[1, 0].item()) > .01
    print("contribution-band native smoke passed")


if __name__ == "__main__":
    main()
