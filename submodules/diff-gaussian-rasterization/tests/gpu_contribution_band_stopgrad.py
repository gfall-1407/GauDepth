"""Small native check of the contribution-band surrogate gradient."""

import torch

from gpu_contribution_band_smoke import W, H, make_raster, scene, check_close
from utils.contribution_band_reference import band_depth_stopgrad_weights


def main():
    assert torch.cuda.is_available()
    full, sg = make_raster(1), make_raster(2)
    baseline = make_raster(0)
    means, opacity, scales, rotations, colors = scene()
    scales = scales.clone().requires_grad_()
    colors = colors.clone().requires_grad_()
    inputs = dict(means3D=means, means2D=torch.zeros_like(means),
                  colors_precomp=colors, opacities=opacity, scales=scales,
                  rotations=rotations)
    full_out, sg_out, base_out = full(**inputs), sg(**inputs), baseline(**inputs)
    for a, b in zip(full_out, sg_out):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for i in (0, 1, 4, 5):
        torch.testing.assert_close(sg_out[i], base_out[i], rtol=0, atol=0)

    depth = sg_out[2][0, H // 2, W // 2]
    _, expected_di, _ = band_depth_stopgrad_weights([.25, .4, .5], [1., 1.5, 2.])
    grad_mean, grad_opacity, grad_scale = torch.autograd.grad(
        depth, (means, opacity, scales), retain_graph=True)
    torch.testing.assert_close(grad_opacity, torch.zeros_like(grad_opacity), atol=1e-5, rtol=0)
    torch.testing.assert_close(grad_scale, torch.zeros_like(grad_scale), atol=1e-4, rtol=0)
    for i in range(3):
        check_close(grad_mean[i, 2].item(), expected_di[i], atol=1e-2)
    full_opacity = torch.autograd.grad(full_out[2][0, H // 2, W // 2], opacity,
                                       retain_graph=True)[0]
    assert full_opacity.abs().max().item() > 1e-2

    rgb_full = torch.autograd.grad(full_out[0].sum(), (means, opacity, scales, colors),
                                   retain_graph=True)
    rgb_sg = torch.autograd.grad(sg_out[0].sum(), (means, opacity, scales, colors),
                                 retain_graph=True)
    for a, b in zip(rgb_full, rgb_sg):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    assert rgb_sg[1].abs().max().item() > 0

    query = torch.tensor([[[.01, .005, 1.]]], device="cuda", requires_grad=True)
    sampled_full, inside_full = full.sample_depth(query, means, opacity, scales=scales,
                                                   rotations=rotations)
    sampled_sg, inside_sg = sg.sample_depth(query, means, opacity, scales=scales,
                                            rotations=rotations)
    torch.testing.assert_close(sampled_full, sampled_sg, rtol=0, atol=0)
    assert inside_full.item() and inside_sg.item()
    sample_grads = torch.autograd.grad(sampled_sg[0, 0, 2], (means, opacity, scales, query),
                                       retain_graph=True)
    full_sample_grads = torch.autograd.grad(
        sampled_full[0, 0, 2], (means, opacity, scales, query), retain_graph=True)
    torch.testing.assert_close(sample_grads[1], torch.zeros_like(opacity), atol=1e-5, rtol=0)
    torch.testing.assert_close(sample_grads[2], torch.zeros_like(scales), atol=1e-4, rtol=0)
    assert full_sample_grads[1].abs().max().item() > 1e-3
    assert full_sample_grads[2].abs().max().item() > 1e-3
    assert sample_grads[0][:, :2].abs().max().item() > 1e-3  # di geometry remains
    assert (full_sample_grads[0][:, :2] - sample_grads[0][:, :2]).abs().max().item() > 1e-3
    assert (full_sample_grads[3] - sample_grads[3]).abs().max().item() > 1e-3
    assert torch.isfinite(sample_grads[3]).all()
    query_x_grad = torch.autograd.grad(sampled_sg[0, 0, 0], query)[0][0, 0, 0]
    check_close(query_x_grad.item(), sampled_sg[0, 0, 2].item(), atol=2e-2)

    for alpha in ([.2, .5, .5], [.2000004, .5, .4999997499998752]):
        edge_opacity = torch.tensor(alpha, device="cuda")[:, None].requires_grad_()
        edge_inputs = dict(inputs, opacities=edge_opacity)
        edge_full = full(**edge_inputs)[2][0, H // 2, W // 2]
        edge_sg = sg(**edge_inputs)[2][0, H // 2, W // 2]
        torch.testing.assert_close(edge_full, edge_sg, rtol=0, atol=0)
        edge_grad = torch.autograd.grad(edge_sg, edge_opacity)[0]
        torch.testing.assert_close(edge_grad, torch.zeros_like(edge_grad), rtol=0, atol=1e-5)

    # Empty acceptance and the weak-alpha fallback retain the baseline forward.
    weak_opacity = torch.full_like(opacity[:1], .001)
    weak = dict(means3D=means[:1], means2D=torch.zeros_like(means[:1]),
                colors_precomp=colors[:1], opacities=weak_opacity,
                scales=scales[:1], rotations=rotations[:1])
    torch.testing.assert_close(sg(**weak)[2], baseline(**weak)[2], rtol=0, atol=0)
    empty = dict(weak, opacities=torch.zeros_like(weak_opacity))
    torch.testing.assert_close(sg(**empty)[2], baseline(**empty)[2], rtol=0, atol=0)
    print("contribution-band stopgrad native smoke passed")


if __name__ == "__main__":
    main()
