"""Inference-only first crossing of the EXISTING native cumulative opacity.

This is a GOF-like control, not the joint-total-variation proposal or a new
view-independent field. No derivative or Expected fallback is invented here.
"""
import math
import numpy as np


def reference_opacity(z, means, sigma, alpha):
    """CPU formula reference without native tile/alpha/termination culling."""
    z = np.asarray(z, dtype=np.float64)
    means, sigma, alpha = [np.asarray(x, dtype=np.float64) for x in (means, sigma, alpha)]
    if np.any(sigma <= 0) or np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError("invalid profile")
    u = np.minimum((z[..., None] - means) / sigma, 0.)
    local = alpha * np.exp(-.5*u*u)
    return -np.expm1(np.log1p(-local).sum(axis=-1))


def first_crossing_numpy(query, count, near=.2001, far=8., steps=22):
    """Reference bracketing policy; invalid/no crossing has zero depth."""
    lo, hi = np.full(count, near), np.full(count, far)
    a, b = query(lo), query(hi)
    valid = np.isfinite(a) & np.isfinite(b) & (a < .5) & (b >= .5)
    for _ in range(steps):
        mid = (lo+hi)/2
        value = query(mid)
        valid &= np.isfinite(value)
        hi = np.where(value >= .5, mid, hi)
        lo = np.where(value < .5, mid, lo)
    return np.where(valid, (lo+hi)/2, 0.), valid


def render_collective_half(camera, gaussians, pipe, background, kernel_size,
                           batch_pixels=131072, steps=22, near=.2001, far=8.):
    """All image pixels, bounded query batches, native integrate CUDA unchanged.

    Native points are culled at camera z <= .2; explicitly reject a crossing
    already reached at near. Invalid roots stay zero and never use a baseline.
    Values are axial camera depth, matching mesh_extract.py.
    """
    import torch
    from gaussian_renderer import integrate
    if torch.is_grad_enabled():
        raise RuntimeError("inference-only: run under torch.no_grad(); no training gradient exists")
    if not (steps >= 1 and 0.2 < near < far <= 8 and batch_pixels > 0):
        raise ValueError("invalid domain or query budget")
    w, h = camera.image_width, camera.image_height
    device = gaussians.get_xyz.device
    c2w = torch.linalg.inv(camera.world_view_transform.T)
    depth = torch.zeros(h*w, dtype=torch.float32, device=device)
    statuses = torch.zeros(h*w, dtype=torch.uint8, device=device)
    # status: 1 valid, 2 native outside/nonfinite, 3 near clipped, 4 no far crossing
    max_projection_error = 0.
    for start in range(0, h*w, batch_pixels):
        end = min(start+batch_pixels, h*w)
        pixel = torch.arange(start, end, device=device)
        x, y = (pixel % w).float(), (pixel // w).float()
        rays = torch.stack(((x-camera.Cx)/camera.Fx,
                            (y-camera.Cy)/camera.Fy, torch.ones_like(x)), -1)
        world_rays = rays @ c2w[:3, :3].T
        origin = c2w[:3, 3]
        def query(z):
            points = origin + world_rays*z[:, None]
            result = integrate(points, camera, gaussians, pipe, background, kernel_size)
            value = result["alpha_integrated"].reshape(-1)
            inside = result["inside"].reshape(-1).bool()
            return value, inside & torch.isfinite(value)
        # Test the same pixel convention as native preprocessPointsCUDA.
        p = origin + world_rays
        clip = torch.cat((p, torch.ones_like(p[:, :1])), 1) @ camera.full_proj_transform
        projected = ((clip[:, :2]/clip[:, 3:4]+1)*torch.tensor([w, h], device=device)-1)/2
        error = float(torch.max(torch.abs(projected-torch.stack((x,y),-1))))
        max_projection_error = max(max_projection_error, error)
        if error > .002:
            raise RuntimeError("camera ray/native pixel projection mismatch")
        lo, hi = torch.full_like(x, near), torch.full_like(x, far)
        alo, il = query(lo)
        ahi, ih = query(hi)
        valid_native = il & ih
        near_clipped = valid_native & (alo >= .5)
        no_crossing = valid_native & ~near_clipped & (ahi < .5)
        valid = valid_native & ~near_clipped & ~no_crossing
        status = torch.full_like(pixel, 2, dtype=torch.uint8)
        status[near_clipped] = 3
        status[no_crossing] = 4
        for _ in range(steps):
            mid = (lo+hi)/2
            value, inside = query(mid)
            valid &= inside
            hi = torch.where(value >= .5, mid, hi)
            lo = torch.where(value < .5, mid, lo)
        status[valid] = 1
        depth[start:end] = torch.where(valid, (lo+hi)/2, 0.)
        statuses[start:end] = status
    counts = {str(i): int((statuses == i).sum()) for i in (1,2,3,4)}
    return depth.reshape(1,h,w), statuses.reshape(h,w), dict(
        counts=counts, pixels=h*w, steps=steps, near=near, far=far,
        axial_bracket_width=(far-near)/(2**steps),
        max_projection_error_pixels=max_projection_error,
        field="existing native cumulative opacity; GOF-like inference control",
        no_expected_fallback=True, training_gradient=False)
