"""Differentiable *discretized* joint total-variation depth. Research prototype.

G_i(t)=a_i exp(-.5 ((t-m_i)/s_i)^2), h_i=-.5 log(1-G_i).
joint integrates |d sum(h_i)|; sep integrates sum |dh_i|.
The joint solver is the exact first crossing for piecewise-linear H on its
knots, NOT a certified integral of the continuous field. Refinement is required.
No normal, shortest-axis surface, Expected fallback, or RGB replacement.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class DepthResult:
    # valid is only a root/conditioning guard; it does NOT certify convergence.
    depth: torch.Tensor
    valid: torch.Tensor
    has_sampled_root: torch.Tensor
    slope: torch.Tensor
    total: torch.Tensor
    near_potential: torch.Tensor
    knots: int
    analytic_slope: torch.Tensor = None


def validate_profiles(mean, sigma, amplitude):
    if mean.ndim != 2 or mean.shape != sigma.shape or mean.shape != amplitude.shape:
        raise ValueError("profiles must have equal [ray, member] shapes")
    if mean.shape[1] == 0:
        raise ValueError("at least one member (zero amplitude padding allowed)")
    if not all(bool(torch.isfinite(x).all()) for x in (mean, sigma, amplitude)):
        raise ValueError("nonfinite profile")
    if bool((sigma <= 0).any()) or bool(((amplitude < 0) | (amplitude >= 1)).any()):
        raise ValueError("require sigma>0 and 0<=amplitude<1; no silent clamp")


def potential(t, mean, sigma, amplitude, sample_chunk=256):
    """H on [ray, sample] positions. Autograd still saves all chunks in total."""
    values = []
    for p in t.split(sample_chunk, dim=-1):
        u = (p[..., None] - mean[:, None, :]) / sigma[:, None, :]
        g = amplitude[:, None, :] * torch.exp(-0.5 * u.square())
        values.append((-0.5 * torch.log1p(-g)).sum(-1))
    return torch.cat(values, dim=-1)


def directional_slope(t, mean, sigma, amplitude):
    u = (t[:, None]-mean)/sigma
    g = amplitude*torch.exp(-0.5*u.square())
    return (-0.5*g*u/(sigma*(1-g))).sum(-1)


def profile_knots(mean, sigma, amplitude, near, far, subdivisions=1):
    """Nested adaptive grids. Knots remain differentiable through sort and clamp.

    Include each component's peak, near-peak scale sqrt(1-a)*s, and tails.
    Midpoint refinement is nested, so sampled total variation cannot decrease.
    Finite knots still do not certify isolation of all continuous extrema.
    """
    if subdivisions not in (1, 2, 4, 8, 16):
        raise ValueError("subdivisions must be a power of two from 1 to 16")
    offsets = mean.new_tensor([-8., -4., -2., -1., -.5, 0., .5, 1., 2., 4., 8.])
    outer = mean[..., None] + sigma[..., None] * offsets
    inner = mean[..., None] + (sigma * torch.sqrt(1-amplitude))[..., None] * offsets[2:-2]
    fixed = torch.linspace(near, far, 33, device=mean.device, dtype=mean.dtype)
    t = torch.cat((outer.flatten(1), inner.flatten(1), fixed.expand(mean.shape[0], -1)), -1)
    t = t.clamp(near, far).sort(-1).values
    if subdivisions > 1:
        f = torch.arange(subdivisions, device=t.device, dtype=t.dtype) / subdivisions
        refined = t[:, :-1, None] + (t[:, 1:] - t[:, :-1])[..., None] * f
        t = torch.cat((refined.flatten(1), t[:, -1:]), -1)
    return t


def first_tv_crossing(t, h, *, threshold=math.log(2), min_slope=1e-4,
                      max_near_potential=1e-4):
    """Exact differentiable crossing of sampled H's cumulative absolute changes.

    Flat/rootless/near-occupied rays have depth 0 and valid=False. No loss should
    use invalid depths. has_sampled_root distinguishes sampled crossing from a
    guard; no sampled crossing does NOT certify no continuous crossing.
    Roots at knots and changing sign branches are nonsmooth. No softened abs.
    """
    if threshold <= 0 or min_slope <= 0 or max_near_potential < 0:
        raise ValueError("invalid root policy")
    increments = (h[:, 1:] - h[:, :-1]).abs()
    cumulative = torch.cat((h.new_zeros((len(h), 1)), increments.cumsum(-1)), -1)
    total = cumulative[:, -1]
    has_root = total >= threshold
    # Leftmost equality; duplicated knots contain zero variation and are skipped.
    j = (cumulative >= threshold).to(torch.int64).argmax(-1).clamp(min=1)
    take = lambda a, k: a.gather(1, k[:, None]).squeeze(1)
    lo, hi = take(t, j-1), take(t, j)
    alo, ahi = take(cumulative, j-1), take(cumulative, j)
    dh, dt = ahi-alo, hi-lo
    safe_dh = torch.where(dh > 0, dh, torch.ones_like(dh))
    slope = dh / torch.where(dt > 0, dt, torch.ones_like(dt))
    root = lo + (threshold-alo) * dt / safe_dh
    valid = has_root & (dt > 0) & (slope >= min_slope)
    valid &= h[:, 0] <= max_near_potential
    valid &= torch.isfinite(root) & torch.isfinite(total)
    return DepthResult(torch.where(valid, root, torch.zeros_like(root)), valid,
                       has_root, slope, total, h[:, 0], t.shape[-1])


def joint_depth(mean, sigma, amplitude, *, near, far, subdivisions=1,
                min_slope=1e-4, max_near_potential=1e-4):
    validate_profiles(mean, sigma, amplitude)
    if not near < far:
        raise ValueError("invalid interval")
    t = profile_knots(mean, sigma, amplitude, near, far, subdivisions)
    h = potential(t, mean, sigma, amplitude)
    result = first_tv_crossing(t, h, min_slope=min_slope,
                              max_near_potential=max_near_potential)
    # A secant may conceal a root at a continuous turning point (e.g. a=.75).
    # This pointwise guard is necessary, not a certification over an interval.
    result.analytic_slope = directional_slope(result.depth, mean, sigma, amplitude).abs()
    result.valid = result.valid & (result.analytic_slope >= min_slope)
    result.depth = torch.where(result.valid, result.depth, torch.zeros_like(result.depth))
    return result


def sep_optical_depth(t, mean, sigma, amplitude, near):
    """Closed-form continuous separated TV, normalized at finite near."""
    h = lambda p: -0.5 * torch.log1p(-amplitude * torch.exp(-0.5*((p-mean)/sigma).square()))
    left = h(torch.full_like(mean, near))
    now = h(t[:, None])
    peak = -0.5 * torch.log1p(-amplitude)
    # Monotone if the component peak is outside [near,t].
    crosses_peak = (mean > near) & (mean < t[:, None])
    return torch.where(crosses_peak, 2*peak-left-now, (now-left).abs()).sum(-1)


def continuous_sep_depth(mean, sigma, amplitude, *, near, far, iterations=48):
    """Reference only, no gradient promise through bisection branch decisions."""
    validate_profiles(mean, sigma, amplitude)
    with torch.no_grad():
        lo = mean.new_full((len(mean),), near)
        hi = mean.new_full((len(mean),), far)
        valid = sep_optical_depth(hi, mean, sigma, amplitude, near) >= math.log(2)
        for _ in range(iterations):
            mid = (lo+hi)/2
            right = sep_optical_depth(mid, mean, sigma, amplitude, near) >= math.log(2)
            hi = torch.where(right, mid, hi)
            lo = torch.where(right, lo, mid)
        return torch.where(valid, (lo+hi)/2, torch.zeros_like(lo)), valid


def quaternion_rotation(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def ray_profiles(origin, direction, xyz, scales, rotations, opacity):
    """Full covariance, axial or metric rays (units follow direction).

    origin/direction [R,3], Gaussian inputs broadcastable [R,K,...] or [K,...].
    Orthogonal residual avoids C-B^2/A catastrophic cancellation. The returned
    amplitude is geometric 3D Gaussian value, not per-pixel splatting alpha.
    """
    rot = quaternion_rotation(rotations)
    delta = origin[:, None, :] - xyz
    local_o = torch.matmul(delta.unsqueeze(-2), rot).squeeze(-2) / scales
    local_d = torch.matmul(direction[:, None, None, :], rot).squeeze(-2) / scales
    a = local_d.square().sum(-1)
    b = (local_d * local_o).sum(-1)
    mean = -b/a
    residual = local_o + mean[..., None]*local_d
    amplitude = opacity.squeeze(-1) * torch.exp(-0.5*residual.square().sum(-1))
    return mean, a.rsqrt(), amplitude
