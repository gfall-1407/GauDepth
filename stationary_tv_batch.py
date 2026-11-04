"""Batched float64 stationary-TV reference/search with a first-order implicit VJP.

This is the same finite sign-search approximation as stationary_tv_depth.py,
not a proof that every critical point has been isolated. Branch/refinement
checks remain necessary. Invalid/rootless rays have a zero depth sentinel and
zero gradient, never a fallback depth. No model, camera, RGB or normal changes.
"""
from dataclasses import dataclass
import math
import torch
from torch.autograd.function import once_differentiable

from joint_tv_depth import validate_profiles, profile_knots, potential
from stationary_tv_depth import slope_at, slope_direction_signal


GUARD_NAMES = ('no_detected_root', 'near_potential', 'root_slope',
               'nonfinite_root_or_TV', 'monotonic_direction', 'root_residual')


class _BatchBranchDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, sigma, amplitude, points, weights, root, slope,
                valid, sample_chunk):
        ctx.save_for_backward(mean, sigma, amplitude, points, weights, slope, valid)
        ctx.sample_chunk = sample_chunk
        return torch.where(valid, root, torch.zeros_like(root))

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        mean, sigma, amplitude, points, weights, slope, valid = ctx.saved_tensors
        # Invalid rows may hold a depth=0 sentinel far outside their physical
        # domain. With a subnormal sigma, evaluating that sentinel first can
        # yield inf and then 0*inf=NaN. Mask BEFORE forming any standardized
        # coordinate; multiplying an already nonfinite derivative by zero is
        # insufficient. Valid rows and their endpoint derivatives are unchanged.
        active = valid[:, None]
        mean = torch.where(active, mean, torch.zeros_like(mean))
        sigma = torch.where(active, sigma, torch.ones_like(sigma))
        amplitude = torch.where(active, amplitude, torch.zeros_like(amplitude))
        points = torch.where(active, points, torch.zeros_like(points))
        weights = torch.where(active, weights, torch.zeros_like(weights))
        # Partial parameter derivatives at fixed prefix endpoints. Interior
        # endpoint motion cancels at H'=0; the moving root is the denominator.
        # Analytic accumulation avoids storing another full endpoint graph.
        gm, gs, ga = (torch.zeros_like(x) for x in (mean, sigma, amplitude))
        for start in range(0, points.shape[1], ctx.sample_chunk):
            x = points[:, start:start + ctx.sample_chunk]
            w = weights[:, start:start + ctx.sample_chunk]
            u = (x[..., None] - mean[:, None, :]) / sigma[:, None, :]
            exponential = torch.exp(-.5 * u.square())
            g = amplitude[:, None, :] * exponential
            coefficient = .5 * w[..., None] / (1 - g)
            gu = g * u
            gm += (coefficient * gu / sigma[:, None, :]).sum(1)
            # Two multiplies avoid u.square() overflow where g already vanished.
            gs += (coefficient * gu * u / sigma[:, None, :]).sum(1)
            ga += (coefficient * exponential).sum(1)
        safe_slope = torch.where(valid, slope, torch.ones_like(slope))
        factor = torch.where(valid, -upstream / safe_slope, torch.zeros_like(upstream))
        return (gm * factor[:, None], gs * factor[:, None], ga * factor[:, None],
                None, None, None, None, None, None)


@dataclass
class BatchStationaryResult:
    depth: torch.Tensor
    valid: torch.Tensor
    has_root: torch.Tensor
    total: torch.Tensor
    near_potential: torch.Tensor
    slope: torch.Tensor
    critical_points: torch.Tensor
    critical_values: torch.Tensor
    critical_count: torch.Tensor
    prefix_count: torch.Tensor
    grid_samples: torch.Tensor
    max_critical_residual: torch.Tensor
    root_residual: torch.Tensor
    max_critical_bracket_width: torch.Tensor
    root_bracket_width: torch.Tensor
    root_branch_margin: torch.Tensor
    monotonic_directions: torch.Tensor
    guard_failure_mask: torch.Tensor
    continuous_error_certified: bool = False

    def metadata(self):
        """Materialize JSON-safe per-ray records only when explicitly requested.

        Search/training need not call this: all branch data are already GPU
        tensors. Padded critical slots are omitted, as are padded directions.
        """
        fields = ('depth', 'valid', 'has_root', 'total', 'near_potential', 'slope',
                  'critical_count', 'prefix_count', 'grid_samples',
                  'max_critical_residual', 'root_residual',
                  'max_critical_bracket_width', 'root_bracket_width',
                  'root_branch_margin')
        scalars = torch.stack([getattr(self, k).detach().double() for k in fields], -1).cpu().tolist()
        cp = self.critical_points.detach().cpu().tolist()
        cv = self.critical_values.detach().cpu().tolist()
        directions = self.monotonic_directions.detach().cpu().tolist()
        failures = self.guard_failure_mask.detach().cpu().tolist()
        records = []
        for row, points, heights, signs, rejected in zip(scalars, cp, cv, directions, failures):
            d = {k: (v if math.isfinite(v) else None) for k, v in zip(fields, row)}
            for k in ('valid', 'has_root'): d[k] = bool(d[k])
            for k in ('critical_count', 'prefix_count', 'grid_samples'): d[k] = int(d[k])
            count = d['critical_count']
            d.update(critical_points=points[:count], critical_values=heights[:count],
                     monotonic_directions=signs[:count + 1],
                     guard_failures=[name for name, failed in zip(GUARD_NAMES, rejected) if failed],
                     continuous_error_certified=False)
            records.append(d)
        return records


def stationary_depth_batch(mean, sigma, amplitude, *, near, far, subdivisions=2,
                           min_slope=1e-4, max_near_potential=1e-4,
                           bisections=40, sample_chunk=128, member_mask=None):
    """Vectorized search and first-order implicit gradients for [R,K] profiles.

    All rays share near/far and numerical policies. member_mask is an optional
    boolean [R,K] tensor on the profile device; False slots contribute neither
    field nor internal search knots nor gradients. Without it, every slot is a
    real member, including a genuine zero-amplitude member and its search knots.
    In particular, amplitude==0 does NOT implicitly designate padding.
    Refinement comparisons are empirical, not completeness proofs.
    The caller should bound ray/member batch sizes as well as sample_chunk.
    """
    validate_profiles(mean, sigma, amplitude)
    if mean.shape[0] == 0 or not near < far or bisections < 24:
        raise ValueError('nonempty ray batch, ordered interval and >=24 bisections required')
    if min_slope <= 0 or max_near_potential < 0 or sample_chunk < 1:
        raise ValueError('invalid root guards or sample chunk')
    if len({x.device for x in (mean, sigma, amplitude)}) != 1:
        raise ValueError('profiles must share a device')
    if member_mask is not None:
        if not isinstance(member_mask, torch.Tensor):
            raise TypeError('member_mask must be a boolean tensor or None')
        if member_mask.dtype != torch.bool:
            raise TypeError('member_mask must have dtype torch.bool')
        if member_mask.shape != mean.shape or member_mask.device != mean.device:
            raise ValueError('member_mask must match profile shape and device')
        # Safe finite placeholders prevent a masked narrow/outlying member
        # from producing 0*inf before its zero amplitude is applied.
        p = (torch.where(member_mask, mean.double(), torch.zeros_like(mean, dtype=torch.float64)),
             torch.where(member_mask, sigma.double(), torch.ones_like(sigma, dtype=torch.float64)),
             amplitude.double() * member_mask)
    else:
        p = tuple(x.double() for x in (mean, sigma, amplitude))
    with torch.no_grad():
        t = profile_knots(*p, near, far, subdivisions, member_mask=member_mask)
        hp = slope_direction_signal(t, *p, chunk=sample_chunk)
        if not bool(torch.isfinite(hp).all()):
            raise ValueError('nonfinite derivative direction signal')
        rays, samples = t.shape
        grid_samples = 1 + (t[:, 1:] != t[:, :-1]).sum(-1)
        sample_index = torch.arange(samples, device=t.device).expand(rays, -1)
        nonzero = hp != 0
        # For each nonzero slope sample, find the preceding nonzero sample in
        # the same ray. Exact-zero samples and duplicate knots add no brackets.
        last_nonzero = torch.where(nonzero, sample_index, -1).cummax(-1).values
        previous = torch.cat((last_nonzero.new_full((rays, 1), -1), last_nonzero[:, :-1]), -1)
        previous_safe = previous.clamp_min(0)
        previous_signal = hp.gather(1, previous_safe)
        changes = nonzero & (previous >= 0) & (hp.signbit() != previous_signal.signbit())
        count = changes.sum(-1)
        max_count = int(count.max())
        lo = t.new_full((rays, max_count), far)
        hi = lo.clone()
        dl = lo.new_zeros(lo.shape)
        if max_count:
            ranks = changes.cumsum(-1) - 1
            row_index, col_index = changes.nonzero(as_tuple=True)
            packed_index = ranks[row_index, col_index]
            lo[row_index, packed_index] = t[row_index, previous_safe[row_index, col_index]]
            hi[row_index, packed_index] = t[row_index, col_index]
            dl[row_index, packed_index] = previous_signal[row_index, col_index]
            for _ in range(bisections):
                mid = (lo + hi) * .5
                dm = slope_direction_signal(mid, *p, chunk=sample_chunk)
                same = dm.signbit() == dl.signbit()
                zero = dm == 0
                lo = torch.where(same | zero, mid, lo)
                hi = torch.where((~same) | zero, mid, hi)
                dl = torch.where(same | zero, dm, dl)
        critical = (lo + hi) * .5
        points = torch.cat((t.new_full((rays, 1), near), critical,
                            t.new_full((rays, 1), far)), -1)
        h = potential(points, *p, sample_chunk=sample_chunk)
        segment_index = torch.arange(max_count + 1, device=t.device)[None, :]
        active_segment = segment_index <= count[:, None]
        directions = slope_direction_signal((points[:, :-1] + points[:, 1:]) * .5,
                                            *p, chunk=sample_chunk).sign()
        directions = torch.where(active_segment, directions, torch.zeros_like(directions))
        dh = h[:, 1:] - h[:, :-1]
        signed = dh * directions
        segment_ok = (signed >= -1e-10) & ((directions != 0) | (dh.abs() <= 1e-12))
        monotone_consistent = (segment_ok | ~active_segment).all(-1)
        variation = torch.cat((h.new_zeros((rays, 1)), dh.abs().cumsum(-1)), -1)
        target = math.log(2)
        total = variation[:, -1]
        has_root = total >= target
        j = (variation >= target).long().argmax(-1).clamp_min(1)
        take = lambda x, index: x.gather(1, index[:, None]).squeeze(1)
        left = take(points, j - 1)
        right = take(points, j)
        base = take(variation, j - 1)
        height = take(h, j - 1)
        segment_left, segment_right = left.clone(), right.clone()
        for _ in range(bisections):
            mid = (left + right) * .5
            accumulated = base + (potential(mid[:, None], *p, sample_chunk=sample_chunk)[:, 0] - height).abs()
            right = torch.where(accumulated >= target, mid, right)
            left = torch.where(accumulated < target, mid, left)
        root = (left + right) * .5
        slope = slope_at(root[:, None], *p, chunk=sample_chunk)[:, 0].abs()
        root = torch.where(has_root, root, torch.zeros_like(root))
        slope = torch.where(has_root, slope, torch.zeros_like(slope))
        prefix_count = torch.where(has_root, j - 1, torch.zeros_like(j))
        point_index = torch.arange(max_count + 2, device=t.device)[None, :]
        # Prefix endpoints are fixed for the partial derivative. All slots after
        # the selected root are duplicate roots with zero signed weights.
        prefix_points = torch.where(point_index < j[:, None], points, root[:, None])
        heights = potential(prefix_points, *p, sample_chunk=sample_chunk)
        residual = (torch.diff(heights, dim=-1).abs().sum(-1) - target).abs()
        residual = torch.where(has_root, residual, torch.full_like(residual, math.inf))
        finite = torch.isfinite(root) & torch.isfinite(total)
        guard_ok = torch.stack((has_root, h[:, 0] <= max_near_potential,
                                slope >= min_slope, finite, monotone_consistent,
                                residual <= 1e-8), -1)
        valid = guard_ok.all(-1)
        prefix_signs = directions * (segment_index < j[:, None]) * valid[:, None]
        weights = torch.zeros_like(points)
        weights[:, :-1] -= prefix_signs
        weights[:, 1:] += prefix_signs
        if max_count:
            critical_mask = torch.arange(max_count, device=t.device)[None, :] < count[:, None]
            critical_residual = torch.where(critical_mask,
                slope_at(critical, *p, chunk=sample_chunk).abs(), 0.).amax(-1)
            critical_width = torch.where(critical_mask, hi - lo, 0.).amax(-1)
        else:
            critical_residual = t.new_zeros(rays)
            critical_width = t.new_zeros(rays)
        root_width = torch.where(has_root, right - left, torch.full_like(root, math.inf))
        branch_margin = torch.where(has_root,
            torch.minimum(root - segment_left, segment_right - root), torch.zeros_like(root))
    depth = _BatchBranchDepth.apply(*p, prefix_points, weights, root, slope, valid, sample_chunk)
    return BatchStationaryResult(depth, valid, has_root, total, h[:, 0], slope,
        critical, h[:, 1:-1], count, prefix_count, grid_samples, critical_residual,
        residual, critical_width, root_width, branch_margin, directions, ~guard_ok)
