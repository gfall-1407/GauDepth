"""Scalar reference for the accepted-order contribution-band depth.

This intentionally models a fixed accepted set. Native culling, alpha clipping,
and early termination are discrete and must be matched by the CUDA caller.
"""

from math import isfinite
from struct import pack, unpack


def _next_up(value):
    """One binary64 ULP above a positive boundary (also on Python 3.8)."""
    bits = unpack(">Q", pack(">d", value))[0]
    return unpack(">d", pack(">Q", bits + 1))[0]


def band_depth_and_grad(alpha, depth, lower=0.25, upper=0.75):
    """Return depth, dD/d(depth_i), dD/d(alpha_i) for accepted members.

    The interval is in normalized front-to-back accumulated contribution,
    not in sorted camera depth. The returned depth has the units of ``depth``.
    """
    if len(alpha) != len(depth) or not alpha:
        raise ValueError("equal-length nonempty alpha and depth are required")
    if not 0 <= lower < upper <= 1:
        raise ValueError("invalid contribution interval")
    if any(not isfinite(a) or not 0 <= a < 1 for a in alpha):
        raise ValueError("alpha must be finite in [0, 1)")
    if any(not isfinite(d) for d in depth):
        raise ValueError("depth must be finite")

    weights = []
    transmittance = []
    t = 1.0
    for a in alpha:
        transmittance.append(t)
        weights.append(t * a)
        t *= 1 - a
    total = sum(weights)
    if total <= 0:
        raise ValueError("no accepted contribution")

    scale = 1.0 / (upper - lower)
    prefix = 0.0
    overlaps = []
    normalized_prefix = []
    for w in weights:
        before = prefix / total
        prefix += w
        after = prefix / total
        normalized_prefix.append(after)
        overlaps.append(max(0.0, min(after, upper) - max(before, lower)))
    result = scale * sum(o * d for o, d in zip(overlaps, depth))
    depth_grad = [scale * o for o in overlaps]

    # First prefix strictly exceeding a boundary owns that boundary. At an
    # exact prefix equality this specifies the right-side subgradient.
    low = next(i for i, p in enumerate(normalized_prefix) if p > _next_up(lower))
    high = next(i for i, p in enumerate(normalized_prefix) if p > _next_up(upper))
    clipped = [depth[low] if i <= low else depth[high] if i >= high else depth[i]
               for i in range(len(alpha))]
    clipped_mean = lower * depth[low] + (upper - lower) * result + (1 - upper) * depth[high]
    weight_grad = [scale * (d - clipped_mean) / total for d in clipped]

    # dw_i/dalpha_k = T_k for i=k, -w_i/(1-alpha_k) for i>k.
    alpha_grad = [0.0] * len(alpha)
    weighted_tail = 0.0
    for k in range(len(alpha) - 1, -1, -1):
        alpha_grad[k] = (
            transmittance[k] * weight_grad[k]
            - weighted_tail / (1 - alpha[k])
        )
        weighted_tail += weights[k] * weight_grad[k]
    return result, depth_grad, alpha_grad
