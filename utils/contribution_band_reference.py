"""Scalar reference for the accepted-order contribution-band depth.

This intentionally models a fixed accepted set. Native culling, alpha clipping,
and early termination are discrete and must be matched by the CUDA caller.
"""

from math import isfinite


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

    # Write D as scale * sum_i H(P_i) * (d_i - d_{i+1}), where
    # H(p)=clip(p-lower, 0, upper-lower). P_last=1 is constant.
    # Thus g_k=dD/dw_k=(scale/A)*(sum_{i>=k} v_i - B),
    # v_i=1_(lower<P_i<upper)*(d_i-d_{i+1}), B=sum_i v_i*P_i.
    v = [
        (depth[i] - depth[i + 1]) if lower < normalized_prefix[i] < upper else 0.0
        for i in range(len(alpha) - 1)
    ]
    b = sum(x * p for x, p in zip(v, normalized_prefix))
    weight_grad = [0.0] * len(alpha)
    suffix = 0.0
    for k in range(len(alpha) - 1, -1, -1):
        if k < len(v):
            suffix += v[k]
        weight_grad[k] = scale * (suffix - b) / total

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
