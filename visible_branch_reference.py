"""CPU arithmetic reference for a photo-selected Gaussian plane branch.

This is deliberately separate from the rasterizer. Native accepted records and
their differentiable alpha/T values must be supplied by a later host adapter.
"""

from dataclasses import dataclass
from math import exp, isfinite, sqrt


TAU = 0.1
EPS = 1e-10


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def sub(x, y):
    return tuple(a - b for a, b in zip(x, y))


@dataclass(frozen=True)
class Member:
    gaussian_id: int
    mu: tuple
    normal: tuple
    alpha: float
    transmittance: float
    footprint: float


@dataclass(frozen=True)
class Camera:
    center: tuple
    camera_to_world: tuple  # three row vectors
    fx: float
    fy: float
    cx: float
    cy: float


def pixel_ray(camera, x, y):
    local = ((x-camera.cx)/camera.fx, (y-camera.cy)/camera.fy, 1.)
    return tuple(dot(row, local) for row in camera.camera_to_world)


def project(camera, point):
    delta = sub(point, camera.center)
    local = tuple(sum(camera.camera_to_world[r][c]*delta[r]
                      for r in range(3)) for c in range(3))
    if local[2] <= EPS:
        return None
    return (camera.fx*local[0]/local[2]+camera.cx,
            camera.fy*local[1]/local[2]+camera.cy)


def bilinear(image, x, y):
    width, height = len(image[0]), len(image)
    if not (0 <= x < width-1 and 0 <= y < height-1):
        return None
    ix, iy = int(x), int(y)
    dx, dy = x-ix, y-iy
    return sum(image[iy+oy][ix+ox] *
               (dx if ox else 1-dx) * (dy if oy else 1-dy)
               for oy in (0, 1) for ox in (0, 1))


def patch_cost(member, reference_camera, reference_image,
               witness_camera, witness_image, x, y, patch_radius=2):
    """Warp a 5x5 reference patch through the member's plane.

    Images are linear grayscale floats. The result is invalid at image bounds,
    occlusions, textureless patches, or near-parallel patch rays; occlusion
    requires an external visibility check and cannot be inferred here.
    """
    ref, witness = [], []
    for oy in range(-patch_radius, patch_radius+1):
        for ox in range(-patch_radius, patch_radius+1):
            px, py = x+ox, y+oy
            color = bilinear(reference_image, px, py)
            ray = pixel_ray(reference_camera, px, py)
            hit = intersection(member, reference_camera.center, ray)
            if color is None or hit is None:
                return None
            point = tuple(o+hit[2]*v for o, v in zip(reference_camera.center, ray))
            coords = project(witness_camera, point)
            if coords is None:
                return None
            sampled = bilinear(witness_image, *coords)
            if sampled is None:
                return None
            ref.append(color); witness.append(sampled)
    return ncc_cost(ref, witness)


def intersection(member, origin, ray):
    n = member.normal
    norm = sqrt(dot(n, n))
    if norm <= 1e-8 or not isfinite(norm):
        return None
    a = dot(n, ray)
    b = dot(n, sub(member.mu, origin))
    if abs(a) <= 1e-8 * norm * sqrt(dot(ray, ray)):
        return None
    t = b / a
    if not isfinite(t) or t <= 0:
        return None
    return (a, b, t)


def same_branch(i, j, depths, projections, half_pixel=0.5):
    """Require the two hypotheses to agree in both witness projections."""
    if depths[i] is None or depths[j] is None:
        return False
    for view in projections:
        pi, pj = view[i], view[j]
        if pi is None or pj is None:
            return False
        if sqrt(sum((x-y)**2 for x, y in zip(pi, pj))) > half_pixel:
            return False
    return True


def choose_branch(depths, projections, cost_sets, perturb_cost_sets=None):
    """Choose one branch from two views; ties or unstable winners are invalid.

    perturb_cost_sets contains fixed ±0.25-pixel reference patch shifts. Each
    item is a pair of complete per-member cost arrays, one for each witness.
    """
    if len(cost_sets) != 2 or len(projections) != 2:
        raise ValueError("exactly two witnesses required")
    count = len(depths)
    if any(len(v) != count for v in (*cost_sets, *projections)):
        raise ValueError("member array length mismatch")
    eligible = [i for i in range(count) if depths[i] is not None and all(
        cost_sets[v][i] is not None and isfinite(cost_sets[v][i]) and
        projections[v][i] is not None
        for v in range(2))]
    if not eligible:
        return (), "no_eligible_hypothesis"

    def winner(costs, view):
        ranked = sorted(eligible, key=lambda i: (costs[view][i], i))
        if len(ranked) > 1 and abs(costs[view][ranked[0]] - costs[view][ranked[1]]) <= EPS:
            if not same_branch(ranked[0], ranked[1], depths, projections):
                return None
        return ranked[0]

    leaders = [winner(cost_sets, v) for v in range(2)]
    if any(i is None for i in leaders) or not same_branch(*leaders, depths, projections):
        return (), "witness_disagreement"
    anchor = leaders[0]
    branch = tuple(i for i in eligible if same_branch(i, anchor, depths, projections))
    if not branch:
        return (), "empty_branch"
    # Competing branches must remain defeated under all fixed small shifts.
    for shifted in perturb_cost_sets or ():
        if len(shifted) != 2:
            raise ValueError("perturbation requires both witnesses")
        if any(len(v) != count for v in shifted):
            raise ValueError("perturbation member count mismatch")
        for v in range(2):
            shifted_winner = winner(shifted, v)
            if shifted_winner not in branch:
                return (), "unstable_branch"
    return branch, "valid"


def candidate_depth(members, origin, ray, costs, branch, fallback):
    """Piecewise differentiable weighted plane LS arithmetic (plain CPU values)."""
    numerator = denominator = 0.0
    for i in branch:
        m = members[i]
        hit = intersection(m, origin, ray)
        if hit is None:
            return fallback, False, "invalid_member"
        a, b, _ = hit
        if not (0 < m.alpha < 1 and 0 <= m.transmittance <= 1 and
                0 <= m.footprint <= 1):
            return fallback, False, "invalid_weight"
        c1, c2 = costs[0][i], costs[1][i]
        weight = (m.transmittance * m.alpha * m.footprint *
                  exp(-(c1 + c2) / (2 * TAU)))
        numerator += weight * a * b
        denominator += weight * a * a
    if not branch or denominator <= EPS:
        return fallback, False, "zero_responsibility"
    depth = numerator / denominator
    if not isfinite(depth):
        return fallback, False, "nonfinite_depth"
    return depth, True, "valid"


def ncc_cost(reference, witness):
    """Fixed 5x5 grayscale or channel-flattened patch; lower is better."""
    if len(reference) != len(witness) or not reference:
        raise ValueError("patch sizes differ")
    xr = sum(reference) / len(reference)
    yw = sum(witness) / len(witness)
    a = [x - xr for x in reference]
    b = [y - yw for y in witness]
    xx, yy = dot(a, a), dot(b, b)
    if xx <= EPS or yy <= EPS:
        return None  # no texture: do not manufacture a winning branch
    ncc = max(-1.0, min(1.0, dot(a, b) / sqrt(xx * yy)))
    return (1.0 - ncc) / 2.0


def plane_loss(expected, candidate, valid, target, weight=1.0):
    """Host loss denominator is preserved for invalid candidate rays."""
    depths = [c if ok else e for e, c, ok in zip(expected, candidate, valid)]
    if len(depths) != len(target) or not depths:
        raise ValueError("loss input mismatch")
    return weight * sum(abs(d-y) for d, y in zip(depths, target)) / len(depths)
