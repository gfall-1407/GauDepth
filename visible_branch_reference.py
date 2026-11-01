"""CPU arithmetic reference for a photo-selected Gaussian plane branch.

This is deliberately separate from the rasterizer. Native accepted records and
their differentiable alpha/T values must be supplied by a later host adapter.
"""

from dataclasses import dataclass
from math import exp, isfinite, sqrt


TAU = 0.1
EPS = 1e-10
PERTURB_OFFSETS = {(-.25, 0.), (.25, 0.), (0., -.25), (0., .25)}


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
    if not all(isfinite(x) for x in local) or local[2] <= EPS:
        return None
    pixel = (camera.fx*local[0]/local[2]+camera.cx,
             camera.fy*local[1]/local[2]+camera.cy)
    return pixel if all(isfinite(x) for x in pixel) else None


def bilinear(image, x, y):
    width, height = len(image[0]), len(image)
    if not isfinite(x) or not isfinite(y):
        raise ValueError("nonfinite sample coordinates")
    if not (0 <= x < width-1 and 0 <= y < height-1):
        return None
    ix, iy = int(x), int(y)
    dx, dy = x-ix, y-iy
    corners = [image[iy+oy][ix+ox] for oy in (0, 1) for ox in (0, 1)]
    if not all(isfinite(v) for v in corners):
        raise ValueError("nonfinite image sample")
    value = sum(corners[2*oy+ox] *
                (dx if ox else 1-dx) * (dy if oy else 1-dy)
                for oy in (0, 1) for ox in (0, 1))
    if not isfinite(value):
        raise ValueError("nonfinite bilinear interpolation")
    return value


def patch_cost(member, reference_camera, reference_image,
               witness_camera, witness_image, x, y, patch_radius=2):
    """Warp a 5x5 reference patch through the member's plane.

    Images are linear grayscale floats. The result is invalid at image bounds,
    occlusions, textureless patches, or near-parallel patch rays; occlusion
    requires an external visibility check and cannot be inferred here.
    """
    return patch_cost_diagnostic(member, reference_camera, reference_image,
                                 witness_camera, witness_image, x, y,
                                 patch_radius)[0]


def patch_cost_diagnostic(member, reference_camera, reference_image,
                          witness_camera, witness_image, x, y, patch_radius=2):
    """Return (NCC cost, failure reason) without dropping invalid evidence."""
    for camera in (reference_camera, witness_camera):
        values = (*camera.center, *camera.camera_to_world[0],
                  *camera.camera_to_world[1], *camera.camera_to_world[2],
                  camera.fx, camera.fy, camera.cx, camera.cy)
        if not all(isfinite(v) for v in values) or camera.fx <= 0 or camera.fy <= 0:
            raise ValueError("corrupt camera")
    if not all(isfinite(v) for v in (*member.mu, *member.normal, x, y)):
        raise ValueError("nonfinite plane or reference pixel")
    ref, witness = [], []
    for oy in range(-patch_radius, patch_radius+1):
        for ox in range(-patch_radius, patch_radius+1):
            px, py = x+ox, y+oy
            color = bilinear(reference_image, px, py)
            if color is None:
                return None, "reference_patch_oob"
            ray = pixel_ray(reference_camera, px, py)
            hit = intersection(member, reference_camera.center, ray)
            if hit is None:
                return None, "invalid_plane_intersection"
            point = tuple(o+hit[2]*v for o, v in zip(reference_camera.center, ray))
            if not all(isfinite(v) for v in point):
                raise ValueError("nonfinite plane intersection")
            delta = sub(point, witness_camera.center)
            local = tuple(sum(witness_camera.camera_to_world[r][c]*delta[r]
                              for r in range(3)) for c in range(3))
            if not all(isfinite(v) for v in local):
                raise ValueError("nonfinite witness camera point")
            if local[2] <= EPS:
                return None, "witness_negative_or_near_zero_depth"
            coords = (witness_camera.fx*local[0]/local[2]+witness_camera.cx,
                      witness_camera.fy*local[1]/local[2]+witness_camera.cy)
            if not all(isfinite(v) for v in coords):
                raise ValueError("nonfinite witness projection")
            sampled = bilinear(witness_image, *coords)
            if sampled is None:
                return None, "witness_patch_oob"
            ref.append(color); witness.append(sampled)
    cost = ncc_cost(ref, witness)
    return (cost, "valid" if cost is not None else "low_texture")


def intersection(member, origin, ray):
    n = member.normal
    if not all(isfinite(x) for x in (*n, *member.mu, *origin, *ray)):
        return None
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
    if (depths[i] is None or depths[j] is None or
            not isfinite(depths[i]) or not isfinite(depths[j])):
        return False
    for view in projections:
        pi, pj = view[i], view[j]
        if (pi is None or pj is None or
                not all(isfinite(x) for x in (*pi, *pj))):
            return False
        if sqrt(sum((x-y)**2 for x, y in zip(pi, pj))) > half_pixel:
            return False
    return True


def choose_branch(depths, projections, cost_sets, perturb_cost_sets=None):
    """Choose one branch from two views; ties or unstable winners are invalid.

    perturb_cost_sets maps all four fixed ±0.25-pixel x/y patch shifts to
    pairs of complete per-member cost arrays, one for each witness.
    """
    if len(cost_sets) != 2 or len(projections) != 2:
        raise ValueError("exactly two witnesses required")
    count = len(depths)
    if any(len(v) != count for v in (*cost_sets, *projections)):
        raise ValueError("member array length mismatch")
    if not isinstance(perturb_cost_sets, dict) or set(perturb_cost_sets) != PERTURB_OFFSETS:
        return (), "missing_perturbations"
    if any(len(shifted) != 2 or any(len(v) != count for v in shifted)
           for shifted in perturb_cost_sets.values()):
        return (), "invalid_perturbation_shape"

    def valid_cost(c):
        return c is not None and isfinite(c) and 0 <= c <= 1

    if not count:
        return (), "no_eligible_hypothesis"
    if any(depths[i] is None or not isfinite(depths[i]) or
           any(not valid_cost(cost_sets[v][i]) or
               projections[v][i] is None or
               not all(isfinite(x) for x in projections[v][i])
               for v in range(2)) for i in range(count)):
        return (), "invalid_hypothesis_data"
    eligible = list(range(count))
    if any(not valid_cost(shifted[v][i])
           for shifted in perturb_cost_sets.values() for v in range(2) for i in eligible):
        return (), "invalid_perturbation_cost"

    def winner(costs, view):
        ranked = sorted(eligible, key=lambda i: (costs[view][i], i))
        minimum = costs[view][ranked[0]]
        tied = [i for i in ranked if costs[view][i]-minimum <= EPS]
        if any(not same_branch(i, j, depths, projections)
               for pos, i in enumerate(tied) for j in tied[pos+1:]):
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
    for shifted in perturb_cost_sets.values():
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
        if not (isfinite(m.alpha) and isfinite(m.transmittance) and
                isfinite(m.footprint) and 0 < m.alpha < 1 and
                0 <= m.transmittance <= 1 and 0 <= m.footprint <= 1):
            return fallback, False, "invalid_weight"
        c1, c2 = costs[0][i], costs[1][i]
        if any(c is None or not isfinite(c) or c < 0 or c > 1 for c in (c1, c2)):
            return fallback, False, "invalid_cost"
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
    if not all(isfinite(x) for x in (*reference, *witness)):
        return None
    xr = sum(reference) / len(reference)
    yw = sum(witness) / len(witness)
    a = [x - xr for x in reference]
    b = [y - yw for y in witness]
    xx, yy = dot(a, a), dot(b, b)
    if not all(isfinite(x) for x in (xx, yy)) or xx <= EPS or yy <= EPS:
        return None  # no texture: do not manufacture a winning branch
    correlation = dot(a, b) / sqrt(xx * yy)
    if not isfinite(correlation):
        return None
    ncc = max(-1.0, min(1.0, correlation))
    return (1.0 - ncc) / 2.0


def plane_loss(expected, candidate, valid, target, weight=1.0):
    """Host loss denominator is preserved for invalid candidate rays."""
    depths = [c if ok else e for e, c, ok in zip(expected, candidate, valid)]
    if len(depths) != len(target) or not depths:
        raise ValueError("loss input mismatch")
    return weight * sum(abs(d-y) for d, y in zip(depths, target)) / len(depths)
