"""In-memory 32-ray adapter; the source-verified capture loader is pending.

The caller owns the native accepted record capture and original image decode.
This module never fabricates costs or silently drops a ray from evaluation.
"""

from dataclasses import dataclass

from visible_branch_reference import (PERTURB_OFFSETS, candidate_depth,
                                      choose_branch, intersection, patch_cost,
                                      pixel_ray, project)


@dataclass(frozen=True)
class RayResult:
    pixel: tuple
    expected_depth: float
    candidate_depth: float
    valid: bool
    reason: str
    accepted_count: int
    branch_ids: tuple
    costs: tuple
    perturb_costs: dict


def evaluate_ray(members, reference_camera, reference_image, witnesses,
                 pixel, expected_depth):
    """Compute all original-image costs including four actual patch shifts.

    witnesses: exactly two (Camera, image) pairs. ``members`` must be the full
    native accepted sequence for this ray; no sorting, filtering or resampling.
    """
    if len(witnesses) != 2:
        raise ValueError("exactly two fixed witnesses required")
    x, y = pixel
    ray = pixel_ray(reference_camera, x, y)
    depths = []
    projections = [[], []]
    costs = [[], []]
    shifted = {offset: [[], []] for offset in PERTURB_OFFSETS}
    for m in members:
        hit = intersection(m, reference_camera.center, ray)
        depths.append(None if hit is None else hit[2])
        point = None if hit is None else tuple(
            o+hit[2]*v for o, v in zip(reference_camera.center, ray))
        for view, (camera, image) in enumerate(witnesses):
            projections[view].append(None if point is None else project(camera, point))
            costs[view].append(patch_cost(m, reference_camera, reference_image,
                                          camera, image, x, y))
            for dx, dy in PERTURB_OFFSETS:
                shifted[(dx, dy)][view].append(patch_cost(
                    m, reference_camera, reference_image,
                    camera, image, x+dx, y+dy))
    branch, reason = choose_branch(depths, projections, costs, shifted)
    depth, valid, depth_reason = candidate_depth(
        members, reference_camera.center, ray, costs, branch, expected_depth)
    if reason != "valid":
        depth, valid = expected_depth, False
    else:
        reason = depth_reason
    return RayResult(pixel, expected_depth, depth, valid, reason,
                     len(members), tuple(members[i].gaussian_id for i in branch),
                     tuple(tuple(c) for c in costs),
                     {offset: tuple(tuple(c) for c in shifted[offset])
                      for offset in sorted(PERTURB_OFFSETS)})


def evaluate_fixed_32(rays):
    """Keep exactly the frozen 32 rays, including every fallback entry."""
    if len(rays) != 32:
        raise ValueError("expected exactly 32 frozen rays")
    return [evaluate_ray(**ray) for ray in rays]
