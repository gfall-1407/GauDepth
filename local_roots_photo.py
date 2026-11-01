"""Frozen local_roots_photo_v1 selection: geometry roots plus image evidence.

Independent of the older per-Gaussian choose_branch and plane LS depth.
"""

from math import hypot, isfinite, sqrt


OFFSETS = ((0., 0.), (-.25, 0.), (.25, 0.), (0., -.25), (0., .25))
THRESHOLD = .25
PIXEL_CLASS = .5
TIE_EPS = 1e-10


def validated_root(root, center_index, fine_root_index):
    if not root.get("matched_admissible"):
        raise ValueError("non-candidate root passed to selector")
    z = root["z"]
    n = root["surface_normal_world"]
    g = root["gradF_world"]
    if (not isfinite(z) or z <= 0 or len(n) != 3 or len(g) != 3 or
            not all(isfinite(v) for v in (*n, *g))):
        raise ValueError(f"corrupt root {center_index}/{fine_root_index}")
    n_norm = sqrt(sum(v*v for v in n))
    g_norm = sqrt(sum(v*v for v in g))
    alignment = abs(sum(a*b for a, b in zip(n, g)) / (n_norm*g_norm))
    if abs(n_norm-1) > 1e-6 or alignment < 1-1e-6:
        raise ValueError(f"normal is not the surface gradient: {center_index}/{fine_root_index}")
    return [center_index, fine_root_index]


def finalize_root(root_id, z, point_world, normal, outside_all_3sigma, evidence):
    """Five samples per witness or an unknown [0,1] interval."""
    intervals = {}
    complete = True
    for view in ("0012", "0050"):
        samples = evidence[view]
        if len(samples) != 5 or [tuple(s["offset"]) for s in samples] != list(OFFSETS):
            raise ValueError("missing or reordered fixed offsets")
        costs = [s["cost"] for s in samples]
        if any(c is not None and (not isfinite(c) or c < 0 or c > 1)
               for c in costs):
            raise ValueError("corrupt observed NCC cost")
        if any(c is None for c in costs):
            intervals[view] = [0., 1.]
            complete = False
        else:
            intervals[view] = [min(costs), max(costs)]
    return dict(root_id=root_id, z=z, point_world=point_world,
                surface_normal_world=normal,
                outside_all_3sigma=outside_all_3sigma, evidence=evidence,
                view_intervals=intervals,
                cost_lower=max(intervals[v][0] for v in intervals),
                cost_upper=max(intervals[v][1] for v in intervals),
                complete_observation=complete)


def compatible(a, b):
    for view in ("0012", "0050"):
        pa = a["evidence"][view][0]["projected_center"]
        pb = b["evidence"][view][0]["projected_center"]
        if pa is None or pb is None or not all(isfinite(x) for x in (*pa, *pb)):
            return False
        if hypot(pa[0]-pb[0], pa[1]-pb[1]) > PIXEL_CLASS:
            return False
    return True


def decide(roots, expected):
    """Keep every possible competitor, including roots with missing evidence."""
    if not isfinite(expected):
        raise ValueError("corrupt native Expected depth")
    if not roots:
        return dict(decision="no_geometry_candidate", best_complete_root=None,
                    possible_competitor_ids=[], unresolved_competitor_ids=[],
                    selected_root=None, output_depth=expected)
    complete = [r for r in roots if r["complete_observation"]]
    if not complete:
        reason = "no_complete_evidence"
        best = None
        possible = roots
    else:
        best = min(complete, key=lambda r: (r["cost_upper"], r["root_id"]))
        possible = [r for r in roots if r["cost_lower"] <= best["cost_upper"]+TIE_EPS]
        reason = "weak_match" if best["cost_upper"] > THRESHOLD else None
    pairs = [(a, b) for i, a in enumerate(possible) for b in possible[i+1:]
             if not compatible(a, b)]
    unresolved = [] if best is None else [r for r in possible
                                          if any(r is a or r is b for a, b in pairs)]
    if reason is None:
        if any(not a["complete_observation"] or not b["complete_observation"]
               for a, b in pairs):
            reason = "conditional_missing_competitor"
        elif pairs:
            reason = "ambiguous_observed_roots"
        else:
            reason = "selected"
    selected = best if reason == "selected" else None
    return dict(decision=reason,
                best_complete_root=None if best is None else best["root_id"],
                possible_competitor_ids=[r["root_id"] for r in possible],
                unresolved_competitor_ids=[r["root_id"] for r in unresolved],
                selected_root=None if selected is None else selected["root_id"],
                output_depth=expected if selected is None else selected["z"])
