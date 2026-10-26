"""Reference-excluded track triangulation; no surface fitting or real-data runner.

Coordinates use NumPy pixel centres (0, 0). Normalized-camera DLT minimizes
an algebraic homogeneous objective, not pixel maximum likelihood or optimal
triangulation. LOO displacement is empirical stability in world units, NOT an
error bound/confidence interval. Shared calibration/localization uncertainty
and the relationship of a feature anchor to a query pixel remain uncertified.
"""

import argparse
import json

import numpy as np


def triangulate_track(observations, cameras, excluded_reference):
    """Return a dict with status/valid/reason, xyz, used_views, loo_xyz,
    max_loo_displacement and reprojection_residual_px (per-view L2 norms).

    observations maps view name to xy[2]; cameras maps name to K[3,3],
    c2w[4,4], w2c[4,4]. At least four non-reference views are required.
    All supplied non-reference observations are used; no residual-based view
    removal, GT, existing SfM XYZ or model depth participates. On any full/LOO
    failure xyz and spread are None; diagnostic partial LOO entries may remain.
    Numerical degeneracy tolerances do not certify physical uncertainty.
    """
    names = sorted(name for name in observations if name != excluded_reference)
    result = dict(status="unknown", valid=False, reason=None, xyz=None,
                  used_views=names, loo_xyz={}, max_loo_displacement=None,
                  reprojection_residual_px={})

    def unknown(reason):
        result["reason"] = reason
        return result

    if len(names) < 4:
        return unknown("fewer_than_four_non_reference_views")
    prepared = {}
    try:
        for name in names:
            xy = np.asarray(observations[name], dtype=np.float64)
            camera = cameras[name]
            K = np.asarray(camera["K"], dtype=np.float64)
            c2w = np.asarray(camera["c2w"], dtype=np.float64)
            w2c = np.asarray(camera["w2c"], dtype=np.float64)
            if xy.shape != (2,) or K.shape != (3, 3) or c2w.shape != (4, 4) or w2c.shape != (4, 4):
                return unknown("invalid_shape:" + str(name))
            if not all(np.isfinite(a).all() for a in (xy, K, c2w, w2c)):
                return unknown("nonfinite_input:" + str(name))
            if (not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(w2c @ c2w, np.eye(4), atol=1e-7, rtol=1e-7)):
                return unknown("inconsistent_camera_transforms:" + str(name))
            ray = np.linalg.solve(K, np.r_[xy, 1.0])
            if not np.isfinite(ray).all() or abs(ray[2]) <= 1e-12 * np.linalg.norm(ray):
                return unknown("invalid_normalized_ray:" + str(name))
            prepared[name] = (xy, K, c2w, w2c, ray[:2] / ray[2])
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return unknown("invalid_input:" + type(exc).__name__)

    def solve(selected):
        centres = np.array([prepared[n][2][:3, 3] for n in selected])
        baseline = np.max(np.linalg.norm(centres - centres[0], axis=1))
        if baseline <= 1e-12 * max(1.0, np.max(np.linalg.norm(centres, axis=1))):
            return None, "zero_baseline"
        rows = []
        for name in selected:
            P = prepared[name][3][:3]
            u, v = prepared[name][4]
            rows.extend((u * P[2] - P[0], v * P[2] - P[1]))
        A = np.asarray(rows)
        try:
            _, s, vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, "svd_failed"
        if not np.isfinite(s).all() or s[0] <= 0 or s[-2] <= 1e-10 * s[0]:
            return None, "rank_deficient"
        # A non-unique smallest singular vector is not a usable DLT solution.
        if s[-2] - s[-1] <= 1e-10 * s[0]:
            return None, "ambiguous_null_direction"
        h = vh[-1]
        if not np.isfinite(h).all() or abs(h[3]) <= 1e-12 * np.linalg.norm(h):
            return None, "point_at_infinity"
        xyz = h[:3] / h[3]
        if not np.isfinite(xyz).all():
            return None, "nonfinite_xyz"
        # Check even the omitted LOO view; a behind-camera anchor is unknown.
        for name in names:
            q = prepared[name][3] @ np.r_[xyz, 1.0]
            if not np.isfinite(q).all() or q[2] <= 1e-12 * max(1.0, np.linalg.norm(q[:3])):
                return None, "nonpositive_depth:" + str(name)
        return xyz, None

    xyz, reason = solve(names)
    if reason:
        return unknown("full:" + reason)
    loo = {}
    for omitted in names:
        point, reason = solve([n for n in names if n != omitted])
        if reason:
            return unknown("loo:" + str(omitted) + ":" + reason)
        loo[omitted] = point
        result["loo_xyz"][omitted] = point.tolist()
    for name in names:
        xy, K, _, w2c, _ = prepared[name]
        q = K @ (w2c @ np.r_[xyz, 1.0])[:3]
        if not np.isfinite(q).all() or abs(q[2]) <= 1e-12 * np.linalg.norm(q):
            return unknown("invalid_reprojection:" + str(name))
        residual = float(np.linalg.norm(q[:2] / q[2] - xy))
        if not np.isfinite(residual):
            return unknown("nonfinite_residual:" + str(name))
        result["reprojection_residual_px"][name] = residual
    spread = float(max(np.linalg.norm(point - xyz) for point in loo.values()))
    if not np.isfinite(spread):
        return unknown("nonfinite_loo_displacement")
    result.update(status="valid", valid=True, reason="ok", xyz=xyz.tolist(),
                  max_loo_displacement=spread)
    return result


def surface_failure_controls():
    """Exact counterexamples: empirical LOO stability is not surface coverage."""
    angle=np.arange(16)*2*np.pi/16
    radius=.5;curvature=.4;true_depth=2.
    cloud=np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.full(16,true_depth+curvature*radius**2)))
    def plane_depth(points):
        center=points.mean(axis=0);_,_,v=np.linalg.svd(points-center,full_matrices=False)
        n=v[-1];return float(n@center/n[2]),float(np.sqrt(np.mean(((points-center)@n)**2)))
    full,residual=plane_depth(cloud)
    deleted=np.array([plane_depth(np.delete(cloud,i,axis=0))[0] for i in range(16)])
    assert abs(full-2.1)<1e-12 and residual<1e-12 and np.ptp(deleted)<1e-12
    assert full-true_depth>.09
    # The central ray is inside the ring's projected convex hull, but its
    # true curved-surface intersection is not identified by the anchor plane.
    K=np.array([[160.,0,63.5],[0,160.,47.5],[0,0,1.]])
    cameras={};biased={};obs={};point=np.array([.12,-.08,2.])
    for i,xy in enumerate(((0,0),(-.4,0),(.4,0),(0,-.4),(0,.4))):
        c2w=np.eye(4);c2w[:2,3]=xy;w2c=np.linalg.inv(c2w);name=str(i)
        cameras[name]=dict(K=K,c2w=c2w,w2c=w2c)
        q=K@(w2c@np.r_[point,1.])[:3];obs[name]=q[:2]/q[2]
        wrong=K.copy();wrong[0,0]*=1.001;wrong[1,1]*=1.001
        biased[name]=dict(K=wrong,c2w=c2w,w2c=w2c)
    anchor=triangulate_track(obs,biased,"0")
    assert anchor["valid"] and abs(anchor["xyz"][2]-2.002)<1e-10
    assert anchor["max_loo_displacement"]<1e-10
    return dict(status="PASS_constructed_failures",ring=dict(true_query_depth=true_depth,proposed_plane_depth=full,fit_rms=residual,delete_one_depth_range=float(np.ptp(deleted)),query_inside_convex_hull=True),
        shared_focal_bias=dict(relative_focal_error=.001,true_depth=2.,triangulated_depth=anchor["xyz"][2],max_loo_displacement=anchor["max_loo_displacement"]),
        limitation="These refute empirical-LOO-as-coverage, not attribute DTU errors. Curvature, holes, surface membership and calibration bounds are additional assumptions.")


def _check():
    K = np.array([[700., 0, 320], [0, 680., 240], [0, 0, 1]])
    xyz = np.array([.12, -.08, 3.])
    centres = [[0, 0, 0], [-.4, 0, 0], [.4, .1, 0], [0, -.4, .05], [.1, .4, -.05]]
    cameras, observations = {}, {}
    for index, centre in enumerate(centres):
        c2w = np.eye(4)
        c2w[:3, 3] = centre
        w2c = np.linalg.inv(c2w)
        name = str(index)
        cameras[name] = dict(K=K, c2w=c2w, w2c=w2c)
        q = K @ (w2c @ np.r_[xyz, 1])[:3]
        observations[name] = q[:2] / q[2]
    exact = triangulate_track(observations, cameras, "0")
    assert exact["valid"] and np.linalg.norm(np.array(exact["xyz"]) - xyz) < 1e-10
    assert "0" not in exact["used_views"] and len(exact["loo_xyz"]) == 4
    changed = dict(observations, **{"0": observations["0"] + 100})
    assert triangulate_track(changed, cameras, "0") == exact
    insufficient = triangulate_track({n: observations[n] for n in ("0", "1", "2", "3")}, cameras, "0")
    assert not insufficient["valid"]
    same_cameras = {n: cameras["0"] for n in observations}
    same_obs = {n: observations["0"] for n in observations}
    zero = triangulate_track(same_obs, same_cameras, "0")
    assert not zero["valid"] and "zero_baseline" in zero["reason"]
    loo_cameras, loo_obs = dict(same_cameras), dict(same_obs)
    loo_cameras["4"], loo_obs["4"] = cameras["4"], observations["4"]
    invalid_loo = triangulate_track(loo_obs, loo_cameras, "0")
    assert not invalid_loo["valid"] and invalid_loo["reason"] == "loo:4:zero_baseline"
    behind = {}
    for name, camera in cameras.items():
        q = K @ (camera["w2c"] @ np.array([.12, -.08, -3., 1]))[:3]
        behind[name] = q[:2] / q[2]
    back = triangulate_track(behind, cameras, "0")
    assert not back["valid"] and "nonpositive_depth" in back["reason"]
    noise = np.array([[0, 0], [.08, -.03], [-.04, .06], [.02, .05], [-.07, -.04]])
    noisy = triangulate_track({n: observations[n] + noise[int(n)] for n in observations}, cameras, "0")
    assert noisy["valid"] and np.isfinite(noisy["max_loo_displacement"]) and noisy["max_loo_displacement"] > 0
    assert all(np.isfinite(point).all() for point in noisy["loo_xyz"].values())
    print(json.dumps(dict(check="PASS", exact_xyz=exact["xyz"],
                          noisy_max_loo_displacement=noisy["max_loo_displacement"],
                          zero_baseline_reason=zero["reason"], behind_reason=back["reason"],
                          surface_failures=surface_failure_controls(),
                          limitation="LOO displacement is empirical stability, not a true error bound"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    _check()
