"""Conditional bounded-error stereo intervals, not a certificate of true image matching.

If the correspondence/total calibration error really is <= tau pixels, its true
positive depth belongs to the returned interval. LK and forward-back agreement
do NOT establish that assumption. Unbounded/degenerate evidence stays unknown.
"""
from __future__ import annotations

import argparse
import json
import numpy as np


def _ray_terms(xy, ci, cj):
    xy = np.asarray(xy, dtype=np.float64)
    ray = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(ci["K"]).T
    world_ray = ray @ ci["c2w"][:3, :3].T
    a = world_ray @ cj["w2c"][:3, :3].T
    b = ci["c2w"][:3, 3] @ cj["w2c"][:3, :3].T + cj["w2c"][:3, 3]
    return a, b


def ray_projection(xy, depth, ci, cj):
    a, b = _ray_terms(xy, ci, cj)
    local = a * np.asarray(depth, dtype=np.float64)[:, None] + b
    image = local @ cj["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = image[:, :2] / image[:, 2, None]
    uv[local[:, 2] <= 0] = np.nan
    return uv, local[:, 2]


def depth_interval(xy, E, q, ci, cj, tau=.5):
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("Positive finite pixel error bound required")
    K = np.asarray(cj["K"])
    if abs(K[0, 1])+abs(K[1, 0]) > 1e-10:
        raise ValueError("This minimal interval solver requires zero-skew PINHOLE K")
    a, b = _ray_terms(xy, ci, cj)
    uv, z = ray_projection(xy, E, ci, cj)
    q = np.asarray(q, dtype=np.float64)
    f = np.array([K[0, 0], K[1, 1]])
    c = K[:2, 2]
    direction = (a[:, :2]*b[2]-b[:2]*a[:, 2, None])*f
    length = np.linalg.norm(direction, axis=1)
    v = direction / np.maximum(length[:, None], 1e-30)
    delta = q - uv
    perpendicular = delta - v*np.sum(delta*v, axis=1)[:, None]
    perp2 = np.sum(perpendicular**2, axis=1)
    along = np.sum((q-c)*v, axis=1)
    radius = np.sqrt(np.maximum(tau*tau-perp2, 0.))
    alpha = np.sum(v*a[:, :2]*f, axis=1)
    beta = np.sum(v*b[:2]*f, axis=1)
    ends = np.column_stack((along-radius, along+radius))
    denominator = ends*a[:, 2, None]-alpha[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        depths = (beta[:, None]-ends*b[2])/denominator
    other_z = a[:, 2, None]*depths+b[2]
    valid = (np.isfinite(q).all(axis=1) & np.isfinite(uv).all(axis=1) & (z > 0) &
             (length > 1e-12) & (perp2 <= tau*tau+1e-12) &
             (np.abs(denominator) > 1e-12).all(axis=1) &
             np.isfinite(depths).all(axis=1) & (depths > 0).all(axis=1) & (other_z > 0).all(axis=1))
    lo, hi = np.min(depths, axis=1), np.max(depths, axis=1)
    # Endpoints must actually satisfy the two-dimensional reprojection bound.
    for d in (lo, hi):
        projected, _ = ray_projection(xy, d, ci, cj)
        valid &= np.linalg.norm(projected-q, axis=1) <= tau+1e-6
    lo[~valid] = np.nan; hi[~valid] = np.nan
    return lo, hi, valid


def _patch_valid(mask, xy, radius=4):
    import cv2
    safe = cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    finite = np.isfinite(xy).all(axis=1)
    p = np.floor(np.where(np.isfinite(xy), xy, 0)).astype(np.int64)
    h, w = mask.shape
    inside = finite & (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0]+1 < w) & (p[:, 1]+1 < h)
    out = np.zeros(len(xy), dtype=bool)
    x, y = p[inside, 0], p[inside, 1]
    out[inside] = (safe[y, x] & safe[y+1, x] & safe[y, x+1] & safe[y+1, x+1]).astype(bool)
    return out


def track_pair(ref_gray, target_gray, xy, initial_uv):
    import cv2
    xy, initial_uv = np.asarray(xy, dtype=float), np.asarray(initial_uv, dtype=float)
    ref = np.rint(np.clip(np.nan_to_num(ref_gray), 0, 1)*255).astype(np.uint8)
    target = np.rint(np.clip(np.nan_to_num(target_gray), 0, 1)*255).astype(np.uint8)
    seed_valid = np.isfinite(xy).all(axis=1) & np.isfinite(initial_uv).all(axis=1)
    p = np.ascontiguousarray(np.nan_to_num(xy).astype(np.float32).reshape(-1, 1, 2))
    initial = np.ascontiguousarray(np.nan_to_num(initial_uv).astype(np.float32).reshape(-1, 1, 2))
    kwargs = dict(winSize=(9, 9), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    q, status, _ = cv2.calcOpticalFlowPyrLK(ref, target, p, initial, **kwargs)
    if q is None:
        return dict(q=np.full(xy.shape, np.nan), valid=np.zeros(len(xy), bool), fb_error=np.full(len(xy), np.nan))
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(target, ref, q, p.copy(), **kwargs)
    q = q.reshape(-1, 2).astype(float)
    if back is None:
        return dict(q=q, valid=np.zeros(len(xy), bool), fb_error=np.full(len(xy), np.nan))
    back = back.reshape(-1, 2).astype(float)
    fb = np.linalg.norm(back-xy, axis=1)
    valid = seed_valid & status.ravel().astype(bool) & status_back.ravel().astype(bool) & (fb <= .5)
    valid &= _patch_valid(np.isfinite(ref_gray), xy) & _patch_valid(np.isfinite(target_gray), q)
    return dict(q=q, valid=valid, fb_error=fb)


def solve_intervals(E, lower, upper, valid, fit=(0, 2, 4, 6), gate=(1, 3)):
    E = np.asarray(E, dtype=np.float64)
    lower, upper, valid = np.asarray(lower), np.asarray(upper), np.asarray(valid, dtype=bool)
    if lower.shape != upper.shape or valid.shape != lower.shape or lower.shape[1:] != E.shape:
        raise ValueError("Expected per-view [J,N] intervals and [N] E")
    fit, gate = list(fit), list(gate)
    good = valid & np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (lower <= upper)
    fit_n = np.sum(good[fit], axis=0)
    lo = np.max(np.where(good[fit], lower[fit], -np.inf), axis=0)
    hi = np.min(np.where(good[fit], upper[fit], np.inf), axis=0)
    feasible = (fit_n >= 3) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    candidate = np.minimum(np.maximum(E, lo), hi)
    change = feasible & ((E < lo-1e-12) | (E > hi+1e-12))
    gate_ok = np.all(good[gate] & (candidate >= lower[gate]-1e-12) & (candidate <= upper[gate]+1e-12), axis=0)
    gate_refutes_E = np.any(good[gate] & ((E < lower[gate]-1e-12) | (E > upper[gate]+1e-12)), axis=0)
    accept = change & gate_ok & gate_refutes_E
    selected = E.copy(); selected[accept] = candidate[accept]
    reason = np.full(len(E), "insufficient_correspondences", dtype="<U32")
    reason[fit_n >= 3] = "incompatible_intervals"
    reason[feasible] = "baseline_not_refuted"
    reason[change] = "gate_unverified"
    reason[accept] = "resolved_outside_baseline"
    lo[fit_n < 3] = np.nan; hi[fit_n < 3] = np.nan
    return dict(selected=selected, accept=accept, lo=lo, hi=hi, fit_n=fit_n, reason=reason)


def project_joint_intervals(E, lower, upper, valid):
    """Closest feasible depth for ALL five decision constraints; audit stays unused.

    Conditional on true depth belonging to this intersection, convex projection
    cannot increase scalar depth error. Matching/calibration bounds are not certified.
    """
    E=np.asarray(E,dtype=np.float64);lower,upper=np.asarray(lower),np.asarray(upper)
    valid=np.asarray(valid,dtype=bool)
    if lower.shape!=(5,len(E)) or upper.shape!=lower.shape or valid.shape!=lower.shape:
        raise ValueError("Exactly five preassigned decision intervals required")
    good=valid&np.isfinite(lower)&np.isfinite(upper)&(lower>0)&(lower<=upper)
    complete=good.all(axis=0)
    lo=np.max(np.where(good,lower,-np.inf),axis=0);hi=np.min(np.where(good,upper,np.inf),axis=0)
    feasible=complete&(lo<=hi)&np.isfinite(lo)&np.isfinite(hi)
    change=feasible&((E<lo-1e-12)|(E>hi+1e-12))
    selected=E.copy();selected[change]=np.clip(E,lo,hi)[change]
    reason=np.full(len(E),"insufficient_joint_constraints",dtype="<U32")
    reason[complete]="incompatible_joint_intervals";reason[feasible]="baseline_inside_joint";reason[change]="projected_to_joint"
    lo[~complete]=np.nan;hi[~complete]=np.nan
    return dict(selected=selected,accept=change,lo=lo,hi=hi,fit_n=good[:3].sum(axis=0),reason=reason)


def synthetic_check():
    import cv2
    K = np.array([[160., 0, 63.5], [0, 160., 47.5], [0, 0, 1.]])
    def camera(x):
        c2w = np.eye(4); c2w[0, 3] = x
        return dict(K=K, c2w=c2w, w2c=np.linalg.inv(c2w), image_size=(96, 128))
    ci = camera(0)
    xy = np.array([[63.5, 47.5], [50., 40.], [70., 60.]])
    true = np.full(3, 2.)
    lower, upper, allvalid = [], [], []
    baselines = (.3, -.15, .25, -.18, .22, -.2, .26, -.21)
    for baseline in baselines:
        cj = camera(baseline);q, _ = ray_projection(xy, true, ci, cj)
        q += [.2, .1]
        lo, hi, valid = depth_interval(xy, np.full(3, 2.2), q, ci, cj)
        assert valid.all() and (lo <= true).all() and (hi >= true).all()
        lower.append(lo);upper.append(hi);allvalid.append(valid)
    lower, upper, allvalid = np.array(lower), np.array(upper), np.array(allvalid)
    baseline = solve_intervals(true, lower, upper, allvalid)
    assert not baseline["accept"].any();np.testing.assert_array_equal(baseline["selected"], true)
    # A gate may legitimately reject the nearest fit-boundary point under
    # different bounded perturbations. The positive correction control uses
    # coherent exact correspondences and fit parallax stronger than gate.
    clean = []
    for baseline in baselines:
        cj = camera(baseline);q, _ = ray_projection(xy, true, ci, cj)
        clean.append(depth_interval(xy, np.full(3, 2.2), q, ci, cj))
    cl, cu, cv = (np.array([x[k] for x in clean]) for k in range(3))
    wrong = solve_intervals(np.full(3, 2.2), cl, cu, cv)
    assert wrong["accept"].all() and (np.abs(wrong["selected"]-true) < .2).all()
    cu_gate = cu.copy(); cl_gate = cl.copy()
    cl_gate[[1, 3]] = 1.5; cu_gate[[1, 3]] = 2.5
    assert not solve_intervals(np.full(3, 2.2), cl_gate, cu_gate, cv)["accept"].any()
    q, _ = ray_projection(xy, true, ci, ci)
    assert not depth_interval(xy, true, q, ci, ci)[2].any()
    q, _ = ray_projection(xy, true, ci, camera(.2));q[:, 1] += 1.
    assert not depth_interval(xy, true, q, ci, camera(.2))[2].any()
    pure = np.full((96, 128), .5)
    assert not track_pair(pure, pure, xy, xy)["valid"].any()
    rng = np.random.default_rng(20261026)
    textured = rng.uniform(.1, .9, (96, 128))
    shifted = cv2.warpAffine(textured, np.float32([[1, 0, 3], [0, 1, 0]]), (128, 96))
    # Real reference samples are integer pixel centres. Fractional LK source
    # locations can converge to an alias; the FB gate must then reject them.
    tracked_xy = np.rint(xy)
    tracked = track_pair(textured, shifted, tracked_xy, tracked_xy+[3., 0.])
    assert tracked["valid"].all()
    np.testing.assert_allclose(tracked["q"], tracked_xy+[3., 0.], rtol=0, atol=.05)
    # The nearest fit boundary may fail a gate even when the joint set is nonempty.
    jl=np.array([[1.],[1.],[1.],[2.],[2.]])
    ju=np.array([[3.],[3.],[3.],[4.],[4.]])
    jv=np.ones((5,1),bool)
    assert not solve_intervals(np.array([.5]),jl,ju,jv,fit=[0,1,2],gate=[3,4])["accept"][0]
    joint=project_joint_intervals(np.array([.5]),jl,ju,jv)
    np.testing.assert_array_equal(joint["selected"],[2.])
    for truth in np.linspace(2.,3.,21):
        for start in np.linspace(.5,4.5,21):
            value=project_joint_intervals(np.array([start]),jl,ju,jv)["selected"][0]
            assert abs(value-truth)<=abs(start-truth)+1e-12
    jl_bad=jl.copy();jl_bad[4]=3.5
    assert not project_joint_intervals(np.array([.5]),jl_bad,ju,jv)["accept"][0]
    jv[4]=False
    assert not project_joint_intervals(np.array([.5]),jl,ju,jv)["accept"][0]
    print(json.dumps(dict(status="PASS", synthetic_only=True,
        true_in_intervals=True, correct_baseline_unchanged=True, wrong_baseline_moved_toward_truth=True,
        zero_baseline="unknown", off_epipolar="unknown", pure_colour="unknown", LK_translation="PASS",joint_projection="PASS",
        limitation="Conditional <=.5px correspondence/calibration error bound, NOT certified by LK/FB")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--check", action="store_true")
    if p.parse_args().check:synthetic_check()
