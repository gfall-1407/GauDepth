"""Finite joint-member hypothesis search; CPU only, no weights or depth readout.

Responsibilities are relative assignments in this fixed-energy model, not calibrated
probabilities. Hypothesis members are internal hypotheses, not physical surfaces.
"""
import argparse
import json

import numpy as np


def fit(depth, normal, valid, K, normal_weight=1., scale=np.log1p(.005),
        nu=4., angle=30., chunk_size=256):
    """Exhaustive global minimum over distinct member features (repeats allowed).

    Exact objective ties use the first lexicographic candidate tuple. Near-optimal
    counts and next gap use tolerance 1e-10; near ties do not imply uniqueness.
    Memory is O(valid_count * chunk_size), time O(valid_count * candidates**K).
    Invalid rows remain unknown even when normal_weight=0. All outputs are JSON-safe.
    """
    if K not in (1, 2) or normal_weight not in (0., 1.):
        raise ValueError('Require K=1 or 2 and normal_weight=0 or 1')
    if not (np.isfinite([scale, nu, angle]).all() and scale > 0 and nu > 0
            and 0 < angle <= 180 and isinstance(chunk_size, (int, np.integer))
            and chunk_size > 0):
        raise ValueError('Require positive finite scales and integer chunk_size')
    depth, normal = np.asarray(depth, np.float64), np.asarray(normal, np.float64)
    valid = np.asarray(valid, bool)
    if depth.ndim != 1 or normal.shape != (len(depth), 3) or valid.shape != depth.shape:
        raise ValueError('Expected depth/valid (N,) and normal (N,3)')
    # Scaled norm avoids overflow while retaining analyze's length > 1e-8 rule.
    largest = np.max(np.abs(normal), axis=1)
    safe = np.isfinite(normal).all(axis=1) & (largest > 0)
    unit = np.zeros_like(normal)
    np.divide(normal, largest[:, None], out=unit, where=safe[:, None])
    length = np.linalg.norm(unit, axis=1)
    good = valid & safe & (largest > 1e-8 / np.maximum(length, 1.))
    good &= np.isfinite(depth) & (depth > 0)
    ids = np.flatnonzero(good)
    labels, responsibility = np.full(len(depth), -1), np.zeros((len(depth), K))
    settings = dict(K=int(K), normal_weight=float(normal_weight), scale=float(scale),
                    nu=float(nu), angle=float(angle), chunk_size=int(chunk_size),
                    tie_tolerance=1e-10, prior='equal', candidate='joint_member_medoid')
    result = dict(labels=labels.tolist(), responsibilities=responsibility.tolist(),
                  hypothesis_member_indices=[], objective=None, hypothesis_pair_count=0,
                  near_optimal_count=0, next_objective_gap=None, collapsed=False,
                  mean_entropy=None, valid_count=int(len(ids)), unknown_count=len(depth),
                  fit_settings=settings)
    if not len(ids):
        return result
    unit = unit[good] / length[good, None]
    logz = np.log(depth[good])
    logz -= logz.mean()
    features = np.column_stack((logz, unit))
    centers, first = np.unique(features, axis=0, return_index=True)
    count = len(centers)

    def energy(which):
        delta = np.abs(logz[:, None] - centers[which, 0])
        with np.errstate(divide='ignore'):
            log_square = 2 * (np.log(delta) - np.log(scale)) - np.log(nu)
        e = ((nu + 1) / 2) * np.logaddexp(0., log_square)
        if normal_weight:
            theta = np.arccos(np.clip(unit @ centers[which, 1:].T, -1., 1.))
            e += normal_weight * np.minimum((theta / np.deg2rad(angle)) ** 2, 1.)
        return e

    def objectives():
        if K == 1:
            for start in range(0, count, chunk_size):
                js = np.arange(start, min(start + chunk_size, count))
                yield [(int(j),) for j in js], energy(js).mean(axis=0)
        else:
            for i in range(count):
                ei = energy([i])
                for start in range(i, count, chunk_size):
                    js = np.arange(start, min(start + chunk_size, count))
                    value = (np.log(2.) - np.logaddexp(-ei, -energy(js))).mean(axis=0)
                    yield [(i, int(j)) for j in js], value

    best, chosen = float('inf'), None
    for pairs, values in objectives():
        at = int(np.argmin(values))
        if values[at] < best:
            best, chosen = float(values[at]), pairs[at]
    near, following = 0, float('inf')
    for _, values in objectives():
        near += int(np.count_nonzero(values <= best + 1e-10))
        higher = values[values > best + 1e-10]
        if len(higher):
            following = min(following, float(higher.min()))
    e = energy(list(chosen))
    # Subtract row minimum before softmax, stable even for very large energies.
    relative = -(e - e.min(axis=1, keepdims=True))
    r = np.exp(relative)
    r /= r.sum(axis=1, keepdims=True)
    hard = np.argmin(e, axis=1)
    collapsed = False
    if K == 2:
        tied = np.abs(e[:, 0] - e[:, 1]) <= 1e-10
        hard[tied] = -1
        collapsed = chosen[0] == chosen[1] or bool(tied.all())
    labels[ids], responsibility[ids] = hard, r
    entropy = -np.sum(r * np.log(np.maximum(r, np.finfo(float).tiny)), axis=1)
    result.update(labels=labels.tolist(), responsibilities=responsibility.tolist(),
                  hypothesis_member_indices=ids[first[list(chosen)]].tolist(),
                  objective=best, hypothesis_pair_count=count if K == 1 else count*(count+1)//2,
                  near_optimal_count=near,
                  next_objective_gap=None if not np.isfinite(following) else following-best,
                  collapsed=collapsed, mean_entropy=float(entropy.mean()),
                  unknown_count=int(np.count_nonzero(labels == -1)))
    return result


def self_check():
    n = np.tile([0., 0., 1.], (6, 1))
    d, v = np.array([1., 1., 1., 2., 2., 2.]), np.ones(6, bool)
    one = fit(np.ones(6), n, v, 1)
    spread = n.copy(); spread[3:] = [0., 1., 0.]
    assert fit(np.ones(6), spread, v, 1)['labels'] == one['labels'] == [0]*6
    two = fit(d, n, v, 2)
    assert two['labels'][:3] == [0]*3 and two['labels'][3:] == [1]*3
    directions = fit(np.ones(6), spread, v, 2)
    assert len(set(directions['labels'])) == 2 and -1 not in directions['labels']
    same = fit(np.ones(6), n, v, 2)
    assert same['collapsed'] and same['labels'] == [-1]*6
    assert np.allclose(same['responsibilities'], .5)
    assert same['hypothesis_pair_count'] == same['near_optimal_count'] == 1
    tied = fit(np.ones(6), spread, v, 2, normal_weight=0.)
    assert tied['hypothesis_member_indices'] == [0, 0]
    assert tied['near_optimal_count'] == tied['hypothesis_pair_count'] == 3
    assert tied['next_objective_gap'] is None
    # Independent direct, unchunked enumeration including diagonal pairs.
    ds = np.array([1., 1.002, 1.02, 1.03])
    ns = np.array([[0., 0., 1.], [0., 1., 1.], [1., 0., 1.], [1., 0., 0.]])
    ns /= np.linalg.norm(ns, axis=1)[:, None]
    for k in (1, 2):
        for weight in (0., 1.):
            e = 2.5*np.log1p(((np.log(ds)[:, None]-np.log(ds))/np.log1p(.005))**2/4)
            e += weight*np.minimum((np.arccos(np.clip(ns@ns.T, -1, 1))/np.deg2rad(30))**2, 1)
            vals = ([e[:, i].mean() for i in range(4)] if k == 1 else
                    [(np.log(2)-np.logaddexp(-e[:, i], -e[:, j])).mean()
                     for i in range(4) for j in range(i, 4)])
            base = fit(ds, ns, v[:4], k, weight, chunk_size=1)
            assert np.isclose(base['objective'], min(vals), atol=1e-12, rtol=0)
            assert base['near_optimal_count'] == sum(x <= min(vals)+1e-10 for x in vals)
            higher = [x for x in vals if x > min(vals)+1e-10]
            assert np.isclose(base['next_objective_gap'], min(higher)-min(vals))
            q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
            p = [3, 0, 2, 1]
            for dd, nn in ((ds[p], ns[p]), (ds*1e100, ns), (ds, ns@q),
                           (np.tile(ds, 2), np.tile(ns, (2, 1)))):
                got = fit(dd, nn, np.ones(len(dd), bool), k, weight, chunk_size=2)
                assert np.isclose(got['objective'], base['objective'], atol=1e-10, rtol=0)
    extreme = fit([np.nextafter(0., 1.), 1e308], n[:2], v[:2], 2, scale=1e-300)
    json.dumps(extreme, allow_nan=False)
    assert np.isfinite(extreme['objective'])
    bad = fit([0., np.nan, 1., 1.], [[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1]],
              [True, True, True, False], 2)
    assert bad['labels'] == [-1]*4 and bad['responsibilities'] == [[0., 0.]]*4
    assert bad['objective'] is None and bad['unknown_count'] == 4
    partial = fit([1., 0.], n[:2], v[:2], 1)
    assert partial['labels'] == [0, -1] and partial['responsibilities'] == [[1.], [0.]]
    empty = fit([], np.empty((0, 3)), [], 1)
    assert empty['labels'] == [] and empty['objective'] is None
    print('self-check passed: global enumeration, degeneracy, invariances, stability, invalid/empty')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    self_check()
