"""Conditional first-hit controls and audit of fixed native accepted rays.

Uniform finite-support hazards are analytic controls, not a Gaussian-to-surface model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "数据"


def indexed(suffix):
    index = json.loads((DATA / "来源索引.json").read_text(encoding="utf8"))
    rows = [r for r in index["files"] if r["path"].replace("\\", "/").endswith(suffix)]
    if len(rows) != 1:
        raise ValueError((suffix, len(rows)))
    path = DATA / rows[0]["path"]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != rows[0]["sha256"]:
        raise ValueError(f"input SHA mismatch: {path}")
    return path, digest


def uniform_hazards(alpha, intervals, representative):
    """Integrate independent constant hazards exactly on each support segment."""
    alpha = np.asarray(alpha, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    d = np.asarray(representative, dtype=float)
    n = len(alpha)
    if n == 0 or intervals.shape != (n, 2) or d.shape != (n,) or not np.all(np.isfinite(intervals)):
        raise ValueError("shape or finite interval error")
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0) or np.any(alpha >= 1) or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] <= intervals[:, 0]) or not np.all(np.isfinite(d)):
        raise ValueError("invalid alpha, interval, or representative")
    tau = -np.log1p(-alpha)
    rate = tau / np.diff(intervals, axis=1).ravel()
    endpoints = sorted(set(intervals.ravel()))
    mass = np.zeros(n)
    moment = np.zeros(n)
    prefix_integral = np.zeros(n - 1)
    L = np.zeros(n)
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        h = np.where((intervals[:, 0] <= left) & (intervals[:, 1] >= right), rate, 0.)
        width = right - left
        total = h.sum()
        if total > 0:
            T = math.exp(-L.sum())
            q = -math.expm1(-total * width) / total
            local = T * h * q
            mass += local
            x = total * width
            if x < 1e-3:
                first = width**2 * (0.5 - x/3 + x*x/8 - x**3/30 + x**4/144)
            else:
                first = (-math.expm1(-x) - x * math.exp(-x)) / total**2
            moment += left * local + T * h * first
            for k in range(1, n):
                early = h[:k].sum()
                late_L = L[k:].sum()
                # Prefix integrand evaluated analytically as two exponential integrals.
                if early:
                    q_early = -math.expm1(-early * width) / early
                    prefix_integral[k-1] += math.exp(-L[:k].sum()) * early * (
                        q_early - math.exp(-late_L) * q)
        L += h * width
    native_T = np.r_[1., np.cumprod(1 - alpha)[:-1]]
    w = native_T * alpha
    A = w.sum()
    delta = np.cumsum(w - mass)[:-1]
    if A <= 0:
        return dict(A=0., native_weights=w.tolist(), continuous_weights=mass.tolist())
    internal = float((moment - d * mass).sum() / A)
    between = float(np.dot(delta, np.diff(d)) / A)
    direct = float((moment.sum() - np.dot(w, d)) / A)
    return dict(A=float(A), native_weights=w.tolist(), continuous_weights=mass.tolist(),
                prefix_delta=delta.tolist(), prefix_integral=prefix_integral.tolist(),
                internal=internal, between=between, direct=direct,
                residual=float(direct - internal - between),
                total_residual=float(A - mass.sum()))


def controls():
    cases = {
        "single": ([.5], [[1., 2.]], [1.5]),
        "ordered_disjoint": ([.5, .5], [[1., 2.], [3., 4.]], [1.5, 3.5]),
        "partial_overlap": ([.5, .5], [[1., 3.], [2., 4.]], [2., 3.]),
        "identical": ([.5, .5], [[1., 3.], [1., 3.]], [2., 2.]),
        "reverse_disjoint": ([.5, .5], [[3., 4.], [1., 2.]], [3.5, 1.5]),
        "tiny": ([1e-12, 2e-12], [[1., 2.], [1.5, 3.]], [2., 1.]),
        "zero": ([0., 0.], [[1., 2.], [2., 3.]], [1., 2.]),
        "three": ([.2, .4, .3], [[1., 3.], [1.5, 2.5], [2., 4.]], [3., 1., 2.]),
    }
    out = {name: uniform_hazards(*args) for name, args in cases.items()}
    for name, row in out.items():
        if name == "zero":
            np.testing.assert_equal(row["A"], 0.)
            continue
        np.testing.assert_allclose([row["total_residual"], row["residual"]], 0, atol=1e-12)
        np.testing.assert_allclose(row["prefix_delta"], row["prefix_integral"], atol=1e-12)
        numeric = numerical_oracle(*cases[name])
        coarse = numerical_oracle(*cases[name], nodes_count=128)
        np.testing.assert_allclose(numeric["mass"], coarse["mass"], rtol=1e-11, atol=1e-20)
        np.testing.assert_allclose(numeric["moment"], coarse["moment"], rtol=1e-11, atol=1e-20)
        np.testing.assert_allclose(row["continuous_weights"], numeric["mass"], rtol=1e-10, atol=1e-20)
        np.testing.assert_allclose(row["direct"] * row["A"] + np.dot(row["native_weights"], cases[name][2]), numeric["moment"], rtol=1e-10, atol=1e-20)
        np.testing.assert_allclose(row["prefix_integral"], numeric["prefix"], rtol=1e-10, atol=1e-20)
        if name == "ordered_disjoint":
            np.testing.assert_allclose(row["prefix_delta"], 0, atol=1e-12)
        if name == "identical":
            np.testing.assert_allclose(row["continuous_weights"], [.375, .375], atol=1e-12)
        if name == "reverse_disjoint":
            np.testing.assert_allclose(row["prefix_delta"], .25, atol=1e-12)
    for bad in (float("nan"), float("inf"), -0.1, 1.):
        try:
            uniform_hazards([bad], [[1., 2.]], [1.])
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid alpha accepted: {bad}")
    for a in (1e-8, 1e-10, 1e-12):
        tiny = uniform_hazards([a], [[1., 2.]], [1.5])
        oracle = numerical_oracle([a], [[1., 2.]], [1.5])
        mean = (tiny["direct"] * tiny["A"] + 1.5*tiny["A"]) / tiny["A"]
        oracle_mean = oracle["moment"] / oracle["mass"][0]
        assert 1. <= mean <= 2.
        np.testing.assert_allclose(mean, oracle_mean, atol=1e-10, rtol=0)
    assert replay_transmittance(np.empty(0)).shape == (0,)
    return out


def replay_transmittance(alpha):
    return np.r_[1., np.cumprod(1-alpha)[:-1]] if len(alpha) else np.empty(0)


def numerical_oracle(alpha, intervals, representative, nodes_count=256):
    """Independent Gauss-Legendre quadrature of pointwise T(t) * lambda(t)."""
    alpha = np.asarray(alpha, float)
    intervals = np.asarray(intervals, float)
    tau = -np.log1p(-alpha)
    rate = tau / (intervals[:, 1] - intervals[:, 0])
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    ends = sorted(set(intervals.ravel()))
    mass = np.zeros(len(alpha))
    moment = 0.
    prefix = np.zeros(len(alpha)-1)
    for a, b in zip(ends[:-1], ends[1:]):
        t = (a+b)/2 + (b-a)/2 * nodes
        dt = (b-a)/2 * weights
        L = np.clip(t[:, None]-intervals[:, 0], 0, intervals[:, 1]-intervals[:, 0]) * rate
        h = ((t[:, None] >= intervals[:, 0]) & (t[:, None] < intervals[:, 1])) * rate
        T = np.exp(-L.sum(axis=1))
        mass += ((dt*T)[:, None]*h).sum(axis=0)
        moment += float(np.sum(dt*t*T*h.sum(axis=1)))
        for k in range(1, len(alpha)):
            integrand = np.exp(-L[:, :k].sum(axis=1)) * (-np.expm1(-L[:, k:].sum(axis=1))) * h[:, :k].sum(axis=1)
            prefix[k-1] += np.sum(dt*integrand)
    return dict(mass=mass, moment=moment, prefix=prefix)


def native_audit():
    prefix = "20261026_夜间深度方法/confirm_v1/scan37/"
    trace, trace_sha = indexed(prefix + "view_000.npz")
    meta, meta_sha = indexed(prefix + "metadata.json")
    m = json.loads(meta.read_text(encoding="utf8"))
    if m["views"][0]["npz_sha256"] != trace_sha:
        raise ValueError("metadata trace mismatch")
    model, model_sha = indexed("20261026_夜间深度方法/inputs/scan37/point_cloud.ply")
    cfg, cfg_sha = indexed("20261026_夜间深度方法/inputs/scan37/cfg_args")
    if model_sha != m["model_sha256"] or cfg_sha != m["cfg_sha256"]:
        raise ValueError("metadata model/config mismatch")
    with np.load(trace) as f:
        offsets = f["offsets"]
        members = len(f["gaussian_id"])
        if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2
                or offsets[0] != 0 or offsets[-1] != members or np.any(np.diff(offsets) < 0)
                or any(len(f[key]) != members for key in ("depth", "alpha", "transmittance", "weight"))):
            raise ValueError("invalid accepted-list structure")
        if len(offsets)-1 < 32:
            raise ValueError("fewer than 32 distinct recorded rays")
        selected = np.linspace(0, len(f["offsets"]) - 2, 32, dtype=int)
        if len(set(selected)) != 32:
            raise ValueError("duplicate selected rays")
        rows = []
        for ray in selected:
            a, b = map(int, f["offsets"][ray:ray+2])
            alpha = f["alpha"][a:b].astype(float)
            T = f["transmittance"][a:b].astype(float)
            w = f["weight"][a:b].astype(float)
            depth = f["depth"][a:b].astype(float)
            if not (np.isfinite(alpha).all() and np.isfinite(T).all() and np.isfinite(w).all() and np.isfinite(depth).all()):
                raise ValueError(f"nonfinite native ray {ray}")
            if np.any(alpha < 0) or np.any(alpha >= 1) or np.any(T < 0) or np.any(T > 1) or np.any(w < 0):
                raise ValueError(f"invalid native range {ray}")
            replay_T = replay_transmittance(alpha)
            rows.append(dict(ray=int(ray), count=int(b-a), native_mass=float(w.sum()),
                             max_weight_product_error=float(np.max(abs(w-T*alpha))) if b>a else None,
                             max_transmittance_recurrence_error=float(np.max(abs(T-replay_T))) if b>a else None))
    return dict(trace_sha256=trace_sha, metadata_sha256=meta_sha, model_sha256=model_sha,
                cfg_sha256=cfg_sha, recorded_commit=m["git_commit"],
                host_binary_sha256=m["host_binary_sha256"], host_build_provenance=m["host_build_provenance"],
                mapping_verified=False, rays=rows,
                limitation="Native accepted-list audit only; no Gaussian spatial hazard mapping or GT claim.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["controls", "native"])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = controls() if args.mode == "controls" else native_audit()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(payload)
    print(payload)
