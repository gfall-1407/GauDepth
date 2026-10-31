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
    if intervals.shape != (n, 2) or d.shape != (n,) or not np.all(np.isfinite(intervals)):
        raise ValueError("shape or finite interval error")
    if np.any(alpha < 0) or np.any(alpha >= 1) or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] <= intervals[:, 0]) or not np.all(np.isfinite(d)):
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
            first = (1 - math.exp(-total * width) * (1 + total * width)) / total**2
            moment += left * local + T * h * first
            for k in range(1, n):
                early = h[:k].sum()
                late_L = L[k:].sum()
                late_h = h[k:].sum()
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
    }
    out = {name: uniform_hazards(*args) for name, args in cases.items()}
    for name, row in out.items():
        np.testing.assert_allclose([row["total_residual"], row["residual"]], 0, atol=1e-12)
        np.testing.assert_allclose(row["prefix_delta"], row["prefix_integral"], atol=1e-12)
        if name == "ordered_disjoint":
            np.testing.assert_allclose(row["prefix_delta"], 0, atol=1e-12)
        if name == "identical":
            np.testing.assert_allclose(row["continuous_weights"], [.375, .375], atol=1e-12)
        if name == "reverse_disjoint":
            np.testing.assert_allclose(row["prefix_delta"], .25, atol=1e-12)
    return out


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
        selected = np.linspace(0, len(f["offsets"]) - 2, 32, dtype=int)
        rows = []
        for ray in selected:
            a, b = map(int, f["offsets"][ray:ray+2])
            alpha = f["alpha"][a:b].astype(float)
            T = f["transmittance"][a:b].astype(float)
            w = f["weight"][a:b].astype(float)
            depth = f["depth"][a:b].astype(float)
            if not (np.isfinite(alpha).all() and np.isfinite(T).all() and np.isfinite(w).all() and np.isfinite(depth).all()):
                raise ValueError(f"nonfinite native ray {ray}")
            replay_T = np.r_[1., np.cumprod(1-alpha)[:-1]]
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
