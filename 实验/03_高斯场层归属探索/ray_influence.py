"""Per-ray influence diagnostics for already accepted Gaussian contributions."""
from __future__ import annotations

import math
import sys

import numpy as np

_STAT_NAMES = ("accepted", "total_w", "min_w", "max_w", "top1_fraction", "top10_count_mass",
    "n90_count_fraction", "weighted_median", "tail_mass", "tail_mean", "tail_shift",
    "far_high_mass", "far_high_count", "far_high_mean", "has_far_high", "far_high_shift",
    "far_single_p", "far_single_depth", "far_single_loo", "has_far_single", "far_single_shift") + tuple(
    f"{name}_{suffix}" for name in ("all", "top10", "top50", "cover90")
    for suffix in ("unweighted_sd", "weighted_sd", "sd_ratio"))


def _numpy_reference(depth, weight, counts):
    """Independent scalar reference used by --check (one row at a time)."""
    result = []
    for b, n in enumerate(counts):
        z, w = depth[b, :n].astype(float), weight[b, :n].astype(float)
        result.append(_one_ray(z, w))
    return result


def _one_ray(z, w):
    n = len(z)
    nan = float("nan")
    result = {"mean": [nan] * 6, "sizes": [0, 0, 0, 0], "stats": {}}
    if not n:
        return result
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("accepted weights must be finite and > 0")
    total = math.fsum(w.tolist())
    order = np.argsort(-w, kind="stable")
    c10, c50 = math.ceil(.1 * n), math.ceil(.5 * n)
    c90 = int(np.searchsorted(np.cumsum(w[order]), .9 * total, side="left") + 1)
    sets = [np.arange(n), order[:c10], order[:c50], order[:c90]]
    result["sizes"] = [len(s) for s in sets]
    finite_all = np.isfinite(z).all()

    def avg(ix):
        if not len(ix) or not np.isfinite(z[ix]).all():
            return nan
        sw = math.fsum(w[ix].tolist())
        return math.fsum((w[ix] * z[ix]).tolist()) / sw

    means = [avg(s) for s in sets]
    result["mean"][:4] = means
    med = nan
    if finite_all:
        ix = np.argsort(z, kind="stable")
        med = float(z[ix[np.searchsorted(np.cumsum(w[ix]), .5 * total, side="left")]])
    top10 = sets[1]
    far = top10[z[top10] > med] if np.isfinite(med) else np.array([], dtype=int)
    far_mask = np.zeros(n, dtype=bool)
    far_mask[far] = True
    far_mass = math.fsum(w[far].tolist()) if len(far) else 0.
    base = means[0]
    result["mean"][4] = avg(np.flatnonzero(~far_mask)) if len(far) else base
    candidates = []
    if len(far) and np.isfinite(base):
        for j in far:
            p = w[j] / total
            if p < 1:
                loo = p * (z[j] - base) / (1 - p)
                if loo > 0:
                    candidates.append((loo, -int(j), int(j), p))
    if candidates:
        _, _, j, p = max(candidates)
        result["mean"][5] = avg(np.array([k for k in range(n) if k != j]))
        single = (j, p, float(z[j]), float(candidates[0][0]))
        # retrieve the selected member's LOO exactly
        loo = p * (z[j] - base) / (1 - p)
    else:
        single, loo = None, nan
        result["mean"][5] = base

    def variance(ix, weighted):
        if not len(ix) or not np.isfinite(z[ix]).all():
            return nan
        m = avg(ix) if weighted else float(np.mean(z[ix]))
        q = np.sum(w[ix] * (z[ix] - m) ** 2) / np.sum(w[ix]) if weighted else np.mean((z[ix] - m) ** 2)
        return float(q)

    uw, ww, uvars, wvars = [], [], [], []
    for ix in sets:
        uw.append(float(np.std(z[ix])) if len(ix) and np.isfinite(z[ix]).all() else (0. if len(ix) == 1 and np.isfinite(z[ix]).all() else nan))
        wvars.append(variance(ix, True))
        uvars.append(variance(ix, False))
        ww.append(math.sqrt(wvars[-1]) if np.isfinite(wvars[-1]) else nan)
    ratios = [a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else nan for a, b in zip(ww, uw)]
    stats = dict(accepted=n, total_w=total, min_w=float(np.min(w)), max_w=float(np.max(w)),
                 top1_fraction=float(np.max(w) / total), top10_count_mass=float(np.sum(w[top10]) / total),
                 n90_count_fraction=float(c90 / n), weighted_median=med,
                 tail_mass=float(np.sum(w[sets[3]]) / total), tail_mean=means[3],
                 tail_shift=means[3] - base if np.isfinite(means[3]) and np.isfinite(base) else nan,
                 far_high_mass=float(far_mass / total), far_high_count=len(far),
                 far_high_mean=avg(far), has_far_high=int(bool(len(far))),
                 far_high_shift=result["mean"][4] - base if len(far) and np.isfinite(result["mean"][4]) and np.isfinite(base) else nan,
                 far_single_p=single[1] if single else nan, far_single_depth=single[2] if single else nan,
                 far_single_loo=loo, far_single_shift=result["mean"][5] - base if single and np.isfinite(result["mean"][5]) and np.isfinite(base) else nan,
                 has_far_single=int(single is not None))
    for i, name in enumerate(("all", "top10", "top50", "cover90")):
        stats[f"{name}_unweighted_sd"] = uw[i]
        stats[f"{name}_weighted_sd"] = ww[i]
        stats[f"{name}_sd_ratio"] = ratios[i]
    result["stats"] = stats
    result["uniform_variance4"] = uvars
    result["weighted_variance4"] = wvars
    return result


def diagnostics_torch(depth, weight, counts):
    """Compute accepted-ray diagnostics. Inputs are [B,cap], counts is [B].

    Results are tensors on the input device: mean [B,6], sizes [B,4],
    variance arrays [B,4], and a dict of [B] tensors.
    """
    import torch

    if depth.ndim != 2 or weight.shape != depth.shape:
        raise ValueError("depth and weight must have matching [B,cap] shapes")
    counts = torch.as_tensor(counts, device=depth.device, dtype=torch.long)
    if counts.shape != (depth.shape[0],) or torch.any(counts < 0) or torch.any(counts > depth.shape[1]):
        raise ValueError("counts must be [B] and within [0,cap]")
    rows = []
    stat_names = None
    for b in range(depth.shape[0]):
        n = int(counts[b].item())
        d = depth[b, :n]
        w = weight[b, :n]
        if n and (not bool(torch.isfinite(w).all()) or bool(torch.any(w <= 0))):
            raise ValueError("accepted weights must be finite and > 0")
        # Preserve the input precision for the public calculation.
        rows.append(_torch_one(d, w))
        if rows[-1]["stats"]:
            stat_names = rows[-1]["stats"].keys()
    device, dtype = depth.device, depth.dtype
    mean = torch.stack([r["mean"] for r in rows]) if rows else depth.new_empty((0, 6))
    sizes = torch.stack([r["sizes"] for r in rows]) if rows else torch.empty((0, 4), device=device, dtype=torch.long)
    uv = torch.stack([r["uniform_variance4"] for r in rows]) if rows else depth.new_empty((0, 4))
    wv = torch.stack([r["weighted_variance4"] for r in rows]) if rows else depth.new_empty((0, 4))
    zero_keys = {"accepted", "total_w", "min_w", "max_w", "top1_fraction", "top10_count_mass", "n90_count_fraction", "tail_mass", "far_high_mass", "far_high_count", "has_far_high", "has_far_single"}
    stats = {k: torch.stack([r["stats"].get(k, depth.new_tensor(0.) if k in zero_keys else depth.new_tensor(float("nan"))) for r in rows]) for k in _STAT_NAMES}
    return {"mean": mean, "sizes": sizes, "stats": stats,
            "uniform_variance4": uv, "weighted_variance4": wv}


def _torch_one(z, w):
    import torch
    nan = z.new_tensor(float("nan"))
    zero = z.new_tensor(0.)
    n = z.numel()
    if n == 0:
        empty_stats = {k: z.new_tensor(0.) for k in ("accepted", "total_w", "min_w", "max_w", "top1_fraction", "top10_count_mass", "n90_count_fraction", "tail_mass", "far_high_mass", "far_high_count", "has_far_high", "has_far_single")}
        empty_stats.update({k: nan for k in ("weighted_median", "tail_mean", "tail_shift", "far_high_mean", "far_high_shift", "far_single_p", "far_single_depth", "far_single_loo", "far_single_shift")})
        for name in ("all", "top10", "top50", "cover90"):
            empty_stats.update({f"{name}_unweighted_sd": nan, f"{name}_weighted_sd": nan, f"{name}_sd_ratio": nan})
        return {"mean": torch.full((6,), float("nan"), device=z.device, dtype=z.dtype),
                "sizes": torch.zeros(4, device=z.device, dtype=torch.long),
                "uniform_variance4": torch.full((4,), float("nan"), device=z.device, dtype=z.dtype),
                "weighted_variance4": torch.full((4,), float("nan"), device=z.device, dtype=z.dtype), "stats": empty_stats}
    total = w.sum()
    order = torch.argsort(w, descending=True, stable=True)
    c10, c50 = math.ceil(.1*n), math.ceil(.5*n)
    c90 = int(torch.searchsorted(torch.cumsum(w[order], 0), .9 * total, right=False).item()) + 1
    sets = [torch.arange(n, device=z.device), order[:c10], order[:c50], order[:c90]]
    finite = torch.isfinite(z)
    def avg(ix):
        if ix.numel() == 0 or not bool(finite[ix].all()): return nan
        return (w[ix] * z[ix]).sum() / w[ix].sum()
    means = [avg(ix) for ix in sets]
    med = nan
    if bool(finite.all()):
        zo = torch.argsort(z, stable=True)
        k = int(torch.searchsorted(torch.cumsum(w[zo], 0), .5 * total, right=False).item())
        med = z[zo[k]]
    far = sets[1][z[sets[1]] > med] if bool(torch.isfinite(med)) else torch.empty(0, device=z.device, dtype=torch.long)
    mask = torch.ones(n, device=z.device, dtype=torch.bool); mask[far] = False
    remaining = torch.arange(n, device=z.device)[mask]
    nofar = avg(remaining) if far.numel() else means[0]
    best_j, best_loo = None, nan
    if far.numel() and bool(torch.isfinite(means[0])):
        for j in far.tolist():
            p = w[j] / total
            loo = p * (z[j] - means[0]) / (1 - p) if bool(p < 1) else nan
            if bool(torch.isfinite(loo)) and bool(loo > 0) and (best_j is None or bool(loo > best_loo)):
                best_j, best_loo = j, loo
    nosingle = avg(torch.cat((torch.arange(best_j, device=z.device), torch.arange(best_j+1, n, device=z.device)))) if best_j is not None else means[0]
    variances, uniform = [], []
    for ix in sets:
        if ix.numel() == 0 or not bool(finite[ix].all()):
            variances.append(nan); uniform.append(nan); continue
        if ix.numel() == 1:
            variances.append(zero); uniform.append(zero); continue
        m = avg(ix); um = z[ix].mean()
        variances.append((w[ix] * (z[ix]-m).square()).sum()/w[ix].sum())
        uniform.append((z[ix]-um).square().mean())
    u_sd = torch.stack([torch.sqrt(v) if bool(torch.isfinite(v)) else nan for v in uniform])
    w_sd = torch.stack([torch.sqrt(v) if bool(torch.isfinite(v)) else nan for v in variances])
    ratio = torch.where((u_sd != 0) & torch.isfinite(u_sd) & torch.isfinite(w_sd), w_sd/u_sd, nan)
    far_mass = w[far].sum() if far.numel() else zero
    values = dict(accepted=z.new_tensor(float(n)), total_w=total, min_w=w.min(), max_w=w.max(),
        top1_fraction=w.max()/total, top10_count_mass=w[sets[1]].sum()/total,
        n90_count_fraction=z.new_tensor(c90/n), weighted_median=med,
        tail_mass=w[sets[3]].sum()/total, tail_mean=means[3],
        tail_shift=means[3]-means[0], far_high_mass=far_mass/total,
        far_high_count=z.new_tensor(float(far.numel())), far_high_mean=avg(far),
        has_far_high=z.new_tensor(float(far.numel()>0)),
        far_high_shift=nofar-means[0] if far.numel() else nan,
        far_single_p=w[best_j]/total if best_j is not None else nan,
        far_single_depth=z[best_j] if best_j is not None else nan, far_single_loo=best_loo,
        far_single_shift=nosingle-means[0] if best_j is not None else nan,
        has_far_single=z.new_tensor(float(best_j is not None)))
    for i, name in enumerate(("all", "top10", "top50", "cover90")):
        values[f"{name}_unweighted_sd"] = u_sd[i]
        values[f"{name}_weighted_sd"] = w_sd[i]
        values[f"{name}_sd_ratio"] = ratio[i]
    return {"mean": torch.stack(means + [nofar, nosingle]),
        "sizes": torch.tensor([len(x) for x in sets], device=z.device, dtype=torch.long),
        "uniform_variance4": torch.stack(uniform), "weighted_variance4": torch.stack(variances), "stats": values}


def _check():
    import torch
    z = np.array([[1., 2., 2., 8., 0.], [3., 3., 3., 0., 0.],
                  [9., 0., 0., 0., 0.], [np.nan, 1., 0., 0., 0.],
                  [1., 100., 101., 0., 0.], [0., 0., 0., 0., 0.]])
    w = np.array([[.1, .2, .2, .4, .1], [.5, .25, .25, 0, 0],
                  [1., 0, 0, 0, 0], [.5, .5, 0, 0, 0],
                  [.01, .98, .01, 0, 0], [0, 0, 0, 0, 0]])
    counts = np.array([5, 3, 1, 2, 3, 0])
    ref = _numpy_reference(z, w, counts)
    out = diagnostics_torch(torch.tensor(z, dtype=torch.float64), torch.tensor(w, dtype=torch.float64), counts)
    for i, r in enumerate(ref):
        assert np.allclose(out["mean"][i].numpy(), r["mean"], equal_nan=True, atol=1e-12, rtol=0)
        assert np.array_equal(out["sizes"][i].numpy(), r["sizes"])
        assert np.allclose(out["uniform_variance4"][i].numpy(), r["uniform_variance4"], equal_nan=True, atol=1e-12, rtol=0)
        assert np.allclose(out["weighted_variance4"][i].numpy(), r["weighted_variance4"], equal_nan=True, atol=1e-12, rtol=0)
        for k, v in r["stats"].items():
            assert np.allclose(out["stats"][k][i].item(), v, equal_nan=True, atol=1e-12, rtol=0), (i, k, v, out["stats"][k][i])
    # Deleting all members is undefined and must remain NaN.
    all_far = diagnostics_torch(torch.tensor([[1., 9.]], dtype=torch.float64), torch.tensor([[.1, .9]], dtype=torch.float64), [2])
    assert torch.isnan(all_far["mean"][0, 4])
    print("ray_influence --check passed")


if __name__ == "__main__" and "--check" in sys.argv:
    _check()
