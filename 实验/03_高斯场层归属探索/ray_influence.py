"""Vectorized diagnostics on already accepted ray contributions."""
from __future__ import annotations

import math
import sys

import numpy as np

_SD_NAMES = ("all", "top10", "top50", "cover90")
_STAT_NAMES = ("accepted", "total_w", "min_w", "max_w", "top1_fraction",
    "top10_count_mass", "n90_count_fraction", "weighted_median", "tail_mass",
    "tail_mean", "tail_shift", "far_high_mass", "far_high_count", "far_high_mean",
    "has_far_high", "far_high_shift", "far_single_p", "far_single_depth",
    "far_single_loo", "has_far_single", "far_single_shift") + tuple(
    f"{name}_{suffix}" for name in _SD_NAMES
    for suffix in ("unweighted_sd", "weighted_sd", "sd_ratio",
                   "unweighted_sub_all", "weighted_sub_all"))


def _numpy_reference(depth, weight, counts):
    """Independent scalar per-ray reference, used only by --check."""
    return [_one_ray(depth[b, :int(n)].astype(np.float64),
                     weight[b, :int(n)].astype(np.float64))
            for b, n in enumerate(counts)]


def _one_ray(z, w):
    """Readable NumPy oracle for a single ray."""
    n = len(z)
    nan = float("nan")
    if not n:
        return dict(mean=[nan] * 6, sizes=[0] * 4,
                    uniform_variance4=[nan] * 4, weighted_variance4=[nan] * 4,
                    stats={k: (0. if k in {"accepted", "total_w", "min_w", "max_w",
                          "top1_fraction", "top10_count_mass", "n90_count_fraction",
                          "tail_mass", "far_high_mass", "far_high_count",
                          "has_far_high", "has_far_single"} else nan)
                           for k in _STAT_NAMES})
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("accepted weights must be finite and > 0")
    total = math.fsum(w.tolist())
    order = np.argsort(-w, kind="stable")
    k10, k50 = math.ceil(.1*n), math.ceil(.5*n)
    k90 = int(np.searchsorted(np.cumsum(w[order]), (.9-1e-12)*total, side="left") + 1)
    sets = [np.arange(n), order[:k10], order[:k50], order[:k90]]
    def avg(ix):
        if len(ix) == 0 or not np.isfinite(z[ix]).all(): return nan
        return math.fsum((w[ix]*z[ix]).tolist()) / math.fsum(w[ix].tolist())
    means = [avg(ix) for ix in sets]
    median = nan
    if np.isfinite(z).all():
        zi = np.argsort(z, kind="stable")
        median = float(z[zi[np.searchsorted(np.cumsum(w[zi]), .5*total, side="left")]])
    far = sets[1][z[sets[1]] > median] if np.isfinite(median) else np.array([], dtype=int)
    remain = np.setdiff1d(np.arange(n), far, assume_unique=True)
    nofar = avg(remain) if len(far) else means[0]
    tail = np.setdiff1d(np.arange(n), sets[3], assume_unique=True)
    candidates = []
    for j in sorted(far.tolist()):  # stable tie resolution follows acceptance order
        p = w[j]/total
        loo = p*(z[j]-means[0])/(1-p) if p < 1 and np.isfinite(means[0]) else nan
        if np.isfinite(loo) and loo > 0: candidates.append((loo, int(j), p))
    if candidates:
        max_loo = max(x[0] for x in candidates)
        loo, j, p = next(x for x in candidates if x[0] == max_loo)
        nosingle = avg(np.delete(np.arange(n), j))
    else:
        j, p, loo, nosingle = None, nan, nan, means[0]
    uvar, wvar, usd, wsd = [], [], [], []
    for ix in sets:
        if not len(ix) or not np.isfinite(z[ix]).all():
            uv = wv = nan
        elif len(ix) == 1:
            uv = wv = 0.
        else:
            mu, uwmu = avg(ix), float(np.mean(z[ix]))
            uv = float(np.mean((z[ix]-uwmu)**2))
            wv = float(np.sum(w[ix]*(z[ix]-mu)**2)/np.sum(w[ix]))
        uvar.append(uv); wvar.append(wv)
        usd.append(math.sqrt(uv) if np.isfinite(uv) else nan)
        wsd.append(math.sqrt(wv) if np.isfinite(wv) else nan)
    stats = dict(accepted=n, total_w=total, min_w=float(w.min()), max_w=float(w.max()),
        top1_fraction=float(w.max()/total), top10_count_mass=float(w[sets[1]].sum()/total),
        n90_count_fraction=k90/n, weighted_median=median,
        tail_mass=float(w[tail].sum()/total), tail_mean=avg(tail),
        tail_shift=means[3]-means[0],
        far_high_mass=float(w[far].sum()/total), far_high_count=len(far),
        far_high_mean=avg(far), has_far_high=int(bool(len(far))),
        far_high_shift=(nofar-means[0]) if len(far) and np.isfinite(nofar) and np.isfinite(means[0]) else (0. if not len(far) and np.isfinite(means[0]) else nan),
        far_single_p=p, far_single_depth=float(z[j]) if j is not None else nan,
        far_single_loo=loo, has_far_single=int(j is not None),
        far_single_shift=(nosingle-means[0]) if j is not None and np.isfinite(nosingle) and np.isfinite(means[0]) else (0. if j is None and np.isfinite(means[0]) else nan))
    for i, name in enumerate(_SD_NAMES):
        stats[f"{name}_unweighted_sd"], stats[f"{name}_weighted_sd"] = usd[i], wsd[i]
        stats[f"{name}_sd_ratio"] = wsd[i]/usd[i] if np.isfinite(usd[i]) and usd[i] != 0 and np.isfinite(wsd[i]) else nan
        stats[f"{name}_unweighted_sub_all"] = usd[i]/usd[0] if np.isfinite(usd[i]) and np.isfinite(usd[0]) and usd[0] != 0 else nan
        stats[f"{name}_weighted_sub_all"] = wsd[i]/wsd[0] if np.isfinite(wsd[i]) and np.isfinite(wsd[0]) and wsd[0] != 0 else nan
    return dict(mean=means+[nofar,nosingle], sizes=[len(x) for x in sets],
                uniform_variance4=uvar, weighted_variance4=wvar, stats=stats)


def diagnostics_torch(depth, weight, counts):
    """Vectorized [B,cap] accepted ray diagnostics; calculations use float64."""
    import torch
    if depth.ndim != 2 or weight.shape != depth.shape:
        raise ValueError("depth and weight must have matching [B,cap] shapes")
    device = depth.device
    z, w = depth.to(torch.float64), weight.to(torch.float64)
    counts = torch.as_tensor(counts, device=device, dtype=torch.long)
    if counts.shape != (depth.shape[0],): raise ValueError("counts must have shape [B]")
    cap = depth.shape[1]
    pos = torch.arange(cap, device=device)[None, :]
    valid = pos < counts[:, None]
    # One batch-level synchronization only for input validation.
    if bool(((~torch.isfinite(w) | (w <= 0)) & valid).any()):
        raise ValueError("accepted weights must be finite and > 0")
    B = depth.shape[0]
    nan = torch.full((B,), float("nan"), device=device, dtype=torch.float64)
    zero = torch.zeros((B,), device=device, dtype=torch.float64)
    countf = counts.to(torch.float64)
    ww = torch.where(valid, w, torch.zeros_like(w))
    total = ww.sum(1)
    order = torch.argsort(torch.where(valid, w, -torch.inf), dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(cap, device=device).expand(B, cap))
    k10 = torch.ceil(.1*countf).long(); k50 = torch.ceil(.5*countf).long()
    sorted_w = torch.gather(ww, 1, order)
    cumw = sorted_w.cumsum(1)
    k90 = ((cumw < ((.9-1e-12)*total[:,None])).sum(1)+1).clamp(min=0)
    # Empty rows have zero subsets; nonempty cover sets use threshold crossing.
    k90 = torch.where(counts > 0, k90.clamp(max=counts), torch.zeros_like(k90))
    subsets = torch.stack((valid, valid & (rank < k10[:,None]),
                           valid & (rank < k50[:,None]), valid & (rank < k90[:,None])), dim=1)
    sizes = subsets.sum(2)
    finite_z = torch.isfinite(z) | ~valid
    def weighted_mean(mask):
        sw = torch.where(mask, ww, 0.).sum(1)
        num = torch.where(mask, ww*z, 0.).sum(1)
        ok = mask.sum(1)>0
        finite_selected = (~(mask & ~torch.isfinite(z))).all(1)
        return torch.where(ok & (sw>0) & finite_selected, num/sw.clamp_min(torch.finfo(z.dtype).tiny), nan)
    means4 = torch.stack([weighted_mean(subsets[:,i]) for i in range(4)], 1)
    # Stable weighted median by stable depth ordering; nonfinite accepted depth
    # leaves the median undefined while preserving all other set NaN semantics.
    zo = torch.argsort(torch.where(valid, z, torch.inf), dim=1, stable=True)
    zw = torch.gather(ww, 1, zo)
    zcum = zw.cumsum(1)
    medidx = ((zcum < (.5*total[:,None])) & valid).sum(1).clamp(max=max(cap-1,0))
    med = torch.gather(torch.gather(z,1,zo),1,medidx[:,None]).squeeze(1) if cap else nan
    med = torch.where((counts>0)&finite_z.all(1), med, nan)
    top10 = subsets[:,1]
    far = top10 & (z > med[:,None]) & valid
    far_count = far.sum(1)
    far_mass = torch.where(far, ww, 0.).sum(1)
    far_mean = weighted_mean(far)
    keepfar = valid & ~far
    nofar = weighted_mean(keepfar)
    nofar = torch.where(far_count==0, means4[:,0], nofar)
    farflag = far_count>0
    farshift = nofar-means4[:,0]
    # Per-member LOO influence. argmax chooses first acceptance index on ties.
    p = ww / total[:,None].clamp_min(torch.finfo(z.dtype).tiny)
    loo = p*(z-means4[:,0,None])/(1-p).clamp_min(torch.finfo(z.dtype).tiny)
    eligible = far & (loo>0) & torch.isfinite(loo) & (p<1)
    maxloo = torch.where(eligible, loo, -torch.inf).max(1).values
    has_single = torch.isfinite(maxloo)
    winner = eligible & (loo == maxloo[:,None])
    # winner is unique because max selects same value ties; argmax over bool is first index.
    winner_idx = winner.to(torch.int64).argmax(1)
    onehot = torch.arange(cap, device=device)[None,:] == winner_idx[:,None]
    singlep = (ww*onehot).sum(1)/total.clamp_min(torch.finfo(z.dtype).tiny)
    singlez = (torch.where(onehot,z,0.)).sum(1)
    keepsingle = valid & ~onehot & has_single[:,None]
    singlesumw = torch.where(keepsingle,ww,0.).sum(1)
    singlesumz = torch.where(keepsingle,ww*z,0.).sum(1)
    nosingle = torch.where(has_single & (singlesumw>0), singlesumz/singlesumw.clamp_min(torch.finfo(z.dtype).tiny), means4[:,0])
    singlep = torch.where(has_single,singlep,nan); singlez = torch.where(has_single,singlez,nan)
    singleloo = torch.where(has_single,maxloo,nan)
    singleshift = nosingle-means4[:,0]
    # Tail is the complement of the smallest prefix covering 90% contribution.
    tail = valid & ~subsets[:,3]
    tail_mass = torch.where(tail,ww,0.).sum(1)
    tail_mean = weighted_mean(tail)
    tail_shift = means4[:,3]-means4[:,0]
    far_high_mass = far_mass/total.clamp_min(torch.finfo(z.dtype).tiny)
    top1mass = torch.where(valid,w,-torch.inf).max(1).values
    top1mass = torch.where(counts>0, top1mass/total.clamp_min(torch.finfo(z.dtype).tiny), zero)
    stat = dict(accepted=countf, total_w=total, min_w=torch.where(valid,w,torch.inf).min(1).values,
        max_w=torch.where(valid,w,-torch.inf).max(1).values, top1_fraction=top1mass,
        top10_count_mass=torch.where(subsets[:,1],ww,0.).sum(1)/total.clamp_min(torch.finfo(z.dtype).tiny),
        n90_count_fraction=k90.to(torch.float64)/countf.clamp_min(1), weighted_median=med,
        tail_mass=tail_mass/total.clamp_min(torch.finfo(z.dtype).tiny), tail_mean=tail_mean,
        tail_shift=tail_shift, far_high_mass=far_high_mass, far_high_count=far_count.to(torch.float64),
        far_high_mean=far_mean, has_far_high=farflag.to(torch.float64), far_high_shift=farshift,
        far_single_p=singlep, far_single_depth=singlez, far_single_loo=singleloo,
        has_far_single=has_single.to(torch.float64), far_single_shift=singleshift)
    stat["min_w"] = torch.where(counts>0, stat["min_w"], zero)
    stat["max_w"] = torch.where(counts>0, stat["max_w"], zero)
    var_u, var_w = [], []
    for i,name in enumerate(_SD_NAMES):
        mask = subsets[:,i]
        nset = sizes[:,i]
        muw = means4[:,i]
        sw = torch.where(mask,ww,0.).sum(1)
        muu = torch.where(mask,z,0.).sum(1)/nset.clamp_min(1)
        du = torch.where(mask,z-muu[:,None],0.)
        dw = torch.where(mask,z-muw[:,None],0.)
        vu = (du.square().sum(1)/nset.clamp_min(1))
        vw = torch.where(mask,ww*dw.square(),0.).sum(1)/sw.clamp_min(torch.finfo(z.dtype).tiny)
        ok = (nset>0) & (~(mask & ~torch.isfinite(z))).all(1)
        vu = torch.where(ok,vu,nan); vw = torch.where(ok,vw,nan)
        vu = torch.where(ok & (nset==1),zero,vu); vw = torch.where(ok & (nset==1),zero,vw)
        var_u.append(vu); var_w.append(vw)
        su, swd = torch.sqrt(vu), torch.sqrt(vw)
        stat[f"{name}_unweighted_sd"] = su; stat[f"{name}_weighted_sd"] = swd
        stat[f"{name}_sd_ratio"] = torch.where((su!=0)&torch.isfinite(su)&torch.isfinite(swd),swd/su,nan)
        stat[f"{name}_unweighted_sub_all"] = torch.where((su==su)&(torch.sqrt(var_u[0])!=0),su/torch.sqrt(var_u[0]),nan)
        stat[f"{name}_weighted_sub_all"] = torch.where((swd==swd)&(torch.sqrt(var_w[0])!=0),swd/torch.sqrt(var_w[0]),nan)
    # Empty rows use explicit empty-set conventions.
    empty = counts==0
    for key in ("weighted_median","tail_mean","far_high_mean","far_single_p","far_single_depth","far_single_loo","far_single_shift"):
        stat[key] = torch.where(empty,nan,stat[key])
    stat["tail_shift"] = torch.where(empty,nan,stat["tail_shift"])
    stat["far_high_shift"] = torch.where(empty,nan,stat["far_high_shift"])
    means4 = torch.where(empty[:,None],nan[:,None],means4)
    nofar = torch.where(empty,nan,nofar)
    meansout = torch.cat((means4,nofar[:,None],nosingle[:,None]),1)
    meansout[:,5] = torch.where(empty,nan,meansout[:,5])
    return {"mean":meansout,"sizes":sizes,"stats":stat,
            "uniform_variance4":torch.stack(var_u,1),"weighted_variance4":torch.stack(var_w,1)}


def _check():
    import torch
    z = np.array([[1.,2.,2.,8.,0.],[3.,3.,3.,0.,0.],[9.,0.,0.,0.,0.],
                  [np.nan,1.,0.,0.,0.],[100.,1.,1.,1.,1.],[0.,0.,0.,0.,0.]])
    w = np.array([[.1,.2,.2,.4,.1],[.5,.25,.25,0,0],[1.,0,0,0,0],
                  [.5,.5,0,0,0],[.2,.2,.2,.2,.2],[0,0,0,0,0]])
    counts=np.array([5,3,1,2,5,0])
    ref=_numpy_reference(z,w,counts)
    out=diagnostics_torch(torch.tensor(z,dtype=torch.float64),torch.tensor(w,dtype=torch.float64),counts)
    for i,r in enumerate(ref):
        for key, expected in (("mean",r["mean"]),("sizes",r["sizes"]),
            ("uniform_variance4",r["uniform_variance4"]),("weighted_variance4",r["weighted_variance4"])):
            assert np.allclose(out[key][i].cpu().numpy(),expected,equal_nan=True,atol=1e-12,rtol=0),(i,key)
        for k,v in r["stats"].items():
            assert np.allclose(out["stats"][k][i].item(),v,equal_nan=True,atol=1e-12,rtol=0),(i,k,v,out["stats"][k][i])
    nofar=diagnostics_torch(torch.tensor([[1.,9.]],dtype=torch.float64),torch.tensor([[.1,.9]],dtype=torch.float64),[2])
    assert nofar["stats"]["has_far_high"].item()==0 and nofar["stats"]["far_high_shift"].item()==0
    print("ray_influence --check passed")


if __name__=="__main__" and "--check" in sys.argv:
    _check()
