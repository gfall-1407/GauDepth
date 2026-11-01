"""Frozen local_plane_training_cpu_v1: CPU geometry reference, no training."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch


CAPTURE_SHA = "74fbd05c3a2c10549adca00f885cdb39c525d003ce120d1a3d593c7feb78d6ef"
ANCHOR_OFFSETS = np.array([[0., 0.], [-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
POLE = 1e-10


class InvalidWindow(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise InvalidWindow("invalid_quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rotmat_torch(q):
    w, x, y, z = (q / torch.linalg.vector_norm(q)).unbind()
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))).reshape(3, 3)


def member_plane(mu_world, quat, axis, Rref, tref):
    mu = Rref @ mu_world + tref
    n = Rref @ rotmat(quat)[:, axis]
    d = float(n @ mu)
    if not np.isfinite(mu).all() or not np.isfinite(n).all() or abs(d) <= POLE*max(1., np.linalg.norm(mu)):
        raise InvalidWindow("member_plane_pole")
    return n / d, d


def member_plane_torch(mu_world, quat, axis, Rref, tref):
    mu = Rref @ mu_world + tref
    n = Rref @ rotmat_torch(quat)[:, axis]
    return n / (n @ mu)


def geometry(Kr, Ks, R, t, uv):
    rays = (np.linalg.inv(Kr) @ np.c_[uv, np.ones(len(uv))].T).T
    base = (Ks @ R @ rays.T).T
    trans = Ks @ t
    return rays, base, trans


def project(pi, geom):
    rays, base, trans = geom
    v = base + np.outer(rays @ pi, trans)
    norms = np.linalg.norm(v, axis=1)
    if not np.isfinite(v).all() or np.any(np.abs(v[:, 2]) <= POLE*np.maximum(1., norms)):
        raise InvalidWindow("homography_pole")
    return v[:, :2] / v[:, 2, None], v[:, 2], v


def project_torch(pi, geom):
    rays, base, trans = (torch.as_tensor(g, dtype=pi.dtype) for g in geom)
    v = base + (rays @ pi)[:, None]*trans[None, :]
    return v[:, :2] / v[:, 2, None]


def jacobian(pi, geom):
    rays, _, trans = geom
    pix, den, _ = project(pi, geom)
    return ((trans[:2][None, :] - pix*trans[2]) / den[:, None])[:, :, None] * rays[:, None, :]


def dists(pix, latent, geom):
    return np.mean(np.sum((pix-project(latent, geom)[0][None, :, :])**2, axis=2), axis=1)


def merge_identical(planes, mass):
    """Geometric sort and exact-plane merge, independent of ID/row order."""
    entries = sorted(((tuple(np.where(p == 0, 0., p)), float(m)) for p, m in zip(planes, mass)), key=lambda x: x[0])
    keys, groups = [], []
    for key, m in entries:
        if not keys or key != keys[-1]:
            keys.append(key)
            groups.append([])
        groups[-1].append(m)
    return np.array(keys, dtype=np.float64), np.array([math.fsum(g) for g in groups])


def objective(pix, q, latent, assignment, geom):
    return math.fsum(float(q[i]*dists(pix[i:i+1], latent[int(assignment[i])], geom)[0]) for i in range(len(q)))


def assign(pix, latent, geom):
    order = sorted(range(len(latent)), key=lambda j: tuple(latent[j]))
    ds = np.stack([dists(pix, latent[j], geom) for j in order], axis=1)
    return np.asarray(order, dtype=int)[np.argmin(ds, axis=1)]


def fit_cluster(pix, q, initial, geom):
    latent = initial.copy()
    stalled = 0
    for _ in range(3):
        proj, old_den, _ = project(latent, geom)
        jac = jacobian(latent, geom)
        residual = proj[None, :, :] - pix
        g = np.einsum("i,ajk,iaj->k", q, jac, residual)/len(proj)
        H = np.einsum("i,ajk,ajl->kl", q, jac, jac)/len(proj)
        diag = np.diag(H)
        A = H + 1e-3*np.diag(diag)
        norms = np.linalg.norm(A, axis=0)
        if not np.isfinite(A).all() or np.any(diag <= 0) or np.any(norms <= 0) or np.linalg.cond(A/norms) > 1e12:
            raise InvalidWindow("gn_singular")
        delta = np.linalg.solve(A, -g)
        old = objective(pix, q, [latent], np.zeros(len(q), dtype=int), geom)
        accepted = False
        for power in range(16):
            trial = latent + (2.**-power)*delta
            try:
                _, den, _ = project(trial, geom)
                if np.any(den*old_den <= 0):
                    continue
                value = objective(pix, q, [trial], np.zeros(len(q), dtype=int), geom)
            except InvalidWindow:
                continue
            if value < old:
                latent = trial
                accepted = True
                break
        if not accepted:
            stalled += 1
    return latent, stalled


def fit(planes, mass, geom):
    positive = mass > 0
    if not np.isfinite(planes[positive]).all() or not np.isfinite(mass).all() or np.any(mass < 0):
        raise InvalidWindow("nonfinite_or_negative_input")
    if not np.any(mass > 0):
        raise InvalidWindow("zero_mass")
    unique, merged = merge_identical(planes[positive], mass[positive])
    q = merged / math.fsum(merged)
    pix = np.stack([project(p, geom)[0] for p in unique])
    init = np.einsum("i,ij->j", q, unique)
    project(init, geom)
    models = []
    for M in (1, 2):
        if M == 1:
            latent = [init.copy()]
        else:
            if len(unique) < 2:
                models.append(dict(M=2, reason="identical_seed"))
                continue
            first = sorted(range(len(unique)), key=lambda i: (-dists(pix[i:i+1], init, geom)[0], tuple(unique[i])))[0]
            second = sorted(range(len(unique)), key=lambda i: (-dists(pix[i:i+1], unique[first], geom)[0], tuple(unique[i])))[0]
            if first == second or np.array_equal(unique[first], unique[second]):
                models.append(dict(M=2, reason="identical_seed"))
                continue
            latent = [unique[first].copy(), unique[second].copy()]
        stalled = 0
        try:
            for _ in range(8):
                a = assign(pix, latent, geom)
                if len(set(a)) != M:
                    raise InvalidWindow("empty_cluster")
                updated = [fit_cluster(pix[a == j], q[a == j], latent[j], geom) for j in range(M)]
                latent = [row[0] for row in updated]
                stalled += sum(row[1] for row in updated)
            a = assign(pix, latent, geom)
            if len(set(a)) != M:
                raise InvalidWindow("empty_cluster_final")
            residual = objective(pix, q, latent, a, geom)
            models.append(dict(M=M, latent=[p.tolist() for p in latent], objective=residual+(M-1),
                               data_residual=residual, reason=None, stalled=stalled))
        except (InvalidWindow, np.linalg.LinAlgError) as exc:
            models.append(dict(M=M, reason=str(exc)))
    finite = [m for m in models if m["reason"] is None]
    if not finite:
        raise InvalidWindow("both_models_failed:"+str([m["reason"] for m in models]))
    best = min(finite, key=lambda m: (m["objective"], m["M"]))
    if len(finite) == 2:
        a, b = finite
        if abs(a["objective"]-b["objective"]) <= 1e-10*max(1., abs(a["objective"]), abs(b["objective"])):
            best = a
    latent = np.array(best["latent"])
    assignment = np.full(len(mass), -1, dtype=int)
    full_dist = np.zeros(len(mass), dtype=np.float64)
    positive_indices = np.flatnonzero(positive)
    assignment[positive] = assign(np.stack([project(p, geom)[0] for p in planes[positive]]), latent, geom)
    for i in positive_indices:
        full_dist[i] = dists(project(planes[i], geom)[0][None, :, :], latent[assignment[i]], geom)[0]
    best = dict(best, assignment=assignment.tolist(), member_D=full_dist.tolist(),
                numerator=math.fsum(float(m*d) for m, d in zip(mass, full_dist)),
                Z=math.fsum(float(m) for m in mass))
    return best, models


def frozen_objective_torch(xyz, quat, axes, q, latent, assignment, Rref, tref, geom):
    R = torch.as_tensor(Rref, dtype=torch.float64)
    t = torch.as_tensor(tref, dtype=torch.float64)
    targets = [torch.as_tensor(project(np.asarray(p), geom)[0], dtype=torch.float64) for p in latent]
    terms = []
    for i in range(len(q)):
        if q[i] == 0:
            continue
        pi = member_plane_torch(xyz[i], quat[i], int(axes[i]), R, t)
        pred = project_torch(pi, geom)
        terms.append(float(q[i])*torch.mean(torch.sum((pred-targets[int(assignment[i])])**2, dim=1)))
    return torch.stack(terms).sum()
