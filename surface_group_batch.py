"""Chunked Torch implementation of the frozen v1 candidate for renderer support.

The renderer supplies detached native accepted IDs/alpha. Selection is detached;
the final equal-weight plane LS keeps center, rotation, and query gradients.
"""

import math

import torch

from surface_group_reference import rotation_matrix_wxyz


def grouped_depth_from_support(pc, camera, xy, ids, alpha, count, overflow,
                               native_z, chunk_size=256):
    """Return camera-z [Q] and per-query audit masks from renderer support.

    xy is continuous pixel coordinates [Q,2], native_z [Q], ids/alpha [Q,K].
    This never sends per-query tensors to CPU. K must be <=128; overflow falls
    back per query. Camera transform follows row-vector world_view_transform.
    """
    if (xy.ndim != 2 or xy.shape[1] != 2 or ids.ndim != 2 or
            alpha.shape != ids.shape or ids.shape[0] != xy.shape[0] or
            count.shape != (xy.shape[0],) or overflow.shape != count.shape or
            native_z.shape != count.shape or ids.shape[1] > 128 or chunk_size < 1):
        raise ValueError('support shape/capacity mismatch')
    n = pc.get_xyz.shape[0]
    if n == 0:
        return native_z, {'candidate': torch.zeros_like(count, dtype=torch.bool)}
    wvt = camera.world_view_transform
    world_centers = pc.get_xyz
    world_rotations = rotation_matrix_wxyz(pc.get_rotation)
    scales = pc.get_scaling  # Unfiltered, un-AA physical axes.
    centers = world_centers @ wvt[:3, :3] + wvt[3, :3]
    rotations = wvt[:3, :3].T @ world_rotations
    depths, audit = [], {'candidate': [], 'invalid_count': [], 'first_group_size': [],
                        'group_switch': [], 'no_valid': [],
                        'overflow': [], 'low_alpha': [], 'footprint_fail': [],
                        'insufficient_distinct': []}
    for start in range(0, xy.shape[0], chunk_size):
        stop = min(start + chunk_size, xy.shape[0])
        out, rows = _chunk(centers, rotations, scales, xy[start:stop],
                           ids[start:stop], alpha[start:stop], count[start:stop],
                           overflow[start:stop], native_z[start:stop],
                           camera.Fx, camera.Fy, camera.Cx, camera.Cy)
        depths.append(out)
        for key in audit:
            audit[key].append(rows[key])
    return torch.cat(depths), {key: torch.cat(rows) for key, rows in audit.items()}


def _chunk(centers, rotations, scales, xy, ids, alpha, count, overflow,
           native_z, fx, fy, cx, cy):
    q, k = ids.shape
    device = ids.device
    pos = torch.arange(k, device=device)[None, :]
    padding = pos >= count[:, None]
    member = ~padding & (ids >= 0) & (ids < centers.shape[0])
    safe = ids.long().clamp(0, centers.shape[0] - 1)
    m = centers[safe]
    rot = rotations[safe]
    scale = scales[safe]
    v = torch.cat(((xy - xy.new_tensor([cx, cy])) /
                   xy.new_tensor([fx, fy]), torch.ones_like(xy[:, :1])), dim=1)

    with torch.no_grad():
        md, rd, sd, ad, vd = m.detach(), rot.detach(), scale.detach(), alpha.detach(), v.detach()
        sorted_scale, axis = torch.sort(sd, dim=-1)
        ni = torch.gather(rd, 3, axis[:, :, :1, None].expand(-1, -1, 3, -1)).squeeze(-1)
        inc = (ni * vd[:, None, :]).sum(-1)
        t = (ni * md).sum(-1) / inc
        valid = (member & torch.isfinite(md).all(-1) & torch.isfinite(rd).all(-1).all(-1) &
                 torch.isfinite(sd).all(-1) & (sd > 0).all(-1) &
                 torch.isfinite(ad) & (ad >= 0) & (ad < 1) &
                 (sorted_scale[:, :, 1] / sorted_scale[:, :, 0] > 1.001) &
                 (inc.abs() / torch.linalg.norm(vd, dim=1)[:, None] >= .1) &
                 torch.isfinite(t) & (t > 0))
        invalid_count = (member & ~valid).sum(1)
        # Stable tau sort after stable global-ID sort; padding goes last.
        id_order = torch.argsort(torch.where(valid, ids.long(), torch.iinfo(torch.int64).max),
                                 dim=1, stable=True)
        tau_by_id = torch.gather(torch.where(valid, t, torch.inf), 1, id_order)
        tau_order = torch.argsort(tau_by_id, dim=1, stable=True)
        order = torch.gather(id_order, 1, tau_order)
        gather3 = order[:, :, None].expand(-1, -1, 3)
        gather33 = order[:, :, None, None].expand(-1, -1, 3, 3)
        ms = torch.gather(md, 1, gather3)
        ns = torch.gather(ni, 1, gather3)
        rs = torch.gather(rd, 1, gather33)
        ss = torch.gather(sd, 1, gather3)
        ax = torch.gather(axis[:, :, 0], 1, order)
        ts = torch.gather(t, 1, order)
        al = torch.gather(ad, 1, order)
        vs = torch.gather(valid, 1, order)
        nv = vs.sum(1)
        normal_dot = torch.bmm(ns, ns.transpose(1, 2)).abs()
        angle_ok = normal_dot >= math.cos(math.radians(10.))
        proj = torch.bmm(ns, ms.transpose(1, 2))
        diag = torch.diagonal(proj, dim1=1, dim2=2)
        gap = torch.maximum((proj - diag[:, :, None]).abs(),
                            (proj.transpose(1, 2) - diag[:, None, :]).abs())
        delta = ts * max(1/fx, 1/fy)
        pair_ok = angle_ok & (gap <= delta[:, :, None] + delta[:, None, :])
        i = torch.arange(k, device=device)
        previous = i[:, None] < i[None, :]
        prefix_ok = (pair_ok | ~previous[None, :, :]).all(dim=1)
        break_at = torch.where((~prefix_ok) & (i[None, :] < nv[:, None]),
                               i[None, :], k).min(dim=1).values
        size = torch.minimum(break_at, nv)
        first = (i[None, :] < size[:, None])
        group_alpha = 1 - torch.prod(torch.where(first, 1 - al, 1.), dim=1)
        c0 = (ns * vd[:, None, :]).sum(-1)
        b0 = (ns * ms).sum(-1)
        denom0 = torch.where(first, c0*c0, 0.).sum(1)
        z0 = torch.where(first, c0*b0, 0.).sum(1) / denom0
        point = z0[:, None] * vd
        local = torch.matmul(rs.transpose(-1, -2), (point[:, None, :] - ms)[..., None]).squeeze(-1)
        q_axes = (local / ss).square()
        q_axes.scatter_(2, ax[:, :, None], 0.)
        footprint = q_axes.sum(-1)
        footprint_fail = (first & (~torch.isfinite(footprint) | (footprint > 4.))).any(1)
        cov = torch.matmul(rs * (ss*ss)[:, :, None, :], rs.transpose(-1, -2))
        cov = (cov + cov.transpose(-1, -2)) / 2
        distinct = first & ((ms != ms[:, :1]).any(-1) |
                            (cov != cov[:, :1]).any(-1).any(-1))
        insufficient = ~distinct.any(1)
        low_alpha = group_alpha < .5
        good = (~overflow.bool() & (count <= k) & (size >= 2) & ~insufficient &
                ~low_alpha & ~footprint_fail & torch.isfinite(denom0) &
                (denom0 > 0) & torch.isfinite(z0) & (z0 > 0))
    live_m = torch.gather(m, 1, gather3)
    live_n = torch.gather(torch.gather(rot, 1, gather33), 3,
                          ax[:, :, None, None].expand(-1, -1, 3, 1)).squeeze(-1)
    # Sanitize inactive entries before products, so a NaN in an excluded member
    # cannot leak through a zero upstream gradient during autograd.
    live_m = torch.where(first[:, :, None], live_m, 0.)
    live_n = torch.where(first[:, :, None], live_n, 0.)
    live_c = (live_n * v[:, None, :]).sum(-1)
    live_b = (live_n * live_m).sum(-1)
    candidate = torch.where(first, live_c * live_b, 0.).sum(1) / torch.where(
        first, live_c * live_c, 0.).sum(1).clamp_min(torch.finfo(native_z.dtype).tiny)
    output = torch.where(good, candidate, native_z)
    rows = dict(candidate=good, invalid_count=invalid_count, first_group_size=size,
                group_switch=size < nv, no_valid=nv == 0,
                overflow=overflow, low_alpha=low_alpha, footprint_fail=footprint_fail,
                insufficient_distinct=insufficient)
    return output, rows
