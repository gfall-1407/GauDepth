"""Chunked Torch implementation of the frozen v1 candidate for renderer support.

The renderer supplies detached native accepted IDs/alpha. Selection is detached;
the final equal-weight plane LS keeps center, rotation, and query gradients.
"""

import math

import torch

from surface_group_reference import rotation_matrix_wxyz


def native_sample_xy(points_world, camera):
    """Match sample_forward.cu preprocessPointsCUDA's full-projective pixels.

    transformPoint4x4 uses the stored transposed full_proj_transform; native
    divides by homogeneous w+1e-7 before ndc2Pix(v,S)=((v+1)S-1)/2.
    """
    matrix = camera.full_proj_transform
    finite_points = torch.isfinite(points_world).all(-1)
    safe_points = torch.where(torch.isfinite(points_world), points_world, 0.)
    hom = safe_points @ matrix[:3, :] + matrix[3, :]
    denominator = hom[..., 3:4] + 1e-7
    valid = (finite_points & torch.isfinite(hom).all(-1) &
             torch.isfinite(denominator[..., 0]) & (denominator[..., 0] != 0))
    safe_denominator = torch.where(valid[..., None], denominator, 1.)
    ndc_xy = hom[..., :2] / safe_denominator
    size = ndc_xy.new_tensor([camera.image_width, camera.image_height])
    pixels = ((ndc_xy + 1.) * size - 1.) * .5
    return torch.where(valid[..., None], pixels, torch.full_like(pixels, float('nan')))


def compose_sample_points(candidate_z, query_xy, candidate_mask, native_points, camera):
    """Construct continuous candidate camera points without contaminating fallback."""
    if (candidate_z.ndim != 1 or query_xy.shape != (candidate_z.numel(), 2) or
            candidate_mask.shape != candidate_z.shape or
            native_points.numel() != candidate_z.numel() * 3):
        raise ValueError('sample candidate shape mismatch')
    choose = candidate_mask.bool() & torch.isfinite(query_xy).all(-1) & torch.isfinite(candidate_z)
    safe_xy = torch.where(choose[:, None], query_xy, 0.)
    safe_z = torch.where(choose, candidate_z, 1.)
    ray = torch.cat(((safe_xy - safe_xy.new_tensor([camera.Cx, camera.Cy])) /
                     safe_xy.new_tensor([camera.Fx, camera.Fy]),
                     torch.ones_like(safe_xy[:, :1])), dim=1)
    point = (safe_z[:, None] * ray).reshape_as(native_points)
    return torch.where(choose.reshape(native_points.shape[:-1])[..., None], point, native_points)


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
    empty_audit = dict(candidate=torch.zeros_like(count, dtype=torch.bool),
                       invalid_count=torch.zeros_like(count),
                       accepted_count=count, valid_count=torch.zeros_like(count),
                       invalid_basic=torch.zeros_like(count),
                       invalid_short_axis=torch.zeros_like(count),
                       invalid_grazing=torch.zeros_like(count),
                       invalid_tau=torch.zeros_like(count),
                       first_group_size=torch.zeros_like(count),
                       first_group_alpha=torch.zeros_like(count, dtype=alpha.dtype),
                       group_switch=torch.zeros_like(count, dtype=torch.bool),
                       boundary_angle_fail=torch.zeros_like(count, dtype=torch.bool),
                       boundary_distance_fail=torch.zeros_like(count, dtype=torch.bool),
                       no_valid=torch.ones_like(count, dtype=torch.bool),
                       overflow=overflow.bool(),
                       low_alpha=torch.zeros_like(count, dtype=torch.bool),
                       footprint_fail=torch.zeros_like(count, dtype=torch.bool),
                       insufficient_distinct=torch.zeros_like(count, dtype=torch.bool))
    n = pc.get_xyz.shape[0]
    if n == 0 or xy.shape[0] == 0:
        return native_z, empty_audit
    wvt = camera.world_view_transform
    # GaussianModel.get_rotation normalizes _rotation before returning. If a
    # latent quaternion is nonfinite, masking only that normalized result can
    # still backpropagate 0*NaN through normalization. Sanitize the latent first.
    raw_centers = pc.get_xyz
    raw_quaternions = pc._rotation if hasattr(pc, '_rotation') else pc.get_rotation
    raw_scales = pc.get_scaling
    quaternion_norm = torch.linalg.norm(torch.where(torch.isfinite(raw_quaternions),
                                                     raw_quaternions, 0.), dim=-1)
    primitive_valid = (torch.isfinite(raw_centers).all(-1) &
                       torch.isfinite(raw_quaternions).all(-1) &
                       torch.isfinite(quaternion_norm) & (quaternion_norm > 0) &
                       torch.isfinite(raw_scales).all(-1) & (raw_scales > 0).all(-1))
    world_centers = torch.where(torch.isfinite(raw_centers), raw_centers, 0.)
    identity_q = raw_quaternions.new_tensor([1., 0., 0., 0.])
    safe_quaternions = torch.where(primitive_valid[:, None], raw_quaternions, identity_q)
    world_rotations = rotation_matrix_wxyz(safe_quaternions)
    scales = torch.where(primitive_valid[:, None], raw_scales, 1.)  # Unfiltered, un-AA.
    centers = world_centers @ wvt[:3, :3] + wvt[3, :3]
    rotations = wvt[:3, :3].T @ world_rotations
    depths, audit = [], {key: [] for key in empty_audit}
    for start in range(0, xy.shape[0], chunk_size):
        stop = min(start + chunk_size, xy.shape[0])
        out, rows = _chunk(centers, rotations, scales, xy[start:stop],
                           ids[start:stop], alpha[start:stop], count[start:stop],
                           overflow[start:stop], native_z[start:stop],
                           camera.Fx, camera.Fy, camera.Cx, camera.Cy, primitive_valid)
        depths.append(out)
        for key in audit:
            audit[key].append(rows[key])
    return torch.cat(depths), {key: torch.cat(rows) for key, rows in audit.items()}


def _chunk(centers, rotations, scales, xy, ids, alpha, count, overflow,
           native_z, fx, fy, cx, cy, primitive_valid):
    q, k = ids.shape
    device = ids.device
    pos = torch.arange(k, device=device)[None, :]
    padding = pos >= count[:, None]
    member = ~padding & (ids >= 0) & (ids < centers.shape[0])
    safe = ids.long().clamp(0, centers.shape[0] - 1)
    m = centers[safe]
    rot = rotations[safe]
    scale = scales[safe]
    query_valid = torch.isfinite(xy).all(-1)
    safe_xy = torch.where(torch.isfinite(xy), xy, 0.)
    v = torch.cat(((safe_xy - xy.new_tensor([cx, cy])) /
                   xy.new_tensor([fx, fy]), torch.ones_like(xy[:, :1])), dim=1)

    with torch.no_grad():
        md, rd, sd, ad, vd = m.detach(), rot.detach(), scale.detach(), alpha.detach(), v.detach()
        sorted_scale, axis = torch.sort(sd, dim=-1)
        ni = torch.gather(rd, 3, axis[:, :, :1, None].expand(-1, -1, 3, -1)).squeeze(-1)
        inc = (ni * vd[:, None, :]).sum(-1)
        t = (ni * md).sum(-1) / inc
        basic = (member & primitive_valid[safe] & query_valid[:, None] &
                 torch.isfinite(md).all(-1) & torch.isfinite(rd).all(-1).all(-1) &
                 torch.isfinite(sd).all(-1) & (sd > 0).all(-1) &
                 torch.isfinite(ad) & (ad >= 0) & (ad < 1))
        short_ok = sorted_scale[:, :, 1] / sorted_scale[:, :, 0] > 1.001
        incidence_ok = inc.abs() / torch.linalg.norm(vd, dim=1)[:, None] >= .1
        tau_ok = torch.isfinite(t) & (t > 0)
        valid = basic & short_ok & incidence_ok & tau_ok
        invalid_count = (member & ~valid).sum(1)
        invalid_basic = (member & ~basic).sum(1)
        invalid_short_axis = (basic & ~short_ok).sum(1)
        invalid_grazing = (basic & short_ok & ~incidence_ok).sum(1)
        invalid_tau = (basic & short_ok & incidence_ok & ~tau_ok).sum(1)
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
        distance_ok = gap <= delta[:, :, None] + delta[:, None, :]
        pair_ok = angle_ok & distance_ok
        i = torch.arange(k, device=device)
        previous = i[:, None] < i[None, :]
        prefix_ok = (pair_ok | ~previous[None, :, :]).all(dim=1)
        break_at = torch.where((~prefix_ok) & (i[None, :] < nv[:, None]),
                               i[None, :], k).min(dim=1).values
        size = torch.minimum(break_at, nv)
        boundary = i[None, None, :] == break_at[:, None, None]
        boundary_angle_fail = ((~angle_ok) & previous[None, :, :] & boundary).any(1).any(1)
        boundary_distance_fail = ((~distance_ok) & previous[None, :, :] & boundary).any(1).any(1)
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
                first_group_alpha=group_alpha,
                accepted_count=count, valid_count=nv,
                invalid_basic=invalid_basic, invalid_short_axis=invalid_short_axis,
                invalid_grazing=invalid_grazing, invalid_tau=invalid_tau,
                group_switch=size < nv, no_valid=nv == 0,
                boundary_angle_fail=boundary_angle_fail,
                boundary_distance_fail=boundary_distance_fail,
                overflow=overflow, low_alpha=low_alpha, footprint_fail=footprint_fail,
                insufficient_distinct=insufficient)
    return output, rows
