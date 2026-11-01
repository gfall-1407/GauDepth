"""One-query Torch reference for the frozen v1 front-group depth candidate.

The accepted IDs and their native alpha come from the renderer. This reference
does not replace rendering, and discrete selection intentionally has no gradient.
All geometry and returned depth are in camera coordinates (camera z).
"""

import math
import torch


MAX_ACCEPTED = 128
ANGLE_LIMIT = math.radians(10.)
INCIDENCE_MIN = .1
SHORT_RATIO_MIN = 1.001
FOOTPRINT_Q_MAX = 4.
ALPHA_MIN = .5


def rotation_matrix_wxyz(quaternions):
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z+w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1
    ).reshape(-1, 3, 3)


def _same_geometry_signature(center, rotation, scales):
    c = center.detach().double().cpu().numpy().copy()
    r = rotation.detach().double().cpu().numpy()
    s = scales.detach().double().cpu().numpy()
    cov = (r * (s*s)[None, :]) @ r.T
    cov = (cov + cov.T) / 2
    c[c == 0] = 0.
    cov[cov == 0] = 0.
    return c.tobytes(), cov.tobytes()


def one_ray_reference(centers_camera, quaternions_wxyz, scales_unaa, alpha,
                      ray_v, fx, fy, baseline_z, gaussian_ids=None,
                      max_accepted=MAX_ACCEPTED):
    """Return (candidate or original depth, audit info) for one accepted list.

    Invalid member geometry is excluded and counted; this may miss foreground.
    The first depth-ordered complete-link *contiguous segment* alone is tested.
    """
    n = centers_camera.shape[0]
    info = dict(accepted_count=n, valid_count=0, invalid_count=0,
                invalid_reasons={}, selected=[], groups=[], edges=[], pairs=[],
                reason='', candidate_valid=False, group_alpha=None, tau=[],
                footprint_q=[], thresholds=dict(max_accepted=max_accepted,
                    angle_limit_rad=ANGLE_LIMIT, incidence_min=INCIDENCE_MIN,
                    short_ratio_min=SHORT_RATIO_MIN,
                    footprint_q_max=FOOTPRINT_Q_MAX, group_alpha_min=ALPHA_MIN))

    def fallback(reason):
        info['reason'] = reason
        return baseline_z, info

    if max_accepted < 1:
        raise ValueError('max_accepted must be positive')
    if n == 0:
        return fallback('empty')
    if n > max_accepted:
        return fallback('overflow')
    if gaussian_ids is None:
        gaussian_ids = torch.arange(n, device=centers_camera.device)
    if (centers_camera.shape != (n, 3) or quaternions_wxyz.shape != (n, 4) or
            scales_unaa.shape != (n, 3) or alpha.shape != (n,) or
            ray_v.shape != (3,) or gaussian_ids.shape != (n,)):
        raise ValueError('one-ray input shape mismatch')
    if (not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0 or fy <= 0 or
            not bool(torch.isfinite(ray_v).all()) or not bool(ray_v[2] == 1)):
        raise ValueError('invalid camera ray or focal length')

    with torch.no_grad():
        m0, q0, s0, a0, v0 = (x.detach() for x in
                             (centers_camera, quaternions_wxyz, scales_unaa, alpha, ray_v))
        ids = gaussian_ids.detach()
        rotation = rotation_matrix_wxyz(torch.where(torch.isfinite(q0), q0, 0.))
        normals = torch.empty_like(m0)
        axes = torch.full((n,), -1, dtype=torch.long, device=m0.device)
        tau = torch.full((n,), float('nan'), dtype=m0.dtype, device=m0.device)
        reasons = [None] * n
        ray_norm = torch.linalg.norm(v0)
        for i in range(n):
            if (not bool(torch.isfinite(m0[i]).all()) or not bool(torch.isfinite(q0[i]).all()) or
                    not bool(torch.isfinite(s0[i]).all()) or not bool(torch.isfinite(a0[i])) or
                    bool((s0[i] <= 0).any()) or not bool(0 <= a0[i] < 1) or
                    not bool(torch.linalg.norm(q0[i]) > 0)):
                reasons[i] = 'invalid_member'
                continue
            sorted_scale, sorted_axis = torch.sort(s0[i])
            if not bool(sorted_scale[1] / sorted_scale[0] > SHORT_RATIO_MIN):
                reasons[i] = 'ambiguous_short_axis'
                continue
            axis = sorted_axis[0]
            ni = rotation[i, :, axis]
            ai = torch.dot(ni, v0)
            if not bool(torch.abs(ai) / ray_norm >= INCIDENCE_MIN):
                reasons[i] = 'grazing'
                continue
            ti = torch.dot(ni, m0[i]) / ai
            if not bool(torch.isfinite(ti)) or not bool(ti > 0):
                reasons[i] = 'nonpositive_intersection'
                continue
            axes[i], normals[i], tau[i] = axis, ni, ti
        valid = [i for i, reason in enumerate(reasons) if reason is None]
        info['valid_count'] = len(valid)
        info['invalid_count'] = n - len(valid)
        info['invalid_reasons'] = {str(i): reason for i, reason in enumerate(reasons) if reason}
        info['tau'] = [float(t) if bool(torch.isfinite(t)) else None for t in tau]
        if not valid:
            return fallback('no_valid_members')
        order = sorted(valid, key=lambda i: (float(tau[i]), int(ids[i])))

        def compatible(i, j):
            ni, nj = normals[i], normals[j]
            angle = torch.atan2(torch.linalg.norm(torch.linalg.cross(ni, nj)),
                                torch.abs(torch.dot(ni, nj)))
            delta_m = m0[j] - m0[i]
            gap = torch.maximum(torch.abs(torch.dot(ni, delta_m)),
                                torch.abs(torch.dot(nj, delta_m)))
            distance_limit = (tau[i] + tau[j]) * max(1/fx, 1/fy)
            good = bool(angle <= ANGLE_LIMIT and gap <= distance_limit)
            info['pairs'].append(dict(i=i, j=j, angle_rad=float(angle),
                distance=float(gap), distance_limit=float(distance_limit), compatible=good))
            if good:
                info['edges'].append([min(i, j), max(i, j)])
            return good

        # Do not backfill an older group. Rejection starts a new contiguous
        # segment, even when the new member is compatible with a still older one.
        groups = [[order[0]]]
        for i in order[1:]:
            if all(compatible(i, j) for j in groups[-1]):
                groups[-1].append(i)
            else:
                groups.append([i])
        info['groups'] = groups
        chosen = groups[0]
        info['selected'] = chosen
        if len({_same_geometry_signature(m0[i], rotation[i], s0[i]) for i in chosen}) < 2:
            return fallback('insufficient_distinct_geometry')
        group_alpha = 1 - torch.prod(1 - a0[chosen])
        info['group_alpha'] = float(group_alpha)
        if not bool(group_alpha >= ALPHA_MIN):
            return fallback('low_group_alpha')
        c = (normals[chosen] * v0).sum(dim=-1)
        b = (normals[chosen] * m0[chosen]).sum(dim=-1)
        denom = (c*c).sum()
        if not bool(torch.isfinite(denom)) or not bool(denom > 0):
            return fallback('invalid_denominator')
        z0 = (c*b).sum() / denom
        if not bool(torch.isfinite(z0)) or not bool(z0 > 0):
            return fallback('invalid_ls_depth')
        point = z0 * v0
        for i in chosen:
            local = rotation[i].T @ (point - m0[i])
            tangential = [j for j in range(3) if j != int(axes[i])]
            footprint_q = ((local[tangential] / s0[i, tangential])**2).sum()
            info['footprint_q'].append(dict(i=i, q=float(footprint_q)))
            if not bool(torch.isfinite(footprint_q)) or not bool(footprint_q <= FOOTPRINT_Q_MAX):
                return fallback('outside_group_footprint')
        chosen_axes = axes[chosen]

    live_rotation = rotation_matrix_wxyz(quaternions_wxyz[chosen])
    live_normals = live_rotation[torch.arange(len(chosen), device=m0.device), :, chosen_axes]
    live_c = (live_normals * ray_v).sum(dim=-1)
    live_b = (live_normals * centers_camera[chosen]).sum(dim=-1)
    depth = (live_c * live_b).sum() / (live_c * live_c).sum()
    info['candidate_valid'], info['reason'] = True, 'accepted'
    return depth, info
