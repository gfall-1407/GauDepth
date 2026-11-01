"""Small-ray Torch reference for the proposed hard-branch depth heuristic.

This is not a training/rendering implementation. Inputs are already accepted
members for ONE query, in native acceptance order, in camera coordinates.
"""

import math
import torch


MAX_ACCEPTED = 128
ANGLE_LIMIT = math.radians(10.)
ANGLE_TOL = 1e-6
INCIDENCE_MIN = .1
SHORT_GAP_MIN = 1e-3
SCALE_RATIO_MIN = 1e-6
FOOTPRINT_Q_MAX = 4.
DIST_TOL_REL = 1e-7
ALPHA_MIN = .5


def rotation_matrix_wxyz(quaternions):
    """Quaternion-to-rotation, with Torch autograd and no fixed device."""
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1
    ).reshape(-1, 3, 3)


def _complete_link_groups(positions, edges):
    groups = [(i,) for i in positions]
    while True:
        choice = next(((a, b) for a in range(len(groups)) for b in range(a+1, len(groups))
                       if all((min(i, j), max(i, j)) in edges for i in groups[a] for j in groups[b])), None)
        if choice is None:
            return groups
        a, b = choice
        groups = ([g for k, g in enumerate(groups) if k not in (a, b)] +
                  [tuple(sorted(groups[a] + groups[b]))])
        groups.sort(key=lambda g: min(positions.index(i) for i in g))


def one_ray_reference(centers_camera, quaternions_wxyz, scales_unaa, alpha,
                      ray_v, fx, fy, baseline_z, max_accepted=MAX_ACCEPTED):
    """Return (depth, info). Discrete decisions detach; chosen LS stays live.

    A rejected earliest group falls back to baseline_z. No later group is
    allowed to claim first visibility. All depths use camera-z coordinates.
    """
    n = centers_camera.shape[0]
    info = dict(accepted_count=n, selected=[], groups=[], edges=[], reason='',
                candidate_valid=False, group_alpha=None, tau=[], footprint_q=[],
                pairs=[], order_ambiguous=False, p3_witness=None,
                thresholds=dict(max_accepted=max_accepted,
                                         angle_limit_rad=ANGLE_LIMIT, angle_tol_rad=ANGLE_TOL,
                                         incidence_min=INCIDENCE_MIN, short_gap_min=SHORT_GAP_MIN,
                                         scale_ratio_min=SCALE_RATIO_MIN,
                                         footprint_q_max=FOOTPRINT_Q_MAX,
                                         distance_tol_rel=DIST_TOL_REL,
                                         group_alpha_min=ALPHA_MIN))

    def fallback(reason):
        info['reason'] = reason
        return baseline_z, info

    if n == 0:
        return fallback('empty')
    if max_accepted < 1:
        raise ValueError('max_accepted must be positive')
    if n > max_accepted:
        return fallback('overflow')
    if (centers_camera.shape != (n, 3) or quaternions_wxyz.shape != (n, 4) or
            scales_unaa.shape != (n, 3) or alpha.shape != (n,) or ray_v.shape != (3,)):
        raise ValueError('one-ray input shape mismatch')
    if (not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0 or fy <= 0 or
            not torch.isfinite(ray_v).all() or ray_v[2] != 1):
        raise ValueError('invalid camera ray or focal length')

    with torch.no_grad():
        if (not torch.isfinite(centers_camera).all() or
                not torch.isfinite(quaternions_wxyz).all() or
                not torch.isfinite(scales_unaa).all() or
                not torch.isfinite(alpha).all() or
                (scales_unaa <= 0).any() or (alpha < 0).any() or (alpha >= 1).any() or
                (torch.linalg.norm(quaternions_wxyz, dim=-1) <= 0).any()):
            return fallback('invalid_member')
        scales, axis = torch.sort(scales_unaa.detach(), dim=-1)
        if ((scales[:, 0] / scales[:, 2]) <= SCALE_RATIO_MIN).any() or (
                (1 - scales[:, 0] / scales[:, 1]) <= SHORT_GAP_MIN).any():
            return fallback('ambiguous_short_axis')
        rotation = rotation_matrix_wxyz(quaternions_wxyz.detach())
        normals = rotation[torch.arange(n, device=axis.device), :, axis[:, 0]]
        unit_ray = ray_v.detach() / torch.linalg.norm(ray_v.detach())
        incidence = (normals * ray_v.detach()).sum(dim=-1)
        if ((normals * unit_ray).sum(dim=-1).abs() < INCIDENCE_MIN).any():
            return fallback('grazing')
        tau = (normals * centers_camera.detach()).sum(dim=-1) / incidence
        if (not torch.isfinite(tau).all() or (tau <= 0).any()):
            return fallback('nonpositive_intersection')
        info['tau'] = tau.cpu().tolist()
        # Each member's own plane intersection must be within its 2-sigma
        # tangential footprint. This test never clips the intersection.
        points = tau[:, None] * ray_v.detach()[None, :]
        local = torch.einsum('nji,nj->ni', rotation, points - centers_camera.detach())
        q = (local / scales_unaa.detach()).square()
        q[torch.arange(n, device=axis.device), axis[:, 0]] = 0
        footprint_q = q.sum(dim=-1)
        info['footprint_q'] = footprint_q.cpu().tolist()
        if (footprint_q > FOOTPRINT_Q_MAX).any():
            return fallback('outside_member_footprint')
        order = sorted(range(n), key=lambda i: (float(tau[i]), i))
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                angle = torch.atan2(torch.linalg.norm(torch.linalg.cross(normals[i], normals[j])),
                                    torch.abs(torch.dot(normals[i], normals[j])))
                delta_m = centers_camera[j].detach() - centers_camera[i].detach()
                gap = torch.maximum(torch.abs(torch.dot(normals[i], delta_m)),
                                    torch.abs(torch.dot(normals[j], delta_m)))
                distance_limit = (tau[i] + tau[j]) * max(1/fx, 1/fy)
                tolerance = DIST_TOL_REL * max(1., abs(float(tau[i])), abs(float(tau[j])))
                compatible = bool(angle <= ANGLE_LIMIT + ANGLE_TOL and gap <= distance_limit + tolerance)
                info['pairs'].append(dict(i=i, j=j, angle_rad=float(angle),
                                          distance=float(gap), distance_limit=float(distance_limit),
                                          distance_tolerance=tolerance, compatible=compatible))
                if compatible:
                    edges.add((i, j))
        groups = _complete_link_groups(order, edges)
        for b in order:
            neighbors = [i for i in order if i != b and (min(i, b), max(i, b)) in edges]
            for ai, a in enumerate(neighbors):
                for c in neighbors[ai + 1:]:
                    if (min(a, c), max(a, c)) not in edges:
                        info['order_ambiguous'] = True
                        info['p3_witness'] = [a, b, c]
                        break
                if info['order_ambiguous']:
                    break
            if info['order_ambiguous']:
                break
        info['edges'] = sorted([list(e) for e in edges])
        info['groups'] = [list(g) for g in groups]
        first = groups[0]
        info['selected'] = list(first)
        signatures = set()
        for i in first:
            c = centers_camera[i].detach().double().cpu().numpy().copy()
            r = rotation[i].detach().double().cpu().numpy()
            s = scales_unaa[i].detach().double().cpu().numpy()
            cov = (r * (s*s)[None, :]) @ r.T
            c[c == 0] = 0.
            cov[cov == 0] = 0.
            signatures.add((c.tobytes(), cov.tobytes()))
        if len(signatures) < 2:
            return fallback('insufficient_distinct_geometry')
        group_alpha = 1 - torch.prod(1 - alpha.detach()[list(first)])
        info['group_alpha'] = float(group_alpha)
        if group_alpha < ALPHA_MIN:
            return fallback('low_group_alpha')
        chosen = list(first)
        chosen_axes = axis[chosen, 0].detach()

    # Recompute normal with live quaternion; center, rotation and query remain
    # differentiable. Scale and opacity only control the detached branch here.
    live_rotation = rotation_matrix_wxyz(quaternions_wxyz[chosen])
    live_normals = live_rotation[torch.arange(len(chosen), device=chosen_axes.device), :, chosen_axes]
    m = centers_camera[chosen]
    c = (live_normals * ray_v).sum(dim=-1)
    b = (live_normals * m).sum(dim=-1)
    depth = (c * b).sum() / (c * c).sum()
    info['candidate_valid'] = True
    info['reason'] = 'accepted'
    return depth, info
