"""Fixed two-root raw-parameter bridge; no renderer, optimizer, or root selection.

The caller supplies the separately reviewed selected_root implementation. All
Gaussian IDs are gathered from one shared parameter table; the saved PLY filter
is a detached constant. Camera rays are world-space and camera-z parameterized.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOTS = ((17, 2), (24, 6))
EXPECTED_CAPTURE_SHA = '74fbd05c3a2c10549adca00f885cdb39c525d003ce120d1a3d593c7feb78d6ef'
EXPECTED_PLY_SHA = '733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7'
DTYPE = torch.float64


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def quat_matrix(raw):
    """Host build_rotation convention: quaternion is (w,x,y,z)."""
    if raw.ndim != 2 or raw.shape[1] != 4 or not bool(torch.isfinite(raw).all()):
        raise ValueError('invalid_raw_quaternion')
    norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    if bool((norm <= 0).any()):
        raise ValueError('zero_raw_quaternion')
    w, x, y, z = (raw / norm).unbind(1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1).reshape(-1, 3, 3)


def activate(xyz, log_scale, raw_quat, logit, saved_filter):
    """Match GaussianModel.get_scaling_n_opacity_with_3D_filter."""
    n = xyz.shape[0]
    if (xyz.shape != (n, 3) or log_scale.shape != (n, 3) or
            raw_quat.shape != (n, 4) or logit.shape != (n, 1) or
            saved_filter.shape != (n, 1)):
        raise ValueError('invalid_raw_parameter_shape')
    for name, value in (('xyz', xyz), ('log_scale', log_scale), ('filter', saved_filter)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite_' + name)
    if bool((saved_filter < 0).any()):
        raise ValueError('negative_saved_filter')
    # +Inf is a valid saved raw logit (sigmoid=1); NaN is never valid.
    if bool(torch.isnan(logit).any()):
        raise ValueError('nan_raw_logit')
    scale = torch.exp(log_scale)
    scale_sq = scale.square()
    filtered_sq = scale_sq + saved_filter.square()
    filtered = torch.sqrt(filtered_sq)
    coefficient = torch.sqrt(scale_sq.prod(1) / filtered_sq.prod(1))
    amplitude = torch.sigmoid(logit) * coefficient[:, None]
    if not bool(torch.isfinite(filtered).all()) or not bool(torch.isfinite(amplitude).all()):
        raise ValueError('nonfinite_filtered_parameters')
    return dict(scale_unaa=scale, scale_render=filtered,
                opacity_render=amplitude, rotation=quat_matrix(raw_quat))


def world_camera_ray(world_view_transform, intrinsics, xy):
    """Reproduce the captured float32 world-view matrix route, without normalizing."""
    camera_R = world_view_transform[:3, :3]
    camera_T = world_view_transform[3, :3]
    fx, fy, cx, cy = intrinsics
    v = torch.stack(((xy[0]-cx)/fx, (xy[1]-cy)/fy, xy.new_tensor(1.)))
    origin = -(camera_R @ camera_T)
    ray = camera_R @ v
    forward = camera_R[:, 2]
    # The recorded float32 camera axes have about 2e-8 non-orthogonality.
    # This is only a source-integrity check; do not force ray.dot(forward)=1.
    if not bool(torch.isfinite(ray).all()) or abs(float(ray @ forward)-1) > 1e-6:
        raise ValueError('not_camera_z_ray')
    return origin, ray, forward


def load_fixed_capture(capture_path, roots_path, metadata_path):
    if sha256(capture_path) != EXPECTED_CAPTURE_SHA:
        raise ValueError('wrong_capture_sha256')
    roots = json.loads(Path(roots_path).read_text(encoding='utf-8'))
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if (roots['capture_sha256'] != EXPECTED_CAPTURE_SHA or
            roots['ply_sha256'] != EXPECTED_PLY_SHA or
            metadata['ply_sha256'] != EXPECTED_PLY_SHA):
        raise ValueError('source_version_mismatch')
    with np.load(capture_path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    ids = arrays['unique_id']
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid_global_ids')
    rows = {int(value): i for i, value in enumerate(ids)}
    root_rows = {}
    for center, index in ROOTS:
        reference = next(r for r in roots['rays'] if r['center_index'] == center)
        query = reference['query_index']
        count = int(arrays['count'][query])
        member_ids = [int(value) for value in arrays['ids'][query, :count]]
        if count < 2 or any(value not in rows for value in member_ids):
            raise ValueError('incomplete_root_members')
        if hashlib.sha256(json.dumps(member_ids).encode()).hexdigest() != reference['member_ids_sha256']:
            raise ValueError('root_member_ids_mismatch')
        candidates = reference['grid_512']['roots']
        if index < 1 or index >= len(candidates)-1 or not candidates[index]['admissible']:
            raise ValueError('invalid_frozen_root')
        z0 = candidates[index]['z']
        bracket = ((candidates[index-1]['z']+z0)/2,
                   (z0+candidates[index+1]['z'])/2)
        root_rows[center] = dict(query=query, ids=member_ids, indices=[rows[i] for i in member_ids],
                                 z=z0, normal=candidates[index]['surface_normal_world'],
                                 bracket=bracket, reference=reference)
    return arrays, root_rows, metadata


def shared_raw_leaves(arrays):
    result = {}
    for key, source in (('xyz', 'center'), ('log_scale', 'raw_scale'),
                        ('raw_quat', 'raw_quaternion'), ('logit', 'opacity_logit')):
        result[key] = torch.tensor(arrays[source], dtype=DTYPE, requires_grad=True)
    result['saved_filter'] = torch.tensor(arrays['filter_3D'], dtype=DTYPE)
    return result


def activation_parity(arrays, activated):
    """Compare every captured unique ID, including those shared across rays."""
    comparisons = (('scale_unaa', activated['scale_unaa'], arrays['scale_unaa']),
                   ('scale_render', activated['scale_render'], arrays['scale_render']),
                   ('opacity_render', activated['opacity_render'], arrays['opacity_render']))
    errors = {}
    for name, actual, saved in comparisons:
        expected = torch.tensor(saved, dtype=DTYPE)
        difference = (actual.detach()-expected).abs()
        relative = difference / torch.maximum(expected.abs(), expected.new_tensor(1e-6))
        errors[name] = dict(max_abs=float(difference.max()), max_relative=float(relative.max()))
        if not bool(torch.isfinite(difference).all()) or not bool(
                torch.allclose(actual.detach(), expected, atol=2e-7, rtol=2e-5)):
            raise ValueError('capture_activation_mismatch_' + name)
    normalized = torch.nn.functional.normalize(
        torch.tensor(arrays['raw_quaternion'], dtype=DTYPE), dim=1)
    quat_saved = torch.tensor(arrays['quaternion'], dtype=DTYPE)
    if not torch.allclose(normalized, quat_saved, atol=2e-7, rtol=2e-5):
        raise ValueError('capture_quaternion_mismatch')
    return errors


def root_from_shared(arrays, row, raw, selected_root):
    activated = activate(raw['xyz'], raw['log_scale'], raw['raw_quat'],
                         raw['logit'], raw['saved_filter'])
    indices = row['indices']
    xy = torch.tensor(arrays['xy'][row['query']], dtype=DTYPE)
    origin, ray, forward = world_camera_ray(
        torch.tensor(arrays['world_view_transform'], dtype=DTYPE),
        torch.tensor(arrays['intrinsics'], dtype=DTYPE), xy)
    omega = torch.zeros((len(indices), 3), dtype=DTYPE)
    depth, normal, diagnostic = selected_root(
        raw['xyz'][indices], activated['scale_render'][indices], omega,
        activated['opacity_render'][indices, 0], activated['rotation'][indices],
        origin, ray, row['bracket'])
    return depth, normal, diagnostic, activated


def fixed_root_vjp(arrays, rows, selected_root):
    """Fresh branch solve, actual-normal VJP, and same-bracket finite differences.

    No root or member is reselected by a method output. The finite-difference
    probes reconstruct filtered parameters and solve the root again each time.
    A branch failure is an explicit invalid result, not a zero gradient.
    """
    raw = shared_raw_leaves(arrays)
    activated = activate(raw['xyz'], raw['log_scale'], raw['raw_quat'],
                         raw['logit'], raw['saved_filter'])
    parity = activation_parity(arrays, activated)
    objective = None
    outputs = {}
    for center, _ in ROOTS:
        row = rows[center]
        depth, normal, diagnostic, _ = root_from_shared(arrays, row, raw, selected_root)
        if abs(float(depth.detach())-row['z']) > 1e-5:
            raise ValueError('fixed_root_depth_mismatch')
        expected_normal = torch.tensor(row['normal'], dtype=DTYPE)
        if not torch.allclose(normal.detach(), expected_normal, atol=1e-5, rtol=1e-5):
            raise ValueError('actual_surface_normal_mismatch')
        term = 1.3*depth + normal @ normal.new_tensor([.2, -.3, .4])
        objective = term if objective is None else objective+term
        outputs[center] = dict(depth=float(depth.detach()), normal=normal.detach().tolist(),
                               root_value=diagnostic['root_value'],
                               member_count=len(row['indices']))
    names = ('xyz', 'log_scale', 'raw_quat', 'logit')
    grads = torch.autograd.grad(objective, [raw[name] for name in names])
    grad_map = dict(zip(names, grads))
    for name, value in grad_map.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite_raw_gradient_' + name)
    # Select a deterministic shared ID if present; otherwise the first ID of
    # each fixed ray. A repeated ID always points to one leaf and accumulates.
    overlap = set(rows[17]['ids']) & set(rows[24]['ids'])
    if overlap:
        uid = min(overlap)
        candidates = [rows[17]['indices'][rows[17]['ids'].index(uid)]]
    else:
        candidates = [rows[center]['indices'][0] for center, _ in ROOTS]
    checks = []
    for name in names:
        for index in candidates:
            if name == 'logit' and not bool(torch.isfinite(raw[name][index]).all()):
                checks.append(dict(name=name, row=index, status='infinite_logit_no_finite_difference'))
                continue
            component = 0
            key = (index, component)
            step = 1e-5
            values = []
            for sign in (1, -1):
                changed = dict(raw)
                changed[name] = raw[name].detach().clone()
                changed[name][key] += sign*step
                try:
                    total = 0.
                    for center, _ in ROOTS:
                        z, n, _, _ = root_from_shared(arrays, rows[center], changed, selected_root)
                        total += float((1.3*z+n@n.new_tensor([.2, -.3, .4])).detach())
                    values.append(total)
                except (ValueError, RuntimeError) as exc:
                    checks.append(dict(name=name, row=index, status='invalid_branch', reason=str(exc)))
                    values = []
                    break
            if len(values) != 2:
                continue
            finite_difference = (values[0]-values[1])/(2*step)
            analytic = float(grad_map[name][key])
            relative = abs(finite_difference-analytic)/max(1., abs(finite_difference), abs(analytic))
            checks.append(dict(name=name, row=index, status='compared',
                               analytic=analytic, finite_difference=finite_difference,
                               relative_error=relative))
            if not np.isfinite(relative) or relative > 1e-4:
                raise ValueError('raw_parameter_vjp_mismatch_' + name)
    return dict(roots=outputs, activation_parity=parity, checks=checks,
                shared_ids=sorted(overlap), objective=float(objective.detach()))
