"""CPU audit of fixed captured accepted lists; no model labels or threshold tuning."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from surface_group_batch import grouped_depth_from_support
from surface_group_reference import rotation_matrix_wxyz


COS10 = math.cos(math.radians(10.))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_group(group, centers, rotations, scales, axis, normals, alpha, ray):
    if len(group) < 2:
        return dict(qualified=False, reason='singleton', alpha=float(1-np.prod(1-alpha[group])))
    signatures = set()
    for i in group:
        cov = (rotations[i] * (scales[i]**2)[None, :]) @ rotations[i].T
        cov = (cov + cov.T) / 2
        center, cov = centers[i].copy(), cov.copy()
        center[center == 0] = 0
        cov[cov == 0] = 0
        signatures.add((center.tobytes(), cov.tobytes()))
    if len(signatures) < 2:
        return dict(qualified=False, reason='duplicate', alpha=float(1-np.prod(1-alpha[group])))
    a = normals[group] @ ray
    b = np.sum(normals[group] * centers[group], axis=1)
    denom = float(np.sum(a*a))
    depth = float(np.sum(a*b) / denom) if np.isfinite(denom) and denom > 0 else None
    if depth is not None and not np.isfinite(depth):
        depth = None
    support_alpha = float(1-np.prod(1-alpha[group]))
    if support_alpha < .5:
        return dict(qualified=False, reason='low_alpha', alpha=support_alpha, depth=depth)
    if depth is None or depth <= 0:
        return dict(qualified=False, reason='invalid_ls', alpha=support_alpha, depth=depth)
    point = depth * ray
    q_values = []
    for i in group:
        local = rotations[i].T @ (point - centers[i])
        tangent = [j for j in range(3) if j != axis[i]]
        q_values.append(float(np.sum((local[tangent] / scales[i, tangent])**2)))
    if not np.isfinite(q_values).all() or max(q_values) > 4.:
        return dict(qualified=False, reason='footprint', alpha=support_alpha,
                    depth=depth, footprint_max=max(q_values))
    return dict(qualified=True, reason='accepted', alpha=support_alpha,
                depth=depth, footprint_max=max(q_values))


def audit_ray(row, captured, prepared):
    count = int(captured['count'][row])
    capacity = captured['ids'].shape[1]
    result = dict(flat_pixel=int(captured['flat_pixel'][row]),
                  accepted_count=count, overflow=bool(captured['overflow'][row]),
                  invalid_reasons={}, first_boundary=None, groups=[])
    if count > capacity or result['overflow']:
        result['reason'] = 'overflow_unresolved'
        return result
    raw = captured['ids'][row, :count]
    lookup = np.searchsorted(captured['unique_id'], raw)
    m, r, s, n, axes = (prepared[key][lookup] for key in
                        ('center', 'rotation', 'scale', 'normal', 'axis'))
    alpha = captured['alpha'][row, :count]
    xy = captured['xy'][row]
    fx, fy, cx, cy = captured['focal']
    ray = np.array([(xy[0]-cx)/fx, (xy[1]-cy)/fy, 1.], dtype=np.float32)
    ray_norm = np.linalg.norm(ray)
    valid, taus = [], {}
    for i in range(count):
        reason = None
        if (not np.isfinite(m[i]).all() or not np.isfinite(r[i]).all() or
                not np.isfinite(s[i]).all() or np.any(s[i] <= 0) or
                not np.isfinite(alpha[i]) or not 0 <= alpha[i] < 1 or
                not prepared['primitive_valid'][lookup[i]]):
            reason = 'basic'
        elif np.sort(s[i])[1] / np.sort(s[i])[0] <= 1.001:
            reason = 'short_axis'
        else:
            incidence = float(np.dot(n[i], ray))
            if abs(incidence) / ray_norm < .1:
                reason = 'grazing'
            else:
                tau = float(np.dot(n[i], m[i]) / incidence)
                if not np.isfinite(tau) or tau <= 0:
                    reason = 'tau'
                else:
                    valid.append(i)
                    taus[i] = tau
        if reason:
            result['invalid_reasons'][str(int(raw[i]))] = reason
    valid.sort(key=lambda i: (taus[i], int(raw[i])))
    result['valid_count'] = len(valid)
    if not valid:
        result['reason'] = 'no_valid'
        return result
    groups = [[valid[0]]]
    for i in valid[1:]:
        old = groups[-1]
        dots = np.abs(n[old] @ n[i])
        delta = m[i] - m[old]
        distance = np.maximum(np.abs(np.sum(n[old] * delta, axis=1)),
                              np.abs(delta @ n[i]))
        limit = (np.array([taus[j] for j in old]) + taus[i]) * max(1/fx, 1/fy)
        angle_ok, distance_ok = dots >= COS10, distance <= limit
        if np.all(angle_ok & distance_ok):
            old.append(i)
        else:
            if result['first_boundary'] is None:
                result['first_boundary'] = dict(angle_fail=bool((~angle_ok).any()),
                                                distance_fail=bool((~distance_ok).any()))
            groups.append([i])
    for group in groups:
        verdict = audit_group(group, m, r, s, axes, n, alpha, ray)
        result['groups'].append(dict(ids=[int(raw[i]) for i in group], **verdict))
    result['reason'] = result['groups'][0]['reason']
    result['later_qualified_groups'] = sum(g['qualified'] for g in result['groups'][1:])
    return result


def summarize_rows(rows, capacity):
    """Keep the frozen v1 K=128 domain distinct from wider CPU-only captures."""
    eligible = [row for row in rows if row['accepted_count'] <= 128 and not row['overflow']]
    wider = [row for row in rows if 128 < row['accepted_count'] <= capacity and not row['overflow']]
    overflow = [row for row in rows if row['overflow']]
    strata = {'success': [], 'overflow': [], 'over_v1_cap': [],
              'low_support': [], 'other': []}
    for row in rows:
        label = ('overflow' if row['overflow'] else
                 'over_v1_cap' if row['accepted_count'] > 128 else
                 'success' if row['reason'] == 'accepted' else
                 'low_support' if row['reason'] in ('singleton', 'duplicate', 'low_alpha')
                 else 'other')
        if len(strata[label]) < 4:
            strata[label].append(row['flat_pixel'])
    return dict(count_le128=len(eligible), count_129_to_capture_cap=len(wider),
                capture_overflow=len(overflow),
                cpu_reason_count_captured=dict(Counter(row['reason'] for row in rows)),
                native_v1_reason_count=dict(Counter(row['reason'] for row in eligible)),
                later_qualified_after_first_failure_v1_eligible=sum(
                    row['reason'] != 'accepted' and row.get('later_qualified_groups', 0) > 0
                    for row in eligible),
                first_singleton_v1_eligible=sum(
                    bool(row['groups']) and len(row['groups'][0]['ids']) == 1 for row in eligible),
                first_alpha_pass_v1_eligible=sum(
                    bool(row['groups']) and row['groups'][0]['alpha'] >= .5 for row in eligible),
                boundary_angle_fail_v1_eligible=sum(
                    bool(row['first_boundary']) and row['first_boundary']['angle_fail']
                    for row in eligible),
                boundary_distance_fail_v1_eligible=sum(
                    bool(row['first_boundary']) and row['first_boundary']['distance_fail']
                    for row in eligible),
                fixed_examples=strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    metadata_path = Path(args.capture).with_name('metadata.json')
    capture_hash = sha256(args.capture)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['result_sha256'] != capture_hash:
            raise ValueError('captured support SHA256 differs from metadata')
    with np.load(args.capture) as source:
        data = {key: source[key] for key in source.files}
    query_count, capacity = data['ids'].shape
    if (data['alpha'].shape != (query_count, capacity) or
            any(data[key].shape != (query_count,) for key in
                ('count', 'overflow', 'flat_pixel', 'native_expected')) or
            data['xy'].shape != (query_count, 2) or
            np.any(data['count'] < 0) or
            np.any(data['overflow'] != (data['count'] > capacity)) or
            np.any((data['ids'] >= 0) & ~np.isin(data['ids'], data['unique_id']))):
        raise ValueError('invalid captured support shape/count/ID')
    centers, quaternion, scales = data['center'], data['quaternion'], data['scale_unaa']
    primitive_valid = (np.isfinite(centers).all(axis=1) & np.isfinite(quaternion).all(axis=1) &
                       (np.linalg.norm(np.where(np.isfinite(quaternion), quaternion, 0), axis=1) > 0) &
                       np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1))
    q_safe = np.where(primitive_valid[:, None], quaternion,
                      np.array([1, 0, 0, 0], dtype=quaternion.dtype))
    rot_world = rotation_matrix_wxyz(torch.from_numpy(q_safe)).numpy()
    wvt = data['world_view_transform']
    rotation = wvt[:3, :3].T @ rot_world
    center = np.where(np.isfinite(centers), centers, 0) @ wvt[:3, :3] + wvt[3, :3]
    safe_scale = np.where(primitive_valid[:, None], scales, 1)
    axis = np.argmin(safe_scale, axis=1)
    normal = rotation[np.arange(len(rotation)), :, axis]
    prepared = dict(center=center, rotation=rotation, scale=safe_scale,
                    axis=axis, normal=normal, primitive_valid=primitive_valid)
    rows = [audit_ray(i, data, prepared) for i in range(len(data['count']))]
    comparable = np.flatnonzero((data['count'] <= 128) & ~data['overflow'])
    parity = dict(comparable=int(len(comparable)), verdict_mismatch=None,
                  group_size_mismatch=None, invalid_count_mismatch=None,
                  depth_max_abs=None, depth_over_2e5=None)
    if len(comparable) and len(data['unique_id']):
        raw_ids = data['ids'][comparable, :128]
        local_id = np.searchsorted(data['unique_id'], np.maximum(raw_ids, 0))
        local_id = np.where(raw_ids >= 0, local_id, -1).astype(np.int32)
        pc = SimpleNamespace(get_xyz=torch.from_numpy(centers),
                             _rotation=torch.from_numpy(quaternion),
                             get_scaling=torch.from_numpy(scales))
        fx, fy, cx, cy = data['focal']
        camera = SimpleNamespace(world_view_transform=torch.from_numpy(wvt),
                                 Fx=float(fx), Fy=float(fy), Cx=float(cx), Cy=float(cy))
        batch_depth, batch_audit = grouped_depth_from_support(
            pc, camera, torch.from_numpy(data['xy'][comparable]),
            torch.from_numpy(local_id), torch.from_numpy(data['alpha'][comparable, :128]),
            torch.from_numpy(data['count'][comparable]),
            torch.zeros(len(comparable), dtype=torch.bool),
            torch.from_numpy(data['native_expected'][comparable]), chunk_size=64)
        expected_accept = np.array([rows[i]['reason'] == 'accepted' for i in comparable])
        expected_group_size = np.array([len(rows[i]['groups'][0]['ids']) if rows[i]['groups'] else 0
                                        for i in comparable])
        expected_invalid = np.array([len(rows[i]['invalid_reasons']) for i in comparable])
        batch_accept = batch_audit['candidate'].numpy()
        parity['verdict_mismatch'] = int(np.count_nonzero(batch_accept != expected_accept))
        parity['group_size_mismatch'] = int(np.count_nonzero(
            batch_audit['first_group_size'].numpy() != expected_group_size))
        parity['invalid_count_mismatch'] = int(np.count_nonzero(
            batch_audit['invalid_count'].numpy() != expected_invalid))
        both = np.flatnonzero(batch_accept & expected_accept)
        if len(both):
            cpu_depth = np.array([rows[comparable[i]]['groups'][0]['depth'] for i in both])
            difference = np.abs(batch_depth.numpy()[both] - cpu_depth)
            parity['depth_max_abs'] = float(difference.max())
            parity['depth_over_2e5'] = int(np.count_nonzero(
                difference > 2e-5 * np.maximum(1., np.abs(cpu_depth))))
    parity['ok'] = (parity['comparable'] > 0 and parity['verdict_mismatch'] == 0 and
                    parity['group_size_mismatch'] == 0 and
                    parity['invalid_count_mismatch'] == 0 and
                    parity['depth_over_2e5'] in (None, 0))
    summary = dict(capture=str(args.capture), capture_sha256=capture_hash,
                   queries=len(rows), cpu_v1_parity=parity,
                   **summarize_rows(rows, capacity))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'rays.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
