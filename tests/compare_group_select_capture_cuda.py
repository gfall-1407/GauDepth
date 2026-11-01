"""Differential CUDA/Torch check on the fixed scan37 support capture.

Run after installing this branch's extension:
  python tests/compare_group_select_capture_cuda.py --capture /path/to/support.npz
No rendering or training is performed. The capture must be materialized by the
Library workflow or already present in the authorized G1 result directory.
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surface_group_batch import grouped_depth_from_support
from surface_group_reference import rotation_matrix_wxyz
from diff_gaussian_rasterization import _C

AUDIT_PATH = ROOT / '实验' / '07_训练读出接入' / 'audit_captured_support.py'
spec = importlib.util.spec_from_file_location('capture_audit', AUDIT_PATH)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def prepared_geometry(data):
    center, quat, scale = data['center'], data['quaternion'], data['scale_unaa']
    valid = (np.isfinite(center).all(1) & np.isfinite(quat).all(1) &
             (np.linalg.norm(np.where(np.isfinite(quat), quat, 0), axis=1) > 0) &
             np.isfinite(scale).all(1) & (scale > 0).all(1))
    safe_q = np.where(valid[:, None], quat, np.array([1, 0, 0, 0], dtype=quat.dtype))
    world_rot = rotation_matrix_wxyz(torch.from_numpy(safe_q)).numpy()
    wvt = data['world_view_transform']
    rot = wvt[:3, :3].T @ world_rot
    cam_center = np.where(np.isfinite(center), center, 0) @ wvt[:3, :3] + wvt[3, :3]
    safe_scale = np.where(valid[:, None], scale, 1)
    axis = np.argmin(safe_scale, axis=1)
    normal = rot[np.arange(len(rot)), :, axis]
    return dict(center=cam_center, rotation=rot, scale=safe_scale,
                axis=axis, normal=normal, primitive_valid=valid)


def margin_report(row, data, geometry):
    """Threshold proximity for a mismatched ray; values are absolute distances."""
    count = int(data['count'][row])
    if count == 0:
        return {'empty': True}
    raw = data['ids'][row, :count]
    local = np.searchsorted(data['unique_id'], raw)
    m, n, s = (geometry[key][local] for key in ('center', 'normal', 'scale'))
    fx, fy, cx, cy = data['focal']
    px, py = data['xy'][row]
    v = np.array([(px-cx)/fx, (py-cy)/fy, 1], dtype=np.float32)
    inc = n @ v
    tau = np.sum(n*m, axis=1) / inc
    margins = {'short_ratio': float(np.nanmin(np.sort(s, axis=1)[:, 1] /
                                             np.sort(s, axis=1)[:, 0] - 1.001)),
               'incidence': float(np.nanmin(np.abs(inc) / np.linalg.norm(v) - .1))}
    angle, distance = [], []
    for i in range(count):
        for j in range(i+1, count):
            if not np.isfinite(tau[[i,j]]).all() or (tau[[i,j]] <= 0).any():
                continue
            delta = m[j]-m[i]
            angle.append(abs(abs(float(n[i] @ n[j]))-math.cos(math.radians(10))))
            gap = max(abs(float(n[i] @ delta)), abs(float(n[j] @ delta)))
            limit = float((tau[i]+tau[j])*max(1/fx,1/fy))
            distance.append(abs(gap-limit))
    margins['pair_angle'] = min(angle, default=None)
    margins['pair_distance'] = min(distance, default=None)
    return margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA extension required')
    with np.load(args.capture) as source:
        data = {key: source[key] for key in source.files}
    if data['ids'].shape[1] < 128 or data['alpha'].shape != data['ids'].shape:
        raise ValueError('capture capacity/shape mismatch')
    rows = np.flatnonzero((data['count'] <= 128) & ~data['overflow'])
    if len(rows) != 920:
        raise ValueError(f'expected fixed 920 eligible rays, got {len(rows)}')
    raw_ids = data['ids'][rows, :128]
    local = np.searchsorted(data['unique_id'], np.maximum(raw_ids, 0))
    valid_ids = raw_ids >= 0
    if np.any(local[valid_ids] >= len(data['unique_id'])):
        raise ValueError('capture ID lookup exceeds unique IDs')
    if np.any(data['unique_id'][local[valid_ids]] != raw_ids[valid_ids]):
        raise ValueError('capture ID lookup failed')
    local = np.where(raw_ids >= 0, local, -1).astype(np.int32)
    geometry = prepared_geometry(data)
    fx, fy, cx, cy = map(float, data['focal'])
    inputs = (torch.from_numpy(data['xy'][rows]), torch.from_numpy(local),
              torch.from_numpy(data['alpha'][rows, :128]),
              torch.from_numpy(data['count'][rows]),
              torch.zeros(len(rows), dtype=torch.bool),
              torch.from_numpy(data['native_expected'][rows]))

    def run(device):
        xyz = torch.from_numpy(data['center']).to(device).requires_grad_()
        quat = torch.from_numpy(data['quaternion']).to(device).requires_grad_()
        scale = torch.from_numpy(data['scale_unaa']).to(device)
        pc = SimpleNamespace(get_xyz=xyz, _rotation=quat, get_scaling=scale)
        camera = SimpleNamespace(world_view_transform=torch.from_numpy(
            data['world_view_transform']).to(device), Fx=fx, Fy=fy, Cx=cx, Cy=cy)
        xy, ids, alpha, count, overflow, native = [x.to(device) for x in inputs]
        xy.requires_grad_(); native.requires_grad_()
        depth, audit = grouped_depth_from_support(
            pc, camera, xy, ids, alpha, count, overflow, native,
            chunk_size=64, selection='first_qualified')
        grad = torch.autograd.grad(depth.sum(), (xyz, quat, xy, native),
                                   allow_unused=True)
        return depth.detach().cpu(), {k:v.detach().cpu() for k,v in audit.items()}, \
               [None if x is None else x.detach().cpu() for x in grad]

    cpu_depth, cpu_audit, cpu_grad = run('cpu')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    gpu_depth, gpu_audit, gpu_grad = run('cuda')
    torch.cuda.synchronize()
    print(f'920-ray CUDA elapsed_ms={(time.perf_counter()-start)*1000:.1f} '
          f'peak_bytes={torch.cuda.max_memory_allocated()}')
    for key in cpu_audit:
        a, b = cpu_audit[key], gpu_audit[key]
        if a.dtype.is_floating_point:
            mismatch = ~torch.isclose(a, b, rtol=2e-4, atol=2e-6, equal_nan=True)
        else:
            mismatch = a != b
        if mismatch.any():
            index = int(mismatch.nonzero()[0, 0]); row = int(rows[index])
            raise AssertionError(f'audit {key} mismatch at capture row {row}: '
                                 f'cpu={a[index]} cuda={b[index]} '
                                 f'margins={margin_report(row,data,geometry)}')
    for index in range(4):
        a,b=cpu_grad[index],gpu_grad[index]
        if a is None or b is None:
            assert a is None and b is None
            continue
        maxdiff=(a-b).abs().max().item()
        print(f'grad {index} maxdiff={maxdiff:.3g}')
        torch.testing.assert_close(a,b,rtol=2e-3,atol=2e-5)
    torch.testing.assert_close(cpu_depth,gpu_depth,rtol=2e-4,atol=2e-5)

    # Compare selected original IDs, order and group index to the independent
    # captured-ray audit, not merely two outputs of the same CUDA kernel.
    c = geometry
    cuda_geometry = [torch.from_numpy(c[key]).to('cuda').contiguous() for key in
                     ('center','rotation','scale','primitive_valid')]
    xy, ids, alpha, count, overflow, _ = [x.to('cuda') for x in inputs]
    slots, axes, audit_i, _, flags = _C.select_support_groups(
        *cuda_geometry, xy, ids, alpha, count, overflow, fx, fy, cx, cy)
    slots, audit_i, flags = slots.cpu().numpy(), audit_i.cpu().numpy(), flags.cpu().numpy()
    for index,row in enumerate(rows):
        trace = audit_module.audit_ray(int(row),data,c)
        qualified = next((g for g,group in enumerate(trace['groups'])
                          if group['qualified']), -1)
        expected = trace['groups'][qualified]['ids'] if qualified >= 0 else []
        chosen_slots = slots[index, slots[index] >= 0]
        actual = data['ids'][row, chosen_slots].tolist()
        if actual != expected or audit_i[index,7] != qualified or bool(flags[index,0]) != (qualified>=0):
            raise AssertionError(f'selection mismatch at capture row {row}: '
                                 f'expected={expected} actual={actual} '
                                 f'group={qualified}/{audit_i[index,7]} '
                                 f'margins={margin_report(int(row),data,c)}')
    print('PASS: 920 captured rays audit/slots/depth/grad versus Torch and independent trace')


if __name__ == '__main__':
    main()
