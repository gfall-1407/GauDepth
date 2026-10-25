"""NumPy research approximation of the preserved host; require saved-host parity first.

Geometry formulas are derived from Inria/RaDe-GS render_forward.cu and retain
its research-only LICENSE.md terms. This is not the installed CUDA renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

F = np.float32


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _unit(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def load_model(path):
    # GaussianModel.save_ply writes one binary little-endian float vertex table.
    # Refuse other formats instead of adding a broad PLY conversion dependency.
    fields, count = [], None
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        if stream.readline().strip() != b'format binary_little_endian 1.0':
            raise ValueError('Expected original binary little-endian Gaussian PLY')
        for _ in range(1024):
            line = stream.readline().decode('ascii').strip().split()
            if line == ['end_header']:
                break
            if not line or line[0] == 'comment':
                continue
            if line[:2] == ['element', 'vertex'] and count is None:
                count = int(line[2])
            elif line[:2] == ['property', 'float'] and count is not None and len(line) == 3:
                fields.append((line[2], '<f4'))
            else:
                raise ValueError(f'Unsupported Gaussian PLY header line: {line}')
        else:
            raise ValueError('PLY header exceeds 1024 lines')
        offset = stream.tell()
    if count is None or not fields:
        raise ValueError('Missing original vertex table')
    p = np.memmap(path, dtype=np.dtype(fields), offset=offset, shape=(count,), mode='r')
    def cols(names):
        return np.stack([p[n] for n in names], axis=1).astype(np.float32)
    xyz = cols(('x', 'y', 'z'))
    raw_s = cols(('scale_0', 'scale_1', 'scale_2'))
    raw_q = cols(('rot_0', 'rot_1', 'rot_2', 'rot_3'))
    s2 = np.exp(raw_s) ** F(2)
    filtered = s2 + np.asarray(p['filter_3D'], dtype=F)[:, None] ** F(2)
    opacity = F(1) / (F(1) + np.exp(-np.asarray(p['opacity'], dtype=F)))
    opacity *= np.sqrt(np.prod(s2, axis=1) / np.prod(filtered, axis=1))
    quat = _unit(raw_q)
    r, x, y, z = quat.T
    rotation = np.empty((len(x), 3, 3), dtype=F)
    rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rotation[:, 0, 1] = 2 * (x*y - r*z)
    rotation[:, 0, 2] = 2 * (x*z + r*y)
    rotation[:, 1, 0] = 2 * (x*y + r*z)
    rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rotation[:, 1, 2] = 2 * (y*z - r*x)
    rotation[:, 2, 0] = 2 * (x*z - r*y)
    rotation[:, 2, 1] = 2 * (y*z + r*x)
    rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    if not all(np.isfinite(a).all() for a in (xyz, filtered, opacity, rotation)):
        raise ValueError('Nonfinite model activations; no implicit repair')
    return dict(xyz=xyz, scales=np.sqrt(filtered), opacity=opacity, rotation=rotation)


def preprocess(model, view, kernel_size=0.):
    """Vectorized float32 camera preprocessing; no RGB or fitted parameters."""
    xyz = model['xyz']
    V = np.asarray(view['world_view_transform'], dtype=F)
    R = V[:3, :3].T
    t = xyz @ V[:3, :3] + V[3, :3]
    distance = np.sqrt(np.sum(t*t, axis=1))
    height, width = map(int, view['image_size'])
    fx, fy = map(F, view['focal'])
    tanx, tany = [F(math.tan(float(v) / 2)) for v in view['fov']]
    clipped = t.copy()
    clipped[:, 0] = np.clip(t[:, 0]/t[:, 2], -F(1.3)*tanx, F(1.3)*tanx)*t[:, 2]
    clipped[:, 1] = np.clip(t[:, 1]/t[:, 2], -F(1.3)*tany, F(1.3)*tany)*t[:, 2]
    u, v = clipped[:, 0]/clipped[:, 2], clipped[:, 1]/clipped[:, 2]
    tz = clipped[:, 2]
    # GLM's constructor is column-major; these arrays use conventional rows.
    Q = R[None, :, :] @ model['rotation']
    J = np.zeros((len(xyz), 2, 3), dtype=F)
    J[:, 0, 0], J[:, 1, 1] = fx/tz, fy/tz
    J[:, 0, 2] = -(fx*clipped[:, 0])/(tz*tz)
    J[:, 1, 2] = -(fy*clipped[:, 1])/(tz*tz)
    M = (J @ Q) * model['scales'][:, None, :]
    cov = M @ np.swapaxes(M, 1, 2)
    a0, b, c0 = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    a, c = a0+F(kernel_size), c0+F(kernel_size)
    determinant = a*c-b*b
    coefficient = np.sqrt(np.maximum(F(1e-6), a0*c0-b*b)/np.maximum(F(1e-6), determinant))
    conic = np.stack((c/determinant, -b/determinant, a/determinant,
                      model['opacity']*coefficient), axis=1)
    mid = F(.5)*(a+c)
    radius = np.ceil(F(3)*np.sqrt(mid+np.sqrt(np.maximum(F(.1), mid*mid-determinant))))
    # Reconstruct the actual Camera full projection, rather than fx*x/z.
    P = np.zeros((4, 4), dtype=F)
    P[0, 0] = F(1/math.tan(float(view['fov'][0])/2))
    P[1, 1] = F(1/math.tan(float(view['fov'][1])/2))
    P[2, 2], P[2, 3], P[3, 2] = F(100/99.99), F(-1/99.99), F(1)
    full = V @ P.T
    homogeneous = xyz @ full[:3, :] + full[3, :]
    ndc = homogeneous[:, :2] / (homogeneous[:, 3, None] + F(1e-7))
    xy = ((ndc + F(1))*np.array([width, height], dtype=F)-F(1))*F(.5)
    grid = np.array([(width+15)//16, (height+15)//16])
    # C++ truncates towards zero before clamping; floor differs for negatives.
    rect_min = np.clip(np.trunc((xy-radius[:, None])/F(16)).astype(np.int64), 0, grid)
    rect_max = np.clip(np.trunc((xy+radius[:, None]+F(15))/F(16)).astype(np.int64), 0, grid)
    usable = ((t[:, 2] > F(.2)) & (determinant != 0) &
              np.all(rect_max > rect_min, axis=1) & np.isfinite(conic).all(axis=1))
    # Inverse covariance in camera coordinates, including the rank-one branch.
    invq = Q/model['scales'][:, None, :]
    invcov = invq @ np.swapaxes(invq, 1, 2)
    weak = np.min(model['scales'], axis=1) <= F(1e-7)
    if weak.any():
        indices = np.flatnonzero(weak)
        axis = Q[indices, :, np.argmin(model['scales'][indices], axis=1)]
        invcov[indices] = axis[:, :, None]*axis[:, None, :]
    uvh = np.stack((u, v, np.ones_like(u)), axis=1)
    mn = _unit(np.einsum('nij,nj->ni', invcov, uvh))
    denominator = np.maximum(np.sum(mn*uvh, axis=1), F(1e-7))
    mx, my, mz = mn.T
    plane0 = ((1+v*v)*mx-u*v*my-u*mz)/denominator
    plane1 = (-u*v*mx+(1+u*u)*my-v*mz)/denominator
    length = np.sqrt(np.sum(clipped*clipped, axis=1))
    factor = length/(u*u+v*v+F(1))
    plane = np.stack((plane0*factor/fx, plane1*factor/fy, distance), axis=1)
    rx, ry, rz = -plane0*factor, -plane1*factor, -np.ones_like(u)
    normal = np.stack((rx/tz+clipped[:, 0]/length*rz,
                       ry/tz+clipped[:, 1]/length*rz,
                       -clipped[:, 0]/(tz*tz)*rx-clipped[:, 1]/(tz*tz)*ry+tz/length*rz), axis=1)
    normal = _unit(normal)
    degenerate = ~np.isfinite(mn[:, 0])
    plane[degenerate] = 0
    normal[degenerate] = np.array([0, 0, -1], dtype=F)
    return dict(xy=xy, conic=conic, plane=plane, normal=normal,
                degenerate=degenerate, rect_min=rect_min, rect_max=rect_max,
                usable=usable, sort_depth=distance, width=width, height=height,
                focal=np.array([fx, fy]), tile_cache={})


def trace(prepared, pixel_xy):
    x, y = map(int, pixel_xy)
    if not (0 <= x < prepared['width'] and 0 <= y < prepared['height']):
        raise ValueError('Pixel outside view')
    tile = (x//16, y//16)
    cache = prepared['tile_cache']
    if tile not in cache:
        tile_arr = np.array(tile)
        valid = (prepared['usable'] & np.all(prepared['rect_min'] <= tile_arr, axis=1) &
                 np.all(prepared['rect_max'] > tile_arr, axis=1))
        ids = np.flatnonzero(valid)
        # Positive float32 distance key; stable ties retain original Gaussian ID.
        cache[tile] = ids[np.argsort(prepared['sort_depth'][ids], kind='stable')]
    candidates = cache[tile]
    delta = prepared['xy'][candidates] - np.array([x, y], dtype=F)
    dx, dy = delta.T
    c = prepared['conic'][candidates]
    power = -F(.5)*(c[:, 0]*dx*dx+c[:, 2]*dy*dy)-c[:, 1]*dx*dy
    alpha = np.minimum(F(.99), c[:, 3]*np.exp(power))
    keep = (power <= 0) & (alpha >= F(1/255))
    candidates, delta, alpha = candidates[keep], delta[keep], alpha[keep]
    nx = F(F(x)-F((prepared['width']-1)/2))/prepared['focal'][0]
    ny = F(F(y)-F((prepared['height']-1)/2))/prepared['focal'][1]
    rln = F(1)/np.sqrt(nx*nx+ny*ny+F(1))
    T, D, median = F(1), F(0), F(0)
    records, accepted = [], []
    for gid, d, a in zip(candidates, delta, alpha):
        nxt = F(T*F(1-a))
        if nxt < F(.0001):
            break
        pl = prepared['plane'][gid]
        tt = F(F(pl[0]*d[0]+pl[1]*d[1])+pl[2])
        w = F(T*a)
        records.append([F(tt*rln), w, a, T, *prepared['normal'][gid], F(prepared['degenerate'][gid])])
        accepted.append(gid)
        D = F(D+F(tt*w))
        if T > F(.5):
            median = tt
        T = nxt
    replay = np.array([F(D*rln/F(1-T)) if accepted else F(0), F(median*rln), F(1-T)])
    return np.asarray(accepted, dtype=np.int32), np.asarray(records, dtype=F).reshape(-1, 8), replay


def compare(a, b, rtol, atol):
    if a.shape != b.shape:
        return dict(pass_=False, shape_a=list(a.shape), shape_b=list(b.shape))
    difference = np.abs(a.astype(float)-b.astype(float))
    return dict(pass_=bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)),
                max_abs=float(np.nanmax(difference)) if difference.size else 0.,
                rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--view', type=int, default=0)
    parser.add_argument('--parity-rays', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads(args.metadata.read_text(encoding='utf-8'))
    view = next(v for v in meta['views'] if v['view_index'] == args.view)
    raw_path = args.metadata.parent/view['file']
    if sha256(args.model) != meta['model_sha256'] or sha256(raw_path) != view['npz_sha256']:
        raise ValueError('Model/raw parity inputs do not match saved host metadata')
    started = time.monotonic()
    model = load_model(args.model)
    prepared = preprocess(model, view, meta['effective_model_args']['kernel_size'])
    raw = np.load(raw_path)
    rows, all_ids, all_records, offsets, replays = [], [], [], [0], []
    for ray in range(min(args.parity_rays, len(raw['pixel_xy']))):
        ids, records, replay = trace(prepared, raw['pixel_xy'][ray])
        sl = slice(int(raw['offsets'][ray]), int(raw['offsets'][ray+1]))
        old_ids = raw['gaussian_id'][sl]
        same_ids = bool(np.array_equal(ids, old_ids))
        row = dict(ray=ray, pixel_xy=raw['pixel_xy'][ray].tolist(), cpu_members=len(ids), host_members=len(old_ids), exact_ids_order=same_ids)
        if same_ids:
            row['depth'] = compare(records[:, 0], raw['depth'][sl], 2e-5, 2e-6)
            row['weight'] = compare(records[:, 1], raw['weight'][sl], 2e-5, 2e-6)
            row['normal'] = compare(records[:, 4:7], raw['normal'][sl], 2e-4, 2e-5)
        row['replay'] = compare(replay, raw['host_target'][ray], 2e-5, 2e-6)
        rows.append(row)
        all_ids.append(ids); all_records.append(records); offsets.append(offsets[-1]+len(ids)); replays.append(replay)
    passed = all(r['exact_ids_order'] and all(r[k]['pass_'] for k in ('depth', 'weight', 'normal', 'replay')) for r in rows)
    commit = subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(status='parity_passed' if passed else 'parity_failed', computed_commit=commit,
                  model_sha256=sha256(args.model), raw_sha256=sha256(raw_path), metadata_sha256=sha256(args.metadata),
                  view=view['image_name'], rays=len(rows), elapsed_seconds=time.monotonic()-started, rows=rows,
                  interpretation='Independent CPU source-formula approximation calibrated on saved original-host rays; not proof of installed CUDA source or all-pixel parity.')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'cpu_parity_rays.npz', offsets=np.array(offsets), gaussian_id=np.concatenate(all_ids),
                        records=np.concatenate(all_records), replay=np.array(replays), pixel_xy=raw['pixel_xy'][:len(rows)])
    (args.output/'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
