"""Finite, offline cross-view plane probe. No model training or Gaussian replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
REPO = Path(__file__).resolve().parents[2]
SIGMA = 1.0
TAU = .05
SEED = 20261101
OFF = np.array([(x, y) for y in range(-4, 5) for x in range(-4, 5)], float)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def camera_loader():
    spec = importlib.util.spec_from_file_location('colmap_loader', REPO / 'scene/colmap_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scan(scan):
    trace = next((ROOT / '数据/原始射线记录').glob(f'20261026*/confirm_v1/scan{scan}/view_000.npz'))
    meta_path = trace.with_name('metadata.json')
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    view = next(v for v in meta['views'] if v['view_index'] == 0)
    assert digest(trace) == view['npz_sha256']
    obs = next((ROOT / '数据/原始输入与基线模型').glob(f'20261026*/observations_v1/scan{scan}'))
    loader = camera_loader()
    cameras = loader.read_intrinsics_binary(str(obs / 'sparse/0/cameras.bin'))
    images = loader.read_extrinsics_binary(str(obs / 'sparse/0/images.bin'))
    byname = {v.name: v for v in images.values()}
    bundle = []
    for index in range(3):
        name = f'{index:04d}.png'
        pose = byname[name]
        cam = cameras[pose.camera_id]
        assert cam.model == 'PINHOLE' and (cam.width, cam.height) == (1554, 1162)
        fx, fy, cx, cy = cam.params
        K = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1.]], float)
        R = loader.qvec2rotmat(pose.qvec)
        t = pose.tvec
        photo = np.asarray(Image.open(obs / 'images' / name).convert('RGB'), float) / 255
        gray = photo @ np.array([.299, .587, .114])
        bundle.append(dict(K=K, R=R, t=t, gray=gray))
    vR = np.asarray(view['world_view_transform'])[:3, :3].T
    vt = np.asarray(view['world_view_transform'])[3, :3]
    assert np.max(np.abs(bundle[0]['R'] - vR)) < 1e-4
    assert np.max(np.abs(bundle[0]['t'] - vt)) < 1e-4
    assert abs(bundle[0]['K'][0, 0] / 2 - view['focal'][0]) < .001
    hashes = dict(trace=digest(trace), metadata=digest(meta_path),
        images=[digest(obs / 'images' / f'{i:04d}.png') for i in range(3)],
        cameras=digest(obs / 'sparse/0/cameras.bin'), extrinsics=digest(obs / 'sparse/0/images.bin'))
    index = json.loads((ROOT / '数据/来源索引.json').read_text(encoding='utf8'))
    known = {entry['path'].replace('\\', '/'): entry['sha256'] for entry in index['files']}
    for path in [trace, meta_path, *(obs/'images'/f'{i:04d}.png' for i in range(3)),
                 obs/'sparse/0/cameras.bin', obs/'sparse/0/images.bin']:
        relative = path.relative_to(ROOT / '数据').as_posix()
        assert known[relative] == digest(path), relative
    return np.load(trace), meta, bundle, hashes


def bilinear(im, xy):
    h, w = im.shape
    if not np.isfinite(xy).all() or np.any(xy[:, 0] < 0) or np.any(xy[:, 1] < 0) or np.any(xy[:, 0] >= w-1) or np.any(xy[:, 1] >= h-1):
        return None
    x = np.floor(xy[:, 0]).astype(int)
    y = np.floor(xy[:, 1]).astype(int)
    u = xy[:, 0] - x
    v = xy[:, 1] - y
    return ((1-u)*(1-v)*im[y, x] + u*(1-v)*im[y, x+1]
        + (1-u)*v*im[y+1, x] + u*v*im[y+1, x+1])


def zncc(a, b):
    if a is None or b is None:
        return np.nan
    a = a-a.mean(); b = b-b.mean()
    av = np.mean(a*a); bv = np.mean(b*b)
    if av <= 1e-8 or bv <= 1e-8:
        return np.nan
    return float(1 - np.clip(np.mean(a*b)/np.sqrt(av*bv), -1, 1))


def warp(pi, patch, c0, cj):
    rays = (np.linalg.inv(c0['K']) @ np.column_stack((patch, np.ones(len(patch)))).T).T
    den = rays @ pi
    if np.any(den <= 1e-8):
        return None
    X0 = rays / den[:, None]
    Xw = (X0-c0['t']) @ c0['R']
    Xj = Xw @ cj['R'].T + cj['t']
    if np.any(Xj[:, 2] <= 0):
        return None
    q = Xj @ cj['K'].T
    return q[:, :2]/q[:, 2, None]


def plane(depth, normal, ray):
    den = depth*np.dot(normal, ray)
    if abs(den) < 1e-10:
        return None
    return normal/den


def ray_case(f, i, cams):
    x, y = map(int, f['pixel_xy'][i])
    center = np.array([2*x+.5, 2*y+.5], float)
    patch = center+OFF
    ref = bilinear(cams[0]['gray'], patch)
    ray = np.linalg.inv(cams[0]['K']) @ np.r_[center, 1.]
    a, b = map(int, f['offsets'][i:i+2])
    d = f['depth'][a:b].astype(float)
    w = f['weight'][a:b].astype(float)
    if len(d) == 0 or not np.isfinite(d).all() or not np.isfinite(w).all() or np.any(d <= 0) or np.any(w <= 0):
        return None
    q = w/w.sum()
    native_depth = float(f['host_target'][i, 0])
    native_normal = f['normal_target'][i].astype(float)
    if not np.isfinite(native_normal).all() or np.linalg.norm(native_normal) < 1e-8:
        return None
    axis_uv = [warp(plane(v, native_normal, ray), center[None], cams[0], cams[1]) for v in (float(d.min()), float(d.max()))]
    if any(v is None for v in axis_uv):
        return None
    axis = axis_uv[1][0]-axis_uv[0][0]
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-5:
        return None
    axis /= axis_len
    member_uv = [warp(plane(v, native_normal, ray), center[None], cams[0], cams[1]) for v in d]
    if any(v is None for v in member_uv):
        return None
    xi = np.array([np.dot(v[0], axis) for v in member_uv])
    return dict(i=i, xy=[x,y], center=center, patch=patch, ref=ref, ray=ray, d=d, q=q, axis=axis,
        xi=xi, native_depth=native_depth, native_normal=native_normal)


def cost(pi, case, cams, joint):
    if pi is None or case['ref'] is None:
        return np.inf
    uv = warp(pi, case['patch'], cams[0], cams[1])
    if uv is None:
        return np.inf
    photo = zncc(case['ref'], bilinear(cams[1]['gray'], uv))
    if not np.isfinite(photo):
        return np.inf
    if not joint:
        return photo/TAU
    xi = np.dot(uv[40], case['axis'])
    vals = case['q']*np.exp(-.5*((xi-case['xi'])/SIGMA)**2)
    support = vals.sum()
    return float(photo/TAU-np.log(support)) if support > 0 else np.inf


def heldout(pi, case, cams):
    if pi is None or case['ref'] is None:
        return None
    uv = warp(pi, case['patch'], cams[0], cams[2])
    value = zncc(case['ref'], bilinear(cams[2]['gray'], uv)) if uv is not None else np.nan
    return float(value) if np.isfinite(value) else None


def derivatives(f, p, h=1e-4):
    # Central finite differences of the complete nonsmooth bilinear-image objective.
    e = np.eye(3)*h
    f0 = f(p)
    if not np.isfinite(f0):
        return None, None
    grad = np.empty(3); H = np.empty((3, 3))
    for j in range(3):
        fp, fm = f(p+e[j]), f(p-e[j])
        if not np.isfinite(fp+fm): return None, None
        grad[j] = (fp-fm)/(2*h)
        H[j,j] = (fp-2*f0+fm)/h**2
        for k in range(j):
            vals = [f(p+e[j]+e[k]), f(p+e[j]-e[k]), f(p-e[j]+e[k]), f(p-e[j]-e[k])]
            if not np.isfinite(vals).all(): return None, None
            H[j,k] = H[k,j] = (vals[0]-vals[1]-vals[2]+vals[3])/(4*h*h)
    return grad, H


def support_gradient_check():
    """Derivative test of the accepted-depth/weight path, not Gaussian parameters."""
    xi = np.array([2.1, 3.4, 4.0], float)
    logits = np.log([.2, .3, .5])
    target = 3.1
    def value(x, l):
        q = np.exp(l-l.max()); q /= q.sum()
        return -np.log(np.sum(q*np.exp(-.5*((target-x)/SIGMA)**2)))
    q = np.exp(logits-logits.max()); q /= q.sum()
    responsibility = q*np.exp(-.5*((target-xi)/SIGMA)**2)
    responsibility /= responsibility.sum()
    analytic_x = responsibility*(xi-target)/SIGMA**2
    analytic_l = q-responsibility
    h = 1e-6
    numeric_x = np.array([(value(xi+np.eye(3)[j]*h,logits)-value(xi-np.eye(3)[j]*h,logits))/(2*h) for j in range(3)])
    numeric_l = np.array([(value(xi,logits+np.eye(3)[j]*h)-value(xi,logits-np.eye(3)[j]*h))/(2*h) for j in range(3)])
    err = max(np.max(np.abs(analytic_x-numeric_x)), np.max(np.abs(analytic_l-numeric_l)))
    assert err < 1e-8
    return dict(max_absolute_error=float(err), xi_gradient=analytic_x.tolist(), logit_gradient=analytic_l.tolist())


def synthetic_check():
    K = np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    a = dict(K=K,R=np.eye(3),t=np.zeros(3))
    b = dict(K=K,R=np.eye(3),t=np.array([-.1,0,0]))
    uv = warp(np.array([0,0,.5]),np.array([[50.,50.],[51.,50.]]),a,b)
    error = float(np.max(np.abs(uv-np.array([[45.,50.],[46.,50.]]))))
    assert error < 1e-12
    assert abs(zncc(np.arange(81.),np.arange(81.)*2+5)) < 1e-12
    assert np.isnan(zncc(np.ones(81),np.arange(81.)))
    return dict(max_warp_error_px=error, zncc_affine_invariant=True, textureless_invalid=True)


def refine(start, case, cams, joint):
    p = start.copy()
    fun = lambda v: cost(v, case, cams, joint)
    for step in range(8):
        g, H = derivatives(fun, p)
        if g is None: return None
        eig = np.linalg.eigvalsh(H)
        if np.linalg.norm(g) < 1e-4 and eig[0] > 1e-6:
            return dict(pi=p, steps=step, hessian_eigen=eig.tolist(), cost=fun(p))
        damping = max(1e-4, 1e-4-eig[0])
        direction = np.linalg.solve(H+damping*np.eye(3), g)
        accepted = False
        for damp in (1, .5, .25, .125, .0625):
            trial = p-damp*direction
            if fun(trial) < fun(p): p = trial; accepted = True; break
        if not accepted: return None
    g, H = derivatives(fun, p)
    if g is None or np.linalg.norm(g) >= 1e-4 or np.linalg.eigvalsh(H)[0] <= 1e-6:
        return None
    return dict(pi=p, steps=8, hessian_eigen=np.linalg.eigvalsh(H).tolist(), cost=fun(p))


def solve(case, cams, joint):
    # A fixed 129-position epipolar profile seeds at most four local basins.
    lo = max(.2, float(case['d'].min())*.9)
    hi = float(case['d'].max())*1.1
    c0, c1 = cams[:2]
    transform = c1['R'] @ c0['R'].T
    a = transform @ case['ray']
    b = c1['t']-transform @ c0['t']
    axis = case['axis']
    A = axis[0]*c1['K'][0,0]*a[0] + axis[1]*c1['K'][1,1]*a[1]
    B = axis[0]*c1['K'][0,0]*b[0] + axis[1]*c1['K'][1,1]*b[1]
    C = axis[0]*c1['K'][0,2] + axis[1]*c1['K'][1,2]
    def position(depth): return (A*depth+B)/(a[2]*depth+b[2])+C
    locations = np.linspace(position(lo), position(hi), 129)-C
    depths = (B-locations*b[2])/(locations*a[2]-A)
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        return None, 0, 0
    starts = [plane(v, case['native_normal'], case['ray']) for v in depths]
    vals = np.array([cost(p, case, cams, joint) for p in starts])
    minima = [j for j in range(1,128) if np.isfinite(vals[j-1:j+2]).all() and vals[j] < vals[j-1] and vals[j] < vals[j+1]]
    minima.sort(key=lambda j: vals[j])
    branches = []
    for j in minima[:4]:
        branch = refine(starts[j], case, cams, joint)
        if branch is not None and all(np.linalg.norm(branch['pi']-old['pi']) > 1e-3 for old in branches):
            branches.append(branch)
    if len(branches) != 1:
        return None, len(branches), len(minima)
    return branches[0], len(branches), len(minima)


def evaluate_scan(scan):
    f, meta, cams, hashes = load_scan(scan)
    rng = np.random.default_rng(SEED+scan)
    indices = sorted(rng.choice(len(f['pixel_xy']), size=32, replace=False).tolist())
    rows = []
    for i in indices:
        start = time.perf_counter()
        case = ray_case(f, i, cams)
        row = dict(index=i, valid_input=case is not None)
        if case is not None:
            baseline = plane(case['native_depth'], case['native_normal'], case['ray'])
            row['native_heldout'] = heldout(baseline, case, cams)
            row['branches'] = {}
            for name, joint in [('photo',False),('joint',True)]:
                branch, count, minima = solve(case, cams, joint)
                row['branches'][name] = dict(reliable=count==1, count=count, grid_minima=minima,
                    heldout=heldout(branch['pi'], case, cams) if branch else row['native_heldout'],
                    fallback=branch is None, cost=float(branch['cost']) if branch else None,
                    pi=branch['pi'].tolist() if branch else None)
        row['seconds'] = time.perf_counter()-start
        rows.append(row)
    return dict(scan=scan, rows=rows, hashes=hashes, source_commit=meta['git_commit'],
        normal_semantics=meta['normal_semantics'], trace_model_sha256=meta['model_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(status='partial', synthetic=synthetic_check(), support_gradient=support_gradient_check(),
        parameters=dict(seed=SEED, rays_per_scan=32, sigma_px=SIGMA,tau=TAU,
        patch='9x9 original pixels', profile_positions=129, max_branches=4, max_newton_steps=8), scans=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scan in (37,40):
        result['scans'].append(evaluate_scan(scan))
        args.output.write_text(json.dumps(result, indent=2),encoding='utf8')
    for item in result['scans']:
        rows = item['rows']
        def stats(name):
            values = [r.get('native_heldout') if name == 'native' else r['branches'][name]['heldout']
                      for r in rows if r['valid_input']]
            valid = [v for v in values if v is not None]
            return dict(valid_count=len(valid), denominator=32,
                mean=float(np.mean(valid)) if valid else None,
                median=float(np.median(valid)) if valid else None)
        item['summary'] = dict(total_rays=32, native_heldout_valid=sum(r.get('native_heldout') is not None for r in rows),
            photo_reliable=sum(r['branches']['photo']['reliable'] for r in rows if r['valid_input']),
            joint_reliable=sum(r['branches']['joint']['reliable'] for r in rows if r['valid_input']),
            photo_fallback=sum(r['branches']['photo']['fallback'] for r in rows if r['valid_input']),
            joint_fallback=sum(r['branches']['joint']['fallback'] for r in rows if r['valid_input']),
            seconds_per_ray_mean=float(np.mean([r['seconds'] for r in rows])),
            heldout={name:stats(name) for name in ('native','photo','joint')})
        summary = item['summary']
        summary['joint_reliable_improved'] = sum(r['branches']['joint']['reliable']
            and r['native_heldout'] is not None
            and r['branches']['joint']['heldout'] is not None
            and r['branches']['photo']['heldout'] is not None
            and r['branches']['joint']['heldout'] < min(r['native_heldout'],r['branches']['photo']['heldout'])
            for r in rows if r['valid_input'])
        held = summary['heldout']
        summary['gate_passed'] = bool(summary['joint_reliable_improved'] >= 8
            and all(held[name]['valid_count'] == 32 for name in held)
            and all(held['joint'][metric] < min(held['native'][metric],held['photo'][metric])
                    for metric in ('mean','median')))
    result['gate_passed'] = all(s['summary']['gate_passed'] for s in result['scans'])
    result['limitations'] = ('The observed images are training views of the frozen original model; view2 is held out '
        'from the local solver only. No GT reconstruction metric or Gaussian parameter backpropagation. '
        'Host warp_patch_ncc uses squared signed correlation and a 5e-6 variance gate, whereas this '
        'probe uses signed 1-ZNCC on grayscale with variance >1e-8; native, photo and joint are '
        'compared under this same probe metric. No reliable isolated converged joint branch was '
        'found, so implicit-solution derivatives were not evaluated. The 32-ray denominator is '
        'retained; unavailable photometric costs are null and invalidate the pass gate.')
    result['status']='complete'
    args.output.write_text(json.dumps(result, indent=2),encoding='utf8')


if __name__ == '__main__': main()
