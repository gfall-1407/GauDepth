"""Conservative local scan-plane references for saved G1 rays; not pixel GT."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree, ConvexHull, QhullError
from plyfile import PlyData

SCANS = (24, 65, 105, 110)
PROTOCOL = dict(radius_px=.75, nearest_px=.25, front_gap_mm=2., min_points=6,
                max_rms_mm=.1, max_planarity=.05, min_incidence=.2,
                max_extrapolation_mm=.25)
REFERENCE_RADII_PX = (.75, 1.5, 3.)


def stable_reference(refs):
    """Coverage repair uses GT only, with fixed cross-footprint agreement."""
    a, b = refs[1:]
    if a['status'] == b['status'] == 'accepted':
        gap = abs(a['reference_mm'] - b['reference_mm'])
        if gap <= .1:
            result = dict(a)
        else:
            result = dict(a, status='cross_scale_disagreement', reference_mm=None)
    else:
        gap = None
        result = dict(a, status='missing_multiscale_reference', reference_mm=None)
    result['cross_scale_gap_mm'] = gap
    for label, ref in zip(('075', '15', '30'), refs):
        result[f'status_radius_{label}'] = ref['status']
        result[f'reference_radius_{label}_mm'] = ref['reference_mm']
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def local_reference(points_mm, uv, pixel, focal, center):
    """Uses only projected scan points. No prediction enters the acceptance gates."""
    info = dict(status='sparse', points=len(points_mm), front_points=0,
                reference_mm=None, rms_mm=None, rms_z_mm=None,
                planarity=None, incidence=None, projected_components=0)
    if len(points_mm) < PROTOCOL['min_points']:
        return info
    order = np.argsort(points_mm[:, 2], kind='stable')
    points_mm, uv = points_mm[order], uv[order]
    cuts = np.flatnonzero(np.diff(points_mm[:, 2]) > PROTOCOL['front_gap_mm']) + 1
    info['projected_components'] = len(cuts) + 1
    end = int(cuts[0]) if len(cuts) else len(points_mm)
    points_mm, uv = points_mm[:end], uv[:end]
    info['front_points'] = end
    if end < PROTOCOL['min_points']:
        info['status'] = 'front_sparse'
        return info
    if np.linalg.norm(uv - pixel, axis=1).min() > PROTOCOL['nearest_px']:
        info['status'] = 'no_close_scan_projection'
        return info
    try:
        hull = ConvexHull(uv)
    except QhullError:
        info['status'] = 'degenerate_projection'
        return info
    if np.any(hull.equations[:, :2] @ pixel + hull.equations[:, 2] > 1e-9):
        info['status'] = 'pixel_outside_scan_hull'
        return info
    centroid = points_mm.mean(0)
    delta = points_mm - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(delta.T @ delta / end)
    normal = eigenvectors[:, 0]
    rms = float(np.sqrt(max(eigenvalues[0], 0.)))
    planarity = float(max(eigenvalues[0], 0.) / max(eigenvalues[1], 1e-30))
    info.update(rms_mm=rms, planarity=planarity)
    if eigenvalues[1] <= 1e-8 or rms > PROTOCOL['max_rms_mm'] or planarity > PROTOCOL['max_planarity']:
        info['status'] = 'nonplanar_scan_patch'
        return info
    ray = np.r_[(pixel - center) / focal, 1.]
    denominator = float(normal @ ray)
    incidence = abs(denominator) / np.linalg.norm(ray)
    info['incidence'] = float(incidence)
    if incidence < PROTOCOL['min_incidence']:
        info['status'] = 'grazing_scan_patch'
        return info
    depth = float((normal @ centroid) / denominator)
    if not np.isfinite(depth) or depth <= 0:
        info['status'] = 'nonpositive_reference'
        return info
    margin = PROTOCOL['max_extrapolation_mm']
    if depth < points_mm[:, 2].min() - margin or depth > points_mm[:, 2].max() + margin:
        info['status'] = 'depth_extrapolation'
        return info
    info.update(status='accepted', reference_mm=depth,
                rms_z_mm=float(rms / abs(denominator)))
    return info


def self_check():
    a = np.linspace(-.7, .7, 9)
    uv = np.array([(x, y) for x in a for y in a])
    q = np.c_[uv / 100., np.ones(len(uv))]
    z = 100. / (1. - .1 * q[:, 0])
    front = q * z[:, None]
    r = local_reference(front, uv, np.zeros(2), np.array([100., 100.]), np.zeros(2))
    assert r['status'] == 'accepted', r
    np.testing.assert_allclose(r['reference_mm'], 100., atol=1e-9)
    r = local_reference(np.r_[front, q * (z[:, None] + 100.)], np.r_[uv, uv],
                        np.zeros(2), np.array([100., 100.]), np.zeros(2))
    assert r['status'] == 'accepted' and r['projected_components'] == 2, r
    np.testing.assert_allclose(r['reference_mm'], 100., atol=1e-9)
    assert local_reference(front[:3], uv[:3], np.zeros(2), np.ones(2), np.zeros(2))['status'] == 'sparse'
    outside = local_reference(front, uv, np.array([1., 0.]), np.array([100., 100.]), np.zeros(2))
    assert outside['status'] != 'accepted'
    good = dict(r, status='accepted', reference_mm=100.)
    assert stable_reference([good, good, good])['status'] == 'accepted'
    shifted = dict(good, reference_mm=100.2)
    assert stable_reference([good, good, shifted])['reference_mm'] is None
    print('scan-plane, front occlusion, sparse and extrapolation checks passed')


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Independent output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True).strip():
        raise ValueError('Commit the source before a data run')
    start = time.monotonic()
    inputs, rows, views = [], [], []
    sampled = {}
    for path in sorted(args.ray_root.glob('scan*/metadata.json')):
        m = read_json(path)
        if m.get('hostname') != 'autodl-container-d6a7459179-693ab6ee' or not m.get('model_sha256'):
            continue
        scan = int(Path(m['model'].replace('\\', '/')).parts[-4].replace('scan', ''))
        if scan not in SCANS:
            continue
        for view in m['views']:
            key = (scan, view['image_name'])
            if key in sampled:
                raise ValueError(f'Duplicate saved view: {key}')
            sampled[key] = (path, m, view)
    if len(sampled) != 12:
        raise ValueError(f'Expected 12 pre-existing saved views, got {len(sampled)}')
    for scan in SCANS:
        root = args.inputs / f'scan{scan}'
        old = read_json(args.influence_root / f'scan{scan}/metadata.json')
        gt_path = root / f'stl{scan:03d}_total.ply'
        camera_path = root / 'cameras.npz'
        plane_path = root / 'ObsMask' / f'Plane{scan}.mat'
        assert sha(gt_path) == old['gt_sha256']
        assert sha(camera_path) == old['cameras_npz_sha256']
        with np.load(camera_path) as camera:
            transform = camera['scale_mat_0'].astype(float)
        scale = float(transform[0, 0])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3) * scale, atol=1e-6)
        vertex = PlyData.read(gt_path)['vertex']
        gt = np.column_stack([vertex[k] for k in ('x', 'y', 'z')]).astype(float)
        plane = loadmat(plane_path)['P'].reshape(-1)
        keep = np.isfinite(gt).all(1) & ((gt @ plane[:3] + plane[3]) > 0)
        inputs.append(dict(scan=scan, gt_sha256=sha(gt_path), camera_sha256=sha(camera_path),
                           plane_sha256=sha(plane_path), gt_total=len(gt), gt_above=int(keep.sum()),
                           scale_to_mm=scale))
        scene = (gt[keep] - transform[:3, 3]) / scale
        del vertex, gt
        for (s, name), (meta_path, meta, view) in sorted(sampled.items()):
            if s != scan:
                continue
            assert meta['model_sha256'] == old['model_sha256']
            data_path = meta_path.parent / view['file']
            assert sha(data_path) == view['npz_sha256']
            world_view = np.asarray(view['world_view_transform'], float).T
            rot, trans = world_view[:3, :3], world_view[:3, 3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=2e-6)
            np.testing.assert_allclose(-rot.T @ trans, view['camera_center'], atol=2e-6)
            h, w = view['image_size']
            focal = np.asarray(view['focal'], float)
            center = np.array([(w - 1) / 2., (h - 1) / 2.])
            cam_mm = (scene @ rot.T + trans) * scale
            positive = cam_mm[:, 2] > 0
            cam_mm = cam_mm[positive]
            uv = cam_mm[:, :2] / cam_mm[:, 2, None] * focal + center
            inside = ((uv >= [-1., -1.]) & (uv <= [w, h])).all(1)
            cam_mm, uv = cam_mm[inside], uv[inside]
            tree = cKDTree(uv)
            with np.load(data_path) as rays:
                pixels = rays['pixel_xy'].astype(float)
            radii = REFERENCE_RADII_PX if args.multiscale else (PROTOCOL['radius_px'],)
            neighborhoods = [tree.query_ball_point(pixels, radius) for radius in radii]
            statuses = collections.Counter()
            for ri, pixel in enumerate(pixels):
                refs = []
                for neighborhood in neighborhoods:
                    neighbors = np.asarray(neighborhood[ri], dtype=int)
                    refs.append(local_reference(cam_mm[neighbors], uv[neighbors], pixel, focal, center))
                ref = stable_reference(refs) if args.multiscale else refs[0]
                statuses[ref['status']] += 1
                rows.append(dict(scan=scan, image_name=name, ray_index=ri,
                                 pixel_x=int(pixel[0]), pixel_y=int(pixel[1]), **ref))
            views.append(dict(scan=scan, image_name=name, sampled_rays=len(pixels),
                              accepted=statuses['accepted'], statuses=dict(statuses)))
            print(json.dumps(views[-1]), flush=True)
            del tree, cam_mm, uv
            if time.monotonic() - start > 600:
                raise TimeoutError('Reference batch exceeded its 600-second bound')
        del scene
    write_csv(out / 'ray_reference.csv', rows)
    write_csv(out / 'view_coverage.csv', [{k: v for k, v in x.items() if k != 'statuses'} for x in views])
    report = dict(commit=commit, protocol=PROTOCOL, reference_radii_px=list(radii),
                  multiscale=args.multiscale,
                  stable_reference_rule='Both 1.5px and 3px references must independently pass identical plane gates and differ by <=0.1mm; use 1.5px depth. This coverage check was added after the 0.75px-only run had 30/3072 coverage, before inspecting prediction errors.',
                  inputs=inputs, views=views,
                  sampled_rays=len(rows), accepted=sum(x['status'] == 'accepted' for x in rows),
                  interpretation='Local front scan-plane reference on a conservative coverage subset; not exact pixel GT, not whole-image coverage; thresholds fixed before predictions; above-plane GT only.')
    (out / 'reference_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--multiscale', action='store_true')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--ray-root', type=Path)
    p.add_argument('--influence-root', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.self_check:
        self_check()
    else:
        if any(getattr(args, k) is None for k in ('inputs', 'ray_root', 'influence_root', 'output')):
            p.error('Data runs require inputs, ray-root, influence-root and output')
        run(args)


if __name__ == '__main__':
    main()
