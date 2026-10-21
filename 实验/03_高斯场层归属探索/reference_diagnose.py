"""Read-only CPU comparison of sampled Gaussian contributions with expected-mesh hits.

The mesh is an in-model reference, not independent ground truth. Band membership
is descriptive only and must not be interpreted as physical surface labels.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from analyze import group_ray


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def camera_ray(stored_w2c, center, x, y, width, height, fx, fy):
    """Return world origin, non-unit world direction, and camera ray (z=1)."""
    m = np.asarray(stored_w2c, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError('world_view_transform must be 4x4')
    d_cam = np.array([(x - (width - 1) / 2) / fx,
                      (y - (height - 1) / 2) / fy, 1.], dtype=np.float64)
    return np.asarray(center, dtype=np.float64), m[:3, :3] @ d_cam, d_cam


def world_to_camera(stored_w2c, point):
    m = np.asarray(stored_w2c, dtype=np.float64)
    return (m.T @ np.r_[np.asarray(point, dtype=np.float64), 1.])[:3]


def project_camera(point_camera, width, height, fx, fy):
    p = np.asarray(point_camera, dtype=np.float64)
    if p[2] <= 0:
        return None
    return np.array([fx * p[0] / p[2] + (width - 1) / 2,
                     fy * p[1] / p[2] + (height - 1) / 2])


def grouped_mass_metrics(weights, labels):
    """Count groups needed for 90% of all near-band mass; unknown stays denominator."""
    w, lab = np.asarray(weights, float), np.asarray(labels)
    good = np.isfinite(w) & (w > 0)
    total = float(w[good].sum())
    unknown = float(w[good & (lab < 0)].sum())
    masses = np.array([w[good & (lab == k)].sum() for k in np.unique(lab[lab >= 0])], float)
    masses = np.sort(masses)[::-1]
    max_fraction = float(masses[0] / total) if total > 0 and len(masses) else (0. if total > 0 else None)
    if total > 0 and len(masses) and masses.sum() >= .9 * total:
        n90 = int(np.searchsorted(np.cumsum(masses), .9 * total, side='left') + 1)
    else:
        n90 = None
    return dict(near_mass=total, undetermined_near_mass=unknown,
                undetermined_near_mass_fraction=unknown / total if total > 0 else None,
                groups_touching_band=int(len(masses)), max_group_near_mass_fraction=max_fraction,
                groups_for_90pct_near_mass=n90)


def weighted_normal_angle(normals_camera, weights, valid, band_mask, reference_normal_camera):
    n = np.asarray(normals_camera, float)
    w = np.asarray(weights, float)
    valid = np.asarray(valid, bool)
    lengths = np.linalg.norm(n, axis=1)
    good = band_mask & valid & np.isfinite(n).all(axis=1) & (lengths > 1e-8) & np.isfinite(w) & (w > 0)
    if not good.any():
        return None
    # Reference surface normals are unoriented; mesh winding cannot change the angle.
    ref = np.asarray(reference_normal_camera, dtype=float)
    ref_length = np.linalg.norm(ref)
    if not np.isfinite(ref).all() or ref_length <= 1e-8:
        return None
    dots = np.abs((n[good] / lengths[good, None]) @ (ref / ref_length))
    angles = np.degrees(np.arccos(np.clip(dots, -1., 1.)))
    return float(np.average(angles, weights=w[good]))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_scan(metadata):
    paths = [metadata.get('model', ''), metadata.get('model_path', ''),
             metadata.get('effective_model_args', {}).get('model_path', ''),
             metadata.get('args', {}).get('model_path', '')]
    scans = set()
    for p in paths:
        match = re.search(r'(scan\d+)', str(p), re.I)
        if match:
            scans.add(match.group(1).lower())
    if len(scans) != 1:
        raise ValueError(f'Missing or inconsistent scan identity: {scans}')
    return scans.pop()


def run(inputs, model_root, output, voxel=.002, bands=(2, 4, 8), eta=.01, angle=15.):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be new or empty: {output}')
    files = []
    for p in inputs:
        p = Path(p)
        files.extend(sorted(p.glob('view_*.npz')) if p.is_dir() else [p])
    files = sorted(set(p.resolve() for p in files))
    if not files:
        raise ValueError('No view_*.npz inputs found')
    output.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError('Open3D is required for real mesh ray casting; pure helpers remain importable') from e
    root = Path(model_root)
    scenes, mesh_hashes = {}, {}
    ray_rows, summary_map, manifests = [], {}, []
    for path in files:
        meta_path = path.parent / 'metadata.json'
        if not meta_path.is_file():
            raise FileNotFoundError(f'Missing metadata: {meta_path}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        scan = find_scan(meta)
        mesh_path = root / scan / 'recon_post.ply'
        if not mesh_path.is_file():
            raise FileNotFoundError(f'Missing expected-z reference mesh: {mesh_path}')
        if mesh_path not in scenes:
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(mesh_path)))
            if mesh.vertex.positions.shape[0] == 0 or mesh.triangle.indices.shape[0] == 0:
                raise ValueError(f'Empty mesh: {mesh_path}')
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh)
            scenes[mesh_path] = scene
            mesh_hashes[str(mesh_path)] = sha256(mesh_path)
        scene = scenes[mesh_path]
        view_index = int(path.stem.split('_')[-1])
        view = next((v for v in meta.get('views', []) if int(v['view_index']) == view_index), None)
        if view is None:
            raise ValueError(f'No metadata view_index={view_index} for {path}')
        if sha256(path) != view['npz_sha256']:
            raise ValueError(f'NPZ hash differs from sampler metadata: {path}')
        H, W = map(int, view['image_size'])
        fx, fy = map(float, view['focal'])
        mat = np.asarray(view['world_view_transform'], dtype=np.float64)
        center = np.asarray(view['camera_center'], dtype=np.float64)
        with np.load(path, allow_pickle=False) as z:
            off, depth, weight = z['offsets'], z['depth'], z['weight']
            normals, valid = z['normal'], z['normal_valid']
            qual, xy, replay = z['qualitative'], z['pixel_xy'], z['replay']
            nr = len(off) - 1
            if off[0] != 0 or off[-1] != len(depth) or np.any(np.diff(off) < 0):
                raise ValueError(f'Invalid ragged offsets: {path}')
            if any(len(a) != len(depth) for a in (weight, normals, valid)) or len(qual) != nr or len(xy) != nr:
                raise ValueError(f'Invalid NPZ schema: {path}')
            for r in range(nr):
                x, y = map(int, xy[r])
                origin, direction, dcam = camera_ray(mat, center, x, y, W, H, fx, fy)
                ray = np.r_[origin, direction].astype(np.float32)[None, :]
                hit = scene.cast_rays(o3d.core.Tensor(ray))
                t_hit = float(hit['t_hit'].numpy()[0])
                hit_id = int(hit['geometry_ids'].numpy()[0])
                has_hit = math.isfinite(t_hit) and hit_id != 0xFFFFFFFF
                sl = slice(int(off[r]), int(off[r + 1]))
                d, w, n_cam, nv = depth[sl].astype(float), weight[sl].astype(float), normals[sl].astype(float), valid[sl]
                # Sampler normals already use camera coordinates (verified by host replay).
                partitions = {geom: group_ray(d, n_cam, nv, eta, angle, geom) for geom in (False, True)}
                if has_hit:
                    point = origin + t_hit * direction
                    p_cam = world_to_camera(mat, point)
                    z_ref = float(p_cam[2])
                    projected = project_camera(p_cam, W, H, fx, fy)
                    pixel_error = float(np.linalg.norm(projected - [x, y])) if projected is not None else None
                    camera_ray_t_error = float(abs(t_hit - z_ref))
                    # A forward z=1 ray has t equal to camera-z, up to mesh/raycast float32 error.
                    accepted = z_ref > 0 and pixel_error is not None and pixel_error <= 2e-3 and camera_ray_t_error <= max(2e-3, abs(z_ref) * 2e-5)
                    status = 'hit' if accepted else 'hit_replay_mismatch'
                else:
                    z_ref, pixel_error, camera_ray_t_error, accepted = None, None, None, False
                    status = 'miss'
                expected_z = float(replay[r, 0]) if np.isfinite(replay[r, 0]) else None
                for half_band in bands:
                    half_world = float(half_band * voxel)
                    # Ray parameter is z along camera ray with z=1, so z half-width equals world-band/||d_cam||.
                    half_z = half_world / float(np.linalg.norm(dcam))
                    if accepted:
                        near = np.isfinite(d) & np.isfinite(w) & (w > 0) & (np.abs(d - z_ref) <= half_z)
                        for geom in (False, True):
                            labels, flags = partitions[geom]
                            metrics = grouped_mass_metrics(w[near], labels[near])
                            primitive_n_world = hit['primitive_normals'].numpy()[0].astype(np.float64)
                            primitive_n_cam = mat[:3, :3].T @ primitive_n_world
                            normal_angle = weighted_normal_angle(n_cam, w, nv, near, primitive_n_cam)
                            row = dict(scan=scan, view=path.stem, ray=r, pixel_x=x, pixel_y=y,
                                       qualitative=bool(qual[r]), status=status, expected_z=expected_z,
                                       reference_z=z_ref, expected_minus_reference_z=(expected_z-z_ref) if expected_z is not None else None,
                                       expected_reference_abs_error=abs(expected_z-z_ref) if expected_z is not None else None,
                                       t_hit=t_hit, t_camera_z_abs_error=camera_ray_t_error,
                                       replay_pixel_error=pixel_error, voxel=voxel, half_band_voxels=half_band,
                                       half_band_world=half_world, half_band_camera_z=half_z,
                                       near_member_count=int(near.sum()), original_near_weight=float(w[near].sum()),
                                       method='depth_normal' if geom else 'depth', normal_angle_weighted_mean_deg=normal_angle,
                                       normal_angle_note='unoriented angle; descriptive, not confidence', **metrics,
                                       candidate_undetermined_count=int(np.sum(near & (labels < 0))),
                                       candidate_groups_touching_band=int(len(set(labels[near]) - {-1})),
                                       eta=eta, angle_degrees=angle, **flags)
                            ray_rows.append(row)
                            if not bool(qual[r]):
                                key = (scan, path.stem, half_band, row['method'])
                                summary_map.setdefault(key, []).append(row)
                    else:
                        for method in ('depth', 'depth_normal'):
                            row = dict(scan=scan, view=path.stem, ray=r, pixel_x=x, pixel_y=y,
                                       qualitative=bool(qual[r]), status=status, expected_z=expected_z,
                                       reference_z=None, expected_minus_reference_z=None,
                                       expected_reference_abs_error=None, t_hit=t_hit if math.isfinite(t_hit) else None,
                                       t_camera_z_abs_error=None, replay_pixel_error=None, voxel=voxel,
                                       half_band_voxels=half_band, half_band_world=half_world,
                                       half_band_camera_z=half_z, near_member_count=None, original_near_weight=None,
                                       method=method, normal_angle_weighted_mean_deg=None,
                                       normal_angle_note='unoriented angle; descriptive, not confidence',
                                       near_mass=None, undetermined_near_mass=None,
                                       undetermined_near_mass_fraction=None, groups_touching_band=None,
                                       max_group_near_mass_fraction=None, groups_for_90pct_near_mass=None,
                                       candidate_undetermined_count=None, candidate_groups_touching_band=None,
                                       eta=eta, angle_degrees=angle, ambiguous_insertions=None,
                                       compatible_cross_group_pairs=None)
                            ray_rows.append(row)
                            if not bool(qual[r]):
                                key = (scan, path.stem, half_band, method)
                                summary_map.setdefault(key, []).append(row)
        manifests.append(dict(npz=str(path), npz_sha256=sha256(path), metadata=str(meta_path),
                              metadata_sha256=sha256(meta_path), scan=scan, mesh=str(mesh_path),
                              mesh_sha256=mesh_hashes[str(mesh_path)], source_sha256=sha256(Path(__file__)),
                              analyze_sha256=sha256(Path(__file__).with_name('analyze.py'))))
    write_csv(output / 'rays.csv', ray_rows)
    summaries = []
    for (scan, view, band, method), rows in sorted(summary_map.items()):
        metrics = ['expected_reference_abs_error', 'near_member_count', 'near_mass',
                   'undetermined_near_mass_fraction', 'groups_touching_band',
                   'max_group_near_mass_fraction', 'groups_for_90pct_near_mass',
                   'normal_angle_weighted_mean_deg']
        summaries.append(dict(scan=scan, view=view, half_band_voxels=band, method=method,
                              random_rays=len(rows), mesh_hits=sum(r['status'] == 'hit' for r in rows),
                              mesh_misses=sum(r['status'] == 'miss' for r in rows),
                              replay_mismatches=sum(r['status'] == 'hit_replay_mismatch' for r in rows),
                              **{f'mean_{k}': (float(np.mean([r[k] for r in rows if r.get(k) is not None]))
                                  if any(r.get(k) is not None for r in rows) else None) for k in metrics}))
    write_csv(output / 'summary.csv', summaries)
    payload = dict(inputs=manifests, settings=dict(voxel=voxel, half_band_voxels=list(bands), eta=eta,
                   angle_degrees=angle, raycast='Open3D t.geometry.RaycastingScene, float32 rays',
                   projection_pixel_tolerance=2e-3, hit_t_camera_z_tolerance='max(2e-3, |z|*2e-5)'),
                   interpretation='Same-model expected-z reference mesh; bands are descriptive only. No physical membership labels, precision/recall, new depth, fusion, or training.',
                   summaries=summaries, qualitative_rows_excluded_from_summary=True)
    (output / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def self_check():
    # Pure CPU camera/projection and voxel conversion sanity checks.
    identity = np.eye(4)
    origin, direction, dcam = camera_ray(identity, [0, 0, 0], 6, 4, 11, 9, 10, 10)
    assert np.allclose(direction, [.1, 0, 1]) and np.allclose(dcam, [.1, 0, 1])
    p = origin + 2.5 * direction
    pc = world_to_camera(identity, p)
    assert np.allclose(project_camera(pc, 11, 9, 10, 10), [6, 4])
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    stored = np.eye(4); stored[:3, :3] = rot
    center = np.array([1., 2., 3.])
    stored[3, :3] = -center @ rot
    _, dw, dc = camera_ray(stored, center, 5, 4, 11, 9, 10, 10)
    pw = center + 3. * dw
    assert np.allclose(world_to_camera(stored, pw), dc * 3.)
    assert np.allclose(project_camera(world_to_camera(stored, pw), 11, 9, 10, 10), [5, 4])
    world_normal = np.array([1., 0., 0.])
    camera_normal = stored[:3, :3].T @ world_normal
    assert weighted_normal_angle([camera_normal], [1.], [True], np.array([True]), camera_normal) == 0.
    assert find_scan({'model': '/models/scan24/point_cloud.ply',
                      'effective_model_args': {'model_path': '/models/scan24'}}) == 'scan24'
    assert [b * .002 for b in (2, 4, 8)] == [.004, .008, .016]
    m = grouped_mass_metrics([.5, .3, .2], [0, 1, -1])
    assert m['groups_for_90pct_near_mass'] is None and m['undetermined_near_mass_fraction'] == .2
    assert grouped_mass_metrics([.5, .3, .2], [0, 0, 1])['groups_for_90pct_near_mass'] == 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, help='NPZ directory or file; repeatable')
    parser.add_argument('--model-root', required=True, help='Directory containing scan24/recon_post.ply, etc.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--voxel', type=float, default=.002)
    parser.add_argument('--band', type=float, action='append', help='Half-width in voxels; defaults to 2, 4, 8')
    parser.add_argument('--eta', type=float, default=.01)
    parser.add_argument('--angle', type=float, default=15.)
    parser.add_argument('--self-check', action='store_true', help='Run pure CPU geometry/metric checks')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print('pure CPU self-check passed')
    else:
        result = run(args.input, Path(args.model_root), Path(args.output), args.voxel,
                     args.band or (2, 4, 8), args.eta, args.angle)
        print(json.dumps(result, ensure_ascii=False))
