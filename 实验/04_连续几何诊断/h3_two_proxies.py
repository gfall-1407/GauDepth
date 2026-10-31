"""Fixed H3 native-median versus depth50 readout comparison on two scan proxies.

Neither one-way 3D nearest point nor frontmost scan point inside one pixel is
certified visible first-hit geometry. The latter has no expanded search window.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

import all_scan_proxy as source


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def statistics(rows, key):
    a = np.asarray([row[key] for row in rows if np.isfinite(row[key])], float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else np.nan,
                median=float(np.median(a)) if len(a) else np.nan,
                p95=float(np.quantile(a, .95)) if len(a) else np.nan)


def pixel_footprint(tree, uv, xy):
    candidates = tree.query_ball_point(xy, np.sqrt(.5))
    return [j for j in candidates if np.all(uv[j]-xy >= -.5) and np.all(uv[j]-xy < .5)]


def check_trace_binding(native_row, trace_sha, key):
    if native_row['trace_sha256'] != trace_sha:
        raise ValueError(f'Native/trace SHA mismatch: {key}')


def summarize(rows, scan, role):
    group = [row for row in rows if row['scan'] == scan and (role == 'all' or row['h3_fixed'])]
    nn = [row for row in group if np.isfinite(row['native_nn_mm']) and np.isfinite(row['depth50_nn_mm'])]
    foot = [row for row in group if np.isfinite(row['native_front_axial_mm']) and np.isfinite(row['depth50_front_axial_mm'])]
    return dict(scan=scan, role=role, rays=len(group), depth50_defined=sum(np.isfinite(row['depth50_mm']) for row in group),
        nearest_paired=len(nn), front_footprint_available=sum(row['front_candidate_n']>0 for row in group),
        front_paired=len(foot), front_missing=len(group)-len(foot),
        native_nn_mean_mm=statistics(nn, 'native_nn_mm')['mean'],
        depth50_nn_mean_mm=statistics(nn, 'depth50_nn_mm')['mean'],
        native_nn_median_mm=statistics(nn, 'native_nn_mm')['median'],
        depth50_nn_median_mm=statistics(nn, 'depth50_nn_mm')['median'],
        native_nn_p95_mm=statistics(nn, 'native_nn_mm')['p95'],
        depth50_nn_p95_mm=statistics(nn, 'depth50_nn_mm')['p95'],
        depth50_nn_wins=sum(row['depth50_nn_mm'] < row['native_nn_mm'] for row in nn),
        native_nn_wins=sum(row['native_nn_mm'] < row['depth50_nn_mm'] for row in nn),
        nn_ties=sum(row['native_nn_mm'] == row['depth50_nn_mm'] for row in nn),
        same_nearest_gt=sum(row['same_nearest_gt'] for row in nn),
        native_nn_projection_median_px=statistics(nn, 'native_nn_projection_px')['median'],
        depth50_nn_projection_median_px=statistics(nn, 'depth50_nn_projection_px')['median'],
        native_front_mean_mm=statistics(foot, 'native_front_axial_mm')['mean'],
        depth50_front_mean_mm=statistics(foot, 'depth50_front_axial_mm')['mean'],
        native_front_median_mm=statistics(foot, 'native_front_axial_mm')['median'],
        depth50_front_median_mm=statistics(foot, 'depth50_front_axial_mm')['median'],
        native_front_p95_mm=statistics(foot, 'native_front_axial_mm')['p95'],
        depth50_front_p95_mm=statistics(foot, 'depth50_front_axial_mm')['p95'],
        depth50_front_wins=sum(row['depth50_front_axial_mm'] < row['native_front_axial_mm'] for row in foot),
        native_front_wins=sum(row['native_front_axial_mm'] < row['depth50_front_axial_mm'] for row in foot),
        front_ties=sum(row['native_front_axial_mm'] == row['depth50_front_axial_mm'] for row in foot),
        native_before_front=sum(row['native_minus_front_mm'] < 0 for row in foot),
        depth50_before_front=sum(row['depth50_minus_front_mm'] < 0 for row in foot))


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    fixed = json.loads(args.fixed_ids.read_text(encoding='utf-8'))
    if fixed['count'] != 2522 or len(fixed['ids']) != 2522:
        raise ValueError('Expected the original fixed 2522 H3 IDs')
    fixed_ids = {tuple(x) for x in fixed['ids']}
    if len(fixed_ids) != 2522:
        raise ValueError('Duplicate H3 IDs')
    if sha(args.native_rays) != fixed['native_csv_sha256']:
        raise ValueError('Native source changed after H3 selection')
    native = read_csv(args.native_rays)
    native_map = {(int(row['scan']), row['view'], int(row['ray'])): row for row in native}
    if len(native_map) != len(native) or len(native) != 19968 or not fixed_ids <= native_map.keys():
        raise ValueError('Native ray set incomplete or duplicated')
    index, records = source.source_files(args.data_root)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    git_status = subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
    if git_status:
        raise ValueError('Commit the experiment source before running')
    rows, inputs = [], []
    recomputed_ids = set()
    start = time.monotonic()
    for scan in source.SCANS:
        items = source.scan_records(records, scan)
        gt_items = source.named(items, f'stl{scan:03d}_total.ply')
        cam_items = source.named(items, 'cameras.npz')
        if len(gt_items) != 1 or not cam_items or any(x['sha256'] != cam_items[0]['sha256'] for x in cam_items):
            raise ValueError(f'GT/camera ambiguous: {scan}')
        gt_item, cam_item = gt_items[0], cam_items[0]
        gt_path, cam_path = args.data_root / gt_item['path'], args.data_root / cam_item['path']
        if sha(gt_path) != gt_item['sha256'] or sha(cam_path) != cam_item['sha256'] or cam_item['sha256'] != fixed['camera_sha256'][str(scan)]:
            raise ValueError(f'GT/camera SHA mismatch: {scan}')
        transform = np.load(cam_path)['scale_mat_0'].astype(float)
        scale = float(transform[0, 0])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3)*scale, atol=1e-6)
        vertex = PlyData.read(gt_path)['vertex']
        gt = np.column_stack([vertex[key] for key in ('x','y','z')]).astype(float)
        original_id = np.arange(len(gt))
        finite = np.isfinite(gt).all(1)
        scene = (gt[finite]-transform[:3, 3])/scale
        original_id = original_id[finite]
        del vertex, gt
        inputs.append(dict(scan=scan, gt_sha256=gt_item['sha256'], camera_sha256=cam_item['sha256'], scale_to_mm=scale))
        for trace, _, view in source.view_records(args.data_root, items):
            path = args.data_root / trace['path']
            if sha(path) != trace['sha256']:
                raise ValueError(f'Trace SHA mismatch: {path}')
            with np.load(path) as data:
                pixel = data['pixel_xy'].astype(float)
            world_view = np.asarray(view['world_view_transform'], float).T
            rot, trans = world_view[:3, :3], world_view[:3, 3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=2e-6)
            height, width = view['image_size']
            focal = np.asarray(view['focal'], float)
            center = np.array([(width-1)/2., (height-1)/2.])
            cam = (scene @ rot.T+trans)*scale
            forward = cam[:, 2] > 0
            cam, ids = cam[forward], original_id[forward]
            uv = cam[:, :2]/cam[:, 2, None]*focal+center
            visible_frame = ((uv >= [-1.,-1.]) & (uv <= [width,height])).all(1)
            cam, uv, ids = cam[visible_frame], uv[visible_frame], ids[visible_frame]
            tree3 = cKDTree(cam)
            tree2 = cKDTree(uv)
            for ray, xy in enumerate(pixel):
                key = (scan, view['image_name'], ray)
                if key not in native_map:
                    raise ValueError(f'Trace/native ID missing: {key}')
                n = native_map[key]
                check_trace_binding(n, trace['sha256'], key)
                native_z = float(n['native_median'])*scale
                depth50_z = float(n['depth50'])*scale
                if np.isfinite(depth50_z) and abs(depth50_z-native_z) > 1.0:
                    recomputed_ids.add(key)
                direction = np.r_[(xy-center)/focal, 1.]
                footprint = pixel_footprint(tree2, uv, xy)
                if footprint:
                    front = min(footprint, key=lambda j: cam[j, 2])
                    front_z = float(cam[front, 2])
                    front_id = int(ids[front])
                    front_offset = float(np.linalg.norm(uv[front]-xy))
                else:
                    front_z, front_id, front_offset = np.nan, -1, np.nan
                row = dict(scan=scan, view=view['image_name'], ray=ray,
                    h3_fixed=key in fixed_ids, pixel_x=int(xy[0]), pixel_y=int(xy[1]),
                    trace_sha256=trace['sha256'], native_mm=native_z, depth50_mm=depth50_z,
                    front_candidate_n=len(footprint), front_z_mm=front_z,
                    front_gt_id=front_id, front_projection_px=front_offset)
                for label, z in (('native', native_z), ('depth50', depth50_z)):
                    valid = np.isfinite(z) and z > 0 and len(cam)>0
                    if valid:
                        dist, j = tree3.query(direction*z, k=1)
                        row[f'{label}_nn_mm'] = float(dist)
                        row[f'{label}_nn_gt_id'] = int(ids[j])
                        row[f'{label}_nn_projection_px'] = float(np.linalg.norm(uv[j]-xy))
                        row[f'{label}_nn_gt_minus_pred_z_mm'] = float(cam[j, 2]-z)
                    else:
                        row[f'{label}_nn_mm'] = np.nan
                        row[f'{label}_nn_gt_id'] = -1
                        row[f'{label}_nn_projection_px'] = np.nan
                        row[f'{label}_nn_gt_minus_pred_z_mm'] = np.nan
                    row[f'{label}_front_axial_mm'] = abs(z-front_z) if valid and np.isfinite(front_z) else np.nan
                    row[f'{label}_minus_front_mm'] = z-front_z if valid and np.isfinite(front_z) else np.nan
                row['same_nearest_gt'] = row['native_nn_gt_id'] >= 0 and row['native_nn_gt_id'] == row['depth50_nn_gt_id']
                rows.append(row)
            print(json.dumps(dict(scan=scan, view=view['image_name'], rays=len(pixel),
                                  h3_fixed=sum(row['scan']==scan and row['view']==view['image_name'] and row['h3_fixed'] for row in rows))), flush=True)
        del scene
    if recomputed_ids != fixed_ids:
        raise ValueError(f'Frozen H3 ID mismatch: {len(recomputed_ids ^ fixed_ids)} IDs differ')
    if len(rows) != 19968 or sum(row['h3_fixed'] for row in rows) != 2522:
        raise ValueError('Incomplete fixed cohort or auxiliary sample')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'rays.csv', rows)
    write_csv(args.output/'scans.csv', [summarize(rows, scan, role) for scan in source.SCANS for role in ('H3','all')])
    (args.output/'metadata.json').write_text(json.dumps(dict(
        commit=commit, script_sha256=sha(Path(__file__)), helper_sha256=sha(Path(source.__file__)),
        fixed_ids_sha256=sha(args.fixed_ids), native_csv_sha256=sha(args.native_rays), source_index_sha256=sha(index),
        scans=list(source.SCANS), rays=len(rows), h3_fixed=2522,
        footprint='Half-open pixel square [u-.5,u+.5) x [v-.5,v+.5), fixed before evaluation; no expansion or prediction-based selection. Frontmost scan point in camera-z.',
        nearest='One-way Euclidean distance from predicted ray point to closest independent DTU scan point in camera-mm; projected offset, signed z relation, and same-point identity recorded.',
        limitation='Neither proxy certifies visible first-hit geometry. No object mask applied, matching prior proxy. Depth50 is a fixed-weight readout sensitivity, not a new renderer or algorithm.',
        input_sha256=inputs, elapsed_seconds=time.monotonic()-start), indent=2), encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-ids',type=Path,required=True)
    parser.add_argument('--native-rays',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
