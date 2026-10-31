"""Prediction-blind audit of a frozen selection of saved DTU rays.

Selection must be frozen independently. This program never reads method depth,
method error, host_target, or the broad nearest-point proxy output.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

import all_scan_proxy as sources


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    selected = json.loads(args.selection.read_text(encoding='utf-8'))['selected']
    if len({(x['scan'], x['view'], x['ray']) for x in selected}) != len(selected):
        raise ValueError('Duplicate selection IDs')
    ref_path = Path(__file__).resolve().parents[1] / '03_高斯场层归属探索' / 'ray_gt_reference.py'
    spec = importlib.util.spec_from_file_location('dtu_reference', ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    index_path, records = sources.source_files(args.data_root)
    rows, inputs = [], []
    for scan in sorted({x['scan'] for x in selected}):
        items = sources.scan_records(records, scan)
        gt_item = sources.named(items, f'stl{scan:03d}_total.ply')
        cam_item = sources.named(items, 'cameras.npz')
        if len(gt_item) != 1 or not cam_item or any(x['sha256'] != cam_item[0]['sha256'] for x in cam_item):
            raise ValueError(f'GT/camera ambiguity: scan {scan}')
        gt_path = args.data_root / gt_item[0]['path']
        cam_path = args.data_root / cam_item[0]['path']
        if sha(gt_path) != gt_item[0]['sha256'] or sha(cam_path) != cam_item[0]['sha256']:
            raise ValueError(f'GT/camera hash mismatch: scan {scan}')
        transform = np.load(cam_path)['scale_mat_0'].astype(float)
        scale = float(transform[0, 0])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3)*scale, atol=1e-6)
        vertex = PlyData.read(gt_path)['vertex']
        gt = np.column_stack([vertex[key] for key in ('x','y','z')]).astype(float)
        gt = gt[np.isfinite(gt).all(1)]
        scene = (gt-transform[:3, 3])/scale
        inputs.append(dict(scan=scan, gt_sha256=gt_item[0]['sha256'], camera_sha256=cam_item[0]['sha256']))
        for trace, meta, view in sources.view_records(args.data_root, items):
            keys = [x for x in selected if x['scan']==scan and x['view']==view['image_name']]
            if not keys:
                continue
            trace_path = args.data_root / trace['path']
            if sha(trace_path) != trace['sha256']:
                raise ValueError(f'Trace hash mismatch: {trace_path}')
            # Only pixel_xy is read from the trace; no estimator/host field enters this audit.
            with np.load(trace_path) as data:
                pixels = data['pixel_xy'].astype(float)
            world_view = np.asarray(view['world_view_transform'], float).T
            rot, trans = world_view[:3, :3], world_view[:3, 3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=2e-6)
            height, width = view['image_size']
            focal = np.asarray(view['focal'], float)
            center = np.array([(width-1)/2., (height-1)/2.])
            cam = (scene @ rot.T + trans)*scale
            cam = cam[cam[:, 2] > 0]
            uv = cam[:, :2]/cam[:, 2, None]*focal+center
            inside = ((uv >= [-1.,-1.]) & (uv <= [width,height])).all(1)
            cam, uv = cam[inside], uv[inside]
            tree = cKDTree(uv)
            for x in keys:
                ray = int(x['ray'])
                if ray < 0 or ray >= len(pixels):
                    raise ValueError(f'Invalid ray ID: {x}')
                pixel = pixels[ray]
                row = dict(scan=scan, view=view['image_name'], ray=ray, labels=x['labels'],
                           pixel_xy=pixel.tolist(), trace_sha256=trace['sha256'],
                           nearest_projected_scan_px=float(tree.query(pixel, k=1)[0]))
                refs = []
                for radius in (1.5, 3.):
                    neighbors = tree.query_ball_point(pixel, radius)
                    points = cam[neighbors]
                    result = ref.local_reference(points, uv[neighbors], pixel, focal, center)
                    z_quantiles = np.quantile(points[:, 2], [0,.1,.5,.9,1]).tolist() if len(points) else []
                    refs.append(result)
                    row[f'radius_{radius}_candidate_n'] = len(neighbors)
                    row[f'radius_{radius}_z_quantiles_mm'] = z_quantiles
                    row[f'radius_{radius}_fit'] = result
                if refs[0]['status']==refs[1]['status']=='accepted':
                    gap = abs(refs[0]['reference_mm']-refs[1]['reference_mm'])
                    row['status'] = 'accepted' if gap<=.1 else 'cross_scale_disagreement'
                    row['cross_scale_gap_mm'] = gap
                else:
                    row['status'] = 'missing_multiscale_reference'
                    row['cross_scale_gap_mm'] = None
                rows.append(row)
    if len(rows) != len(selected):
        raise ValueError('Some selected rays were not found in metadata')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'reference_blind.json').write_text(json.dumps(dict(
        selection_sha256=sha(args.selection), script_sha256=sha(Path(__file__)),
        source_helper_sha256=sha(Path(sources.__file__)), reference_code_sha256=sha(ref_path),
        source_index_sha256=sha(index_path), inputs=inputs, rays=sorted(rows,key=lambda x:(x['scan'],x['view'],x['ray'])),
        interpretation='GT-only local scan plane quality audit; still not proof of visible first surface. No predictions were accessed.'
    ), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(selected=len(selected),statuses={k:sum(x['status']==k for x in rows) for k in sorted({x['status'] for x in rows})})))


if __name__=='__main__':
    main()
