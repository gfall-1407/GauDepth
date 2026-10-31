"""Fixed within-view depth-displacement placebo for the existing two scan proxies.

The comparison keeps native rays, scan points, cameras and pixel footprints fixed.
It tests readout correspondence, not certified first-surface accuracy.
"""
import argparse
from collections import defaultdict
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


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return (int(row['scan']), row['view'], int(row['ray']))


def finite(value):
    return bool(np.isfinite(float(value)))


def mapping_for(rows, rays_sha):
    groups = defaultdict(list)
    undefined = []
    for row in rows:
        k = key(row)
        if finite(row['depth50_mm']):
            groups[(k[0], k[1], row['h3_fixed'] == 'True')].append(k)
        else:
            undefined.append(list(k))
    pairs, singleton = [], []
    for group in sorted(groups):
        members = sorted(groups[group])
        if len(members) == 1:
            singleton.append(list(members[0]))
            continue
        for i, receiver in enumerate(members):
            pairs.append([*receiver, members[(i+1) % len(members)][2]])
    return dict(definition='Within the same scan, view and pre-frozen H3/non-H3 layer, sort ray IDs and give each receiver the next ray ID delta cyclically once. No replacement seed or proxy-based selection. Undefined depth50 and singleton groups have no donor.',
                source_rays_sha256=rays_sha, pairs=sorted(pairs), singleton=sorted(singleton), undefined=sorted(undefined))


def mean(values):
    return float(np.mean(values)) if values else np.nan


def summarize(rows, scan, role):
    group = [r for r in rows if r['scan'] == scan and (role == 'all' or (r['h3_fixed'] if role == 'H3' else not r['h3_fixed']))]
    paired_nn = [r for r in group if np.isfinite(r['real_nn_mm']) and np.isfinite(r['placebo_nn_mm'])]
    paired_front = [r for r in group if np.isfinite(r['real_front_mm']) and np.isfinite(r['placebo_front_mm'])]
    result = dict(scan=scan, role=role, rays=len(group), depth50_defined=sum(np.isfinite(r['real_mm']) for r in group),
        placebo_no_donor=sum(r['placebo_status'] == 'no_donor' for r in group),
        placebo_nonpositive=sum(r['placebo_status'] == 'nonpositive' for r in group),
        placebo_nonfinite=sum(r['placebo_status'] == 'nonfinite' for r in group),
        placebo_valid=sum(r['placebo_status'] == 'valid' for r in group),
        nn_paired=len(paired_nn), front_paired=len(paired_front),
        front_footprint_available=sum(r['front_candidate_n'] > 0 for r in group),
        front_missing=len(group)-len(paired_front))
    for label, subset, field in [('nn', paired_nn, 'nn_mm'), ('front', paired_front, 'front_mm')]:
        for method in ('native', 'real', 'placebo'):
            result[f'{method}_{label}_mean_mm'] = mean([r[f'{method}_{field}'] for r in subset])
        result[f'real_{label}_wins'] = sum(r[f'real_{field}'] < r[f'placebo_{field}'] for r in subset)
        result[f'placebo_{label}_wins'] = sum(r[f'placebo_{field}'] < r[f'real_{field}'] for r in subset)
        result[f'{label}_ties'] = sum(r[f'placebo_{field}'] == r[f'real_{field}'] for r in subset)
    return result


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze(args):
    if args.mapping.exists():
        raise ValueError('Mapping already exists')
    rows = csv_rows(args.previous_rays)
    if len(rows) != 19968 or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Previous rays incomplete or duplicated')
    mapping = mapping_for(rows, sha(args.previous_rays))
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_text(json.dumps(mapping, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps(dict(mapping_sha256=sha(args.mapping), pairs=len(mapping['pairs']),
                          singleton=len(mapping['singleton']), undefined=len(mapping['undefined']))))


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    if sha(args.mapping) != args.mapping_sha256:
        raise ValueError('Frozen mapping SHA changed')
    previous = csv_rows(args.previous_rays)
    if len(previous) != 19968 or len({key(r) for r in previous}) != len(previous):
        raise ValueError('Previous rays incomplete or duplicated')
    mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    if mapping != mapping_for(previous, sha(args.previous_rays)):
        raise ValueError('Frozen mapping differs from the deterministic rule')
    prev = {key(r): r for r in previous}
    donors = {(s, v, r): (s, v, d) for s, v, r, d in mapping['pairs']}
    index, records = source.source_files(args.data_root)
    prior_meta = json.loads(args.previous_metadata.read_text(encoding='utf-8'))
    if prior_meta['source_index_sha256'] != sha(index) or prior_meta['script_sha256'] != args.previous_script_sha256:
        raise ValueError('Previous source metadata changed')
    prior_inputs = {x['scan']: x for x in prior_meta['input_sha256']}
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True):
        raise ValueError('Commit script before running')
    start = time.monotonic()
    rows, semantic, seen = [], [], set()
    for scan in source.SCANS:
        items = source.scan_records(records, scan)
        gt_item = source.named(items, f'stl{scan:03d}_total.ply')[0]
        cam_item = source.named(items, 'cameras.npz')[0]
        original = prior_inputs[scan]
        if (gt_item['sha256'] != original['gt_sha256'] or cam_item['sha256'] != original['camera_sha256'] or
                sha(args.data_root / gt_item['path']) != gt_item['sha256'] or sha(args.data_root / cam_item['path']) != cam_item['sha256']):
            raise ValueError(f'GT/camera changed for scan {scan}')
        scale_mat = np.load(args.data_root / cam_item['path'])['scale_mat_0'].astype(float)
        scale = float(scale_mat[0, 0])
        if scale != original['scale_to_mm']:
            raise ValueError(f'Camera scale changed for scan {scan}')
        vertex = PlyData.read(args.data_root / gt_item['path'])['vertex']
        gt = np.column_stack([vertex[x] for x in ('x', 'y', 'z')]).astype(float)
        ids = np.arange(len(gt))
        finite_gt = np.isfinite(gt).all(1)
        gt, source_ids = gt[finite_gt], ids[finite_gt]
        scene = (gt-scale_mat[:3, 3])/scale
        for trace, _, view in source.view_records(args.data_root, items):
            path = args.data_root / trace['path']
            if sha(path) != trace['sha256']:
                raise ValueError(f'Trace changed: {path}')
            with np.load(path) as data:
                pixels = data['pixel_xy'].astype(float)
                offsets = data['offsets']
                depths = data['depth'].astype(float)
                weights = data['weight'].astype(float)
                transmittance = data['transmittance'].astype(float)
            world_view = np.asarray(view['world_view_transform'], float).T
            cam = (scene @ world_view[:3, :3].T+world_view[:3, 3])*scale
            positive = cam[:, 2] > 0
            cam, ids = cam[positive], source_ids[positive]
            height, width = view['image_size']
            focal = np.asarray(view['focal'], float)
            center = np.array([(width-1)/2., (height-1)/2.])
            uv = cam[:, :2]/cam[:, 2, None]*focal+center
            in_frame = ((uv >= [-1.,-1.]) & (uv <= [width,height])).all(1)
            cam, uv, ids = cam[in_frame], uv[in_frame], ids[in_frame]
            tree = cKDTree(cam)
            for i, xy in enumerate(pixels):
                k = (scan, view['image_name'], i)
                old = prev[k]
                if old['trace_sha256'] != trace['sha256'] or (int(old['pixel_x']), int(old['pixel_y'])) != tuple(xy):
                    raise ValueError(f'Previous ray/trace mismatch: {k}')
                seen.add(k)
                lo, hi = map(int, offsets[i:i+2])
                z, w, t = depths[lo:hi], weights[lo:hi], transmittance[lo:hi]
                inversion = bool(np.any(z[1:] < z[:-1]))
                ties = bool(np.unique(z).size < len(z))
                order = np.argsort(z, kind='stable')
                cumulative = np.cumsum(w[order])
                defined = bool(len(z)>0 and w.sum() >= .5 and cumulative[-1] >= .5)
                if defined != finite(old['depth50_mm']):
                    raise ValueError(f'Depth50 defined branch mismatch: {k}')
                if defined and abs(float(z[order[np.searchsorted(cumulative, .5)]])*scale-float(old['depth50_mm'])) > 1e-4:
                    raise ValueError(f'Depth50 replay mismatch: {k}')
                native_choice = int(np.flatnonzero(t > .5)[-1]) if len(z) and np.any(t > .5) else -1
                native_replay = float(z[native_choice])*scale if native_choice >= 0 else 0.
                if abs(native_replay-float(old['native_mm'])) > 1e-4:
                    raise ValueError(f'Native readout replay mismatch: {k}')
                semantic.append(dict(scan=scan, view=k[1], ray=i, members=len(z), inversion=inversion,
                    ties=ties, depth50_defined=defined, native_threshold_member=native_choice,
                    ordered_defined=bool(not inversion and defined),
                    ordered_equal=bool(not inversion and defined and abs(float(old['depth50_mm'])-native_replay) < 1e-4)))
                donor = donors.get(k)
                if donor is None:
                    placebo = np.nan
                    status = 'no_donor'
                else:
                    delta = float(prev[donor]['depth50_mm'])-float(prev[donor]['native_mm'])
                    placebo = float(old['native_mm'])+delta
                    status = 'valid' if np.isfinite(placebo) and placebo>0 else 'nonpositive' if np.isfinite(placebo) else 'nonfinite'
                front_z = float(old['front_z_mm'])
                if status == 'valid':
                    direction = np.r_[(xy-center)/focal, 1.]
                    nearest, nearest_idx = tree.query(direction*placebo, k=1)
                    placebo_nn = float(nearest)
                    placebo_gt_id = int(ids[nearest_idx])
                    placebo_offset = float(np.linalg.norm(uv[nearest_idx]-xy))
                    placebo_gt_minus_pred_z = float(cam[nearest_idx, 2]-placebo)
                    placebo_front = abs(placebo-front_z) if np.isfinite(front_z) else np.nan
                else:
                    placebo_nn, placebo_gt_id, placebo_offset, placebo_gt_minus_pred_z, placebo_front = np.nan, -1, np.nan, np.nan, np.nan
                rows.append(dict(scan=scan, view=k[1], ray=i, h3_fixed=old['h3_fixed']=='True',
                    donor_ray=donor[2] if donor else -1, placebo_status=status,
                    native_mm=float(old['native_mm']), real_mm=float(old['depth50_mm']), placebo_mm=placebo,
                    native_nn_mm=float(old['native_nn_mm']), real_nn_mm=float(old['depth50_nn_mm']), placebo_nn_mm=placebo_nn,
                    placebo_nn_gt_id=placebo_gt_id, placebo_nn_projection_px=placebo_offset,
                    placebo_nn_gt_minus_pred_z_mm=placebo_gt_minus_pred_z,
                    placebo_same_real_nn_gt=bool(placebo_gt_id >= 0 and placebo_gt_id == int(old['depth50_nn_gt_id'])),
                    native_front_mm=float(old['native_front_axial_mm']), real_front_mm=float(old['depth50_front_axial_mm']), placebo_front_mm=placebo_front,
                    front_z_mm=front_z, front_candidate_n=int(old['front_candidate_n'])))
            print(json.dumps(dict(scan=scan, view=view['image_name'], rays=len(pixels))), flush=True)
    if seen != prev.keys() or len(rows) != 19968:
        raise ValueError('Incomplete source ray set')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'rays.csv', rows)
    write_csv(args.output/'semantics.csv', semantic)
    summaries = [summarize(rows, scan, role) for scan in source.SCANS for role in ('H3', 'nonH3', 'all')]
    write_csv(args.output/'scans.csv', summaries)
    (args.output/'metadata.json').write_text(json.dumps(dict(commit=commit, script_sha256=sha(Path(__file__)),
        source_index_sha256=sha(index), previous_rays_sha256=sha(args.previous_rays), previous_metadata_sha256=sha(args.previous_metadata),
        mapping_sha256=sha(args.mapping), mapping_definition=mapping['definition'], scans=list(source.SCANS),
        rays=len(rows), semantics=dict(inverted=sum(x['inversion'] for x in semantic),
            tied=sum(x['ties'] for x in semantic), ordered_defined=sum(x['ordered_defined'] for x in semantic),
            ordered_equal=sum(x['ordered_equal'] for x in semantic)), elapsed_seconds=time.monotonic()-start,
        limitation='Placebo and real depth50 are fixed-readout sensitivity comparisons on scan-point proxies, not certified visible first-hit geometry.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--previous-rays', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--mapping-sha256')
    parser.add_argument('--previous-metadata', type=Path)
    parser.add_argument('--previous-script-sha256')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args)
    else:
        if not all((args.mapping_sha256, args.previous_metadata, args.previous_script_sha256, args.data_root, args.output)):
            parser.error('run needs mapping SHA, previous metadata/script SHA, data root and output')
        run(args)
