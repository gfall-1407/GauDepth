"""Fixed, prediction-blind DTU scan-plane proxy on the 15 saved G1 ray sets.

This measures selected saved rays only. A local plane fitted to the DTU point
cloud is neither an exact pixel first hit nor the official DTU mesh metric.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import time

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


SCANS = (24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def analyze_hypotheses(proxy_dir, native_dir, output):
    """Join immutable ray IDs; never use reference error to pick a member."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    proxy = read_csv(proxy_dir / 'rays.csv')
    native = read_csv(native_dir / 'native_rays.csv')
    pmeta = json.loads((proxy_dir / 'metadata.json').read_text(encoding='utf-8'))
    nmeta = json.loads((native_dir / 'metadata.json').read_text(encoding='utf-8'))
    scales = {int(item['scan']): item['scale_to_mm'] for item in pmeta['inputs']}
    def key(row):
        return (int(row['scan']), row['view'], int(row['ray']))
    pmap = {key(row): row for row in proxy}
    nmap = {key(row): row for row in native}
    if len(pmap) != len(proxy) or len(nmap) != len(native) or set(pmap) != set(nmap):
        raise ValueError('Ray IDs are duplicated or mismatched')
    joined = []
    for k, p in pmap.items():
        n = nmap[k]
        if p['trace_sha256'] != n['trace_sha256']:
            raise ValueError(f'Trace mismatch: {k}')
        scale = scales[k[0]]
        impact = float(n['max_deletion_influence']) * scale
        depth50 = float(n['depth50']) * scale
        native_median = float(p['native_median_mm'])
        target = float(p['reference_mm'])
        without = float(n['expected_without_max_influence']) * scale
        joined.append(dict(scan=k[0], view=k[1], ray=k[2], reference_valid=p['status']=='accepted',
            branch=n['branch'], opacity=float(n['opacity']),
            max_deletion_impact_mm=impact,
            max_influence_weight_fraction=float(n['max_influence_weight_fraction']),
            depth50_mm=depth50,
            depth50_minus_native_mm=depth50-native_median,
            threshold_margin=float(n['threshold_margin']),
            next_depth_gap_mm=float(n['next_depth_gap']) * scale,
            expected_abs_reference_mm=float(p['expected_abs_mm']),
            median_abs_reference_mm=float(p['native_median_abs_mm']),
            without_max_influence_abs_reference_mm=abs(without-target),
            depth50_abs_reference_mm=abs(depth50-target),
            expected_nearest_scan_mm=float(p['expected_nearest_scan_mm']),
            median_nearest_scan_mm=float(p['native_median_nearest_scan_mm'])))
    write_csv(output / 'joined_rays.csv', joined)
    summaries = []
    for scan in sorted(scales):
        group = [row for row in joined if row['scan']==scan]
        accepted = [row for row in group if row['reference_valid']]
        fallback = [row for row in group if row['branch']=='last_without_crossing']
        impacts = np.asarray([row['max_deletion_impact_mm'] for row in group if np.isfinite(row['max_deletion_impact_mm'])])
        ordered = np.asarray([abs(row['depth50_minus_native_mm']) for row in group if np.isfinite(row['depth50_minus_native_mm'])])
        depth50_ref = [row for row in accepted if np.isfinite(row['depth50_abs_reference_mm'])]
        deletion_ref = [row for row in accepted if np.isfinite(row['without_max_influence_abs_reference_mm'])]
        high_impact_ref = [row for row in deletion_ref if row['max_deletion_impact_mm'] > 1.]
        high_disagreement_ref = [row for row in depth50_ref if abs(row['depth50_minus_native_mm']) > 1.]
        summaries.append(dict(scan=scan, rays=len(group), reference_valid=len(accepted),
            no_crossing=len(fallback), no_crossing_reference_valid=sum(row['reference_valid'] for row in fallback),
            max_deletion_impact_mm_median=float(np.median(impacts)) if len(impacts) else np.nan,
            max_deletion_impact_mm_p95=float(np.quantile(impacts,.95)) if len(impacts) else np.nan,
            impact_over_1mm=int((impacts>1.).sum()),
            impact_over_1mm_reference_paired=len(high_impact_ref),
            impact_over_1mm_reference_improved=sum(row['without_max_influence_abs_reference_mm'] < row['expected_abs_reference_mm'] for row in high_impact_ref),
            depth50_defined=len(ordered), depth50_disagrees_over_1mm=int((ordered>1.).sum()),
            depth50_disagrees_over_1mm_reference_paired=len(high_disagreement_ref),
            depth50_disagrees_over_1mm_reference_improved=sum(row['depth50_abs_reference_mm'] < row['median_abs_reference_mm'] for row in high_disagreement_ref),
            depth50_disagreement_mm_p95=float(np.quantile(ordered,.95)) if len(ordered) else np.nan,
            near_threshold_and_gap_over_1mm=sum(np.isfinite(row['threshold_margin']) and row['threshold_margin']<=.01 and np.isfinite(row['next_depth_gap_mm']) and row['next_depth_gap_mm']>1. for row in group),
            deletion_reference_paired=len(deletion_ref),
            deletion_reference_improved=sum(row['without_max_influence_abs_reference_mm'] < row['expected_abs_reference_mm'] for row in deletion_ref),
            depth50_reference_paired=len(depth50_ref),
            depth50_reference_improved_vs_native=sum(row['depth50_abs_reference_mm'] < row['median_abs_reference_mm'] for row in depth50_ref)))
    write_csv(output / 'hypothesis_scans.csv', summaries)
    (output / 'metadata.json').write_text(json.dumps(dict(
        script_sha256=sha(Path(__file__)), proxy_metadata_sha256=sha(proxy_dir/'metadata.json'),
        native_metadata_sha256=sha(native_dir/'metadata.json'),
        command_argv=[sys.executable, *sys.argv],
        fixed_descriptive_bands=dict(influence_mm=1., threshold_margin=.01, neighboring_depth_gap_mm=1.),
        interpretation='H1-H3 mechanism audit on all saved rays; reference comparisons are restricted to the unchanged independent scan-plane coverage. Deleting one fixed-weight member is algebraic attribution, not re-rendering.'
    ), indent=2), encoding='utf-8')


def source_files(root):
    index = root / '来源索引.json'
    records = json.loads(index.read_text(encoding='utf-8'))['files']
    for item in records:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'Missing or wrong size: {path}')
    return index, records


def scan_records(records, scan):
    return [item for item in records if re.search(rf'scan0*{scan}(?!\d)', item['path'], re.I)]


def named(items, name):
    return [item for item in items if PureWindowsPath(item['path']).name.lower() == name.lower()]


def view_records(root, items):
    traces = [item for item in items if re.fullmatch(r'view_\d+\.npz', PureWindowsPath(item['path']).name, re.I)]
    if len(traces) != 3:
        raise ValueError(f'Expected three saved views, got {len(traces)}')
    result = []
    for trace in sorted(traces, key=lambda item: item['path']):
        path = root / trace['path']
        meta_path = path.parent / 'metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        view = next((v for v in meta['views'] if v['file'] == path.name), None)
        if view is None or view['npz_sha256'] != trace['sha256']:
            raise ValueError(f'Metadata view/hash mismatch: {path}')
        result.append((trace, meta, view))
    return result


def estimate(z, w, host_median):
    if len(z) == 0 or not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w < 0):
        return (np.nan, np.nan)
    total = float(w.sum())
    if total <= 0:
        return (np.nan, np.nan)
    expected = float(np.dot(z, w) / total)
    return expected, float(host_median) if np.isfinite(host_median) else np.nan


def positive_depth(value):
    return bool(np.isfinite(value) and value > 0)


def strongest_deletion(z, w):
    """Largest algebraic leave-one-out shift among positive, deletable members."""
    mass = float(w.sum())
    if mass <= 0:
        return -1, np.nan, np.nan
    mean = float(np.dot(w, z) / mass)
    rest = mass - w
    valid = (w > 0) & (rest > 0)
    if not valid.any():
        return -1, np.nan, np.nan
    shifts = np.abs(w * (z-mean) / np.where(valid, rest, np.nan))
    member = int(np.argmax(np.where(valid, shifts, -np.inf)))
    without = float((mass*mean-w[member]*z[member])/rest[member])
    return member, float(shifts[member]), without


def native_semantics(root, output, scans):
    """Replay the exact pre-T>0.5 native branch without GT or fitted thresholds."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    index, records = source_files(root)
    rows = []
    for scan in scans:
        for trace, _, view in view_records(root, scan_records(records, scan)):
            path = root / trace['path']
            if sha(path) != trace['sha256']:
                raise ValueError(f'Trace hash mismatch: {path}')
            with np.load(path) as data:
                offsets = data['offsets']
                z = data['depth'].astype(float)
                w = data['weight'].astype(float)
                alpha = data['alpha'].astype(float)
                pre_t = data['transmittance'].astype(float)
                host = data['host_target'].astype(float)
            if offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) != len(host)+1 or offsets[0] != 0 or offsets[-1] != len(z) or any(x.shape != z.shape or x.ndim != 1 for x in (w, alpha, pre_t)) or z.ndim != 1 or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f'Invalid arrays: {path}')
            if host.ndim != 2 or host.shape[1] < 3 or not np.isfinite(host[:, :3]).all():
                raise ValueError(f'Invalid host targets: {path}')
            for i in range(len(host)):
                lo, hi = map(int, offsets[i:i+2])
                dz, dw, da, dt = z[lo:hi], w[lo:hi], alpha[lo:hi], pre_t[lo:hi]
                if not len(dz):
                    rows.append(dict(scan=scan, view=view['image_name'], ray=i, members=0,
                                     branch='empty', replay_abs=np.nan, expected_replay_abs=np.nan,
                                     opacity_replay_abs=np.nan, max_talpha_residual=np.nan,
                                     max_t_recurrence_residual=np.nan,
                                     opacity=0., max_deletion_influence=np.nan, threshold_margin=np.nan,
                                     next_depth_gap=np.nan, depth50=np.nan,
                                     native_median=np.nan,
                                     max_influence_member=-1, max_influence_weight_fraction=np.nan,
                                     expected_without_max_influence=np.nan,
                                     trace_sha256=trace['sha256']))
                    continue
                if not (np.isfinite(dz).all() and np.isfinite(dw).all() and np.isfinite(da).all() and np.isfinite(dt).all()) or np.any(dw < 0) or np.any(da < 0) or np.any(da >= 1) or np.any(dt < 0) or np.any(dt > 1) or np.any(dt[1:] > dt[:-1]):
                    raise ValueError(f'Nonfinite or invalid native member: {path}, ray {i}')
                before = np.flatnonzero(dt > .5)
                chosen = int(before[-1]) if len(before) else -1
                replay = float(dz[chosen]) if chosen >= 0 else 0.
                branch = 'crossed_half' if chosen >= 0 and dt[chosen]*(1-da[chosen]) <= .5 else 'last_without_crossing' if chosen == len(dz)-1 else 'no_pre_t_above_half'
                mass = float(dw.sum())
                mu = float(np.dot(dw, dz)/mass) if mass > 0 else np.nan
                max_member, max_impact, without = strongest_deletion(dz, dw)
                depth_order = np.argsort(dz, kind='stable')
                depth_cumulative = np.cumsum(dw[depth_order])
                depth50 = float(dz[depth_order[np.searchsorted(depth_cumulative, .5)]]) if mass >= .5 and depth_cumulative[-1] >= .5 else np.nan
                rows.append(dict(scan=scan, view=view['image_name'], ray=i, members=len(dz),
                    branch=branch, replay_abs=abs(replay-host[i, 1]),
                    native_median=float(host[i, 1]),
                    expected_replay_abs=abs(mu-host[i, 0]),
                    opacity_replay_abs=abs(mass-host[i, 2]),
                    max_talpha_residual=float(np.max(np.abs(dw-dt*da))),
                    max_t_recurrence_residual=float(np.max(np.abs(dt[1:]-dt[:-1]*(1-da[:-1])))) if len(dt)>1 else np.nan,
                    opacity=float(1-dt[-1]*(1-da[-1])),
                    max_deletion_influence=max_impact,
                    max_influence_member=max_member,
                    max_influence_weight_fraction=float(dw[max_member]/mass) if max_member >= 0 else np.nan,
                    expected_without_max_influence=without,
                    threshold_margin=float(np.min(np.abs(dt-.5))),
                    next_depth_gap=abs(float(dz[chosen+1]-dz[chosen])) if 0 <= chosen < len(dz)-1 else np.nan,
                    depth50=depth50,
                    trace_sha256=trace['sha256']))
        print(json.dumps(dict(scan=scan, rays=sum(row['scan'] == scan for row in rows))), flush=True)
    write_csv(output / 'native_rays.csv', rows)
    summaries = []
    for scan in scans:
        group = [row for row in rows if row['scan'] == scan]
        def finite_mean(key):
            vals = [row[key] for row in group if np.isfinite(row[key])]
            return float(np.mean(vals)) if vals else np.nan
        summaries.append(dict(scan=scan, rays=len(group),
            branches=json.dumps({key: sum(row['branch'] == key for row in group) for key in sorted({row['branch'] for row in group})}),
            max_replay_abs=max(row['replay_abs'] for row in group if np.isfinite(row['replay_abs'])),
            max_expected_replay_abs=max(row['expected_replay_abs'] for row in group if np.isfinite(row['expected_replay_abs'])),
            max_opacity_replay_abs=max(row['opacity_replay_abs'] for row in group if np.isfinite(row['opacity_replay_abs'])),
            max_talpha_residual=max(row['max_talpha_residual'] for row in group if np.isfinite(row['max_talpha_residual'])),
            max_t_recurrence_residual=max((row['max_t_recurrence_residual'] for row in group if np.isfinite(row['max_t_recurrence_residual'])), default=np.nan),
            no_deletable_member=sum(row['max_influence_member'] < 0 for row in group),
            mean_opacity=finite_mean('opacity'),
            mean_max_deletion_influence_scene_units=finite_mean('max_deletion_influence'),
            depth50_defined=sum(np.isfinite(row['depth50']) for row in group),
            depth50_differs_native=sum(np.isfinite(row['depth50']) and abs(row['depth50']-row['native_median']) > 1e-6 for row in group)))
    write_csv(output / 'native_scans.csv', summaries)
    source = Path(__file__).resolve().parents[2] / 'submodules' / 'diff-gaussian-rasterization' / 'cuda_rasterizer' / 'render_forward.cu'
    (output / 'metadata.json').write_text(json.dumps(dict(
        script_sha256=sha(Path(__file__)), renderer_sha256=sha(source),
        source_index_sha256=sha(index), scans=list(scans),
        command_argv=[sys.executable, *sys.argv], data_root=str(root.resolve()),
        median_definition='Last accepted member with pre-transmittance T>0.5; if threshold never crosses and initial T=1, this is the final accepted member.',
        influence_definition='Absolute change in normalized expected after deleting exactly one accepted member while holding other weights fixed; no GT choice or learned threshold.',
        recurrence_definition='alpha and pre-T range/monotonicity are input checks. w-T*alpha and adjacent accepted pre-T recurrence are measured residuals only; no newly chosen pass tolerance claims host equivalence.',
        units='Native scene camera-z for depths/influence; threshold_margin is dimensionless.'
    ), indent=2), encoding='utf-8')


def run(root, output, scans):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    index, records = source_files(root)
    ref_path = Path(__file__).resolve().parents[1] / '03_高斯场层归属探索' / 'ray_gt_reference.py'
    spec = importlib.util.spec_from_file_location('dtu_proxy_reference', ref_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rows, inputs = [], []
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    git_status = subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
    for scan in scans:
        items = scan_records(records, scan)
        gt_item = named(items, f'stl{scan:03d}_total.ply')
        cam_item = named(items, 'cameras.npz')
        if len(gt_item) != 1 or not cam_item:
            raise ValueError(f'GT/camera unavailable or ambiguous for scan {scan}')
        gt_path = root / gt_item[0]['path']
        cam_path = root / cam_item[0]['path']
        if sha(gt_path) != gt_item[0]['sha256'] or sha(cam_path) != cam_item[0]['sha256']:
            raise ValueError(f'GT/camera hash mismatch: scan {scan}')
        if any(item['sha256'] != cam_item[0]['sha256'] for item in cam_item):
            raise ValueError(f'Multiple camera versions: scan {scan}')
        with np.load(cam_path) as camera:
            transform = camera['scale_mat_0'].astype(float)
        scale = float(transform[0, 0])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3) * scale, atol=1e-6)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'Invalid scale: scan {scan}')
        ply = PlyData.read(gt_path)['vertex']
        gt_mm = np.column_stack([ply[key] for key in ('x', 'y', 'z')]).astype(float)
        gt_mm = gt_mm[np.isfinite(gt_mm).all(1)]
        scene = (gt_mm - transform[:3, 3]) / scale
        del gt_mm, ply
        views = view_records(root, items)
        model = [item for item in named(items, 'point_cloud.ply') if item['sha256'] == views[0][1]['model_sha256']]
        if len(model) != 1 or any(meta['model_sha256'] != model[0]['sha256'] for _, meta, _ in views):
            raise ValueError(f'Trace/model hash mismatch: scan {scan}')
        if sha(root / model[0]['path']) != model[0]['sha256']:
            raise ValueError(f'Model hash mismatch: scan {scan}')
        cfg = [item for item in named(items, 'cfg_args') if item['sha256'] == views[0][1]['cfg_sha256']]
        if len(cfg) != 1 or any(meta['cfg_sha256'] != cfg[0]['sha256'] for _, meta, _ in views):
            raise ValueError(f'Trace/config hash mismatch: scan {scan}')
        if sha(root / cfg[0]['path']) != cfg[0]['sha256']:
            raise ValueError(f'Config hash mismatch: scan {scan}')
        inputs.append(dict(scan=scan, gt=gt_item[0]['path'], gt_sha256=gt_item[0]['sha256'],
                           camera=cam_item[0]['path'], camera_sha256=cam_item[0]['sha256'],
                           model=model[0]['path'], model_sha256=model[0]['sha256'],
                           config=cfg[0]['path'], config_sha256=cfg[0]['sha256'],
                           gt_points=len(scene), scale_to_mm=scale))
        for trace, meta, view in views:
            path = root / trace['path']
            if sha(path) != trace['sha256']:
                raise ValueError(f'Trace hash mismatch: {path}')
            world_view = np.asarray(view['world_view_transform'], float).T
            rot, trans = world_view[:3, :3], world_view[:3, 3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=2e-6)
            np.testing.assert_allclose(-rot.T @ trans, view['camera_center'], atol=2e-6)
            height, width = view['image_size']
            focal = np.asarray(view['focal'], float)
            center = np.array([(width - 1) / 2., (height - 1) / 2.])
            cam_mm = (scene @ rot.T + trans) * scale
            cam_mm = cam_mm[cam_mm[:, 2] > 0]
            uv = cam_mm[:, :2] / cam_mm[:, 2, None] * focal + center
            inside = ((uv >= [-1., -1.]) & (uv <= [width, height])).all(1)
            cam_mm, uv = cam_mm[inside], uv[inside]
            tree = cKDTree(uv)
            spatial_tree = cKDTree(cam_mm)
            with np.load(path) as rays:
                xy = rays['pixel_xy'].astype(float)
                offsets = rays['offsets']
                depths = rays['depth'].astype(float) * scale
                weight = rays['weight'].astype(float)
                host = rays['host_target'][:, 1].astype(float) * scale
            if offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) != len(xy) + 1 or offsets[0] != 0 or offsets[-1] != len(depths) or depths.ndim != 1 or weight.shape != depths.shape or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f'Invalid ray offsets: {path}')
            if xy.ndim != 2 or xy.shape[1] != 2 or host.shape != (len(xy),) or not np.isfinite(xy).all():
                raise ValueError(f'Invalid sampled pixels or host median: {path}')
            # Freeze the 1.5/3 px agreement rule from ray_gt_reference.py.
            near15 = tree.query_ball_point(xy, 1.5)
            near30 = tree.query_ball_point(xy, 3.)
            estimates = [estimate(depths[int(offsets[i]):int(offsets[i+1])],
                                  weight[int(offsets[i]):int(offsets[i+1])], host[i])
                         for i in range(len(xy))]
            nearest = np.full((len(xy), 2), np.nan)
            ray_direction = np.column_stack(((xy - center) / focal, np.ones(len(xy))))
            for method in range(2):
                z = np.asarray([pair[method] for pair in estimates])
                finite = np.isfinite(z) & (z > 0)
                if finite.any() and len(cam_mm):
                    nearest[finite, method] = spatial_tree.query(ray_direction[finite] * z[finite, None], k=1)[0]
            for i, pixel in enumerate(xy):
                a = reference.local_reference(cam_mm[near15[i]], uv[near15[i]], pixel, focal, center)
                b = reference.local_reference(cam_mm[near30[i]], uv[near30[i]], pixel, focal, center)
                if a['status'] == b['status'] == 'accepted':
                    gap = abs(a['reference_mm'] - b['reference_mm'])
                    status = 'accepted' if gap <= .1 else 'cross_scale_disagreement'
                else:
                    gap = np.nan
                    status = 'missing_multiscale_reference'
                lo, hi = map(int, offsets[i:i + 2])
                expected, median = estimates[i]
                target = a['reference_mm'] if status == 'accepted' else np.nan
                rows.append(dict(scan=scan, view=view['image_name'], ray=i,
                    x=int(pixel[0]), y=int(pixel[1]), status=status,
                    status_1p5=a['status'], status_3p0=b['status'],
                    reference_mm=target, reference_rms_z_mm=a['rms_z_mm'] if status == 'accepted' else np.nan,
                    cross_scale_gap_mm=gap, expected_mm=expected, native_median_mm=median,
                    expected_abs_mm=abs(expected - target) if positive_depth(expected) else np.nan,
                    native_median_abs_mm=abs(median - target) if positive_depth(median) else np.nan,
                    expected_nearest_scan_mm=nearest[i, 0],
                    native_median_nearest_scan_mm=nearest[i, 1],
                    accepted_members=hi-lo, trace_sha256=trace['sha256']))
            print(json.dumps(dict(scan=scan, view=view['image_name'], rays=len(xy),
                                  accepted=sum(row['status'] == 'accepted' for row in rows if row['scan'] == scan and row['view'] == view['image_name']))), flush=True)
            del cam_mm, uv, tree, spatial_tree
        del scene
    write_csv(output / 'rays.csv', rows)
    summaries = []
    for scan in scans:
        group = [row for row in rows if row['scan'] == scan]
        valid = [row for row in group if row['status'] == 'accepted']
        paired = [row for row in valid if np.isfinite(row['expected_abs_mm']) and np.isfinite(row['native_median_abs_mm'])]
        spatial = [row for row in group if np.isfinite(row['expected_nearest_scan_mm']) and np.isfinite(row['native_median_nearest_scan_mm'])]
        summaries.append(dict(scan=scan, sampled=len(group), reference_accepted=len(valid), paired=len(paired),
            expected_invalid_on_reference=sum(not positive_depth(row['expected_mm']) for row in valid),
            native_median_invalid_on_reference=sum(not positive_depth(row['native_median_mm']) for row in valid),
            expected_invalid_on_all=sum(not np.isfinite(row['expected_nearest_scan_mm']) for row in group),
            native_median_invalid_on_all=sum(not np.isfinite(row['native_median_nearest_scan_mm']) for row in group),
            expected_mae_mm=float(np.mean([row['expected_abs_mm'] for row in paired])) if paired else np.nan,
            native_median_mae_mm=float(np.mean([row['native_median_abs_mm'] for row in paired])) if paired else np.nan,
            median_minus_expected_mae_mm=float(np.mean([row['native_median_abs_mm'] - row['expected_abs_mm'] for row in paired])) if paired else np.nan,
            nearest_scan_paired=len(spatial),
            expected_nearest_scan_mean_mm=float(np.mean([row['expected_nearest_scan_mm'] for row in spatial])) if spatial else np.nan,
            native_median_nearest_scan_mean_mm=float(np.mean([row['native_median_nearest_scan_mm'] for row in spatial])) if spatial else np.nan,
            median_minus_expected_nearest_scan_mean_mm=float(np.mean([row['native_median_nearest_scan_mm']-row['expected_nearest_scan_mm'] for row in spatial])) if spatial else np.nan,
            median_nearest_scan_wins=sum(row['native_median_nearest_scan_mm'] < row['expected_nearest_scan_mm'] for row in spatial),
            reference_statuses=json.dumps({key: sum(row['status'] == key for row in group) for key in sorted({row['status'] for row in group})})))
    write_csv(output / 'scans.csv', summaries)
    (output / 'metadata.json').write_text(json.dumps(dict(
        commit=commit, git_status=git_status, script_sha256=sha(Path(__file__)),
        command_argv=[sys.executable, *sys.argv], data_root=str(root.resolve()),
        source_index_sha256=sha(index), reference_code_sha256=sha(ref_path),
        protocol=reference.PROTOCOL, reference_radii_px=[1.5, 3.0], agreement_mm=.1,
        definition='Prediction-blind DTU scan-point local front-plane agreement; no prediction-selected anchor. Exact 3D nearest scan-point distance is measured at each predicted ray point with no reference gating.',
        limitation='Neither proxy is pixel first-hit GT. Object masks are not used even where present, so background/stand contamination is possible. Principal-point/crop provenance is not fully resolved. Plane comparisons require the same positive finite outputs on the same accepted rays; nearest-scan distances use all positive finite saved depths but can match occluded/back surfaces. Missing/negative/nonfinite depths remain in sampled denominators and are counted invalid. No whole-image claims.',
        methods=['expected', 'native_median'], scans=list(scans), inputs=inputs,
        elapsed_seconds=time.monotonic()-start), indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scans', type=int, nargs='+', default=SCANS)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--analyze-native', type=Path)
    parser.add_argument('--proxy-results', type=Path)
    args = parser.parse_args()
    if any(scan not in SCANS for scan in args.scans) or len(set(args.scans)) != len(args.scans):
        parser.error('Scans must be unique members of the fixed DTU list')
    if args.analyze_native:
        if not args.proxy_results:
            parser.error('--proxy-results is required with --analyze-native')
        analyze_hypotheses(args.proxy_results, args.analyze_native, args.output)
    elif args.data_root is None:
        parser.error('--data-root is required for a data run')
    elif args.native_only:
        native_semantics(args.data_root, args.output, args.scans)
    else:
        run(args.data_root, args.output, args.scans)


if __name__ == '__main__':
    main()
