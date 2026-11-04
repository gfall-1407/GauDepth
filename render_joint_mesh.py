"""Bounded read-only-model Expected/joint depth rendering and matched TSDF fusion.

Run on the authorized Linux GPU server with an outer ``timeout`` as backup.
The process also arms a POSIX hard wall timer. Limited-view/pixel runs are
diagnostics and cannot produce meshes. No source dataset or input PLY is written.
"""
import argparse
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from joint_tv_inputs import sha256, MODEL_IDENTITIES
from joint_tv_model import candidate_depths, rays_for_pixels, build_support_index, FIELD_PROTOCOL
from joint_training_anchors import load_cameras
from train_joint_matched import native_config, runtime_identity


NEAR, FAR = FIELD_PROTOCOL['near'], FIELD_PROTOCOL['far']


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def output_root(path):
    out = Path(path).resolve()
    root = Path('/root/joint_tv_runs').resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('fresh output must be beneath /root/joint_tv_runs')
    root.mkdir(parents=True, exist_ok=True)
    if root.stat().st_dev != Path('/root').stat().st_dev:
        raise ValueError('output root must be on the system filesystem')
    if shutil.disk_usage(root).free < (3 << 30):
        raise ValueError('at least 3 GiB system-disk space required')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    return out


def contained_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('render artifact must be an existing file inside its run')
    return path


def check_deadline(start, limit):
    if time.monotonic() - start >= limit:
        raise TimeoutError('internal wall budget exhausted')


def read_identity(args):
    source = Path(args.source_path).resolve()
    if source.name != f'scan{args.scan}':
        raise ValueError('explicit scan does not match source directory')
    actual = sha256(args.input_ply)
    if actual != args.input_sha256.lower():
        raise ValueError('input PLY SHA mismatch')
    for known in MODEL_IDENTITIES.values():
        if actual == known['ply_sha256'] and source.name != known['scene']:
            raise ValueError('known input PLY belongs to another scene')
    identity = dict(input_ply=str(Path(args.input_ply).resolve()), input_sha256=actual,
                    source_path=str(source), scan=args.scan, declared_iteration=args.iteration,
                    colmap_sha256={n: sha256(source/'sparse'/'0'/n)
                                   for n in ('cameras.bin', 'images.bin')})
    sidecar = Path(args.input_ply).parent / 'snapshot.json'
    if sidecar.is_file():
        snapshot = json.loads(sidecar.read_text())
        if snapshot.get('ply_sha256') != actual or snapshot.get('global_iteration') != args.iteration:
            raise ValueError('snapshot sidecar does not match PLY/iteration')
        original = snapshot.get('input_identity', {})
        if original.get('colmap_sha256') != identity['colmap_sha256']:
            raise ValueError('training snapshot camera identity does not match extraction source')
        identity.update(snapshot_sha256=sha256(sidecar), snapshot=snapshot)
    return identity


def require_settings(settings):
    required = ('white_background', 'kernel_size', 'sh_degree', 'use_decoupled_appearance', 'eval')
    if not isinstance(settings, dict) or any(key not in settings for key in required):
        raise ValueError('complete native renderer settings required')
    if type(settings['white_background']) is not bool or type(settings['eval']) is not bool:
        raise ValueError('native background/eval settings must be boolean')
    if type(settings['sh_degree']) is not int or not 0 <= settings['sh_degree'] <= 3:
        raise ValueError('invalid native SH degree')
    if not isinstance(settings['kernel_size'], (int, float)) or isinstance(settings['kernel_size'], bool):
        raise ValueError('invalid native kernel size type')
    if not math.isfinite(settings['kernel_size']) or not 0 <= settings['kernel_size'] <= 1:
        raise ValueError('invalid native kernel size')
    if type(settings['use_decoupled_appearance']) is not int or settings['use_decoupled_appearance'] not in (0, 3):
        raise ValueError('only the matched pilot NO/fresh-PGSR native setup is supported')


def resolve_render_config(args, identity):
    """Use recorded settings; conflicting CLI values never silently override them."""
    snapshot = identity.get('snapshot')
    current_runtime = runtime_identity()
    if snapshot is not None:
        required = ('field_protocol', 'geometry_protocol', 'native_config', 'resolution',
                    'runtime_identity', 'training_head')
        if any(key not in snapshot for key in required):
            raise ValueError('snapshot lacks the complete training/extraction contract')
        if snapshot['field_protocol'] != FIELD_PROTOCOL:
            raise ValueError('complete training/extraction FIELD_PROTOCOL mismatch')
        if not isinstance(snapshot['training_head'], str) or not re.fullmatch(r'[0-9a-f]{40}', snapshot['training_head']):
            raise ValueError('snapshot lacks an exact training source SHA')
        if snapshot['training_head'] != args.expected_head:
            raise ValueError('extract the training snapshot using its exact recorded source commit')
        protocol = snapshot['geometry_protocol']
        for key, expected in (('near', NEAR), ('far', FAR), ('opacity_factor', FIELD_PROTOCOL['opacity_factor'])):
            if key not in protocol or protocol[key] != expected:
                raise ValueError('training/extraction geometry mismatch: ' + key)
        if any(key not in protocol for key in ('member_epsilon', 'max_members')):
            raise ValueError('snapshot lacks actual support thresholds')
        config = snapshot['native_config']
        if args.native_config:
            supplied = native_config(args.native_config)
            if supplied['sha256'] != config.get('sha256') or supplied['settings'] != config.get('settings'):
                raise ValueError('supplied native config conflicts with snapshot')
        resolution = snapshot['resolution']
        epsilon, max_members = protocol['member_epsilon'], protocol['max_members']
        stored_runtime = snapshot['runtime_identity']
        for key in ('torch_version', 'cuda_version', 'rasterizer_wrapper_sha256', 'rasterizer_binary_sha256'):
            if key not in stored_runtime or stored_runtime[key] != current_runtime[key]:
                raise ValueError('training/extraction native runtime mismatch: ' + key)
        provenance = 'snapshot training contract'
    else:
        if not args.native_config:
            raise ValueError('original PLY without snapshot requires --native-config')
        config = native_config(args.native_config)
        # native_config already validates a literal Namespace expression. Read
        # its recorded camera resolution without eval or a second hidden default.
        expression = ast.parse(Path(args.native_config).read_text(), mode='eval').body
        raw = {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords}
        if 'resolution' not in raw:
            raise ValueError('native config must declare resolution')
        resolution = raw['resolution']
        epsilon = FIELD_PROTOCOL['default_epsilon'] if args.member_epsilon is None else args.member_epsilon
        max_members = FIELD_PROTOCOL['default_max_members'] if args.max_members is None else args.max_members
        provenance = 'explicit literal native config; no prior training snapshot contract'
    settings = config.get('settings') if isinstance(config, dict) else None
    require_settings(settings)
    if type(resolution) is not int or resolution not in (1, 2, 4, 8):
        raise ValueError('recorded resolution must be explicit 1/2/4/8; automatic resizing is not silently reinterpreted')
    expected = dict(background='white' if settings['white_background'] else 'black',
                    kernel_size=settings['kernel_size'], sh_degree=settings['sh_degree'],
                    resolution=resolution, member_epsilon=epsilon, max_members=max_members)
    for name, value in expected.items():
        explicit = getattr(args, name)
        if explicit is not None and explicit != value:
            raise ValueError('explicit --'+name.replace('_', '-')+' conflicts with recorded settings')
        setattr(args, name, value)
    if not 0 < args.member_epsilon < 1 or type(args.max_members) is not int or not 1 <= args.max_members <= 4096:
        raise ValueError('invalid resolved support budget')
    return dict(native_config=config, renderer_settings=settings, resolution=resolution,
                runtime_identity=current_runtime, provenance=provenance,
                runtime_claim='matching recorded binary hashes, not an independent compile-source verification')


def diag_vector(diag, name, count, dtype):
    if name not in diag:
        raise ValueError('candidate diagnostics missing ' + name)
    value = diag[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=dtype).reshape(-1)
    if len(vector) != count:
        raise ValueError('candidate diagnostic length mismatch: ' + name)
    return vector


def summarize_candidate(diag, valid, count):
    has_root = diag_vector(diag, 'has_root', count, bool)
    overflow = diag_vector(diag, 'overflow', count, bool)
    convergence = diag_vector(diag, 'convergence', count, bool)
    members = diag_vector(diag, 'member_counts', count, np.int64)
    # Rootless but stable rays have valid=False and convergence=True. Never turn
    # unresolved numerical results or support overflow into empty background.
    numerical = overflow | ~convergence | (has_root & ~valid)
    failures = jsonable(diag.get('guard_failures'))
    if not isinstance(failures, list) or len(failures) != count:
        raise ValueError('candidate guard_failures must identify every ray')
    rootless_sentinels = {'no_detected_root', 'root_slope', 'root_residual'}
    for i, reasons in enumerate(failures):
        if not isinstance(reasons, list):
            raise ValueError('candidate guard_failures row is malformed')
        # A missing root naturally has no root slope/residual. Near occupancy,
        # nonfinite fields and unresolved monotonic directions are separate
        # failures even if this finite search found no threshold crossing.
        if not has_root[i] and any(reason not in rootless_sentinels for reason in reasons):
            numerical[i] = True
    return has_root, numerical, members


def render_view(g, camera, args, out, index, start, limit, pipe, background):
    w, h = int(camera.image_width), int(camera.image_height)
    stem = f'{index:05d}_' + re.sub(r'[^0-9A-Za-z_.-]', '_', camera.image_name)
    state_path = out/'views'/f'{stem}.json'
    artifact = out/'views'/f'{stem}.npz'
    state = dict(status='started_not_complete', index=index, image_name=camera.image_name,
                 width=w, height=h, fx=float(camera.Fx), fy=float(camera.Fy),
                 cx=float(camera.Cx), cy=float(camera.Cy), processed_pixels=0,
                 total_pixels=w*h, numerical_failure_pixels=0, no_detected_root_pixels=0,
                 alpha_mask_excluded_pixels=0, methods=list(args.methods))
    state['world_to_camera'] = camera.world_view_transform.T.detach().cpu().double().tolist()
    write_json(state_path, state)
    begin = time.perf_counter()
    torch.cuda.synchronize()
    package = render(camera, g, pipe, background, args.kernel_size, require_depth=True)
    torch.cuda.synchronize()
    state['native_render_seconds'] = time.perf_counter() - begin
    color = np.ascontiguousarray(package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    expected = package['expected_depth'].reshape(h, w).cpu().numpy().astype(np.float32)
    del package
    allowed = np.ones((h, w), dtype=bool)
    if camera.gt_mask is not None:
        allowed = camera.gt_mask.detach().cpu().numpy().reshape(h, w) >= .5
    state['alpha_mask_excluded_pixels'] = int((~allowed).sum())
    selection = np.ones(w*h, dtype=bool)
    if args.max_pixels:
        selection[:] = False
        selection[np.linspace(0, w*h-1, min(args.max_pixels, w*h), dtype=np.int64)] = True
    selection = selection.reshape(h, w)
    processed = np.zeros((h, w), dtype=bool)
    joint = np.zeros((h, w), dtype=np.float32)
    joint_valid = np.zeros((h, w), dtype=bool)
    has_root_map = np.zeros((h, w), dtype=bool)
    convergence_map = np.zeros((h, w), dtype=bool)
    member_hist = np.zeros(args.max_members+1, dtype=np.int64)
    guard_counts = Counter()
    candidate_start = time.perf_counter()
    support_index = build_support_index(g, camera, epsilon=args.member_epsilon) if 'joint' in args.methods else None
    processed_tiles = 0

    def save_arrays(partial=False):
        ev = np.isfinite(expected) & (expected >= NEAR) & (expected <= FAR) & allowed & processed
        arrays = dict(color=color, processed=processed, allowed=allowed)
        if 'expected' in args.methods:
            arrays.update(expected=np.where(ev, expected, 0).astype(np.float32), expected_valid=ev)
        if 'joint' in args.methods:
            jv = joint_valid & allowed & processed
            arrays.update(joint=np.where(jv, joint, 0).astype(np.float32), joint_valid=jv,
                          joint_has_root=has_root_map, joint_convergence=convergence_map)
        path = artifact.with_name(artifact.stem + '.partial.npz') if partial else artifact
        np.savez(path, **arrays)
        return path, {m: int(arrays[m+'_valid'].sum()) for m in args.methods}

    try:
        for y0 in range(0, h, args.tile_size):
            for x0 in range(0, w, args.tile_size):
                check_deadline(start, limit)
                y1, x1 = min(h, y0+args.tile_size), min(w, x0+args.tile_size)
                yy, xx = np.nonzero(selection[y0:y1, x0:x1])
                if not len(xx):
                    continue
                xx, yy = xx+x0, yy+y0
                # Masked pixels are explicitly processed/excluded, equally for both readouts.
                masked = ~allowed[yy, xx]
                processed[yy[masked], xx[masked]] = True
                xx, yy = xx[~masked], yy[~masked]
                ids = support_index.query(x0, x1-1, y0, y1-1) if support_index is not None else None
                for offset in range(0, len(xx), args.batch_size):
                    check_deadline(start, limit)
                    bx, by = xx[offset:offset+args.batch_size], yy[offset:offset+args.batch_size]
                    if 'joint' in args.methods:
                        xy = torch.tensor(np.stack((bx, by), axis=-1), device='cuda', dtype=torch.float64)
                        origins, directions = rays_for_pixels(camera, xy)
                        depth, valid, diag = candidate_depths(g, origins, directions,
                            epsilon=args.member_epsilon, max_members=args.max_members,
                            verify=True, require_grad=False, candidate_ids=ids)
                        values = depth.detach().cpu().double().numpy().reshape(-1)
                        good = valid.detach().cpu().numpy().astype(bool).reshape(-1)
                        if len(values) != len(bx) or len(good) != len(bx):
                            raise ValueError('candidate output shape mismatch')
                        has_root, numerical, members = summarize_candidate(diag, good, len(bx))
                        numerical |= good & (~np.isfinite(values) | (values < NEAR) | (values > FAR))
                        if numerical.any():
                            state.update(numerical_failure_pixels=int(numerical.sum()),
                                failed_batch_pixels=np.stack((bx, by), axis=-1).tolist(),
                                failed_batch_diagnostics=jsonable(diag))
                            raise RuntimeError('unresolved candidate/support failure; no mesh may be produced')
                        joint[by, bx] = values.astype(np.float32)
                        joint_valid[by, bx] = good
                        has_root_map[by, bx] = has_root
                        convergence_map[by, bx] = True
                        state['no_detected_root_pixels'] += int((~has_root).sum())
                        member_hist += np.bincount(members, minlength=args.max_members+1)[:args.max_members+1]
                        for reasons in jsonable(diag.get('guard_failures', [])):
                            guard_counts.update(reasons)
                    processed[by, bx] = True
                processed_tiles += 1
                if processed_tiles % 16 == 0:
                    state.update(processed_pixels=int(processed.sum()), elapsed_seconds=time.perf_counter()-begin)
                    write_json(state_path, state)
                    if shutil.disk_usage(out).free < (1 << 30):
                        raise RuntimeError('system-disk free space below 1 GiB')
        torch.cuda.synchronize()
        saved, valid_counts = save_arrays()
        expected_count = int(selection.sum())
        if int(processed.sum()) != expected_count:
            raise RuntimeError('not all selected pixels were processed')
        state.update(status='diagnostic_complete' if args.max_pixels else 'render_complete',
            processed_pixels=int(processed.sum()), selected_pixels=expected_count,
            valid_pixels=valid_counts, candidate_and_tile_seconds=time.perf_counter()-candidate_start,
            elapsed_seconds=time.perf_counter()-begin, artifact=str(saved.relative_to(out)),
            artifact_sha256=sha256(saved), guard_counts=dict(guard_counts), member_histogram=member_hist.tolist())
        write_json(state_path, state)
        return state
    except BaseException as error:
        saved, valid_counts = save_arrays(partial=True)
        state.update(status='failed_not_complete', error=repr(error),
            processed_pixels=int(processed.sum()), valid_pixels=valid_counts,
            partial_artifact=str(saved.relative_to(out)), guard_counts=dict(guard_counts),
            elapsed_seconds=time.perf_counter()-begin)
        write_json(state_path, state)
        raise


def fuse(render_root, render_manifest, out, args, start, limit):
    import open3d as o3d
    import open3d.core as o3c
    from mesh_extract import post_process_mesh
    if render_manifest['diagnostic'] or render_manifest['status'] not in ('render_complete', 'complete'):
        raise ValueError('only complete full-view/full-pixel renders can be fused')
    if len(render_manifest['views']) != render_manifest['camera_count']:
        raise ValueError('incomplete camera set')
    mesh_records = {}
    for method in render_manifest['methods']:
        check_deadline(start, limit)
        directory = out/method
        directory.mkdir(exist_ok=False)
        state = dict(status='started_not_complete', integrated_views=0, method=method)
        write_json(directory/'fusion.json', state)
        device = o3c.Device(args.fusion_device)
        grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(o3c.float32,)*3,
            attr_channels=(1, 1, 3), voxel_size=args.voxel_size, block_resolution=16,
            block_count=args.block_count, device=device)
        try:
            for view in render_manifest['views']:
                check_deadline(start, limit)
                if view['status'] != 'render_complete' or view['processed_pixels'] != view['total_pixels']:
                    raise ValueError('view is not complete')
                artifact = contained_file(render_root, view['artifact'])
                if sha256(artifact) != view['artifact_sha256']:
                    raise ValueError('saved depth artifact hash mismatch')
                with np.load(artifact, allow_pickle=False) as saved:
                    depth = np.ascontiguousarray(saved[method], dtype=np.float32)
                    color = np.ascontiguousarray(saved['color'], dtype=np.float32)
                if not (depth > 0).any():
                    state.setdefault('empty_views', []).append(view['image_name'])
                    state['integrated_views'] += 1
                    write_json(directory/'fusion.json', state)
                    continue
                intrinsic = o3c.Tensor([[view['fx'], 0., view['cx']],
                                       [0., view['fy'], view['cy']], [0., 0., 1.]],
                                      dtype=o3c.float64)
                extrinsic = o3c.Tensor(view['world_to_camera'], dtype=o3c.float64)
                depth_img = o3d.t.geometry.Image(depth).to(device)
                color_img = o3d.t.geometry.Image(color).to(device)
                blocks = grid.compute_unique_block_coordinates(depth_img, intrinsic, extrinsic, 1., FAR)
                grid.integrate(blocks, depth_img, color_img, intrinsic, extrinsic, 1., FAR)
                state['integrated_views'] += 1
                write_json(directory/'fusion.json', state)
            check_deadline(start, limit)
            mesh = grid.extract_triangle_mesh().to_legacy()
            if not len(mesh.triangles):
                raise RuntimeError('empty TSDF mesh')
            mesh.compute_vertex_normals()
            raw = directory/'recon.ply'
            if not o3d.io.write_triangle_mesh(str(raw), mesh):
                raise RuntimeError('mesh write failed')
            cluster_count = len(np.unique(np.asarray(mesh.cluster_connected_triangles()[0])))
            if cluster_count < args.num_cluster:
                raise RuntimeError('fewer mesh clusters than requested; no silent cleanup change')
            post = post_process_mesh(mesh, args.num_cluster)
            if not len(post.triangles):
                raise RuntimeError('postprocessed TSDF mesh is empty')
            cleaned = directory/'recon_post.ply'
            if not o3d.io.write_triangle_mesh(str(cleaned), post):
                raise RuntimeError('postprocessed mesh write failed')
            state.update(status='complete', raw_vertices=len(mesh.vertices), raw_triangles=len(mesh.triangles),
                post_vertices=len(post.vertices), post_triangles=len(post.triangles),
                recon_sha256=sha256(raw), recon_post_sha256=sha256(cleaned))
            write_json(directory/'fusion.json', state)
            mesh_records[method] = dict(directory=str(directory), **state)
        except BaseException as error:
            state.update(status='failed_not_complete', error=repr(error))
            write_json(directory/'fusion.json', state)
            raise
        finally:
            del grid
    return mesh_records


def prepare_evaluation_layout(directory, manifest):
    """Place only a symlink to the immutable PLY beside each completed mesh.

    The legacy evaluator loads a model directory even though its alignment only
    needs camera poses. This layout avoids copying or rewriting input models.
    It does not execute evaluation, install dependencies or touch the dataset.
    """
    identity = manifest['input_identity']
    protocol = manifest['protocol']
    renderer = manifest['renderer_settings']
    iteration = identity['declared_iteration']
    target = directory/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    target.parent.mkdir(parents=True, exist_ok=False)
    target.symlink_to(identity['input_ply'])
    config = dict(sh_degree=renderer['sh_degree'], source_path=identity['source_path'], model_path=str(directory),
        images='images', dataset='dtu', resolution=protocol['resolution'],
        white_background=protocol['background']=='white', data_device='cpu', eval=False,
        use_decoupled_appearance=renderer['use_decoupled_appearance'], disable_filter3D=False, kernel_size=protocol['kernel_size'],
        depth_ratio=0., multi_view_num=8, multi_view_max_angle=30,
        multi_view_min_dis=.01, multi_view_max_dis=1.5)
    (directory/'cfg_args').write_text(str(argparse.Namespace(**config))+'\n')
    # Scene's old loader creates this file if absent. Never launch that path
    # against original data without first satisfying this read-only precondition.
    source_ply_exists = (Path(identity['source_path'])/'sparse'/'0'/'points3D.ply').is_file()
    return dict(status='prepared_not_evaluated', input_is_symlink=True,
        source_points3D_ply_exists=source_ply_exists,
        precondition='source sparse/0/points3D.ply must already exist; otherwise do not launch legacy evaluator',
        command=[sys.executable, 'evaluate_dtu_mesh.py', '--model_path', str(directory),
                 '--source_path', identity['source_path'], '--iteration', str(iteration),
                 '--resolution', str(protocol['resolution']), '--DTU', '<official-DTU-root>'])


def main(args):
    if os.name != 'posix' or not hasattr(signal, 'setitimer'):
        raise RuntimeError('Linux server with POSIX hard timer required')
    if not 0 < args.max_seconds <= 86400:
        raise ValueError('wall budget must be in (0, 86400] seconds')
    if not 1 <= args.batch_size <= 128 or not 1 <= args.tile_size <= 64:
        raise ValueError('invalid batch/tile budget')
    if ((args.member_epsilon is not None and not 0 < args.member_epsilon < 1) or
            (args.max_members is not None and not 1 <= args.max_members <= 4096)):
        raise ValueError('invalid support budget')
    if args.max_views < 0 or args.max_pixels < 0 or args.num_cluster < 1 or args.block_count < 1:
        raise ValueError('invalid extraction budget')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('duplicate readout methods')
    if args.voxel_size <= 0 or (args.kernel_size is not None and args.kernel_size < 0):
        raise ValueError('invalid native/TSDF setting')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if head != args.expected_head or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('unexpected or dirty source checkout')
    out = output_root(args.output)
    start = time.monotonic()
    manifest = dict(status='started_not_complete', head=head, command=dict(vars(args)), views=[])
    write_json(out/'manifest.json', manifest)
    # Default SIGALRM terminates only this process, even while a native operation
    # has not returned to Python. A killed run retains started/partial status.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        if args.fuse_from:
            if args.max_views or args.max_pixels or args.render_only:
                raise ValueError('fuse-from cannot use render diagnostic switches')
            if any(getattr(args, key) is not None for key in
                   ('native_config', 'background', 'kernel_size', 'sh_degree', 'resolution', 'member_epsilon', 'max_members')):
                raise ValueError('fuse-from uses saved render settings; do not supply renderer/support overrides')
            render_root = Path(args.fuse_from).resolve()
            rendered = json.loads((render_root/'manifest.json').read_text())
            if sha256(rendered['input_identity']['input_ply']) != rendered['input_identity']['input_sha256']:
                raise ValueError('rendered input model changed after rendering')
            source = Path(rendered['input_identity']['source_path'])/'sparse'/'0'
            if any(sha256(source/name) != expected
                   for name, expected in rendered['input_identity']['colmap_sha256'].items()):
                raise ValueError('camera calibration changed after rendering')
            manifest.update(render_source=str(render_root), render_source_sha256=sha256(render_root/'manifest.json'),
                            input_identity=rendered['input_identity'], methods=rendered['methods'],
                            protocol=rendered['protocol'], renderer_settings=rendered['renderer_settings'],
                            native_config=rendered['native_config'], runtime_identity=rendered['runtime_identity'],
                            diagnostic=False, camera_count=rendered['camera_count'], views=rendered['views'])
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no local CPU fallback')
            for name in ('input_ply', 'input_sha256', 'source_path', 'scan', 'iteration'):
                if getattr(args, name) is None:
                    raise ValueError('render requires --' + name.replace('_', '-'))
            identity = read_identity(args)
            resolved = resolve_render_config(args, identity)
            cameras, camera_meta = load_cameras(args.source_path, resolution=args.resolution, data_device='cpu')
            if camera_meta['source_sha256'] != identity['colmap_sha256']:
                raise ValueError('camera identity changed during load')
            if not cameras:
                raise ValueError('no cameras')
            manifest.update(input_identity=identity, methods=list(args.methods), camera_count=len(cameras),
                native_config=resolved['native_config'], renderer_settings=resolved['renderer_settings'],
                runtime_identity=resolved['runtime_identity'], render_setting_provenance=resolved['provenance'],
                runtime_claim=resolved['runtime_claim'],
                camera_metadata=camera_meta, diagnostic=bool(args.max_views or args.max_pixels),
                protocol=dict(solver='continuous_stationary_joint_TV', near=NEAR, far=FAR,
                    member_epsilon=args.member_epsilon, max_members=args.max_members,
                    verify_refinement=True, opacity_factor=FIELD_PROTOCOL['opacity_factor'],
                    shared_field_protocol=FIELD_PROTOCOL, tile_size=args.tile_size,
                    batch_size=args.batch_size, resolution=args.resolution, background=args.background,
                    sh_degree=args.sh_degree,
                    kernel_size=args.kernel_size, masked_pixels='image alpha <0.5 excluded in both readouts',
                    expected_range='native expected depth restricted to the same near/far',
                    rootless='stable no detected root -> zero depth and separately counted; no fallback'))
            write_json(out/'manifest.json', manifest)
            (out/'views').mkdir()
            g = GaussianModel(args.sh_degree)
            g.load_ply(args.input_ply)
            background = torch.tensor([1., 1., 1.] if args.background == 'white' else [0., 0., 0.], device='cuda')
            pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
            torch.cuda.reset_peak_memory_stats()
            selected = cameras[:args.max_views] if args.max_views else cameras
            with torch.no_grad():
                for index, camera in enumerate(selected):
                    check_deadline(start, args.max_seconds)
                    record = render_view(g, camera, args, out, index, start, args.max_seconds, pipe, background)
                    manifest['views'].append(record)
                    manifest['elapsed_seconds'] = time.monotonic()-start
                    write_json(out/'manifest.json', manifest)
                    print(json.dumps({k: record[k] for k in ('index', 'image_name', 'status', 'processed_pixels', 'valid_pixels', 'elapsed_seconds')}), flush=True)
            manifest['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['gpu'] = torch.cuda.get_device_name()
            if sha256(args.input_ply) != identity['input_sha256']:
                raise ValueError('input model changed during rendering')
            manifest['status'] = 'diagnostic_complete' if manifest['diagnostic'] else 'render_complete'
            manifest['elapsed_seconds'] = time.monotonic()-start
            count = sum(v['processed_pixels'] for v in manifest['views'])
            all_pixels = sum(c.image_width*c.image_height for c in cameras)
            manifest['pilot_extrapolation_seconds_all_views'] = (manifest['elapsed_seconds']*all_pixels/count) if count else None
            manifest['pilot_extrapolation_note'] = 'linear sampled-wall-time scaling only; not measured dense runtime or a completion promise'
            write_json(out/'manifest.json', manifest)
            if manifest['diagnostic'] or args.render_only:
                return
            render_root, rendered = out, manifest
        manifest['fusion_protocol'] = dict(voxel_size=args.voxel_size, block_resolution=16,
            block_count=args.block_count, depth_scale=1., depth_max=FAR, device=args.fusion_device,
            num_cluster=args.num_cluster, cleanup='mesh_extract.post_process_mesh')
        write_json(out/'manifest.json', manifest)
        manifest['meshes'] = fuse(render_root, rendered, out, args, start, args.max_seconds)
        if sha256(manifest['input_identity']['input_ply']) != manifest['input_identity']['input_sha256']:
            raise ValueError('input model changed during mesh fusion')
        manifest['evaluation_layouts'] = {method: prepare_evaluation_layout(out/method, manifest)
                                          for method in manifest['methods']}
        manifest.update(status='complete', elapsed_seconds=time.monotonic()-start,
                        evaluation_status='not_run_use_project_evaluate_dtu_mesh_pipeline')
        write_json(out/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed_not_complete', error=repr(error), elapsed_seconds=time.monotonic()-start)
        write_json(out/'manifest.json', manifest)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-ply'); parser.add_argument('--input-sha256')
    parser.add_argument('--source-path'); parser.add_argument('--scan', type=int)
    parser.add_argument('--native-config', help='literal native cfg_args; mandatory without a training snapshot')
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--output', required=True); parser.add_argument('--expected-head', required=True)
    parser.add_argument('--methods', nargs='+', choices=('expected', 'joint'), default=['expected', 'joint'])
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=None)
    parser.add_argument('--background', choices=('white', 'black'), default=None)
    parser.add_argument('--kernel-size', type=float, default=None)
    parser.add_argument('--sh-degree', type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument('--member-epsilon', type=float, default=None)
    parser.add_argument('--max-members', type=int, default=None)
    parser.add_argument('--tile-size', type=int, default=16); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-views', type=int, default=0); parser.add_argument('--max-pixels', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--fuse-from', help='complete render run; write fusion into a new --output directory')
    parser.add_argument('--voxel-size', type=float, default=.002)
    parser.add_argument('--block-count', type=int, default=50000)
    parser.add_argument('--fusion-device', default='CPU:0')
    parser.add_argument('--num-cluster', type=int, default=1)
    main(parser.parse_args())
