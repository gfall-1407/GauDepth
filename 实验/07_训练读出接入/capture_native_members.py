"""Read-only native member capture for one DTU view and 32 fixed-seed rays.

Requires the reviewed accepted_gaussians extension. T and w are replayed from
native accepted alpha; they are detached observations, not autograd tensors.
"""

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                 read_intrinsics_binary, read_intrinsics_text, qvec2rotmat)
from diff_gaussian_rasterization import _C
from capture_ray_support import selected_camera, sha256


def native_geometry(blob, count):
    """Decode the first GeometryState arrays (rasterizer_impl.cu::fromChunk)."""
    raw = blob.cpu().numpy()
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError('unexpected native geometry buffer')
    offset = 0
    arrays = {}
    for name, dtype, shape in (
        ('depth_sort', np.float32, (count,)),
        ('ray_plane', np.float32, (count, 4)),
        ('native_normal', np.float32, (count, 4)),
        ('clamped', np.bool_, (count, 3)),
        ('radius', np.int32, (count,)),
        ('mean2d', np.float32, (count, 2)),
        ('conic_opacity', np.float32, (count, 4)),
    ):
        offset = (offset + 127) & ~127
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if offset + size > raw.nbytes:
            raise ValueError(f'geometry buffer too short for {name}')
        arrays[name] = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape)),
                                     offset=offset).copy().reshape(shape)
        offset += size
    return arrays


def read_training_cfg(path):
    expr = ast.parse(path.read_text(encoding='utf-8').strip(), mode='eval').body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != 'Namespace' or expr.args:
        raise ValueError('cfg_args must be a simple Namespace(...)')
    result = {}
    for keyword in expr.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
            raise ValueError(f'unsupported cfg_args field {keyword.arg}')
        result[keyword.arg] = keyword.value.value
    return result


def audit_opacity(logit, activated, filtered):
    """The PLY stores pre-sigmoid logits; their signed infinities are valid limits."""
    if np.isnan(logit).any():
        raise ValueError('NaN opacity logit in original PLY')
    positive, negative = np.isposinf(logit), np.isneginf(logit)
    for name, value in (('sigmoid opacity', activated), ('render opacity', filtered)):
        if not np.isfinite(value).all() or np.any((value < 0.) | (value > 1.)):
            raise ValueError(f'{name} must be finite and in [0, 1]')
    if not np.all(activated[positive] == 1.) or not np.all(activated[negative] == 0.):
        raise ValueError('opacity saturation disagrees with sigmoid logit')
    if np.any(filtered > activated + 1e-6):
        raise ValueError('3D-filtered opacity exceeds unfiltered opacity')
    return dict(posinf=int(positive.sum()), neginf=int(negative.sum()), nan=0)


def replay_members(xy, ids, alpha, counts, ray_planes, mean2d, fx, fy, width, height):
    transmittance = np.zeros_like(alpha, dtype=np.float32)
    weight = np.zeros_like(alpha, dtype=np.float32)
    t = np.zeros_like(alpha, dtype=np.float32)
    di = np.zeros_like(alpha, dtype=np.float32)
    depth = np.zeros(len(xy), dtype=np.float32)
    final_t = np.ones(len(xy), dtype=np.float32)
    for q, (x, y) in enumerate(xy):
        rln = np.float32(1. / math.sqrt(((x - (width - 1) / 2) / fx) ** 2 +
                                         ((y - (height - 1) / 2) / fy) ** 2 + 1.))
        accum = np.float32(0.)
        for j in range(int(counts[q])):
            member = int(ids[q, j])
            a = np.float32(alpha[q, j])
            plane = ray_planes[member]
            transmittance[q, j] = final_t[q]
            weight[q, j] = np.float32(final_t[q] * a)
            dx, dy = np.float32(mean2d[member, 0] - x), np.float32(mean2d[member, 1] - y)
            t[q, j] = np.float32(np.float32(plane[0] * dx) + np.float32(plane[1] * dy) + plane[2])
            di[q, j] = np.float32(t[q, j] * rln)
            accum = np.float32(accum + np.float32(t[q, j] * weight[q, j]))
            final_t[q] = np.float32(final_t[q] * np.float32(1. - a))
        if counts[q]:
            depth[q] = np.float32(np.float32(accum * rln) / np.float32(1. - final_t[q]))
    return transmittance, weight, t, di, final_t, depth


def filter_camera_roster(camera_files, dataset):
    """Reproduce Scene's sorted train-camera split without loading all images."""
    read_extr = read_extrinsics_binary if camera_files[0].suffix == '.bin' else read_extrinsics_text
    read_intr = read_intrinsics_binary if camera_files[1].suffix == '.bin' else read_intrinsics_text
    extr, intr = read_extr(str(camera_files[0])), read_intr(str(camera_files[1]))
    rows = sorted(extr.values(), key=lambda row: Path(row.name).stem)
    if dataset.eval:
        rows = [row for i, row in enumerate(rows) if i % 8 != 0]
    cameras = []
    for row in rows:
        lens = intr[row.camera_id]
        raw_fx = lens.params[0]
        raw_fy = lens.params[1] if lens.model == 'PINHOLE' else raw_fx
        if lens.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError(f'unsupported COLMAP camera {lens.model}')
        if dataset.resolution in (1, 2, 4, 8):
            width = round(lens.width / dataset.resolution)
            height = round(lens.height / dataset.resolution)
        elif dataset.resolution == -1:
            factor = max(1., lens.width / 1600.)
            width, height = int(lens.width / factor), int(lens.height / factor)
        else:
            factor = lens.width / dataset.resolution
            width, height = int(lens.width / factor), int(lens.height / factor)
        cameras.append(SimpleNamespace(
            R=torch.tensor(qvec2rotmat(row.qvec).T.copy(), dtype=torch.float32, device='cuda'),
            T=torch.tensor(row.tvec, dtype=torch.float32, device='cuda'),
            Fx=raw_fx * width / lens.width, Fy=raw_fy * height / lens.height,
            image_width=width, image_height=height))
    return cameras, [row.name for row in rows]


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument('--ply', required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--rays', type=int, default=32)
    parser.add_argument('--support-k', type=int, default=2048)
    parser.add_argument('--cfg', required=True, help='exact training cfg_args or config file')
    args = parser.parse_args()
    dataset = model.extract(args)
    ply, output, cfg = map(lambda p: Path(p).resolve(), (args.ply, args.output, args.cfg))
    if not ply.is_file() or not cfg.is_file() or output.exists():
        raise ValueError('PLY and cfg must exist; output must be new')
    if args.rays != 32 or args.support_k < 128:
        raise ValueError('capture contract: 32 rays, K >= 128')
    training_cfg = read_training_cfg(cfg)
    for key in ('source_path', 'images', 'resolution', 'eval', 'kernel_size',
                'white_background', 'sh_degree', 'disable_filter3D', 'use_decoupled_appearance'):
        if getattr(dataset, key) != training_cfg[key]:
            raise ValueError(f'capture {key}={getattr(dataset, key)!r} disagrees with cfg_args {training_cfg[key]!r}')
    camera, camera_files, image_file = selected_camera(dataset.source_path,
                                                       dataset.images, args.view, dataset)
    height, width = camera.image_height, camera.image_width
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(ply))
    points = gaussians.get_xyz.shape[0]
    saved_filter = gaussians.filter_3D.clone()
    bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.],
                      dtype=torch.float32, device='cuda')
    empty = torch.empty(0, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        if dataset.disable_filter3D:
            gaussians.reset_3D_filter()
            scale, opacity = gaussians.get_scaling, gaussians.get_opacity
        else:
            scale, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
        opacity_stats = audit_opacity(gaussians._opacity.cpu().numpy(),
                                      gaussians.get_opacity.cpu().numpy(), opacity.cpu().numpy())
        rendered, _, native_alpha, native_normal, native_depth, native_median, _, geom, bins, img = (
            _C.rasterize_gaussians(
                bg, gaussians.get_xyz, empty, opacity, scale, gaussians.get_rotation,
                1., empty, camera.world_view_transform, camera.full_proj_transform,
                math.tan(camera.FoVx * .5), math.tan(camera.FoVy * .5),
                dataset.kernel_size, height, width, gaussians.get_features,
                gaussians.active_sh_degree, camera.camera_center, False, True, False))
        rows, cols = np.mgrid[1:height - 1, 1:width - 1]
        valid = (rows * width + cols).reshape(-1)
        if len(valid) < args.rays:
            raise ValueError('image too small for 32 interior rays')
        flat = np.sort(np.random.default_rng(args.seed).choice(valid, args.rays, replace=False))
        centers = np.stack((flat % width, flat // width), axis=1).astype(np.float32)
        offsets = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float32)
        xy_np = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        xy = torch.from_numpy(xy_np).to(device='cuda').contiguous()
        ids, alpha, count, overflow = _C.accepted_gaussians(
            xy, geom, bins, img, points, rendered, width, height, args.support_k, False)
        ids, alpha, count, overflow = [v.cpu().numpy() for v in (ids, alpha, count, overflow)]
        if np.any(overflow) or np.any(count > args.support_k):
            raise ValueError(f'incomplete accepted lists; max count={int(count.max())}; raise K')
        valid_alpha = alpha[ids >= 0]
        if (not np.isfinite(alpha).all() or
                np.any((valid_alpha < 1. / 255. - 1e-6) | (valid_alpha > .99 + 1e-6)) or
                np.any(alpha[ids < 0] != 0.)):
            raise ValueError('accepted alpha is non-finite or outside native range')
        if geom.data_ptr() % 128:
            raise ValueError('geometry allocation violates native 128-byte layout assumption')
        geometry = native_geometry(geom, points)
        unique = np.unique(ids[ids >= 0]).astype(np.int64)
        if np.any(unique >= points):
            raise ValueError('accepted ID outside PLY row range')
        T, w, t, di, final_T, replay_depth = replay_members(
            xy_np, ids, alpha, count, geometry['ray_plane'], geometry['mean2d'], camera.Fx, camera.Fy,
            width, height)
        flat_xy = (xy_np[:, 1] * width + xy_np[:, 0]).astype(np.int64)
        native_flat = native_depth.reshape(-1).cpu().numpy()
        sampled_native = native_flat[flat_xy]
        error = float(np.max(np.abs(sampled_native - replay_depth)))
        if not np.isfinite(error) or error > 2e-3:
            raise ValueError(f'native/replayed depth mismatch: {error}')
        indices = torch.from_numpy(unique).to(device='cuda')
        capture = dict(seed_flat=flat, xy=xy_np, ids=ids, alpha=alpha, count=count,
                       overflow=overflow, T_before=T, weight=w, ray_t=t, di=di,
                       final_T=final_T, replay_depth=replay_depth,
                       native_depth=sampled_native,
                       native_alpha=native_alpha.reshape(-1).cpu().numpy()[flat_xy],
                       native_normal_image=native_normal.reshape(3, -1).cpu().numpy()[:, flat_xy],
                       native_median=native_median.reshape(-1).cpu().numpy()[flat_xy],
                       unique_id=unique,
                       center=gaussians.get_xyz[indices].cpu().numpy(),
                       raw_quaternion=gaussians._rotation[indices].cpu().numpy(),
                       quaternion=gaussians.get_rotation[indices].cpu().numpy(),
                       raw_scale=gaussians._scaling[indices].cpu().numpy(),
                       scale_unaa=gaussians.get_scaling[indices].cpu().numpy(),
                       opacity_logit=gaussians._opacity[indices].cpu().numpy(),
                       opacity_logit_posinf=torch.isposinf(gaussians._opacity[indices]).cpu().numpy(),
                       opacity_logit_neginf=torch.isneginf(gaussians._opacity[indices]).cpu().numpy(),
                       opacity_unaa=gaussians.get_opacity[indices].cpu().numpy(),
                       filter_3D=saved_filter[indices].cpu().numpy(),
                       scale_render=scale[indices].cpu().numpy(),
                       opacity_render=opacity[indices].cpu().numpy(),
                       ray_plane=geometry['ray_plane'][unique],
                       native_normal_member=geometry['native_normal'][unique],
                       mean2d=geometry['mean2d'][unique],
                       conic_opacity=geometry['conic_opacity'][unique],
                       world_view_transform=camera.world_view_transform.cpu().numpy(),
                       full_proj_transform=camera.full_proj_transform.cpu().numpy(),
                       camera_R=camera.R.cpu().numpy(), camera_T=camera.T.cpu().numpy(),
                       intrinsics=np.array([camera.Fx, camera.Fy, camera.Cx, camera.Cy]),
                       image_size=np.array([height, width]))
        filter_cameras, filter_image_names = filter_camera_roster(camera_files, dataset)
        gaussians.compute_3D_filter(filter_cameras)
        filter_delta = (gaussians.filter_3D - saved_filter).abs()
        filter_max = float(filter_delta.max().item())
        filter_mean = float(filter_delta.mean().item())
        nonfinite = {key: int(np.size(value) - np.isfinite(value).sum())
                     for key, value in capture.items()
                     if key != 'opacity_logit' and np.issubdtype(value.dtype, np.floating)}
        if any(nonfinite.values()) or not np.isfinite([filter_max, filter_mean]).all():
            raise ValueError(f'non-finite capture fields: {nonfinite}')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'native_members.npz', **capture)
    source_root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), source_root / 'train.py',
               source_root / 'scene/gaussian_model.py', source_root / 'scene/dataset_readers.py',
               source_root / 'scene/cameras.py',
               source_root / 'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py',
               source_root / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu',
               source_root / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu',
               source_root / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/support_forward.cu']
    binary = Path(_C.__file__)
    metadata = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    ply_path=str(ply), ply_sha256=sha256(ply), ply_row_count=points,
                    id_mapping='unique_id is zero-based original PLY vertex row order (load_ply preserves order)',
                    cfg_path=str(cfg), cfg_sha256=sha256(cfg), view=args.view,
                    resolution=dataset.resolution, eval=dataset.eval,
                    image_path=str(image_file), image_sha256=sha256(image_file),
                    colmap_sha256={str(p): sha256(p) for p in camera_files},
                    source_sha256={str(p.relative_to(source_root)): sha256(p) for p in sources},
                    binary_path=str(binary), binary_sha256=sha256(binary),
                    filter_provenance='PLY field compared with recomputation from sorted same-scene train cameras; saved filter may predate final 100 training iterations',
                    filter_views=[Path(name).stem for name in filter_image_names],
                    train_image_sha256={str(Path(dataset.source_path) / dataset.images / Path(name).name):
                                        sha256(Path(dataset.source_path) / dataset.images / Path(name).name)
                                        for name in filter_image_names},
                    filter_recompute_max_abs=filter_max,
                    filter_recompute_mean_abs=filter_mean,
                    nonfinite_counts=nonfinite,
                    opacity_logit_full_stats=opacity_stats,
                    opacity_logit_selected_stats=dict(
                        posinf=int(capture['opacity_logit_posinf'].sum()),
                        neginf=int(capture['opacity_logit_neginf'].sum()), nan=0),
                    normal_definition='native preprocess computeCov2D camera-space normal; stored float4 GeometryState.normals',
                    T_weight_provenance='float32 replay of native accepted alpha in list order',
                    detached=True, seed=args.seed, rays=args.rays, neighborhood='3x3',
                    support_k=args.support_k, max_count=int(count.max()),
                    depth_replay_max_abs_error=error,
                    result_sha256=sha256(output / 'native_members.npz'))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), result_sha256=metadata['result_sha256'],
                          max_count=metadata['max_count'], depth_error=error)))


if __name__ == '__main__':
    main()
