"""Read-only frozen-host sampling. Run from repository root; no training."""
import argparse
import builtins
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def protect_scene(output):
    """Redirect Scene's only write during load_iteration; restore module globals."""
    from scene import Scene
    import scene.dataset_readers as readers
    scope = Scene.__init__.__globals__
    had_open, old = 'open' in scope, scope.get('open')

    def guarded(file, mode='r', *args, **kwargs):
        if any(c in mode for c in 'wax+'):
            if Path(file).name != 'multi_view.json':
                raise RuntimeError(f'Unexpected Scene write blocked: {file}')
            file = output / 'multi_view.json'
        return builtins.open(file, mode, *args, **kwargs)

    old_store = readers.storePly

    def no_source_write(*args, **kwargs):
        raise RuntimeError('Dataset loader attempted to create source PLY; prepare existing source data explicitly')

    scope['open'] = guarded
    readers.storePly = no_source_write
    try:
        yield
    finally:
        readers.storePly = old_store
        if had_open:
            scope['open'] = old
        else:
            scope.pop('open', None)


def forward(camera, gaussians, pipe, background, kernel):
    import torch
    from diff_gaussian_rasterization import _C
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    empty = torch.empty(0, device='cuda')
    tx, ty = math.tan(camera.FoVx * .5), math.tan(camera.FoVy * .5)
    result = _C.rasterize_gaussians(
        background, gaussians.get_xyz, empty, opacity, scales, gaussians.get_rotation,
        1.0, empty, camera.world_view_transform, camera.full_proj_transform,
        tx, ty, kernel, camera.image_height, camera.image_width, gaussians.get_features,
        gaussians.active_sh_degree, camera.camera_center, False, True, pipe.debug)
    # Host C++ takes float tan_fov before calculating focal lengths.
    fx = np.float32(np.float32(camera.image_width) / (np.float32(2) * np.float32(tx)))
    fy = np.float32(np.float32(camera.image_height) / (np.float32(2) * np.float32(ty)))
    return result, float(fx), float(fy)


def sample(args):
    import torch
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C
    from torch.utils.cpp_extension import load
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    output = Path(args.output).resolve()
    model = Path(args.model_path).resolve()
    # Also accept the explicit point_cloud/iteration_30000 path.
    if model.name.startswith('iteration_') and model.parent.name == 'point_cloud':
        if int(model.name.split('_')[-1]) != args.iteration:
            raise ValueError('Model iteration directory disagrees with --iteration')
        model = model.parent.parent
    if args.iteration <= 0:
        raise ValueError('Only an existing positive iteration is allowed')
    if output == model or model in output.parents:
        raise ValueError('Output must be outside the original model directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = model / 'cfg_args'
    # Parse Namespace literals without executing the model configuration.
    import ast
    node = ast.parse(cfg_path.read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError('Expected cfg_args Namespace(key=literal, ...)')
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg is not None}
    defaults_parser = argparse.ArgumentParser()
    mp, pp = ModelParams(defaults_parser), PipelineParams(defaults_parser)
    values = vars(defaults_parser.parse_args([]))
    values.update(cfg)
    values['model_path'] = str(model)
    if args.source_path:
        values['source_path'] = str(Path(args.source_path).resolve())
    dataset = mp.extract(argparse.Namespace(**values))
    pipe = pp.extract(argparse.Namespace(**values))
    ply = model / 'point_cloud' / f'iteration_{args.iteration}' / 'point_cloud.ply'
    model_hash = sha256(ply)
    gaussians = GaussianModel(dataset.sh_degree)
    with protect_scene(output):
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras() if args.split == 'train' else scene.getTestCameras()
    selected = [int(i) for i in args.views.split(',')]
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(cameras) for i in selected):
        raise ValueError(f'Invalid/duplicate view indices for {len(cameras)} {args.split} cameras')
    build = HERE / '构建'
    build.mkdir(exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    extension = load(name='gaudepth_layers03_reader_v1', sources=[str(HERE / 'reader.cu')],
                     build_directory=str(build), extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=True)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(args.seed)
    source_files = ['reader.cu', 'sample.py', 'analyze.py']
    host_files = ['cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/render_forward.cu',
                  'cuda_rasterizer/config.h']
    metadata = dict(schema=1, git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                    git_remote=git('remote', 'get-url', 'origin'), hostname=platform.node(),
                    model=str(ply), model_sha256=model_hash, cfg_sha256=sha256(cfg_path),
                    args=vars(args), effective_model_args=vars(dataset),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), python=sys.version,
                    numpy=np.__version__, host_package=diff_gaussian_rasterization.__file__,
                    host_binary=_C.__file__, host_binary_sha256=sha256(_C.__file__),
                    reader_binary=extension.__file__, reader_binary_sha256=sha256(extension.__file__),
                    reader_build=str(build), cuda_flags=['-O3', '--use_fast_math'],
                    source_sha256={f: sha256(HERE / f) for f in source_files},
                    host_source_sha256={f: sha256(ROOT / 'submodules/diff-gaussian-rasterization' / f) for f in host_files},
                    normal_semantics='camera-space oriented geometric direction, host buffer xyz; not RGB/rendered normal',
                    normal_invalid='nonfinite, norm <= 1e-8, or zero ray-plane host fallback',
                    host_build_provenance=args.host_build_provenance,
                    replay_tolerance=dict(rtol=2e-5, atol=2e-6),
                    normal_replay_tolerance=dict(rtol=2e-4, atol=2e-5, normalize_eps=1e-12,
                        reduction='original acceptance order, CPU float32 multiply/add; host CUDA may fuse FMA'), views=[])
    atomic_json(output / 'metadata.json', metadata)
    extra = {}
    for spec in args.extra_pixel:
        vi, x, y = map(int, spec.split(':'))
        if vi not in selected:
            raise ValueError('Extra pixel view must appear in --views')
        extra.setdefault(vi, []).append((x, y))
    with torch.no_grad():
        for vi in selected:
            camera = cameras[vi]
            result, fx, fy = forward(camera, gaussians, pipe, background, dataset.kernel_size)
            _, rgb, alpha, normal, expected, median, _, geom, binning, image = result
            H, W = camera.image_height, camera.image_width
            mask = getattr(camera, 'gt_mask', None)
            mask_source = 'camera.gt_mask' if mask is not None else None
            if args.mask_dir:
                from PIL import Image
                path = Path(args.mask_dir) / args.mask_template.format(image_name=camera.image_name, view=vi)
                if not path.is_file():
                    raise FileNotFoundError(f'Explicit GT mask missing: {path}')
                with Image.open(path) as im:
                    arr = np.asarray(im.convert('L').resize((W, H), Image.Resampling.NEAREST))
                mask = torch.from_numpy(arr > 127)
                mask_source = dict(path=str(path.resolve()), sha256=sha256(path), resize='nearest')
            if mask is not None:
                eligible = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1) > .5)
                sampling = 'GT mask > .5, uniform without replacement; includes empty render rays'
            else:
                eligible = np.flatnonzero(alpha.detach().cpu().numpy().reshape(-1) > args.opacity_threshold)
                sampling = f'No GT mask: rendered opacity > {args.opacity_threshold}, NOT verified foreground'
            count = min(args.samples, len(eligible))
            if not count:
                raise RuntimeError(f'View {vi}: no eligible rays')
            pixels = rng.choice(eligible, count, replace=False).astype(np.int32)
            extras = []
            for x, y in extra.get(vi, []):
                if not (0 <= x < W and 0 <= y < H):
                    raise ValueError('Extra pixel outside image')
                extras.append(y * W + x)
            # Extras may coincide with statistical rays; kept separately and never pooled.
            pixels = np.concatenate([pixels, np.asarray(extras, dtype=np.int32)])
            qualitative = np.arange(len(pixels)) >= count
            pix = torch.from_numpy(pixels).to(device=geom.device)
            K, retries = args.capacity, 0
            while True:
                rec, ids, counts, replay = extension.read_rays(
                    geom, binning, image, gaussians.get_xyz.shape[0], W, H, fx, fy, pix, K)
                maximum = int(counts.max())
                if maximum <= K:
                    break
                K, retries = maximum, retries + 1
            target = torch.stack([expected.flatten(), median.flatten(), alpha.flatten()], dim=1)[pix.long()]
            torch.testing.assert_close(replay, target, rtol=2e-5, atol=2e-6)
            errors = (replay - target).abs().cpu().numpy()
            counts_np = counts.cpu().numpy().astype(np.int64)
            offsets = np.r_[0, np.cumsum(counts_np)]
            keep = np.arange(K)[None, :] < counts_np[:, None]
            records = rec.cpu().numpy()[keep]
            gids = ids.cpu().numpy()[keep]
            valid = (np.isfinite(records[:, 4:7]).all(axis=1)
                     & (np.linalg.norm(records[:, 4:7], axis=1) > 1e-8) & (records[:, 7] == 0))
            # Independent normal replay from ALL complete saved records, including invalid normals.
            # Do not filter on normal_valid here: host rendering did not apply that filter either.
            normal_replay = np.zeros((len(pixels), 3), dtype=np.float32)
            for ri in range(len(pixels)):
                for j in range(offsets[ri], offsets[ri + 1]):
                    normal_replay[ri] += records[j, 4:7] * records[j, 1]
            normal_length = np.sqrt(np.sum(normal_replay * normal_replay, axis=1, dtype=np.float32))
            normal_replay /= np.maximum(normal_length, np.float32(1e-12))[:, None]
            normal_target = normal.reshape(3, -1).T[pix.long()].cpu().numpy()
            # Slightly looser than scalar replay: CPU separate mul/add versus CUDA FMA,
            # followed by normalization, can amplify cancellation near zero normals.
            torch.testing.assert_close(torch.from_numpy(normal_replay), torch.from_numpy(normal_target),
                                       rtol=2e-4, atol=2e-5, equal_nan=True)
            normal_error = np.abs(normal_replay - normal_target)
            # Preserve invalid source normals and matching host nonfinite outputs explicitly.
            # The assert still rejects finite/nonfinite mismatches and unequal infinities.
            normal_error_max = [float(col[np.isfinite(col)].max()) if np.isfinite(col).any() else None
                                for col in normal_error.T]
            finite_lengths = normal_length[np.isfinite(normal_length)]
            name = f'view_{vi:03d}.npz'
            # Each view is committed to disk before proceeding to the next one.
            temporary = output / (name + '.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, offsets=offsets, gaussian_id=gids,
                                depth=records[:, 0], weight=records[:, 1], alpha=records[:, 2],
                                transmittance=records[:, 3], normal=records[:, 4:7],
                                normal_valid=valid, plane_degenerate=records[:, 7].astype(bool),
                                pixel_xy=np.c_[pixels % W, pixels // W], qualitative=qualitative,
                                replay=replay.cpu().numpy(), host_target=target.cpu().numpy(),
                                normal_replay=normal_replay, normal_target=normal_target,
                                view_index=np.array(vi), image_size=np.array([H, W]))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, output / name)
            vm = dict(view_index=vi, image_name=camera.image_name, file=name, split=args.split,
                      sampling=sampling, mask_source=mask_source, eligible_pixels=len(eligible),
                      random_samples=count, qualitative_samples=len(extras), seed=args.seed,
                      image_size=[H, W], focal=[fx, fy], fov=[camera.FoVx, camera.FoVy],
                      world_view_transform=camera.world_view_transform.cpu().tolist(),
                      camera_center=camera.camera_center.cpu().tolist(), capacity=K,
                      overflow_retries=retries, total_members=len(gids), invalid_normal=int((~valid).sum()),
                      replay_max_abs=errors.max(axis=0).tolist(),
                      normal_replay_max_abs=normal_error_max,
                      normal_nonfinite_host_rays=int((~np.isfinite(normal_target).all(axis=1)).sum()),
                      normal_min_accumulated_length=float(finite_lengths.min()) if len(finite_lengths) else None,
                      npz_sha256=sha256(output / name))
            metadata['views'].append(vm)
            atomic_json(output / 'metadata.json', metadata)
            print(json.dumps(vm, ensure_ascii=False), flush=True)
            del result, rec, ids, counts, replay, target, geom, binning, image, rgb, normal
    if sha256(ply) != model_hash:
        raise RuntimeError('Original PLY hash changed during sampling')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m', '--model-path', required=True)
    p.add_argument('-s', '--source-path', help='Override stale cfg_args source_path')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--views', default='0', help='Comma-separated camera-list indices')
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--output', required=True)
    p.add_argument('--mask-dir', help='Optional explicit GT masks, missing files are errors')
    p.add_argument('--mask-template', default='{image_name}.png')
    p.add_argument('--opacity-threshold', type=float, default=.5)
    p.add_argument('--extra-pixel', action='append', default=[], help='view:x:y; qualitative only')
    p.add_argument('--capacity', type=int, default=256)
    p.add_argument('--host-build-provenance', required=True,
                   help='Operator-verified installed extension source/build origin; commit alone is insufficient')
    a = p.parse_args()
    if a.samples <= 0 or a.capacity <= 0:
        p.error('samples and capacity must be positive')
    sample(a)
