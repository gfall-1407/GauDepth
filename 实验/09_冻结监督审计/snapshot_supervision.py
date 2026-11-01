"""One frozen DTU camera-pair audit. No optimizer, densification, or model writes."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_text)
from scene.dataset_readers import readColmapCameras
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import depth_to_normal
from utils.loss_utils import PatchMatch
import diff_gaussian_rasterization as raster_module
import warp_patch_ncc


MODES = ('baseline', 'contribution_band', 'contribution_band_sg')
PARAMS = ('xyz', 'scaling', 'rotation', 'opacity')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_cameras(dataset, names):
    sparse = Path(dataset.source_path) / 'sparse' / '0'
    binary = all((sparse / name).is_file() for name in ('images.bin', 'cameras.bin'))
    camera_files = [sparse / ('images.bin' if binary else 'images.txt'),
                    sparse / ('cameras.bin' if binary else 'cameras.txt')]
    if binary:
        extr, intr = read_extrinsics_binary(str(camera_files[0])), read_intrinsics_binary(str(camera_files[1]))
    else:
        extr, intr = read_extrinsics_text(str(camera_files[0])), read_intrinsics_text(str(camera_files[1]))
    picked = {key: value for key, value in extr.items() if Path(value.name).stem in names}
    if len(picked) != len(names) or {Path(x.name).stem for x in picked.values()} != set(names):
        raise ValueError('camera pair missing or ambiguous in COLMAP images')
    infos = readColmapCameras(picked, intr, str(Path(dataset.source_path) / dataset.images))
    cameras = cameraList_from_camInfos(infos, 1., dataset)
    return {info.image_name: camera for info, camera in zip(infos, cameras)}, camera_files, infos


def masked_mean(values, selected):
    if values is None or not bool(selected.any()):
        return None
    return values[selected].mean()


def numeric(value):
    if value is None:
        return None
    number = float(value.detach().item())
    return number if math.isfinite(number) else 'nonfinite'


def gradient(loss, parameters):
    if loss is None or not loss.requires_grad:
        return None
    values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True,
                                 allow_unused=True)
    return {key: (None if grad is None else grad.detach().cpu().float().reshape(-1))
            for key, grad in zip(parameters, values)}


def gradient_summary(grads):
    if grads is None:
        return None
    return {key: (None if value is None else
                  dict(rms=(float(torch.linalg.vector_norm(value) / math.sqrt(value.numel()))
                            if bool(torch.isfinite(value).all()) else None),
                       finite=bool(torch.isfinite(value).all()), elements=value.numel()))
            for key, value in grads.items()}


def cosine(left, right):
    if left is None or right is None:
        return None
    dot = float(torch.dot(left, right))
    norm = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return None if norm == 0 or not math.isfinite(norm) else dot / norm


def assert_native_pm_parity(geo_loss, ncc_loss, audit):
    pm = audit['pm_mask']
    geo = masked_mean(audit['weights'] * audit['pixel_noise'], pm)
    ncc = masked_mean(audit['weights'] * audit['ncc_raw'], audit['ncc_mask'])
    if geo is not None and not torch.allclose(geo.reshape(()), geo_loss.reshape(()), atol=1e-6, rtol=1e-5):
        raise AssertionError('instrumented geometric loss differs from native PM')
    if ncc is not None and not torch.allclose(ncc.reshape(()), ncc_loss.reshape(()), atol=1e-6, rtol=1e-5):
        raise AssertionError('instrumented NCC loss differs from native PM')
    if geo is None and float(geo_loss.detach().item()) != 0:
        raise AssertionError('empty PM mask has a nonzero native geometric loss')
    if ncc is None and float(ncc_loss.detach().item()) != 0:
        raise AssertionError('empty NCC mask has a nonzero native NCC loss')


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument('--ply', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference_view', default='0049')
    parser.add_argument('--nearest_view', default='0012')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    dataset, optimization = model.extract(args), opt.extract(args)
    if (Path(dataset.source_path).name != 'scan110' or dataset.resolution != 2 or
            dataset.use_decoupled_appearance != 3 or args.reference_view != '0049' or
            args.nearest_view != '0012' or args.samples != 1024 or args.seed != 20261101):
        raise ValueError('this frozen audit is only scan110, r2, appearance3, 0049->0012, 1024 pixels, seed20261101')
    output, ply = Path(args.output).resolve(), Path(args.ply).resolve()
    if output.exists() or not ply.is_file() or args.samples < 1:
        raise ValueError('output must be new, PLY must exist, and samples must be positive')
    cameras, camera_files, infos = selected_cameras(dataset, (args.reference_view, args.nearest_view))
    reference, nearest = cameras[args.reference_view], cameras[args.nearest_view]
    if (reference.image_height, reference.image_width) != (nearest.image_height, nearest.image_width):
        raise ValueError('camera pair image dimensions differ')
    height, width = reference.image_height, reference.image_width
    if args.samples > height * width:
        raise ValueError('sample size exceeds image pixels')
    pixels = np.sort(np.random.default_rng(args.seed).choice(height * width, args.samples, replace=False))
    pixel_tensor = torch.from_numpy(pixels.astype(np.int64)).cuda()
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(ply))
    torch.cuda.reset_peak_memory_stats()
    parameters = {name: getattr(gaussians, '_' + name) for name in PARAMS}
    background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.],
                              device='cuda', dtype=torch.float32)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    ply=str(ply), ply_sha256=digest(ply), source=dataset.source_path,
                    camera_files={str(path): digest(path) for path in camera_files},
                    images={info.image_name: dict(path=info.image_path, sha256=digest(info.image_path))
                            for info in infos},
                    reference_view=args.reference_view, nearest_view=args.nearest_view,
                    resolution=dataset.resolution, use_decoupled_appearance=dataset.use_decoupled_appearance,
                    image_size=[height, width], samples=args.samples, seed=args.seed,
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    raster_binary=dict(path=raster_module._C.__file__, sha256=digest(raster_module._C.__file__)),
                    ncc_module=dict(path=warp_patch_ncc.__file__, sha256=digest(warp_patch_ncc.__file__)),
                    kernel_size=dataset.kernel_size, white_background=dataset.white_background,
                    regularization_weights=dict(normal=optimization.lambda_depth_normal,
                                                geo=optimization.lambda_multi_view_geo,
                                                ncc=optimization.lambda_multi_view_ncc),
                    method_names=MODES, not_evaluated=('full_median',),
                    no_optimizer=True, no_densification=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    np.save(output / 'flat_pixels.npy', pixels)
    def forward(mode, track_grad):
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False,
                               convert_SHs_python=False, depth_estimator=mode)
        patchmatch = PatchMatch(optimization.multi_view_patch_size,
                               optimization.multi_view_pixel_noise_th,
                               dataset.kernel_size, pipe, debug=False)
        context = torch.enable_grad() if track_grad else torch.no_grad()
        with context:
            pkg = render(reference, gaussians, pipe, background, dataset.kernel_size, require_depth=True)
            audit = {}
            ncc_loss, geo_loss = patchmatch(gaussians, pkg, reference, nearest,
                                           iteration=7001, depth_normal=None, audit=audit)
            assert_native_pm_parity(geo_loss, ncc_loss, audit)
            normal_map = depth_to_normal(reference, pkg['selected_depth'])
            normal_error = 1-torch.linalg.vecdot(pkg['normal'], normal_map, dim=0).reshape(-1)
            return pkg, audit, normal_error, geo_loss, ncc_loss

    frozen = {}
    t0 = time.perf_counter()
    for mode in MODES:
        pkg, audit, normal_error, geo_loss, ncc_loss = forward(mode, False)
        frozen[mode] = dict(pm=audit['pm_mask'][pixel_tensor].cpu(),
                            inside=audit['inside_mask'][pixel_tensor].cpu(),
                            reprojection_lt_threshold=audit['reprojection_lt_threshold'][pixel_tensor].cpu(),
                            ncc=audit['ncc_mask'][pixel_tensor].cpu(),
                            weight=audit['weights'][pixel_tensor].cpu(),
                            pixel_noise=audit['pixel_noise'][pixel_tensor].cpu(),
                            ncc_raw=None if audit['ncc_raw'] is None else audit['ncc_raw'][pixel_tensor].cpu(),
                            normal=normal_error[pixel_tensor].cpu(),
                            native_full_loss=dict(normal=numeric(normal_error.mean()),
                                                  geo=numeric(geo_loss), ncc=numeric(ncc_loss)))
        del pkg, audit, normal_error, geo_loss, ncc_loss
    common_pm = torch.stack([frozen[m]['pm'] for m in MODES]).all(0)
    common_ncc = torch.stack([frozen[m]['ncc'] for m in MODES]).all(0)
    result = dict(manifest_sha256=digest(output / 'manifest.json'),
                  common_pm_count=int(common_pm.sum()), common_ncc_count=int(common_ncc.sum()),
                  methods={}, gradients={}, cosine={}, within_method_cosine={},
                  seconds_no_grad=time.perf_counter()-t0)
    saved = dict(flat_pixel=pixels, common_pm=common_pm.numpy(), common_ncc=common_ncc.numpy())
    for mode in MODES:
        item = frozen[mode]
        pm, ncc = item['pm'], item['ncc']
        weighted_ncc = None if item['ncc_raw'] is None else item['weight']*item['ncc_raw']
        result['methods'][mode] = dict(inside_count=int(item['inside'].sum()),
                                       reprojection_lt_1px_count=int(item['reprojection_lt_threshold'].sum()),
                                       pm_count=int(pm.sum()), pm_denominator=args.samples,
                                       pm_weight_sum=float(item['weight'][pm].sum()),
                                       ncc_count=int(ncc.sum()), ncc_denominator=int(pm.sum()),
                                       ncc_weight_sum=float(item['weight'][ncc].sum()),
                                       native_full_loss=item['native_full_loss'],
                                       sampled_own_loss=dict(normal=numeric(masked_mean(item['normal'], pm)),
                                                             geo=numeric(masked_mean(item['weight']*item['pixel_noise'], pm)),
                                                             ncc=numeric(masked_mean(weighted_ncc, ncc))),
                                       sampled_common_loss=dict(normal=numeric(masked_mean(item['normal'], common_pm)),
                                                                geo=numeric(masked_mean(item['weight']*item['pixel_noise'], common_pm)),
                                                                ncc=numeric(masked_mean(weighted_ncc, common_ncc))))
        for key in ('inside', 'reprojection_lt_threshold', 'pm', 'ncc', 'weight', 'pixel_noise', 'normal'):
            saved[mode+'_'+key] = item[key].numpy()
        if item['ncc_raw'] is not None:
            saved[mode+'_ncc_raw'] = item['ncc_raw'].numpy()
    np.savez_compressed(output / 'sampled_masks_and_losses.npz', **saved)

    # Second pass preserves the same frozen masks; autograd.grad does not
    # accumulate into parameter .grad or update the model.
    gradients = {}
    for mode in MODES:
        pkg, audit, normal_error, _, _ = forward(mode, True)
        if not torch.equal(audit['pm_mask'][pixel_tensor].cpu(), frozen[mode]['pm']) or \
                not torch.equal(audit['ncc_mask'][pixel_tensor].cpu(), frozen[mode]['ncc']):
            raise AssertionError('PM acceptance changed between frozen and gradient passes')
        pm = common_pm.to(pixel_tensor.device)
        ncc = common_ncc.to(pixel_tensor.device)
        terms = dict(normal=masked_mean(normal_error[pixel_tensor], pm),
                     geo=masked_mean((audit['weights']*audit['pixel_noise'])[pixel_tensor], pm),
                     ncc=masked_mean(None if audit['ncc_raw'] is None else
                                     (audit['weights']*audit['ncc_raw'])[pixel_tensor], ncc))
        coefficients = dict(normal=optimization.lambda_depth_normal,
                            geo=optimization.lambda_multi_view_geo,
                            ncc=optimization.lambda_multi_view_ncc)
        terms = {name: None if value is None else coefficients[name]*value
                 for name, value in terms.items()}
        gradients[mode] = {}
        result['gradients'][mode] = {}
        for term, loss in terms.items():
            grads = gradient(loss, parameters)
            gradients[mode][term] = grads
            result['gradients'][mode][term] = gradient_summary(grads)
        # The normal loss has only two renderer inputs: selected depth and
        # native normal. Their VJPs separate the two gradient routes.
        normal_loss = terms['normal']
        if normal_loss is not None:
            route_tensors = dict(selected_depth=pkg['selected_depth'], native_normal=pkg['normal'])
            active_routes = {name: value for name, value in route_tensors.items() if value.requires_grad}
            upstream = torch.autograd.grad(normal_loss, tuple(active_routes.values()),
                                            retain_graph=True, allow_unused=True)
            upstream_by_name = dict(zip(active_routes, upstream))
            split = {}
            for label, tensor in route_tensors.items():
                upstream_grad = upstream_by_name.get(label)
                if upstream_grad is None:
                    split[label] = None
                else:
                    values = torch.autograd.grad(tensor, tuple(parameters.values()),
                                                 grad_outputs=upstream_grad, retain_graph=True,
                                                 allow_unused=True)
                    split[label] = {name: (None if value is None else value.detach().cpu().float().reshape(-1))
                                    for name, value in zip(parameters, values)}
            result['gradients'][mode]['normal_routes'] = {name: gradient_summary(value)
                                                          for name, value in split.items()}
            result['gradients'][mode]['normal_route_cosine'] = {
                name: cosine(None if split['selected_depth'] is None else split['selected_depth'][name],
                             None if split['native_normal'] is None else split['native_normal'][name])
                for name in PARAMS}
            total_normal = gradients[mode]['normal']
            result['gradients'][mode]['normal_route_sum_residual_rms'] = {}
            for name in PARAMS:
                total = None if total_normal is None else total_normal[name]
                if total is None:
                    result['gradients'][mode]['normal_route_sum_residual_rms'][name] = None
                    continue
                routed = sum((part[name] for part in split.values()
                              if part is not None and part[name] is not None),
                             torch.zeros_like(total))
                result['gradients'][mode]['normal_route_sum_residual_rms'][name] = numeric(
                    torch.linalg.vector_norm(total-routed) / math.sqrt(total.numel()))
        del pkg, audit, normal_error
    for mode in MODES:
        result['within_method_cosine'][mode] = {}
        for left, right in (('normal', 'geo'), ('normal', 'ncc'), ('geo', 'ncc')):
            result['within_method_cosine'][mode][left+'__'+right] = {
                name: cosine(gradients[mode][left][name], gradients[mode][right][name])
                if gradients[mode][left] is not None and gradients[mode][right] is not None
                else None for name in PARAMS}
    for left in MODES:
        for right in MODES:
            if left >= right:
                continue
            result['cosine'][left+'__'+right] = {
                term: {name: cosine(gradients[left][term][name], gradients[right][term][name])
                       if gradients[left][term] is not None and gradients[right][term] is not None
                       else None for name in PARAMS}
                for term in ('normal', 'geo', 'ncc')}
    torch.cuda.synchronize()
    result['seconds_total'] = time.perf_counter()-t0
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / (1024**3)
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise AssertionError('audit unexpectedly accumulated model parameter gradients')
    result['sampled_file_sha256'] = digest(output / 'sampled_masks_and_losses.npz')
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), common_pm=result['common_pm_count'],
                          common_ncc=result['common_ncc_count'], seconds=result['seconds_total'])))


if __name__ == '__main__':
    main()
