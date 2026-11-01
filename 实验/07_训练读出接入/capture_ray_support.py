"""Read-only, fixed-pixel accepted-support capture for one trained DTU view.

Writes only a new result directory. No training, optimizer, or source-data edit.
The captured full-K lists permit an offline group audit without keeping GPU on.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                 read_intrinsics_binary, read_intrinsics_text)
from scene.dataset_readers import readColmapCameras
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
import diff_gaussian_rasterization as raster_module


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def selected_camera(source, image_dir, view_name, dataset):
    sparse = Path(source) / 'sparse' / '0'
    if (sparse / 'images.bin').exists() and (sparse / 'cameras.bin').exists():
        extr = read_extrinsics_binary(str(sparse / 'images.bin'))
        intr = read_intrinsics_binary(str(sparse / 'cameras.bin'))
        camera_files = [sparse / 'images.bin', sparse / 'cameras.bin']
    else:
        extr = read_extrinsics_text(str(sparse / 'images.txt'))
        intr = read_intrinsics_text(str(sparse / 'cameras.txt'))
        camera_files = [sparse / 'images.txt', sparse / 'cameras.txt']
    matches = {key: value for key, value in extr.items()
               if Path(value.name).stem == view_name}
    if len(matches) != 1:
        raise ValueError(f'expected one COLMAP camera named {view_name}, found {len(matches)}')
    infos = readColmapCameras(matches, intr, str(Path(source) / image_dir))
    camera = cameraList_from_camInfos(infos, 1., dataset)[0]
    return camera, camera_files, Path(infos[0].image_path)


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument('--ply', required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--support-k', type=int, default=512)
    args = parser.parse_args()
    dataset = model.extract(args)
    ply, output = Path(args.ply).resolve(), Path(args.output).resolve()
    if not ply.is_file() or output.exists():
        raise FileExistsError('PLY must exist and output directory must be new')
    if args.samples < 1 or args.support_k < 128:
        raise ValueError('samples must be positive; diagnostic support-k must be >=128')
    camera, camera_files, image_file = selected_camera(dataset.source_path,
                                                       dataset.images, args.view, dataset)
    height, width = camera.image_height, camera.image_width
    if args.samples > height * width:
        raise ValueError('samples exceed image pixels')
    flat = np.sort(np.random.default_rng(args.seed).choice(height * width,
                                                            args.samples, replace=False))
    xy_numpy = np.stack((flat % width, flat // width), axis=-1).astype(np.float32)
    xy = torch.from_numpy(xy_numpy).cuda().contiguous()
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(ply))
    background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.],
                              dtype=torch.float32, device='cuda')
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
        kernel_size=dataset.kernel_size, bg=background, scale_modifier=1.,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree, campos=camera.camera_center,
        prefiltered=False, require_depth=True, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    with torch.no_grad():
        scale, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
        outputs = rasterizer(means3D=gaussians.get_xyz,
                             means2D=torch.zeros_like(gaussians.get_xyz),
                             shs=gaussians.get_features, colors_precomp=None,
                             opacities=opacity, scales=scale,
                             rotations=gaussians.get_rotation, cov3D_precomp=None,
                             support_xy=xy, support_k=args.support_k)
        if len(outputs) != 10:
            raise RuntimeError('reviewed support rasterizer must return 10 outputs')
        ids, alpha, count, overflow = outputs[6:]
        unique = torch.unique(ids[ids >= 0].long(), sorted=True)
        flat_cuda = torch.from_numpy(flat).to(device=xy.device, dtype=torch.long)
        captured = dict(flat_pixel=flat, xy=xy_numpy,
                        ids=ids.cpu().numpy(), alpha=alpha.cpu().numpy(),
                        count=count.cpu().numpy(), overflow=overflow.cpu().numpy(),
                        native_expected=outputs[2].reshape(-1).index_select(0, flat_cuda).cpu().numpy(),
                        native_median=outputs[3].reshape(-1).index_select(0, flat_cuda).cpu().numpy(),
                        unique_id=unique.cpu().numpy(),
                        center=gaussians.get_xyz[unique].cpu().numpy(),
                        quaternion=gaussians._rotation[unique].cpu().numpy(),
                        scale_unaa=gaussians.get_scaling[unique].cpu().numpy(),
                        world_view_transform=camera.world_view_transform.cpu().numpy(),
                        focal=np.array([camera.Fx, camera.Fy, camera.Cx, camera.Cy]))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'support.npz', **captured)
    binary = Path(raster_module._C.__file__)
    metadata = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_path=dataset.source_path, ply_path=str(ply), ply_sha256=sha256(ply),
                    colmap_sha256={str(p): sha256(p) for p in camera_files},
                    image_path=str(image_file), image_sha256=sha256(image_file),
                    view=args.view, resolution=dataset.resolution,
                    image_size=[height, width], samples=args.samples, seed=args.seed,
                    pixel_indices_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                    support_k=args.support_k, overflow=int(captured['overflow'].sum()),
                    host_binary=str(binary), host_binary_sha256=sha256(binary),
                    result_sha256=sha256(output / 'support.npz'))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), count=args.samples,
                          overflow=metadata['overflow'], result_sha256=metadata['result_sha256'])))


if __name__ == '__main__':
    main()
