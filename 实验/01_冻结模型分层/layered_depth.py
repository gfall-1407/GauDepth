"""Frozen host forward, complete batched ray readout, offline grouping."""
import os
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from diff_gaussian_rasterization import _C

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MAX_JOBS', '2')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
(HERE / '构建').mkdir(exist_ok=True)
extension = load(name='gaudepth_layered_reader', sources=[str(HERE / 'ray_trace_layered.cu')],
                 build_directory=str(HERE / '构建'),
                 extra_cuda_cflags=['-O3','--use_fast_math'], verbose=True)

def protect_scene_output(output):
    """Redirect the host Scene's unconditional neighbor metadata write only."""
    import scene
    import builtins
    def scene_open(file, mode='r', *a, **kw):
        if 'w' in mode and os.path.basename(os.fspath(file)) == 'multi_view.json':
            file = str(Path(output) / 'multi_view_layered.json')
        return builtins.open(file, mode, *a, **kw)
    scene.Scene.__init__.__globals__['open'] = scene_open

def forward(camera, gaussians, pipe, background, kernel):
    # Exact host Gaussian renderer arguments, using its installed forward.
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    empty = torch.empty(0, device='cuda')
    tx, ty = math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5)
    result = _C.rasterize_gaussians(background, gaussians.get_xyz, empty, opacity,
        scales, gaussians.get_rotation, 1.0, empty, camera.world_view_transform,
        camera.full_proj_transform, tx, ty, kernel, camera.image_height,
        camera.image_width, gaussians.get_features, gaussians.active_sh_degree,
        camera.camera_center, False, True, pipe.debug)
    # The host C++ forward computes these from float tan_fov arguments.
    fx=np.float32(np.float32(camera.image_width)/(np.float32(2)*np.float32(tx)))
    fy=np.float32(np.float32(camera.image_height)/(np.float32(2)*np.float32(ty)))
    return result, float(fx), float(fy)

def depth_maps(camera, gaussians, pipe, background, kernel, methods, validate=False):
    r, fx, fy = forward(camera, gaussians, pipe, background, kernel)
    _, rgb, alpha, normal, expected, median, _, geom, binning, image = r
    H, W = camera.image_height, camera.image_width
    maps = {'expected': expected.cpu(), 'median': median.cpu()}
    active = [m for m in methods if m.startswith('gap_') or m == 'contribution_valley']
    stats = dict(pixels=H*W, invalid=0, adaptive_fallback=0, overflow_retries=0,
                 multi_group=[0,0,0], winner_share_sum=[0.,0.,0.], nonempty=0)
    if active:
        all_depth = np.empty((H*W,3),np.float32)
        expected_flat = expected.flatten().cpu().contiguous()
        target = torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],dim=1)
        for start in range(0,H*W,8192):
            N=min(8192,H*W-start); K=1024
            while True:
                rec, count, replay = extension.read_rays(geom,binning,image,
                    gaussians.get_xyz.shape[0],W,H,fx,fy,start,N,K)
                maximum=int(count.max())
                if maximum<=K: break
                stats['overflow_retries']+=1; K=maximum
            if validate:
                torch.testing.assert_close(replay,target[start:start+N],rtol=2e-5,atol=2e-6)
            out, st = extension.aggregate(rec.cpu(),count.cpu(),expected_flat[start:start+N],
                                          'contribution_valley' in active)
            if validate:
                for m in range(3):
                    single=st[:,2+m]==1
                    torch.testing.assert_close(out[single,m],expected_flat[start:start+N][single],
                                               rtol=2e-5,atol=2e-6)
            all_depth[start:start+N]=out.numpy()
            stats['invalid']+=int(st[:,0].sum());stats['adaptive_fallback']+=int(st[:,1].sum())
            stats['nonempty']+=int((st[:,2]>0).sum())
            for m in range(3):
                stats['multi_group'][m]+=int((st[:,2+m]>1).sum())
                stats['winner_share_sum'][m]+=float(st[:,5+m].sum())
        for i,m in enumerate(['gap_fixed','gap_adaptive','contribution_valley']):
            if m in active: maps[m]=torch.from_numpy(all_depth[:,i].copy().reshape(1,H,W))
    return rgb.cpu(),maps,stats
