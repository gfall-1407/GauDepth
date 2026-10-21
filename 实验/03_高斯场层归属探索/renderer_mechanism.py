"""Synthetic Gaussian origin probes through the unchanged host; no model/data loading.

Labels identify construction origin only, never physical membership in a trained
model. Angular/centre perturbations destroy exact common-plane geometry.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def cases():
    result = []
    for kind, angles, gaps in [('single', [0.], [0.]),
                               ('parallel', [0., 0.], [.005, .02]),
                               ('tilted', [-15., 15.], [.01, .03])]:
        for gap, dz, dn in itertools.product(gaps, (0., .002), (0., 15., 30.)):
            result.append(dict(kind=kind, theta_deg=angles, gap=gap,
                               center_noise=dz, normal_noise_deg=dn,
                               seed=20261022 + len(result), case_index=len(result)))
    return result


def generate(cfg):
    rng = np.random.default_rng(cfg['seed'])
    x, y = np.meshgrid(np.linspace(-.01, .01, 7), np.linspace(-.01, .01, 7))
    origin = np.repeat(np.arange(len(cfg['theta_deg']), dtype=np.int32), 49)
    x, y = np.tile(x.ravel(), len(cfg['theta_deg'])), np.tile(y.ravel(), len(cfg['theta_deg']))
    theta = np.deg2rad(np.asarray(cfg['theta_deg'])[origin])
    z = 1. + origin * cfg['gap'] - np.tan(theta) * x
    z += rng.normal(0, cfg['center_noise'], len(z))
    base = np.c_[np.sin(theta), np.zeros(len(z)), np.cos(theta)]
    tangent = np.c_[np.cos(theta), np.zeros(len(z)), -np.sin(theta)]
    delta = rng.normal(0, np.deg2rad(cfg['normal_noise_deg']), len(z))
    azimuth = rng.uniform(0, 2 * np.pi, len(z))
    normals = np.cos(delta)[:, None] * base + np.sin(delta)[:, None] * (
        np.cos(azimuth)[:, None] * tangent + np.sin(azimuth)[:, None] * [0., 1., 0.])
    # Unit wxyz quaternion taking +z to n, including the antipodal special case.
    quat = np.c_[1 + normals[:, 2], -normals[:, 1], normals[:, 0], np.zeros(len(z))]
    antipodal = np.linalg.norm(quat, axis=1) < 1e-12
    quat[antipodal] = [0., 1., 0., 0.]
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    return dict(means=np.c_[x, y, z].astype(np.float32),
                scales=np.tile([.006, .006, .0005], (len(z), 1)).astype(np.float32),
                rotations=quat.astype(np.float32), opacity=np.full((len(z), 1), .06, np.float32),
                source_origin=origin, generated_normal=normals.astype(np.float32))


def separation(cfg, xs):
    t = np.tan(np.deg2rad(cfg['theta_deg']))
    return float(np.min(cfg['gap'] + (t[0] - t[1]) * xs))


def self_check():
    configs = cases()
    assert len(configs) == 30
    assert [sum(c['kind'] == k for c in configs) for k in ('single', 'parallel', 'tilted')] == [6, 12, 12]
    minimum = []
    for cfg in configs:
        data = generate(cfg)
        q, n = data['rotations'], data['generated_normal']
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=2e-7)
        w, a, b, c = q.T
        z_axis = np.c_[2 * (a * c + w * b), 2 * (b * c - w * a), 1 - 2 * (a*a + b*b)]
        np.testing.assert_allclose(z_axis, n, atol=3e-7)
        origin, means = data['source_origin'], data['means']
        if cfg['center_noise'] == 0:
            theta = np.deg2rad(np.asarray(cfg['theta_deg'])[origin])
            plane_z = 1 + origin * cfg['gap'] - np.tan(theta) * means[:, 0]
            np.testing.assert_allclose(means[:, 2], plane_z, rtol=0, atol=1e-7)
        if cfg['normal_noise_deg'] == 0:
            theta = np.deg2rad(np.asarray(cfg['theta_deg'])[origin])
            np.testing.assert_allclose(n, np.c_[np.sin(theta), np.zeros(len(n)), np.cos(theta)], atol=1e-7)
        if cfg['kind'] == 'tilted':
            grid_sep = separation(cfg, np.array([-.01, .01]))
            # Ray-plane intersections for x pixel 28..36, using host pixel centre.
            slope = (np.arange(28, 37) - 31.5) / (64 / (2 * np.tan(.04)))
            t = np.tan(np.deg2rad(cfg['theta_deg']))
            z0, z1 = 1 / (1 + t[0] * slope), (1 + cfg['gap']) / (1 + t[1] * slope)
            assert grid_sep > 0 and np.min(z1 - z0) > 0
            minimum.append(grid_sep)
    print(json.dumps(dict(self_check='passed', cases=30, rays_per_case=81,
                          minimum_nominal_tilted_grid_separation=min(minimum),
                          torch_imported='torch' in sys.modules)))


def atomic_npz(path, data):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def sync_dir(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def analytic_peak(data, rays):
    """Exact Mahalanobis-density maximum along each ray; NOT the host rasterizer."""
    mu, scale = data['means'].astype(float), data['scales'].astype(float)
    q = data['rotations'].astype(float)
    q /= np.linalg.norm(q, axis=1)[:, None]
    qw, qx, qy, qz = q.T
    axis = np.c_[2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy)]
    assert np.allclose(scale[:,0],scale[:,1])
    a = 1/scale[:,0]**2
    b = 1/scale[:,2]**2-a
    nd = rays@axis.T
    nm = np.sum(axis*mu,axis=1)
    denominator = np.sum(rays*rays,axis=1)[:,None]*a+nd*nd*b
    z = ((rays@mu.T)*a+nd*nm*b)/denominator
    residual = z[:,:,None]*rays[:,None,:]-mu[None,:,:]
    mahal = np.sum(residual*residual,axis=2)*a+np.sum(residual*axis,axis=2)**2*b
    alpha = np.minimum(.99,data['opacity'].ravel()*np.exp(-.5*mahal))
    oriented = axis*np.where(nm>0,-1.,1.)[:,None]
    return z,alpha,oriented


def run_analytic(args):
    from sample import git,sha256,atomic_json
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    px,py=np.meshgrid(np.arange(28,37),np.arange(28,37))
    xy=np.c_[px.ravel(),py.ravel()]
    focal=64/(2*np.tan(.04))
    rays=np.c_[(xy[:,0]-31.5)/focal,(xy[:,1]-31.5)/focal,np.ones(len(xy))]
    meta=dict(schema=1,status='running',backend='analytic_ray_peak_gaussian_CPU',
        codeHEAD=git('rev-parse','HEAD'),git_status=git('status','--short'),
        git_remote=git('remote','get-url','origin'),hostname=platform.node(),
        sourceSHA={'renderer_mechanism.py':sha256(__file__)},numpy=np.__version__,args=vars(args),
        mechanism='z=argmin_t (t*d-mu)^T Sigma^-1(t*d-mu), alpha=opacity*exp(-minimum_Mahalanobis/2)',
        caveat='Independent analytical ray-density mechanism, NOT RaDe-GS projected 2D rasterization. '
        'No host/CUDA replay validation is claimed. Labels indicate construction origins only.',
        acceptance='Stable center-z order; cap alpha .99, discard alpha<1/255, reject current member and stop if next T<1e-4',
        sampling='fixed 81 pixels 28..36, image64, FoV.08, identity camera',
        planned_cases=cases()[args.start:args.start+args.limit],cases=[])
    atomic_json(output/'metadata.json',meta)
    for cfg in meta['planned_cases']:
        data=generate(cfg);z,alpha,normal=analytic_peak(data,rays)
        order=np.argsort(data['means'][:,2],kind='stable')
        offsets=[0];ids=[];records=[];replay=[]
        for ri in range(len(rays)):
            T,D,M=1.,0.,0.
            for i in order:
                al=float(alpha[ri,i])
                if al<1/255:continue
                next_T=T*(1-al)
                if next_T<1e-4:break
                w=T*al
                ids.append(int(i));records.append([z[ri,i],w,al,T,*normal[i]])
                D+=w*z[ri,i]
                if T>.5:M=z[ri,i]
                T=next_T
            offsets.append(len(ids));replay.append([D/(1-T) if T<1 else 0.,M,1-T])
        rec=np.asarray(records,dtype=float).reshape(-1,7);gids=np.asarray(ids,np.int32)
        filename=f"case_{cfg['case_index']:03d}.npz"
        atomic_npz(output/filename,dict(**data,offsets=np.asarray(offsets),gaussian_id=gids,
            depth=rec[:,0],weight=rec[:,1],alpha=rec[:,2],transmittance=rec[:,3],normal=rec[:,4:7],
            normal_valid=np.ones(len(rec),bool),plane_degenerate=np.zeros(len(rec),bool),
            truth_origin=data['source_origin'][gids],pixel_xy=xy,qualitative=np.zeros(len(rays),bool),
            replay=np.asarray(replay),image_size=np.array([64,64])))
        meta['cases'].append(dict(config=cfg,file=filename,status='complete',npz_sha256=sha256(output/filename),
                                  total_members=len(rec),validation='analytic identities only; no host replay'))
        atomic_json(output/'metadata.json',meta)
        print('analytic Gaussian case',cfg['case_index'],'members',len(rec),flush=True)
    meta['status']='complete';atomic_json(output/'metadata.json',meta)


def run(args):
    if args.backend=='analytic_cpu':
        return run_analytic(args)
    # All GPU dependencies are deliberately deferred beyond --self-check.
    from sample import forward, git, sha256, atomic_json
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output is not an empty directory: {output}')
    if platform.node() != 'autodl-container-d6a7459179-693ab6ee':
        raise RuntimeError('GPU run is restricted to the explicitly authorized G1 hostname')
    import torch
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C
    from torch.utils.cpp_extension import load
    from utils.graphics_utils import getProjectionMatrix
    import importlib.metadata

    # reader.cu uses a checked byte-buffer layout, not a public host ABI.
    verified_host_sha = 'f97dcd66f55d9ddbf673c1d467238fe6f983b9f0bf062645a87bb12d20aa91df'
    host_sha = sha256(_C.__file__)
    if host_sha != verified_host_sha:
        raise RuntimeError(f'Host binary differs from the reader-verified binary: {host_sha}')
    output.mkdir(parents=True, exist_ok=True)
    build = HERE / '构建'
    build.mkdir(exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    reader = load(name='gaudepth_layers03_reader_v1', sources=[str(HERE / 'reader.cu')],
                  build_directory=str(build), extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=True)
    camera = SimpleNamespace(FoVx=.08, FoVy=.08, image_width=64, image_height=64,
                             world_view_transform=torch.eye(4, device='cuda'),
                             full_proj_transform=getProjectionMatrix(.01, 100, .08, .08).transpose(0, 1).cuda(),
                             camera_center=torch.zeros(3, device='cuda'))
    px, py = np.meshgrid(np.arange(28, 37), np.arange(28, 37))
    pixels = (py * 64 + px).ravel().astype(np.int32)
    pix = torch.from_numpy(pixels).cuda()
    try:
        direct_url = importlib.metadata.distribution('diff_gaussian_rasterization').read_text('direct_url.json')
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    host_files = ['cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/render_forward.cu', 'cuda_rasterizer/config.h']
    metadata = dict(schema=1, status='running', codeHEAD=git('rev-parse', 'HEAD'),
        git_status=git('status', '--short'), git_remote=git('remote', 'get-url', 'origin'),
        hostname=platform.node(), sourceSHA={n: sha256(HERE / n) for n in
            ('renderer_mechanism.py', 'sample.py', 'reader.cu')},
        hostbinarySHA=host_sha, reader_verified_host_SHA=verified_host_sha, host_binary=_C.__file__,
        host_package=diff_gaussian_rasterization.__file__, host_direct_url=direct_url,
        buildprovenance='GAP: previously verified direct_url source path /root/autodl-tmp/RaDe-GS/'
            'submodules/diff-gaussian-rasterization is missing; current Git HEAD does not prove installed binary source. '
            'Same-buffer scalar and normal replay only checks observed compatibility.',
        host_source_sha256={n: sha256(ROOT / 'submodules/diff-gaussian-rasterization' / n) for n in host_files},
        reader_binary=reader.__file__, reader_binary_sha256=sha256(reader.__file__), reader_build=str(build),
        cuda_flags=['-O3', '--use_fast_math'], torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), python=sys.version, numpy=np.__version__, args=vars(args),
        construction=dict(grid_side=7, xy_range=[-.01, .01], scales=[.006, .006, .0005], opacity=.06,
            base_z=1., features='zeros(N,1,3)', sh_degree=0, quaternion='wxyz; local +z to perturbed normal',
            center_noise='independent Gaussian camera-z noise; xy fixed',
            normal_noise='signed Gaussian angular displacement, uniform tangent azimuth (mechanism_sweep law)'),
        camera=dict(size=[64, 64], fov=[.08, .08], near=.01, far=100., kernel_size=0., background=[0, 0, 0],
            world_view_transform=camera.world_view_transform.cpu().tolist(),
            full_proj_transform=camera.full_proj_transform.cpu().tolist(), camera_center=[0, 0, 0]),
        sampling='fixed 9x9 grid x,y=28..36; no mask; separate from real 3072-ray samples',
        truth='construction origin only; perturbations destroy common planes; NOT physical surface truth',
        normal_semantics='host camera-space oriented geometric direction; not generated_normal',
        normal_invalid='nonfinite, norm <= 1e-8, or zero ray-plane host fallback',
        replay_tolerance=dict(rtol=2e-5, atol=2e-6), normal_replay_tolerance=dict(rtol=2e-4, atol=2e-5,
            normalize_eps=1e-12, reduction='CPU float32 multiply/add in original acceptance order; all members'),
        planned_cases=cases()[args.start:args.start + args.limit], cases=[])
    atomic_json(output / 'metadata.json', metadata)
    with torch.no_grad():
        for cfg in metadata['planned_cases']:
            data = generate(cfg)
            tensor = lambda a: torch.from_numpy(a).cuda()
            gaussians = SimpleNamespace(get_xyz=tensor(data['means']),
                get_scaling_n_opacity_with_3D_filter=(tensor(data['scales']), tensor(data['opacity'])),
                get_rotation=tensor(data['rotations']), get_features=torch.zeros((len(data['means']), 1, 3), device='cuda'),
                active_sh_degree=0)
            result, fx, fy = forward(camera, gaussians, SimpleNamespace(debug=False), torch.zeros(3, device='cuda'), 0.)
            _, rgb, alpha, normal, expected, median, _, geom, binning, image = result
            capacity, capacities = 128, []
            while True:
                capacities.append(capacity)
                rec, ids, counts, replay = reader.read_rays(geom, binning, image, len(data['means']), 64, 64, fx, fy, pix, capacity)
                maximum = int(counts.max())
                if maximum <= capacity:
                    break
                capacity = maximum
            target = torch.stack([expected.flatten(), median.flatten(), alpha.flatten()], dim=1)[pix.long()]
            torch.testing.assert_close(replay, target, rtol=2e-5, atol=2e-6)
            count = counts.cpu().numpy().astype(np.int64)
            offsets = np.r_[0, np.cumsum(count)]
            keep = np.arange(capacity)[None, :] < count[:, None]
            records, gids = rec.cpu().numpy()[keep], ids.cpu().numpy()[keep]
            assert np.all((gids >= 0) & (gids < len(data['means'])))
            valid = (np.isfinite(records[:, 4:7]).all(axis=1) &
                     (np.linalg.norm(records[:, 4:7], axis=1) > 1e-8) & (records[:, 7] == 0))
            normal_replay = np.zeros((len(pixels), 3), np.float32)
            for ri in range(len(pixels)):
                for j in range(offsets[ri], offsets[ri + 1]):
                    normal_replay[ri] += records[j, 4:7] * records[j, 1]
            length = np.sqrt(np.sum(normal_replay * normal_replay, axis=1, dtype=np.float32))
            normal_replay /= np.maximum(length, np.float32(1e-12))[:, None]
            normal_target = normal.reshape(3, -1).T[pix.long()].cpu().numpy()
            np.testing.assert_allclose(normal_replay, normal_target, rtol=2e-4, atol=2e-5, equal_nan=True)
            name = f"case_{cfg['case_index']:03d}.npz"
            atomic_npz(output / name, dict(**data, offsets=offsets, gaussian_id=gids,
                depth=records[:, 0], weight=records[:, 1], alpha=records[:, 2], transmittance=records[:, 3],
                normal=records[:, 4:7], normal_valid=valid, plane_degenerate=records[:, 7].astype(bool),
                truth_origin=data['source_origin'][gids], pixel_xy=np.c_[pixels % 64, pixels // 64],
                qualitative=np.zeros(len(pixels), bool), replay=replay.cpu().numpy(), host_target=target.cpu().numpy(),
                normal_replay=normal_replay, normal_target=normal_target, image_size=np.array([64, 64])))
            record = dict(config=cfg, file=name, status='complete', capacity_history=capacities,
                total_members=len(gids), focal=[fx, fy], invalid_normal=int((~valid).sum()),
                replay_max_abs=(replay - target).abs().max(dim=0).values.cpu().tolist(),
                normal_replay_max_abs=[float(v.max()) if len(v) else None for v in
                    [a[np.isfinite(a)] for a in np.abs(normal_replay - normal_target).T]],
                nominal_grid_min_separation=separation(cfg, np.array([-.01, .01])) if len(cfg['theta_deg']) == 2 else None,
                npz_sha256=sha256(output / name))
            metadata['cases'].append(record)
            atomic_json(output / 'metadata.json', metadata)
            sync_dir(output)
            print(json.dumps(record), flush=True)
            del result, rec, ids, counts, replay, target, geom, binning, image, rgb, normal, gaussians
    metadata['status'] = 'complete'
    atomic_json(output / 'metadata.json', metadata)
    sync_dir(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='New or empty synthetic-result directory')
    parser.add_argument('--backend',choices=('host','analytic_cpu'),default='host',
                        help='host requires G1 GPU; analytic_cpu is an independent model, not host validation')
    parser.add_argument('--start', type=int, default=0, help='Starting case index; original seeds and indices retained')
    parser.add_argument('--limit', type=int, default=30, help='Number of configurations from --start, one seed each')
    parser.add_argument('--self-check', action='store_true', help='Pure NumPy geometry checks; no GPU imports or output')
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1 or args.start + args.limit > 30:
        parser.error('Require start >= 0, limit >= 1, and start + limit <= 30')
    if args.self_check:
        self_check()
    elif not args.output:
        parser.error('--output required unless --self-check')
    else:
        run(args)
