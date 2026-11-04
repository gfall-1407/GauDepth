"""Frozen-model all-view depth/TSDF/DTU comparison; no training or source mutation.

Stage preflight is CPU-only; rendering uses the project's existing CUDA API.
Evaluation still runs evaluate_dtu_mesh.py -> dtu_eval/eval.py.
"""
import argparse
from argparse import Namespace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scene_compare_io import sha256, read_cfg, dump, stage_model, validate_inputs, camera_alignment

METHODS = ('expected', 'median', 'collective_half')


def preflight(args):
    profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    manifest = validate_inputs(args.model, args.source, args.dtu, args.iteration, profile)
    out = Path(args.output).resolve()
    for protected in (Path(args.model).resolve(), Path(args.source).resolve(), Path(args.dtu).resolve()):
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError('output overlaps protected input')
    if out.exists():
        raise FileExistsError('new output root required')
    print(json.dumps(manifest, indent=2))
    return manifest


def load_scene(staged, iteration):
    import torch
    from scene import GaussianModel, Scene
    from arguments import ModelParams, PipelineParams
    parser = argparse.ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    options = vars(parser.parse_args([]))
    options.update(read_cfg(Path(staged)/'cfg_args'))
    dataset, pipe = lp.extract(Namespace(**options)), pp.extract(Namespace(**options))
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, load_iteration=iteration, shuffle=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    return dataset, pipe, model, scene


def render_all(args, manifest):
    import torch
    from gaussian_renderer import render
    from collective_readout import render_collective_half
    from evaluate_dtu_mesh import load_dtu_camera
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    dump(root/'manifest.json', manifest)
    stage_model(args.model, root/'model', args.source, args.iteration)
    dataset, pipe, model, scene = load_scene(root/'model', args.iteration)
    cameras = list(scene.getTrainCameras())
    if [c.image_name for c in cameras] != manifest['image_names']:
        raise RuntimeError('loaded camera roster differs from audited all-view roster')
    profile=manifest['reviewed_profile']
    if len(model.get_xyz) != profile['gaussians']:
        raise RuntimeError('loaded model Gaussian count mismatch')
    roster=[]
    for c in cameras:
        if [c.image_width,c.image_height] != profile['render_dimensions']:
            raise RuntimeError('loaded camera resolution mismatch')
        intrinsic=[c.Fx,c.Fy,c.Cx,c.Cy]
        if not all(np.isfinite(x) for x in intrinsic) or c.Fx<=0 or c.Fy<=0:
            raise RuntimeError('invalid loaded intrinsics')
        if c.Cx!=(c.image_width-1)/2 or c.Cy!=(c.image_height-1)/2:
            raise RuntimeError('native principal-point convention changed')
        roster.append(dict(name=c.image_name,intrinsic=intrinsic,
            world_to_camera=c.world_view_transform.T.cpu().numpy().tolist()))
    dump(root/'camera_roster.json',roster)
    gt = load_dtu_camera(args.dtu)[:len(cameras)]
    centers = [torch.linalg.inv(c.world_view_transform.T)[:3,3].cpu().numpy() for c in cameras]
    alignment = camera_alignment(centers, [c[:,3] for c in gt])
    dump(root/'alignment.json', alignment)
    raster = importlib.import_module('diff_gaussian_rasterization')
    binding = Path(raster._C.__file__).resolve()
    if not binding.is_file():
        raise RuntimeError('loaded native extension missing')
    dump(root/'runtime.json', dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), native=str(binding),
        native_sha256=sha256(binding), wrapper_sha256=sha256(raster.__file__),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
    # Require a build receipt from the coordinator for the exact loaded binary.
    receipt = json.loads(Path(args.extension_receipt).read_text(encoding='utf-8'))
    if receipt.get('native_sha256') != sha256(binding):
        raise ValueError('extension receipt does not match loaded binary')
    if not receipt.get('source_revision') or not receipt.get('build_log_sha256'):
        raise ValueError('source revision and build log SHA required in extension receipt')
    dump(root/'extension_receipt.json', receipt)
    frames = root/'frames'
    frames.mkdir()
    background = torch.tensor([1.,1.,1.], dtype=torch.float32, device='cuda')
    results, started = [], time.monotonic()
    selected = cameras[:1] if args.smoke_one_view else cameras
    with torch.no_grad():
        for index, camera in enumerate(selected):
            tick = time.monotonic()
            native = render(camera, model, pipe, background, dataset.kernel_size)
            candidate, status, audit = render_collective_half(camera, model, pipe,
                background, dataset.kernel_size, batch_pixels=args.batch_pixels, steps=args.steps)
            depths = dict(expected=native['expected_depth'], median=native['median_depth'],
                          collective_half=candidate)
            mask = torch.ones_like(candidate, dtype=torch.bool)
            if camera.gt_mask is not None:
                mask = camera.gt_mask >= .5
            arrays = {}
            coverage = {}
            for method, depth in depths.items():
                raw_finite = torch.isfinite(depth)
                # Same finite/positive/TSDF depth range rule for every method.
                valid = raw_finite & (depth > 0) & (depth <= 8) & mask
                arrays[method] = torch.where(valid, depth, 0.)[0].cpu().numpy()
                coverage[method] = dict(all_pixels=depth.numel(), mask_pixels=int(mask.sum()),
                    valid_pixels=int(valid.sum()), nonfinite_pixels=int((~raw_finite).sum()))
            arrays['color'] = native['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
            arrays['mask'] = mask[0].cpu().numpy()
            arrays['candidate_status'] = status.cpu().numpy()
            arrays['intrinsic'] = np.array([[camera.Fx,0,camera.Cx],[0,camera.Fy,camera.Cy],[0,0,1]],float)
            arrays['extrinsic'] = camera.world_view_transform.T.cpu().numpy().astype(np.float64)
            np.savez(frames/(camera.image_name+'.npz'), **arrays)
            torch.cuda.synchronize()
            record = dict(view=camera.image_name, seconds=time.monotonic()-tick,
                coverage=coverage, candidate=audit)
            results.append(record)
            dump(root/'render_progress.json', dict(completed=len(results), total=len(cameras), views=results))
            print(json.dumps(dict(view=camera.image_name, completed=index+1, total=len(cameras),
                seconds=record['seconds'], coverage=coverage)), flush=True)
    # Protect source identity on both sides of this operation.
    for path, identity in manifest['inputs'].items():
        if sha256(path) != identity['sha256']:
            raise RuntimeError('protected source changed: '+path)
    record = dict(full_scene=not args.smoke_one_view, count=len(results),
                  expected=len(cameras), seconds=time.monotonic()-started, views=results)
    dump(root/'render_complete.json', record)
    return record


def require_full(root, args=None):
    record = json.loads((root/'render_complete.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    if not record['full_scene'] or record['count'] != record['expected']:
        raise RuntimeError('single-view smoke cannot enter scene evaluation')
    actual = sorted(p.stem for p in (root/'frames').glob('*.npz'))
    if actual != manifest['image_names']:
        raise RuntimeError('missing or extra full-image frames')
    if args is not None:
        for name in ('model','source','dtu'):
            if str(Path(getattr(args,name)).resolve()) != manifest[name]:
                raise ValueError('resumed '+name+' differs from original run')
        if args.iteration != manifest['iteration']:
            raise ValueError('resumed iteration differs from original run')
        for path, identity in manifest['inputs'].items():
            if sha256(path) != identity['sha256']:
                raise RuntimeError('protected input changed before fusion/evaluation: '+path)
    return manifest


def fuse_all(args):
    import open3d as o3d
    import open3d.core as o3c
    from mesh_extract import post_process_mesh
    root = Path(args.output).resolve()
    manifest = require_full(root,args)
    for method in METHODS:
        target = root/method
        stage_model(root/'model', target, args.source, args.iteration)
        device = o3c.Device('CPU:0')  # identical to original mesh_extract.py
        volume = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
            attr_dtypes=(o3c.float32,o3c.float32,o3c.float32), attr_channels=(1,1,3),
            voxel_size=.002, block_resolution=16, block_count=50000, device=device)
        for name in manifest['image_names']:
            with np.load(root/'frames'/(name+'.npz'), allow_pickle=False) as frame:
                depth = o3d.t.geometry.Image(np.ascontiguousarray(frame[method])).to(device)
                color = o3d.t.geometry.Image(np.ascontiguousarray(frame['color'])).to(device)
                intrinsic = o3c.Tensor(frame['intrinsic'])
                extrinsic = o3c.Tensor(frame['extrinsic'])
                blocks = volume.compute_unique_block_coordinates(depth,intrinsic,extrinsic,1.,8.)
                volume.integrate(blocks,depth,color,intrinsic,extrinsic,1.,8.)
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = mesh.to_legacy()
        if len(mesh.triangles) == 0:
            raise RuntimeError(method+': empty mesh; no metric may be substituted')
        if not o3d.io.write_triangle_mesh(str(target/'recon.ply'),mesh):
            raise RuntimeError('mesh write failed')
        # Original post_process_mesh has a threshold of >=50 triangles.
        mesh = post_process_mesh(mesh, 1)
        if len(mesh.triangles) == 0:
            raise RuntimeError(method+': empty postprocessed mesh')
        if not o3d.io.write_triangle_mesh(str(target/'recon_post.ply'),mesh):
            raise RuntimeError('postprocessed mesh write failed')
        dump(target/'fusion.json',dict(views=manifest['image_names'], voxel_size=.002,
            block_resolution=16,block_count=50000,depth_scale=1.,depth_max=8.,
            cluster_to_keep=1, implementation='same Open3D VoxelBlockGrid calls as mesh_extract.py',
            triangles=len(mesh.triangles),source_model_sha256=sha256(
                target/'point_cloud'/f'iteration_{args.iteration}'/'point_cloud.ply')))


def evaluate_all(args):
    root = Path(args.output).resolve()
    manifest = require_full(root,args)
    scores = {}
    for method in METHODS:
        target = root/method
        if (target/'vis'/'results.json').exists():
            raise FileExistsError('refuse to overwrite an existing metric')
        command = [sys.executable,'evaluate_dtu_mesh.py','-m',str(target),
                   '--iteration',str(args.iteration),'--DTU',args.dtu]
        with (target/'evaluate.log').open('x',encoding='utf-8') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        score_path = target/'vis'/'results.json'
        score = json.loads(score_path.read_text())
        if set(score) != {'mean_d2s','mean_s2d','overall'} or not all(np.isfinite(v) for v in score.values()):
            raise RuntimeError('invalid DTU metric; inspect native evaluation log')
        scores[method] = score
        dump(root/'scores.json',dict(scene=manifest['scan'],iteration=args.iteration,
             units='mm', criterion='DTU mesh only', scores=scores,
             complete=len(scores)==len(METHODS),
             early_screening_only=True, official_submission_claim=False))
    print(json.dumps(scores,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','source','dtu','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--iteration',type=int,default=7000)
    p.add_argument('--profile',default=str(Path(__file__).with_name('scan40_profile.json')))
    p.add_argument('--stage',choices=('preflight','render','fuse','evaluate','all'),default='preflight')
    p.add_argument('--extension-receipt')
    p.add_argument('--smoke-one-view',action='store_true')
    p.add_argument('--batch-pixels',type=int,default=131072)
    p.add_argument('--steps',type=int,default=22)
    args = p.parse_args()
    if args.smoke_one_view and args.stage not in ('render','preflight'):
        p.error('smoke is rendering only')
    if args.stage in ('render','all') and not args.extension_receipt:
        p.error('render requires the reviewed extension build receipt')
    if args.stage in ('preflight','render','all'):
        manifest = preflight(args)
    if args.stage in ('render','all'):
        render_all(args,manifest)
    if args.stage in ('fuse','all'):
        fuse_all(args)
    if args.stage in ('evaluate','all'):
        evaluate_all(args)


if __name__ == '__main__':
    main()
