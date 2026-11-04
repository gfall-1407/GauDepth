"""Small CPU-only provenance and staging helpers for a frozen scene comparison."""
import ast
import hashlib
import json
from argparse import Namespace
from pathlib import Path
import shutil
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_cfg(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), mode='eval').body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != 'Namespace' or tree.args:
        raise ValueError('cfg must be a literal Namespace')
    if any(k.arg is None for k in tree.keywords):
        raise ValueError('cfg unpacking forbidden')
    return {k.arg: ast.literal_eval(k.value) for k in tree.keywords}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_profile(model, source, dtu, iteration, cfg, image_names, profile):
    from PIL import Image
    if source.name != profile['scene'] or iteration != profile['iteration']:
        raise ValueError('scene/iteration differs from reviewed profile')
    if image_names != profile['image_names']:
        raise ValueError('camera roster differs from reviewed profile')
    for key,value in profile['cfg'].items():
        if cfg.get(key) != value:
            raise ValueError('config differs from reviewed profile: '+key)
    ply=model/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    if sha256(ply) != profile['ply_sha256']:
        raise ValueError('wrong reviewed original PLY SHA256')
    count=None
    with ply.open('rb') as stream:
        for _ in range(512):
            line=stream.readline(1024).decode('ascii').strip()
            if line.startswith('element vertex '):count=int(line.split()[-1])
            if line=='end_header':break
        else:raise ValueError('invalid bounded PLY header')
    if count != profile['gaussians']:
        raise ValueError('wrong reviewed Gaussian count')
    for record in profile['hashes']:
        base={'source':source,'dtu':dtu}[record['root']]
        path=(base/record['relative']).resolve()
        if base not in path.parents or sha256(path) != record['sha256']:
            raise ValueError('reviewed data/calibration fingerprint mismatch: '+str(path))
    for name in image_names:
        with Image.open(source/'images'/(name+'.png')) as img:
            if list(img.size) != profile['raw_dimensions']:
                raise ValueError('source image dimensions changed: '+name)


def validate_inputs(model, source, dtu, iteration, profile=None):
    model, source, dtu = map(lambda p: Path(p).resolve(), (model, source, dtu))
    if iteration not in (7000,7500,8000,30000):
        raise ValueError('declare the frozen model stage explicitly: 7000/7500/8000/30000')
    scan = int(source.name.removeprefix('scan'))
    paths = [model/'cfg_args', model/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply',
             source/'sparse'/'0'/'cameras.bin', source/'sparse'/'0'/'images.bin',
             source/'sparse'/'0'/'points3D.ply',
             dtu/'Points'/'stl'/f'stl{scan:03}_total.ply',
             dtu/'ObsMask'/f'ObsMask{scan}_10.mat', dtu/'ObsMask'/f'Plane{scan}.mat']
    images = sorted((source/'images').glob('*'))
    images = [p for p in images if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    if not images or not (source/'sparse'/'0').is_dir():
        raise FileNotFoundError('full source/images and sparse/0 required')
    paths += images
    stems = [p.stem for p in images]
    if stems != [f'{i:04}' for i in range(len(images))]:
        raise ValueError('native evaluation assumes sequential 0000-based views; audit mapping first')
    paths += [dtu/'Calibration'/'cal18'/f'pos_{i:03}.txt' for i in range(1,65)]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    cfg = read_cfg(model/'cfg_args')
    if cfg.get('eval'):
        raise ValueError('require eval=False: all cameras for common DTU reconstruction/alignment')
    if profile is None:
        raise ValueError('explicit reviewed scene profile required')
    validate_profile(model,source,dtu,iteration,cfg,stems,profile)
    return dict(scan=scan, model=str(model), source=str(source), dtu=str(dtu),
                iteration=iteration, view_count=len(images), image_names=stems,
                reviewed_profile=profile,
                inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in paths})


def stage_model(model, destination, source, iteration):
    model, destination = Path(model).resolve(), Path(destination).resolve()
    if destination == model or model in destination.parents or destination in model.parents:
        raise ValueError('output must be isolated from original model')
    if destination.exists():
        raise FileExistsError(destination)
    original = model/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    target = destination/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    target.parent.mkdir(parents=True)
    shutil.copyfile(original, target)
    if sha256(original) != sha256(target):
        raise RuntimeError('staged PLY differs')
    cfg = read_cfg(model/'cfg_args')
    cfg.update(model_path=str(destination), source_path=str(Path(source).resolve()), eval=False)
    (destination/'cfg_args').write_text(str(Namespace(**cfg)), encoding='utf-8')
    return cfg


def camera_alignment(points, gt_points):
    """Same similarity fit as evaluate_dtu_mesh.py, with residuals reported."""
    p, g = np.asarray(points, float), np.asarray(gt_points, float)
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError('invalid camera correspondences')
    scale = np.linalg.norm(g-g.mean(0), axis=1).mean()/np.linalg.norm(p-p.mean(0), axis=1).mean()
    ps = p*scale
    u, _, vt = np.linalg.svd((ps-ps.mean(0)).T @ (g-g.mean(0)))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = g.mean(0)-r@ps.mean(0)
    residual = np.linalg.norm(ps@r.T+t-g, axis=1)
    if not np.all(np.isfinite(residual)) or residual.max() > 1.:
        raise ValueError('camera alignment residual exceeds fixed 1mm preflight threshold')
    return dict(scale_mm=float(scale), rotation=r.tolist(), translation_mm=t.tolist(),
                rms_mm=float(np.sqrt(np.mean(residual**2))), max_mm=float(residual.max()),
                source='camera centers only; no mesh/GT ICP or metric-dependent tuning')
