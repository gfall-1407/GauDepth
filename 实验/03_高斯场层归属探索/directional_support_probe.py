"""CPU provenance, accepted-ray replay and directional cross-section probe.

The renderer uses a projected local ray-plane approximation, not the exact
ray/ellipsoid maximum. All errors below are diagnostics, not accuracy claims.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPO = Path(__file__).resolve().parents[2]
DATA = ROOT / '数据'
OUT = Path(__file__).resolve().parent / '结果' / '20261028_方向支持'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_sha(commit, path):
    blob = subprocess.check_output(['git', 'show', f'{commit}:{path.as_posix()}'], cwd=REPO)
    return hashlib.sha256(blob).hexdigest()


def indexed(suffix, index):
    matches = [x for x in index['files'] if x['path'].replace('\\', '/').endswith(suffix)]
    assert len(matches) == 1, (suffix, len(matches))
    p = DATA / Path(matches[0]['path'].replace('\\', '/'))
    assert sha(p) == matches[0]['sha256'], str(p)
    return p


def ply(path):
    with path.open('rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        assert 'format binary_little_endian 1.0' in header
        count = int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        props = [x.split()[-1] for x in header if x.startswith('property float ')]
        assert len(props) == sum(x.startswith('property ') for x in header)
        return np.fromfile(f, dtype=np.dtype([(p, '<f4') for p in props]), count=count)


def rotation(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(-1, 3, 3)


def section(mu, scales, rot, origin, direction):
    # In principal coordinates Q = R diag(1/s^2) R^T.
    d = np.einsum('nji,j->ni', rot, direction) / scales
    b = np.einsum('nji,nj->ni', rot, origin - mu) / scales
    a = np.sum(d*d, axis=1)
    return -np.sum(d*b, axis=1)/a, 1/np.sqrt(a)


def synthetic():
    eye = np.eye(3)[None]
    mu = np.array([[0., 0., 4.]])
    s = np.array([[1., 2., .5]])
    for d in ([0, 0, 1], [1, 0, 1], [1, 0, .01]):
        v = np.array(d, float); v /= np.linalg.norm(v)
        t, sig = section(mu, s, eye, np.zeros(3), v)
        expected = (4*v[2]/.25)/(v[0]**2 + v[1]**2/4 + v[2]**2/.25)
        assert np.allclose(t, expected) and np.allclose(sig, 1/np.sqrt(v[0]**2 + v[1]**2/4 + v[2]**2/.25))
    v = np.array([0., 0., 1.]); o = np.array([.7, .4, 0.])
    t, sig = section(mu, s, eye, o, v)
    assert np.allclose([t[0], sig[0]], [4, .5])
    angle = .73
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift = np.array([3., -2., 1.]); k = 2.5
    t2, s2 = section(k*(mu@R.T)+shift, k*s, R[None], k*(o@R.T)+shift, R@v)
    assert np.allclose(t2, k*t) and np.allclose(s2, k*sig)


def main():
    synthetic()
    index = json.loads((DATA / '来源索引.json').read_text(encoding='utf-8'))
    model = indexed('20261026_夜间深度方法/inputs/scan83/point_cloud.ply', index)
    trace = indexed('20261026_夜间深度方法/confirm_v1/scan83/view_032.npz', index)
    meta_path = indexed('20261026_夜间深度方法/confirm_v1/scan83/metadata.json', index)
    cfg = indexed('20261026_夜间深度方法/inputs/scan83/cfg_args', index)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert sha(model) == meta['model_sha256'] == '0c32a97c920e7eef1f0b11cc0a7e74b029222246d347026f5ed2e8c3df83cc00'
    assert sha(cfg) == meta['cfg_sha256']
    view = next(v for v in meta['views'] if v['view_index'] == 32)
    assert sha(trace) == view['npz_sha256'] == '8a752fa19d9fabfc329fecd8742ad9b98ea64e3dae0a579c0de6f8609ef39569'
    f = np.load(trace); vertices = ply(model)
    ids = f['gaussian_id']; offsets = f['offsets']; assert ids.min() >= 0 and ids.max() < len(vertices)
    chosen = np.linspace(0, len(offsets)-2, 32, dtype=int)
    replay = []; errors = {'filtered': [], 'unfiltered': []}; sigmas = {'filtered': [], 'unfiltered': []}
    W = np.array(view['world_view_transform'], dtype=float)
    origin = np.array(view['camera_center'], dtype=float)
    assert np.allclose(np.r_[origin, 1] @ W, [0, 0, 0, 1], atol=1e-5)
    h, w = view['image_size']; fx, fy = view['focal']
    for i in chosen:
        sl = slice(offsets[i], offsets[i+1]); depth = f['depth'][sl]; weight = f['weight'][sl]
        alpha = f['alpha'][sl]; trans = f['transmittance'][sl]
        T = np.float32(1); D = np.float32(0); M = np.float32(0)
        for z, a, wt, recorded_t in zip(depth, alpha, weight, trans):
            assert np.isclose(T, recorded_t, rtol=2e-5, atol=2e-6)
            assert np.isclose(np.float32(T*a), wt, rtol=2e-5, atol=2e-6)
            D = np.float32(D + np.float32(z*wt))
            if T > .5: M = z
            T = np.float32(T*np.float32(1-a))
        opacity = np.float32(1-T)
        replay.append([float(D/opacity) if len(depth) else 0., float(M), float(opacity)])
        p = vertices[ids[sl]]
        if not len(p): continue
        mu = np.stack([p[n] for n in ('x','y','z')], axis=1).astype(float)
        q = np.stack([p[f'rot_{j}'] for j in range(4)], axis=1).astype(float)
        scales = np.exp(np.stack([p[f'scale_{j}'] for j in range(3)], axis=1).astype(float))
        r = rotation(q); x, y = f['pixel_xy'][i]
        vcam = np.array([(x-(w-1)/2)/fx, (y-(h-1)/2)/fy, 1.]); vcam /= np.linalg.norm(vcam)
        v = W[:3,:3] @ vcam; v /= np.linalg.norm(v)
        for name, ss in [('unfiltered', scales), ('filtered', np.sqrt(scales**2 + p['filter_3D'][:,None].astype(float)**2))]:
            t, s = section(mu, ss, r, origin, v)
            errors[name].extend((t*vcam[2] - depth).tolist()); sigmas[name].extend((s*vcam[2]).tolist())
    replay = np.array(replay); target = f['host_target'][chosen]; saved = f['replay'][chosen]
    source = {}
    for name, recorded in meta['source_sha256'].items():
        actual = sha(Path(__file__).parent / name)
        relative = (Path(__file__).parent / name).relative_to(REPO)
        blob = git_sha(meta['git_commit'], relative)
        source[name] = {'recorded': recorded, 'current_worktree': actual, 'recorded_commit_blob': blob, 'recorded_commit_matches': recorded == blob}
    hostsrc = REPO / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu'
    hosthash = sha(hostsrc)
    hostblob = git_sha(meta['git_commit'], hostsrc.relative_to(REPO))
    stats = lambda x: {'count': len(x), 'abs_p50': float(np.quantile(np.abs(x), .5)), 'abs_p95': float(np.quantile(np.abs(x), .95)), 'abs_max': float(np.max(np.abs(x)))}
    report = {'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
      'input_sha256': {'model': sha(model), 'trace': sha(trace), 'metadata': sha(meta_path), 'cfg': sha(cfg)},
      'reader_sources': source, 'host_forward': {'recorded': meta['host_source_sha256']['cuda_rasterizer/render_forward.cu'], 'current_worktree': hosthash, 'recorded_commit_blob': hostblob, 'binary': meta['host_binary_sha256'], 'provenance': meta['host_build_provenance']},
      'model_vertices': len(vertices), 'selected_ray_indices': chosen.tolist(), 'selected_members': int(sum(offsets[i+1]-offsets[i] for i in chosen)),
      'id_bounds_valid': True, 'id_identity_note': 'bounds and model hash only; cannot independently prove reader ID correspondence without source/binary provenance',
      'replay_max_abs_host': np.max(np.abs(replay-target), axis=0).tolist(), 'replay_max_abs_saved': np.max(np.abs(replay-saved), axis=0).tolist(),
      'replay_matches_host': bool(np.allclose(replay, target, rtol=2e-5, atol=2e-6)),
      'exact_section_minus_recorded_depth': {n: stats(np.array(v)) for n,v in errors.items()},
      'section_sigma_z': {n: {'p50': float(np.quantile(v,.5)), 'p95': float(np.quantile(v,.95))} for n,v in sigmas.items()},
      'synthetic_tests': 'passed: frontal, oblique, grazing, off-center, rigid transform and scale',
      'limitations': 'Host installed binary source not established. Render ray-plane is local linear approximation; exact Gaussian ray maximum need not equal trace. No GT or algorithm evaluation.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'probe.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert report['replay_matches_host'], 'Host replay mismatch'


if __name__ == '__main__':
    main()
