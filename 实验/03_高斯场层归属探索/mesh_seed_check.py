"""CPU-only repeat scoring of fixed, already aligned meshes; no new reconstruction."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from sample import atomic_json, git, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
METHODS = ("native_expected", "huber_1mm")
SEEDS = (20261026, 20261027, 20261028)
CHILD = r'''
import sys, random
from pathlib import Path
from multiprocessing.pool import ThreadPool
import numpy as np
import open3d as o3d
script, seed = sys.argv[1], int(sys.argv[2])
sys.argv = [script, *sys.argv[3:]]
random.seed(seed)
np.random.seed(seed)
o3d.io.write_point_cloud = lambda *a, **kw: True
source = Path(script).read_text(encoding="utf-8")
assert source.count("np.random.default_rng()") == 1
assert source.count("mp.Pool()") == 1
source = source.replace("np.random.default_rng()", f"np.random.default_rng({seed})")
source = source.replace("mp.Pool()", "ThreadPool(processes=8)")
scope = sys.modules["__main__"].__dict__
scope.update(__file__=script, ThreadPool=ThreadPool)
exec(compile(source, script, "exec"), scope)
'''

AUDIT_AFTER_SCORE = r'''
# Post-computation inspection only: results.json was already written above.
def _audit_distribution(values):
    x=np.asarray(values).reshape(-1)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid scored distances')
    edges=[0.,.2,.5,1.,2.,5.,10.,20.,np.inf]
    counts=np.histogram(x,bins=edges)[0]
    sums=[float(x[(x>=lo)&(x<hi)].sum()) for lo,hi in zip(edges[:-1],edges[1:])]
    retained=x<max_dist
    return dict(n=len(x),below_cutoff_n=int(retained.sum()),excluded_n=int((~retained).sum()),
        native_truncated_mean=float(x[retained].mean()),untruncated_mean=float(x.mean()),
        capped_mean=float(np.minimum(x,max_dist).mean()),
        quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(x,[0,.5,.9,.95,.99,1.]).tolist())),
        bin_edges_mm=['0','.2','.5','1','2','5','10','20','inf'],bin_counts=counts.tolist(),bin_distance_sums_mm=sums)
_audit_d2s=_audit_distribution(dist_d2s);_audit_s2d=_audit_distribution(dist_s2d)
if not np.isclose(_audit_d2s['native_truncated_mean'],mean_d2s,rtol=1e-12,atol=1e-12):raise ValueError('D2S inspection changed the scored quantity')
if not np.isclose(_audit_s2d['native_truncated_mean'],mean_s2d,rtol=1e-12,atol=1e-12):raise ValueError('S2D inspection changed the scored quantity')
np.savez_compressed(Path(args.vis_out_dir)/'retained_scoring_points.npz',
    d2s_points_mm=data_in_obs,d2s_distance_mm=dist_d2s[:,0],
    s2d_gt_index=np.flatnonzero(above),s2d_distance_mm=dist_s2d[:,0])
_audit=dict(input_vertices_n=len(vertices),input_triangles_n=len(triangles),
    nonzero_triangles_n=len(area2),mesh_area_mm2=float(area2.sum()/2),
    surface_sample_before_thinning_n=len(data_pcd),after_thinning_n=len(data_down),
    bbox_n=len(data_in),grid_inbound_n=int(grid_inbound.sum()),obs_mask_n=len(data_in_obs),
    gt_total_n=len(stl),gt_above_plane_n=len(stl_above),max_distance_mm=float(max_dist),
    downsample_radius_mm=float(thresh),d2s=_audit_d2s,s2d=_audit_s2d)
import json as _audit_json
(Path(args.vis_out_dir)/'score_diagnostics.json').write_text(_audit_json.dumps(_audit,indent=2),encoding='utf-8')
'''


def run(args):
    if git("status", "--short") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require committed clean approved source")
    out = args.output.resolve()
    if not args.output.is_absolute() or out.exists():
        raise ValueError("Require new absolute output")
    out.mkdir(parents=True)
    evaluator = ROOT / "dtu_eval/eval.py"
    evaluator_sha = __import__("hashlib").sha256(evaluator.read_text(encoding="utf-8").encode()).hexdigest()
    cases = {}
    for scan in (24, 37, 106):
        path = args.root / ("dense_pilot_v1" if scan == 24 else f"dense_confirm_v1/scan{scan}_v1")
        m = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if m["status"] != "complete" or m["dtu_eval_sha256"] != evaluator_sha:
            raise ValueError("Incomplete primary experiment or different normalized evaluator source")
        for rel in (f"Points/stl/stl{scan:03d}_total.ply", f"ObsMask/ObsMask{scan}_10.mat", f"ObsMask/Plane{scan}.mat"):
            if sha256(args.root / "score_inputs" / rel) != m["gt_inputs_sha256"][rel]:
                raise ValueError(f"Independent scan input mismatch: {rel}")
        for method in METHODS:
            if sha256(path / method / "recon_aligned.ply") != m["results"][method]["aligned_sha256"]:
                raise ValueError("Frozen aligned mesh mismatch")
        cases[scan] = (path, m)
    state = dict(status="partial", source_commit=git("rev-parse", "HEAD"), seeds=list(SEEDS), jobs=[],
        evaluator_normalized_sha256=evaluator_sha, python=sys.version,
        packages={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "open3d")},
        protocol="Original DTU score source and byte-identical already Calibration-aligned meshes. Only fixed RNG seed and executor change: ThreadPool8 on Windows replaces Pool8; sample_single_tri has no random/shared mutable state, ordered map is retained. Both methods use the same backend and seed. Local-versus-primary seed26 differences are recorded rather than silently mixed. No GPU, rendering, training, mesh fitting or new hyperparameters.")
    atomic_json(out / "metadata.json", state)
    deadline = min(time.monotonic()+args.max_seconds,
        time.monotonic()+dt.datetime.fromisoformat("2026-10-26T07:58:00+08:00").timestamp()-time.time())
    try:
        for scan, (path, primary) in cases.items():
            for seed in SEEDS:
                for method in METHODS:
                    remaining = deadline-time.monotonic()
                    if remaining < 10 or (out / "stop_after_score").exists():
                        raise TimeoutError("Fixed batch deadline or requested complete-score boundary")
                    dest = out / f"scan{scan}_{method}_{seed}"
                    dest.mkdir()
                    mesh = path / method / "recon_aligned.ply"
                    command = [sys.executable, "-c", CHILD, str(evaluator), str(seed), "--data", str(mesh),
                        "--scan", str(scan), "--mode", "mesh", "--dataset_dir", str(args.root / "score_inputs"), "--vis_out_dir", str(dest)]
                    with (dest / "evaluation.log").open("w") as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        state["active"] = dict(scan=scan, method=method, seed=seed, pid=child.pid)
                        atomic_json(out / "metadata.json", state)
                        try:
                            code = child.wait(timeout=min(600, remaining))
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                            raise TimeoutError("Evaluation child stopped at its independent time budget")
                    if code:
                        raise RuntimeError(f"Evaluation failed: {dest}")
                    score = json.loads((dest / "results.json").read_text(encoding="utf-8"))
                    row = dict(scan=scan, method=method, seed=seed, **score,
                        primary_overall=primary["results"][method]["scores"]["overall"],
                        mesh_sha256=sha256(mesh), results_sha256=sha256(dest / "results.json"))
                    state["jobs"].append(row)
                    atomic_json(out / "metadata.json", state)
                    print(json.dumps(row), flush=True)
        with (out / "scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(state["jobs"][0])); writer.writeheader(); writer.writerows(state["jobs"])
        state["status"] = "complete"
        atomic_json(out / "metadata.json", state)
    except BaseException as exc:
        state.update(status="partial", error=f"{type(exc).__name__}: {exc}")
        atomic_json(out / "metadata.json", state)
        raise


def run_training(args):
    """Repeat the scorer on transferred fixed training meshes, without rerendering."""
    if git('status','--porcelain') or git('remote','get-url','origin')!='git@github.com:gfall-1407/GauDepth.git':
        raise ValueError('Require clean committed approved source')
    out=args.output.resolve()
    if not args.output.is_absolute() or out.exists():raise ValueError('New independent output required')
    batch=args.training_batch.resolve();manifest=json.loads(args.transfer_manifest.read_text(encoding='utf-8'))
    if Path(manifest['root']).name!=batch.name:raise ValueError('Transferred batch root differs')
    source_files={x['file']:x['sha256'] for x in manifest['files']}
    arms=args.training_arms;prepared={};scan=None
    evaluator=ROOT/'dtu_eval/eval.py'
    normalized=__import__('hashlib').sha256(evaluator.read_text(encoding='utf-8').encode()).hexdigest()
    teacher=json.loads(args.teacher_metadata.read_text(encoding='utf-8'))
    if teacher['dtu_eval_sha256']!=normalized:raise ValueError('Native evaluation source differs')
    for arm in arms:
        training=json.loads((batch/arm/'metadata.json').read_text(encoding='utf-8'))
        if training['status']!='complete' or training['actual_updates']!=1000:raise ValueError('Partial training')
        this_scan=int(training['args']['scan'])
        if scan is not None and scan!=this_scan:raise ValueError('Mixed scenes in batch')
        scan=this_scan;relative=f'{arm}_native/recon_aligned.ply';mesh=batch/relative
        if sha256(mesh)!=source_files[relative]:raise ValueError('Transferred aligned mesh SHA differs')
        primary_path=batch/f'{arm}_native/metadata.json'
        if sha256(primary_path)!=source_files[f'{arm}_native/metadata.json']:raise ValueError('Primary result metadata differs')
        prepared[arm]=(mesh,json.loads(primary_path.read_text(encoding='utf-8'))['scores'])
    canonical=json.loads((HERE/'G1_Huber全场景输入参考.json').read_text(encoding='utf-8'))['scans'][str(scan)]
    for rel in (f'Points/stl/stl{scan:03d}_total.ply',f'ObsMask/ObsMask{scan}_10.mat',f'ObsMask/Plane{scan}.mat'):
        if sha256(args.root/rel)!=canonical['gt_inputs'][rel]:raise ValueError('Canonical scoring input differs')
    out.mkdir();state=dict(status='partial',source_commit=git('rev-parse','HEAD'),scan=scan,seeds=list(SEEDS),
        arms=arms,jobs=[],evaluator_normalized_sha256=normalized,transfer_manifest_sha256=sha256(args.transfer_manifest),
        protocol='Fixed transferred Calibration-aligned meshes; original scorer with fixed RNG and Windows ordered ThreadPool8, visualization disabled. No model or mesh update. Every seed26 arm must reproduce its remote primary within1e-9 before other-seed evidence is accepted.')
    atomic_json(out/'metadata.json',state);deadline=time.monotonic()+args.max_seconds
    try:
        for seed in SEEDS:
            for arm,(mesh,primary) in prepared.items():
                remaining=deadline-time.monotonic()
                if remaining<10 or (out/'stop_after_score').exists():raise TimeoutError('Full-score boundary stop/deadline')
                dest=out/f'scan{scan}_{arm}_{seed}';dest.mkdir()
                command=[sys.executable,'-c',CHILD,str(evaluator),str(seed),'--data',str(mesh),'--scan',str(scan),
                    '--mode','mesh','--dataset_dir',str(args.root),'--vis_out_dir',str(dest)]
                with (dest/'evaluation.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    state['active']=dict(seed=seed,arm=arm,pid=child.pid);atomic_json(out/'metadata.json',state)
                    try:code=child.wait(timeout=min(600,remaining))
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait();raise TimeoutError('One independent scorer child timed out')
                if code:raise RuntimeError('Scoring failed '+str(dest))
                score=json.loads((dest/'results.json').read_text(encoding='utf-8'))
                row=dict(scan=scan,arm=arm,seed=seed,**score,mesh_sha256=sha256(mesh),
                    results_sha256=sha256(dest/'results.json'),primary_overall=primary['overall'])
                state['jobs'].append(row);atomic_json(out/'metadata.json',state)
                if seed==20261026 and any(abs(score[k]-primary[k])>1e-9 for k in ('mean_d2s','mean_s2d','overall')):
                    raise ValueError('Local seed26/remote primary differs; do not mix scorer backends')
                print(json.dumps(row),flush=True)
        state.update(status='complete',active=None);atomic_json(out/'metadata.json',state)
        with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(state['jobs'][0]));writer.writeheader();writer.writerows(state['jobs'])
    except BaseException as exc:
        state.update(status='partial',error=repr(exc));atomic_json(out/'metadata.json',state);raise


def run_frozen_audit(args):
    """Inspect exactly two original aligned E/gap meshes and preserve distances."""
    if git('status','--porcelain') or git('remote','get-url','origin')!='git@github.com:gfall-1407/GauDepth.git':
        raise ValueError('Require clean committed approved source')
    root=args.frozen_score_audit.resolve();out=args.output.resolve()
    if not args.output.is_absolute() or out.exists():raise ValueError('New independent diagnostic output required')
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'));scan=int(meta['scan'])
    manifest=json.loads(args.transfer_manifest.read_text(encoding='utf-8'))
    if meta['status']!='complete' or Path(manifest['root']).name!=root.parent.name:
        raise ValueError('Expected complete transferred frozen native scene')
    source_files={x['file']:x['sha256'] for x in manifest['files']}
    if sha256(root/'metadata.json')!=source_files[f'{root.name}/metadata.json']:
        raise ValueError('Primary native metadata SHA differs')
    evaluator=ROOT/'dtu_eval/eval.py'
    normalized=__import__('hashlib').sha256(evaluator.read_text(encoding='utf-8').encode()).hexdigest()
    if normalized!=meta['dtu_eval_sha256']:raise ValueError('Native evaluator source differs')
    canonical=json.loads((HERE/'G1_Huber全场景输入参考.json').read_text(encoding='utf-8'))['scans'][str(scan)]
    for rel in (f'Points/stl/stl{scan:03d}_total.ply',f'ObsMask/ObsMask{scan}_10.mat',f'ObsMask/Plane{scan}.mat'):
        if sha256(args.root/rel)!=canonical['gt_inputs'][rel]:raise ValueError('Canonical evaluation input differs')
    for method in ('native_expected','gap_huber'):
        mesh=root/method/'recon_aligned.ply';relative=f'{root.name}/{method}/recon_aligned.ply'
        if sha256(mesh)!=source_files[relative] or sha256(mesh)!=meta['results'][method]['aligned_sha256']:
            raise ValueError('Original aligned mesh SHA differs')
    out.mkdir();state=dict(status='partial',source_commit=git('rev-parse','HEAD'),scan=scan,seed=20261026,jobs=[],
        evaluator_normalized_sha256=normalized,transfer_manifest_sha256=sha256(args.transfer_manifest),
        protocol='Original scorer and fixed seed26 with ordered ThreadPool8. Diagnostics only read arrays after the original three metrics were written. No change to meshes, GT, alignment, sampling, masks or cutoffs. Ray-NN and mesh denominators remain distinct; histogram differences are descriptive, not causal shares.')
    atomic_json(out/'metadata.json',state);deadline=time.monotonic()+args.max_seconds
    try:
        for method in ('native_expected','gap_huber'):
            remaining=deadline-time.monotonic()
            if remaining<10 or (out/'stop_after_score').exists():raise TimeoutError('Full-score boundary stop/deadline')
            dest=out/method;dest.mkdir();mesh=root/method/'recon_aligned.ply'
            command=[sys.executable,'-c',CHILD+AUDIT_AFTER_SCORE,str(evaluator),'20261026','--data',str(mesh),
                '--scan',str(scan),'--mode','mesh','--dataset_dir',str(args.root),'--vis_out_dir',str(dest)]
            with (dest/'evaluation.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['active']=dict(method=method,pid=child.pid);atomic_json(out/'metadata.json',state)
                try:code=child.wait(timeout=min(600,remaining))
                except subprocess.TimeoutExpired:
                    child.kill();child.wait();raise TimeoutError('Bounded diagnostic scorer timed out')
            if code:raise RuntimeError('Native score inspection failed: '+method)
            score=json.loads((dest/'results.json').read_text(encoding='utf-8'));primary=meta['results'][method]['scores']
            if any(abs(score[k]-primary[k])>1e-9 for k in ('mean_d2s','mean_s2d','overall')):
                raise ValueError('Inspection does not reproduce original native score; do not compare distributions')
            row=dict(method=method,scores=score,mesh_sha256=sha256(mesh),
                diagnostics=json.loads((dest/'score_diagnostics.json').read_text(encoding='utf-8')),
                results_sha256=sha256(dest/'results.json'),diagnostics_sha256=sha256(dest/'score_diagnostics.json'),
                retained_points_sha256=sha256(dest/'retained_scoring_points.npz'))
            state['jobs'].append(row);atomic_json(out/'metadata.json',state)
            print(json.dumps(dict(method=method,scores=score,d2s_n=row['diagnostics']['d2s']['n'])),flush=True)
        state.update(status='complete',active=None);atomic_json(out/'metadata.json',state)
    except BaseException as exc:
        state.update(status='partial',error=repr(exc));atomic_json(out/'metadata.json',state);raise


def run_cross_surface(args):
    """Locate the fixed2-20mm error tail relative to the other evaluated pointset."""
    import numpy as np
    from scipy.spatial import cKDTree
    if git('status','--porcelain') or git('remote','get-url','origin')!='git@github.com:gfall-1407/GauDepth.git':
        raise ValueError('Require clean committed approved source')
    root=args.cross_surface_audit.resolve();out=args.output.resolve()
    if not args.output.is_absolute() or out.exists():raise ValueError('New independent output required')
    primary=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if primary['status']!='complete' or len(primary['jobs'])!=2:raise ValueError('Complete two-method score inspection required')
    records={r['method']:r for r in primary['jobs']};data={}
    for name in ('native_expected','gap_huber'):
        file=root/name/'retained_scoring_points.npz'
        if sha256(file)!=records[name]['retained_points_sha256']:raise ValueError('Retained scored points SHA differs')
        with np.load(file) as z:points=z['d2s_points_mm'];errors=z['d2s_distance_mm']
        if points.shape!=(len(errors),3) or not np.isfinite(points).all() or not np.isfinite(errors).all():
            raise ValueError('Invalid retained scoring arrays')
        data[name]=(points,errors)
    out.mkdir();state=dict(status='partial',source_commit=git('rev-parse','HEAD'),completed=[],
        protocol='All source D2S points in the already fixed2<=error<20mm histogram tail. Exact nearest neighbor in the OTHER method retained ObsMask pointset; no GT query, geometry update or fitted threshold. Radius.2mm/native thinning and1mm/prior physical band are descriptive cuts only. Associations are not triangle identities or causal surface creation; excluded other-method points may affect proximity.')
    atomic_json(out/'metadata.json',state);start=time.monotonic()
    try:
        for source,target in (('gap_huber','native_expected'),('native_expected','gap_huber')):
            points,errors=data[source];other,other_error=data[target];index=np.flatnonzero((errors>=2)&(errors<20))
            tree=cKDTree(other,leafsize=32);distance=[];nearest=[]
            for pos in range(0,len(index),4096):
                if time.monotonic()-start>args.max_seconds or (out/'stop_after_chunk').exists():
                    raise TimeoutError('Bounded full-chunk stop/deadline')
                d,i=tree.query(points[index[pos:pos+4096]],k=1,eps=0,workers=1)
                distance.append(d);nearest.append(i);state['active']=dict(source=source,processed=min(pos+4096,len(index)),total=len(index));atomic_json(out/'metadata.json',state)
            distance=np.concatenate(distance);nearest=np.concatenate(nearest);delta=errors[index]-other_error[nearest]
            violations=np.abs(delta)>distance+1e-8
            if violations.any():raise ValueError('Same-GT NN1-Lipschitz consistency failed')
            far=distance>1.;near=~far
            def avg(x):return float(x.mean()) if len(x) else None
            row=dict(source=source,target=target,n=len(index),
                cross_distance_mm_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(distance,[0,.5,.9,.95,.99,1]).tolist())),
                within_native_radius_0p2_n=int((distance<=.2).sum()),within_1mm_n=int(near.sum()),beyond_1mm_n=int(far.sum()),
                source_tail_mean_mm=avg(errors[index]),nearest_other_mean_mm=avg(other_error[nearest]),
                source_error_better_n=int((delta<0).sum()),source_error_worse_n=int((delta>0).sum()),
                beyond_1mm_source_mean_mm=avg(errors[index][far]),beyond_1mm_nearest_other_mean_mm=avg(other_error[nearest][far]),
                nearest_other_below2mm_n=int((other_error[nearest]<2).sum()),lipschitz_violations=0)
            file=out/(source+'_to_'+target+'.npz');np.savez_compressed(file,source_index=index,other_index=nearest,cross_distance_mm=distance)
            row['matches_sha256']=sha256(file);state['completed'].append(row);atomic_json(out/'metadata.json',state);print(json.dumps(row),flush=True)
        state.update(status='complete',active=None);atomic_json(out/'metadata.json',state)
    except BaseException as exc:
        state.update(status='partial',error=repr(exc));atomic_json(out/'metadata.json',state);raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=3600.)
    p.add_argument('--training-batch',type=Path)
    p.add_argument('--transfer-manifest',type=Path)
    p.add_argument('--teacher-metadata',type=Path)
    p.add_argument('--training-arms',nargs='+',default=['control','gap_train','xyz_train'])
    p.add_argument('--frozen-score-audit',type=Path)
    p.add_argument('--cross-surface-audit',type=Path)
    a=p.parse_args()
    if a.cross_surface_audit:run_cross_surface(a)
    elif a.frozen_score_audit:
        if not a.transfer_manifest:p.error('Transferred manifest required')
        run_frozen_audit(a)
    elif a.training_batch:
        if not a.transfer_manifest or not a.teacher_metadata:p.error('Transferred manifest and teacher metadata required')
        run_training(a)
    else:run(a)
