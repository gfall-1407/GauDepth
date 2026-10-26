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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=3600.)
    p.add_argument('--training-batch',type=Path)
    p.add_argument('--transfer-manifest',type=Path)
    p.add_argument('--teacher-metadata',type=Path)
    p.add_argument('--training-arms',nargs='+',default=['control','gap_train','xyz_train'])
    a=p.parse_args()
    if a.training_batch:
        if not a.transfer_manifest or not a.teacher_metadata:p.error('Transferred manifest and teacher metadata required')
        run_training(a)
    else:run(a)
