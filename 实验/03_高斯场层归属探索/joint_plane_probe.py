"""Finite joint 3D tangent-plane hypotheses, known K; analytic mechanism only."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze import pair_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy(means,normal,scale,normal_weight):
    means=np.asarray(means,float);normal=np.asarray(normal,float)
    if means.ndim!=2 or means.shape[1]!=3 or normal.shape!=means.shape or not len(means):raise ValueError('Nonempty (N,3) required')
    if not np.isfinite(means).all() or not np.isfinite(normal).all() or np.any(np.linalg.norm(normal,axis=1)<=1e-8):raise ValueError('Finite means and nonzero finite normals required')
    normal=normal/np.linalg.norm(normal,axis=1)[:,None]
    # Candidate h is an observed Gaussian's tangent plane, with no truth planes.
    residual=np.einsum('ihj,hj->ih',means[:,None,:]-means[None,:,:],normal)
    e=2.5*np.log1p((residual/scale)**2/4.)
    theta=np.arccos(np.clip(np.abs(normal@normal.T),0.,1.))
    e+=normal_weight*np.minimum((theta/np.deg2rad(30.))**2,1.)
    return e


def fit(means,normal,K,scale=.002,normal_weight=0.):
    if K not in (1,2) or scale<=0 or not np.isfinite(scale):raise ValueError('K1/2 and positive scale required')
    e=energy(means,normal,scale,normal_weight);size=len(e)
    candidates=list(itertools.combinations_with_replacement(range(size),K))
    values=np.array([e[:,list(pair)].min(axis=1).mean() for pair in candidates])
    at=int(np.argmin(values));chosen=candidates[at];best=float(values[at])
    selected=e[:,list(chosen)];labels=np.argmin(selected,axis=1)
    if K==2:labels[np.abs(selected[:,0]-selected[:,1])<=1e-10]=-1
    return dict(labels=labels,objective=best,centers=chosen,
                near_optimal_count=int((values<=best+1e-10).sum()),
                hypotheses=len(candidates))


def self_check():
    means=np.array([[0.,0.,1.],[.01,0.,1.],[0.,.01,1.],[0.,0.,1.02],[.01,0.,1.02],[0.,.01,1.02]])
    n=np.tile([0.,0.,1.],(6,1));truth=np.repeat([0,1],3)
    result=fit(means,n,2)
    assert pair_metrics(truth,result['labels'])['f1']==1 and result['objective']==0.
    # Independent explicit normal-distance calculation and direct finite enumeration.
    e=np.array([[2.5*np.log1p((np.dot(means[i]-means[h],n[h])/.002)**2/4.) for h in range(6)] for i in range(6)])
    np.testing.assert_allclose(energy(means,n,.002,0.),e,atol=1e-12)
    brute=min(sum(min(e[i,a],e[i,b]) for i in range(6))/6 for a in range(6) for b in range(a,6))
    assert result['objective']==brute
    q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))
    for mm,nn,scale in ((means@q+np.array([3.,4.,5.]),n@q,.002),(means*1000,n,2.),(means,-n,.002)):
        got=fit(mm,nn,2,scale)
        assert np.isclose(got['objective'],result['objective'],atol=1e-12)
        assert pair_metrics(truth,got['labels'])['f1']==1
    print('joint plane self-check passed',flush=True)


def run(input_dir,baseline,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    meta=json.loads((input_dir/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status']=='complete' and meta['backend']=='analytic_ray_peak_gaussian_CPU'
    out.mkdir(parents=True,exist_ok=True);rows=[];details=[];inputs=[]
    for item in meta['cases']:
        config=item['config'];path=input_dir/item['file'];assert sha(path)==item['npz_sha256']
        labelpath=baseline/f"labels_{config['case_index']:03d}.json"
        old=json.loads(labelpath.read_text(encoding='utf-8'));assert old['source_sha256']==sha(path)
        inputs.append(dict(path=str(path),sha256=sha(path),baseline=str(labelpath),baseline_sha256=sha(labelpath)))
        with np.load(path,allow_pickle=False) as z:
            ids=np.unique(z['gaussian_id']);assert len(ids)>0
            means=z['means'][ids].astype(float);normal=z['generated_normal'][ids].astype(float)
            truth=z['source_origin'][ids];K=len(np.unique(truth))
            source_labels={gid:k for k,gid in enumerate(ids.tolist())}
            for scale,lam in itertools.product((.002,.005),(0.,1.)):
                result=fit(means,normal,K,scale,lam);labels=result['labels']
                per_ray=[];baseline_scores=[];oracle_mismatch=0
                for record in old['rays']:
                    local=np.array([labels[source_labels[g]] for g in record['gaussian_id']])
                    actual=np.asarray(record['truth_origin']);score=pair_metrics(actual,local)['f1']
                    oracle_mismatch+=len(np.unique(actual))!=K
                    assert score is not None, 'Require at least one true pair for this mechanism probe'
                    per_ray.append(score)
                    name='conditional_normal_uniform' if lam else 'conditional_depth_uniform'
                    baseline_scores.append(pair_metrics(actual,record['labels'][name])['f1'])
                score=pair_metrics(truth,labels)
                rows.append(dict(case_index=config['case_index'],kind=config['kind'],gap=config['gap'],
                    center_noise=config['center_noise'],normal_noise_deg=config['normal_noise_deg'],
                    scale=scale,normal_weight=lam,accepted_unique_gaussians=len(ids),oracle_K=K,
                    global_pair_f1=score['f1'],global_false_merge_pairs=score['fp'],global_false_split_pairs=score['fn'],
                    mean_projected_ray_pair_f1=float(np.mean(per_ray)),
                    mean_baseline_conditional_ray_pair_f1=float(np.mean(baseline_scores)),
                    union_vs_ray_oracle_K_mismatches=int(oracle_mismatch),
                    projected_minus_baseline=float(np.mean(per_ray)-np.mean(baseline_scores)),
                    groups=len(set(labels.tolist())-{-1}),unknown_gaussians=int((labels<0).sum()),
                    objective=result['objective'],near_optimal_count=result['near_optimal_count']))
                details.append(dict(case_index=config['case_index'],scale=scale,normal_weight=lam,
                    gaussian_ids=ids.tolist(),labels=labels.tolist(),center_gaussian_ids=[int(ids[i]) for i in result['centers']]))
        csv_write(out/'summary.csv',rows)
        print('joint 3D planes',config['case_index'],'finished',flush=True)
    (out/'assignments.json').write_text(json.dumps(details),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        inputs_note='Only Gaussians accepted on at least one of the 81 analytic rays; each counted once.',
        objective='Finite observed tangent-plane hypotheses, mean minimum robust point-to-plane energy, optional unsigned-axis angle term.',
        evidence='Known accepted-source K and a globally planar generating mechanism; not automatic layer count or physical truth in trained models. '
        'This probe adds 3D means and Gaussian axes, pools all rays, changes candidate/energy and counts each Gaussian once; '
        'comparison with independent per-ray fits is an information/hypothesis ablation, not an equal-information algorithm ranking. '
        'No construction labels or true plane angles enter fit; labels supply K and evaluation only. '
        'Fixed two scales are a sensitivity grid, not selected from evaluation. Single-origin K1 pairing success is tautological. '
        'An axis may flip without changing its geometric plane; this unsigned angle is intentionally different from oriented ray-normal constraints. '
        'A global Gaussian assignment assumes one construction origin across these rays, not a universal physical membership rule.'),indent=2),encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');args=p.parse_args()
    if args.self_check:self_check()
    else:
        if args.input is None or args.baseline is None or args.output is None:p.error('input, baseline, output required')
        run(args.input,args.baseline,args.output)
