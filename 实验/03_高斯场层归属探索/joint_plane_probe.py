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
                    if name not in record['labels']:name=name.removesuffix('_uniform')
                    baseline_scores.append(pair_metrics(actual,record['labels'][name])['f1'])
                score=pair_metrics(truth,labels)
                rows.append(dict(case_index=config['case_index'],kind=config['kind'],gap=config['gap'],
                    curvature=float(config.get('curvature',0.)),generating_seed=config['seed'],
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
        evidence='Known accepted-source K; generating geometry is planar when curvature=0 and curved otherwise; not automatic layer count or physical truth in trained models. '
        'This probe adds 3D means and Gaussian axes, pools all rays, changes candidate/energy and counts each Gaussian once; '
        'comparison with independent per-ray fits is an information/hypothesis ablation, not an equal-information algorithm ranking. '
        'No construction labels or true plane angles enter fit; labels supply K and evaluation only. '
        'Fixed two scales are a sensitivity grid, not selected from evaluation. Single-origin K1 pairing success is tautological. '
        'An axis may flip without changing its geometric plane; this unsigned angle is intentionally different from oriented ray-normal constraints. '
        'A global Gaussian assignment assumes one construction origin across these rays, not a universal physical membership rule. '
        'Near-optimal center tuple counts may include duplicate geometric planes and do not count distinct member partitions. '
        'Curvature controls, when present in input metadata, deliberately mismatch the global planar hypothesis.'),indent=2),encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def curve_candidates(points,normal,d):
    """Observed tangents anchor graph hypotheses; vertical candidates are flagged."""
    points=np.asarray(points,float);normal=np.asarray(normal,float)
    if points.ndim!=2 or points.shape[1]!=3 or normal.shape!=points.shape or not len(points):raise ValueError('Nonempty (N,3) required')
    lengths=np.linalg.norm(normal,axis=1)
    if not np.isfinite(points).all() or not np.isfinite(normal).all() or np.any(lengths<=1e-8):raise ValueError('Invalid member retained as a failed condition')
    normal=normal/lengths[:,None];valid=np.flatnonzero(np.abs(normal[:,2])>1e-3)
    p=points[valid];axis=normal[valid]
    b=-axis[:,0]/axis[:,2]-2*d*p[:,0];c=-axis[:,1]/axis[:,2]-2*d*p[:,1]
    a=p[:,2]-b*p[:,0]-c*p[:,1]-d*(p[:,0]**2+p[:,1]**2)
    return np.column_stack((a,b,c)),valid


def curve_energy(points,normal,coefficients,d,scale,normal_weight):
    points=np.asarray(points,float);normal=np.asarray(normal,float);coefficients=np.asarray(coefficients,float)
    if scale<=0 or not np.isfinite(scale):raise ValueError('Positive finite scale required')
    normal=normal/np.linalg.norm(normal,axis=1)[:,None]
    x=points[:,0,None];y=points[:,1,None];a,b,c=coefficients.T
    gx=b[None,:]+2*d*x;gy=c[None,:]+2*d*y;length=np.sqrt(1+gx*gx+gy*gy)
    residual=(points[:,2,None]-a[None,:]-b[None,:]*x-c[None,:]*y-d*(x*x+y*y))/length
    cost=2.5*np.log1p((residual/scale)**2/4.)
    dot=(-normal[:,0,None]*gx-normal[:,1,None]*gy+normal[:,2,None])/length
    cost+=normal_weight*np.minimum((np.arccos(np.clip(np.abs(dot),0.,1.))/np.deg2rad(30.))**2,1.)
    return cost


def curve_labels(cost):
    labels=np.argmin(cost,axis=1)
    if cost.shape[1]==2:labels[np.abs(cost[:,0]-cost[:,1])<=1e-10]=-1
    return labels


def fit_shared_curve(points,normal,K,scale,normal_weight):
    if K not in (1,2):raise ValueError('Explicit K1/2 oracle required')
    choices=[]
    for d in (0.,20.,50.):
        coeff,valid=curve_candidates(points,normal,d)
        if not len(valid):continue
        cost=curve_energy(points,normal,coeff,d,scale,normal_weight)
        tuples=list(itertools.combinations_with_replacement(range(len(valid)),K))
        values=np.array([cost[:,list(ix)].min(axis=1).mean() for ix in tuples]);at=int(np.argmin(values));ix=list(tuples[at])
        choices.append(dict(d=d,objective=float(values[at]),center_indices=valid[ix],coefficients=coeff[ix],
                            invalid_candidates=len(points)-len(valid),labels=curve_labels(cost[:,ix])))
    if not choices:return dict(status='no_graph_candidates',labels=np.full(len(points),-1,int),choices=[])
    best=min(r['objective'] for r in choices)
    selected=next(r for r in choices if r['objective']<=best+1e-10)
    return dict(status='ok',**selected,choices=[dict(d=r['d'],objective=r['objective']) for r in choices])


def curve_self_check():
    p=np.array([[.003,.002,1.],[-.004,.005,1.01],[.006,-.002,1.02],[.001,.007,1.03]])
    n=np.array([[.1,.2,1.],[-.2,.3,1.],[.4,-.1,1.],[.2,.1,1.]])
    for scale,lam in itertools.product((.002,.005),(0.,1.)):
        coeff,valid=curve_candidates(p,n,0.)
        np.testing.assert_array_equal(valid,np.arange(4))
        got=curve_energy(p,n,coeff,0.,scale,lam)
        np.testing.assert_allclose(got,energy(p,n,scale,lam),atol=1e-11,rtol=1e-10)
        flipped,_=curve_candidates(p,-n,0.)
        np.testing.assert_allclose(flipped,coeff,atol=1e-14)
    xy=np.array([[-.01,-.005],[.005,.01],[.01,-.01]])
    p=np.vstack([np.column_stack((xy,a+20*(xy**2).sum(axis=1))) for a in (1.,1.02)])
    n=np.tile(np.column_stack((-40*xy,np.ones(3))),(2,1));learn=np.array([0,1,3,4]);test=np.array([2,5])
    result=fit_shared_curve(p[learn],n[learn],2,.002,1.)
    assert result['status']=='ok' and result['d']==20 and result['objective']<1e-10
    assert not np.intersect1d(learn[result['center_indices']],test).size
    labels=curve_labels(curve_energy(p[test],n[test],result['coefficients'],result['d'],.002,1.))
    assert labels[0]>=0 and labels[1]>=0 and labels[0]!=labels[1]
    failed=fit_shared_curve(p,np.tile([1.,0.,0.],(6,1)),2,.002,1.)
    assert failed['status']=='no_graph_candidates' and (failed['labels']==-1).all()
    print('shared-curve d0/tangent/gauge/heldout/vertical self-check passed',flush=True)


def run_shared_curve(root,out,case=6):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    previous=root/'gaussian_member_holdout';oldmeta=json.loads((previous/'metadata.json').read_text(encoding='utf-8'))
    assert oldmeta['status']=='complete'
    cached={}
    for line in (previous/'models.jsonl').read_text(encoding='utf-8').splitlines():
        m=json.loads(line)
        if m['case_index']==case and m['protocol']=='random_half':cached[(m['seed_offset'],m['curvature'],m['scale'],m['normal_weight'],m['repeat'])]=m
    with (previous/'replicates.csv').open(encoding='utf-8-sig',newline='') as f:
        baseline={(int(r['seed_offset']),int(r['curvature']),float(r['scale']),float(r['normal_weight']),int(r['repeat'])):r
                  for r in csv.DictReader(f) if int(r['case_index'])==case and r['protocol']=='random_half'}
    assert len(cached)==len(baseline)==120
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];details=[]
    for seed in (1000,2000):
        data={};common=None
        for curvature in (0,20,50):
            folder=root/f'independent_c{curvature}_s{seed}';meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            assert meta['status']=='complete' and meta['backend']=='analytic_ray_peak_gaussian_CPU'
            item=meta['cases'][case];cfg=item['config'];assert cfg['case_index']==case and cfg['seed']==20261022+seed+case
            if curvature==0:reference=cfg
            for name in ('kind','gap','theta_deg','center_noise','normal_noise_deg'):assert cfg[name]==reference[name]
            assert float(cfg.get('curvature',0.))==curvature
            path=folder/item['file'];assert sha(path)==item['npz_sha256']
            assert sha(path)==next(item['sha256'] for item in oldmeta['inputs'] if item['path']==str(path))
            inputs.append(dict(path=str(path),sha256=sha(path)))
            with np.load(path,allow_pickle=False) as z:
                data[curvature]={name:z[name].copy() for name in ('means','generated_normal','source_origin')}
                ids=np.unique(z['gaussian_id']);common=ids if common is None else np.intersect1d(common,ids)
        truth=data[0]['source_origin'][common];K=len(np.unique(truth));assert K==2
        for curvature,scale,lam,repeat in itertools.product((0,20,50),(.002,.005),(0.,1.),range(5)):
            payload=data[curvature];p=payload['means'].astype(float);n=payload['generated_normal'].astype(float)
            np.testing.assert_array_equal(payload['source_origin'][common],truth)
            model=cached[(seed,curvature,scale,lam,repeat)];learn=np.array(model['fit_ids']);test=np.array(model['heldout_ids'])
            order=np.random.default_rng(700000+seed*100+case*7+repeat).permutation(len(common))
            np.testing.assert_array_equal(learn,common[np.sort(order[:len(common)//2])]);np.testing.assert_array_equal(test,common[np.sort(order[len(common)//2:])])
            assert not np.intersect1d(learn,test).size
            centers=np.array(model['center_ids']);assert np.isin(centers,learn).all()
            plane_coeff,valid=curve_candidates(p[centers],n[centers],0.);assert len(valid)==K
            base_cost=curve_energy(p[test],n[test],plane_coeff,0.,scale,lam);base_labels=curve_labels(base_cost)
            base_score=pair_metrics(payload['source_origin'][test],base_labels)
            np.testing.assert_allclose(base_score['f1'],float(baseline[(seed,curvature,scale,lam,repeat)]['heldout_pair_f1']),atol=1e-12,rtol=0)
            try:result=fit_shared_curve(p[learn],n[learn],K,scale,lam)
            except ValueError:result=dict(status='invalid_input',labels=np.full(len(learn),-1,int),choices=[])
            pred=np.full(len(test),-1,int);chosen_ids=[]
            if result['status']=='ok':
                chosen_ids=learn[result['center_indices']].tolist();assert np.isin(chosen_ids,learn).all() and not np.isin(chosen_ids,test).any()
                pred=curve_labels(curve_energy(p[test],n[test],result['coefficients'],result['d'],scale,lam))
            score=pair_metrics(payload['source_origin'][test],pred)
            row=dict(seed_offset=seed,case_index=case,kind=reference['kind'],center_noise=reference['center_noise'],normal_noise_deg=reference['normal_noise_deg'],curvature=curvature,scale=scale,normal_weight=lam,repeat=repeat,
                oracle_common_K=K,status=result['status'],selected_d=result.get('d'),fit_objective=result.get('objective'),
                invalid_graph_candidates=result.get('invalid_candidates',len(learn)),fit_members=len(learn),heldout_members=len(test),
                fit_origins=len(np.unique(payload['source_origin'][learn])),heldout_origins=len(np.unique(payload['source_origin'][test])),
                baseline_pair_f1=base_score['f1'],curve_pair_f1=score['f1'],curve_minus_plane=None if score['f1'] is None else score['f1']-base_score['f1'],
                false_merge_pairs=score['fp'],false_split_pairs=score['fn'],unknown_members=int((pred<0).sum()),groups=len(set(pred.tolist())-{-1}))
            for choice in result['choices']:row['fit_objective_d'+str(int(choice['d']))]=choice['objective']
            for d in (0,20,50):row.setdefault('fit_objective_d'+str(d),None)
            rows.append(row);details.append(dict(**{name:row[name] for name in ('seed_offset','curvature','scale','normal_weight','repeat','status','selected_d')},
                case_index=case,fit_ids=learn.tolist(),heldout_ids=test.tolist(),center_ids=chosen_ids,coefficients=result['coefficients'].tolist() if result['status']=='ok' else None,
                heldout_labels=pred.tolist(),baseline_labels=base_labels.tolist()))
        csv_write(out/'replicates.csv',rows);print('shared curve case',case,'seed',seed,'finished',flush=True)
    assert len(rows)==120
    (out/'models.json').write_text(json.dumps(details),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,case_index=case,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        dependency_sha256={'analyze.py':sha(Path(__file__).parent/'analyze.py')},
        cached_input_sha256={name:sha(previous/name) for name in ('metadata.json','models.jsonl','replicates.csv')},
        evidence='One explicitly selected existing case, two seeds/three curvature conditions/four energy settings/five predetermined random halves. '
        'Finite shared curvature grid [0,20,50] intentionally includes construction truth: extra controlled prior, not general recovery or learned continuous curvature. '
        'Each candidate graph anchored by a fit member position/tangent; only fit members choose d and center tuple. K is common-set source-count oracle. '
        'Normalized graph residual is not exact closest-surface distance for curved graphs. Fixed scales, unsigned axis term. '
        'Candidates with abs(unit n_z)<=1e-3 are invalid, but members remain scored; no-candidate/invalid conditions return unknown, never silently drop. '
        'd0 prediction checked against cached same-heldout plane baseline; curvature objective ties within 1e-10 choose first d. '
        'One common parameter set makes same-member relation fixed by construction; consistency alone is not accuracy. '
        'Related member splits, not independent scenes; no original 3DGS training, automatic K or host GPU evidence.'),indent=2),encoding='utf-8')


def nominal_midpoint_labels(points,cfg):
    assert len(cfg['theta_deg'])==2 and cfg['theta_deg'][0]==cfg['theta_deg'][1] and cfg['gap']>0
    midpoint=1.+cfg['gap']/2-np.tan(np.deg2rad(cfg['theta_deg'][0]))*points[:,0]+float(cfg.get('curvature',0.))*(points[:,:2]**2).sum(axis=1)
    delta=points[:,2]-midpoint;labels=(delta>0).astype(int);labels[np.abs(delta)<=1e-12]=-1
    return labels


def geometry_oracle_self_check():
    cfg=dict(theta_deg=[0.,0.],gap=.01,curvature=20.)
    p=np.array([[.01,.02,1.],[.01,.02,1.01],[.01,.02,1.005]])
    p[:,2]+=20*(p[:,:2]**2).sum(axis=1)
    n=np.tile([.2,.3,1.],(3,1));coeff=np.array([[1.,0.,0.],[1.01,0.,0.]])
    for scale,lam in itertools.product((.002,.005),(0.,1.)):
        pred=curve_labels(curve_energy(p,n,coeff,20.,scale,lam))
        np.testing.assert_array_equal(pred,nominal_midpoint_labels(p,cfg))
        assert pred.tolist()==[0,1,-1]
    print('known-geometry energy/midpoint/unknown self-check passed',flush=True)


def run_geometry_oracle(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];seen=set();details=[]
    for case in range(6,12):
        folder=root/'shared_curve_case6' if case==6 else root/f'shared_curve_noise/case{case}'
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));assert meta['status']=='complete'
        for item in meta['inputs']:assert sha(Path(item['path']))==item['sha256']
        inputs.append(dict(path=str(folder/'models.json'),sha256=sha(folder/'models.json'),metadata_sha256=sha(folder/'metadata.json')))
        models=json.loads((folder/'models.json').read_text(encoding='utf-8'))
        with (folder/'replicates.csv').open(encoding='utf-8-sig',newline='') as f:
            prior={(int(r['seed_offset']),int(r['curvature']),float(r['scale']),float(r['normal_weight']),int(r['repeat'])):r for r in csv.DictReader(f)}
        for m in models:
            seed=m['seed_offset'];curvature=m['curvature'];scale=m['scale'];lam=m['normal_weight'];repeat=m['repeat']
            key=(case,seed,curvature,scale,lam,repeat);assert key not in seen;seen.add(key)
            data_folder=root/f'independent_c{curvature}_s{seed}';data_meta=json.loads((data_folder/'metadata.json').read_text(encoding='utf-8'))
            item=data_meta['cases'][case];cfg=item['config'];assert cfg['case_index']==case and cfg['seed']==20261022+case+seed
            path=data_folder/item['file'];assert sha(path)==item['npz_sha256']
            assert any(str(path)==entry['path'] and sha(path)==entry['sha256'] for entry in meta['inputs'])
            with np.load(path,allow_pickle=False) as z:
                ids=np.array(m['heldout_ids'],int);p=z['means'][ids].astype(float);n=z['generated_normal'][ids].astype(float);truth=z['source_origin'][ids]
                fit_origins=len(np.unique(z['source_origin'][m['fit_ids']]))
            assert not set(ids)&set(m['fit_ids']) and len(ids)==len(set(ids))
            coeff=np.array([[1.+origin*cfg['gap'],-np.tan(np.deg2rad(cfg['theta_deg'][origin])),0.] for origin in range(2)])
            pred=curve_labels(curve_energy(p,n,coeff,curvature,scale,lam))
            independent=nominal_midpoint_labels(p,cfg);np.testing.assert_array_equal(pred,independent)
            score=pair_metrics(truth,pred);curve_pred=np.array(m['heldout_labels'],int);curve_score=pair_metrics(truth,curve_pred)
            np.testing.assert_allclose(curve_score['f1'],float(prior[(seed,curvature,scale,lam,repeat)]['curve_pair_f1']),atol=1e-12,rtol=0)
            upper=np.triu(np.ones((len(ids),len(ids)),bool),1)
            known=(pred[:,None]>=0)&(pred[None,:]>=0)&(curve_pred[:,None]>=0)&(curve_pred[None,:]>=0)&upper
            relation_diff=(pred[:,None]==pred[None,:])!=(curve_pred[:,None]==curve_pred[None,:])
            row=dict(case_index=case,seed_offset=seed,curvature=curvature,scale=scale,normal_weight=lam,repeat=repeat,
                center_noise=cfg['center_noise'],normal_noise_deg=cfg['normal_noise_deg'],heldout_members=len(ids),
                heldout_origins=len(np.unique(truth)),fit_origins=fit_origins,origin0_members=int((truth==0).sum()),origin1_members=int((truth==1).sum()),
                nominal_known_members=int((pred>=0).sum()),nominal_unknown_members=int((pred<0).sum()),
                origin0_crossed_midpoint=int(((truth==0)&(pred==1)).sum()),origin1_crossed_midpoint=int(((truth==1)&(pred==0)).sum()),
                nominal_origin_disagreements=int(((pred>=0)&(pred!=truth)).sum()),
                nominal_pair_f1=score['f1'],nominal_false_merge_pairs=score['fp'],nominal_false_split_pairs=score['fn'],
                curve_pair_f1=curve_score['f1'],curve_false_merge_pairs=curve_score['fp'],curve_false_split_pairs=curve_score['fn'],curve_unknown_members=int((curve_pred<0).sum()),
                curve_minus_nominal_f1=curve_score['f1']-score['f1'],both_known_pairs=int(known.sum()),
                curve_nominal_relation_disagreements=int((known&relation_diff).sum()))
            rows.append(row);details.append(dict(**{k:row[k] for k in ('case_index','seed_offset','curvature','scale','normal_weight','repeat')},
                heldout_ids=ids.tolist(),nominal_labels=pred.tolist(),source_origin=truth.tolist(),curve_labels=curve_pred.tolist()))
        csv_write(out/'replicates.csv',rows);print('known geometry case',case,'finished',flush=True)
    assert len(rows)==720
    (out/'labels.json').write_text(json.dumps(details),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),dependency_sha256={'analyze.py':sha(Path(__file__).parent/'analyze.py')},
        evidence='True construction curvature/gap/angles supply nominal graphs: explicit geometric oracle, no fitting or selection. '
        'Same saved heldout IDs and same four energies; energy labels independently checked against parallel height midpoint. '
        'Normalized graph residual is not closest Euclidean curved-surface distance. Same graph gradients make axis costs equal between the two parallel sources. '
        'Construction-origin labels and closest-nominal assignment can differ after center perturbation; neither is trained-scene physical support truth. '
        'No pair-F1 upper bound, irreducibility, automatic method or original 3DGS training claimed. '
        'Repeated splits and energy configurations reuse members; counts must not be summed as unique Gaussians.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');p.add_argument('--shared-curve-case6','--shared-curve',dest='shared_curve_case6',action='store_true');p.add_argument('--root',type=Path)
    p.add_argument('--curve-case',type=int,choices=range(6,12),default=6);p.add_argument('--geometry-oracle',action='store_true');args=p.parse_args()
    if args.geometry_oracle and args.shared_curve_case6:p.error('choose a single probe mode')
    if args.self_check:geometry_oracle_self_check() if args.geometry_oracle else curve_self_check() if args.shared_curve_case6 else self_check()
    elif args.geometry_oracle:
        if args.root is None or args.output is None:p.error('root and output required')
        run_geometry_oracle(args.root,args.output)
    elif args.shared_curve_case6:
        if args.root is None or args.output is None:p.error('root and output required')
        run_shared_curve(args.root,args.output,args.curve_case)
    else:
        if args.input is None or args.baseline is None or args.output is None:p.error('input, baseline, output required')
        run(args.input,args.baseline,args.output)
