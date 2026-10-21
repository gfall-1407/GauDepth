"""Curvature comparison on exactly the same Gaussian IDs, not changing acceptance."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze import pair_metrics
from joint_plane_probe import fit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[]
    for seed in (1000,2000):
        folders={k:root/f'independent_c{k}_s{seed}' for k in (0,20,50)}
        metas={k:json.loads((folder/'metadata.json').read_text(encoding='utf-8')) for k,folder in folders.items()}
        for meta in metas.values():assert meta['status']=='complete' and len(meta['cases'])==30
        for case in range(30):
            payload={};common=None;accepted={}
            for curvature,folder in folders.items():
                record=metas[curvature]['cases'][case];assert record['config']['case_index']==case
                assert record['config']['seed']==20261022+case+seed
                assert float(record['config'].get('curvature',0.))==curvature
                reference=metas[0]['cases'][case]['config']
                for key in ('kind','gap','theta_deg','center_noise','normal_noise_deg'):
                    assert record['config'][key]==reference[key], (case,key)
                path=folder/record['file'];assert sha(path)==record['npz_sha256']
                inputs.append(dict(path=str(path),sha256=sha(path)))
                with np.load(path,allow_pickle=False) as z:
                    payload[curvature]={key:z[key].copy() for key in ('means','generated_normal','source_origin')}
                    accepted[curvature]=np.unique(z['gaussian_id'])
                    common=accepted[curvature] if common is None else np.intersect1d(common,accepted[curvature])
            assert len(common)>=2
            means0=payload[0]['means'][common].astype(float);truth=payload[0]['source_origin'][common]
            K=len(np.unique(truth))
            for curvature in (0,20,50):
                data=payload[curvature];means=data['means'][common].astype(float)
                np.testing.assert_array_equal(data['source_origin'][common],truth)
                np.testing.assert_array_equal(means[:,:2],means0[:,:2])
                np.testing.assert_allclose(means[:,2]-means0[:,2],curvature*(means0[:,0]**2+means0[:,1]**2),atol=2e-7,rtol=0)
                for scale,lam in itertools.product((.002,.005),(0.,1.)):
                    result=fit(means,data['generated_normal'][common],K,scale,lam)
                    score=pair_metrics(truth,result['labels']);cfg=metas[curvature]['cases'][case]['config']
                    assert score['f1'] is not None, 'This control requires a scoreable common subset'
                    rows.append(dict(seed_offset=seed,case_index=case,kind=cfg['kind'],gap=cfg['gap'],
                        center_noise=cfg['center_noise'],normal_noise_deg=cfg['normal_noise_deg'],
                        curvature=curvature,scale=scale,normal_weight=lam,
                        common_gaussians=len(common),accepted_union_gaussians=len(accepted[curvature]),
                        common_fraction_of_union=len(common)/len(accepted[curvature]),
                        origin0_common_count=int((truth==0).sum()),origin1_common_count=int((truth==1).sum()),
                        oracle_common_K=K,groups=len(set(result['labels'].tolist())-{-1}),
                        nominal_origins=len(cfg['theta_deg']),lost_origin_in_common=K<len(cfg['theta_deg']),
                        unknown_gaussians=int((result['labels']<0).sum()),pair_f1=score['f1'],
                        false_merge_pairs=score['fp'],false_split_pairs=score['fn'],objective=result['objective']))
            write_csv(out/'summary.csv',rows)
            print('common-support curvature',seed,case,'finished',flush=True)
    reference={(r['seed_offset'],r['case_index'],r['scale'],r['normal_weight']):r for r in rows if r['curvature']==0}
    paired=[]
    for row in rows:
        if row['curvature']==0:continue
        base=reference[(row['seed_offset'],row['case_index'],row['scale'],row['normal_weight'])]
        paired.append(dict(**row,planar_reference_pair_f1=base['pair_f1'],
                           curvature_minus_planar=row['pair_f1']-base['pair_f1']))
    write_csv(out/'paired.csv',paired)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:sha(Path(__file__).parent/name) for name in ('common_support_probe.py','joint_plane_probe.py','analyze.py')},
        evidence='Fixed Gaussian ID intersection across kappa=0/20/50, separately per case and new seed offset. '
        'Same labels, same candidate indices, same counts; avoids changing accepted-ID population as the explanation of curvature effects. '
        'Centers x/y identical, center z difference equals kappa*(x*x+y*y) to float32 tolerance; intrinsic axes follow new surface plus matched scalar angular perturbations. '
        'No per-ray score or per-ray acceptance invariance claimed. Intersection selection may discard difficult members; its fraction and per-origin counts are reported. '
        'Still a known-K, globally planar hypothesis evaluated under curved construction; not automatic layers, curved-surface impossibility, or trained-scene evidence. '
        'Only two independent random seed repeats; condition combinations are not independent scenes.'),indent=2),encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def predict(points,normal,centers,axes,scale,normal_weight):
    """Assign new members to already fixed observed tangent planes."""
    normal=np.asarray(normal,float);axes=np.asarray(axes,float)
    normal=normal/np.linalg.norm(normal,axis=1)[:,None]
    axes=axes/np.linalg.norm(axes,axis=1)[:,None]
    residual=np.einsum('ihj,hj->ih',points[:,None,:]-centers[None,:,:],axes)
    costs=2.5*np.log1p((residual/scale)**2/4.)
    costs+=normal_weight*np.minimum((np.arccos(np.clip(np.abs(normal@axes.T),0.,1.))/np.deg2rad(30.))**2,1.)
    labels=np.argmin(costs,axis=1)
    if len(centers)==2:labels[np.abs(costs[:,0]-costs[:,1])<=1e-10]=-1
    return labels,costs


def holdout_self_check():
    from joint_plane_probe import energy
    p=np.array([[0.,0.,1.],[.01,0.,1.],[0.,.01,1.02],[.01,.01,1.02]])
    n=np.tile([0.,0.,1.],(4,1));ix=[0,2]
    labels,e=predict(p,n,p[ix],n[ix],.002,1.)
    np.testing.assert_allclose(e,energy(p,n,.002,1.)[:,ix],atol=1e-12,rtol=0)
    assert labels.tolist()==[0,0,1,1]
    fit_ids=np.array([0,2]);test_ids=np.array([1,3])
    fitted=fit(p[fit_ids],n[fit_ids],2,.002,1.)
    assert set(fit_ids[fitted['centers'],])<=set(fit_ids)
    pred,_=predict(p[test_ids],n[test_ids],p[fit_ids[list(fitted['centers'])]],n[fit_ids[list(fitted['centers'])]],.002,1.)
    assert pred.tolist()==[0,1]
    print('held-out plane prediction self-check passed',flush=True)


def run_holdout(root,out,given_k=None,single_only=False):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[]
    with (out/'models.jsonl').open('w',encoding='utf-8') as models:
        for seed in (1000,2000):
            folders={k:root/f'independent_c{k}_s{seed}' for k in (0,20,50)}
            metas={k:json.loads((folder/'metadata.json').read_text(encoding='utf-8')) for k,folder in folders.items()}
            for meta in metas.values():assert meta['status']=='complete' and len(meta['cases'])==30
            for case in range(30):
                if single_only and metas[0]['cases'][case]['config']['kind']!='single':continue
                data={};common=None
                for curvature,folder in folders.items():
                    item=metas[curvature]['cases'][case];cfg=item['config']
                    assert cfg['case_index']==case and cfg['seed']==20261022+case+seed
                    reference=metas[0]['cases'][case]['config']
                    for key in ('kind','gap','theta_deg','center_noise','normal_noise_deg'):assert cfg[key]==reference[key]
                    path=folder/item['file'];assert sha(path)==item['npz_sha256']
                    inputs.append(dict(path=str(path),sha256=sha(path)))
                    with np.load(path,allow_pickle=False) as z:
                        data[curvature]={k:z[k].copy() for k in ('means','generated_normal','source_origin')}
                        ids=np.unique(z['gaussian_id']);common=ids if common is None else np.intersect1d(common,ids)
                truth=data[0]['source_origin'][common];K=len(np.unique(truth));size=len(common)
                model_K=K if given_k is None else given_k
                splits=[]
                for repeat in range(5):
                    split_seed=700000+seed*100+case*7+repeat
                    order=np.random.default_rng(split_seed).permutation(size)
                    splits.append(('random_half',repeat,split_seed,np.sort(order[:size//2]),np.sort(order[size//2:])))
                x=data[0]['means'][common,0]
                splits.append(('spatial_left_right',0,None,np.flatnonzero(x<0),np.flatnonzero(x>0)))
                for curvature in (0,20,50):
                    p=data[curvature]['means'][common].astype(float);n=data[curvature]['generated_normal'][common].astype(float)
                    np.testing.assert_array_equal(data[curvature]['source_origin'][common],truth)
                    for scale,lam in itertools.product((.002,.005),(0.,1.)):
                        full=fit(p,n,model_K,scale,lam);full_ix=list(full['centers'])
                        for protocol,repeat,split_seed,learn,test in splits:
                            assert len(learn)>=model_K and len(test)>=2 and not np.intersect1d(learn,test).size
                            fitted=fit(p[learn],n[learn],model_K,scale,lam);centers=learn[list(fitted['centers'])]
                            assert np.isin(centers,learn).all() and not np.isin(centers,test).any()
                            pred,test_cost=predict(p[test],n[test],p[centers],n[centers],scale,lam)
                            full_pred,ref_cost=predict(p[test],n[test],p[full_ix],n[full_ix],scale,lam)
                            score=pair_metrics(truth[test],pred);fit_score=pair_metrics(truth[learn],fitted['labels'])
                            ref_score=pair_metrics(truth[test],full_pred)
                            assert score['f1'] is not None and fit_score['f1'] is not None and ref_score['f1'] is not None
                            row=dict(seed_offset=seed,case_index=case,kind=reference['kind'],gap=reference['gap'],
                                center_noise=reference['center_noise'],normal_noise_deg=reference['normal_noise_deg'],
                                curvature=curvature,scale=scale,normal_weight=lam,protocol=protocol,repeat=repeat,
                                split_seed=split_seed,oracle_common_K=K,given_K=model_K,common_members=size,fit_members=len(learn),
                                heldout_members=len(test),excluded_members=size-len(learn)-len(test),
                                fit_origins=len(np.unique(truth[learn])),heldout_origins=len(np.unique(truth[test])),
                                fit_pair_f1=fit_score['f1'],heldout_pair_f1=score['f1'],
                                full_data_reference_on_same_heldout_pairs=ref_score['f1'],
                                heldout_minus_full_reference=score['f1']-ref_score['f1'],
                                heldout_false_merge_pairs=score['fp'],heldout_false_split_pairs=score['fn'],
                                heldout_unknown=int((pred<0).sum()),fit_unknown=int((fitted['labels']<0).sum()),
                                heldout_mean_min_energy=float(test_cost.min(axis=1).mean()),
                                full_data_reference_heldout_energy=float(ref_cost.min(axis=1).mean()),
                                fit_objective=fitted['objective'])
                            rows.append(row)
                            models.write(json.dumps(dict(**{k:row[k] for k in ('seed_offset','case_index','curvature','scale','normal_weight','protocol','repeat')},
                                given_K=model_K,fit_ids=common[learn].tolist(),heldout_ids=common[test].tolist(),
                                center_ids=common[centers].tolist()),ensure_ascii=False)+'\n')
                write_csv(out/'replicates.csv',rows);models.flush()
                print('held-out common Gaussian fit',seed,case,'finished',flush=True)
    summary=[]
    for values in sorted(set((r['seed_offset'],r['case_index'],r['curvature'],r['scale'],r['normal_weight'],r['protocol']) for r in rows)):
        keys=('seed_offset','case_index','curvature','scale','normal_weight','protocol');config=dict(zip(keys,values))
        part=[r for r in rows if all(r[k]==v for k,v in config.items())]
        summary.append(dict(**config,**{k:part[0][k] for k in ('kind','gap','center_noise','normal_noise_deg','oracle_common_K')},
            repeats=len(part),fit_missing_origin_splits=sum(r['fit_origins']<r['oracle_common_K'] for r in part),
            heldout_missing_origin_splits=sum(r['heldout_origins']<r['oracle_common_K'] for r in part),
            given_K=part[0]['given_K'],**{'mean_'+k:float(np.mean([r[k] for r in part])) for k in ('fit_pair_f1','heldout_pair_f1',
                'full_data_reference_on_same_heldout_pairs','heldout_minus_full_reference','heldout_unknown','fit_members','heldout_members','excluded_members')},
            mean_heldout_min_energy=float(np.mean([r['heldout_mean_min_energy'] for r in part])),
            std_heldout_pair_f1=float(np.std([r['heldout_pair_f1'] for r in part])),
            min_heldout_pair_f1=min(r['heldout_pair_f1'] for r in part)))
    write_csv(out/'summary.csv',summary)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:sha(Path(__file__).parent/name) for name in ('common_support_probe.py','joint_plane_probe.py','analyze.py')},
        random_splits=5,split_seed_rule='700000 + seed_offset*100 + case_index*7 + repeat',given_K=given_k,single_source_only=single_only,
        evidence='Common Gaussian IDs across curvature. Fit/test split ignores source labels and is identical across kappa and all energy configurations. '
        'Only fit-member centers/axes form candidate planes and only fit members optimize the objective; held-out features only assign to fixed selected planes. '
        'K is normally the full common-set source count oracle; explicit given_K overrides it for a conditional count stress. No resampling. '
        'Random half tests interpolation among correlated members. Spatial x<0 fitting vs x>0 evaluation omits x=0 and tests one-sided extrapolation, not exact 50%. '
        'Full-data reference is scored on the same held-out pairs but has seen them; it is descriptive resubstitution, not independent validation. '
        'Two generating seed repeats and five dependent member splits do not constitute independent scenes. '
        'K1 pairing is tautological; main result must isolate true K2 and retain coverage loss/unknowns. '
        'Single-source explicit-K runs test whether validation loss can favor extra components; same held-out features used for selection are not an independent final test. '
        'No original 3DGS training or physical count claim.'),indent=2),encoding='utf-8')


def local_self_check():
    holdout_self_check()
    # Fixed planar two-source example: local fitting sees neither held-out ID.
    p=np.array([[0.,0.,1.],[.001,0.,1.02],[0.,.001,1.],[.001,.001,1.02]])
    n=np.tile([0.,0.,1.],(4,1));learn=np.array([0,1]);test=np.array([2,3])
    result=fit(p[learn],n[learn],2,.002,1.)
    centers=learn[list(result['centers'])]
    got,_=predict(p[test],n[test],p[centers],n[centers],.002,1.)
    assert got.tolist()==[0,1] and not np.intersect1d(centers,test).size
    # Empty local evaluation is retained rather than passed off as a perfect score.
    assert pair_metrics([],[])['f1'] is None
    print('local member prediction self-check passed',flush=True)


def run_local(root,out):
    """Fixed x-y neighborhoods; reuse previous train-only global models, no refits."""
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    previous=root/'gaussian_member_holdout'
    oldmeta=json.loads((previous/'metadata.json').read_text(encoding='utf-8'))
    assert oldmeta['status']=='complete'
    # Fit/energy/scoring definitions must match the original cached global run.
    for name in ('joint_plane_probe.py','analyze.py'):
        assert sha(Path(__file__).parent/name)==oldmeta['source_sha256'][name]
    input_hash={str(Path(item['path'])):item['sha256'] for item in oldmeta['inputs']}
    keys=('seed_offset','case_index','curvature','scale','normal_weight','repeat')
    models={};oldrows={}
    for line in (previous/'models.jsonl').read_text(encoding='utf-8').splitlines():
        m=json.loads(line)
        if m['protocol']=='random_half':
            key=tuple(m[k] for k in keys);assert key not in models;models[key]=m
    with (previous/'replicates.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['protocol']=='random_half':
                key=tuple(float(row[k]) if k in ('scale','normal_weight') else int(row[k]) for k in keys)
                oldrows[key]=row
    assert len(models)==len(oldrows)==3600
    out.mkdir(parents=True,exist_ok=True);inputs=[];summaries=[];total=0
    centers_xy=list(itertools.product((-.005,0.,.005),repeat=2))
    with (out/'replicates.csv').open('w',encoding='utf-8-sig',newline='') as detail, (out/'models.jsonl').open('w',encoding='utf-8') as saved:
        writer=None
        for seed,case in itertools.product((1000,2000),range(30)):
            data={};common=None;case_rows=[]
            for curvature in (0,20,50):
                folder=root/f'independent_c{curvature}_s{seed}'
                meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
                assert meta['status']=='complete' and meta['backend']=='analytic_ray_peak_gaussian_CPU'
                item=meta['cases'][case];cfg=item['config']
                assert cfg['case_index']==case and cfg['seed']==20261022+seed+case
                assert float(cfg.get('curvature',0.))==curvature
                if curvature==0:reference=cfg
                for name in ('kind','gap','theta_deg','center_noise','normal_noise_deg'):assert cfg[name]==reference[name]
                path=folder/item['file'];digest=sha(path)
                assert digest==item['npz_sha256']==input_hash[str(path)]
                inputs.append(dict(path=str(path),sha256=digest))
                with np.load(path,allow_pickle=False) as z:
                    data[curvature]={name:z[name].copy() for name in ('means','generated_normal','source_origin')}
                    ids=np.unique(z['gaussian_id']);common=ids if common is None else np.intersect1d(common,ids)
            truth=data[0]['source_origin'][common];K=len(np.unique(truth));size=len(common)
            for curvature in (0,20,50):
                payload=data[curvature];p=payload['means'].astype(float);n=payload['generated_normal'].astype(float)
                np.testing.assert_array_equal(payload['source_origin'][common],truth)
                np.testing.assert_array_equal(p[common,:2],data[0]['means'][common,:2])
                for scale,lam,repeat in itertools.product((.002,.005),(0.,1.),range(5)):
                    key=(seed,case,curvature,scale,lam,repeat);model=models[key]
                    split_seed=700000+seed*100+case*7+repeat
                    order=np.random.default_rng(split_seed).permutation(size)
                    learn=common[np.sort(order[:size//2])];test=common[np.sort(order[size//2:])]
                    np.testing.assert_array_equal(learn,model['fit_ids']);np.testing.assert_array_equal(test,model['heldout_ids'])
                    global_centers=np.asarray(model['center_ids'],int)
                    assert len(global_centers)==K and np.isin(global_centers,learn).all() and not np.isin(global_centers,test).any()
                    global_pred,_=predict(p[test],n[test],p[global_centers],n[global_centers],scale,lam)
                    np.testing.assert_allclose(pair_metrics(payload['source_origin'][test],global_pred)['f1'],float(oldrows[key]['heldout_pair_f1']),atol=1e-12,rtol=0)
                    for radius,(cx,cy) in itertools.product((.005,.01),centers_xy):
                        inside=np.sum((p[:,:2]-np.array([cx,cy]))**2,axis=1)<=radius**2
                        local_learn=learn[inside[learn]];take=inside[test];local_test=test[take]
                        actual=payload['source_origin'][local_test];baseline=global_pred[take]
                        true_pairs=sum(int((actual==origin).sum())*(int((actual==origin).sum())-1)//2 for origin in np.unique(actual))
                        status='empty_heldout' if len(local_test)==0 else 'no_member_pairs' if len(local_test)<2 else 'no_true_pairs' if true_pairs==0 else 'ok'
                        base_score=pair_metrics(actual,baseline);fitted=None;local_centers=np.array([],int)
                        local_pred=None;local_score=None
                        if len(local_learn)>=K:
                            fitted=fit(p[local_learn],n[local_learn],K,scale,lam)
                            local_centers=local_learn[list(fitted['centers'])]
                            assert np.isin(local_centers,local_learn).all() and not np.isin(local_centers,test).any()
                            local_pred,_=predict(p[local_test],n[local_test],p[local_centers],n[local_centers],scale,lam)
                            local_score=pair_metrics(actual,local_pred)
                        row=dict(seed_offset=seed,case_index=case,kind=reference['kind'],curvature=curvature,scale=scale,normal_weight=lam,
                            repeat=repeat,split_seed=split_seed,center_x=cx,center_y=cy,radius=radius,oracle_common_K=K,
                            local_fit_status='ok' if fitted is not None else 'insufficient_fit_members',score_status=status,
                            global_fit_members=len(learn),local_fit_members=len(local_learn),heldout_members=len(local_test),true_same_source_pairs=true_pairs,
                            fit_origins=len(np.unique(payload['source_origin'][local_learn])),heldout_origins=len(np.unique(actual)),
                            global_groups=len(set(baseline.tolist())-{-1}),global_unknown=int((baseline<0).sum()),
                            local_groups=None if local_pred is None else len(set(local_pred.tolist())-{-1}),
                            local_unknown=None if local_pred is None else int((local_pred<0).sum()),
                            global_pair_f1=base_score['f1'],local_pair_f1=None if local_score is None else local_score['f1'])
                        for prefix,score in (('global',base_score),('local',local_score)):
                            for name in ('tp','fp','fn'):row[prefix+'_'+name]=None if score is None else score[name]
                        row['local_minus_global']=None if local_score is None or local_score['f1'] is None or base_score['f1'] is None else local_score['f1']-base_score['f1']
                        if writer is None:writer=csv.DictWriter(detail,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);case_rows.append(row);total+=1
                        saved.write(json.dumps(dict(**{name:row[name] for name in keys+('center_x','center_y','radius')},
                            fit_ids=local_learn.tolist(),heldout_ids=local_test.tolist(),center_ids=local_centers.tolist(),
                            global_center_ids=global_centers.tolist(),local_labels=None if local_pred is None else local_pred.tolist(),
                            global_labels=baseline.tolist()),ensure_ascii=False)+'\n')
            for curvature,scale,lam,radius in itertools.product((0,20,50),(.002,.005),(0.,1.),(.005,.01)):
                part=[r for r in case_rows if (r['curvature'],r['scale'],r['normal_weight'],r['radius'])==(curvature,scale,lam,radius)]
                paired=[r for r in part if r['score_status']=='ok' and r['local_fit_status']=='ok']
                assert all(r['global_pair_f1'] is not None and r['local_pair_f1'] is not None for r in paired)
                result=dict(seed_offset=seed,case_index=case,kind=reference['kind'],oracle_common_K=K,curvature=curvature,scale=scale,
                    normal_weight=lam,radius=radius,neighborhood_splits=len(part),paired_scoreable=len(paired),
                    insufficient_fit=sum(r['local_fit_status']!='ok' for r in part),
                    empty_heldout=sum(r['score_status']=='empty_heldout' for r in part),
                    no_member_pairs=sum(r['score_status']=='no_member_pairs' for r in part),
                    no_true_pairs=sum(r['score_status']=='no_true_pairs' for r in part),
                    fit_missing_origin=sum(r['fit_origins']<K for r in part),heldout_missing_origin=sum(r['heldout_origins']<K for r in part),
                    mean_heldout_members=float(np.mean([r['heldout_members'] for r in part])))
                for name in ('global_pair_f1','local_pair_f1','local_minus_global','global_unknown','local_unknown'):
                    result['paired_mean_'+name]=None if not paired else float(np.mean([r[name] for r in paired]))
                summaries.append(result)
            detail.flush();saved.flush();write_csv(out/'summary.csv',summaries)
            print('local member holdout',seed,case,'finished',flush=True)
    assert total==64800 and len(summaries)==1440
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inputs=inputs,source_sha256={name:sha(Path(__file__).parent/name) for name in ('common_support_probe.py','joint_plane_probe.py','analyze.py')},
        reused_global_files={name:sha(previous/name) for name in ('metadata.json','models.jsonl','replicates.csv')},
        rows=total,centers_xy=centers_xy,radii=[.005,.01],random_splits=5,
        evidence='Only fixed x-y neighborhoods and existing random-half member splits. Global train-only models reused, never refitted. '
        'Local planes fit local training members only. Both models score identical local heldout pairs. K is the full common-ID source-count oracle, even if local origins vanish. '
        'All neighborhoods, coverage failures and undefined scores retained. Paired means condition on a usable local fit and at least one true heldout pair, with denominators shown. '
        'Neighborhoods overlap and correlated member splits are not independent scenes. No cross-neighborhood label matching or surface stitching. '
        'Single-source K1 F1 is tautological. Analytic CPU construction, not host GPU or trained-scene physical truth. No training or automatic count.'),indent=2),encoding='utf-8')


def overlap_pairs(records):
    # Each pair is counted once per neighborhood; numeric labels may permute.
    pairs={}
    for record in records:
        ids=record['heldout_ids'];local=record['local_labels'];glob=record['global_labels']
        assert len(ids)==len(set(ids))==len(glob) and (local is None or len(local)==len(ids))
        for a,b in itertools.combinations(range(len(ids)),2):
            key=tuple(sorted((ids[a],ids[b])))
            v=pairs.setdefault(key,[0]*9);v[0]+=1
            if local is None:v[7]+=1
            elif local[a]<0 or local[b]<0:v[8]+=1
            else:v[1]+=1;v[2 if local[a]==local[b] else 3]+=1
            if glob[a]>=0 and glob[b]>=0:v[4]+=1;v[5 if glob[a]==glob[b] else 6]+=1
    return pairs


def overlap_self_check():
    records=[dict(heldout_ids=[3,8,9],local_labels=[0,0,1],global_labels=[0,0,1]),
             dict(heldout_ids=[3,8,9],local_labels=[7,8,8],global_labels=[4,4,5])]
    got=overlap_pairs(records)
    assert got[(3,8)][:4]==[2,2,1,1] and got[(3,9)][:4]==[2,2,0,2]
    changed=[dict(r,local_labels=[100+x for x in r['local_labels']],global_labels=[200+x for x in r['global_labels']]) for r in records]
    assert overlap_pairs(changed)==got
    unknown=[dict(heldout_ids=[3,8],local_labels=[-1,-1],global_labels=[0,0]),
             dict(heldout_ids=[3,8],local_labels=None,global_labels=[1,1])]
    assert overlap_pairs(unknown)[(3,8)]==[2,0,0,0,2,2,0,1,1]
    print('overlap relation/label-permutation/unknown self-check passed',flush=True)


def run_overlap(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    source=root/'local_member_holdout';meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status']=='complete' and meta['rows']==64800
    for item in meta['inputs']:assert sha(Path(item['path']))==item['sha256']
    keys=('seed_offset','case_index','curvature','scale','normal_weight','repeat','radius')
    def rowkey(r):return tuple(float(r[k]) if k in ('scale','normal_weight','radius') else int(r[k]) for k in keys)
    raw={}
    with (source/'replicates.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=rowkey(r)+(float(r['center_x']),float(r['center_y']));assert key not in raw;raw[key]=r
    assert len(raw)==64800
    out.mkdir(parents=True,exist_ok=True);aggregated={};groups_seen=set();processed=0;truth_cache={};writer=None
    names=('observed_pairs','repeated_observed_pairs','pair_observations','local_known_pair_observations',
           'unfitted_pair_observations','unknown_local_pair_observations','local_repeated_known_pairs',
           'local_conflict_pairs','global_repeated_known_pairs','global_conflict_pairs')
    with (source/'models.jsonl').open(encoding='utf-8') as models, (out/'conditions.csv').open('w',encoding='utf-8-sig',newline='') as detail, (out/'witnesses.jsonl').open('w',encoding='utf-8') as witnesses:
        stream=(json.loads(line) for line in models)
        for key,batch in itertools.groupby(stream,key=rowkey):
            assert key not in groups_seen;groups_seen.add(key);records=list(batch);assert len(records)==9
            config=dict(zip(keys,key));seed,case=key[:2]
            if (seed,case) not in truth_cache:
                with np.load(root/f'independent_c0_s{seed}'/f'case_{case:03d}.npz',allow_pickle=False) as z:truth_cache[(seed,case)]=z['source_origin'].copy()
            truth=truth_cache[(seed,case)]
            flags=[];K=None
            for m in records:
                row=raw[key+(m['center_x'],m['center_y'])];K=int(row['oracle_common_K'])
                assert len(m['heldout_ids'])==int(row['heldout_members'])
                assert not set(m['fit_ids'])&set(m['heldout_ids']) and set(m['center_ids'])<=set(m['fit_ids'])
                flags.append(int(row['fit_origins'])==K and int(row['heldout_origins'])==K)
                processed+=1
            for coverage in ('all','complete_origins','missing_origin'):
                selected=[m for m,complete in zip(records,flags) if coverage=='all' or complete==(coverage=='complete_origins')]
                pairs=overlap_pairs(selected)
                for kind in ('all','same_origin','different_origin'):
                    part={pair:v for pair,v in pairs.items() if kind=='all' or (truth[pair[0]]==truth[pair[1]])==(kind=='same_origin')}
                    counts=[len(part),sum(v[0]>=2 for v in part.values()),sum(v[0] for v in part.values()),
                            sum(v[1] for v in part.values()),sum(v[7] for v in part.values()),sum(v[8] for v in part.values()),
                            sum(v[1]>=2 for v in part.values()),sum(v[2]>0 and v[3]>0 for v in part.values()),
                            sum(v[4]>=2 for v in part.values()),sum(v[5]>0 and v[6]>0 for v in part.values())]
                    assert counts[-1]==0, 'Cached global relation must be invariant within one split'
                    row=dict(**config,oracle_common_K=K,source_coverage=coverage,pair_origin=kind,neighborhoods=len(selected),**dict(zip(names,counts)))
                    row['local_conflict_fraction']=counts[7]/counts[6] if counts[6] else None
                    if writer is None:writer=csv.DictWriter(detail,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    aggkey=(config['curvature'],config['scale'],config['normal_weight'],config['radius'],K,coverage,kind)
                    if aggkey not in aggregated:aggregated[aggkey]=dict(conditions=0,**{name:0 for name in names},conditions_with_known_pairs=0,conditions_with_conflict=0,sum_defined_conflict_fraction=0.)
                    agg=aggregated[aggkey];agg['conditions']+=1
                    for name in names:agg[name]+=row[name]
                    if row['local_conflict_fraction'] is not None:
                        agg['conditions_with_known_pairs']+=1;agg['sum_defined_conflict_fraction']+=row['local_conflict_fraction']
                    agg['conditions_with_conflict']+=row['local_conflict_pairs']>0
                    if coverage=='all' and kind=='all':
                        for pair,v in part.items():
                            if v[2] and v[3]:
                                views=[]
                                for m in selected:
                                    if pair[0] in m['heldout_ids'] and pair[1] in m['heldout_ids'] and m['local_labels'] is not None:
                                        a=m['heldout_ids'].index(pair[0]);b=m['heldout_ids'].index(pair[1]);labels=[m['local_labels'][a],m['local_labels'][b]]
                                        views.append(dict(center_x=m['center_x'],center_y=m['center_y'],labels=labels))
                                witnesses.write(json.dumps(dict(**config,pair=list(pair),same_origin=bool(truth[pair[0]]==truth[pair[1]]),neighborhoods=views))+'\n');break
            if len(groups_seen)%120==0:detail.flush();witnesses.flush();print('overlap groups',len(groups_seen),flush=True)
    assert processed==64800 and len(groups_seen)==7200
    summary=[]
    for key,agg in sorted(aggregated.items()):
        row=dict(zip(('curvature','scale','normal_weight','radius','oracle_common_K','source_coverage','pair_origin'),key));row.update(agg)
        row['pooled_local_conflict_fraction']=agg['local_conflict_pairs']/agg['local_repeated_known_pairs'] if agg['local_repeated_known_pairs'] else None
        row['mean_condition_conflict_fraction']=agg['sum_defined_conflict_fraction']/agg['conditions_with_known_pairs'] if agg['conditions_with_known_pairs'] else None
        summary.append(row)
    write_csv(out/'summary.csv',summary)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        input_files={name:sha(source/name) for name in ('metadata.json','models.jsonl','replicates.csv')},groups=len(groups_seen),processed_neighborhoods=processed,
        evidence='Read-only saved local/global predictions, no fitting or generation. Per seed/case/kappa/scale/angle/repeat/radius; compare pair co-membership, not numeric labels. '
        'Repeated known pairs require at least two neighborhoods with both labels known. Empty/unknown/unfitted observations retained separately. '
        'Counts pool pair-condition units, not independent Gaussian pairs across splits; mean condition ratios also reported. Origin coverage subgroups do not assess cross-subgroup conflicts. '
        'One representative conflict witness per condition, not all conflicts. No voting/closure/stitching or global physical membership axiom.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--heldout',action='store_true');p.add_argument('--self-check',action='store_true')
    p.add_argument('--single-only',action='store_true');p.add_argument('--given-k',type=int,choices=(1,2))
    p.add_argument('--local-heldout',action='store_true')
    p.add_argument('--overlap-audit',action='store_true')
    args=p.parse_args()
    if args.self_check:overlap_self_check() if args.overlap_audit else local_self_check() if args.local_heldout else holdout_self_check()
    else:
        if args.root is None or args.output is None:p.error('root and output required')
        if args.overlap_audit and (args.local_heldout or args.heldout or args.single_only or args.given_k is not None):p.error('overlap audit only reads fixed saved predictions')
        if args.local_heldout and (args.heldout or args.single_only or args.given_k is not None):p.error('local holdout uses its fixed cached global control')
        if (args.single_only or args.given_k is not None) and not args.heldout:p.error('count control requires --heldout')
        if args.given_k is not None and not args.single_only:p.error('--given-k requires --single-only')
        if args.overlap_audit:run_overlap(args.root,args.output)
        elif args.local_heldout:run_local(args.root,args.output)
        elif args.heldout:run_holdout(args.root,args.output,args.given_k,args.single_only)
        else:run(args.root,args.output)
