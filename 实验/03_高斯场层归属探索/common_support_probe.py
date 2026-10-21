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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--heldout',action='store_true');p.add_argument('--self-check',action='store_true')
    p.add_argument('--single-only',action='store_true');p.add_argument('--given-k',type=int,choices=(1,2))
    args=p.parse_args()
    if args.self_check:holdout_self_check()
    else:
        if args.root is None or args.output is None:p.error('root and output required')
        if (args.single_only or args.given_k is not None) and not args.heldout:p.error('count control requires --heldout')
        if args.given_k is not None and not args.single_only:p.error('--given-k requires --single-only')
        if args.heldout:run_holdout(args.root,args.output,args.given_k,args.single_only)
        else:run(args.root,args.output)
