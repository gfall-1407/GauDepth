"""Fixed-K conditional mechanism/scene probes; does not estimate physical layer count."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from analyze import pair_metrics
from analyze import group_ray
from latent_competition import fit
from mechanism_sweep import proposals


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def stats(result):
    labels = np.asarray(result['labels'])
    return dict(objective=result['objective'], mean_entropy=result['mean_entropy'],
                collapsed=result['collapsed'], valid_count=result['valid_count'],
                unknown_count=result['unknown_count'],
                hard_groups=len(set(labels.tolist())-{-1}),
                near_optimal_count=result['near_optimal_count'],
                next_objective_gap=result['next_objective_gap'])


def mechanism(out, repeats=20, seed=20261022,objective='soft_uniform',single_force_two=False):
    rows = []
    conditions = 0
    for sources in (1,2):
        if single_force_two and sources==2:continue
        gaps = (0.,) if sources == 1 else (0.,.005,.02,.05)
        directions = (0.,) if sources == 1 else (0.,10.,30.,60.)
        for gap, angle, dz, dn in itertools.product(gaps,directions,(0.,.002,.008),(0.,5.,15.,30.)):
            config = dict(sources=sources, depth_gap=gap, direction_gap_deg=angle,
                          depth_noise=dz, normal_noise_deg=dn, oracle_K=sources,
                          given_K=2 if single_force_two else sources)
            for rep in range(repeats):
                rng=np.random.default_rng(seed+conditions*repeats+rep)
                d,n,truth=proposals(rng,sources,gap,angle,dz,dn,12)
                for lam in (0.,1.):
                    result=fit(d,n,np.ones(len(d),bool),2 if single_force_two else sources,
                               normal_weight=lam,objective=objective)
                    score=pair_metrics(truth,result['labels'])
                    rows.append(dict(**config, repeat=rep, normal_weight=lam,
                                     pair_f1=score['f1'], false_merge_pairs=score['fp'],
                                     false_split_pairs=score['fn'], **stats(result)))
            conditions+=1
            if conditions % 32 == 0:
                print(f'conditional mechanism {conditions}/204',flush=True)
    write_csv(out/'replicates.csv',rows)
    summary=[]
    keys=('pair_f1','false_merge_pairs','false_split_pairs','objective','mean_entropy',
          'collapsed','unknown_count','hard_groups','near_optimal_count')
    configs=sorted(set(tuple(r[k] for k in
        ('sources','depth_gap','direction_gap_deg','depth_noise','normal_noise_deg','oracle_K','given_K','normal_weight')) for r in rows))
    for values in configs:
        fields=('sources','depth_gap','direction_gap_deg','depth_noise','normal_noise_deg','oracle_K','given_K','normal_weight')
        config=dict(zip(fields,values));part=[r for r in rows if all(r[k]==v for k,v in config.items())]
        summary.append(dict(**config,repeats=len(part),**{f'mean_{k}':float(np.mean([r[k] for r in part])) for k in keys}))
    write_csv(out/'summary.csv',summary)
    return dict(seed=seed,repeats=repeats,conditions=conditions,generated_cases=conditions*repeats,
                oracle_K=not single_force_two,single_source_forced_K2=single_force_two,
                evidence='Known source count only except explicit single-source wrong-K stress; labels only for scoring. '
                'K=1 produces one group by construction and cannot demonstrate layer-number discovery. '
                'Same feature proposals/seeds as mechanism_sweep.py; not a Gaussian renderer.')


def scenes(inputs,out,objective='soft_uniform'):
    files=[]
    for name in inputs:
        p=Path(name);files.extend(sorted(p.glob('view_*.npz')) if p.is_dir() else [p])
    files=sorted(set(p.resolve() for p in files))
    if not files:
        raise ValueError('No scene NPZ files')
    rows=[];manifests=[]
    with (out/'assignments.jsonl').open('w',encoding='utf-8') as assignments:
        for path in files:
            start=time.monotonic()
            meta=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8'))
            view=next(v for v in meta['views'] if v['file']==path.name)
            assert sha(path)==view['npz_sha256']
            with np.load(path,allow_pickle=False) as z:
                for ray in range(len(z['offsets'])-1):
                    if z['qualitative'][ray]:continue
                    a,b=map(int,z['offsets'][ray:ray+2]);d,n,v=[z[k][a:b] for k in ('depth','normal','normal_valid')]
                    for lam in (0.,1.):
                        for K in (1,2):
                            result=fit(d,n,v,K,normal_weight=lam,objective=objective)
                            config=dict(input_dir=path.parent.name,view=path.stem,ray=ray,
                                        conditional_K=K,normal_weight=lam)
                            rows.append(dict(**config,**stats(result)))
                            assignments.write(json.dumps(dict(**config,**result,
                                gaussian_ids=z['gaussian_id'][a:b].tolist()),ensure_ascii=False)+'\n')
            manifests.append(dict(path=str(path),npz_sha256=sha(path)))
            write_csv(out/(path.parent.name+'_'+path.stem+'.csv'),[r for r in rows
                      if r['input_dir']==path.parent.name and r['view']==path.stem])
            print(path.parent.name,path.stem,'seconds',round(time.monotonic()-start,2),flush=True)
    write_csv(out/'rays.csv',rows)
    summaries=[]
    for K,lam in itertools.product((1,2),(0.,1.)):
        part=[r for r in rows if r['conditional_K']==K and r['normal_weight']==lam]
        keys=('objective','mean_entropy','collapsed','unknown_count','hard_groups','near_optimal_count')
        summaries.append(dict(conditional_K=K,normal_weight=lam,rays=len(part),
            **{f'mean_{key}':float(np.mean([r[key] for r in part if r[key] is not None])) for key in keys}))
    write_csv(out/'summary.csv',summaries)
    return dict(inputs=manifests,oracle_K=False,
                evidence='K=1 and K=2 are separate conditional probes, not selected physical layer counts. '
                'Energy and entropy are model diagnostics, not calibrated confidence or geometric accuracy.')


def imbalance(out,objective='soft_uniform'):
    """Fixed two prototypes; change only how often each is represented."""
    rows=[];details=[]
    for na,nb in ((12,12),(4,20),(2,22)):
        truth=np.r_[np.zeros(na,int),np.ones(nb,int)]
        d=1.+.005*truth
        theta=np.deg2rad(30.*truth)
        n=np.c_[np.sin(theta),np.zeros(len(truth)),-np.cos(theta)]
        for lam in (0.,1.):
            result=fit(d,n,np.ones(len(d),bool),2,normal_weight=lam,objective=objective)
            D=2.5*np.log1p(1./4)+lam
            mixed=np.log(2.)-np.logaddexp(0.,-D) if objective=='soft_uniform' else 0.
            repeated_majority=na/(na+nb)*D
            repeated_minority=nb/(na+nb)*D
            assert np.isclose(result['objective'],min(mixed,repeated_majority,repeated_minority),atol=1e-10,rtol=0)
            rows.append(dict(origin_A_members=na,origin_B_members=nb,normal_weight=lam,
                             prototype_energy_gap=D,analytic_mixed_loss=mixed,
                             analytic_repeated_majority_loss=repeated_majority,
                             minority_collapse_threshold=mixed/D,
                             prototype_A_label=result['labels'][0],prototype_B_label=result['labels'][na],
                             **stats(result)))
            details.append(dict(origin_A_members=na,origin_B_members=nb,**result))
    write_csv(out/'summary.csv',rows)
    (out/'assignments.json').write_text(json.dumps(details),encoding='utf-8')
    return dict(K=2,depth_prototypes=[1.,1.005],direction_gap_degrees=30.,noise=0.,
                evidence='Same unique geometric proposals, different Gaussian multiplicities; objective identifies the ablation. '
                'Uniform-soft may duplicate the majority hypothesis; hard-min is a matched objective control. '
                'Collapse is abstention, not confident false membership. Exact analytic three-solution check.')


def clone_control(out, objective):
    """Exploratory single-ray representation counterexample, NOT scene densification."""
    x=np.array([0.,.7,1.9,3.2]);depth=np.exp(x*np.log1p(.005))
    normal=np.tile([0.,0.,-1.],(4,1));color=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.25,.5,.75]])
    energy=2.5*np.log1p(((np.log(depth)[:,None]-np.log(depth))/np.log1p(.005))**2/4.)
    candidates=list(itertools.combinations_with_replacement(range(4),2))
    def replay(ids,alpha):
        T=np.r_[1.,np.cumprod(1.-alpha)[:-1]];w=T*alpha
        assert np.all(alpha>=1/255) and np.all(T*(1-alpha)>=1e-4)
        return np.r_[w.sum(),w@depth[ids],w@color[ids],w@normal[ids]],np.bincount(ids,w,minlength=4)
    reference,reference_mass=replay(np.arange(4),np.full(4,.2))
    baseline=fit(depth,normal,np.ones(4,bool),2,normal_weight=0.,objective=objective)
    base=np.asarray(baseline['labels'])
    def relation(labels):
        i,j=np.triu_indices(4,1)
        return (labels[i]==labels[j])&(labels[i]>=0)&(labels[j]>=0)
    baseline_relation=relation(base);rows=[];details=[]
    for duplicate,copies in itertools.product(range(4),(1,2,4,8,16)):
        multiplicity=np.ones(4,int);multiplicity[duplicate]=copies
        ids=np.repeat(np.arange(4),multiplicity)
        alpha=np.full(len(ids),.2)
        alpha[ids==duplicate]=-np.expm1(np.log1p(-.2)/copies)
        rendered,mass=replay(ids,alpha)
        np.testing.assert_allclose(rendered,reference,atol=2e-14,rtol=0)
        np.testing.assert_allclose(mass,reference_mass,atol=2e-14,rtol=0)
        result=fit(depth[ids],normal[ids],np.ones(len(ids),bool),2,normal_weight=0.,objective=objective)
        positions=np.r_[0,np.cumsum(multiplicity)[:-1]]
        labels=np.asarray(result['labels'])[positions]
        vals=[];partitions=[]
        for a,b in candidates:
            E=energy[:,[a,b]]
            cost=np.minimum(E[:,0],E[:,1]) if objective=='hard_min' else np.log(2.)-np.logaddexp(-E[:,0],-E[:,1])
            vals.append(float(cost@multiplicity/multiplicity.sum()))
            pred=np.argmin(E,axis=1);pred[np.isclose(E[:,0],E[:,1],atol=1e-10,rtol=0)]=-1
            partitions.append(relation(pred))
        np.testing.assert_allclose(result['objective'],min(vals),atol=1e-12,rtol=0)
        old_partition_optimal=any(np.array_equal(p,baseline_relation) for value,p in zip(vals,partitions) if value<=min(vals)+1e-10)
        greedy=group_ray(depth[ids],normal[ids],np.ones(len(ids),bool),geometry=False)[0][positions]
        greedy_base=group_ray(depth,normal,np.ones(4,bool),geometry=False)[0]
        assert np.array_equal(relation(greedy),relation(greedy_base))
        rows.append(dict(duplicated_prototype=duplicate,copies=copies,members=len(ids),
            prototype_labels=json.dumps(labels.tolist()),baseline_labels=json.dumps(base.tolist()),
            changed_original_pairs=int((relation(labels)!=baseline_relation).sum()),
            baseline_partition_still_optimal=old_partition_optimal,
            replay_max_error=float(np.abs(rendered-reference).max()),
            prototype_mass_max_error=float(np.abs(mass-reference_mass).max()),
            original_prototype_unknown=int((labels<0).sum()),**stats(result)))
        details.append(dict(duplicated_prototype=duplicate,copies=copies,original_ids=ids.tolist(),
            alpha=alpha.tolist(),rendered_moments=rendered.tolist(),**result))
    write_csv(out/'summary.csv',rows)
    (out/'assignments.json').write_text(json.dumps(details),encoding='utf-8')
    return dict(K=2,log_depth_units=x.tolist(),base_alpha=.2,base_depth=depth.tolist(),
        evidence='Constructed single-ray counterexample chosen during exploration. Adjacent clones use '
        'alpha_c=1-(1-alpha)^(1/copies), preserving front-to-back opacity, depth/RGB/normal moments '
        'and summed original-prototype contribution. All clones remain accepted. '
        'This alpha identity is pixel-specific, not a globally equivalent Gaussian scene or host replay. '
        'Finite objectives independently enumerated on the four prototypes with multiplicities. '
        'baseline_partition_still_optimal checks all optima within 1e-10, separating tie-breaking from a changed objective. '
        'Exact-feature deduplication would remove this constructed multiplicity effect but is not validated for near-duplicate real Gaussians.')


def ray_clone_scene(inputs, out, mass_mode='uniform'):
    """Preserve accepted-ray algebra, compare original-member relations only."""
    rows=[];manifests=[];details=[]
    for folder in map(Path,inputs):
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        for item in meta['views']:
            path=folder/item['file'];assert sha(path)==item['npz_sha256']
            manifests.append(dict(path=str(path),sha256=sha(path),metadata_sha256=sha(folder/'metadata.json')))
            with np.load(path,allow_pickle=False) as z:
                selected=np.flatnonzero(~z['qualitative'])[:8]
                for ray in selected:
                    a,b=map(int,z['offsets'][ray:ray+2]);sl=slice(a,b)
                    d,n,v,w,alpha=[z[k][sl] for k in ('depth','normal','normal_valid','weight','alpha')]
                    alpha=alpha.astype(float);w=w.astype(float);d=d.astype(float);n=n.astype(float)
                    copies=4;eligible=np.flatnonzero(alpha>=-np.expm1(copies*np.log1p(-1/255)))
                    if not len(eligible):
                        raise ValueError(f'No splittable member: {path} ray {ray}')
                    targets={'max_original_w':int(eligible[np.argmax(w[eligible])]),
                             'min_original_w_eligible':int(eligible[np.argmin(w[eligible])])}
                    def replay(ids,aa):
                        T=np.r_[1.,np.cumprod(1-aa)[:-1]];ww=T*aa
                        assert np.all(aa>=1/255-1e-12) and np.all(T*(1-aa)>=1e-4-1e-10)
                        return np.r_[ww.sum(),ww@d[ids],ww@n[ids]],np.bincount(ids,ww,minlength=len(d))
                    original_ids=np.arange(len(d));reference,mass_reference=replay(original_ids,alpha)
                    def fitting_mass(aa):
                        if mass_mode=='uniform':return None
                        if mass_mode=='alpha':return aa
                        if mass_mode=='tau':return -np.log1p(-aa)
                        return np.r_[1.,np.cumprod(1-aa)[:-1]]*aa
                    base_mass=fitting_mass(alpha)
                    i,j=np.triu_indices(len(d),1);pair_mass=w[i]*w[j]
                    for lam in (0.,1.):
                        base=fit(d,n,v,2,normal_weight=lam,objective='hard_min',member_mass=base_mass);base_labels=np.asarray(base['labels'])
                        base_known=(base_labels[i]>=0)&(base_labels[j]>=0)
                        base_same=base_known&(base_labels[i]==base_labels[j])
                        for strategy,target in targets.items():
                            mult=np.ones(len(d),int);mult[target]=copies;ids=np.repeat(original_ids,mult)
                            aa=alpha[ids].copy();aa[ids==target]=-np.expm1(np.log1p(-alpha[target])/copies)
                            rendered,mass=replay(ids,aa)
                            np.testing.assert_allclose(rendered,reference,atol=2e-12,rtol=0)
                            np.testing.assert_allclose(mass,mass_reference,atol=2e-12,rtol=0)
                            split_mass=fitting_mass(aa)
                            if mass_mode in ('tau','w'):
                                np.testing.assert_allclose(np.bincount(ids,split_mass,minlength=len(d)),base_mass,atol=2e-12,rtol=0)
                            result=fit(d[ids],n[ids],v[ids],2,normal_weight=lam,objective='hard_min',member_mass=split_mass)
                            objective_change=abs(base['objective']-result['objective'])
                            if mass_mode in ('tau','w'):assert objective_change<=2e-10
                            pos=np.r_[0,np.cumsum(mult)[:-1]];labels=np.asarray(result['labels'])[pos]
                            known=base_known&(labels[i]>=0)&(labels[j]>=0)
                            same=(labels[i]==labels[j])&(labels[i]>=0)&(labels[j]>=0)
                            changed=known&(same!=base_same)
                            row=dict(input_dir=folder.name,view=path.stem,ray=int(ray),normal_weight=lam,member_measure=mass_mode,
                                strategy=strategy,original_members=len(d),copies=copies,target_member=target,
                                target_gaussian_id=int(z['gaussian_id'][a+target]),target_original_w=float(w[target]),
                                target_alpha=float(alpha[target]),eligible_members=len(eligible),
                                target_original_mass_fraction=float(w[target]/w.sum()),
                                known_original_pairs=int(known.sum()),changed_original_pairs=int(changed.sum()),
                                changed_pair_fraction=float(changed.sum()/known.sum()) if known.any() else None,
                                changed_original_pair_mass_fraction=float(pair_mass[changed].sum()/pair_mass[known].sum()) if pair_mass[known].sum()>0 else None,
                                original_unknown_before=int((base_labels<0).sum()),original_unknown_after=int((labels<0).sum()),
                                unique_optimum_before=base['near_optimal_count']==1,unique_optimum_after=result['near_optimal_count']==1,
                                collapsed_before=base['collapsed'],collapsed_after=result['collapsed'],
                                absolute_objective_change=objective_change,
                                replay_max_error=float(np.abs(rendered-reference).max()),
                                original_mass_max_error=float(np.abs(mass-mass_reference).max()))
                            rows.append(row)
                            if changed.any() and len(details)<12:
                                details.append(dict(**row,base_labels=base_labels.tolist(),cloned_original_labels=labels.tolist(),
                                    original_gaussian_ids=z['gaussian_id'][sl].tolist(),source_sha256=sha(path)))
            write_csv(out/'rays.csv',rows)
            print('real-ray clone',folder.name,path.stem,'finished',flush=True)
    summary=[]
    for strategy,lam in itertools.product(('max_original_w','min_original_w_eligible'),(0.,1.)):
        part=[r for r in rows if r['strategy']==strategy and r['normal_weight']==lam]
        unique=[r for r in part if r['unique_optimum_before'] and r['unique_optimum_after']]
        summary.append(dict(strategy=strategy,normal_weight=lam,member_measure=mass_mode,cases=len(part),
            changed_cases=sum(r['changed_original_pairs']>0 for r in part),
            joint_unique_optimum_cases=len(unique),unique_changed_cases=sum(r['changed_original_pairs']>0 for r in unique),
            **{key:float(np.mean([r[key] for r in part if r[key] is not None])) for key in
                ('changed_pair_fraction','changed_original_pair_mass_fraction','target_original_mass_fraction','original_unknown_before','original_unknown_after')},
            max_replay_error=max(r['replay_max_error'] for r in part),max_original_mass_error=max(r['original_mass_max_error'] for r in part),
            max_objective_change=max(r['absolute_objective_change'] for r in part)))
    write_csv(out/'summary.csv',summary)
    (out/'witnesses.json').write_text(json.dumps(details),encoding='utf-8')
    return dict(inputs=manifests,per_view_rays=8,normal_weights=[0.,1.],conditional_K=2,member_measure=mass_mode,
        evidence='First eight of the existing randomly sampled rays per view, not a new independent scene sample. '
        'Single accepted-ray alpha algebra only; no host renderer or globally equivalent Gaussian-scene claim. '
        'One eligible member cloned four times adjacently; both max/min original w targets restricted to eligible members. '
        'All original-member contribution masses and alpha-derived opacity/depth/normal moments preserved in float64. '
        'Changed pairs require known labels both before and after; unknown/collapse separately reported. '
        'Positive fitting mass is an explicit ablation, not physical support: uniform, alpha, tau=-log(1-alpha), or alpha-derived float64 w=Talpha. '
        'For tau/w both sides use matching float64 mass; source float32 w only selects targets and describes pair changes. '
        'Additive mass makes this constructed split invariant by design, not evidence of accurate membership. '
        'Fixed conditional K2 sensitivity, not physical accuracy. Pair-mass change is descriptive original rendering weight, not support truth.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mechanism',action='store_true');p.add_argument('--imbalance',action='store_true')
    p.add_argument('--clone-control',action='store_true')
    p.add_argument('--ray-clone-input',action='append')
    p.add_argument('--member-mass',choices=('uniform','alpha','tau','w'),default='uniform')
    p.add_argument('--input',action='append')
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--objective',choices=('soft_uniform','hard_min'),default='soft_uniform')
    p.add_argument('--single-force-two',action='store_true',help='Wrong-K stress on the 12 single-source conditions only')
    args=p.parse_args();out=Path(args.output)
    if args.ray_clone_input and args.objective!='hard_min':p.error('--ray-clone-input requires --objective hard_min')
    if args.member_mass!='uniform' and not args.ray_clone_input:p.error('--member-mass only applies to --ray-clone-input')
    if sum(map(bool,(args.mechanism,args.imbalance,args.input,args.clone_control,args.ray_clone_input)))!=1:p.error('Choose one probe mode')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    if args.single_force_two and not args.mechanism:p.error('--single-force-two requires --mechanism')
    metadata=(mechanism(out,args.repeats,objective=args.objective,single_force_two=args.single_force_two)
              if args.mechanism else imbalance(out,args.objective) if args.imbalance else
              clone_control(out,args.objective) if args.clone_control else
              ray_clone_scene(args.ray_clone_input,out,args.member_mass) if args.ray_clone_input else scenes(args.input,out,args.objective))
    metadata['objective']=args.objective
    metadata['responsibility_note']='Softmax energy assignments; for hard_min they are post-fit descriptions, not the optimized objective.'
    root=Path(__file__).parent
    metadata.update(source_sha256={n:sha(root/n) for n in
        ('competition_probe.py','latent_competition.py','mechanism_sweep.py','analyze.py')})
    metadata['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
