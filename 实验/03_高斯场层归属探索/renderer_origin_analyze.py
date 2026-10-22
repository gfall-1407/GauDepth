"""Score construction-origin recovery on the accepted synthetic renderer members."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from analyze import group_ray, pair_metrics
from latent_competition import fit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def weighted_pair_score(truth, labels, weights):
    i,j=np.triu_indices(len(truth),1)
    mass=np.asarray(weights,float)[i]*np.asarray(weights,float)[j]
    actual=truth[i]==truth[j]
    predicted=(labels[i]==labels[j])&(labels[i]>=0)&(labels[j]>=0)
    tp=float(mass[actual&predicted].sum());fp=float(mass[~actual&predicted].sum())
    fn=float(mass[actual&~predicted].sum())
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else None


def origin_pair_precision(truth, labels, origin):
    """Penalize cross-origin merging even when within-origin recall is perfect."""
    i,j=np.triu_indices(len(truth),1)
    pred=(labels[i]==labels[j])&(labels[i]>=0)&(labels[j]>=0)
    tp=int((pred&(truth[i]==origin)&(truth[j]==origin)).sum())
    fp=int((pred&(truth[i]!=truth[j])&((truth[i]==origin)|(truth[j]==origin))).sum())
    return tp/(tp+fp) if tp+fp else None


def run(inputs,out,measures=('uniform',),objective='soft_uniform'):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    rows=[];manifests=[];seen=set()
    for folder in map(Path,inputs):
        meta_path=folder/'metadata.json';meta=json.loads(meta_path.read_text(encoding='utf-8'))
        assert meta['status']=='complete'
        for item in meta['cases']:
            config=item['config'];case=int(config['case_index'])
            assert case not in seen,f'Duplicate case {case}'
            seen.add(case);path=folder/item['file'];assert sha(path)==item['npz_sha256']
            label_records=[]
            with np.load(path,allow_pickle=False) as z:
                for ray in range(len(z['offsets'])-1):
                    if z['qualitative'][ray]:continue
                    a,b=map(int,z['offsets'][ray:ray+2]);d,n,v,w,truth=[z[k][a:b] for k in
                        ('depth','normal','normal_valid','weight','truth_origin')]
                    K=len(np.unique(truth))
                    labels_by_method={}
                    for geom in (False,True):
                        labels,_=group_ray(d,n,v,.01,15.,geom)
                        labels_by_method['hard_normal' if geom else 'depth']=(labels,None,None)
                    # Both base and weighted ablations use the same conditional K and energy.
                    alpha=z['alpha'][a:b].astype(float)
                    T=np.r_[1.,np.cumprod(1-alpha)[:-1]]
                    for measure in measures:
                        mass={'uniform':None,'tau':-np.log1p(-alpha),'w':T*alpha}[measure]
                        for lam in (0.,1.):
                            result=fit(d,n,v,K,normal_weight=lam,objective=objective,member_mass=mass) if K else None
                            name=('conditional_normal' if lam else 'conditional_depth')
                            if measures!=('uniform',):name+='_'+measure
                            labels_by_method[name]=(
                                np.asarray(result['labels']) if result else np.full(len(d),-1),
                                result['mean_entropy'] if result else None,
                                result['collapsed'] if result else None)
                    total=float(w.astype(float).sum())
                    origin_mass=[float(w[truth==k].astype(float).sum()) for k in range(len(config['theta_deg']))]
                    label_records.append(dict(ray=ray,gaussian_id=z['gaussian_id'][a:b].tolist(),
                        truth_origin=truth.tolist(),labels={name:value[0].tolist() for name,value in labels_by_method.items()}))
                    for method,(labels,entropy,collapsed) in labels_by_method.items():
                        score=pair_metrics(truth,labels)
                        source_recalls=[]
                        for k in range(len(config['theta_deg'])):
                            ix=np.flatnonzero(truth==k)
                            source_recalls.append(pair_metrics(truth[ix],labels[ix])['recall'] if len(ix)>1 else None)
                        rows.append(dict(case_index=case,kind=config['kind'],gap=config['gap'],
                            center_noise=config['center_noise'],normal_noise_deg=config['normal_noise_deg'],
                            ray=ray,method=method,accepted_members=len(d),nominal_origins=len(config['theta_deg']),
                            accepted_origins=K,oracle_K=K if method.startswith('conditional') else None,
                            both_origins_accepted=K==2,
                            min_original_origin_mass_fraction=min(origin_mass)/total if total else None,
                            pair_f1=score['f1'],false_merge_pairs=score['fp'],false_split_pairs=score['fn'],
                            original_pair_mass_f1=weighted_pair_score(truth,labels,w),
                            source0_pair_recall=source_recalls[0],
                            source1_pair_recall=source_recalls[1] if len(source_recalls)>1 else None,
                            source0_pair_precision=origin_pair_precision(truth,labels,0),
                            source1_pair_precision=origin_pair_precision(truth,labels,1) if len(source_recalls)>1 else None,
                            groups=len(set(labels.tolist())-{-1}),unknown_members=int((labels<0).sum()),
                            mean_entropy=entropy,collapsed=collapsed))
            manifests.append(dict(file=str(path),sha256=sha(path),metadata_sha256=sha(meta_path),
                                  runtime_codeHEAD=meta['codeHEAD']))
            print('origin scores case',case,flush=True)
            out.mkdir(parents=True,exist_ok=True);write_csv(out/'rays.csv',rows)
            (out/f'labels_{case:03d}.json').write_text(json.dumps(dict(config=config,source_file=str(path),
                source_sha256=sha(path),rays=label_records)),encoding='utf-8')
    summaries=[]
    metrics=('pair_f1','original_pair_mass_f1','false_merge_pairs','false_split_pairs','groups',
             'unknown_members','both_origins_accepted','min_original_origin_mass_fraction',
             'source0_pair_recall','source1_pair_recall','source0_pair_precision','source1_pair_precision','mean_entropy','collapsed')
    for case in sorted(seen):
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['case_index']==case and r['method']==method]
            config={k:part[0][k] for k in ('case_index','kind','gap','center_noise','normal_noise_deg')}
            summaries.append(dict(**config,method=method,rays=len(part),
                **{f'mean_{key}':float(np.mean([r[key] for r in part if r[key] is not None]))
                    if any(r[key] is not None for r in part) else None for key in metrics}))
    write_csv(out/'summary.csv',summaries)
    root=Path(__file__).parent
    (out/'metadata.json').write_text(json.dumps(dict(inputs=manifests,objective=objective,member_measures=measures,
        status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={n:sha(root/n) for n in ('renderer_origin_analyze.py','latent_competition.py','analyze.py')},
        interpretation='Construction origins of accepted members only. Hidden/pruned members are not assigned background. '
        'Conditional methods receive the oracle number of accepted origins, other methods do not; not an equal-information comparison. '
        'Single-origin conditional K=1 success is tautological. Perturbed origins are not physical truth in trained scenes. '
        'Original pair-mass F1 is supplementary; unweighted and per-origin recall are retained. '
        'Fitting tau/w is an explicit positive-measure ablation, not physical support strength; w derived consistently from float64 alpha. '
        'Single-source K1 success is imposed. All 30 adjacent-pixel cases are correlated, not independent scene samples.'),indent=2),encoding='utf-8')


def ray_profiles(data,rays):
    """Independent full-matrix Gaussian restriction, with camera-z ray parameter."""
    mu=np.asarray(data['means'],float);scales=np.asarray(data['scales'],float);q=np.asarray(data['rotations'],float)
    good=np.isfinite(mu).all(axis=1)&np.isfinite(scales).all(axis=1)&(scales>0).all(axis=1)&np.isfinite(q).all(axis=1)&(np.linalg.norm(q,axis=1)>1e-12)
    with np.errstate(invalid='ignore',divide='ignore'):
        q=np.where(good[:,None],q,np.nan);q=q/np.linalg.norm(q,axis=1)[:,None]
        w,x,y,z=q.T
        rot=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                      [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                      [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]).transpose(2,0,1)
        inv_var=1/np.where(good[:,None],scales,np.nan)**2
        precision=np.einsum('nij,nj,nkj->nik',rot,inv_var,rot)
        denominator=np.einsum('ri,nij,rj->rn',rays,precision,rays)
        numerator=np.einsum('ri,nij,nj->rn',rays,precision,mu)
        peak=numerator/denominator;sigma=1/np.sqrt(denominator)
    valid=good[None,:]&np.isfinite(peak)&np.isfinite(sigma)&(sigma>0)
    return peak,sigma,valid


def patch_intersections(ray,cfg,origin):
    dx,dy,dz=map(float,ray);kappa=float(cfg.get('curvature',0.))
    a=kappa*(dx*dx+dy*dy);b=dz+np.tan(np.deg2rad(cfg['theta_deg'][origin]))*dx;c=1.+origin*cfg['gap']
    roots=np.roots([a,-b,c]) if abs(a)>1e-14 else np.array([c/b]) if abs(b)>1e-14 else np.array([])
    positive=[]
    for root in roots:
        if abs(np.imag(root))>1e-10:continue
        z=float(np.real(root))
        if np.isfinite(z) and z>0 and not any(np.isclose(z,old,atol=1e-10,rtol=1e-10) for old in positive):positive.append(z)
    inside=[z for z in sorted(positive) if abs(z*dx)<=.01+1e-12 and abs(z*dy)<=.01+1e-12]
    return inside,len(positive)-len(inside)


def profile_self_check():
    cfg=dict(theta_deg=[0.,0.],gap=.005,curvature=0.)
    roots,excluded=patch_intersections([.005,-.004,1.],cfg,1)
    np.testing.assert_allclose(roots,[1.005],atol=1e-12);assert excluded==0
    cfg['curvature']=50.
    roots,excluded=patch_intersections([.005,0.,1.],cfg,0)
    assert len(roots)==1 and excluded==1
    np.testing.assert_allclose(roots[0],1.+50*(.005*roots[0])**2,atol=1e-12)
    assert patch_intersections([.05,0.,1.],cfg,0)==([],2)
    assert patch_intersections([0.,0.,-1.],cfg,0)==([],0)
    rays=np.array([[.01,-.02,1.]])
    data=dict(means=np.array([[0.,0.,1.]]),scales=np.array([[.006,.006,.0005]]),rotations=np.array([[1.,0.,0.,0.]]))
    peak,sigma,valid=ray_profiles(data,rays);assert valid.all()
    omega=np.diag(1/data['scales'][0]**2);direction=rays[0]
    np.testing.assert_allclose(sigma[0,0]**2,1/(direction@omega@direction),rtol=1e-12)
    v0=peak[0,0]*direction-data['means'][0];v1=(peak[0,0]+sigma[0,0])*direction-data['means'][0]
    np.testing.assert_allclose(v1@omega@v1-v0@omega@v0,1.,atol=1e-9)
    data['rotations']=np.array([[0.,1.,0.,0.]])
    flipped=ray_profiles(data,rays);np.testing.assert_allclose(flipped[0],peak);np.testing.assert_allclose(flipped[1],sigma)
    data['scales'][0,2]=0.;assert not ray_profiles(data,rays)[2].any()
    print('ray profile/variance/axis/finite-patch-root self-check passed',flush=True)


def run_ray_profiles(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);summaries=[];inputs=[];ray_rows=[];replay_max=0.;total_rays=0;total_members=0
    for seed in (1000,2000):
        for curvature in (0,20,50):
            folder=root/f'independent_c{curvature}_s{seed}'
            meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            assert meta['status']=='complete' and meta['backend']=='analytic_ray_peak_gaussian_CPU'
            for case in range(6,12):
                item=meta['cases'][case];cfg=item['config'];path=folder/item['file']
                assert cfg['case_index']==case and cfg['seed']==20261022+seed+case and float(cfg.get('curvature',0.))==curvature
                assert len(cfg['theta_deg'])==2 and sha(path)==item['npz_sha256']
                inputs.append(dict(path=str(path),sha256=sha(path)))
                with np.load(path,allow_pickle=False) as z:data={name:z[name].copy() for name in z.files}
                assert data['image_size'].tolist()==[64,64] and len(data['pixel_xy'])==81 and not data['qualitative'].any()
                assert len(data['offsets'])==82 and data['offsets'][-1]==len(data['gaussian_id']) and np.all(np.diff(data['offsets'])>=0)
                focal=64/(2*np.tan(.04));xy=data['pixel_xy']
                rays=np.column_stack(((xy[:,0]-31.5)/focal,(xy[:,1]-31.5)/focal,np.ones(81)))
                peak,sigma,valid=ray_profiles(data,rays);ids=data['gaussian_id'];ray_index=np.repeat(np.arange(81),np.diff(data['offsets']))
                assert ((ids>=0)&(ids<len(data['means']))).all()
                np.testing.assert_array_equal(data['truth_origin'],data['source_origin'][ids])
                record_peak=peak[ray_index,ids];record_sigma=sigma[ray_index,ids]
                profile_valid=valid[ray_index,ids]&np.isfinite(data['depth'])
                np.testing.assert_allclose(record_peak[profile_valid],data['depth'][profile_valid],rtol=1e-10,atol=1e-10)
                error=float(np.max(np.abs(record_peak[profile_valid]-data['depth'][profile_valid]))) if profile_valid.any() else None
                if error is not None:replay_max=max(replay_max,error)
                roots=np.full((81,2),np.nan);root_counts=np.zeros((81,2),int);excluded=np.zeros((81,2),int)
                for ri,ray in enumerate(rays):
                    for origin in (0,1):
                        values,rejected=patch_intersections(ray,cfg,origin);root_counts[ri,origin]=len(values);excluded[ri,origin]=rejected
                        if len(values)==1:roots[ri,origin]=values[0]
                with np.errstate(invalid='ignore',divide='ignore'):
                    delta=(roots[ray_index]-record_peak[:,None])/record_sigma[:,None]
                delta[~profile_valid]=np.nan;relative=np.exp(-.5*delta**2)
                eligible=profile_valid&np.isfinite(delta).all(axis=1);source=data['source_origin'][ids]
                source_delta=delta[np.arange(len(ids)),source];other_delta=delta[np.arange(len(ids)),1-source]
                summary_config=dict(seed_offset=seed,case_index=case,curvature=curvature,center_noise=cfg['center_noise'],normal_noise_deg=cfg['normal_noise_deg'])
                for band in (1.,2.):
                    counts=np.full(len(ids),-1,int);counts[eligible]=(np.abs(delta[eligible])<=band).sum(axis=1)
                    for ri in range(81):
                        ix=slice(data['offsets'][ri],data['offsets'][ri+1]);local=counts[ix]
                        ray_rows.append(dict(**summary_config,ray=ri,sigma_band=band,members=len(local),valid_profiles=int(profile_valid[ix].sum()),
                            origin0_roots=int(root_counts[ri,0]),origin1_roots=int(root_counts[ri,1]),excluded_patch_roots=int(excluded[ri].sum()),
                            compatible_none=int((local==0).sum()),compatible_one=int((local==1).sum()),compatible_two=int((local==2).sum()),unknown_members=int((local<0).sum())))
                    known_source=np.isfinite(source_delta);known_other=np.isfinite(other_delta)
                    summaries.append(dict(**summary_config,sigma_band=band,rays=81,members=len(ids),valid_profiles=int(profile_valid.sum()),
                        both_unique_root_rays=int((root_counts==1).all(axis=1).sum()),missing_root_rays=int((root_counts==0).any(axis=1).sum()),
                        ambiguous_root_rays=int((root_counts>1).any(axis=1).sum()),excluded_patch_roots=int(excluded.sum()),
                        compatible_none=int((counts==0).sum()),compatible_one=int((counts==1).sum()),compatible_two=int((counts==2).sum()),unknown_members=int((counts<0).sum()),
                        source_delta_known=int(known_source.sum()),source_within_band=int((np.abs(source_delta[known_source])<=band).sum()),
                        other_delta_known=int(known_other.sum()),other_within_band=int((np.abs(other_delta[known_other])<=band).sum()),
                        mean_source_relative_peak=float(np.mean(np.exp(-.5*source_delta[known_source]**2))) if known_source.any() else None,
                        mean_other_relative_peak=float(np.mean(np.exp(-.5*other_delta[known_other]**2))) if known_other.any() else None,
                        mean_sigma_z=float(np.mean(record_sigma[profile_valid])) if profile_valid.any() else None,depth_replay_max_abs=error))
                    assert sum(int((counts==v).sum()) for v in (-1,0,1,2))==len(ids)
                np.savez_compressed(out/f'profile_s{seed}_c{curvature}_case{case}.npz',gaussian_id=ids,ray_index=ray_index,
                    peak=record_peak,sigma_z=record_sigma,profile_valid=profile_valid,source_origin=source,
                    delta=delta,relative_peak=relative,root_depths=roots,root_counts=root_counts,excluded_patch_roots=excluded)
                total_rays+=81;total_members+=len(ids);write_csv(out/'summary.csv',summaries);write_csv(out/'rays.csv',ray_rows)
                print('ray profile',seed,curvature,case,'members',len(ids),'finished',flush=True)
    assert len(inputs)==36 and total_rays==2916 and len(summaries)==72 and len(ray_rows)==5832
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),total_rays=total_rays,total_members=total_members,depth_replay_max_abs=replay_max,
        interpretation='Read-only existing analytic CPU Gaussian records; independent full precision matrix peak/variance and finite-patch nominal intersections. '
        'Camera-z ray parameter, FOV .08/image64/identity camera; roots retained only positive inside nominal [-.01,.01] x-y patch. '
        'Relative Gaussian ray-profile peak response exp(-delta^2/2) and fixed 1/2 sigma bands are geometric diagnostics, not physical support probabilities or rendering contribution. '
        'No w/alpha/T-based assignment, no impurity declaration, no threshold-defined layer or forced winner. '
        'All original accepted records retained; invalid profiles or nonunique/missing intersections remain unknown. Hidden/pruned members not inferred. '
        'Rays/members repeat Gaussians; measures pool accepted record counts, not independent scenes. Acceptance composition changes with curvature. '
        'Source origins and nominal graphs are construction controls, not trained-scene truth or host 2D rasterization; no fitting/generation/training.'),indent=2),encoding='utf-8')


def residual_components(mu,axes,rays,surface,peak,cfg,origins):
    kappa=float(cfg.get('curvature',0.));mu=np.asarray(mu,float);axes=np.asarray(axes,float);rays=np.asarray(rays,float)
    tangent=np.tan(np.deg2rad(np.asarray(cfg['theta_deg'])))[origins]
    g=np.column_stack((-tangent+2*kappa*mu[:,0],2*kappa*mu[:,1]))
    h=rays[:,2]-np.sum(g*rays[:,:2],axis=1);nd=np.sum(axes*rays,axis=1)
    nominal_height=1.+origins*cfg['gap']-tangent*mu[:,0]+kappa*np.sum(mu[:,:2]**2,axis=1)
    anchor_error=mu[:,2]-nominal_height
    valid=np.isfinite(mu).all(axis=1)&np.isfinite(axes).all(axis=1)&np.isfinite(rays).all(axis=1)&np.isfinite(surface)&np.isfinite(peak)&(np.abs(h)>1e-8)&(np.abs(nd)>1e-8)
    with np.errstate(invalid='ignore',divide='ignore'):
        t_nom=(mu[:,2]-np.sum(g*mu[:,:2],axis=1))/h
        t_cov=np.sum(axes*mu,axis=1)/nd
        a=(kappa*np.sum((surface[:,None]*rays[:,:2]-mu[:,:2])**2,axis=1)-anchor_error)/h
        components=np.column_stack((a,t_nom-t_cov,t_cov-peak));total=surface-peak;direct=surface-t_nom
    valid&=np.isfinite(components).all(axis=1)
    components[~valid]=np.nan;total[~valid]=np.nan
    return components,total,valid,direct


def decomposition_self_check():
    for kappa in (0.,50.):
        cfg=dict(theta_deg=[0.,0.],gap=.005,curvature=kappa)
        mu=np.array([[.005,0.,1.+kappa*.005**2+1e-6]])
        n=np.array([[-2*kappa*.005,0.,1.]]);n/=np.linalg.norm(n,axis=1)[:,None]
        q=np.column_stack((1+n[:,2],-n[:,1],n[:,0],np.zeros(1)));q/=np.linalg.norm(q,axis=1)[:,None]
        ray=np.array([[.001,0.,1.]])
        peak,_,_=ray_profiles(dict(means=mu,scales=np.array([[.006,.006,.0005]]),rotations=q),ray)
        surface=np.array(patch_intersections(ray[0],cfg,0)[0])
        parts,total,valid,direct=residual_components(mu,n,ray,surface,peak[:,0],cfg,np.array([0]))
        assert valid.all();np.testing.assert_allclose(parts.sum(axis=1),total,atol=1e-12);np.testing.assert_allclose(parts[:,0],direct,atol=1e-12)
        flipped=residual_components(mu,-n,ray,surface,peak[:,0],cfg,np.array([0]))
        np.testing.assert_allclose(flipped[0],parts,atol=1e-12)
    bad=residual_components(np.array([[0.,0.,1.]]),np.array([[0.,0.,1.]]),np.array([[1.,0.,0.]]),np.array([1.]),np.array([1.]),dict(theta_deg=[0.,0.],gap=.005),np.array([0]))
    assert not bad[2].any() and np.isnan(bad[0]).all()
    print('residual decomposition identity/anchor/axis/degeneracy self-check passed',flush=True)


def run_decomposition(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    profile_root=root/'ray_profile_compatibility';meta=json.loads((profile_root/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status']=='complete';input_sha={item['path']:item['sha256'] for item in meta['inputs']}
    out.mkdir(parents=True,exist_ok=True);summaries=[];inputs=[]
    for kappa in (0,20,50):
        folder=root/f'independent_c{kappa}_s1000';manifest=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        record=manifest['cases'][6];cfg=record['config'];path=folder/record['file']
        assert cfg['case_index']==6 and cfg['center_noise']==0 and cfg['normal_noise_deg']==0 and float(cfg.get('curvature',0.))==kappa
        assert sha(path)==record['npz_sha256']==input_sha[str(path)]
        saved=profile_root/f'profile_s1000_c{kappa}_case6.npz';inputs.append(dict(path=str(path),sha256=sha(path),profile_path=str(saved),profile_sha256=sha(saved)))
        with np.load(path,allow_pickle=False) as z:data={name:z[name].copy() for name in z.files}
        with np.load(saved,allow_pickle=False) as z:profile={name:z[name].copy() for name in z.files}
        ids=data['gaussian_id'];ri=np.repeat(np.arange(81),np.diff(data['offsets']))
        np.testing.assert_array_equal(ids,profile['gaussian_id']);np.testing.assert_array_equal(ri,profile['ray_index'])
        np.testing.assert_allclose(profile['peak'][profile['profile_valid']],data['depth'][profile['profile_valid']],rtol=1e-10,atol=1e-10)
        source=data['source_origin'][ids];np.testing.assert_array_equal(source,profile['source_origin'])
        xy=data['pixel_xy'];focal=64/(2*np.tan(.04));rays=np.column_stack(((xy[:,0]-31.5)/focal,(xy[:,1]-31.5)/focal,np.ones(81)))[ri]
        q=data['rotations'].astype(float);q/=np.linalg.norm(q,axis=1)[:,None];w,x,y,z=q.T
        axes=np.column_stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)))[ids]
        surface=profile['root_depths'][ri,source]
        components,total,valid,direct=residual_components(data['means'][ids],axes,rays,surface,profile['peak'],cfg,source)
        valid&=profile['profile_valid']&np.isfinite(profile['sigma_z'])&(profile['sigma_z']>0)
        a_error=float(np.max(np.abs(components[valid,0]-direct[valid]))) if valid.any() else None
        closure=float(np.max(np.abs(components[valid].sum(axis=1)-total[valid]))) if valid.any() else None
        if valid.any():np.testing.assert_allclose(components[valid,0],direct[valid],atol=1e-10,rtol=0);np.testing.assert_allclose(components[valid].sum(axis=1),total[valid],atol=1e-10,rtol=0)
        normalized=np.full_like(components,np.nan);normalized[valid]=components[valid]/profile['sigma_z'][valid,None]
        row=dict(case_index=6,seed_offset=1000,curvature=kappa,records=len(ids),valid_records=int(valid.sum()),unknown_records=int((~valid).sum()),analytic_first_term_max_abs_error=a_error,closure_max_abs_error=closure)
        for j,name in enumerate(('curvature_anchor','axis_representation','finite_tangent_precision')):
            for quantile in (.5,.95):row[name+('_median_abs_sigma' if quantile==.5 else '_p95_abs_sigma')]=float(np.quantile(np.abs(normalized[valid,j]),quantile)) if valid.any() else None
        row['total_median_abs_sigma']=float(np.median(np.abs(total[valid]/profile['sigma_z'][valid]))) if valid.any() else None
        row['total_p95_abs_sigma']=float(np.quantile(np.abs(total[valid]/profile['sigma_z'][valid]),.95)) if valid.any() else None
        summaries.append(row)
        np.savez_compressed(out/f'decomposition_c{kappa}.npz',gaussian_id=ids,ray_index=ri,source_origin=source,valid=valid,components=components,total=total,normalized_components=normalized,sigma_z=profile['sigma_z'])
        write_csv(out/'summary.csv',summaries);print('residual decomposition',kappa,'finished',flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',inputs=inputs,profile_metadata_sha256=sha(profile_root/'metadata.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        interpretation='Existing noiseless case6 seed1000 only; repeated identical second seed omitted. No fitting/generation/new bands. '
        'Known nominal curve is oracle. A=curvature Taylor remainder minus float32 center-anchor error; B=nominal vs stored covariance-axis tangent-plane intersection; C=finite tangential precision peak offset. '
        'Both first-term formula and A+B+C identity checked per valid accepted record. Grazing denominators or invalid records remain unknown. '
        'Camera-z units normalized by original ray-profile sigma; component magnitudes are not independent causal variance shares. '
        'This explains a controlled geometry discrepancy, not a new assignment/correction algorithm or physical support truth.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append');p.add_argument('--output');p.add_argument('--root',type=Path)
    p.add_argument('--ray-profile',action='store_true');p.add_argument('--profile-decomposition',action='store_true');p.add_argument('--self-check',action='store_true')
    p.add_argument('--member-measures',default='uniform',help='comma-separated uniform,tau,w')
    p.add_argument('--objective',choices=('soft_uniform','hard_min'),default='soft_uniform')
    args=p.parse_args();measures=tuple(args.member_measures.split(','))
    if args.ray_profile and args.profile_decomposition:p.error('choose a single mode')
    if args.self_check:
        if args.profile_decomposition:decomposition_self_check()
        elif args.ray_profile:profile_self_check()
        else:p.error('self-check requires a geometry diagnostic mode')
    elif args.profile_decomposition:
        if args.root is None or args.output is None:p.error('root and output required')
        run_decomposition(args.root,Path(args.output))
    elif args.ray_profile:
        if args.root is None or args.output is None:p.error('root and output required')
        run_ray_profiles(args.root,Path(args.output))
    else:
        if not args.input or args.output is None:p.error('input and output required')
        if not measures or len(set(measures))!=len(measures) or any(m not in ('uniform','tau','w') for m in measures):p.error('Invalid measures')
        run(args.input,Path(args.output),measures,args.objective)
