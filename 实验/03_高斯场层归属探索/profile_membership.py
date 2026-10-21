"""Common Gaussian ray-response bands, not confidence intervals or physical labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze import pair_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ray_sigma(data,ray):
    q=data['rotations'].astype(float);q/=np.linalg.norm(q,axis=1)[:,None]
    w,x,y,z=q.T
    axis=np.c_[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]
    scale=data['scales'].astype(float)
    assert np.allclose(scale[:,0],scale[:,1])
    a=1/scale[:,0]**2;b=1/scale[:,2]**2-a
    return 1/np.sqrt(a*np.dot(ray,ray)+b*(axis@ray)**2)


def organize(depth,sigma,c):
    depth,sigma=np.asarray(depth,float),np.asarray(sigma,float)
    good=np.isfinite(depth)&(depth>0)&np.isfinite(sigma)&(sigma>0)
    low,high=depth-c*sigma,depth+c*sigma
    remaining=np.flatnonzero(good);labels=np.full(len(depth),-1,int);anchors=[];witness=[]
    while len(remaining):
        first=int(remaining[np.argmin(high[remaining])]);point=high[first]
        cover=(low[remaining]<=point)&(high[remaining]>=point)
        labels[remaining[cover]]=len(anchors);anchors.append(float(point));witness.append(first)
        remaining=remaining[~cover]
    support=(low[:,None]<=np.asarray(anchors))&(high[:,None]>=np.asarray(anchors))&good[:,None]
    assert np.all(support[good].any(axis=1))
    for i,k in enumerate(witness):
        if i:assert low[k]>high[witness[i-1]]
    unique=labels.copy();unique[support.sum(axis=1)>1]=-1
    return labels,unique,anchors,support


def self_check():
    lab,unique,anchors,support=organize([1.,1.01,1.02],[.03,.001,.001],1.)
    assert len(anchors)==2 and unique[0]==-1 and support[0].sum()==2
    assert len(organize([1.,1.0001],[.001,.001],2.)[2])==1
    assert len(organize([1.,1.02],[.001,.001],2.)[2])==2
    assert len(organize([],[],2.)[2])==0
    shifted=organize(np.array([1.,1.01,1.02])*1000,np.array([.03,.001,.001])*1000,1.)
    assert np.array_equal(shifted[0],lab) and np.array_equal(shifted[1],unique)
    print('common response band self-check passed')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(folder,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert meta['backend']=='analytic_ray_peak_gaussian_CPU' and meta['status']=='complete'
    rows=[];witnesses=[]
    focal=64/(2*np.tan(.04))
    for case in meta['cases']:
        path=folder/case['file'];assert sha(path)==case['npz_sha256'];cfg=case['config']
        with np.load(path,allow_pickle=False) as z:
            for ri in range(len(z['offsets'])-1):
                a,b=map(int,z['offsets'][ri:ri+2]);ids=z['gaussian_id'][a:b]
                xy=z['pixel_xy'][ri];ray=np.r_[(xy-31.5)/focal,1.]
                sigma=ray_sigma(z,ray)[ids];depth=z['depth'][a:b];truth=z['truth_origin'][a:b]
                # Oracle diagnostic only; labels do NOT set the response-band multiplier.
                pair_scale=np.abs(depth[:,None]-depth[None,:])/(sigma[:,None]+sigma[None,:])
                same=truth[:,None]==truth[None,:]
                within=float(pair_scale[same].max()) if len(depth) else 0.
                union=float(pair_scale.max()) if len(depth) else 0.
                for c in (1.,2.,3.):
                    lab,unique,anchors,support=organize(depth,sigma,c)
                    for name,labels in [('minimum_band_partition',lab),('unique_band_support',unique)]:
                        score=pair_metrics(truth,labels)
                        rows.append(dict(case_index=cfg['case_index'],kind=cfg['kind'],gap=cfg['gap'],
                            center_noise=cfg['center_noise'],normal_noise_deg=cfg['normal_noise_deg'],
                            ray=ri,c=c,method=name,anchors=len(anchors),members=len(depth),
                            critical_c_within_origin=within,critical_c_all_members=union,
                            pair_f1=score['f1'],false_merge_pairs=score['fp'],false_split_pairs=score['fn'],
                            unknown_members=int((labels<0).sum()),multiple_support_members=int((support.sum(axis=1)>1).sum())))
                    witnesses.append(dict(case_index=cfg['case_index'],ray=ri,c=c,anchors=anchors,
                        gaussian_ids=ids.tolist(),response_sigma=sigma.tolist(),partition=lab.tolist(),
                        support_hypothesis_indices=[np.flatnonzero(s).tolist() for s in support]))
    out.mkdir(parents=True,exist_ok=True);write_csv(out/'rays.csv',rows)
    summaries=[]
    for key in sorted(set((r['case_index'],r['c'],r['method']) for r in rows)):
        part=[r for r in rows if (r['case_index'],r['c'],r['method'])==key]
        config={k:part[0][k] for k in ('case_index','kind','gap','center_noise','normal_noise_deg','c','method')}
        summaries.append(dict(**config,rays=len(part),**{f'mean_{k}':float(np.mean([r[k] for r in part]))
            for k in ('anchors','pair_f1','false_merge_pairs','false_split_pairs','unknown_members',
                      'multiple_support_members','critical_c_within_origin','critical_c_all_members')}))
    write_csv(out/'summary.csv',summaries)
    (out/'witnesses.json').write_text(json.dumps(witnesses),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(input_metadata_sha256=sha(folder/'metadata.json'),
        input_sha256={v['file']:v['npz_sha256'] for v in meta['cases']},
        source_sha256=sha(__file__),sigma='1/sqrt(ray^T covariance^-1 ray)',
        band='peak_z +/- c*sigma, equivalent relative profile response >= exp(-c^2/2)',
        interpretation='Response width is geometric extent, NOT measurement uncertainty. Minimum interval piercing '
        'is a minimum common-band hypothesis count, NOT physical layer count. Ambiguous memberships remain unknown '
        'in unique-support output; all members retained. Only analytic synthetic origin labels available.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input');p.add_argument('--output');p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:self_check()
    elif args.input and args.output:run(Path(args.input),Path(args.output))
    else:p.error('Specify --input and --output')
