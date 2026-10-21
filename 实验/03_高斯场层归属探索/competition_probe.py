"""Fixed-K conditional mechanism/scene probes; does not estimate physical layer count."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from analyze import pair_metrics
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


def mechanism(out, repeats=20, seed=20261022):
    rows = []
    conditions = 0
    for sources in (1,2):
        gaps = (0.,) if sources == 1 else (0.,.005,.02,.05)
        directions = (0.,) if sources == 1 else (0.,10.,30.,60.)
        for gap, angle, dz, dn in itertools.product(gaps,directions,(0.,.002,.008),(0.,5.,15.,30.)):
            config = dict(sources=sources, depth_gap=gap, direction_gap_deg=angle,
                          depth_noise=dz, normal_noise_deg=dn, oracle_K=sources)
            for rep in range(repeats):
                rng=np.random.default_rng(seed+conditions*repeats+rep)
                d,n,truth=proposals(rng,sources,gap,angle,dz,dn,12)
                for lam in (0.,1.):
                    result=fit(d,n,np.ones(len(d),bool),sources,normal_weight=lam)
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
        ('sources','depth_gap','direction_gap_deg','depth_noise','normal_noise_deg','oracle_K','normal_weight')) for r in rows))
    for values in configs:
        fields=('sources','depth_gap','direction_gap_deg','depth_noise','normal_noise_deg','oracle_K','normal_weight')
        config=dict(zip(fields,values));part=[r for r in rows if all(r[k]==v for k,v in config.items())]
        summary.append(dict(**config,repeats=len(part),**{f'mean_{k}':float(np.mean([r[k] for r in part])) for k in keys}))
    write_csv(out/'summary.csv',summary)
    return dict(seed=seed,repeats=repeats,conditions=conditions,generated_cases=conditions*repeats,
                oracle_K=True,evidence='Known source count only; labels used exclusively for scoring. '
                'K=1 produces one group by construction and cannot demonstrate layer-number discovery. '
                'Same feature proposals/seeds as mechanism_sweep.py; not a Gaussian renderer.')


def scenes(inputs,out):
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
                            result=fit(d,n,v,K,normal_weight=lam)
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


def imbalance(out):
    """Fixed two prototypes; change only how often each is represented."""
    rows=[];details=[]
    for na,nb in ((12,12),(4,20),(2,22)):
        truth=np.r_[np.zeros(na,int),np.ones(nb,int)]
        d=1.+.005*truth
        theta=np.deg2rad(30.*truth)
        n=np.c_[np.sin(theta),np.zeros(len(truth)),-np.cos(theta)]
        for lam in (0.,1.):
            result=fit(d,n,np.ones(len(d),bool),2,normal_weight=lam)
            D=2.5*np.log1p(1./4)+lam
            mixed=np.log(2.)-np.logaddexp(0.,-D)
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
                evidence='Same unique geometric proposals, different Gaussian multiplicities. '
                'Equal-prior soft-mixture objective may duplicate the majority hypothesis despite known K. '
                'Collapse is abstention, not confident false membership. Exact analytic three-solution check.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mechanism',action='store_true');p.add_argument('--imbalance',action='store_true')
    p.add_argument('--input',action='append')
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=20)
    args=p.parse_args();out=Path(args.output)
    if sum(map(bool,(args.mechanism,args.imbalance,args.input)))!=1:p.error('Choose --mechanism, --imbalance, or --input')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    metadata=(mechanism(out,args.repeats) if args.mechanism else
              imbalance(out) if args.imbalance else scenes(args.input,out))
    root=Path(__file__).parent
    metadata.update(source_sha256={n:sha(root/n) for n in
        ('competition_probe.py','latent_competition.py','mechanism_sweep.py','analyze.py')})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
