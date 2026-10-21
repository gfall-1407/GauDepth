"""Score construction-origin recovery on the accepted synthetic renderer members."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

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


def run(inputs,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    rows=[];manifests=[];seen=set()
    for folder in map(Path,inputs):
        meta_path=folder/'metadata.json';meta=json.loads(meta_path.read_text(encoding='utf-8'))
        assert meta['status']=='complete'
        for item in meta['cases']:
            config=item['config'];case=int(config['case_index'])
            assert case not in seen,f'Duplicate case {case}'
            seen.add(case);path=folder/item['file'];assert sha(path)==item['npz_sha256']
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
                    for lam in (0.,1.):
                        result=fit(d,n,v,K,normal_weight=lam) if K else None
                        labels_by_method['conditional_normal' if lam else 'conditional_depth']=(
                            np.asarray(result['labels']) if result else np.full(len(d),-1),
                            result['mean_entropy'] if result else None,
                            result['collapsed'] if result else None)
                    total=float(w.astype(float).sum())
                    origin_mass=[float(w[truth==k].astype(float).sum()) for k in range(len(config['theta_deg']))]
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
                            groups=len(set(labels.tolist())-{-1}),unknown_members=int((labels<0).sum()),
                            mean_entropy=entropy,collapsed=collapsed))
            manifests.append(dict(file=str(path),sha256=sha(path),metadata_sha256=sha(meta_path),
                                  runtime_codeHEAD=meta['codeHEAD']))
            print('origin scores case',case,flush=True)
    out.mkdir(parents=True,exist_ok=True);write_csv(out/'rays.csv',rows)
    summaries=[]
    metrics=('pair_f1','original_pair_mass_f1','false_merge_pairs','false_split_pairs','groups',
             'unknown_members','both_origins_accepted','min_original_origin_mass_fraction',
             'source0_pair_recall','source1_pair_recall','mean_entropy','collapsed')
    for case in sorted(seen):
        for method in ('depth','hard_normal','conditional_depth','conditional_normal'):
            part=[r for r in rows if r['case_index']==case and r['method']==method]
            config={k:part[0][k] for k in ('case_index','kind','gap','center_noise','normal_noise_deg')}
            summaries.append(dict(**config,method=method,rays=len(part),
                **{f'mean_{key}':float(np.mean([r[key] for r in part if r[key] is not None]))
                    if any(r[key] is not None for r in part) else None for key in metrics}))
    write_csv(out/'summary.csv',summaries)
    root=Path(__file__).parent
    (out/'metadata.json').write_text(json.dumps(dict(inputs=manifests,
        source_sha256={n:sha(root/n) for n in ('renderer_origin_analyze.py','latent_competition.py','analyze.py')},
        interpretation='Construction origins of accepted members only. Hidden/pruned members are not assigned background. '
        'Conditional methods receive the oracle number of accepted origins, other methods do not; not an equal-information comparison. '
        'Single-origin conditional K=1 success is tautological. Perturbed origins are not physical truth in trained scenes. '
        'Original pair-mass F1 is supplementary; unweighted and per-origin recall are retained.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.input,Path(args.output))
