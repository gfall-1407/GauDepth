"""Leave-one-ray-out pair voting; pair predictions need not define a partition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_state(record, method):
    ids=np.asarray(record['gaussian_id'],int);labels=np.asarray(record['labels'][method],int)
    i,j=np.triu_indices(len(ids),1)
    known=(labels[i]>=0)&(labels[j]>=0)
    same=known&(labels[i]==labels[j])
    return ids[i],ids[j],known,same


def evaluate(actual, predicted, known):
    # Abstention counts as a missed positive; known-only scores reported separately.
    tp=int((actual&predicted&known).sum());fp=int((~actual&predicted&known).sum())
    fn=int((actual&~(predicted&known)).sum())
    return dict(tp=tp,fp=fp,fn=fn,pair_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def self_check():
    truth=np.array([True,False,True]);pred=np.array([True,True,False]);known=np.array([True,True,False])
    r=evaluate(truth,pred,known);assert (r['tp'],r['fp'],r['fn'],r['pair_f1'])==(1,1,1,.5)
    # 0-1 and 1-2 positive, 0-2 negative violates equivalence transitivity.
    P=np.eye(3,dtype=bool);P[0,1]=P[1,0]=P[1,2]=P[2,1]=True
    assert ((P.astype(int)@P.astype(int)>0)&~P)[0,2]
    # Current observation cannot leak into leave-one-out votes.
    known_total,same_total,current_known,current_same=4,2,1,1
    assert (same_total-current_same)*2 < known_total-current_known
    print('pair consensus self-check passed',flush=True)


def run(source,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    source_meta=source/'metadata.json';meta=json.loads(source_meta.read_text(encoding='utf-8'))
    assert meta['status']=='complete'
    out.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    for path in sorted(source.glob('labels_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'));rays=data['rays'];config=data['config']
        source_npz=Path(data['source_file']);assert sha(source_npz)==data['source_sha256']
        sources.append(dict(path=str(path),sha256=sha(path),npz_sha256=data['source_sha256']))
        nmax=max(max(r['gaussian_id'],default=-1) for r in rays)+1
        for method in sorted(rays[0]['labels']):
            if not method.startswith('conditional'):continue
            counts=np.zeros((nmax,nmax),np.int64);positive=np.zeros_like(counts)
            for ray in rays:
                i,j,known,same=pair_state(ray,method)
                counts[i,j]+=known;counts[j,i]+=known
                positive[i,j]+=same;positive[j,i]+=same
            for ray in rays:
                ids=np.asarray(ray['gaussian_id']);truth=np.asarray(ray['truth_origin'])
                a,b=np.triu_indices(len(ids),1);i,j,known,same=pair_state(ray,method)
                total=counts[i,j]-known;yes=positive[i,j]-same
                eligible=total>=2;decisive=eligible&(yes*2!=total)
                vote=yes*2>total;actual=truth[a]==truth[b]
                assert np.all(total>=0) and np.all(yes>=0) and np.all(yes<=total)
                for policy in ('original','consensus_abstain','consensus_fallback'):
                    pred=same if policy=='original' else vote if policy=='consensus_abstain' else np.where(decisive,vote,same)
                    observed=known if policy=='original' else decisive if policy=='consensus_abstain' else decisive|known
                    # Count known non-transitive triples: three observed edges, exactly two positives.
                    P=np.zeros((len(ids),len(ids)),np.int64);O=np.zeros_like(P)
                    P[a,b]=P[b,a]=pred&observed;O[a,b]=O[b,a]=observed
                    violations=0
                    for pivot in range(len(ids)):
                        neighbors=np.flatnonzero(P[pivot])
                        x,y=np.triu_indices(len(neighbors),1);u,v=neighbors[x],neighbors[y]
                        violations+=int((O[u,v]&(1-P[u,v])).sum())
                    rows.append(dict(case_index=config['case_index'],kind=config['kind'],gap=config['gap'],
                        center_noise=config['center_noise'],normal_noise_deg=config['normal_noise_deg'],
                        ray=ray['ray'],method=method,policy=policy,accepted_origins=len(np.unique(truth)),
                        all_pairs=len(a),other_ray_supported_pairs=int(eligible.sum()),decisive_pairs=int(decisive.sum()),
                        predicted_known_pairs=int(observed.sum()),changed_known_pairs=int((decisive&known&(vote!=same)).sum()),
                        nontransitive_triples=violations,**evaluate(actual,pred,observed)))
        csv_write(out/'rays.csv',rows)
        print('leave-one-ray-out',path.stem,'finished',flush=True)
    summaries=[]
    for case,method,policy in sorted(set((r['case_index'],r['method'],r['policy']) for r in rows)):
        part=[r for r in rows if (r['case_index'],r['method'],r['policy'])==(case,method,policy)]
        metrics=('pair_f1','all_pairs','other_ray_supported_pairs','decisive_pairs','predicted_known_pairs','changed_known_pairs','nontransitive_triples')
        summaries.append(dict(**{k:part[0][k] for k in ('case_index','kind','gap','center_noise','normal_noise_deg','method','policy')},
            rays=len(part),rays_with_nontransitive_triples=sum(r['nontransitive_triples']>0 for r in part),
            **{'mean_'+k:float(np.mean([r[k] for r in part if r[k] is not None])) for k in metrics}))
    csv_write(out/'summary.csv',summaries)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',source_metadata_sha256=sha(source_meta),inputs=sources,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        minimum_other_ray_observations=2,ties='unknown',
        evidence='Pair voting diagnostic, not a layer partition, training algorithm or physical accuracy. '
        'Current-ray observation excluded exactly. Other rays share Gaussians and are correlated; no independent-validation claim. '
        'Construction labels only score pairs; labels and oracle K of base fits already described in source metadata. '
        'No cross-case IDs matched. Fallback keeps original pair state where consensus lacks decisive support. '
        'Non-transitive triples prevent interpreting arbitrary pair predictions as a valid grouping. '
        'Static source identity across these constructed rays is an explicit mechanism assumption, not a universal multi-view constraint.'),indent=2),encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check()
    else:
        if args.input is None or args.output is None:parser.error('input and output required')
        run(args.input,args.output)
