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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.root,args.output)
