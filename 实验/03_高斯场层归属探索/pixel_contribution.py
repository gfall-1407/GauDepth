"""Per-pixel Gaussian compositing weights from saved original-host accepted records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(weights):
    w=np.asarray(weights,float)
    if not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Accepted weights must be finite and positive; do not silently discard invalid records')
    total=float(w.sum());ranked=np.sort(w)[::-1];cumulative=np.cumsum(ranked)
    result=dict(accepted_gaussians=len(w),weight_sum=total,
                min_weight=float(w.min()) if len(w) else None,max_weight=float(w.max()) if len(w) else None)
    for k in (1,3,5,10,20,50,100):
        result[f'top{k}_fraction']=float(ranked[:k].sum()/total) if total else None
    for percent in (50,90,95,99):
        result[f'n{percent}']=int(np.searchsorted(cumulative,percent/100*total-1e-12*total)+1) if total else None
    return result


def self_check():
    assert describe([])['min_weight'] is None and describe([])['accepted_gaussians']==0
    assert describe([.9,.1])['n90']==1
    uniform=describe(np.ones(100)*.01)
    assert uniform['n90']==90 and np.isclose(uniform['top10_fraction'],.1)
    for w in ([.1,.7,.2],[.2,.1,.7]):
        assert describe(w)['n90']==2 and np.isclose(describe(w)['top1_fraction'],.7)
    try:describe([0.])
    except ValueError:pass
    else:raise AssertionError('Invalid accepted weight not rejected')
    print('pixel contribution empty/dominant/uniform/permutation/invalid self-check passed',flush=True)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows):
    result=dict(pixels=len(rows),empty_pixels=sum(r['accepted_gaussians']==0 for r in rows),metrics={})
    for key in ('accepted_gaussians','min_weight','max_weight','weight_sum','background_weight',
                'top1_fraction','top3_fraction','top5_fraction','top10_fraction','top20_fraction','top50_fraction','top100_fraction','n50','n90','n95','n99'):
        values=np.array([r[key] for r in rows if r[key] is not None],float)
        result['metrics'][key]=dict(valid_pixels=len(values),mean=float(values.mean()) if len(values) else None,
            **{name:float(np.quantile(values,q)) if len(values) else None for name,q in (('min',0.),('p05',.05),('median',.5),('p95',.95),('max',1.))})
    return result


def run(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];max_replay_error=0.;skipped_qualitative=0
    for scene in ('scan24','scan105','scan65','scan110'):
        folders=[scene,scene+'_more'] if scene in ('scan24','scan105') else [scene]
        seen=set();model_sha=None
        for folder in folders:
            directory=root/folder;meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            if model_sha is None:model_sha=meta['model_sha256']
            assert meta['model_sha256']==model_sha
            inputs.append(dict(path=str(directory/'metadata.json'),sha256=sha(directory/'metadata.json')))
            for view in meta['views']:
                assert view['view_index'] not in seen;seen.add(view['view_index'])
                path=directory/view['file'];assert sha(path)==view['npz_sha256']
                inputs.append(dict(path=str(path),sha256=sha(path),scene=scene,view=view['view_index'],sampling=view['sampling']))
                with np.load(path,allow_pickle=False) as z:
                    offsets=z['offsets'];weight=z['weight'];alpha=z['alpha'];transmittance=z['transmittance']
                    assert offsets[0]==0 and offsets[-1]==len(weight) and np.all(np.diff(offsets)>=0)
                    assert len(weight)==len(alpha)==len(transmittance)
                    np.testing.assert_allclose(weight.astype(float),alpha.astype(float)*transmittance.astype(float),atol=2e-6,rtol=2e-5)
                    for ray in range(len(offsets)-1):
                        if z['qualitative'][ray]:skipped_qualitative+=1;continue
                        start,end=map(int,offsets[ray:ray+2]);w=weight[start:end]
                        total=float(w.astype(float).sum());opacity=float(z['host_target'][ray,2])
                        np.testing.assert_allclose(total,opacity,atol=2e-6,rtol=2e-5)
                        max_replay_error=max(max_replay_error,abs(total-opacity))
                        row=dict(scene=scene,view=view['view_index'],ray=ray,x=int(z['pixel_xy'][ray,0]),y=int(z['pixel_xy'][ray,1]),
                                 evaluated_gaussians=None,tile_candidates=None,host_opacity=opacity,background_weight=1.-opacity,**describe(w))
                        rows.append(row)
        assert len(seen)==3
    assert len(rows)==3072 and sum(r['accepted_gaussians'] for r in rows)==258345
    write_csv(out/'pixels.csv',rows)
    groups={'all_sampled_pixels':rows}
    for scene in ('scan24','scan105','scan65','scan110'):
        groups[scene]=[r for r in rows if r['scene']==scene]
        for view in sorted({r['view'] for r in groups[scene]}):
            groups[f'{scene}/view{view}']=[r for r in groups[scene] if r['view']==view]
    summary={name:summarize(subset) for name,subset in groups.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    long=[]
    for scope,report in summary.items():
        for metric,values in report['metrics'].items():long.append(dict(scope=scope,metric=metric,**values))
    write_csv(out/'summary.csv',long)
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),inputs=inputs,max_weight_sum_opacity_error=max_replay_error,skipped_qualitative=skipped_qualitative,
        evidence='Original-host 4 scenes, 3 views each, 256 mask-sampled pixels per view, not all image pixels. Equal pixel weighting. '
        'accepted_gaussians counts records entering color compositing; attempted/evaluated/rejected counts unavailable and remain null. '
        'w=T*alpha is the scalar coefficient in sum(w_i*c_i)+(1-opacity)*background, not RGB magnitude or surface support. '
        'Min/max are absolute positive accepted w; top-k and n50/90/95/99 normalize by sum accepted w, excluding background. '
        'nXX ranks individual Gaussians by descending w, not depth, acceptance order or candidate groups; numerical cutoff tolerance1e-12 relative. '
        'Empty pixels have accepted0 and null min/max/concentration; no accepted member removed. '
        'Saved NPZ hashes and weight/alpha/T/opacity replay checked. No rerendering or training.'),indent=2),encoding='utf-8')
    print(json.dumps(summary['all_sampled_pixels'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');args=p.parse_args()
    if args.self_check:self_check()
    else:
        if args.root is None or args.output is None:p.error('root/output required')
        run(args.root,args.output)
