"""Per-pixel Gaussian compositing weights from saved original-host accepted records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

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
    np.testing.assert_array_equal(retained_mask(np.array([[0.,.499,.5,1.]]),(1,4)),[[False,False,True,True]])
    empty_summary=column_summary(dict(accepted=np.array([0]),min_weight=np.array([np.nan])))
    assert empty_summary['accepted']['mean']==0 and empty_summary['min_weight']['valid_pixels']==0
    assert empty_summary['min_weight']['median'] is None
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


def retained_mask(mask,shape):
    mask=np.asarray(mask)
    if mask.shape==(1,*shape):mask=mask[0]
    if mask.shape!=shape or not np.isfinite(mask).all():raise ValueError('Missing, nonfinite or incorrectly sized GT mask')
    return mask>=.5 # Exact boundary used by mesh_extract.py: discard mask < .5.


def column_summary(columns):
    result={}
    for name,column in columns.items():
        values=np.asarray(column,float);values=values[np.isfinite(values)]
        result[name]=dict(valid_pixels=len(values),mean=float(values.mean()) if len(values) else None,
            **{stat:float(np.quantile(values,q)) if len(values) else None for stat,q in (('min',0.),('p05',.05),('median',.5),('p95',.95),('max',1.))})
    return result


def concentration_torch(weights,counts):
    import torch
    w=weights.double();valid=torch.arange(w.shape[1],device=w.device)[None,:]<counts[:,None]
    w=torch.where(valid,w,0.)
    assert bool(torch.isfinite(w).all()) and bool((w[valid]>0).all())
    ranked=torch.sort(w,dim=1,descending=True).values;total=ranked.sum(dim=1);cumulative=ranked.cumsum(dim=1)
    columns=[]
    for k in (1,3,5,10,20,50,100):columns.append(torch.where(total>0,ranked[:,:k].sum(dim=1)/total,float('nan')))
    for percent in (50,90,95,99):
        n=(cumulative<(percent/100*total-1e-12*total)[:,None]).sum(dim=1)+1
        columns.append(torch.where(total>0,n.double(),float('nan')))
    return torch.stack(columns,dim=1),total,w


def prepare_reader(out):
    """Compile only, plus CPU tensor arithmetic checks; never claims a CUDA kernel test."""
    import os
    import torch
    from torch.utils.cpp_extension import load
    here=Path(__file__).resolve().parent;build=here/'构建_pixel_stats';build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
    examples=np.full((4,100),np.nan,np.float32);lengths=np.array([2,100,0,3],np.int32)
    examples[0,:2]=[.9,.1];examples[1]=.01;examples[3,:3]=[.1,.7,.2]
    got,_,_=concentration_torch(torch.from_numpy(examples),torch.from_numpy(lengths))
    names=[f'top{k}_fraction' for k in (1,3,5,10,20,50,100)]+[f'n{p}' for p in (50,90,95,99)]
    expected=np.array([[describe(row[:n])[key] for key in names] for row,n in zip(examples,lengths)],float)
    np.testing.assert_allclose(got.numpy(),expected,rtol=1e-10,atol=1e-10,equal_nan=True)
    print('CPU torch concentration agrees with independent NumPy cases; compiling reader for sm89',flush=True)
    extension=load(name='gaudepth_pixel_stats_reader_v2',sources=[str(here/'reader.cu')],build_directory=str(build),
                   extra_cuda_cflags=['-O3','--use_fast_math'],verbose=True)
    out.mkdir(parents=True,exist_ok=True)
    payload=dict(status='compiled_cpu_arithmetic_checked',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 binary=extension.__file__,binary_sha256=sha(extension.__file__),reader_source_sha256=sha(here/'reader.cu'),
                 torch=torch.__version__,target_arch='8.9',gpu_available=torch.cuda.is_available(),cuda_kernel_executed=False)
    (out/'prepare.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)


def masked_frame(extension,geom,binning,image,P,W,H,fx,fy,expected,median,alpha,mask,output,view_index):
    """Every retained pixel, exact ranked concentration in bounded chunks; no raw-member archive."""
    import os
    import torch
    keep=retained_mask(mask,(H,W));pixels=np.flatnonzero(keep.ravel()).astype(np.int32)
    dense_counts,dense_weights=extension.read_pixel_stats(geom,binning,image,P,W,H)
    torch.testing.assert_close(dense_weights[:,:,2],alpha.reshape(H,W),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(1.-dense_weights[:,:,3],alpha.reshape(H,W),rtol=2e-5,atol=2e-6)
    assert torch.equal(dense_counts[:,:,6],dense_counts[:,:,7]),'Host last-position mismatch'
    assert torch.equal(dense_counts[:,:,1],dense_counts[:,:,2:6].sum(dim=2).to(torch.int32)),'Stage count mismatch'
    assert bool((dense_counts[:,:,0]>=dense_counts[:,:,1]).all())
    pixel_gpu=torch.from_numpy(pixels).to(device=geom.device).long()
    counts=dense_counts.reshape(-1,8)[pixel_gpu].cpu().numpy()
    weights=dense_weights.reshape(-1,4)[pixel_gpu].cpu().numpy()
    del dense_counts,dense_weights
    # GPU numerical-path check against independently defined CPU examples before scanning a frame.
    example=np.array([[.9,.1,0.],[.1,.1,.1],[0.,0.,0.]],np.float32);lengths=np.array([2,3,0],np.int32)
    got,_,_=concentration_torch(torch.from_numpy(example).to(geom.device),torch.from_numpy(lengths).to(geom.device))
    concentration_names=[f'top{k}_fraction' for k in (1,3,5,10,20,50,100)]+[f'n{p}' for p in (50,90,95,99)]
    reference=np.array([[describe(row[:length])[name] for name in concentration_names] for row,length in zip(example,lengths)],dtype=float)
    np.testing.assert_allclose(got.cpu().numpy(),reference,atol=1e-10,rtol=1e-10,equal_nan=True)
    concentration=np.empty((len(pixels),11),np.float32)
    maximum_replay_error=np.zeros(3,float)
    for start in range(0,len(pixels),512):
        stop=min(start+512,len(pixels));local=pixel_gpu[start:stop].to(torch.int32)
        capacity=max(1,int(counts[start:stop,5].max()))
        rec,gids,number,replay=extension.read_rays(geom,binning,image,P,W,H,fx,fy,local,capacity)
        np.testing.assert_array_equal(number.cpu().numpy(),counts[start:stop,5])
        target=torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],dim=1)[local.long()]
        torch.testing.assert_close(replay,target,rtol=2e-5,atol=2e-6)
        maximum_replay_error=np.maximum(maximum_replay_error,(replay-target).abs().amax(dim=0).cpu().numpy())
        values,total,w=concentration_torch(rec[:,:,1],number)
        torch.testing.assert_close(total,alpha.flatten()[local.long()].double(),rtol=2e-5,atol=2e-6)
        wmax=w.max(dim=1).values;wmin=torch.where(torch.arange(capacity,device=w.device)[None,:]<number[:,None],w,float('inf')).min(dim=1).values
        wmax=torch.where(number>0,wmax,float('nan'));wmin=torch.where(number>0,wmin,float('nan'))
        np.testing.assert_allclose(wmin.cpu().numpy(),weights[start:stop,0],atol=1e-10,rtol=2e-5,equal_nan=True)
        np.testing.assert_allclose(wmax.cpu().numpy(),weights[start:stop,1],atol=1e-10,rtol=2e-5,equal_nan=True)
        concentration[start:stop]=values.cpu().numpy().astype(np.float32)
        del rec,gids,number,replay,values,total,w
    count_names=['tile_candidates','evaluated','power_rejected','alpha_rejected','early_stop_triggered','accepted','last_accepted_position','host_last_accepted_position']
    weight_names=['min_weight','max_weight','weight_sum','background_weight']
    columns={name:counts[:,i] for i,name in enumerate(count_names)}
    columns.update({name:weights[:,i] for i,name in enumerate(weight_names)})
    columns.update({name:concentration[:,i] for i,name in enumerate(concentration_names)})
    summary=dict(retained_pixels=len(pixels),image_pixels=H*W,masked_out_pixels=H*W-len(pixels),
                 empty_contribution_pixels=int((counts[:,5]==0).sum()),metrics=column_summary(columns))
    name=f'masked_pixel_stats_{view_index:03d}.npz';temporary=output/(name+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,pixel_index=pixels,counts=counts,weights=weights,concentration=concentration,image_size=np.array([H,W]))
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,output/name)
    return dict(file=name,npz_sha256=sha(output/name),count_fields=count_names,weight_fields=weight_names,
                concentration_fields=concentration_names,mask_threshold='keep >= 0.5, identical to mesh_extract.py',
                retained_mask_sha256=hashlib.sha256(np.packbits(keep.ravel()).tobytes()).hexdigest(),
                summary=summary,replay_max_abs=maximum_replay_error.tolist(),
                evidence='Every GT-mask retained pixel; empty-contribution pixels remain. Not opacity-filtered. '
                'Exact ranked weights computed in float64 chunks, saved as float32; nXX stores integer counts or NaN. '
                'Only retained-pixel aggregates archived, no RGB/depth/full Gaussian lists. '
                'All retained rays expected/median/opacity, extrema and accepted counts replayed; full-frame opacity/last-position/stage accounting checked.')


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


def run_masked_scans(out,provenance):
    import socket
    from sample import atomic_json
    if socket.gethostname()!='autodl-container-d6a7459179-693ab6ee':raise ValueError('Only authorized G1')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent;repo=here.parent.parent
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    batch=dict(status='running',git_commit=commit,stages=[],scans={},started_at=time.time())
    atomic_json(out/'batch.json',batch)

    def stage(scene,label,views,seconds):
        if (out/'stop_after_stage').exists():raise RuntimeError('Controller requested no new stage')
        target=out/(scene+'_'+label)
        command=[sys.executable,str(here/'sample.py'),'-m',f'/root/autodl-tmp/GauDepth/outtest/{scene}',
                 '-s',f'/root/autodl-tmp/DTU/{scene}','--iteration','30000','--split','all','--views',views,
                 '--mask-statistics','--host-build-provenance',provenance,'--output',str(target)]
        record=dict(scene=scene,label=label,command=command,maximum_seconds=seconds,started_at=time.time())
        batch['stages'].append(record);atomic_json(out/'batch.json',batch)
        with (out/(scene+'_'+label+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                record.update(status='timeout',exit_code=child.returncode,finished_at=time.time());atomic_json(out/'batch.json',batch)
                raise RuntimeError(f'{scene}/{label} timeout')
        record.update(status='complete' if code==0 else 'failed',exit_code=code,finished_at=time.time());atomic_json(out/'batch.json',batch)
        if code:raise RuntimeError(f'{scene}/{label} failed: inspect preserved log')
        meta=json.loads((target/'metadata.json').read_text(encoding='utf-8'))
        assert meta['status']=='complete' and meta['original_model_unchanged'] and meta['git_commit']==commit
        print('stage complete',scene,label,len(meta['views']),'views',flush=True)
        return meta

    try:
        for scene in ('scan24','scan105','scan65','scan110'):
            parts=[]
            if scene=='scan24':
                pilot=stage(scene,'pilot','0',600);parts.append(pilot)
                remaining=[str(c['view_index']) for c in pilot['camera_inventory'] if c['view_index']!=0]
                if remaining:parts.append(stage(scene,'remaining',','.join(remaining),1800))
            else:parts.append(stage(scene,'all','all',1800))
            names=[v['image_name'] for part in parts for v in part['views']]
            expected=[c['image_name'] for c in parts[0]['camera_inventory']]
            assert len(names)==len(set(names)) and set(names)==set(expected),'Camera coverage incomplete or duplicate'
            assert len({part['model_sha256'] for part in parts})==1
            metrics={}
            for part in parts:
                for name,value in part['masked_summary']['metrics'].items():
                    acc=metrics.setdefault(name,dict(valid_pixels=0,total=0.))
                    acc['valid_pixels']+=value['valid_pixels'];acc['total']+=(value['pixel_weighted_mean'] or 0.)*value['valid_pixels']
            batch['scans'][scene]=dict(views=len(names),expected_views=len(expected),image_names=names,model_sha256=parts[0]['model_sha256'],
                retained_pixels=sum(p['masked_summary']['retained_pixels'] for p in parts),
                empty_contribution_pixels=sum(p['masked_summary']['empty_contribution_pixels'] for p in parts),
                metrics={k:dict(valid_pixels=v['valid_pixels'],pixel_weighted_mean=v['total']/v['valid_pixels'] if v['valid_pixels'] else None) for k,v in metrics.items()})
            atomic_json(out/'batch.json',batch)
            print('scan complete',scene,len(names),'views',flush=True)
        batch.update(status='complete',finished_at=time.time());atomic_json(out/'batch.json',batch)
    except BaseException as exc:
        batch.update(status='failed',error=str(exc),finished_at=time.time());atomic_json(out/'batch.json',batch);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true')
    p.add_argument('--run-masked-scans',action='store_true');p.add_argument('--prepare-reader',action='store_true');p.add_argument('--host-build-provenance');args=p.parse_args()
    if args.self_check:self_check()
    elif args.prepare_reader:
        if args.output is None:p.error('output required')
        prepare_reader(args.output)
    elif args.run_masked_scans:
        if args.output is None or not args.host_build_provenance:p.error('output/provenance required')
        run_masked_scans(args.output,args.host_build_provenance)
    else:
        if args.root is None or args.output is None:p.error('root/output required')
        run(args.root,args.output)
