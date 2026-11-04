"""Bounded support/index/gradient checks and batched-core timing; no training."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import torch
from scene.gaussian_model import GaussianModel
from joint_tv_inputs import MODEL_IDENTITIES,verify_inputs,sha256
from joint_tv_convergence import gradient_delta
from joint_tv_model import (FIELD_PROTOCOL,rays_for_pixels,select_members,
                            candidate_depths,build_support_index)
from joint_training_anchors import load_cameras
from train_joint_matched import create_output,checked_source,write_json

PARAMETERS=('_xyz','_scaling','_rotation','_opacity')


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if not 0<args.max_seconds<=600:raise ValueError('bounded budget required')
    head=checked_source(args.expected_head);out=create_output(args.output)
    identity=verify_inputs(args.input_ply,args.source_path,args.model_id)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    result=dict(status='started_not_complete',head=head,input_identity=identity,
        protocol=FIELD_PROTOCOL,support_epsilons=[1e-6,1e-8],scope='numerical_pipeline_probe_not_training_or_quality')
    write_json(out/'result.json',result)
    def deadline():
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('pipeline probe deadline')
    try:
        g=GaussianModel(3);g.load_ply(args.input_ply)
        cameras,metadata=load_cameras(args.source_path,2,'cpu')
        cam=next(c for c in cameras if c.image_name=='0049')
        x=torch.linspace(0,cam.image_width-1,8,device='cuda',dtype=torch.float64)
        y=torch.linspace(0,cam.image_height-1,8,device='cuda',dtype=torch.float64)
        xx,yy=torch.meshgrid(x,y,indexing='xy');xy=torch.stack((xx.flatten(),yy.flatten()),-1)
        origin,direction=rays_for_pixels(cam,xy)
        full,sup=select_members(g,origin,direction,epsilon=1e-8,max_members=2048)
        if bool(sup['overflow'].any()):raise RuntimeError('lower-cutoff union exceeds2048; no truncation allowed')
        # Exact full scan versus conservative projected sphere candidates, at
        # both cutoffs, on unchanged pixels including image/tile boundaries.
        index_checks=[]
        for eps in (1e-6,1e-8):
            deadline();ref,reference=select_members(g,origin,direction,epsilon=eps,max_members=2048)
            index=build_support_index(g,cam,eps);candidate_counts=[]
            for i,p in enumerate(xy.cpu().tolist()):
                deadline();u,v=p;tile_x=int(u)//16*16;tile_y=int(v)//16*16
                ids=index.query(tile_x,min(tile_x+15,cam.image_width-1),tile_y,min(tile_y+15,cam.image_height-1))
                # Fractional diagnostic pixels can be beyond the final integer
                # pixel of this tile; enclose the actual queried ray as well.
                ids=index.query(min(tile_x,u),max(tile_x+15,u),min(tile_y,v),max(tile_y+15,v))
                chosen,_=select_members(g,origin[i:i+1],direction[i:i+1],epsilon=eps,max_members=2048,candidate_ids=ids)
                if not torch.equal(ref[i],chosen[0]):raise AssertionError('support index dropped/changed members')
                candidate_counts.append(len(ids))
            index_checks.append(dict(epsilon=eps,rays=64,exact_id_matches=64,
                                     candidate_counts=candidate_counts,member_counts=reference['member_counts'].tolist()))
        result['index_checks']=index_checks
        # Fixed union parameter vectors, no norm-only or differently indexed
        # gradient comparisons. Rootless rays are reported, never valid passes.
        cutoff=[]
        for i in (0,9,19,27,42,49,60,63):
            deadline();ids=full[i]
            if not len(ids):
                cutoff.append(dict(ray_index=i,status='empty_union'));continue
            raw=tuple(getattr(g,n)[ids].detach().double().clone().requires_grad_() for n in PARAMETERS)
            local=SimpleNamespace(**dict(zip(PARAMETERS,raw)),filter_3D=g.filter_3D[ids].detach().double())
            records=[];gradients=[]
            for eps in (1e-6,1e-8):
                torch.cuda.synchronize();begin=time.perf_counter()
                d,v,diag=candidate_depths(local,origin[i:i+1],direction[i:i+1],epsilon=eps,max_members=2048)
                grads=tuple(z.detach() for z in torch.autograd.grad(d.sum(),raw))
                torch.cuda.synchronize()
                records.append(dict(epsilon=eps,depth=float(d.detach()),valid=bool(v),diagnostic=diag,
                                    seconds=time.perf_counter()-begin))
                gradients.append(grads)
            changes={n:gradient_delta(a,b) for n,a,b in zip(PARAMETERS,*gradients)}
            comparable=records[0]['valid'] and records[1]['valid']
            depth_delta=abs(records[0]['depth']-records[1]['depth'])
            cutoff.append(dict(ray_index=i,status='evaluated',union_members=len(ids),levels=records,
                valid_comparable=comparable,depth_delta=depth_delta,gradient_changes=changes,
                observed_cutoff_pass=bool(comparable and depth_delta<=1e-5 and all(x['passed'] for x in changes.values()))))
        result['cutoff_checks']=cutoff
        # Same eight declared pixels: single and batch8 must agree regardless of
        # other rows' padding width. This also measures a useful launch batch.
        selected=torch.tensor([0,9,19,27,42,49,60,63],device='cuda')
        torch.cuda.synchronize();begin=time.perf_counter()
        db,vb,di=candidate_depths(g,origin[selected],direction[selected],require_grad=False)
        torch.cuda.synchronize();batch_time=time.perf_counter()-begin
        singles=[];single_time=0.
        for i in selected.tolist():
            deadline();torch.cuda.synchronize();begin=time.perf_counter()
            d,v,diag=candidate_depths(g,origin[i:i+1],direction[i:i+1],require_grad=False)
            torch.cuda.synchronize();single_time+=time.perf_counter()-begin
            singles.append((float(d),bool(v),diag['convergence'][0]))
        depths=db.cpu().tolist();valid=vb.cpu().tolist()
        for k,(d,v,c) in enumerate(singles):
            if v!=valid[k] or c!=di['convergence'][k] or abs(d-depths[k])>1e-8:
                raise AssertionError('batch composition changed readout/status')
        result['batch_timing']=dict(rays=8,batch_seconds=batch_time,single_sum_seconds=single_time,
            observed_speedup=single_time/batch_time,depths=depths,valid=valid,diagnostic=di)
        if sha256(args.input_ply)!=identity['input_sha256']:raise AssertionError('original model changed')
        result.update(status='complete',seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
            original_ply_unchanged=True)
        write_json(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc),seconds=time.monotonic()-started)
        write_json(out/'result.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-ply',required=True);p.add_argument('--source-path',required=True)
    p.add_argument('--model-id',choices=sorted(MODEL_IDENTITIES),required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-seconds',type=float,default=600)
    main(p.parse_args())
