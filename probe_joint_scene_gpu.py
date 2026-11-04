"""Bounded CUDA feasibility probe. No training, GT use, or input writes.

The sparse sample measures numerical behavior/cost, NOT reconstruction quality.
Source is synchronized by Git. Invoke from an isolated checkout on G6.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from joint_tv_depth import ray_profiles, joint_depth, continuous_sep_depth
from joint_tv_convergence import PROBE_PROTOCOL, refinement_comparison

EXPECTED_SHA = '51f5f92bd6015ae98509b58b5aa293d110f4b4f13b8217916fdea0a819d84187'


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def camera_rays(source, name, side=8):
    base=Path(source)/'sparse'/'0'
    intr=read_intrinsics_binary(str(base/'cameras.bin'))
    ext=read_extrinsics_binary(str(base/'images.bin'))
    cam=next(v for v in ext.values() if Path(v.name).stem==name)
    k=intr[cam.camera_id]
    if k.model=='PINHOLE': fx,fy,cx,cy=k.params
    elif k.model=='SIMPLE_PINHOLE': fx,cx,cy=k.params; fy=fx
    else: raise ValueError('unsupported camera model '+k.model)
    # Match Camera's r=2 centered principal point convention, not raw COLMAP cx.
    w,h=round(k.width/2),round(k.height/2)
    fx,fy=fx*w/k.width,fy*h/k.height
    xx,yy=np.meshgrid(np.linspace(0,w-1,side),np.linspace(0,h-1,side))
    pixels=np.stack((xx.ravel(),yy.ravel()),-1)
    d=np.stack(((pixels[:,0]-(w-1)/2)/fx,(pixels[:,1]-(h-1)/2)/fy,np.ones(len(pixels))),-1)
    r=qvec2rotmat(cam.qvec)
    center=-r.T@cam.tvec
    o=torch.tensor(np.broadcast_to(center,d.shape).copy(),device='cuda',dtype=torch.float32)
    return o,torch.tensor(d@r,device='cuda',dtype=torch.float32),pixels,(w,h)


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(args.output).resolve()
    # Every generated file must stay in the dedicated disposable experiment area.
    if not out.is_relative_to(Path('/root/autodl-tmp/joint_tv_runs').resolve()):
        raise ValueError('output must be under /root/autodl-tmp/joint_tv_runs')
    if out.exists(): raise FileExistsError(out)
    journal=out.with_suffix('.rows.jsonl')
    if journal.exists(): raise FileExistsError(journal)
    if not 0 < args.member_epsilon < 1 or not 1 <= args.max_members <= 512:
        raise ValueError('invalid support budget')
    if not 1 <= args.side <= 8 or not 0 < args.max_seconds <= 240:
        raise ValueError('invalid probe budget')
    if sha256(args.input_ply)!=EXPECTED_SHA: raise ValueError('wrong original 7k PLY')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=args.expected_head: raise ValueError('wrong code commit')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('dirty experiment checkout')
    protocol=dict(PROBE_PROTOCOL,member_epsilon=args.member_epsilon,max_members=args.max_members)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='started_not_complete',head=head,protocol=protocol),indent=2)+'\n')
    journal.touch(exist_ok=False)
    def save_row(row):
        line=json.dumps(row,allow_nan=False)
        with journal.open('a') as f: f.write(line+'\n')
        print(line,flush=True)
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    g=GaussianModel(3);g.load_ply(args.input_ply)
    origin,direction,pixels,size=camera_rays(args.source_path,args.view,args.side)
    scales,alpha=g.get_scaling_n_opacity_with_3D_filter
    # Explicit finite-field definition for native fully opaque parameters.
    # This is a 1e-6 relative opacity change; sep control uses the identical field.
    alpha=alpha*protocol['opacity_factor']
    mean_parts=[];sigma_parts=[];amp_parts=[]
    torch.cuda.synchronize(); begin=time.perf_counter()
    with torch.no_grad():
        for s in range(0,len(g.get_xyz),8192):
            m,sd,a=ray_profiles(origin,direction,g.get_xyz[s:s+8192],scales[s:s+8192],
                               g.get_rotation[s:s+8192],alpha[s:s+8192])
            mean_parts.append(m);sigma_parts.append(sd);amp_parts.append(a)
        means=torch.cat(mean_parts,-1);sigmas=torch.cat(sigma_parts,-1);amps=torch.cat(amp_parts,-1)
    torch.cuda.synchronize(); selection_seconds=time.perf_counter()-begin
    rows=[]
    for i in range(len(origin)):
        if time.monotonic()-started>args.max_seconds: raise TimeoutError('probe time limit')
        # Full-line peak culling is conservative for finite near/far interval.
        keep=amps[i]>=args.member_epsilon
        ids=keep.nonzero().flatten()
        omitted_bound=float((-torch.log1p(-amps[i,~keep].double())).sum())
        row=dict(pixel=pixels[i].tolist(),members=len(ids),omitted_TV_upper_bound=omitted_bound)
        if len(ids)==0 or len(ids)>args.max_members:
            row['status']='empty' if len(ids)==0 else 'over_member_budget_no_truncation'
            rows.append(row);save_row(row);continue
        xyz=g._xyz[ids].detach().clone().requires_grad_()
        ls=g._scaling[ids].detach().clone().requires_grad_()
        q=g._rotation[ids].detach().clone().requires_grad_()
        op=g._opacity[ids].detach().clone().requires_grad_()
        params={'xyz':xyz,'log_scale':ls,'rotation':q,'opacity_logit':op}
        def profiles():
            scales2=ls.exp().square()
            filtered2=scales2+g.filter_3D[ids].square()
            a=op.sigmoid()*torch.sqrt(scales2.prod(-1)/filtered2.prod(-1))[:,None]*protocol['opacity_factor']
            return ray_profiles(origin[i:i+1],direction[i:i+1],xyz,filtered2.sqrt(),q,a)
        records=[];comparisons=[];previous_grads=None
        for n in protocol['refinement_levels']:
            torch.cuda.synchronize();begin=time.perf_counter()
            # Rebuild the small upstream graph at every level; do not retain old
            # member-knot graphs, and never compare gradients on different IDs.
            r=joint_depth(*profiles(),near=protocol['near_axial'],far=protocol['far_axial'],
                          subdivisions=n,min_slope=protocol['min_secant_slope'],
                          max_near_potential=protocol['max_near_H'])
            grads={name:grad.detach() for name,grad in zip(params,torch.autograd.grad(
                r.depth.sum(),tuple(params.values())))}
            torch.cuda.synchronize()
            record=dict(subdivisions=n,depth=float(r.depth),valid=bool(r.valid),
                has_sampled_root=bool(r.has_sampled_root),total=float(r.total),near_H=float(r.near_potential),
                slope=float(r.slope),analytic_slope=float(r.analytic_slope) if bool(r.valid) else None,
                knots=r.knots,forward_backward_seconds=time.perf_counter()-begin,
                gradients={name:dict(finite=bool(torch.isfinite(v).all()),
                           norm=float(v.double().norm()) if bool(torch.isfinite(v).all()) else None)
                           for name,v in grads.items()})
            if previous_grads is not None:
                comparisons.append(refinement_comparison(records[-1],record,previous_grads,grads))
            records.append(record);previous_grads=grads
        with torch.no_grad():
            sd,sv=continuous_sep_depth(*profiles(),near=protocol['near_axial'],far=protocol['far_axial'])
        row.update(status='evaluated',levels=records,sep_depth=float(sd),sep_root=bool(sv),
                   refinement_comparisons=comparisons,
                   observed_all_pairs_passed=all(c['observed_pair_passed'] for c in comparisons),
                   continuous_error_certified=False)
        rows.append(row);save_row(row)
    report=dict(status='complete',scope='numerical_cost_probe_not_quality_or_training',head=head,
        input_ply=args.input_ply,input_sha256=EXPECTED_SHA,gpu=torch.cuda.get_device_name(),
        gaussian_count=len(g.get_xyz),view=args.view,image_size=size,
        protocol=protocol,member_epsilon=args.member_epsilon,
        max_members=args.max_members,selection_seconds=selection_seconds,
        total_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),rows=rows)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-ply',required=True);p.add_argument('--source-path',required=True)
    p.add_argument('--output',required=True);p.add_argument('--expected-head',required=True)
    p.add_argument('--view',default='0049');p.add_argument('--side',type=int,default=8)
    p.add_argument('--member-epsilon',type=float,default=1e-6)
    p.add_argument('--max-members',type=int,default=512)
    p.add_argument('--max-seconds',type=float,default=240)
    main(p.parse_args())
