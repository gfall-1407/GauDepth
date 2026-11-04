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
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    if sha256(args.input_ply)!=EXPECTED_SHA: raise ValueError('wrong original 7k PLY')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=args.expected_head: raise ValueError('wrong code commit')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('dirty experiment checkout')
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    g=GaussianModel(3);g.load_ply(args.input_ply)
    origin,direction,pixels,size=camera_rays(args.source_path,args.view,args.side)
    scales,alpha=g.get_scaling_n_opacity_with_3D_filter
    # Explicit finite-field definition for native fully opaque parameters.
    # This is a 1e-6 relative opacity change; sep control uses the identical field.
    alpha=alpha*(1-1e-6)
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
            rows.append(row);continue
        xyz=g._xyz[ids].detach().clone().requires_grad_()
        ls=g._scaling[ids].detach().clone().requires_grad_()
        q=g._rotation[ids].detach().clone().requires_grad_()
        op=g._opacity[ids].detach().clone().requires_grad_()
        scales2=ls.exp().square()
        filtered2=scales2+g.filter_3D[ids].square()
        a=op.sigmoid()*torch.sqrt(scales2.prod(-1)/filtered2.prod(-1))[:,None]*(1-1e-6)
        p=ray_profiles(origin[i:i+1],direction[i:i+1],xyz,filtered2.sqrt(),q,a)
        records=[]
        for n in (1,2,4):
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.no_grad(): r=joint_depth(*p,near=.2001,far=8.,subdivisions=n)
            torch.cuda.synchronize()
            records.append(dict(subdivisions=n,depth=float(r.depth),valid=bool(r.valid),
                has_sampled_root=bool(r.has_sampled_root),total=float(r.total),near_H=float(r.near_potential),
                slope=float(r.slope),analytic_slope=float(r.analytic_slope),
                knots=r.knots,seconds=time.perf_counter()-begin))
        sd,sv=continuous_sep_depth(*p,near=.2001,far=8.)
        torch.cuda.synchronize();begin=time.perf_counter()
        r=joint_depth(*p,near=.2001,far=8.,subdivisions=2)
        r.depth.sum().backward()
        torch.cuda.synchronize()
        grads={name:dict(finite=bool(torch.isfinite(x.grad).all()),norm=float(x.grad.norm()))
               for name,x in [('xyz',xyz),('log_scale',ls),('rotation',q),('opacity_logit',op)]}
        row.update(status='evaluated',levels=records,sep_depth=float(sd),sep_root=bool(sv),
                   forward_backward_seconds=time.perf_counter()-begin,gradients=grads)
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(scope='numerical_cost_probe_not_quality_or_training',head=head,
        input_ply=args.input_ply,input_sha256=EXPECTED_SHA,gpu=torch.cuda.get_device_name(),
        gaussian_count=len(g.get_xyz),view=args.view,image_size=size,
        opacity_policy='(1-1e-6)*native_3D_filtered_opacity',member_epsilon=args.member_epsilon,
        max_members=args.max_members,selection_seconds=selection_seconds,
        total_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),rows=rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-ply',required=True);p.add_argument('--source-path',required=True)
    p.add_argument('--output',required=True);p.add_argument('--expected-head',required=True)
    p.add_argument('--view',default='0049');p.add_argument('--side',type=int,default=8)
    p.add_argument('--member-epsilon',type=float,default=1e-6)
    p.add_argument('--max-members',type=int,default=512)
    p.add_argument('--max-seconds',type=float,default=240)
    main(p.parse_args())
