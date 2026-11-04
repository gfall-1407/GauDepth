"""Bounded fixed-input CUDA diagnosis; no optimization or input writes.

Compare float32/float64 sampled TV and continuous stationary-point TV on the
same 64 rays and support. Finite-difference forwards independently re-search.
Finite sign sampling is not an exhaustive continuous-error certificate.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import torch
from scene.gaussian_model import GaussianModel
from probe_joint_scene_gpu import camera_rays
from joint_tv_inputs import verify_inputs, MODEL_IDENTITIES
from joint_tv_depth import ray_profiles, joint_depth, profile_knots, potential
from joint_tv_convergence import gradient_delta
from stationary_tv_depth import stationary_depth, slope_at

NAMES=('xyz','log_scale','rotation','opacity_logit')
NEAR,FAR=.2001,8.


def profiles(raw, filt, origin, direction):
    xyz,ls,q,op=raw
    s2=ls.exp().square();f2=s2+filt.square()
    alpha=op.sigmoid()*torch.sqrt(s2.prod(-1)/f2.prod(-1))[:,None]*(1-1e-6)
    return ray_profiles(origin,direction,xyz,f2.sqrt(),q,alpha)


def branch(r):
    return dict(valid=bool(r.valid),has_root=bool(r.has_root),
                critical_count=len(r.critical_points),prefix_count=r.prefix_count,
                directions=r.monotonic_directions.tolist())


def compare(a,b,ga,gb):
    compatible=a['valid'] and b['valid'] and a['has_root']==b['has_root']
    gd={k:gradient_delta(x,y) for k,x,y in zip(NAMES,ga,gb)}
    same_branch=a.get('branch')==b.get('branch') if 'branch' in a and 'branch' in b else None
    critical_delta=None
    if same_branch:
        critical_delta=max((abs(x-y) for x,y in zip(a['critical_points'],b['critical_points'])),default=0.)
    passed=bool(compatible and abs(a['depth']-b['depth'])<=1e-5 and
                abs(a['total']-b['total'])<=1e-5 and all(v['passed'] for v in gd.values()))
    if same_branch is not None: passed=passed and same_branch and critical_delta<=1e-7
    return dict(valid_comparable=bool(compatible),depth_delta=abs(a['depth']-b['depth']),
                tv_delta=abs(a['total']-b['total']),gradients=gd,same_branch=same_branch,
                max_critical_position_delta=critical_delta,
                observed_passed=passed,continuous_error_certified=False)


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if not 0<args.max_seconds<=600: raise ValueError('time budget must be <=600s')
    out=Path(args.output).resolve();root=Path('/root/joint_tv_runs').resolve()
    if not out.is_relative_to(root): raise ValueError('dedicated system-disk output required')
    root.mkdir(parents=True,exist_ok=True)
    if root.stat().st_dev!=Path('/root').stat().st_dev or shutil.disk_usage(root).free<2<<30:
        raise ValueError('system disk must have >=2GiB free')
    journal=out.with_suffix('.rows.jsonl')
    if out.exists() or journal.exists(): raise FileExistsError(out)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=args.expected_head or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('unexpected or dirty source')
    identity=verify_inputs(args.input_ply,args.source_path,args.model_id)
    out.parent.mkdir(parents=True,exist_ok=True);journal.touch(exist_ok=False)
    protocol=dict(near=NEAR,far=FAR,opacity_factor=1-1e-6,member_epsilon=1e-6,
        selection='same fixed float32 full-line support as previous probe',max_members=512,
        methods=['pl32_4','pl32_8','pl64_4','pl64_8','stationary64_1','stationary64_2','stationary64_4'],
        root_bisections=40,fd_rays='first four valid stationary rays in unchanged 8x8 grid',
        fd_directions='one deterministic random unit direction for each parameter group',
        continuous_error_certified=False)
    manifest=dict(status='started_not_complete',head=head,input_identity=identity,protocol=protocol)
    out.write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def deadline():
        if time.monotonic()-start>args.max_seconds: raise TimeoutError('probe deadline')
    g=GaussianModel(3);g.load_ply(args.input_ply)
    if identity['expected_rows'] is not None and len(g.get_xyz)!=identity['expected_rows']:
        raise ValueError('model row count mismatch')
    origins,directions,pixels,size=camera_rays(args.source_path,'0049',8)
    with torch.no_grad():
        scales,alpha=g.get_scaling_n_opacity_with_3D_filter
        parts=[]
        for j in range(0,len(g.get_xyz),8192):
            parts.append(ray_profiles(origins,directions,g.get_xyz[j:j+8192],scales[j:j+8192],
                                      g.get_rotation[j:j+8192],alpha[j:j+8192]*(1-1e-6))[2])
        amps=torch.cat(parts,-1)
    rows=[];fd_count=0
    for i in range(64):
        deadline();ids=(amps[i]>=1e-6).nonzero().flatten()
        row=dict(ray_index=i,pixel=pixels[i].tolist(),members=len(ids))
        if not 0<len(ids)<=512:
            row['status']='support_budget_skip_no_truncation'
        else:
            row['status']='evaluated';records={};gradients={};quantization={}
            raw_base=tuple(x[ids].detach() for x in (g._xyz,g._scaling,g._rotation,g._opacity))
            for dtype,label in ((torch.float32,'pl32'),(torch.float64,'pl64'),(torch.float64,'stationary64')):
                for n in ((1,2,4) if label=='stationary64' else (4,8)):
                    deadline();raw=tuple(x.to(dtype).clone().requires_grad_() for x in raw_base)
                    f=g.filter_3D[ids].to(dtype);o=origins[i:i+1].to(dtype);d=directions[i:i+1].to(dtype)
                    torch.cuda.synchronize();begin=time.perf_counter()
                    p=profiles(raw,f,o,d)
                    r=(stationary_depth if label=='stationary64' else joint_depth)(
                        *p,near=NEAR,far=FAR,subdivisions=n)
                    grads=tuple(x.detach() for x in torch.autograd.grad(r.depth.sum(),raw))
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                    rec=dict(depth=float(r.depth.detach()),valid=bool(r.valid),total=float(r.total.detach()),
                        has_root=bool(r.has_root if label=='stationary64' else r.has_sampled_root),
                        seconds=seconds,gradient_norms={k:float(x.double().norm()) for k,x in zip(NAMES,grads)})
                    if label=='stationary64':
                        rec.update(branch=branch(r),critical_points=r.critical_points.tolist(),
                            max_critical_residual=r.max_critical_residual,
                            max_critical_bracket_width=r.max_critical_bracket_width,
                            root_bracket_width=r.root_bracket_width if math.isfinite(r.root_bracket_width) else None,
                            root_residual=r.root_residual if math.isfinite(r.root_residual) else None,
                            root_branch_margin=r.root_branch_margin)
                    elif n==8:
                        with torch.no_grad():
                            t=profile_knots(*p,NEAR,FAR,n);h=potential(t,*p)
                            dt=t[:,1:]-t[:,:-1];dh=h[:,1:]-h[:,:-1]
                            # Same-sign derivative endpoints identify a conservative
                            # diagnostic subset; this is not monotonic certification.
                            hp=slope_at(t,*p)
                            signal=(hp[:,:-1]!=0)&(hp[:,1:]!=0)&(hp[:,:-1].signbit()==hp[:,1:].signbit())
                            region=(t[:,1:]<=r.depth.detach()[:,None]) & (dt>0)
                            quantization[label]=dict(positive_intervals=int((dt>0).sum()),
                                zero_delta_signal_intervals=int(((dt>0)&(dh==0)&signal).sum()),
                                prefix_intervals=int(region.sum()),
                                prefix_zero_delta_signal_intervals=int((region&(dh==0)&signal).sum()))
                    key=f'{label}_{n}';records[key]=rec;gradients[key]=grads
            pairs=(('pl32_4','pl32_8'),('pl64_4','pl64_8'),
                   ('pl32_8','pl64_8'),('pl64_8','stationary64_4'),
                   ('stationary64_1','stationary64_2'),('stationary64_2','stationary64_4'))
            comparisons={a+'__'+b:compare(records[a],records[b],gradients[a],gradients[b]) for a,b in pairs}
            fd=[]
            if fd_count<4 and records['stationary64_4']['valid']:
                fd_count+=1
                raw=tuple(x.double().clone() for x in raw_base)
                f=g.filter_3D[ids].double();o=origins[i:i+1].double();d=directions[i:i+1].double()
                generator=torch.Generator(device='cuda').manual_seed(713+i)
                for k,name in enumerate(NAMES):
                    v=torch.randn(raw[k].shape,generator=generator,device='cuda',dtype=torch.float64)
                    v=v/v.norm();analytic=float((gradients['stationary64_4'][k]*v).sum())
                    for eps in ((1e-5,3e-6) if k==0 else (1e-4,3e-5)):
                        deadline();values=[];branches=[]
                        for sign in (-1,1):
                            pert=list(raw);pert[k]=raw[k]+sign*eps*v
                            rr=stationary_depth(*profiles(pert,f,o,d),near=NEAR,far=FAR,subdivisions=4)
                            values.append(float(rr.depth));branches.append(branch(rr))
                        numeric=(values[1]-values[0])/(2*eps)
                        stable=all(x==records['stationary64_4']['branch'] for x in branches)
                        error=abs(numeric-analytic)
                        fd.append(dict(parameter=name,eps=eps,analytic=analytic,numeric=numeric,
                            abs_error=error,branch_stable=stable,
                            passed=stable and error<=1e-5+.01*max(abs(numeric),abs(analytic))))
            row.update(records=records,comparisons=comparisons,quantization=quantization,finite_differences=fd)
        rows.append(row)
        line=json.dumps(row,allow_nan=False)
        with journal.open('a') as stream: stream.write(line+'\n')
        print(line,flush=True)
    manifest.update(status='complete',rows=rows,total_seconds=time.monotonic()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                    image_size=size,view='0049',gaussian_count=len(g.get_xyz))
    out.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-ply',required=True);parser.add_argument('--source-path',required=True)
    parser.add_argument('--model-id',required=True,choices=sorted(MODEL_IDENTITIES))
    parser.add_argument('--expected-head',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=float,default=600)
    main(parser.parse_args())
