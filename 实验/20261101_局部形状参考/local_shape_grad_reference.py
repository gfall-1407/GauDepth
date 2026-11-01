"""Torch CPU first-order reference for one fixed density-ridge root branch.

Outputs camera-z depth and actual world surface normal, with gradients to
means, positive scales/amplitudes, world rotation tangent, origin and ray.
The discrete root choice and shortest-axis ordering are intentionally fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import local_shape_reference as old

DTYPE = torch.float64
GAP_EPS = 1e-10
FZ_EPS = 1e-10


def tensor(x, grad=False):
    return torch.tensor(x, dtype=DTYPE, requires_grad=grad)


def skew(w):
    zero = w[0]*0
    return torch.stack((torch.stack((zero,-w[2],w[1])),
        torch.stack((w[2],zero,-w[0])),torch.stack((-w[1],w[0],zero))))


class DominantEigenline(torch.autograd.Function):
    """Leading symmetric eigenvector JVP/VJP; lower eigenvalue tie is allowed."""
    @staticmethod
    def forward(ctx, B, orient):
        values, vectors = torch.linalg.eigh(B)
        n = vectors[:,-1]
        if abs(float(torch.dot(n,orient))) <= 1e-10:
            raise ValueError('invalid_eigenline_orientation_chart')
        if torch.dot(n,orient)<0: n=-n
        gap = values[-1]-values[-2]
        if not torch.isfinite(gap) or gap <= GAP_EPS:
            raise ValueError('invalid_dominant_eigen_gap')
        ctx.save_for_backward(B,n,values[-1])
        return n

    @staticmethod
    def backward(ctx, grad_n):
        B,n,lam = ctx.saved_tensors
        I = torch.eye(3,dtype=B.dtype,device=B.device)
        P = I-torch.outer(n,n)
        v = torch.linalg.solve(lam*I-B+torch.outer(n,n),P@grad_n)
        dB = .5*(torch.outer(v,n)+torch.outer(n,v))
        return dB,None


def field(mu,scale,omega,amplitude,base_rotation,x,orient):
    """Return F, spatial gradient F, curvature, eigen gap and residual scale."""
    rotations = torch.stack([torch.matrix_exp(skew(w)) @ r for w,r in zip(omega,base_rotation)])
    ordered = torch.argsort(scale.detach(),dim=1)
    sorted_scale = torch.gather(scale.detach(),1,ordered)
    if torch.any((sorted_scale[:,1:]-sorted_scale[:,:-1])/sorted_scale[:,1:] <= 1e-10):
        raise ValueError('invalid_scale_axis_tie')
    shortest = ordered[:,0]
    middle = ordered[:,1]
    axis = torch.stack([rotations[i,:,int(shortest[i])] for i in range(len(mu))])
    gamma = 1-(scale[torch.arange(len(mu)),shortest]/scale[torch.arange(len(mu)),middle])**2
    A = gamma[:,None,None]*axis[:,:,None]*axis[:,None,:]
    Q = rotations @ torch.diag_embed(scale**-2) @ rotations.transpose(1,2)
    delta = x[None,:]-mu
    b = -torch.einsum('nij,nj->ni',Q,delta)
    log_phi = torch.log(amplitude)-.5*torch.einsum('ni,nij,nj->n',delta,Q,delta)
    weight = torch.softmax(log_phi,0)
    score = (weight[:,None]*b).sum(0)
    B = (weight[:,None,None]*A).sum(0)
    n = DominantEigenline.apply(B,orient)
    values = torch.linalg.eigvalsh(B.detach())
    gap = values[-1]-values[-2]
    H = (weight[:,None,None]*(b[:,:,None]*b[:,None,:]-Q)).sum(0)-torch.outer(score,score)
    I = torch.eye(3,dtype=DTYPE)
    P = I-torch.outer(n,n)
    lam = n@B@n
    M = lam*I-B+torch.outer(n,n)
    dB = torch.einsum('n,nj,nab->jab',weight,b-score[None,:],A)
    dn = torch.stack([torch.linalg.solve(M,P@dB[j]@n) for j in range(3)])
    gradF = H@n+torch.stack([torch.dot(score,dn[j]) for j in range(3)])
    F = torch.dot(n,score)
    residual_scale = torch.sqrt(torch.sum(weight*torch.diagonal(Q,dim1=1,dim2=2).sum(1)))
    return dict(F=F,gradF=gradF,curvature=torch.dot(n,H@n),gap=gap,
        residual_scale=residual_scale,guide=n)


def selected_root(mu,scale,omega,amplitude,base_rotation,origin,ray,bracket):
    """Fixed bracket branch; no derivative through bisection or discrete selection."""
    for name,value in [('mu',mu),('scale',scale),('omega',omega),('amplitude',amplitude),
                       ('base_rotation',base_rotation),('origin',origin),('ray',ray)]:
        if value.dtype != DTYPE or not bool(torch.isfinite(value).all()): raise ValueError('invalid_'+name)
    if mu.ndim != 2 or mu.shape[1] != 3 or scale.shape != mu.shape or omega.shape != mu.shape:
        raise ValueError('invalid_parameter_shape')
    if amplitude.shape != (len(mu),) or base_rotation.shape != (len(mu),3,3):
        raise ValueError('invalid_parameter_shape')
    I = torch.eye(3,dtype=DTYPE)
    if any(not torch.allclose(r.T@r,I,atol=1e-10,rtol=0)
           or abs(float(torch.linalg.det(r))-1)>1e-10 for r in base_rotation):
        raise ValueError('invalid_base_rotation')
    if origin.shape != (3,) or ray.shape != (3,) or abs(float(ray[2].detach())-1)>1e-12:
        raise ValueError('ray_must_have_camera_z_one')
    if bool(torch.any(scale<=0)) or bool(torch.any(amplitude<=0)):
        raise ValueError('nonpositive_scale_or_amplitude')
    R = torch.stack([torch.matrix_exp(skew(w))@r for w,r in zip(omega,base_rotation)])
    gs = [old.gaussian(mu[i].detach().tolist(),scale[i].detach().tolist(),
        float(amplitude[i].detach()),R[i].detach().tolist()) for i in range(len(mu))]
    z0 = old.root_in_bracket(gs,ray.detach().tolist(),*bracket,origin=origin.detach().tolist())
    x0 = origin+z0*ray
    state = field(mu,scale,omega,amplitude,base_rotation,x0,ray)
    Fz = torch.dot(state['gradF'],ray)
    if (float(state['curvature'].detach()) >= 0 or abs(float(Fz.detach())) <= FZ_EPS
        or float(state['gap']) <= GAP_EPS
        or abs(float((state['F']/state['residual_scale']).detach())) > 1e-8):
        raise ValueError('invalid_unstable_or_nonroot_branch')
    # First-order implicit root value. The centered zero has the same forward z0.
    z = z0-(state['F']-state['F'].detach())/Fz.detach()
    actual = field(mu,scale,omega,amplitude,base_rotation,origin+z*ray,ray)['gradF']
    length = torch.linalg.vector_norm(actual)
    if not bool(torch.isfinite(length)) or float(length.detach()) <= 1e-12:
        raise ValueError('invalid_surface_normal')
    normal = actual/length
    facing = float(torch.dot(normal,ray).detach())
    if abs(facing) <= 1e-10: raise ValueError('invalid_surface_orientation_chart')
    if facing>0: normal=-normal
    guide = state['guide'].detach().numpy()
    actual = normal.detach().numpy()
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(guide,actual)),-1,1))))
    return z,normal,dict(gap=float(state['gap']),Fz=float(Fz.detach()),
        guide_actual_angle_degrees=angle,
        residual=float((state['F']/state['residual_scale']).detach()),root_value=z0)


def cases():
    n,tangent,other = np.array([.6,0,.8]),np.array([.8,0,-.6]),np.array([0,1.,0])
    rot = np.column_stack((tangent,other,n))
    center = np.array([0.,0.,2.])
    plane = [old.gaussian(center+a*tangent+b*other,[.55+.04*i,.35+.03*i,.06+.006*i],
        .2+.11*i,rot.tolist()) for i,(a,b) in enumerate([(-.5,-.2),(-.3,.3),(.1,-.1),(.4,.25),(.6,-.35)])]
    separated = [old.gaussian([0.,0.,2.+a],[.5,.4,.1],.5) for a in (-.15,.15)]
    curved = [old.gaussian([-.25,0.,1.96],[.5,.35,.1],.6),
        old.gaussian([.3,.05,2.08],[.55,.4,.13],.8)]
    curved = old.shifted(curved,0,'rotation',1,.25)
    curved = old.shifted(curved,1,'rotation',1,-.3)
    return [('coplanar',plane,[.3,.1,1.],(1.1,2.2)),
        ('two_layers_near',separated,[0.,0.,1.],(1.6,1.95)),
        ('two_layers_far',separated,[0.,0.,1.],(2.05,2.4)),
        ('varying_normal',curved,[.08,.02,1.],(1.8,2.25))]


def check_case(name,gs,ray_values,bracket):
    mu=tensor([g['mu'] for g in gs],True)
    scale=tensor([g['scale'] for g in gs],True)
    omega=tensor(np.zeros((len(gs),3)),True)
    amplitude=tensor([g['amplitude'] for g in gs],True)
    base=tensor([g['rotation'] for g in gs])
    origin=tensor([0.,0.,0.],True)
    ray=tensor(ray_values,True)
    inputs=[mu,scale,omega,amplitude,origin,ray]
    start=time.perf_counter()
    z,normal,diagnostic=selected_root(*inputs[:4],base,origin,ray,bracket)
    forward_seconds=time.perf_counter()-start
    baseline=old.field(gs,old.mul(float(z.detach()),ray_values),ray_values)
    old_normal=np.asarray(baseline['surface_normal'])
    if np.dot(old_normal,ray_values)>0: old_normal=-old_normal
    normal_reference_error=float(np.max(np.abs(normal.detach().numpy()-old_normal)))
    assert normal_reference_error<1e-10
    loss=1.3*z+normal@tensor([.2,-.3,.4])
    gradients=torch.autograd.grad(loss,inputs)
    forward_backward_seconds=time.perf_counter()-start
    errors={}
    for slot,(group,parameter,gradient) in enumerate(zip(
        ('mean','scale','rotation','amplitude','origin','query_ray'),inputs,gradients)):
        samples=list(np.ndindex(parameter.shape))
        if group=='query_ray': samples=[(0,),(1,)]  # z component is fixed to one.
        maximum=0.
        for index in samples:
            step=1e-5
            copies=[]
            for sign in (1,-1):
                changed=[v.detach().clone() for v in inputs]
                changed[slot][index]+=sign*step
                zp,np_,_=selected_root(*changed[:4],base,changed[4],changed[5],bracket)
                copies.append(float(1.3*zp+np_@tensor([.2,-.3,.4])))
            fd=(copies[0]-copies[1])/(2*step)
            analytic=float(gradient[index])
            maximum=max(maximum,abs(fd-analytic)/max(1.,abs(fd),abs(analytic)))
        errors[group]=maximum
    assert max(errors.values())<1e-4,(name,errors)
    gradcheck_passed = None
    if name == 'varying_normal':
        xy=tensor(ray_values[:2],True)
        def output(m,s,w,a,o,xy):
            query=torch.cat((xy,torch.ones(1,dtype=DTYPE)))
            depth,surface,_=selected_root(m,s,w,a,base,o,query,bracket)
            return depth,surface
        gradcheck_passed=bool(torch.autograd.gradcheck(output,
            (mu,scale,omega,amplitude,origin,xy),eps=1e-6,atol=1e-5,rtol=1e-3))
        assert gradcheck_passed
    return dict(name=name,z=float(z.detach()),normal=normal.detach().tolist(),
        diagnostic=diagnostic,max_relative_error=errors,gradcheck=gradcheck_passed,
        normal_reference_max_error=normal_reference_error,
        forward_seconds=forward_seconds,forward_backward_seconds=forward_backward_seconds,
        seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(status='partial',dtype='float64',cases=[])
    for case in cases():
        report['cases'].append(check_case(*case))
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    gs=cases()[0][1]
    mu=tensor([g['mu'] for g in gs]);scale=tensor([g['scale'] for g in gs])
    omega=tensor(np.zeros((len(gs),3)));amp=tensor([g['amplitude'] for g in gs])
    base=tensor([g['rotation'] for g in gs]);origin=tensor([0.,0.,0.]);ray=tensor([.3,.1,1.])
    invalid={}
    for name in ('nonpositive_scale','nonpositive_amplitude','scale_axis_tie','zero_dominant_gap','non_camera_z_ray'):
        values=[v.clone() for v in (mu,scale,omega,amp,base,origin,ray)]
        if name=='nonpositive_scale': values[1][0,0]=0
        elif name=='nonpositive_amplitude': values[3][0]=0
        elif name=='scale_axis_tie': values[1][0,0]=values[1][0,1]
        elif name=='zero_dominant_gap': values[1][:]=.5
        else: values[6][2]=2
        try: selected_root(*values,(1.1,2.2))
        except ValueError as error: invalid[name]=str(error)
        else: raise AssertionError('Invalid input accepted: '+name)
    report['invalid_checks']=invalid
    report['status']='pass'
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__': main()
