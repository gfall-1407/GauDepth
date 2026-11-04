"""Continuous-branch joint TV with refined stationary points and an implicit VJP.

Finite sign sampling is NOT an exhaustive critical-point certificate. Compare
refinements/branch signatures. Forward searches and backward field evaluations
use float64; original model parameters can remain float32. No RGB or normal change.
"""
from dataclasses import dataclass
import math
import torch
from joint_tv_depth import validate_profiles, profile_knots, potential


def slope_at(t, mean, sigma, amplitude, chunk=256):
    values=[]
    for x in t.split(chunk,dim=-1):
        u=(x[...,None]-mean[:,None,:])/sigma[:,None,:]
        g=amplitude[:,None,:]*torch.exp(-.5*u.square())
        values.append((-.5*g*u/(sigma[:,None,:]*(1-g))).sum(-1))
    return torch.cat(values,-1)


class _BranchDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, sigma, amplitude, points, weights, root, slope):
        ctx.save_for_backward(mean,sigma,amplitude,points,weights,slope)
        return root.clone()

    @staticmethod
    def backward(ctx, upstream):
        mean,sigma,amplitude,points,weights,slope=ctx.saved_tensors
        with torch.enable_grad():
            p=tuple(x.detach().requires_grad_(True) for x in (mean,sigma,amplitude))
            # Interior point motion vanishes because H'=0. Root motion is the
            # denominator; this computes the partial parameter derivative only.
            optical=(potential(points,*p)*weights).sum()
            grads=torch.autograd.grad(optical,p,create_graph=False)
        factor=-upstream/slope
        return tuple(g*factor[:,None] for g in grads)+(None,)*4


@dataclass
class StationaryResult:
    depth: torch.Tensor
    valid: torch.Tensor
    has_root: torch.Tensor
    total: torch.Tensor
    near_potential: torch.Tensor
    slope: torch.Tensor
    critical_points: torch.Tensor
    critical_values: torch.Tensor
    prefix_count: int
    grid_samples: int
    max_critical_residual: float
    root_residual: float
    max_critical_bracket_width: float
    root_bracket_width: float
    root_branch_margin: float
    monotonic_directions: torch.Tensor
    continuous_error_certified: bool = False


def stationary_depth(mean, sigma, amplitude, *, near, far, subdivisions=2,
                     min_slope=1e-4,max_near_potential=1e-4,bisections=40):
    """One-ray reference kernel. Gradients valid only within a stable branch.

    No absolute-value smoothing, secant-root gradient, or fallback depth.
    Model/field support must be fixed during the derivative comparison.
    """
    validate_profiles(mean,sigma,amplitude)
    if mean.shape[0]!=1 or not near<far or bisections<24:
        raise ValueError('one ray, ordered interval and >=24 bisections required')
    if min_slope<=0 or max_near_potential<0:
        raise ValueError('invalid root guards')
    # Promotion remains in the parameter graph; selection and bracketing do not.
    p=tuple(x.double() for x in (mean,sigma,amplitude))
    with torch.no_grad():
        t=torch.unique(profile_knots(*p,near,far,subdivisions).flatten(),sorted=True)
        hp=slope_at(t[None],*p)[0]
        if not bool(torch.isfinite(hp).all()): raise ValueError('nonfinite Hprime')
        # Skip zero samples, including underflow-flat tails. A zero with a sign
        # change on its two sides remains bracketed; a stationary touch is not a
        # TV turning point. Sign-sampling can still miss even numbers of roots.
        nz=(hp!=0).nonzero().flatten()
        # signbit comparison avoids underflow in products of tiny derivatives.
        change=(hp[nz[:-1]].signbit()!=hp[nz[1:]].signbit()) if len(nz)>1 else nz.new_zeros(0,dtype=torch.bool)
        lo=t[nz[:-1]][change].clone();hi=t[nz[1:]][change].clone()
        dl=hp[nz[:-1]][change].clone()
        for _ in range(bisections):
            if lo.numel()==0: break
            mid=(lo+hi)*.5;dm=slope_at(mid[None],*p)[0]
            same=dm.signbit()==dl.signbit();zero=dm==0
            lo=torch.where(same|zero,mid,lo)
            hi=torch.where((~same)|zero,mid,hi)
            dl=torch.where(same|zero,dm,dl)
        critical=(lo+hi)*.5
        points=torch.cat((t.new_tensor([near]),critical,t.new_tensor([far])))
        h=potential(points[None],*p)[0]
        directions=slope_at(((points[:-1]+points[1:])*.5)[None],*p)[0].sign()
        signed=(h[1:]-h[:-1])*directions
        monotone_consistent=bool(((signed>=-1e-10) &
                                 ((directions!=0)|((h[1:]-h[:-1]).abs()<=1e-12))).all())
        variation=torch.cat((h.new_zeros(1),(h[1:]-h[:-1]).abs().cumsum(0)))
        target=math.log(2)
        has=variation[-1]>=target
        root=t.new_zeros(1);slope=t.new_zeros(1);prefix_count=0;residual=math.inf
        root_width=math.inf;branch_margin=0.
        prefix_points=points[:1]
        if bool(has):
            j=int((variation>=target).nonzero()[0])
            left,right=points[j-1].clone(),points[j].clone()
            base=variation[j-1];height=h[j-1]
            for _ in range(bisections):
                mid=(left+right)*.5
                a=base+(potential(mid.reshape(1,1),*p)[0,0]-height).abs()
                right=torch.where(a>=target,mid,right)
                left=torch.where(a<target,mid,left)
            root=((left+right)*.5).reshape(1)
            root_width=float(right-left)
            branch_margin=float(torch.minimum(root[0]-points[j-1],points[j]-root[0]))
            slope=slope_at(root[None],*p).abs().flatten()
            prefix_count=j-1
            prefix_points=torch.cat((points[:j],root))
            heights=potential(prefix_points[None],*p)[0]
            residual=float(((heights[1:]-heights[:-1]).abs().sum()-target).abs())
        valid=has & (h[0]<=max_near_potential) & (slope[0]>=min_slope)
        valid=valid & torch.isfinite(root[0]) & torch.isfinite(variation[-1])
        valid=valid & monotone_consistent & (residual<=1e-8)
        if bool(valid):
            heights=potential(prefix_points[None],*p)[0]
            signs=directions[:j]
            weights=torch.zeros_like(heights)
            weights[:-1]-=signs;weights[1:]+=signs
        critical_residual=float(slope_at(critical[None],*p).abs().max()) if len(critical) else 0.
    if bool(valid):
        depth=_BranchDepth.apply(*p,prefix_points[None],weights[None],root,slope)
    else:
        depth=sum(x.sum()*0 for x in p).reshape(1)
    return StationaryResult(depth,valid.reshape(1),has.reshape(1),variation[-1:].detach(),
        h[:1].detach(),slope.detach(),critical.detach(),h[1:-1].detach(),prefix_count,
        len(t),critical_residual,residual,float((hi-lo).max()) if len(lo) else 0.,
        root_width,branch_margin,directions.detach())
