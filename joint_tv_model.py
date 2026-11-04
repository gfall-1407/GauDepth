"""Shared continuous field/support/readout for training and dense extraction.

Support is freshly selected from full raw-parameter float64 profiles. No model,
dataset or optimizer writes. Conservative projected spheres only accelerate the
candidate list; the exact per-ray amplitude cutoff is still applied afterward.
"""
from dataclasses import dataclass
import math
import time
import torch
import torch.nn.functional as F
from joint_tv_depth import ray_profiles, validate_profiles
from stationary_tv_batch import stationary_depth_batch

FIELD_PROTOCOL=dict(near=.2001,far=8.,opacity_factor=1-1e-6,
    support_dtype='float64_raw_parameter_chain',field_dtype='float64',
    default_epsilon=1e-6,default_max_members=1024,
    support_refresh='every_call_detached_selection',refinements=[4,8,16],
    depth_atol=1e-5,tv_atol=1e-5,critical_position_atol=1e-7,
    no_root='zero_sentinel_with_separate_mask_no_fallback',
    continuous_error_certified=False)
GUARD_NAMES=('no_detected_root','near_potential','root_slope',
             'nonfinite_root_or_TV','monotonic_direction','root_residual')


def rays_for_pixels(camera,xy):
    xy=torch.as_tensor(xy,device=camera.camera_center.device,dtype=torch.float64)
    if xy.ndim!=2 or xy.shape[1]!=2: raise ValueError('xy must be [R,2]')
    local=torch.stack(((xy[:,0]-camera.Cx)/camera.Fx,
                       (xy[:,1]-camera.Cy)/camera.Fy,torch.ones_like(xy[:,0])),-1)
    origin=camera.camera_center.double()[None].expand(len(xy),-1)
    return origin,local@camera.R.double().T


def profiles_for_ids(g,origin,direction,ids):
    xyz=g._xyz[ids].double();ls=g._scaling[ids].double()
    s2=ls.exp().square();filtered2=s2+g.filter_3D[ids].double().square()
    alpha=g._opacity[ids].double().sigmoid()*torch.sqrt(
        s2.prod(-1)/filtered2.prod(-1))[:,None]*FIELD_PROTOCOL['opacity_factor']
    return ray_profiles(origin.double(),direction.double(),xyz,filtered2.sqrt(),
                        g._rotation[ids].double(),alpha)


@torch.no_grad()
def select_members(g,origin,direction,*,epsilon=1e-6,max_members=1024,candidate_ids=None):
    if not 0<epsilon<1 or not 1<=max_members<=4096: raise ValueError('invalid support budget')
    if origin.ndim!=2 or origin.shape!=direction.shape or origin.shape[1]!=3:
        raise ValueError('origins/directions must be [R,3]')
    if not 1<=len(origin)<=128: raise ValueError('ray batch must have 1..128 rays')
    if not g._xyz.is_cuda or not origin.is_cuda: raise RuntimeError('CUDA required')
    ids=torch.arange(len(g._xyz),device=g._xyz.device) if candidate_ids is None else candidate_ids
    if ids.dtype!=torch.long or ids.ndim!=1 or not ids.is_cuda:
        raise ValueError('candidate IDs must be a CUDA long vector')
    if len(ids) and (int(ids.min())<0 or int(ids.max())>=len(g._xyz)):
        raise ValueError('candidate ID out of bounds')
    # Sorted unique IDs preserve a stable parameter ordering for comparisons.
    ids=torch.unique(ids,sorted=True)
    chunks=[]
    for part in ids.split(8192):
        if part.numel():
            profiles=profiles_for_ids(g,origin,direction,part)
            validate_profiles(*profiles)
            chunks.append(profiles[2])
    amps=torch.cat(chunks,-1) if chunks else origin.new_empty((len(origin),0),dtype=torch.float64)
    keep=amps>=epsilon
    counts=keep.sum(-1)
    selected=[ids[keep[i]] if int(counts[i])<=max_members else ids[:0] for i in range(len(origin))]
    inside_omitted=torch.where(keep,torch.zeros_like(amps),-torch.log1p(-amps)).sum(-1)
    inside_omitted=torch.where(counts>max_members,(-torch.log1p(-amps)).sum(-1),inside_omitted)
    # Outside a correctly built conservative support index, each peak is <eps.
    outside=len(g._xyz)-len(ids)
    bound=inside_omitted+outside*(-math.log1p(-epsilon))
    return selected,dict(member_counts=counts,overflow=counts>max_members,
        candidate_count=len(ids),omitted_TV_upper_bound=bound,
        bound_uses_loose_outside_index_term=outside>0)


def _pad_rows(g,origin,direction,ids):
    width=max(1,max(map(len,ids)));all_p=[[],[],[]];masks=[]
    for i,idx in enumerate(ids):
        if len(idx):
            p=profiles_for_ids(g,origin[i:i+1],direction[i:i+1],idx)
        else:
            # The zero ray still has a gradient path to model parameters.
            zero=sum(getattr(g,name)[0].double().sum()*0
                     for name in ('_xyz','_scaling','_rotation'))
            # A PLY may contain a saturated +/-Inf opacity logit. Its
            # activated opacity is finite, while raw Inf * 0 would be NaN.
            zero=zero+g._opacity[0].double().sigmoid().sum()*0
            p=(zero.reshape(1,1),zero.reshape(1,1)+1,zero.reshape(1,1))
        for k in range(3): all_p[k].append(F.pad(p[k],(0,width-p[k].shape[1]),value=1 if k==1 else 0))
        masks.append(torch.arange(width,device=origin.device)<len(idx))
    return tuple(torch.cat(x,0) for x in all_p),torch.stack(masks)


def refinement_agreement(a,b):
    same=(a.has_root==b.has_root)&(a.valid==b.valid)
    same=same&(a.critical_count==b.critical_count)&(a.prefix_count==b.prefix_count)
    width=max(a.critical_points.shape[1],b.critical_points.shape[1])
    aa=F.pad(a.critical_points,(0,width-a.critical_points.shape[1]))
    bb=F.pad(b.critical_points,(0,width-b.critical_points.shape[1]))
    mask=torch.arange(width,device=aa.device)[None]<a.critical_count[:,None]
    delta=torch.where(mask,(aa-bb).abs(),torch.zeros_like(aa))
    position_ok=(delta<=FIELD_PROTOCOL['critical_position_atol']).all(-1)
    dw=max(a.monotonic_directions.shape[1],b.monotonic_directions.shape[1])
    da=F.pad(a.monotonic_directions,(0,dw-a.monotonic_directions.shape[1]))
    db=F.pad(b.monotonic_directions,(0,dw-b.monotonic_directions.shape[1]))
    dm=torch.arange(dw,device=da.device)[None]<=a.critical_count[:,None]
    direction_ok=((da==db)|~dm).all(-1)
    return (same&position_ok&direction_ok&
            ((a.depth.detach()-b.depth.detach()).abs()<=FIELD_PROTOCOL['depth_atol'])&
            ((a.total-b.total).abs()<=FIELD_PROTOCOL['tv_atol']))


def candidate_depths(g,origins,directions,*,epsilon=1e-6,max_members=1024,
                     verify=True,require_grad=True,candidate_ids=None):
    """Return depth, usable-root mask, and JSON-safe per-ray diagnostics.

    Rootless but refinement-stable rays have valid=False/convergence=True.
    Overflow never truncates members: it is an explicit unusable ray. A failed
    4/8 agreement is retried at 8/16 for the whole batch; failure remains visible.
    Forward branch agreement does not certify all continuous extrema were found.
    """
    start=time.perf_counter()
    ids,support=select_members(g,origins,directions,epsilon=epsilon,
                              max_members=max_members,candidate_ids=candidate_ids)
    with torch.enable_grad() if require_grad else torch.no_grad():
        p,member_mask=_pad_rows(g,origins,directions,ids)
        options=dict(near=FIELD_PROTOCOL['near'],far=FIELD_PROTOCOL['far'],member_mask=member_mask)
        if verify:
            with torch.no_grad(): coarse=stationary_depth_batch(*(x.detach() for x in p),subdivisions=4,**options)
        fine=stationary_depth_batch(*p,subdivisions=8,**options);levels=8
        convergence=refinement_agreement(coarse,fine) if verify else torch.ones_like(fine.valid)
        if verify and bool((~convergence&~support['overflow']).any()):
            old=fine
            fine=stationary_depth_batch(*p,subdivisions=16,**options);levels=16
            convergence=refinement_agreement(old,fine)
        convergence=convergence&~support['overflow']
        valid=fine.valid&convergence
        depth=torch.where(valid,fine.depth,torch.zeros_like(fine.depth))
    failures=fine.guard_failure_mask.detach().cpu().tolist()
    diag=dict(protocol=FIELD_PROTOCOL,epsilon=epsilon,max_members=max_members,
        has_root=fine.has_root.detach().cpu().tolist(),valid=valid.detach().cpu().tolist(),
        convergence=convergence.detach().cpu().tolist(),overflow=support['overflow'].cpu().tolist(),
        member_counts=support['member_counts'].cpu().tolist(),candidate_count=support['candidate_count'],
        omitted_TV_upper_bound=support['omitted_TV_upper_bound'].cpu().tolist(),
        bound_uses_loose_outside_index_term=support['bound_uses_loose_outside_index_term'],
        critical_count=fine.critical_count.detach().cpu().tolist(),
        prefix_count=fine.prefix_count.detach().cpu().tolist(),levels_used=levels,
        guard_failures=[[name for name,bad in zip(GUARD_NAMES,row) if bad] for row in failures],
        seconds=time.perf_counter()-start)
    return depth,valid,diag


@dataclass
class ProjectedSupportIndex:
    xmin: torch.Tensor
    xmax: torch.Tensor
    ymin: torch.Tensor
    ymax: torch.Tensor
    active: torch.Tensor
    epsilon: float

    @torch.no_grad()
    def query(self,xmin,xmax,ymin,ymax):
        if xmin>xmax or ymin>ymax: raise ValueError('invalid tile bounds')
        return (self.active&(self.xmin<=xmax)&(self.xmax>=xmin)&
                (self.ymin<=ymax)&(self.ymax>=ymin)).nonzero().flatten()


@torch.no_grad()
def build_support_index(g,camera,epsilon=1e-6):
    """Conservative full-LINE sphere projection; behind-camera points retained.

    Each Gaussian's amplitude>=epsilon ellipsoid is contained in a sphere of
    radius max(filtered_scale)*sqrt(2log(alpha/epsilon)). A line through a pixel
    can pass that sphere only within these exact perspective bounds. Spheres
    crossing z=0 get infinite bounds. No near/far or visibility heuristic.
    Rebuild after any model update; a stale index is not a valid support list.
    """
    if not 0<epsilon<1: raise ValueError('invalid epsilon')
    s2=g._scaling.double().exp().square();f2=s2+g.filter_3D.double().square()
    alpha=(g._opacity.double().sigmoid().flatten()*
           torch.sqrt(s2.prod(-1)/f2.prod(-1))*FIELD_PROTOCOL['opacity_factor'])
    if (not bool(torch.isfinite(alpha).all()) or not bool(torch.isfinite(g._xyz).all()) or
        not bool(torch.isfinite(g._rotation).all()) or bool((g._rotation.double().norm(dim=-1)==0).any())):
        raise ValueError('nonfinite model or zero quaternion')
    active=alpha>=epsilon
    radius=f2.max(-1).values.sqrt()*torch.sqrt(2*torch.log((alpha/epsilon).clamp_min(1.)))
    # rays_for_pixels uses local @ R.T. Invert that exact stored matrix instead
    # of assuming its rounded float32 entries are perfectly orthonormal.
    world_to_ray=torch.linalg.inv(camera.R.double().T)
    radius=radius*torch.linalg.matrix_norm(world_to_ray,ord=2)*(1+1e-12)
    xyz=(g._xyz.double()-camera.camera_center.double())@world_to_ray
    z=xyz[:,2];den=z.square()-radius.square();cross=den<=0
    safe=torch.where(cross,torch.ones_like(den),den)
    bounds=[]
    for coord,focal,center in ((xyz[:,0],camera.Fx,camera.Cx),(xyz[:,1],camera.Fy,camera.Cy)):
        delta=radius*torch.sqrt((coord.square()+z.square()-radius.square()).clamp_min(0))
        lo=(coord*z-delta)/safe*focal+center
        hi=(coord*z+delta)/safe*focal+center
        # Tiny outward padding covers arithmetic rounding; no support is removed.
        lo=torch.where(cross,torch.full_like(lo,-torch.inf),lo-1e-4)
        hi=torch.where(cross,torch.full_like(hi,torch.inf),hi+1e-4)
        bounds.extend((lo,hi))
    if not bool(torch.isfinite(radius).all()): raise ValueError('nonfinite model scale')
    return ProjectedSupportIndex(*bounds,active,epsilon)
