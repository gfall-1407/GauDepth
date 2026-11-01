"""Fixed shared-parameter R direction and signed-ZNCC member-plane check, CPU only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_plane_training_cpu import ANCHOR_OFFSETS, geometry, member_plane, rotmat_torch, sha256
from run_local_plane_training_cpu import camera_matrix, camera_roster, verify_reference_camera, window_members

RESULT_SHA = "df05b6b018c7b38fc71d1185d119a6dbb58d665f09cd6de8998bb9b1b6cfa8dc"
IMAGE_SHA = {"0049":"0971df06cc6d8ff431d45e841b361bd141c5258d2bc908432212ed2fb10f4a36",
             "0012":"d9807bacab63b255cc4a7fff0a33d31b6825a4aa78083fdab20650e327e5e48a",
             "0058":"22bc7fe2995b3d7b2a66bf9d81a0c9c1ec8c87fc4bf65f6a87e0c31cd652f8dc"}
Z_ALL = 27.96946798809682
GRID_EPS = 1e-6
TEXTURE_EPS = 1e-10
FD_H = 1e-6
PATCH = np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3)], dtype=np.float64)
DT = torch.float64


def batch_rotmat(raw):
    q = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    w,x,y,z = q.unbind(1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=1).reshape(-1,3,3)


def planes_torch(mu, quat, axes, Rref, tref):
    rotation=batch_rotmat(quat)
    n_world=rotation[torch.arange(len(axes)),:,torch.as_tensor(axes,dtype=torch.long)]
    R=torch.as_tensor(Rref,dtype=DT)
    t=torch.as_tensor(tref,dtype=DT)
    n=n_world@R.T
    center=mu@R.T+t
    return n/torch.sum(n*center,dim=1,keepdim=True)


def project_torch_batch(pi, rays, base, trans):
    t=trans[None,None,:] if trans.ndim==1 else trans[:,None,:]
    v=base+torch.sum(pi[:,None,:]*rays,dim=2)[:,:,None]*t
    return v[:,:,:2]/v[:,:,2,None]


def direction_from_frozen(cap, saved, ref, dst):
    ids_all=np.asarray(cap["unique_id"],dtype=np.int64)
    id_to_row={int(g):i for i,g in enumerate(ids_all)}
    if len(id_to_row)!=len(ids_all):raise ValueError("duplicate_global_ids")
    Rrel=dst["R"]@ref["R"].T
    trel=dst["t"]-Rrel@ref["t"]
    Kr,Ks=camera_matrix(ref),camera_matrix(dst)
    occ=[]
    for p,row in enumerate(saved["windows"]):
        if row["index"]!=p or row["status"]!="valid":raise ValueError("frozen_window_invalid")
        ids,mass=window_members(cap,p,id_to_row)
        if ids!=row["ids"] or not np.allclose(mass,row["mass"],rtol=0,atol=1e-14):
            raise ValueError("frozen_member_or_mass_mismatch")
        xy=2*np.asarray(cap["xy"][9*p+4],dtype=np.float64)+.5
        if not np.array_equal(xy,np.asarray(row["raw_center"])):raise ValueError("frozen_pixel_mismatch")
        geom=geometry(Kr,Ks,Rrel,trel,xy+ANCHOR_OFFSETS)
        latent=np.asarray(row["best"]["latent"],dtype=np.float64)
        latent_pixels=np.stack([project_torch_batch(torch.as_tensor(x[None],dtype=DT),
            torch.as_tensor(geom[0][None],dtype=DT),torch.as_tensor(geom[1][None],dtype=DT),
            torch.as_tensor(geom[2],dtype=DT))[0].numpy() for x in latent])
        for j,(gid,m) in enumerate(zip(ids,mass)):
            if m<=0:continue
            index=id_to_row[gid]
            axis=int(row["axes"][j])
            if axis!=int(np.argmin(cap["raw_scale"][index])):raise ValueError("axis_changed")
            a=int(row["best"]["assignment"][j])
            occ.append(dict(p=p,gid=gid,index=index,axis=axis,m=float(m),
                            rays=geom[0],base=geom[1],trans=geom[2],
                            target=latent_pixels[a],center=xy))
    if not math.isclose(math.fsum(x["m"] for x in occ),Z_ALL,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError("global_mass_mismatch")
    indices=np.array([x["index"] for x in occ],dtype=np.int64)
    axes=np.array([x["axis"] for x in occ],dtype=np.int64)
    xyz=torch.tensor(cap["center"],dtype=DT,requires_grad=True)
    quat=torch.tensor(cap["raw_quaternion"],dtype=DT,requires_grad=True)
    pi=planes_torch(xyz[indices],quat[indices],axes,ref["R"],ref["t"])
    rays=torch.tensor(np.stack([x["rays"] for x in occ]),dtype=DT)
    base=torch.tensor(np.stack([x["base"] for x in occ]),dtype=DT)
    trans=torch.tensor(np.stack([x["trans"] for x in occ]),dtype=DT)
    target=torch.tensor(np.stack([x["target"] for x in occ]),dtype=DT)
    pred=project_torch_batch(pi,rays,base,trans)
    D=torch.mean(torch.sum((pred-target)**2,dim=2),dim=1)
    mass=torch.tensor([x["m"] for x in occ],dtype=DT)
    R=torch.sum(mass*D)/Z_ALL
    gx,gq=torch.autograd.grad(R,(xyz,quat))
    if not torch.isfinite(R) or not torch.isfinite(gx).all() or not torch.isfinite(gq).all():
        raise ValueError("nonfinite_R_or_gradient")
    if not math.isclose(float(R),saved["R"],rel_tol=1e-9,abs_tol=1e-9):
        raise ValueError("R_does_not_match_frozen_result")
    M=np.zeros(len(ids_all),dtype=np.float64)
    np.add.at(M,indices,mass.numpy())
    center=ref["center"]
    ell2=float(np.sum(M*np.sum((xyz.detach().numpy()-center)**2,axis=1))/sum(M))
    q0=quat.detach().numpy()
    r2=np.sum(q0*q0,axis=1)
    if not np.isfinite(ell2) or ell2<=0 or np.any(r2[M>0]<=0):raise ValueError("invalid_direction_scale")
    qhat=q0/np.sqrt(r2)[:,None]
    gq0=gq.detach().numpy()
    vmu=-ell2*gx.detach().numpy()
    vq=-(r2[:,None]/4)*(gq0-qhat*np.sum(qhat*gq0,axis=1)[:,None])
    vmu[M==0]=0;vq[M==0]=0
    norm=np.sqrt(np.sum(vmu*vmu,axis=1)/ell2+4*np.sum(vq*vq,axis=1)/r2)
    S=float(np.max(norm[M>0]))
    if not np.isfinite(S) or S<=0:raise ValueError("zero_or_nonfinite_direction")
    dmu,dq=vmu/S,vq/S
    dot=float(np.sum(gx.detach().numpy()*dmu)+np.sum(gq0*dq))
    if not np.isfinite(dot) or dot>=0:raise ValueError("not_a_descent_direction")
    return dict(occ=occ,ids=ids_all,xyz=xyz.detach().numpy(),quat=q0,axes=axes,
                M=M,dmu=dmu,dq=dq,ell2=ell2,S=S,gR_dot_d=dot,R=float(R),
                positive_occurrences=len(occ),unique_positive_ids=int(np.count_nonzero(M)))


def save_direction(direction,path):
    path=Path(path)
    if path.exists():raise ValueError("direction_path_exists")
    np.savez_compressed(path,ids=direction["ids"],mass_by_id=direction["M"],
        center=direction["xyz"],raw_quaternion=direction["quat"],
        d_center=direction["dmu"],d_raw_quaternion=direction["dq"])
    return sha256(path)


def load_gray(path):
    rgb=np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)/255.
    if rgb.shape!=(1162,1554,3):raise ValueError("unexpected_image_size")
    return .299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]


def bilinear_np(image,xy):
    H,W=image.shape
    if not np.isfinite(xy).all() or np.any(xy[:,0]<0) or np.any(xy[:,0]>=W-1) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=H-1):
        raise ValueError("out_of_bounds")
    ix=np.floor(xy[:,0]).astype(int);iy=np.floor(xy[:,1]).astype(int)
    dx=xy[:,0]-ix;dy=xy[:,1]-iy
    corners=np.stack((image[iy,ix],image[iy,ix+1],image[iy+1,ix],image[iy+1,ix+1]),axis=1)
    value=(1-dx)*(1-dy)*corners[:,0]+dx*(1-dy)*corners[:,1]+(1-dx)*dy*corners[:,2]+dx*dy*corners[:,3]
    return value,ix,iy,corners


def photo_projection_np(pi,ref,dst,uv):
    Kr,Ks=camera_matrix(ref),camera_matrix(dst)
    rays=(np.linalg.inv(Kr)@np.c_[uv,np.ones(len(uv))].T).T
    denom=rays@pi
    if not np.isfinite(denom).all() or np.any(np.abs(denom)<=1e-10*np.maximum(1.,np.linalg.norm(rays,axis=1))):
        raise ValueError("plane_intersection_pole")
    z=1/denom
    if np.any(z<=0):raise ValueError("nonpositive_reference_depth")
    R=dst["R"]@ref["R"].T;t=dst["t"]-R@ref["t"]
    X=(R@((rays*z[:,None]).T)).T+t
    if not np.isfinite(X).all() or np.any(X[:,2]<=0):raise ValueError("nonpositive_target_depth")
    pix=(Ks@X.T).T
    xy=pix[:,:2]/pix[:,2,None]
    if not np.isfinite(xy).all():raise ValueError("nonfinite_projection")
    return xy


def photo_cost_torch(eps,rows,ref,dst,source_patch,target_corners):
    indices=np.array([r["index"] for r in rows],dtype=int)
    axes=np.array([r["axis"] for r in rows],dtype=int)
    xyz=torch.tensor(np.stack([r["xyz"] for r in rows]),dtype=DT)+eps[:,None]*torch.tensor(np.stack([r["dmu"] for r in rows]),dtype=DT)
    quat=torch.tensor(np.stack([r["quat"] for r in rows]),dtype=DT)+eps[:,None]*torch.tensor(np.stack([r["dq"] for r in rows]),dtype=DT)
    pi=planes_torch(xyz,quat,axes,ref["R"],ref["t"])
    uv=torch.tensor(np.stack([r["uv"] for r in rows]),dtype=DT)
    Kr=torch.tensor(camera_matrix(ref),dtype=DT)
    Ks=torch.tensor(camera_matrix(dst),dtype=DT)
    rays=(torch.linalg.inv(Kr)@torch.cat((uv,torch.ones((len(rows),25,1),dtype=DT)),dim=2).transpose(1,2)).transpose(1,2)
    z=1/torch.sum(rays*pi[:,None,:],dim=2)
    R=torch.tensor(dst["R"]@ref["R"].T,dtype=DT)
    t=torch.tensor(dst["t"]-dst["R"]@ref["R"].T@ref["t"],dtype=DT)
    X=(rays*z[:,:,None])@R.T+t
    pix=X@Ks.T
    xy=pix[:,:,:2]/pix[:,:,2,None]
    ix=torch.tensor(np.stack([r["ix"] for r in rows]),dtype=DT)
    iy=torch.tensor(np.stack([r["iy"] for r in rows]),dtype=DT)
    dx,dy=xy[:,:,0]-ix,xy[:,:,1]-iy
    c=torch.tensor(np.stack(target_corners),dtype=DT)
    tgt=(1-dx)*(1-dy)*c[:,:,0]+dx*(1-dy)*c[:,:,1]+(1-dx)*dy*c[:,:,2]+dx*dy*c[:,:,3]
    src=torch.tensor(np.stack(source_patch),dtype=DT)
    src=src-src.mean(dim=1,keepdim=True)
    tgt=tgt-tgt.mean(dim=1,keepdim=True)
    corr=torch.sum(src*tgt,dim=1)/torch.sqrt(torch.sum(src*src,dim=1)*torch.sum(tgt*tgt,dim=1))
    return (1-corr)/2
