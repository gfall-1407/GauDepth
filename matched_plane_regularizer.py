"""Frozen 32-window local plane data residual; shared-ID Torch gather, no refitting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from local_plane_training_cpu import ANCHOR_OFFSETS, CAPTURE_SHA, geometry, project, sha256
from run_local_plane_training_cpu import camera_matrix, camera_roster, verify_reference_camera, window_members


RESULT_SHA = "df05b6b018c7b38fc71d1185d119a6dbb58d665f09cd6de8998bb9b1b6cfa8dc"
EXPECTED_PLY_SHA = "733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7"
EXPECTED_PLY_BYTES = 36010340
EXPECTED_PLY_ROWS = 142892
EXPECTED_R0 = 140.34713758283957
Z_ALL = 27.96946798809682


def batch_rotation(raw):
    norm=torch.linalg.vector_norm(raw,dim=1,keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm<=0).any()):
        raise ValueError("invalid_positive_mass_raw_quaternion")
    w,x,y,z=(raw/norm).unbind(1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=1).reshape(-1,3,3)


class FrozenPlaneRegularizer:
    def __init__(self,capture_path,results_path,camera_dir,device="cpu"):
        capture_path,results_path,camera_dir=map(Path,(capture_path,results_path,camera_dir))
        if sha256(capture_path)!=CAPTURE_SHA or sha256(results_path)!=RESULT_SHA:
            raise ValueError("frozen_input_sha_mismatch")
        saved=json.loads(results_path.read_text(encoding="utf-8"))
        if (saved["valid_count"]!=32 or saved["R"]!=EXPECTED_R0 or
                saved["protocol"]!="local_plane_training_cpu_v1"):
            raise ValueError("frozen_result_identity_mismatch")
        for name,expected in saved["camera_sha256"].items():
            if sha256(camera_dir/name)!=expected:
                raise ValueError("frozen_camera_sha_mismatch")
        with np.load(capture_path,allow_pickle=False) as z:cap={k:z[k].copy() for k in z.files}
        small=camera_roster(camera_dir,777,581)
        verify_reference_camera(cap,small["0049"])
        raw=camera_roster(camera_dir,1554,1162)
        ref,dst=raw["0049"],raw["0050"]
        Rrel=dst["R"]@ref["R"].T
        trel=dst["t"]-Rrel@ref["t"]
        Kr,Ks=camera_matrix(ref),camera_matrix(dst)
        unique=list(map(int,cap["unique_id"]))
        if len(unique)!=len(set(unique)) or len(unique)!=1836:
            raise ValueError("unexpected_unique_ID_table")
        id_rows={value:i for i,value in enumerate(unique)}
        rows=[]
        for p,row in enumerate(saved["windows"]):
            ids,mass=window_members(cap,p,id_rows)
            if (row["index"]!=p or row["status"]!="valid" or ids!=row["ids"] or
                    not np.allclose(mass,row["mass"],atol=1e-14,rtol=0)):
                raise ValueError("frozen_window_changed")
            center=2*np.asarray(cap["xy"][9*p+4],dtype=np.float64)+.5
            if not np.array_equal(center,np.asarray(row["raw_center"])):
                raise ValueError("window_pixel_changed")
            geom=geometry(Kr,Ks,Rrel,trel,center+ANCHOR_OFFSETS)
            latent=np.asarray(row["best"]["latent"],dtype=np.float64)
            for j,(gid,m) in enumerate(zip(ids,mass)):
                if m<=0:continue
                axis=int(row["axes"][j])
                if axis!=int(np.argmin(cap["raw_scale"][id_rows[gid]])):
                    raise ValueError("short_axis_changed")
                assignment=int(row["best"]["assignment"][j])
                rows.append((gid,axis,float(m),geom[0],geom[1],geom[2],
                             project(latent[assignment],geom)[0]))
        if len(rows)!=2420 or len(set(r[0] for r in rows))!=1836 or not math.isclose(
                math.fsum(r[2] for r in rows),Z_ALL,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("frozen_occurrence_identity_mismatch")
        self.device=torch.device(device)
        def tensor(x,dtype=torch.float64):return torch.as_tensor(np.asarray(x),dtype=dtype,device=self.device)
        self.ids=tensor([r[0] for r in rows],torch.long)
        self.axes=tensor([r[1] for r in rows],torch.long)
        self.mass=tensor([r[2] for r in rows])
        self.rays=tensor(np.stack([r[3] for r in rows]))
        self.base=tensor(np.stack([r[4] for r in rows]))
        self.trans=tensor(np.stack([r[5] for r in rows]))
        self.target=tensor(np.stack([r[6] for r in rows]))
        self.ref_R=tensor(ref["R"])
        self.ref_t=tensor(ref["t"])
        self.count=len(rows)

    def __call__(self,xyz,raw_quaternion):
        if (xyz.ndim!=2 or raw_quaternion.ndim!=2 or xyz.shape[1]!=3 or
                raw_quaternion.shape[1]!=4 or xyz.shape[0]!=raw_quaternion.shape[0] or
                xyz.shape[0]!=EXPECTED_PLY_ROWS):
            raise ValueError("wrong_full_model_parameter_shape")
        if xyz.device!=self.device or raw_quaternion.device!=self.device:
            raise ValueError("wrong_regularizer_device")
        mu=xyz[self.ids].to(torch.float64)
        quat=raw_quaternion[self.ids].to(torch.float64)
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(quat).all()):
            raise ValueError("nonfinite_positive_mass_parameter")
        rotation=batch_rotation(quat)
        n_world=rotation[torch.arange(self.count,device=self.device),:,self.axes]
        n=n_world@self.ref_R.T
        center=mu@self.ref_R.T+self.ref_t
        plane_d=torch.sum(n*center,dim=1)
        if (not bool(torch.isfinite(plane_d).all()) or
                bool((plane_d.abs()<=1e-10*torch.maximum(torch.ones_like(plane_d),torch.linalg.vector_norm(center,dim=1))).any())):
            raise ValueError("positive_mass_member_plane_pole")
        pi=n/plane_d[:,None]
        v=self.base+torch.sum(pi[:,None,:]*self.rays,dim=2)[:,:,None]*self.trans[:,None,:]
        den=v[:,:,2]
        if (not bool(torch.isfinite(v).all()) or
                bool((den.abs()<=1e-10*torch.maximum(torch.ones_like(den),torch.linalg.vector_norm(v,dim=2))).any())):
            raise ValueError("positive_mass_homography_pole")
        pred=v[:,:,:2]/den[:,:,None]
        D=torch.mean(torch.sum((pred-self.target)**2,dim=2),dim=1)
        R=torch.sum(self.mass*D)/Z_ALL
        if not bool(torch.isfinite(R)):
            raise ValueError("nonfinite_regularizer")
        return R
