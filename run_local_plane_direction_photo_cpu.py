"""One fixed scan110 CPU direction and two fixed photo witnesses; no optimizer."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from local_plane_training_cpu import CAPTURE_SHA, sha256, member_plane
from run_local_plane_training_cpu import camera_matrix, camera_roster, verify_reference_camera
from local_plane_direction_photo_cpu import (DT, FD_H, GRID_EPS, IMAGE_SHA, PATCH,
    RESULT_SHA, TEXTURE_EPS, Z_ALL, bilinear_np, direction_from_frozen,
    load_gray, photo_cost_torch, photo_projection_np, save_direction)


def evaluate_R_eps(direction, ref, eps):
    from local_plane_direction_photo_cpu import planes_torch, project_torch_batch
    occ=direction["occ"]
    inds=np.array([o["index"] for o in occ],dtype=int)
    xyz=torch.tensor(direction["xyz"][inds]+eps*direction["dmu"][inds],dtype=DT)
    quat=torch.tensor(direction["quat"][inds]+eps*direction["dq"][inds],dtype=DT)
    pi=planes_torch(xyz,quat,np.array([o["axis"] for o in occ]),ref["R"],ref["t"])
    rays=torch.tensor(np.stack([o["rays"] for o in occ]),dtype=DT)
    base=torch.tensor(np.stack([o["base"] for o in occ]),dtype=DT)
    trans=torch.tensor(np.stack([o["trans"] for o in occ]),dtype=DT)
    target=torch.tensor(np.stack([o["target"] for o in occ]),dtype=DT)
    pred=project_torch_batch(pi,rays,base,trans)
    D=torch.mean(torch.sum((pred-target)**2,dim=2),dim=1)
    m=torch.tensor([o["m"] for o in occ],dtype=DT)
    return float(torch.sum(m*D)/Z_ALL)


def direction_permutation_check(cap,saved,ref,dst,direction):
    perm=np.random.default_rng(20261101).permutation(len(cap["unique_id"]))
    copy=dict(cap)
    for key in ("unique_id","center","raw_quaternion","raw_scale"):
        copy[key]=cap[key][perm]
    alt=direction_from_frozen(copy,saved,ref,dst)
    map_alt={int(g):i for i,g in enumerate(alt["ids"])}
    errors=[]
    for i,g in enumerate(direction["ids"]):
        j=map_alt[int(g)]
        errors.extend((np.max(np.abs(direction["dmu"][i]-alt["dmu"][j])),
                       np.max(np.abs(direction["dq"][i]-alt["dq"][j]))))
    max_error=float(max(errors))
    if max_error>1e-10 or not math.isclose(direction["R"],alt["R"],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError("shared_id_permutation_changed_direction")
    return max_error


def frozen_photo_rows(direction, ref, dst, source, target):
    rows=[];eligible=[];patches=[];corners=[]
    sources={}
    for p in range(32):
        center=next(x["center"] for x in direction["occ"] if x["p"]==p)
        uv=center+PATCH
        try:
            src,_,_,_=bilinear_np(source,uv)
            if float(np.sum((src-src.mean())**2))<=TEXTURE_EPS:
                raise ValueError("low_source_texture")
            sources[p]=(uv,src,None)
        except ValueError as exc:
            sources[p]=(uv,None,str(exc))
    for o in direction["occ"]:
        p=o["p"];uv,src,src_reason=sources[p]
        row=dict(window=p,id=o["gid"],mass=o["m"],status="missing")
        reason=src_reason
        if reason is None:
            try:
                ix=o["index"]
                pi,_=member_plane(direction["xyz"][ix],direction["quat"][ix],o["axis"],ref["R"],ref["t"])
                target_xy=photo_projection_np(pi,ref,dst,uv)
                tgt,ix0,iy0,corner=bilinear_np(target,target_xy)
                frac=target_xy-np.floor(target_xy)
                if np.min(np.minimum(frac,1-frac))<=GRID_EPS:
                    raise ValueError("target_grid_boundary")
                if float(np.sum((tgt-tgt.mean())**2))<=TEXTURE_EPS:
                    raise ValueError("low_target_texture")
                row["status"]="eligible"
                row["initial_cost"]=float((1-np.dot(src-src.mean(),tgt-tgt.mean())/
                    np.sqrt(np.sum((src-src.mean())**2)*np.sum((tgt-tgt.mean())**2)))/2)
                if not np.isfinite(row["initial_cost"]) or not (-1e-12<=row["initial_cost"]<=1+1e-12):
                    raise ValueError("nonfinite_or_out_of_range_cost")
                photo=dict(o,xyz=direction["xyz"][ix],quat=direction["quat"][ix],
                    dmu=direction["dmu"][ix],dq=direction["dq"][ix],uv=uv,ix=ix0,iy=iy0)
                eligible.append(photo);patches.append(src);corners.append(corner)
            except ValueError as exc:
                reason=str(exc)
                row["status"]="missing"
                row.pop("initial_cost",None)
        if row["status"]=="missing":row["reason"]=reason
        rows.append(row)
    return rows,eligible,patches,corners


def evaluate_photo(direction,ref,dst,source,target,view):
    rows,eligible,patches,corners=frozen_photo_rows(direction,ref,dst,source,target)
    if not eligible:
        return dict(view=view,rows=rows,covered_mass=0.,known_cost_contribution=None,
                    known_directional_derivative=None,status="no_eligible_members")
    eps=torch.zeros(len(eligible),dtype=DT,requires_grad=True)
    cost=photo_cost_torch(eps,eligible,ref,dst,patches,corners)
    deriv=torch.autograd.grad(cost.sum(),eps)[0].detach().numpy()
    fd_safe=np.ones(len(eligible),dtype=bool)
    fd_reason=[None]*len(eligible)
    for i,r in enumerate(eligible):
        for sign in (-1,1):
            try:
                pi,_=member_plane(r["xyz"]+sign*FD_H*r["dmu"],
                    r["quat"]+sign*FD_H*r["dq"],r["axis"],ref["R"],ref["t"])
                # Do not reselect the initial pixel stencil or reject OOB at ±h.
                photo_projection_np(pi,ref,dst,r["uv"])
            except ValueError as exc:
                fd_safe[i]=False
                fd_reason[i]=str(exc)
                break
    safe=np.flatnonzero(fd_safe)
    fd=np.full(len(eligible),np.nan)
    with torch.no_grad():
        if len(safe):
            selected=[eligible[i] for i in safe]
            source_selected=[patches[i] for i in safe]
            corners_selected=[corners[i] for i in safe]
            fd[safe]=(photo_cost_torch(torch.full((len(safe),),FD_H,dtype=DT),selected,ref,dst,source_selected,corners_selected)-
                photo_cost_torch(torch.full((len(safe),),-FD_H,dtype=DT),selected,ref,dst,source_selected,corners_selected)).numpy()/(2*FD_H)
    actual=cost.detach().numpy()
    tol=1e-7+1e-4*np.maximum(abs(deriv),abs(fd))
    if not np.isfinite(deriv).all() or not np.isfinite(fd[safe]).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite_photo_derivative")
    counter=0
    for row in rows:
        if row["status"]!="eligible":continue
        row.update(ad_derivative=float(deriv[counter]),fd_derivative=float(fd[counter]) if fd_safe[counter] else None,
            fd_error=float(abs(deriv[counter]-fd[counter])) if fd_safe[counter] else None,
            fd_tolerance=float(tol[counter]) if fd_safe[counter] else None,
            fd_pass=bool(fd_safe[counter] and abs(deriv[counter]-fd[counter])<=tol[counter]),
            fd_failure_reason=fd_reason[counter],cost_torch=float(actual[counter]))
        counter+=1
    mass=np.array([x["m"] for x in eligible])
    covered=float(np.sum(mass)/Z_ALL)
    cost_contrib=float(np.sum(mass*actual)/Z_ALL)
    derivative=float(np.sum(mass*deriv)/Z_ALL)
    per_window=[]
    for p in range(32):
        chosen=[r for r in rows if r["window"]==p and r["status"]=="eligible"]
        per_window.append(dict(window=p,covered_mass=sum(r["mass"] for r in chosen)/Z_ALL,
            known_cost_contribution=sum(r["mass"]*r["cost_torch"] for r in chosen)/Z_ALL,
            known_directional_derivative=sum(r["mass"]*r["ad_derivative"] for r in chosen)/Z_ALL,
            eligible_count=len(chosen)))
    return dict(view=view,rows=rows,per_window=per_window,covered_mass=covered,
        missing_mass_fraction=1-covered,known_cost_contribution=cost_contrib,
        known_directional_derivative=derivative,
        positive_derivative_mass_fraction=float(np.sum(mass[deriv>tol])/Z_ALL),
        negative_derivative_mass_fraction=float(np.sum(mass[deriv< -tol])/Z_ALL),
        near_zero_derivative_mass_fraction=float(np.sum(mass[abs(deriv)<=tol])/Z_ALL),
        fd_pass_count=int(np.count_nonzero(fd_safe & (abs(deriv-fd)<=tol))),fd_total=len(eligible),
        reasons=dict(Counter(r.get("reason","eligible") for r in rows)),status="measured")


def run(capture_path,results_path,camera_dir,image_dir,outdir):
    started=time.perf_counter()
    capture_path,results_path,camera_dir,image_dir,outdir=map(Path,(capture_path,results_path,camera_dir,image_dir,outdir))
    if outdir.exists():raise ValueError("output_directory_exists")
    if sha256(capture_path)!=CAPTURE_SHA or sha256(results_path)!=RESULT_SHA:
        raise ValueError("frozen_input_sha_mismatch")
    outdir.mkdir(parents=True)
    with np.load(capture_path,allow_pickle=False) as z:cap={k:z[k].copy() for k in z.files}
    saved=json.loads(results_path.read_text(encoding="utf-8"))
    if saved["protocol"]!="local_plane_training_cpu_v1" or saved["valid_count"]!=32:
        raise ValueError("frozen_result_mismatch")
    for name,sha in saved["camera_sha256"].items():
        if sha256(camera_dir/name)!=sha:raise ValueError("colmap_sha_mismatch")
    small=camera_roster(camera_dir,777,581)
    verify_reference_camera(cap,small["0049"])
    raw=camera_roster(camera_dir,1554,1162)
    ref=raw["0049"]
    direction=direction_from_frozen(cap,saved,ref,raw["0050"])
    perm_error=direction_permutation_check(cap,saved,ref,raw["0050"],direction)
    plus=evaluate_R_eps(direction,ref,FD_H)
    minus=evaluate_R_eps(direction,ref,-FD_H)
    fd_R=(plus-minus)/(2*FD_H)
    tol_R=1e-7+1e-4*max(abs(fd_R),abs(direction["gR_dot_d"]))
    if abs(fd_R-direction["gR_dot_d"])>tol_R or fd_R>=0 or plus>=direction["R"]:
        raise ValueError("R_direction_FD_failed")
    direction_hash=save_direction(direction,outdir/"direction.npz")
    frozen=dict(protocol="local_plane_direction_photo_cpu_v1",base_commit="6ce1b1b739247f5dbfb3eb68fc0a318fa0ba580d",
        capture_sha256=CAPTURE_SHA,results_sha256=RESULT_SHA,direction_sha256=direction_hash,
        ell2=direction["ell2"],S=direction["S"],R=direction["R"],
        gR_dot_d=direction["gR_dot_d"],R_FD=fd_R,R_FD_tolerance=tol_R,
        R_plus=plus,R_minus=minus,permuted_id_max_error=perm_error,
        positive_occurrences=direction["positive_occurrences"],unique_positive_ids=direction["unique_positive_ids"],
        target_color_views=["0012","0058"],Z_all=Z_ALL)
    (outdir/"frozen_direction.json").write_text(json.dumps(frozen,indent=2)+"\n",encoding="utf-8")
    # The direction and its hash are durable before any target image color is opened.
    images={}
    for view in ("0049","0012","0058"):
        path=image_dir/(view+".png")
        if sha256(path)!=IMAGE_SHA[view]:raise ValueError("image_sha_mismatch_"+view)
        images[view]=load_gray(path)
    views={v:evaluate_photo(direction,ref,raw[v],images["0049"],images[v],v) for v in ("0012","0058")}
    status="indeterminate"
    if all(x["status"]=="measured" and x["fd_pass_count"]==x["fd_total"] for x in views.values()):
        ds=[views[v]["known_directional_derivative"] for v in ("0012","0058")]
        tol=1e-7+1e-4*max(map(abs,ds))
        if all(d>tol for d in ds):status="both_fixed_evaluable_subsets_opposed"
        elif ds[0]>tol and ds[1]<-tol or ds[1]>tol and ds[0]<-tol:status="view_conflict"
        elif all(d< -tol for d in ds):status="both_fixed_evaluable_subsets_support"
    summary=dict(frozen=frozen,image_sha256=IMAGE_SHA,views=views,status=status,
        unseen_derivative="unknown_for_all_missing_mass",occlusion="unverified",
        elapsed_sec=time.perf_counter()-started)
    (outdir/"result.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("capture","results","camera-dir","image-dir","outdir"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    print(json.dumps({"status":run(a.capture,a.results,a.camera_dir,a.image_dir,a.outdir)["status"]}))
