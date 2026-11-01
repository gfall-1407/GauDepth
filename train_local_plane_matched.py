"""Controlled 1000-update host A/B from one 7k PLY; execution requires reviewed GPU approval."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from matched_plane_regularizer import (EXPECTED_PLY_BYTES, EXPECTED_PLY_ROWS,
    EXPECTED_PLY_SHA, EXPECTED_R0, FrozenPlaneRegularizer)
from scene import GaussianModel, Scene
from utils.graphics_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import L1_loss_appearance, PatchMatch, l1_loss, ssim
from local_plane_training_cpu import sha256


START_VIRTUAL = 7001
UPDATES = 1000
SAVED_UPDATES = (0, 500, 1000)
MAX_ACTIVE_SECONDS = 90*60
EXTENSION_PACKAGES=("diff_gaussian_rasterization","warp_patch_ncc",
                    "simple_knn","fused_ssim")
EXTENSION_SOURCE_DIRS=("submodules/diff-gaussian-rasterization",
                       "submodules/warp-patch-ncc","submodules/simple-knn")
EXTENSION_SOURCE_SUFFIXES={".cu",".cpp",".cc",".cuh",".h",".hpp",".py",".inl"}


def extension_sources():
    root=Path(__file__).parent
    found={}
    for directory in EXTENSION_SOURCE_DIRS:
        base=root/directory
        if not base.is_dir():raise ValueError("extension_source_directory_missing:"+directory)
        items=[p for p in base.rglob("*") if p.is_file() and p.suffix in EXTENSION_SOURCE_SUFFIXES]
        if not items:raise ValueError("extension_source_directory_empty:"+directory)
        for path in items:
            found[path.relative_to(root).as_posix()]=sha256(path)
    return dict(sorted(found.items()))


def loaded_extension_modules():
    for package in EXTENSION_PACKAGES:importlib.import_module(package)
    found={}
    for name,module in tuple(sys.modules.items()):
        if not any(name==p or name.startswith(p+".") for p in EXTENSION_PACKAGES):continue
        file=getattr(module,"__file__",None)
        if file and any(file.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            path=Path(file).resolve()
            found[name]=dict(path=str(path),sha256=sha256(path))
    for package in EXTENSION_PACKAGES:
        if not any(n==package or n.startswith(package+".") for n in found):
            raise ValueError("compiled_extension_not_loaded:"+package)
    return dict(sorted(found.items()))


def verify_build_identity(expected_head,build_manifest_path):
    current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    if current!=expected_head:raise ValueError("reviewed_git_head_mismatch")
    build=json.loads(Path(build_manifest_path).read_text(encoding="utf-8"))
    if build.get("reviewed_head")!=expected_head:
        raise ValueError("extension_manifest_identity_mismatch")
    actual=loaded_extension_modules()
    if build.get("modules")!=actual:raise ValueError("extension_binary_identity_mismatch")
    sources=extension_sources()
    if build.get("sources")!=sources:
        raise ValueError("extension_source_manifest_incomplete")
    provenance=build.get("external_build_provenance",{}).get("fused_ssim")
    if not isinstance(provenance,dict) or not provenance.get("build_source") or not provenance.get("build_evidence_path") or not provenance.get("build_evidence_sha256"):
        raise ValueError("fused_ssim_build_provenance_unverified")
    evidence=Path(provenance["build_evidence_path"]).resolve()
    if not evidence.is_file() or sha256(evidence)!=provenance["build_evidence_sha256"]:
        raise ValueError("fused_ssim_build_evidence_mismatch")
    return dict(reviewed_head=current,modules=actual,sources=sources,
                external_build_provenance=build["external_build_provenance"])


def require_empty_output(path):
    path=Path(path)
    if path.exists():
        raise ValueError("output_root_already_exists")
    if not path.parent.is_dir():
        raise ValueError("output_parent_missing")


def verify_original_ply(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=EXPECTED_PLY_BYTES or sha256(path)!=EXPECTED_PLY_SHA:
        raise ValueError("wrong_original_7k_ply")
    from plyfile import PlyData
    if len(PlyData.read(str(path))["vertex"])!=EXPECTED_PLY_ROWS:
        raise ValueError("wrong_original_7k_vertex_count")


def camera_signature(cameras):
    return [(c.image_name,tuple(map(int,c.nearest_id)),int(c.uid)) for c in cameras]


def fixed_schedule(cameras,seed=0,updates=UPDATES):
    rng=random.Random(seed)
    stack=[];out=[]
    for _ in range(updates):
        if not stack:stack=list(range(len(cameras)))
        index=stack.pop(rng.randint(0,len(stack)-1))
        near=cameras[index].nearest_id
        neighbor=None if len(near)==0 else int(rng.sample(near,1)[0])
        out.append((index,neighbor))
    return out


def snapshot_hash(gaussians):
    h=hashlib.sha256()
    for key in ("_xyz","_features_dc","_features_rest","_scaling","_rotation",
                "_opacity","filter_3D","_appearance_embeddings"):
        value=getattr(gaussians,key)
        if value is None:raise ValueError("missing_initial_parameter_"+key)
        h.update(key.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    h.update(str(gaussians.active_sh_degree).encode())
    return h.hexdigest()


FROZEN_OPT=dict(iterations=30000,position_lr_init=.00016,position_lr_final=.0000016,
    position_lr_delay_mult=.01,position_lr_max_steps=30000,feature_lr=.0025,
    opacity_lr=.05,scaling_lr=.005,rotation_lr=.001,
    appearance_embeddings_lr=.001,appearance_network_lr=.001,pgsr_appearance_lr=.001,
    gs_appearance_lr_init=.01,gs_appearance_lr_final=.001,gs_appearance_lr_delay_steps=0,
    gs_appearance_lr_delay_mult=0.,percent_dense=.01,lambda_dssim=.2,
    lambda_depth_normal=.05,densification_interval=100,opacity_reset_interval=3000,
    densify_from_iter=500,densify_until_iter=15000,regularization_from_iter=7000,
    densify_grad_threshold=.0002,lambda_multi_view_geo=.02,lambda_multi_view_ncc=.3,
    multi_view_patch_size=3,multi_view_pixel_noise_th=1.)
FROZEN_DATASET=dict(sh_degree=3,images="images",dataset="",resolution=2,
    white_background=False,data_device="cuda",eval=False,use_decoupled_appearance=3,
    disable_filter3D=False,kernel_size=0.,depth_ratio=0.,multi_view_num=8,
    multi_view_max_angle=30,multi_view_min_dis=.01,multi_view_max_dis=1.5)
FROZEN_PIPE=dict(convert_SHs_python=False,compute_cov3D_python=False,debug=False)


def verify_frozen_configuration(dataset,opt,pipe):
    for label,actual,frozen in (("optimization",vars(opt),FROZEN_OPT),
            ("dataset",vars(dataset),FROZEN_DATASET),("pipeline",vars(pipe),FROZEN_PIPE)):
        checked={k:v for k,v in actual.items() if k in frozen}
        if checked!=frozen or (set(actual)-set(frozen))-(
                {"source_path","model_path"} if label=="dataset" else set()):
            raise ValueError("frozen_"+label+"_configuration_changed")
    return dict(optimization=dict(vars(opt)),dataset=dict(vars(dataset)),pipeline=dict(vars(pipe)))


def initial_opacity_boundary(gaussians):
    raw=gaussians._opacity.detach()
    allowed=torch.isposinf(raw)
    if not bool((torch.isfinite(raw)|allowed).all()):raise ValueError("invalid_initial_raw_opacity")
    for name in ("_xyz","_scaling","_rotation","_features_dc","_features_rest"):
        if not bool(torch.isfinite(getattr(gaussians,name)).all()):
            raise ValueError("invalid_initial_raw_"+name)
    verify_activated_parameters(gaussians)
    return allowed.clone()


def verify_activated_parameters(gaussians):
    for name in ("get_opacity","get_opacity_with_3D_filter",
                 "get_scaling","get_scaling_with_3D_filter"):
        if not bool(torch.isfinite(getattr(gaussians,name)).all()):
            raise ValueError("nonfinite_activated_"+name)


def verify_updated_parameters(gaussians,allowed_opacity_inf_mask,iteration):
    for group in gaussians.optimizer.param_groups:
        for param in group["params"]:
            if group["name"]=="opacity":
                if param.shape!=allowed_opacity_inf_mask.shape or not bool(
                        (torch.isposinf(param)==allowed_opacity_inf_mask).all()) or not bool(
                        (torch.isfinite(param)|allowed_opacity_inf_mask).all()):
                    raise ValueError(f"nonfinite_or_changed_opacity_boundary_after_{iteration}")
            elif not bool(torch.isfinite(param).all()):
                raise ValueError(f"nonfinite_parameter_{group['name']}_after_{iteration}")
    verify_activated_parameters(gaussians)


def initial_optimizer_hash(gaussians):
    state=gaussians.optimizer.state_dict()
    if state["state"]:
        raise ValueError("expected_fresh_Adam_state")
    return hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()


def host_rgb_loss(gaussians,viewpoint_cam,rendered_image,opt):
    gt=viewpoint_cam.original_image.cuda()
    Ll1=L1_loss_appearance(rendered_image,gt,gaussians,viewpoint_cam.uid)
    return ((1.0-opt.lambda_dssim)*Ll1+
            opt.lambda_dssim*(1.0-ssim(rendered_image.unsqueeze(0),gt.unsqueeze(0))))


def host_step_loss(gaussians,viewpoint_cam,nearest_cam,pipe,bg,kernel_size,patchmatch,opt,iteration):
    """Keep main train.py's Expected normal and PatchMatch branch unchanged."""
    render_pkg=render(viewpoint_cam,gaussians,pipe,bg,kernel_size,require_depth=True)
    rgb=host_rgb_loss(gaussians,viewpoint_cam,render_pkg["render"],opt)
    normal_depth=depth_to_normal(viewpoint_cam,render_pkg["expected_depth"])
    normal_error=1-torch.linalg.vecdot(render_pkg["normal"],normal_depth,dim=0)
    normal_loss=normal_error.mean()
    ncc,geo=patchmatch(gaussians,render_pkg,viewpoint_cam,nearest_cam,iteration,normal_depth)
    host=rgb+opt.lambda_depth_normal*normal_loss+opt.lambda_multi_view_ncc*ncc+opt.lambda_multi_view_geo*geo
    return host,dict(rgb=rgb,normal=normal_loss,ncc=ncc,geo=geo)


def lambda_from_initial(rgb0,R0):
    rgb0,R0=float(rgb0),float(R0)
    if not np.isfinite(rgb0) or not np.isfinite(R0) or rgb0<=0 or R0<=0:
        raise ValueError("invalid_initial_rgb_or_R")
    return .01*rgb0/R0


def stage_input_ply(input_ply,arm_dir):
    path=Path(arm_dir)/"point_cloud"/"iteration_7000"/"point_cloud.ply"
    path.parent.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(input_ply,path)
    if sha256(path)!=EXPECTED_PLY_SHA:raise ValueError("staged_model_sha_mismatch")
    return path


def save_boundary(scene,gaussians,arm_dir,update_count,filter_hash):
    if update_count not in SAVED_UPDATES:raise ValueError("wrong_boundary")
    iteration=7000+update_count
    if snapshot_filter_hash(gaussians)!=filter_hash:raise ValueError("saved_filter_changed")
    if gaussians.get_xyz.shape[0]!=EXPECTED_PLY_ROWS:raise ValueError("topology_changed")
    if update_count:
        scene.save(iteration)
    checkpoint=Path(arm_dir)/f"chkpnt{iteration}.pth"
    if checkpoint.exists():raise ValueError("checkpoint_exists")
    torch.save(dict(model=gaussians.capture(),filter_3D=gaussians.filter_3D,
                    iteration=iteration,completed_updates=update_count),checkpoint)
    report=dict(iteration=iteration,completed_updates=update_count,
        ply=str(Path(arm_dir)/"point_cloud"/f"iteration_{iteration}"/"point_cloud.ply"),
        checkpoint=str(checkpoint))
    if not Path(report["ply"]).is_file() or not checkpoint.is_file():
        raise ValueError("boundary_artifact_missing")
    (Path(arm_dir)/f"complete_{iteration}.json").write_text(
        json.dumps(report,sort_keys=True)+"\n",encoding="utf-8")
    return report


def save_failure_state(scene,gaussians,arm_dir,completed):
    """Best-effort separate partial state; never label it a 500/1000 result."""
    iteration=7000+completed
    marker=Path(arm_dir)/f"complete_{iteration}.json"
    if completed in SAVED_UPDATES and marker.is_file():
        report=json.loads(marker.read_text(encoding="utf-8"))
        if (report.get("completed_updates")==completed and
                all(Path(report[k]).is_file() for k in ("ply","checkpoint"))):
            return None
    partial=Path(arm_dir)/"partial"/f"iteration_{iteration}"/"point_cloud.ply"
    partial.parent.mkdir(parents=True,exist_ok=False)
    gaussians.save_ply(str(partial))
    checkpoint=Path(arm_dir)/f"partial_{iteration}.pth"
    torch.save(dict(model=gaussians.capture(),filter_3D=gaussians.filter_3D,
                    iteration=iteration,completed_updates=completed,partial=True,
                    partial_ply=str(partial)),checkpoint)
    return str(checkpoint)


def snapshot_filter_hash(gaussians):
    return hashlib.sha256(gaussians.filter_3D.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def initialize_arm(parser_args,model_args,opt,arm_dir,input_ply):
    stage_input_ply(input_ply,arm_dir)
    dataset=model_args
    # Keep the CLI's original options, changing only the independent output path.
    from argparse import Namespace
    saved_args=Namespace(**vars(parser_args))
    saved_args.model_path=str(arm_dir)
    (Path(arm_dir)/"cfg_args").write_text(str(saved_args),encoding="utf-8")
    dataset.model_path=str(arm_dir)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    gaussians=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,gaussians,load_iteration=7000,shuffle=False)
    gaussians.spatial_lr_scale=scene.cameras_extent
    gaussians.max_radii2D=torch.zeros((EXPECTED_PLY_ROWS,),device="cuda")
    gaussians.training_setup(opt)
    if gaussians.app_model!=gaussians.App_model.PGSR:
        raise ValueError("wrong_appearance_model")
    cameras=list(scene.getTrainCameras())
    if not cameras or not any(c.image_name=="0049" for c in cameras):
        raise ValueError("reference_camera_missing")
    return scene,gaussians,cameras


def manifest_for_run(args,input_hash,camera_sig,schedule,initial_hash,optimizer_hash,filter_hash,rgb0,R0,lam,configuration,opacity_boundary):
    schedule_json=json.dumps([list(x) for x in schedule],separators=(",",":"))
    return dict(protocol="local_plane_matched_1000_v1",source_main="8474727dd1c1f4fee77a3dc02a6cbc0b38075f21",
        frozen_reference_commit="6ce1b1b739247f5dbfb3eb68fc0a318fa0ba580d",
        input_ply_sha256=input_hash,capture_sha256="74fbd05c3a2c10549adca00f885cdb39c525d003ce120d1a3d593c7feb78d6ef",
        results_sha256="df05b6b018c7b38fc71d1185d119a6dbb58d665f09cd6de8998bb9b1b6cfa8dc",
        source_path=args.source_path,output_root=args.output_root,
        resolution=args.resolution,appearance=args.use_decoupled_appearance,
        depth_ratio=args.depth_ratio,initial_hash=initial_hash,
        initial_optimizer_hash=optimizer_hash,filter_hash=filter_hash,
        camera_signature=camera_sig,schedule=[list(x) for x in schedule],
        schedule_sha256=hashlib.sha256(schedule_json.encode()).hexdigest(),seed=0,
        virtual_steps=[START_VIRTUAL,START_VIRTUAL+UPDATES-1],updates=UPDATES,
        saved_updates=list(SAVED_UPDATES),densify=False,prune=False,opacity_reset=False,
        filter_recompute=False,new_adam=True,init_rgb0049=rgb0,init_R=R0,
        lambda_rule="0.01*init_rgb0049/init_R",lambda_value=lam,
        frozen_host_configuration=configuration,
        initial_positive_inf_opacity_count=int(opacity_boundary.sum()),
        initial_positive_inf_opacity_mask_sha256=hashlib.sha256(
            opacity_boundary.contiguous().cpu().numpy().tobytes()).hexdigest(),
        scientific_scope="fixed_32_windows_1836_ids_1.28_percent_local_conditional_continuation")


def train_arm(label,scene,gaussians,cameras,schedule,reg,lam,pipe,dataset,opt,arm_dir,deadline,filter_hash,allowed_opacity_inf_mask=None):
    if allowed_opacity_inf_mask is None:
        allowed_opacity_inf_mask=torch.isposinf(gaussians._opacity.detach())
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    patchmatch=PatchMatch(opt.multi_view_patch_size,opt.multi_view_pixel_noise_th,
        kernel_size=dataset.kernel_size,pipe=pipe,debug=True,model_path=str(arm_dir))
    checkpoints=[save_boundary(scene,gaussians,arm_dir,0,filter_hash)]
    log_path=Path(arm_dir)/"steps.jsonl"
    completed=0
    gaussians._matched_completed_updates=0
    with log_path.open("x",encoding="utf-8") as log:
        for update,(main_idx,neighbor_idx) in enumerate(schedule,1):
            if time.monotonic()>deadline:raise TimeoutError(f"{label}:time_limit_after_{completed}_updates")
            iteration=7000+update
            gaussians.update_learning_rate(iteration)
            if iteration%1000==0:gaussians.oneupSHdegree()
            cam=cameras[main_idx]
            neighbor=None if neighbor_idx is None else cameras[neighbor_idx]
            host,parts=host_step_loss(gaussians,cam,neighbor,pipe,bg,dataset.kernel_size,
                                      patchmatch,opt,iteration)
            R=reg(gaussians._xyz,gaussians._rotation) if label=="B" else None
            loss=host+(lam*R if R is not None else 0.)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"{label}:nonfinite_loss_at_{iteration}")
            loss.backward()
            for group in gaussians.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                        raise ValueError(f"{label}:nonfinite_gradient_{group['name']}_at_{iteration}")
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            completed=update
            gaussians._matched_completed_updates=completed
            with torch.no_grad():
                verify_updated_parameters(gaussians,allowed_opacity_inf_mask,iteration)
                if label=="B":
                    reg(gaussians._xyz,gaussians._rotation)
            if update==1 or update%100==0:
                row=dict(arm=label,completed_updates=update,virtual_iteration=iteration,
                    main=cam.image_name,neighbor=neighbor.image_name if neighbor is not None else None,
                    host=float(host.detach()),rgb=float(parts["rgb"].detach()),
                    normal=float(parts["normal"].detach()),ncc=float(parts["ncc"].detach()),
                    geo=float(parts["geo"].detach()),R=float(R.detach()) if R is not None else None,
                    loss=float(loss.detach()))
                log.write(json.dumps(row)+"\n");log.flush()
            if update in (500,1000):
                checkpoints.append(save_boundary(scene,gaussians,arm_dir,update,filter_hash))
    if completed!=UPDATES:raise ValueError("not_exactly_1000_steps")
    return checkpoints


def evaluate_train_rgb(gaussians,cameras,pipe,dataset,opt,deadline):
    """All training cameras only; evaluation GT depth/mesh is separate."""
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    records=[]
    with torch.no_grad():
        for cam in cameras:
            if time.monotonic()>deadline:raise TimeoutError("RGB_evaluation_time_limit")
            rendered=render(cam,gaussians,pipe,bg,dataset.kernel_size,require_depth=True)["render"]
            gt=cam.original_image.cuda()
            rgb=host_rgb_loss(gaussians,cam,rendered,opt)
            embedding=gaussians.get_appearance_embedding(cam.uid)
            adjusted=torch.addcmul(embedding[1],torch.exp(embedding[0]),rendered)
            pred=torch.clamp(adjusted,0.,1.)
            records.append(dict(image=cam.image_name,host_rgb_loss=float(rgb),
                appearance_l1=float(l1_loss(pred,gt)),appearance_psnr=float(psnr(pred,gt).mean())))
    return dict(scope="all_training_cameras_not_heldout",camera_count=len(records),
        mean_host_rgb_loss=sum(x["host_rgb_loss"] for x in records)/len(records),
        mean_appearance_l1=sum(x["appearance_l1"] for x in records)/len(records),
        mean_appearance_psnr=sum(x["appearance_psnr"] for x in records)/len(records),
        per_camera=records)


def run(args,parser,lp,op,pp):
    started=time.monotonic()
    build_identity=verify_build_identity(args.expected_head,args.extension_manifest)
    if args.resolution!=2 or args.use_decoupled_appearance!=3 or args.depth_ratio!=0.0:
        raise ValueError("wrong_frozen_host_configuration")
    input_ply=Path(args.input_ply)
    verify_original_ply(input_ply)
    root=Path(args.output_root)
    require_empty_output(root)
    initial_free_bytes=shutil.disk_usage(root.parent).free
    if initial_free_bytes<3*(1<<30):
        raise ValueError("insufficient_output_disk_free_space")
    if not Path(args.source_path).is_dir():raise ValueError("DTU_source_missing")
    if not torch.cuda.is_available():raise ValueError("CUDA_required_for_host_run")
    dataset=lp.extract(args);opt=op.extract(args);pipe=pp.extract(args)
    configuration=verify_frozen_configuration(dataset,opt,pipe)
    root.mkdir()
    reg=FrozenPlaneRegularizer(args.capture,args.frozen_results,
        Path(args.source_path)/"sparse"/"0",device="cuda")
    scene_A,gaussians_A,cameras_A=initialize_arm(args,dataset,opt,root/"A",input_ply)
    camera_sig=camera_signature(cameras_A)
    schedule=fixed_schedule(cameras_A)
    initial_hash=snapshot_hash(gaussians_A)
    opacity_boundary=initial_opacity_boundary(gaussians_A)
    optimizer_hash=initial_optimizer_hash(gaussians_A)
    filter_hash=snapshot_filter_hash(gaussians_A)
    R0=float(reg(gaussians_A._xyz,gaussians_A._rotation).detach())
    if not math.isclose(R0,EXPECTED_R0,rel_tol=1e-5,abs_tol=1e-4):
        raise ValueError("initial_R_not_reproduced")
    ref=next(c for c in cameras_A if c.image_name=="0049")
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    with torch.no_grad():
        pkg=render(ref,gaussians_A,pipe,bg,dataset.kernel_size,require_depth=True)
        rgb0=float(host_rgb_loss(gaussians_A,ref,pkg["render"],opt))
    lam=lambda_from_initial(rgb0,R0)
    manifest=manifest_for_run(args,EXPECTED_PLY_SHA,camera_sig,schedule,initial_hash,optimizer_hash,filter_hash,rgb0,R0,lam,configuration,opacity_boundary)
    manifest["extension_identity"]=build_identity
    manifest["initial_output_free_bytes"]=initial_free_bytes
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    deadline=started+MAX_ACTIVE_SECONDS
    reports={}
    rgb_reports={}
    for label in ("A","B"):
        if label=="A":scene,gaussians,cameras=scene_A,gaussians_A,cameras_A
        else:
            dataset.model_path=str(root/"B")
            scene,gaussians,cameras=initialize_arm(args,dataset,opt,root/"B",input_ply)
            if (camera_signature(cameras)!=camera_sig or snapshot_hash(gaussians)!=initial_hash or
                    initial_optimizer_hash(gaussians)!=optimizer_hash or snapshot_filter_hash(gaussians)!=filter_hash):
                raise ValueError("matched_initial_state_failed")
            if not bool(torch.equal(initial_opacity_boundary(gaussians),opacity_boundary)):
                raise ValueError("matched_opacity_boundary_failed")
            if not math.isclose(float(reg(gaussians._xyz,gaussians._rotation).detach()),R0,rel_tol=0,abs_tol=1e-8):
                raise ValueError("matched_initial_R_failed")
        try:
            reports[label]=train_arm(label,scene,gaussians,cameras,schedule,reg,lam,pipe,dataset,opt,
                                      root/label,deadline,filter_hash,opacity_boundary)
        except Exception as exc:
            completed=getattr(gaussians,"_matched_completed_updates",0)
            partial=None
            try:
                partial=save_failure_state(scene,gaussians,root/label,completed)
            except Exception as save_exc:
                partial="save_failed:"+repr(save_exc)
            (root/"failure.json").write_text(json.dumps(dict(arm=label,error=repr(exc),
                completed_updates=completed,partial_checkpoint=partial,
                complete_reports=reports),indent=2)+"\n",encoding="utf-8")
            raise
        try:
            rgb_reports[label]=evaluate_train_rgb(gaussians,cameras,pipe,dataset,opt,deadline)
        except Exception as exc:
            (root/"failure.json").write_text(json.dumps(dict(arm=label,stage="RGB_evaluation",
                error=repr(exc),complete_reports=reports),indent=2)+"\n",encoding="utf-8")
            raise
        (root/"progress.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
        (root/"rgb_train.json").write_text(json.dumps(rgb_reports,indent=2)+"\n",encoding="utf-8")
        if label=="A":
            scene_A=gaussians_A=cameras_A=None
            del scene,gaussians,cameras
            gc.collect()
            torch.cuda.empty_cache()
    return reports


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    for name in ("input-ply","capture","frozen-results","output-root"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--extension-manifest",required=True)
    args=parser.parse_args()
    run(args,parser,lp,op,pp)
