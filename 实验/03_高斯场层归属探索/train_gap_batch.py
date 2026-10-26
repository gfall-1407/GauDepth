"""Finite paired PLY-adaptation pilot, then unchanged native expected reconstruction."""
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from photo_surface_probe import HERE, REPO, source_info, save, sha256
sys.path.insert(0,str(REPO))


def evaluate(model, destination, gt_root):
    import torch
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from diff_gaussian_rasterization import _C
    from sample import protect_scene
    import mesh_extract
    import evaluate_dtu_mesh
    from dense_huber_mesh import cfg_values, native_extract, native_evaluate, HOST_SHA
    if sha256(_C.__file__)!=HOST_SHA:raise ValueError("Original CUDA binary changed")
    if destination.exists():raise ValueError("Do not overwrite native evaluation")
    destination.mkdir()
    training=json.loads((model/"metadata.json").read_text(encoding="utf-8"))
    if training["status"]!="complete" or training["actual_updates"]!=1000:raise ValueError("Partial training cannot enter main evaluation")
    canonical=json.loads((HERE/"G1_Huber全场景输入参考.json").read_text(encoding="utf-8"))["scans"][str(training["args"]["scan"])]
    for relative,digest in canonical["gt_inputs"].items():
        if sha256(gt_root/relative)!=digest:raise ValueError("Evaluation GT input SHA differs")
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    values=vars(p.parse_args([]));values.update(cfg_values(model/"cfg_args"));values["model_path"]=str(model)
    ns=argparse.Namespace(**values);dataset=mp.extract(ns);pipe=pp.extract(ns)
    ply=model/f"point_cloud/iteration_{training['iteration']}/point_cloud.ply"
    if sha256(ply)!=training["ply_sha256"]:raise ValueError("Trained PLY SHA changed")
    g=GaussianModel(dataset.sh_degree)
    with protect_scene(destination):scene=Scene(dataset,g,load_iteration=training["iteration"],shuffle=False)
    scene.model_path=dataset.model_path=str(destination)
    started=time.time()
    with torch.no_grad():
        native_extract(mesh_extract,dataset,pipe,g,scene,render)
        native_evaluate(evaluate_dtu_mesh,dataset,gt_root,g,scene)
    scores=json.loads((destination/"vis/results.json").read_text())
    result=dict(scores=scores,seconds=time.time()-started,trained_ply_sha256=sha256(ply),
        original_C_sha256=sha256(_C.__file__),depth_readout="unchanged native expected; no gap postprocessing",
        mesh_sha256=sha256(destination/"recon_post.ply"),results_sha256=sha256(destination/"vis/results.json"))
    save(destination/"metadata.json",result)
    return result


def main(args):
    if socket.gethostname()!="autodl-container-5d5147b55f-67549ee6":raise ValueError("Only G0")
    root=args.output.resolve()
    if not args.output.is_absolute() or root.exists():raise ValueError("Independent new output required")
    root.mkdir(parents=True)
    state=dict(status="partial",source=source_info(),steps=[],started=time.time(),design="lambda0 vs.01, matched PLY warm-start, 1000 actual updates, seed20261027")
    def record(status,**kw):state.update(status=status,**kw);save(root/"batch.json",state)
    signal.signal(signal.SIGTERM,lambda *_:(root/"stop_batch").touch())
    def check():
        if (root/"stop_batch").exists():raise InterruptedError("Guard requested boundary stop")
    def train(name,weight,smoke=False):
        check();command=[sys.executable,str(HERE/"train_gap_adapt.py"),"--scan","40","--model-root",str(args.model_root),
            "--source-root",str(args.source_root),"--teacher-root",str(args.teacher_root),"--output",str(root/name),
            "--lambda-gap",str(weight),"--steps","1000","--seed","20261027","--max-seconds","900"]
        if smoke:command.extend(["--smoke-steps","10"])
        record("running",active=name,command=command)
        with (root/(name+".log")).open("xb") as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        meta_path=root/name/"metadata.json"
        m=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        state["steps"].append(dict(name=name,returncode=r.returncode,status=m.get("status")))
        record("step_saved")
        if r.returncode or m.get("status")!=("smoke_complete" if smoke else "complete"):raise RuntimeError("Failed/partial "+name)
        return m
    try:
        train("smoke",.01,True)
        control=train("control",0.)
        treated=train("gap_train",.01)
        for field in ("initial_tensor_sha256","schedule_sha256","options"):
            if control[field]!=treated[field]:raise ValueError("Unmatched training arms: "+field)
        check();results={}
        for name in ("control","gap_train"):
            record("evaluating",active=name)
            # Isolate CUDA/large tensors and evaluator state between arms.
            command=[sys.executable,str(HERE/"train_gap_batch.py"),"--evaluate-model",str(root/name),
                "--output",str(root/(name+"_native")),"--gt-root",str(args.gt_root)]
            with (root/(name+"_native.log")).open("xb") as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError("Native evaluation failed "+name)
            results[name]=json.loads((root/(name+"_native")/"metadata.json").read_text())
            record("evaluation_saved",results=results);check()
        record("complete_waiting_transfer",finished_compute=time.time())
        while not (root/"transfer_verified").exists():check();time.sleep(2)
        record("complete",finished=time.time())
    except BaseException as e:record("failed_or_stopped",error=repr(e),finished=time.time());raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model-root",type=Path,default=Path("/root/GauDepth_HuberRepair_20261026/inputs_G1/outtest"))
    p.add_argument("--source-root",type=Path,default=Path("/root/autodl-tmp/DTU"))
    p.add_argument("--teacher-root",type=Path,default=Path("/root/GauDepth_Gap_20261027/scan40_v1/cache40"))
    p.add_argument("--gt-root",type=Path,default=Path("/root/autodl-tmp/DTU_MVS_Data"))
    p.add_argument("--evaluate-model",type=Path)
    a=p.parse_args()
    if a.evaluate_model:evaluate(a.evaluate_model,a.output,a.gt_root)
    else:main(a)
