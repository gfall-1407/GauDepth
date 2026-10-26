"""Post-decision sampled NN summary; never generates or selects predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from photo_surface_probe import sha256, save, source_info

SCANS=(24,37,40,55,63,65,69,83,97,105,106,110,114,118,122)
METHODS=("E","H","WM","gap_huber")
LABEL="sampled one-way scan NN, not native reconstruction Overall/pixel GT"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_view(root, record):
    scan,view=int(record["scan"]),str(record["view"])
    scene=root/f"scan{scan}_view{view}"
    path=scene/"metadata.json"; metadata=read_json(path)
    if (metadata["status"]!="complete" or metadata["scan"]!=scan or metadata["view"]!=view
            or record["status"]!="complete" or metadata["source"]!=record["source"]):
        raise ValueError("Incomplete/different view record: "+str(scene))
    dp,npth=scene/"decisions.npz",scene/"nn.npy"
    for p,key in ((dp,"decision_sha256"),(npth,"nn_sha256")):
        if sha256(p)!=record[key] or record[key]!=metadata[key]:
            raise ValueError("Sealed prediction/NN SHA differs: "+str(p))
    with np.load(dp,allow_pickle=False) as f:
        names=f["methods"].tolist(); paired=f["paired"].copy(); pred=f["depth_mm"].copy()
        gap=f["gap"].copy(); xy=f["pixel_xy"].copy()
    if len(set(names))!=len(names) or not set(METHODS).issubset(names):
        raise ValueError("Missing/duplicate method labels")
    indices=[names.index(name) for name in METHODS]
    nn=np.load(npth,allow_pickle=False)
    n=len(paired)
    if (paired.dtype.kind!="b" or paired.shape!=(n,) or pred.shape!=(n,len(names))
            or nn.shape!=pred.shape or gap.shape!=(n,) or gap.dtype.kind!="b" or xy.shape!=(n,2)):
        raise ValueError("Prediction/audit shapes differ")
    nn,pred=nn[:,indices],pred[:,indices]
    if not np.array_equal(np.isfinite(nn),np.broadcast_to(paired[:,None],nn.shape)):
        raise ValueError("Four methods do not share the sealed paired NN denominator")
    if not np.isfinite(pred[paired]).all() or np.any(pred[paired]<=0) or np.any(nn[paired]<0):
        raise ValueError("Invalid paired prediction/NN")
    if int(paired.sum())!=metadata["paired"] or n!=metadata["total"] or paired.sum()==0:
        raise ValueError("Empty/different paired count")
    lineage=dict(scan=scan,view=view,prediction_source=metadata["source"],
                 raw_npz_sha256=metadata["raw_npz_sha256"],input_lineage=metadata["lineage"],
                 metadata_path=str(path),metadata_sha256=sha256(path),
                 decision_path=str(dp),decision_sha256=sha256(dp),
                 nn_path=str(npth),nn_sha256=sha256(npth),total=n,random=metadata["random"],
                 paired=int(paired.sum()),prediction_seconds=float(metadata["prediction_seconds"]))
    return dict(scan=scan,view=view,nn=nn[paired],pred=pred[paired],gap=gap[paired],
                gap_all=int(gap.sum()),ray=np.flatnonzero(paired),xy=xy[paired],lineage=lineage)


def summarize(args):
    source=source_info(); root=args.root.resolve(); output=args.output.resolve()
    targets=[output]+([args.csv.resolve()] if args.csv else [])
    if len(set(targets))!=len(targets) or any(path.exists() for path in targets):
        raise ValueError("Require distinct new output files; never overwrite")
    if any(not path.parent.is_dir() for path in targets): raise ValueError("Output parent must exist")
    controller_path=root.parent/(root.name+"_controller.json"); controller=read_json(controller_path)
    steps=controller["steps"]
    if (controller["status"]!="complete" or controller["awake"]!="released" or len(steps)!=15
            or any(step.get("returncode")!=0 for step in steps)
            or {int(step["scan"]) for step in steps}!=set(SCANS)):
        raise ValueError("Controller incomplete, failed or not released")
    view_data=[]; inputs={str(controller_path):sha256(controller_path)}
    for scan in SCANS:
        child=root/f"scan{scan}"; mp=child/"metadata.json"; meta=read_json(mp)
        if meta["status"]!="complete" or meta["scans"]!=[scan]: raise ValueError("Partial/different child source")
        step=next(s for s in steps if int(s["scan"])==scan)
        records=meta["completed"]
        if ({int(r["view"]) for r in records}!=set(step["views"])
                or len(records)!=len(step["views"])): raise ValueError("Child view inventory differs from controller")
        for record in records:
            if int(record["scan"])!=scan: raise ValueError("Cross-scan child record")
            view_data.append(load_view(child,record))
        inputs[str(mp)]=sha256(mp)
    if len(view_data)!=43: raise ValueError("Require exactly 43 new completed views")
    pilot=controller["reused_pilot"]; pilot_path=Path(pilot["path"])
    if sha256(pilot_path)!=pilot["sha256"]: raise ValueError("Pilot metadata SHA differs")
    pmeta=read_json(pilot_path)
    wanted={(37,"0000"),(40,"0000")}
    if pmeta["status"]!="complete" or {(int(s),str(v)) for s,v in pilot["scan_views"]}!=wanted:
        raise ValueError("Incomplete/different reused pilot")
    selected=[r for r in pmeta["completed"] if (int(r["scan"]),str(r["view"])) in wanted]
    if len(selected)!=2: raise ValueError("Pilot missing/duplicate records")
    view_data.extend(load_view(pilot_path.parent,r) for r in selected)
    inputs[str(pilot_path)]=sha256(pilot_path)
    keys=[(v["scan"],v["view"]) for v in view_data]
    if len(keys)!=45 or len(set(keys))!=45 or any(sum(s==scan for s,_ in keys)!=3 for scan in SCANS):
        raise ValueError("Require exactly 45 unique scan/view pairs and three per scene")
    scenes=[]; csv_rows=[]; grouped={}
    for scan in SCANS:
        views=sorted((v for v in view_data if v["scan"]==scan),key=lambda v:v["view"])
        nn=np.concatenate([v["nn"] for v in views]); pred=np.concatenate([v["pred"] for v in views])
        gap=np.concatenate([v["gap"] for v in views]); grouped[scan]=(views,nn,pred)
        stats={}
        for j,name in enumerate(METHODS):
            delta=nn[:,j]-nn[:,0]; changed=pred[:,j]!=pred[:,0]
            stats[name]=dict(mean=float(nn[:,j].mean()),median=float(np.median(nn[:,j])),
                             p95=float(np.quantile(nn[:,j],.95)),changed=int(changed.sum()),
                             improved=int((delta<0).sum()),worsened=int((delta>0).sum()),
                             unchanged_nn=int((delta==0).sum()))
        commits=sorted({v["lineage"]["prediction_source"]["commit"] for v in views})
        row=dict(scan=scan,N=len(nn),gap_true_paired=int(gap.sum()),gap_true_all=int(sum(v["gap_all"] for v in views)),
                 methods=stats,prediction_commits=commits,prediction_seconds=sum(v["lineage"]["prediction_seconds"] for v in views))
        scenes.append(row)
        csv_rows.append(dict(scan=scan,N=len(nn),E_NN=stats["E"]["mean"],H_NN=stats["H"]["mean"],WM_NN=stats["WM"]["mean"],gap_NN=stats["gap_huber"]["mean"],
                             gap_true=int(gap.sum()),changed=stats["gap_huber"]["changed"],wins=stats["gap_huber"]["improved"],losses=stats["gap_huber"]["worsened"],prediction_seconds=row["prediction_seconds"],prediction_commits=";".join(commits)))
    means=np.array([[row["methods"][method]["mean"] for method in METHODS] for row in scenes])
    equal_means=means.mean(axis=0)
    rng=np.random.default_rng(20261027); draws=rng.integers(0,15,size=(10000,15)); comparisons={}
    for j,baseline in enumerate(METHODS[:3]):
        delta=means[:,3]-means[:,j]; boot=delta[draws].mean(axis=1)
        loo=(delta.sum()-delta)/14
        comparisons["gap_vs_"+baseline]=dict(mean_delta_mm=float(delta.mean()),scene_wins=int((delta<0).sum()),
             scene_ties=int((delta==0).sum()),scene_losses=int((delta>0).sum()),
             scene_deltas_mm={str(scan):float(d) for scan,d in zip(SCANS,delta)},
             leave_one_scene_out_mean_delta_range_mm=[float(loo.min()),float(loo.max())],
             fixed_set_bootstrap_sensitivity_95pct_mm=np.quantile(boot,[.025,.975]).tolist())
    # Each point's contribution uses its own scene denominator, not the global ray pool.
    candidates=[]
    for scan,(views,nn,_) in grouped.items():
        offset=0
        for view in views:
            d=view["nn"][:,3]-view["nn"][:,0]; i=int(np.argmin(d))
            candidates.append((float(d[i]/(15*len(nn))),scan,view,i,offset+i,float(d[i])))
            offset+=len(view["nn"])
    chosen=min(candidates,key=lambda x:x[0]); contribution,scan,view,i,scene_index,raw_delta=chosen
    robustness=dict(diagnostic_only=True,baseline="E",before_equal_scene_mean_delta_mm=float(equal_means[3]-equal_means[0]))
    if raw_delta<0:
        _,nn,_=grouped[scan]; keep=np.arange(len(nn))!=scene_index
        if not keep.any(): raise ValueError("Cannot remove sole scene point")
        new_scene=nn[keep].mean(axis=0); old_scene=nn.mean(axis=0)
        after=equal_means+(new_scene-old_scene)/15
        robustness.update(removed=dict(scan=scan,view=view["view"],rayindex=int(view["ray"][i]),xy=view["xy"][i].tolist(),raw_gap_minus_E_mm=raw_delta,
                         equal_scene_contribution_mm=contribution),
                         scene_E_before_mm=float(old_scene[0]),scene_gap_before_mm=float(old_scene[3]),
                         scene_E_after_mm=float(new_scene[0]),scene_gap_after_mm=float(new_scene[3]),
                         after_equal_scene_E_mm=float(after[0]),after_equal_scene_gap_mm=float(after[3]),
                         after_equal_scene_mean_delta_mm=float(after[3]-after[0]),still_improves=bool(after[3]<after[0]))
    else: robustness.update(removed=None,reason="No favorable point",still_improves=False)
    report=dict(status="complete",summary_source=source,overall_label=LABEL,scene_count=15,view_count=45,
                methods=list(METHODS),scene_equal_mean_NN_mm=dict(zip(METHODS,equal_means.tolist())),scenes=scenes,comparisons=comparisons,
                remove_most_favorable_single_point=robustness,prediction_commits=sorted({v["lineage"]["prediction_source"]["commit"] for v in view_data}),
                prediction_seconds_total=sum(v["lineage"]["prediction_seconds"] for v in view_data),
                timing_interpretation="Includes all Huber baseline solutions; not measured deployment efficiency.",
                bootstrap=dict(seed=20261027,resamples=10000,unit="15 fixed scene means",interpretation="Fixed-set resampling sensitivity only; no population inference or strict independent confidence guarantee."),
                aggregation="Concatenate paired rays within each scene, then average the 15 scene means equally. Scene median/p95 are per-scene only; unequal ray counts are not pooled for scene-equal averages.",
                input_metadata_sha256=inputs,input_views=[v["lineage"] for v in view_data],
                limitation="Post-decision statistics only; NN never feeds prediction selection. Sampled views/scenes and reused pilot are exploratory, with their actual prediction commits retained.")
    # Exclusive file creation prevents overwriting an output created during input validation.
    with output.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    if args.csv:
        with args.csv.resolve().open("x",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--csv",type=Path)
    summarize(parser.parse_args())
