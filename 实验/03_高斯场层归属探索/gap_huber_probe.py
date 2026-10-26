"""Fixed 1mm proposal-support gap contrast; proposal gaps are not physical empty space."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from photo_surface_probe import HERE, sha256, save, source_info
from depth_holdout import huber_exact
from consensus_depth import _weighted_median, _check_camera_projection


def estimate(z, w, host_expected, teacher_band=None):
    """Camera-z mm. Any invalid member makes the entire ray unknown, never filtered.

    teacher_band is an optional empirical (lower, upper) camera-z mm interval.
    It can veto a proposal-gap Huber update, never supply the selected depth.
    """
    z,w=np.asarray(z,float),np.asarray(w,float)
    out=dict(E=float(host_expected),H=float(host_expected),WM=float(host_expected),
             gap_huber=float(host_expected),gap=False,min_distance_mm=np.nan,status="unknown_empty")
    if teacher_band is not None: out["prior_confirmed_gap"]=float(host_expected)
    if len(z)==0 and len(w)==0: return out
    if (z.ndim!=1 or z.shape!=w.shape or not np.isfinite(z).all() or not np.isfinite(w).all()
            or np.any(z<=0) or np.any(w<0) or not np.isfinite(w.sum()) or w.sum()<=0):
        out["status"]="unknown_invalid_members"; return out
    E=float(np.sum(w*z,dtype=np.float64)/w.sum(dtype=np.float64))
    if not np.isfinite(E) or E<=0:
        out["status"]="unknown_invalid_expected"; return out
    H,_=huber_exact(z,w,h=1.)
    minimum=float(np.min(np.abs(z[w>0]-E))); gap=minimum>1.
    out.update(E=E,H=float(H),WM=float(_weighted_median(z,w)),min_distance_mm=minimum,
               gap=gap,gap_huber=float(H) if gap else E,status="valid")
    if teacher_band is not None:
        lo,hi=teacher_band
        allowed=np.isfinite([lo,hi]).all() and 0<lo<=hi and lo<=H<=hi and not lo<=E<=hi
        out["prior_confirmed_gap"]=float(H) if gap and allowed else E
    return out


def locate_sample(results,scan):
    base=results/"20261026_夜间深度方法"
    roots=[base/name/f"scan{scan}" for name in ("confirm_v1","samples_v2","challenge_106_114_v1")]
    present=[p for p in roots if (p/"metadata.json").is_file()]
    if len(present)!=1: raise ValueError(f"Ambiguous/missing sample roots for scan{scan}: {present}")
    return present[0]


def geometry(results,scan,reference,sample):
    paths=[results/f"20261023_双模型影响诊断/influence_G1_v1/scan{scan}/metadata.json",
           results/f"20261026_夜间深度方法/dense_confirm_v1/scan{scan}_v1/metadata.json",
           results/f"20261026_Huber全场景/scan{scan}_v1/metadata.json"]
    choices=[]
    for path in paths:
        if not path.is_file(): continue
        meta=json.loads(path.read_text(encoding="utf-8"))
        if meta["status"]!="complete": continue
        if any(meta[k]!=sample[k] or meta[k]!=reference[k] for k in ("model_sha256","cfg_sha256")):
            raise ValueError("Geometry model/cfg differs: "+str(path))
        if meta["cameras_npz_sha256"]!=reference["source_inputs"][f"scan{scan}/cameras.npz"]:
            raise ValueError("Geometry cameras SHA differs")
        choices.append((path,meta))
    if not choices: raise ValueError("No verified geometry metadata")
    path,meta=choices[0]
    for _,other in choices[1:]:
        np.testing.assert_allclose(other["alignment"]["transform"],meta["alignment"]["transform"],rtol=0,atol=1e-6)
        np.testing.assert_allclose(other["alignment"]["scale_to_dtu_mm"],meta["alignment"]["scale_to_dtu_mm"],rtol=0,atol=1e-8)
    alignment=np.asarray(meta["alignment"]["transform"],float); scale=float(meta["alignment"]["scale_to_dtu_mm"])
    if alignment.shape!=(4,4) or not np.isfinite(alignment).all() or not np.isfinite(scale) or scale<=0: raise ValueError("Invalid alignment")
    camera=results/f"20261026_夜间深度方法/inputs/scan{scan}/cameras.npz"
    if sha256(camera)!=meta["cameras_npz_sha256"]: raise ValueError("Camera file SHA mismatch")
    return alignment,scale,camera,dict(geometry_metadata=str(path),geometry_sha256=sha256(path),cameras_sha256=sha256(camera))


def teacher_band(root,scan,name,xy,size,scale,results):
    """Read only calibration/raw fields, never the prior's NN scores."""
    scene=root/f"scan{scan}_view{name}"; path=scene/"metadata.json"
    top=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    record=json.loads(path.read_text(encoding="utf-8"))
    if top["status"]!="complete" or record["status"]!="complete" or record["scan"]!=scan or record["reference"]!=name:
        raise ValueError("Incomplete/different teacher source")
    hashes={str(path):sha256(path),str(root/"metadata.json"):sha256(root/"metadata.json")}
    for filename,digest in record["sealed_sha256"].items():
        if sha256(scene/filename)!=digest: raise ValueError("Teacher sealed SHA mismatch")
        hashes[str(scene/filename)]=digest
    rawpath=scene/"raw_prior.npz"
    if "raw_prior.npz" not in record["sealed_sha256"]: raise ValueError("Unsealed full-image prior")
    with np.load(rawpath) as f: raw=f["raw"].copy()
    if raw.shape!=size: raise ValueError("Prior/sample image dimensions differ")
    # The prior uses the dense-cache camera-z units; verify scale/model/camera provenance.
    cache=results/(f"20261026_夜间深度方法/dense_confirm_v1/scan{scan}_v1" if scan==37 else f"20261026_Huber全场景/scan{scan}_v1")
    dense=json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(dense["alignment"]["scale_to_dtu_mm"],scale,rtol=0,atol=1e-8)
    vm=next(v for v in dense["views"] if v["name"]==name)
    if vm["sha256"]!=record["frame_sha256"]: raise ValueError("Teacher cache lineage differs")
    hashes[str(cache/"metadata.json")]=sha256(cache/"metadata.json")
    cal=record["calibration"]; bands=np.full((len(xy),2),np.nan)
    if cal["valid"]:
        a,b,q=cal["a"],cal["b"],cal["q"]
        if not np.isfinite([a,b,q]).all() or a<=0 or q<0: raise ValueError("Invalid teacher calibration")
        mu=a*raw[xy[:,1],xy[:,0]]+b
        valid=np.isfinite(mu)&(mu-q>0)&np.isfinite(mu+q)
        bands[valid,0]=scale/(mu[valid]+q); bands[valid,1]=scale/(mu[valid]-q)
    return bands,hashes,cal


def run(args):
    source=source_info(); results=HERE/"结果"; output=args.output.resolve()
    if not args.output.is_absolute() or output.exists(): raise ValueError("Require new absolute output")
    output.mkdir(parents=True); start=time.monotonic()
    canonical_path=HERE/"G1_Huber全场景输入参考.json"
    canonical=json.loads(canonical_path.read_text(encoding="utf-8"))["scans"]
    state=dict(status="partial",source=source,h_mm=1.,scans=args.scans,views=args.views,completed=[],canonical_sha256=sha256(canonical_path),
               limitation="Fixed proposal-support gap is not physical empty space or a correctness guarantee. Teacher bands are empirical veto only. Scan NN is not pixel GT/mesh/Overall. Unknown rays preserve host E and are explicitly excluded from the predeclared common valid-ray evaluation; no member pruning or GT-based gate selection.")
    save(output/"metadata.json",state)
    def check():
        if time.monotonic()-start>args.max_seconds or (output/"stop_after_view").exists(): raise InterruptedError("Finite deadline/stop")
    try:
        for scan in args.scans:
            check(); root=locate_sample(results,scan); mp=root/"metadata.json"; sample=json.loads(mp.read_text(encoding="utf-8")); reference=canonical[str(scan)]
            if sample["status"]!="complete" or sample["git_remote"]!="git@github.com:gfall-1407/GauDepth.git" or sample.get("git_status"):
                raise ValueError("Invalid sampling lineage")
            if any(sample[k]!=reference[k] for k in ("model_sha256","cfg_sha256")): raise ValueError("Sample model/cfg mismatch")
            alignment,scale,camera,lineage=geometry(results,scan,reference,sample)
            lineage.update(sample_metadata=str(mp),sample_metadata_sha256=sha256(mp),sample_commit=sample["git_commit"],model_sha256=sample["model_sha256"],cfg_sha256=sample["cfg_sha256"])
            for view in args.views:
                check(); matches=[v for v in sample["views"] if v["view_index"]==view]
                if len(matches)!=1: raise ValueError("Missing/ambiguous sampled view")
                vm=matches[0]; name=vm["image_name"]; path=root/vm["file"]
                if sha256(path)!=vm["npz_sha256"]: raise ValueError("Ray NPZ SHA mismatch")
                with np.load(path) as f:
                    offsets=f["offsets"].copy(); z=f["depth"].astype(float); w=f["weight"].astype(float); xy=f["pixel_xy"].copy()
                    host=f["host_target"].astype(float); qualitative=f["qualitative"].astype(bool); size=tuple(map(int,f["image_size"]))
                    if int(f["view_index"])!=view: raise ValueError("NPZ view mismatch")
                n=len(xy)
                if offsets.dtype.kind not in "iu" or offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(z) or np.any(np.diff(offsets)<0) or z.shape!=w.shape or host.shape!=(n,3) or qualitative.shape!=(n,): raise ValueError("Invalid ray record structure")
                if xy.shape!=(n,2) or xy.dtype.kind not in "iu" or len(np.unique(xy,axis=0))!=n or size!=tuple(vm["image_size"]) or np.any(xy<0) or np.any(xy[:,0]>=size[1]) or np.any(xy[:,1]>=size[0]): raise ValueError("Invalid pixel inventory")
                if int((~qualitative).sum())!=vm["random_samples"] or int(qualitative.sum())!=vm["qualitative_samples"]: raise ValueError("Sampling counts differ")
                camera_errors=_check_camera_projection(camera,vm,alignment)
                methods=["E","H","WM","gap_huber"]; bands=None; teacher_hashes={}; cal=None; teacher_seconds=0.
                if args.teacher_root:
                    tick=time.monotonic(); bands,teacher_hashes,cal=teacher_band(args.teacher_root,scan,name,xy,size,scale,results); teacher_seconds=time.monotonic()-tick; methods.append("prior_confirmed_gap")
                tick=time.monotonic(); predictions=[]; details=[]
                for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
                    check(); record=estimate(z[a:b]*scale,w[a:b],host[i,0]*scale,None if bands is None else bands[i]); details.append(record)
                    predictions.append([record[m] for m in methods])
                    if record["status"]=="valid": np.testing.assert_allclose(record["E"]/scale,host[i,0],rtol=sample["replay_tolerance"]["rtol"],atol=sample["replay_tolerance"]["atol"])
                seconds=time.monotonic()-tick; predictions=np.asarray(predictions,float); status=np.array([v["status"] for v in details]); nonempty=np.diff(offsets)>0
                paired=(~qualitative)&nonempty&(status=="valid")&np.isfinite(predictions[:,0])&(predictions[:,0]>0)
                scene=output/f"scan{scan}_view{name}"; scene.mkdir()
                np.savez_compressed(scene/"decisions.npz",pixel_xy=xy,methods=methods,depth_mm=predictions,paired=paired,status=status,gap=np.array([v["gap"] for v in details]),min_distance_mm=np.array([v["min_distance_mm"] for v in details]),teacher_band_mm=bands if bands is not None else np.empty((0,2)))
                sealed=sha256(scene/"decisions.npz")
                report=dict(status="partial",source=source,scan=scan,view=name,methods=methods,total=n,random=int((~qualitative).sum()),paired=int(paired.sum()),unknown_counts=dict(Counter(status.tolist())),prediction_seconds=seconds,teacher_load_band_seconds=teacher_seconds,raw_npz_sha256=vm["npz_sha256"],lineage=lineage,teacher_input_sha256=teacher_hashes,teacher_calibration=cal,decision_sha256=sealed,camera_errors=camera_errors,h_mm=1.,scale_to_dtu_mm=scale)
                save(scene/"metadata.json",report); check()
                # Decisions and provenance are sealed before scanned geometry is opened.
                from scipy.spatial import cKDTree
                from plyfile import PlyData
                gt=results/f"20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"; digest=sha256(gt)
                if digest!=reference["gt_inputs"][f"Points/stl/stl{scan:03d}_total.ply"]: raise ValueError("GT SHA mismatch")
                vertex=PlyData.read(str(gt))["vertex"]; tree=cKDTree(np.column_stack([np.asarray(vertex[k],float) for k in ("x","y","z")]))
                fx,fy=vm["focal"]; h,width=size; K=np.array([[fx,0,(width-1)/2],[0,fy,(h-1)/2],[0,0,1.]])
                rays=np.column_stack((xy[paired],np.ones(paired.sum())))@np.linalg.inv(K).T; pose=np.linalg.inv(np.asarray(vm["world_view_transform"]).T)
                nn=np.full(predictions.shape,np.nan); summaries={}
                for j,method in enumerate(methods):
                    check(); world=(rays*(predictions[paired,j]/scale)[:,None])@pose[:3,:3].T+pose[:3,3]; mm=world@alignment[:3,:3].T+alignment[:3,3]
                    nn[paired,j]=tree.query(mm,k=1,workers=1)[0]; values=nn[paired,j]; delta=values-nn[paired,0]
                    changed=predictions[paired,j]!=predictions[paired,0]
                    summaries[method]=dict(n=len(values),mean=float(values.mean()) if len(values) else None,median=float(np.median(values)) if len(values) else None,p95=float(np.quantile(values,.95)) if len(values) else None,changed=int(changed.sum()),improved=int((delta<0).sum()),worsened=int((delta>0).sum()),changed_delta_mean=float(delta[changed].mean()) if changed.any() else None)
                np.save(scene/"nn.npy",nn)
                if sha256(scene/"decisions.npz")!=sealed or sha256(path)!=vm["npz_sha256"]: raise ValueError("Sealed inputs changed")
                report.update(status="complete",nn_proxy=summaries,gt_sha256=digest,nn_sha256=sha256(scene/"nn.npy")); save(scene/"metadata.json",report)
                state["completed"].append(report); save(output/"metadata.json",state)
        check(); state.update(status="complete",seconds=time.monotonic()-start); save(output/"metadata.json",state)
    except BaseException as exc:
        state.update(status="failed_or_stopped",error=repr(exc)); save(output/"metadata.json",state); raise


def synthetic_check():
    one=estimate([9.9,10.,10.1],[1,2,1],10); assert one["gap_huber"]==one["E"] and not one["gap"]
    z=np.array([7.,13.]); w=np.array([.6,.4]); two=estimate(z,w,0)
    assert two["gap"] and two["gap_huber"]==two["H"] and two["H"]!=two["E"]
    split=estimate(np.repeat(z,2),np.repeat(w/2,2),0)
    for k in ("E","H","WM","gap_huber","min_distance_mm"): np.testing.assert_allclose(two[k],split[k],rtol=0,atol=1e-12)
    veto=estimate(z,w,0,(20,21)); assert veto["prior_confirmed_gap"]==veto["E"]
    assert estimate([],[],5)["status"]=="unknown_empty" and estimate([np.nan],[1],5)["status"]=="unknown_invalid_members"
    assert estimate([1],[-1],5)["gap_huber"]==5
    symmetric=estimate(z,[.5,.5],10); assert symmetric["gap"] and symmetric["E"]==10
    # Symmetric mode positions with unequal mass: truth can still equal accurate E.
    true_surface=two["E"]; harm=abs(two["gap_huber"]-true_surface); assert harm>0
    print(json.dumps(dict(check="PASS",asymmetric_mass_counterexample_E=two["E"],H=two["H"],true_surface=true_surface,gate_error_mm=harm,limitation="Even symmetric proposal mode positions do not establish a physical gap. Equal-mass symmetric proposals also pass gap although true surface may be E; no correctness guarantee.")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--check",action="store_true")
    p.add_argument("--scans",type=int,nargs="+",default=[37,40]); p.add_argument("--views",type=int,nargs="+",default=[0])
    p.add_argument("--output",type=Path); p.add_argument("--max-seconds",type=int,default=900); p.add_argument("--teacher-root",type=Path)
    args=p.parse_args()
    if args.check: synthetic_check()
    else:
        if args.output is None or not 0<args.max_seconds<=900: p.error("New output and max-seconds in (0,900] required")
        if len(set(args.scans))!=len(args.scans) or len(set(args.views))!=len(args.views): p.error("Duplicate scans/views")
        run(args)
