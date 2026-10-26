"""Fixed sparse-SfM observation contrast for bounded stereo, never scan-GT selection."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import numpy as np
from bounded_correspondence import depth_interval, solve_intervals, ray_projection, _ray_terms, project_joint_intervals
from photo_objective import bilinear
from photo_surface_probe import HERE, REPO, sha256, save, source_info
from consensus_depth import _check_camera_projection


def audit_surface(args):
    """Post-decision symmetric local-plane diagnostic, never a depth selector."""
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    out=args.output.resolve();target=out/"surface_audit.json"
    if target.exists():raise ValueError("Do not overwrite surface audit")
    report=dict(status="partial",source=source_info(),k_per_method=20,completed=[],
        limitation="Post-hoc diagnostic only; local PCA plane is not pixel GT. Identical symmetric union of E/D nearest-neighbor sets defines each comparison plane. All points retained; no planarity-based selection. Same SfM cameras/training data; repeated tracks are not independent.")
    spec=importlib.util.spec_from_file_location("sfm_surface_loader",REPO/"scene/colmap_loader.py")
    loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
    verified=json.loads((args.observation_root.parent/"selected_local_verified.json").read_text())
    for rel,v in verified["files"].items():assert sha256(args.observation_root/rel)==v["sha256"]
    started=time.monotonic();all_delta=[];ids_by={37:[],40:[]};accepted_ids={37:[],40:[]}
    for scan in (37,40):
        cache=(HERE/"结果/20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan==37 else HERE/"结果/20261026_Huber全场景/scan40_v1")
        cached=json.loads((cache/"metadata.json").read_text(encoding="utf-8"));vm={v["name"]:v for v in cached["views"]};transform=np.asarray(cached["alignment"]["transform"])
        gt=HERE/f"结果/20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"
        assert sha256(gt)==cached["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]
        vertex=PlyData.read(str(gt))["vertex"];points=np.column_stack([np.asarray(vertex[k],float) for k in ("x","y","z")]);tree=cKDTree(points)
        source=args.observation_root/f"scan{scan}";intr=loader.read_intrinsics_binary(source/"sparse/0/cameras.bin")
        extr={Path(v.name).stem:v for v in loader.read_extrinsics_binary(source/"sparse/0/images.bin").values()}
        with np.load(args.observation_root.parent/f"sfm_interval_v1/scan{scan}_view0000/decisions.npz") as f:old_ids=f["point_ids"].copy()
        for ref in ("0012","0024","0036","0048"):
            if time.monotonic()-started>args.max_seconds or (out/"stop_after_score").exists():raise InterruptedError("Finite audit deadline/score stop")
            meta=json.loads((out/f"ref{ref}/metadata.json").read_text(encoding="utf-8"));assert meta["status"]=="complete"
            row=next(r for r in meta["completed"] if r["scan"]==scan);scene=out/f"ref{ref}/scan{scan}_view{ref}"
            assert sha256(scene/"decisions.npz")==row["decisions_sha256"]
            with np.load(scene/"decisions.npz") as f:d={k:f[k].copy() for k in f.files}
            assert not np.intersect1d(d["point_ids"],old_ids).size
            r=extr[ref];k=intr[r.camera_id];h,w=vm[ref]["image_size"];fx,fy,cx,cy=k.params;sx,sy=w/k.width,h/k.height
            K=np.array([[sx*fx,0,sx*cx-.5],[0,sy*fy,sy*cy-.5],[0,0,1.]])
            w2c=np.eye(4);w2c[:3,:3]=r.qvec2rotmat();w2c[:3,3]=r.tvec;pose=np.linalg.inv(w2c.astype(np.float32).astype(float))
            rays=np.column_stack((d["xy"],np.ones(len(d["E"]))))@np.linalg.inv(K).T
            xyz=[]
            for depth in (d["E"],d["selected"]):
                world=rays*depth[:,None]@pose[:3,:3].T+pose[:3,3];xyz.append(world@transform[:3,:3].T+transform[:3,3])
            nn=[];indices=[]
            for x in xyz:
                distance,index=tree.query(x,k=20,workers=1);nn.append(distance[:,0]);indices.append(index)
            plane=np.empty((2,len(d["E"])));quality=np.empty(len(d["E"]));roughness=quality.copy()
            for i in range(len(d["E"])):
                cloud=points[np.unique(np.r_[indices[0][i],indices[1][i]])];center=cloud.mean(axis=0)
                eigen,vec=np.linalg.eigh((cloud-center).T@(cloud-center)/len(cloud));normal=vec[:,0]
                plane[:,i]=[abs((x[i]-center)@normal) for x in xyz]
                quality[i]=eigen[0]/max(eigen.sum(),1e-30);roughness[i]=np.sqrt(max(eigen[0],0.))
            for label,values in zip(("expected","sfm_bounded"),nn):
                np.testing.assert_allclose(np.mean(values),row["nn_proxy"][label]["mean"],rtol=0,atol=1e-10)
            delta=nn[1]-nn[0];all_delta.extend(delta);accept=d["accept"]
            ids_by[scan].extend(d["point_ids"].tolist());accepted_ids[scan].extend(d["point_ids"][accept].tolist())
            result=dict(scan=scan,view=ref,n=len(delta),accepted=int(accept.sum()),decision_sha256=row["decisions_sha256"],gt_sha256=sha256(gt),decision_source=meta["source"],
                nn_means=[float(np.mean(x)) for x in nn],plane_abs_means=[float(x.mean()) for x in plane],
                changed_nn_mean_delta=float(delta[accept].mean()),changed_plane_mean_delta=float((plane[1]-plane[0])[accept].mean()),
                changed_nn_worse_fraction=float((delta[accept]>0).mean()),changed_plane_worse_fraction=float((plane[1,accept]>plane[0,accept]).mean()),
                plane_eigen_ratio_median=float(np.median(quality)),plane_eigen_ratio_p95=float(np.quantile(quality,.95)),plane_roughness_median_mm=float(np.median(roughness)))
            audit=scene/"geometry_audit.npz"
            if audit.exists():raise ValueError("Do not overwrite geometry audit")
            np.savez_compressed(audit,nn=np.array(nn),local_plane_abs=plane,plane_eigen_ratio=quality,plane_roughness_mm=roughness,accept=accept)
            result["geometry_audit_sha256"]=sha256(audit);report["completed"].append(result);save(target,report);print(json.dumps(result),flush=True)
            assert sha256(scene/"decisions.npz")==row["decisions_sha256"]
    delta=np.asarray(all_delta)
    report.update(status="complete",n=len(delta),nn_delta_mean=float(delta.mean()),nn_improvement_sum_mm=float(-delta[delta<0].sum()),nn_worsening_sum_mm=float(delta[delta>0].sum()),largest_single_improvement_mm=float(-delta.min()),nn_delta_mean_without_largest_improvement=float((delta.sum()-delta.min())/(len(delta)-1)),unique_feature_ids={str(k):len(set(v)) for k,v in ids_by.items()},unique_changed_ids={str(k):len(set(v)) for k,v in accepted_ids.items()},seconds=time.monotonic()-started)
    save(target,report)


def check_coordinates(args):
    """Original full-resolution double SfM versus current cached camera convention."""
    out=args.output.resolve();target=out/"coordinate_parity.json"
    if target.exists():raise ValueError("Do not overwrite coordinate parity")
    meta=json.loads((out/"metadata.json").read_text(encoding="utf-8"));assert meta["status"]=="complete"
    spec=importlib.util.spec_from_file_location("sfm_parity_loader",REPO/"scene/colmap_loader.py")
    loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
    verified=json.loads((args.observation_root.parent/"selected_local_verified.json").read_text())
    for rel,v in verified["files"].items():assert sha256(args.observation_root/rel)==v["sha256"]
    report=dict(status="partial",source=source_info(),decision_source=meta["source"],scans=[],limitation="No GT/no new decisions. Full-resolution coordinates and original K/pose compared, no K replacement or tau tuning.")
    def perpendicular(xy,E,q,ci,cj):
        p,_=ray_projection(xy,E,ci,cj);a,b=_ray_terms(xy,ci,cj)
        direction=(a[:,:2]*b[2]-b[:2]*a[:,2,None])*np.diag(cj["K"])[:2]
        unit=direction/np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-30)
        diff=q-p;return np.linalg.norm(diff-unit*np.sum(diff*unit,axis=1)[:,None],axis=1),p
    for row in meta["completed"]:
        scan=row["scan"];path=out/f"scan{scan}_view0000/decisions.npz"
        assert sha256(path)==row["decisions_sha256"]
        with np.load(path) as f:d={k:f[k].copy() for k in f.files}
        base=args.observation_root/f"scan{scan}"
        intr=loader.read_intrinsics_binary(base/"sparse/0/cameras.bin")
        extr={Path(v.name).stem:v for v in loader.read_extrinsics_binary(base/"sparse/0/images.bin").values()}
        cache=(HERE/"结果/20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan==37 else HERE/"结果/20261026_Huber全场景/scan40_v1")
        cached=json.loads((cache/"metadata.json").read_text(encoding="utf-8"));vm={v["name"]:v for v in cached["views"]}
        raw,half,scales={},{},{}
        for name in ["0000",*d["camera_names"].tolist()]:
            r=extr[name];k=intr[r.camera_id];fx,fy,cx,cy=k.params;h,w=vm[name]["image_size"]
            sx,sy=w/k.width,h/k.height;assert abs(sx-sy)<1e-12
            scales[name]=sx;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            w2c=np.eye(4);w2c[:3,:3]=r.qvec2rotmat();w2c[:3,3]=r.tvec
            raw[name]=dict(K=K,w2c=w2c,c2w=np.linalg.inv(w2c))
            Kh=K.copy();Kh[:2]*=sx;Kh[:2,2]-=.5
            wh=w2c.astype(np.float32).astype(float);half[name]=dict(K=Kh,w2c=wh,c2w=np.linalg.inv(wh))
        raw_values=[];half_values=[];projection_deltas=[]
        for j,name in enumerate(d["camera_names"].tolist()):
            q=d["q"][j];use=np.isfinite(q).all(axis=1)
            if not use.any():continue
            s=scales[name]
            rp,uv=perpendicular((d["xy"]+.5)/scales["0000"],d["E"],(q+.5)/s,raw["0000"],raw[name])
            hp,hv=perpendicular(d["xy"],d["E"],q,half["0000"],half[name])
            raw_values.extend((rp*s)[use]);half_values.extend(hp[use]);projection_deltas.extend(np.linalg.norm(uv*s-.5-hv,axis=1)[use])
        rv,hv,pd=map(np.asarray,(raw_values,half_values,projection_deltas))
        report["scans"].append(dict(scan=scan,pairs=len(rv),raw_scaled_perpendicular_median_px=float(np.median(rv)),current_perpendicular_median_px=float(np.median(hv)),max_perpendicular_difference_px=float(np.max(np.abs(rv-hv))),max_projection_difference_px=float(pd.max()),half_pixel_classification_disagreements=int(((rv>.5)!=(hv>.5)).sum()),decision_sha256=row["decisions_sha256"]))
        assert sha256(path)==row["decisions_sha256"];save(target,report)
    report["status"]="complete";save(target,report);print(json.dumps(report),flush=True)


def diagnose(args):
    """No new decisions/GT: explain the sealed all-track failure funnel."""
    out = args.output.resolve()
    meta = json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["parameters"]["all_track_views"]
    target = out/"diagnostics.json"
    if target.exists():raise ValueError("Do not overwrite completed diagnostics")
    report = dict(source=source_info(), decision_source=meta["source"], status="partial", scans=[],
        limitation="No scan GT loaded. Same calibrated SfM solution; not independent matching truth. Solver-invalid can be unbounded rather than impossible. No threshold/view/output change.")
    spec = importlib.util.spec_from_file_location("sfm_diagnostic_loader", REPO/"scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec);sys.modules[spec.name] = loader;spec.loader.exec_module(loader)
    frozen = json.loads((HERE/"G1_Huber原缓存SHA.json").read_text(encoding="utf-8"))
    verified = json.loads((args.observation_root.parent/"selected_local_verified.json").read_text())
    for rel,v in verified["files"].items():assert sha256(args.observation_root/rel)==v["sha256"]
    def summary(v):
        v = np.asarray(v);v = v[np.isfinite(v)]
        return dict(n=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),fraction_le_half=float((v<=.5).mean())) if len(v) else dict(n=0)
    for row in meta["completed"]:
        scan = row["scan"]; scene = out/f"scan{scan}_view0000"
        assert sha256(scene/"decisions.npz")==row["decisions_sha256"]
        with np.load(scene/"decisions.npz") as f:d={k:f[k].copy() for k in f.files}
        source = args.observation_root/f"scan{scan}"
        intr = loader.read_intrinsics_binary(source/"sparse/0/cameras.bin")
        extr = {Path(v.name).stem:v for v in loader.read_extrinsics_binary(source/"sparse/0/images.bin").values()}
        cache = (HERE/"结果/20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan==37 else HERE/"结果/20261026_Huber全场景/scan40_v1")
        cached = json.loads((cache/"metadata.json").read_text(encoding="utf-8"));vm={v["name"]:v for v in cached["views"]}
        cameras={};masks={};tracks={};names=d["camera_names"].tolist()
        for name in ["0000",*names]:
            frame=cache/vm[name]["file"];assert sha256(frame)==frozen[str(scan)][vm[name]["file"]]
            with np.load(frame) as f:masks[name]=f["gt_mask"].copy().squeeze().astype(float)
            r=extr[name];k=intr[r.camera_id];assert k.model=="PINHOLE"
            h,w=vm[name]["image_size"];sx,sy=w/k.width,h/k.height;fx,fy,cx,cy=k.params
            K=np.array([[sx*fx,0,sx*cx-.5],[0,sy*fy,sy*cy-.5],[0,0,1.]])
            w2c=np.eye(4);w2c[:3,:3]=r.qvec2rotmat();w2c[:3,3]=r.tvec;w2c=w2c.astype(np.float32).astype(float)
            cameras[name]=dict(K=K,w2c=w2c,c2w=np.linalg.inv(w2c),image_size=(h,w))
            ids,count=np.unique(r.point3D_ids[r.point3D_ids>=0],return_counts=True)
            tracks[name]=set(ids[count==1].tolist())
        xy,E,roles=d["xy"],d["E"],d["role_views"];assigned=(roles>=0).all(axis=0)
        available=np.array([sum(int(pid) in tracks[name] for name in names) for pid in d["point_ids"]])
        geometry=np.zeros(roles.shape,bool);maskok=geometry.copy();off=geometry.copy()
        residual=np.full(roles.shape,np.nan);perp=residual.copy()
        for j,name in enumerate(names):
            use=np.any(roles==j,axis=0)
            if not use.any():continue
            q=d["q"][j];p0,z=ray_projection(xy,E,cameras["0000"],cameras[name])
            a,b=_ray_terms(xy,cameras["0000"],cameras[name])
            direction=(a[:,:2]*b[2]-b[:2]*a[:,2,None])*np.diag(cameras[name]["K"])[:2]
            unit=direction/np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-30)
            diff=q-p0;pd=np.linalg.norm(diff-unit*np.sum(diff*unit,axis=1)[:,None],axis=1)
            _,_,ok=depth_interval(xy,E,q,cameras["0000"],cameras[name],tau=.5)
            m,inside=bilinear(masks[name],q);mok=inside&(m>=1-1e-9)
            for slot in range(5):
                take=roles[slot]==j
                geometry[slot,take]=ok[take];maskok[slot,take]=mok[take]
                off[slot,take]=pd[take]>.5+1e-8
                residual[slot,take]=np.linalg.norm(diff[take],axis=1);perp[slot,take]=pd[take]
        np.testing.assert_array_equal(geometry&maskok,d["valid"])
        def funnel(slots):
            use=roles[slots]>=0;g,m=geometry[slots],maskok[slots]
            return dict(assigned_pairs=int(use.sum()),valid_pairs=int((g&m&use).sum()),solver_only_reject=int((~g&m&use).sum()),mask_only_reject=int((g&~m&use).sum()),both_reject=int((~g&~m&use).sum()),off_epipolar_half_px=int((off[slots]&use).sum()),solver_reject_without_off_epipolar=int((~g&~off[slots]&use).sum()),E_residual_px=summary(residual[slots][use]),perpendicular_residual_px=summary(perp[slots][use]))
        valid=d["valid"];lower,upper=d["lower"],d["upper"]
        candidate=np.clip(E,d["lo"],d["hi"])
        gv=valid[3:].all(axis=0)
        insideD=((candidate>=lower[3:]-1e-12)&(candidate<=upper[3:]+1e-12)).all(axis=0)
        excludesE=((E<lower[3:]-1e-12)|(E>upper[3:]+1e-12)).any(axis=0)
        gate_reject=d["reason"]=="gate_unverified"
        joint_lo=np.maximum(d["lo"],np.max(lower[3:],axis=0));joint_hi=np.minimum(d["hi"],np.min(upper[3:],axis=0))
        notref=d["reason"]=="baseline_not_refuted"
        insideE=gv&((E>=lower[3:]-1e-12)&(E<=upper[3:]+1e-12)).all(axis=0)
        reason=dict(zip(*np.unique(d["reason"],return_counts=True)))
        data=dict(scan=scan,samples=len(E),decision_sha256=row["decisions_sha256"],role_assigned=int(assigned.sum()),raw_tracks_below_six=int((~assigned&(available<6)).sum()),raw_tracks_six_but_geometry_role_short=int((~assigned&(available>=6)).sum()),assigned_but_fit_below_three=int((assigned&(d["fit_n"]<3)).sum()),reason_counts={k:int(v) for k,v in reason.items()},fit=funnel([0,1,2]),gate=funnel([3,4]),gate_reject=dict(total=int(gate_reject.sum()),invalid_interval=int((gate_reject&~gv).sum()),valid_but_candidate_outside=int((gate_reject&gv&~insideD).sum()),valid_contains_candidate_but_E_not_excluded=int((gate_reject&gv&insideD&~excludesE).sum()),candidate_outside_but_joint_intersection_nonempty=int((gate_reject&gv&~insideD&(joint_lo<=joint_hi)).sum())),baseline_in_fit=dict(n=int(notref.sum()),also_in_both_valid_gate=int((notref&insideE).sum())),raw_track_count_histogram={str(k):int((available==k).sum()) for k in np.unique(available)})
        report["scans"].append(data);save(target,report);print(json.dumps(data),flush=True)
        assert sha256(scene/"decisions.npz")==row["decisions_sha256"]
    report["status"]="complete";save(target,report)


def run(args):
    ref = args.reference
    if len(ref)!=4 or not ref.isdigit():raise ValueError("Four-digit reference required")
    out = args.output.resolve()
    if not args.output.is_absolute() or out.exists():
        raise ValueError("New independent output required")
    out.mkdir(parents=True)
    state = dict(status="partial", source=source_info(), completed=[], parameters=dict(
        scans=[37, 40], reference=ref, samples=512, exclude_original_probe_features=(ref!="0000"), tau_pixels=.5,
        fit=[0, 2, 4, 6], gate=[1, 3], audit=[5, 7], seed=20261026,
        all_track_views=args.all_track_views,joint_constraints=args.joint_constraints),
        limitation="Sparse SfM feature observations already used for model initialization, not new training holdout or depth truth. E is bilinearly sampled at fractional feature coordinates, not a native pixel rerender. Scan NN is a diagnostic, not pixel GT or mesh gain. Total true matching/calibration bound .5px remains an assumption.")
    if args.all_track_views:
        state["parameters"].update(fit="3 largest geometric parallax among observed tracks",
            gate="next2 geometric parallax",audit="all remaining observed tracks",minimum_available_nonref_views=6)
    if args.joint_constraints:
        if not args.all_track_views:raise ValueError("Joint variant only on identical all-track roles")
        state["parameters"]["estimator"]="nearest E in joint intersection of all five valid decision intervals; no gate preprojection heuristic"
    save(out/"metadata.json", state)
    started = time.monotonic()
    verified = json.loads((args.observation_root.parent/"selected_local_verified.json").read_text())
    for rel, v in verified["files"].items():
        if sha256(args.observation_root/rel) != v["sha256"]:
            raise ValueError("Observation changed")
    spec = importlib.util.spec_from_file_location("sfm_colmap_loader", REPO/"scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec);sys.modules[spec.name] = loader;spec.loader.exec_module(loader)
    results = HERE/"结果"
    frozen = json.loads((HERE/"G1_Huber原缓存SHA.json").read_text(encoding="utf-8"))
    for scan in (37, 40):
        if time.monotonic()-started > args.max_seconds or (out/"stop_after_scene").exists():
            raise InterruptedError("Local finite deadline/scene stop")
        cache = (results/"20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan == 37
                 else results/"20261026_Huber全场景/scan40_v1")
        meta = json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
        source = args.observation_root/f"scan{scan}"
        intr = loader.read_intrinsics_binary(source/"sparse/0/cameras.bin")
        extr = {Path(v.name).stem:v for v in loader.read_extrinsics_binary(source/"sparse/0/images.bin").values()}
        names = next(json.loads(line)["nearest_name"] for line in (cache/"multi_view.json").read_text().splitlines()
                     if json.loads(line)["ref_name"] == ref)
        if args.all_track_views:
            names = sorted(name for name in extr if name != ref)
        frames, cameras, features = {}, {}, {}
        transform = np.asarray(meta["alignment"]["transform"])
        scale = float(meta["alignment"]["scale_to_dtu_mm"])
        vm_by = {v["name"]:v for v in meta["views"]}
        for name in [ref, *names]:
            vm = vm_by[name];assert sha256(cache/vm["file"]) == frozen[str(scan)][vm["file"]]
            with np.load(cache/vm["file"]) as f:
                frames[name] = {k:f[k].copy().squeeze() for k in ("expected", "gt_mask")}
            row = extr[name];k = intr[row.camera_id];assert k.model == "PINHOLE"
            h, w = vm["image_size"];sx, sy = w/k.width, h/k.height
            fx, fy, cx, cy = k.params
            K = np.array([[sx*fx, 0, sx*cx-.5], [0, sy*fy, sy*cy-.5], [0, 0, 1.]])
            np.testing.assert_allclose([K[0, 0], K[1, 1]], vm["focal"], rtol=0, atol=3e-4)
            np.testing.assert_allclose(K[:2, 2], [(w-1)/2., (h-1)/2.], rtol=0, atol=1e-8)
            w2c = np.eye(4);w2c[:3, :3] = row.qvec2rotmat();w2c[:3, 3] = row.tvec
            w2c = w2c.astype(np.float32).astype(float)
            _check_camera_projection(source/"cameras.npz",dict(image_name=name,world_view_transform=w2c.T.tolist()),transform)
            cameras[name] = dict(K=K,w2c=w2c,c2w=np.linalg.inv(w2c),image_size=(h,w))
            ids = row.point3D_ids;xy = row.xys*np.array([sx,sy])-.5
            good = ids >= 0
            unique, counts = np.unique(ids[good], return_counts=True)
            allowed = set(unique[counts==1].tolist())
            features[name] = {int(pid):uv for pid,uv in zip(ids,xy) if int(pid) in allowed}
        ids = np.array(sorted(features[ref]),dtype=np.int64)
        xy = np.array([features[ref][int(pid)] for pid in ids])
        E, valid = bilinear(frames[ref]["expected"],xy)
        mask, inside = bilinear(frames[ref]["gt_mask"].astype(float),xy)
        unseen=np.ones(len(ids),bool)
        if ref!="0000":
            with np.load(args.observation_root.parent/f"sfm_interval_v1/scan{scan}_view0000/decisions.npz") as old:
                unseen=~np.isin(ids,old["point_ids"])
        eligible = np.flatnonzero(valid & inside & np.isfinite(E) & (E>0) & (mask>=1-1e-9) & unseen)
        take = np.sort(np.random.default_rng(20261026+scan).choice(eligible,size=min(512,len(eligible)),replace=False))
        ids,xy,E = ids[take],xy[take],E[take].astype(float)
        if len(ids) == 0:raise ValueError("No foreground sparse observations")
        if args.all_track_views and ref == "0000":
            previous = args.observation_root.parent/f"sfm_interval_v1/scan{scan}_view0000/decisions.npz"
            with np.load(previous) as old:
                np.testing.assert_array_equal(ids,old["point_ids"])
                np.testing.assert_array_equal(xy,old["xy"])
                np.testing.assert_array_equal(E,old["E"])
        roles = None
        if args.all_track_views:
            # Availability and camera geometry only: no q residual/scan score
            # chooses roles. Reserve >=1 observed view for coordinate audit.
            seen = np.array([[int(pid) in features[name] for pid in ids] for name in names])
            parallax = np.zeros(seen.shape)
            step = np.maximum(np.abs(E), 1.)*1e-5
            for j, name in enumerate(names):
                p0,z0 = ray_projection(xy,E,cameras[ref],cameras[name])
                p1,z1 = ray_projection(xy,E+step,cameras[ref],cameras[name])
                score = np.linalg.norm(p1-p0,axis=1)/step
                parallax[j] = np.where(seen[j] & np.isfinite(score) & (z0>0) & (z1>0),score,0.)
            roles = np.full((5,len(ids)),-1,dtype=int)
            for k in range(len(ids)):
                order = np.argsort(-parallax[:,k],kind="stable")
                available = order[parallax[order,k]>1e-12]
                if len(available)>=6:roles[:,k] = available[:5]
            lower = np.full((5,len(ids)),np.nan)
            decision_views = range(len(names));fit_roles,gate_roles = [0,1,2],[3,4]
        else:
            lower = np.full((8,len(ids)),np.nan)
            decision_views = (0,2,4,6,1,3);fit_roles,gate_roles = [0,2,4,6],[1,3]
        upper = lower.copy();valid = np.zeros(lower.shape,bool)
        q_all = np.full((len(names),len(ids),2),np.nan)
        # Only fit/gate observations enter the decision. Audit feature matches are read afterwards.
        for j in decision_views:
            name = names[j]
            used = np.any(roles==j,axis=0) if roles is not None else np.ones(len(ids),bool)
            q = np.array([features[name].get(int(pid),[np.nan,np.nan]) if take else [np.nan,np.nan] for pid,take in zip(ids,used)])
            lo,hi,ok = depth_interval(xy,E,q,cameras[ref],cameras[name],tau=.5)
            mask,inside = bilinear(frames[name]["gt_mask"].astype(float),q)
            if roles is None:
                lower[j],upper[j],valid[j] = lo,hi,ok & inside & (mask>=1-1e-9)
            else:
                for slot in range(5):
                    use = roles[slot]==j
                    lower[slot,use],upper[slot,use],valid[slot,use] = lo[use],hi[use],(ok & inside & (mask>=1-1e-9))[use]
            q_all[j] = q
        result = (project_joint_intervals(E,lower,upper,valid) if args.joint_constraints
                  else solve_intervals(E,lower,upper,valid,fit=fit_roles,gate=gate_roles))
        D,accept = result["selected"],result["accept"]
        scene = out/f"scan{scan}_view{ref}";scene.mkdir()
        np.savez_compressed(scene/"decisions.npz",point_ids=ids,xy=xy,E=E,selected=D,accept=accept,
                            lower=lower,upper=upper,valid=valid,q=q_all,
                            role_views=roles if roles is not None else np.empty((0,len(ids)),int),
                            camera_names=np.array(names),**{k:v for k,v in result.items() if k not in ("selected","accept")})
        sealed = sha256(scene/"decisions.npz")
        audit_errors = []
        audit_available = 0
        for j in (range(len(names)) if roles is not None else (5,7)):
            name = names[j]
            unused = ~np.any(roles==j,axis=0) if roles is not None else np.ones(len(ids),bool)
            q = np.array([features[name].get(int(pid),[np.nan,np.nan]) if take else [np.nan,np.nan] for pid,take in zip(ids,unused)])
            audit_available += int((accept & np.isfinite(q).all(axis=1)).sum())
            pE,_ = ray_projection(xy,E,cameras[ref],cameras[name]);pD,_ = ray_projection(xy,D,cameras[ref],cameras[name])
            audit_errors.append(np.column_stack((np.linalg.norm(pE-q,axis=1),np.linalg.norm(pD-q,axis=1))))
        error = np.stack(audit_errors);audit_take = accept[None,:] & np.isfinite(error).all(axis=2)
        assert sha256(scene/"decisions.npz") == sealed
        from scipy.spatial import cKDTree
        from plyfile import PlyData
        gt = results/f"20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"
        assert sha256(gt) == meta["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]
        vertex = PlyData.read(str(gt))["vertex"];tree = cKDTree(np.column_stack([np.asarray(vertex[k],float) for k in ("x","y","z")]))
        rays = np.column_stack((xy,np.ones(len(xy)))) @ np.linalg.inv(cameras[ref]["K"]).T
        pose = cameras[ref]["c2w"];errors = {}
        for label,depth in (("expected",E),("sfm_bounded",D)):
            world = (rays*depth[:,None]) @ pose[:3,:3].T+pose[:3,3]
            points = world @ transform[:3,:3].T+transform[:3,3]
            errors[label] = tree.query(points,k=1,eps=0,workers=1)[0]
        keys,counts = np.unique(result["reason"],return_counts=True)
        row = dict(scan=scan,view=ref,samples=len(ids),accepted=int(accept.sum()),
            reason_counts=dict(zip(keys.tolist(),counts.astype(int).tolist())),decisions_sha256=sealed,
            audit_pairs=int(audit_take.sum()),audit_available_track_pairs=audit_available,
            audit_coverage=float(audit_take.sum()/audit_available) if audit_available else None,
            audit_coverage_definition="finite pairs / available unused-track pairs for changed features",
            audit_before_mean_px=float(error[:,:,0][audit_take].mean()) if audit_take.any() else None,
            audit_after_mean_px=float(error[:,:,1][audit_take].mean()) if audit_take.any() else None,
            nn_proxy={k:dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in errors.items()},
            accepted_nn_delta_mean_mm=float((errors["sfm_bounded"]-errors["expected"])[accept].mean()) if accept.any() else None,
            accepted_nn_worse_fraction=float((errors["sfm_bounded"][accept]>errors["expected"][accept]).mean()) if accept.any() else None,
            gt_loaded_after_sealed_decision=True)
        np.savez_compressed(scene/"audit.npz",errors=error,valid=audit_take)
        save(scene/"metadata.json",row);state["completed"].append(row);save(out/"metadata.json",state);print(json.dumps(row),flush=True)
    state.update(status="complete",seconds=time.monotonic()-started);save(out/"metadata.json",state)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--observation-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--max-seconds",type=int,default=900)
    p.add_argument("--all-track-views",action="store_true",help="Same interval rule; geometry-only fit3/gate2 among all available tracks, reserve audit")
    p.add_argument("--diagnose-only",action="store_true",help="No new depth decisions/GT; explain sealed all-track intervals")
    p.add_argument("--coordinate-check",action="store_true",help="No new depth decisions/GT; original raw SfM versus current coordinate convention")
    p.add_argument("--reference",default="0000",help="Nonzero references exclude original 0000 probe feature IDs")
    p.add_argument("--surface-audit",action="store_true",help="Post-hoc local-plane/paired-NN audit of fixed four-reference holdout; never selects depth")
    p.add_argument("--joint-constraints",action="store_true",help="Same five decision roles, joint feasible-set projection instead of fit-then-gate")
    args=p.parse_args()
    if args.surface_audit:audit_surface(args)
    elif args.coordinate_check:check_coordinates(args)
    elif args.diagnose_only:diagnose(args)
    else:run(args)
