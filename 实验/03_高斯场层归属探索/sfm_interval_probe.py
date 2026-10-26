"""Fixed sparse-SfM observation contrast for bounded stereo, never scan-GT selection."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import numpy as np
from bounded_correspondence import depth_interval, solve_intervals, ray_projection
from photo_objective import bilinear
from photo_surface_probe import HERE, REPO, sha256, save, source_info
from consensus_depth import _check_camera_projection


def run(args):
    out = args.output.resolve()
    if not args.output.is_absolute() or out.exists():
        raise ValueError("New independent output required")
    out.mkdir(parents=True)
    state = dict(status="partial", source=source_info(), completed=[], parameters=dict(
        scans=[37, 40], reference="0000", samples=512, tau_pixels=.5,
        fit=[0, 2, 4, 6], gate=[1, 3], audit=[5, 7], seed=20261026,
        all_track_views=args.all_track_views),
        limitation="Sparse SfM feature observations already used for model initialization, not new training holdout or depth truth. E is bilinearly sampled at fractional feature coordinates, not a native pixel rerender. Scan NN is a diagnostic, not pixel GT or mesh gain. Total true matching/calibration bound .5px remains an assumption.")
    if args.all_track_views:
        state["parameters"].update(fit="3 largest geometric parallax among observed tracks",
            gate="next2 geometric parallax",audit="all remaining observed tracks",minimum_available_nonref_views=6)
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
                     if json.loads(line)["ref_name"] == "0000")
        if args.all_track_views:
            names = sorted(name for name in extr if name != "0000")
        frames, cameras, features = {}, {}, {}
        transform = np.asarray(meta["alignment"]["transform"])
        scale = float(meta["alignment"]["scale_to_dtu_mm"])
        vm_by = {v["name"]:v for v in meta["views"]}
        for name in ["0000", *names]:
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
        ids = np.array(sorted(features["0000"]),dtype=np.int64)
        xy = np.array([features["0000"][int(pid)] for pid in ids])
        E, valid = bilinear(frames["0000"]["expected"],xy)
        mask, inside = bilinear(frames["0000"]["gt_mask"].astype(float),xy)
        eligible = np.flatnonzero(valid & inside & np.isfinite(E) & (E>0) & (mask>=1-1e-9))
        take = np.sort(np.random.default_rng(20261026+scan).choice(eligible,size=min(512,len(eligible)),replace=False))
        ids,xy,E = ids[take],xy[take],E[take].astype(float)
        if len(ids) == 0:raise ValueError("No foreground sparse observations")
        if args.all_track_views:
            previous = out.parent/f"sfm_interval_v1/scan{scan}_view0000/decisions.npz"
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
                p0,z0 = ray_projection(xy,E,cameras["0000"],cameras[name])
                p1,z1 = ray_projection(xy,E+step,cameras["0000"],cameras[name])
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
            lo,hi,ok = depth_interval(xy,E,q,cameras["0000"],cameras[name],tau=.5)
            mask,inside = bilinear(frames[name]["gt_mask"].astype(float),q)
            if roles is None:
                lower[j],upper[j],valid[j] = lo,hi,ok & inside & (mask>=1-1e-9)
            else:
                for slot in range(5):
                    use = roles[slot]==j
                    lower[slot,use],upper[slot,use],valid[slot,use] = lo[use],hi[use],(ok & inside & (mask>=1-1e-9))[use]
            q_all[j] = q
        result = solve_intervals(E,lower,upper,valid,fit=fit_roles,gate=gate_roles)
        D,accept = result["selected"],result["accept"]
        scene = out/f"scan{scan}_view0000";scene.mkdir()
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
            pE,_ = ray_projection(xy,E,cameras["0000"],cameras[name]);pD,_ = ray_projection(xy,D,cameras["0000"],cameras[name])
            audit_errors.append(np.column_stack((np.linalg.norm(pE-q,axis=1),np.linalg.norm(pD-q,axis=1))))
        error = np.stack(audit_errors);audit_take = accept[None,:] & np.isfinite(error).all(axis=2)
        assert sha256(scene/"decisions.npz") == sealed
        from scipy.spatial import cKDTree
        from plyfile import PlyData
        gt = results/f"20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"
        assert sha256(gt) == meta["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]
        vertex = PlyData.read(str(gt))["vertex"];tree = cKDTree(np.column_stack([np.asarray(vertex[k],float) for k in ("x","y","z")]))
        rays = np.column_stack((xy,np.ones(len(xy)))) @ np.linalg.inv(cameras["0000"]["K"]).T
        pose = cameras["0000"]["c2w"];errors = {}
        for label,depth in (("expected",E),("sfm_bounded",D)):
            world = (rays*depth[:,None]) @ pose[:3,:3].T+pose[:3,3]
            points = world @ transform[:3,:3].T+transform[:3,3]
            errors[label] = tree.query(points,k=1,eps=0,workers=1)[0]
        keys,counts = np.unique(result["reason"],return_counts=True)
        row = dict(scan=scan,view="0000",samples=len(ids),accepted=int(accept.sum()),
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
    run(p.parse_args())
