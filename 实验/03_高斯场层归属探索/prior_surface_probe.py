"""Frozen external shape prior contrast; empirical residual bands are not physical bounds."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from photo_surface_probe import HERE, REPO, sha256, save, source_info
from photo_objective import bilinear
from surface_patch import triangulate_track

DEPENDENCY_SHA = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


def calibrate(raw, z, fit):
    """Ten fixed Huber IRLS steps; disjoint ID-parity calibration, empirical 95% q."""
    raw, z, fit = np.asarray(raw, float), np.asarray(z, float), np.asarray(fit, bool)
    result = dict(valid=False, reason="insufficient_anchors", a=None, b=None, q=None)
    good = np.isfinite(raw) & np.isfinite(z) & (z > 0)
    train, cal = good & fit, good & ~fit
    result.update(n_fit=int(train.sum()), n_calibration=int(cal.sum()))
    if min(train.sum(), cal.sum()) < 20:
        return result
    x, y = raw[train], 1 / z[train]
    if not np.isfinite(1/z[good]).all():
        result["reason"] = "nonfinite_inverse_depth"; return result
    center, scale = x.mean(), x.std()
    if scale <= 1e-10 * max(1., np.max(np.abs(x))):
        result["reason"] = "degenerate_raw_prior"; return result
    A = np.column_stack(((x-center)/scale, np.ones(len(x))))
    weights = np.ones(len(x))
    for _ in range(10):
        weighted = A * np.sqrt(weights[:, None])
        if np.linalg.cond(weighted) > 1e10:
            result["reason"] = "ill_conditioned_fit"; return result
        coef = np.linalg.lstsq(weighted, y*np.sqrt(weights), rcond=None)[0]
        residual = y - A @ coef
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        cutoff = 1.345 * max(mad, 1e-12 * max(1., np.median(np.abs(y))))
        weights = np.minimum(1., cutoff / np.maximum(np.abs(residual), 1e-30))
    a, b = float(coef[0]/scale), float(coef[1]-coef[0]*center/scale)
    if not np.isfinite([a, b]).all() or a <= 0 or np.any(a*raw[good]+b <= 0):
        result["reason"] = "nonpositive_or_invalid_mapping"; return result
    residual = np.abs(1/z[cal] - (a*raw[cal]+b))
    rank = min(len(residual), int(np.ceil((len(residual)+1)*.95)))
    q = float(np.sort(residual)[rank-1])
    if not np.isfinite(q):
        result["reason"] = "nonfinite_empirical_residual"; return result
    result.update(valid=True, reason="ok", a=a, b=b, q=q, quantile_rank=rank)
    return result


def readout(E, raw, calibration):
    """Unknown/unbounded bands preserve E; teacher independently falls back if invalid."""
    E, raw = np.asarray(E, float), np.asarray(raw, float)
    teacher, selected = E.copy(), E.copy()
    lo, hi = np.full(E.shape, np.nan), np.full(E.shape, np.nan)
    status = np.full(E.shape, "unknown_calibration", dtype="U32")
    if calibration["valid"]:
        mu = calibration["a"]*raw + calibration["b"]; q = calibration["q"]
        valid = np.isfinite(mu) & (mu > 0)
        with np.errstate(over="ignore",divide="ignore",invalid="ignore"):inverse=1/mu
        valid &= np.isfinite(inverse) & (inverse > 0)
        teacher[valid] = inverse[valid]
        bounded = valid & np.isfinite(q) & (q >= 0) & (mu-q > 0) & np.isfinite(mu+q)
        lo[bounded], hi[bounded] = 1/(mu[bounded]+q), 1/(mu[bounded]-q)
        bounded &= np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
        status[:] = "unknown_or_unbounded"
        status[bounded] = "within_band"
        selected[bounded] = np.clip(E[bounded], lo[bounded], hi[bounded])
        status[bounded & (selected != E)] = "projected"
    return dict(teacher=teacher, selected=selected, lower=lo, upper=hi,
                accept=selected != E, status=status)


def dependency_info(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    record = dict(remote=git("remote", "get-url", "origin"), commit=git("rev-parse", "HEAD"))
    if (record["remote"] != "https://github.com/DepthAnything/Depth-Anything-V2.git"
            or record["commit"] != DEPENDENCY_SHA or git("status", "--porcelain")):
        raise ValueError("Require exact clean official DepthAnythingV2 dependency")
    return record


def run(args):
    from PIL import Image
    from consensus_depth import _check_camera_projection
    out, obs = args.output.resolve(), args.observation_root.resolve()
    if not args.output.is_absolute() or out.exists() or obs == out or obs in out.parents:
        raise ValueError("Require a new absolute independent output directory")
    info = source_info(); dependency = dependency_info(args.dependency_root)
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA mismatch")
    out.mkdir(parents=True); started = time.monotonic()
    state = dict(status="partial", source=info, dependency=dependency, completed=[],
                 checkpoint_sha256=args.checkpoint_sha256, parameters=vars(args).copy(),
                 design=dict(scans=[37,40],anchor_seed=20261027,max_anchors=256,max_pixels=512,fit="even ID",calibration="odd ID",irls_steps=10,huber=1.345,empirical_quantile=.95,input_size=518),
                 limitation="External frozen affine inverse-depth prior; SfM anchors share joint BA/training inputs. ID parity is not independent training holdout. LOO spread is empirical stability; calibration residual band has no physical coverage guarantee. Scan NN is not pixel GT, mesh or Overall. No GauDepth CUDA extension is loaded.")
    state["parameters"] = {k:str(v) if isinstance(v, Path) else v for k,v in state["parameters"].items()}
    save(out/"metadata.json", state)
    def check():
        if time.monotonic()-started > args.max_seconds or (out/"stop_after_scene").exists():
            raise InterruptedError("Finite probe deadline/stop")
    try:
        verified_path = obs.parent/"selected_local_verified.json"
        verified = json.loads(verified_path.read_text())
        if verified["status"] != "selected_subset_complete": raise ValueError("Incomplete inputs")
        if len(verified["files"]) != 24: raise ValueError("Expected fixed 24 observation files")
        for rel, record in verified["files"].items():
            check()
            if sha256(obs/rel) != record["sha256"]: raise ValueError("Input SHA mismatch: "+rel)
        state.update(observation_manifest_sha256=sha256(verified_path), observation_files=verified["files"])
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.dependency_root.resolve()))
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        if not torch.cuda.is_available(): raise ValueError("CUDA inference required")
        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48,96,192,384])
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location="cpu"))
        model.eval().requires_grad_(False).cuda()
        state.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        spec = importlib.util.spec_from_file_location("prior_colmap", REPO/"scene/colmap_loader.py")
        loader = importlib.util.module_from_spec(spec); sys.modules[spec.name] = loader; spec.loader.exec_module(loader)
        frozen = json.loads((HERE/"G1_Huber原缓存SHA.json").read_text(encoding="utf-8"))
        results = obs.parent.parent
        for scan in (37, 40):
            check(); scene = out/f"scan{scan}_view{args.reference}"; scene.mkdir()
            cache = results/("20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan == 37 else "20261026_Huber全场景/scan40_v1")
            meta = json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
            if meta["status"] != "complete": raise ValueError("Incomplete source cache")
            views = {v["name"]:v for v in meta["views"]}; ref = args.reference; vm = views[ref]
            frame = cache/vm["file"]; frame_sha = sha256(frame)
            if frame_sha != frozen[str(scan)][vm["file"]] or frame_sha != vm["sha256"]: raise ValueError("Frozen cache SHA mismatch")
            with np.load(frame) as f: Emap, mask = f["expected"].copy(), f["gt_mask"].copy()
            transform = np.asarray(meta["alignment"]["transform"], float)
            intr = loader.read_intrinsics_binary(obs/f"scan{scan}/sparse/0/cameras.bin")
            extr = {Path(r.name).stem:r for r in loader.read_extrinsics_binary(obs/f"scan{scan}/sparse/0/images.bin").values()}
            if len(extr) != 49 or set(extr) != set(views): raise ValueError("Require all 49 cameras")
            cameras, tracks, checks = {}, {}, []
            for name, row in extr.items():
                check(); k = intr[row.camera_id]; h,w = views[name]["image_size"]
                if k.model != "PINHOLE": raise ValueError("PINHOLE required")
                fx,fy,cx,cy = k.params; sx,sy = w/k.width,h/k.height
                K = np.array([[sx*fx,0,sx*cx-.5],[0,sy*fy,sy*cy-.5],[0,0,1.]])
                np.testing.assert_allclose([K[0,0],K[1,1]], views[name]["focal"], atol=3e-4, rtol=0)
                w2c = np.eye(4); w2c[:3,:3] = row.qvec2rotmat(); w2c[:3,3] = row.tvec; w2c = w2c.astype(np.float32).astype(float)
                re,te = _check_camera_projection(obs/f"scan{scan}/cameras.npz", dict(image_name=name,world_view_transform=w2c.T.tolist()), transform)
                cameras[name] = dict(K=K,w2c=w2c,c2w=np.linalg.inv(w2c))
                checks.append(dict(name=name,rotation_error=re,translation_error=te))
                ids,counts=np.unique(row.point3D_ids[row.point3D_ids>=0],return_counts=True)
                unique=set(ids[counts==1].tolist());checks[-1]["ambiguous_point_ids"]=int((counts>1).sum())
                for pid,xy in zip(row.point3D_ids, row.xys):
                    if int(pid) in unique: tracks.setdefault(int(pid), {})[name] = xy*np.array([sx,sy])-.5
            h,w = vm["image_size"]; Emap,mask = Emap.reshape(h,w),mask.reshape(h,w)
            image = [p for p in (obs/f"scan{scan}/images").iterdir() if p.stem == ref]
            if len(image) != 1: raise ValueError("Reference original image unavailable")
            with Image.open(image[0]) as im:
                k = intr[extr[ref].camera_id]
                if im.size != (k.width,k.height): raise ValueError("Original image dimensions")
                bgr = np.asarray(im.convert("RGB").resize((w,h)))[:,:,::-1].copy()
            check(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
            with torch.inference_mode(): raw = model.infer_image(bgr, input_size=518).astype(np.float32)
            torch.cuda.synchronize(); inference_seconds = time.monotonic()-tick; check()
            if raw.shape != (h,w): raise ValueError("Prior dimensions")
            np.savez_compressed(scene/"raw_prior.npz",raw=raw)
            row = extr[ref]; xyref = np.array([tracks.get(int(pid),{}).get(ref,[np.nan,np.nan]) for pid in row.point3D_ids])
            fg,inside = bilinear(mask, xyref); candidates = [int(pid) for pid,ok in zip(row.point3D_ids,inside & (fg>=.5)) if ok and pid>=0 and ref in tracks.get(int(pid),{}) and len(tracks[int(pid)])>=5]
            rng = np.random.default_rng(20261027+scan); ids = np.sort(rng.choice(np.unique(candidates), min(256,len(set(candidates))), replace=False))
            projected = np.full((len(ids),2),np.nan); z = np.full(len(ids),np.nan); values=z.copy(); spread=z.copy(); reasons=[]; tick=time.monotonic()
            ci = cameras[ref]
            for i,pid in enumerate(ids):
                check(); tri = triangulate_track(tracks[int(pid)],cameras,ref); reason=tri["reason"]
                if tri["valid"]:
                    q = ci["w2c"]@np.r_[tri["xyz"],1]; uv = ci["K"]@q[:3]; uv=uv[:2]/uv[2]
                    fg,ok=bilinear(mask,uv[None]); val,vok=bilinear(raw,uv[None])
                    if q[2]>0 and ok[0] and fg[0]>=.5 and vok[0] and np.isfinite(val[0]):
                        projected[i],z[i],values[i],spread[i]=uv,q[2],val[0],tri["max_loo_displacement"]
                    else: reason="invalid_projected_anchor"
                reasons.append(reason)
            triangulation_seconds=time.monotonic()-tick; tick=time.monotonic(); fit=ids%2==0; calibration=calibrate(values,z,fit); calibration_seconds=time.monotonic()-tick
            tick=time.monotonic(); eligible=np.flatnonzero((mask>=.5)&np.isfinite(Emap)&(Emap>0)); old_sha=None
            if ref == "0000":
                old=obs.parent/f"photo_probe_v1/scan{scan}_view0000/decisions.npz"; old_sha=sha256(old)
                with np.load(old) as f: xy=f["xy"].copy(); old_E=f["E"].copy()
                if len(xy)!=512 or not np.equal(xy,np.rint(xy)).all(): raise ValueError("Original 512 integer pixels required")
                x,y=xy.astype(int).T; E=Emap[y,x].astype(float); np.testing.assert_array_equal(E,old_E)
            else:
                index=np.sort(rng.choice(eligible,min(512,len(eligible)),replace=False)); y,x=np.divmod(index,w); xy=np.column_stack((x,y)); E=Emap[y,x].astype(float)
            if not np.all((mask[y,x]>=.5)&np.isfinite(E)&(E>0)): raise ValueError("Invalid common denominator")
            decision=readout(E,raw[y,x],calibration); readout_seconds=time.monotonic()-tick
            anchor_valid=np.isfinite(values)&np.isfinite(z)&(z>0)
            np.savez_compressed(scene/"calibration.npz",ids=ids,fit=fit&anchor_valid,calibration=(~fit)&anchor_valid,projected_xy=projected,z=z,raw=values,loo_displacement=spread,reason=np.asarray(reasons,dtype="U128"))
            np.savez_compressed(scene/"decisions.npz",xy=xy,E=E,raw=raw[y,x],params=json.dumps(calibration),**decision)
            sealed={name:sha256(scene/name) for name in ("calibration.npz","decisions.npz","raw_prior.npz")}
            record=dict(status="partial",scan=scan,reference=ref,sealed_sha256=sealed,calibration=calibration,triangulation_reason_counts=dict(Counter(reasons)),samples=len(E),old_decision_sha256=old_sha,frame_sha256=frame_sha,image_sha256=sha256(image[0]),camera_checks=checks,timings=dict(inference=inference_seconds,triangulation=triangulation_seconds,calibration=calibration_seconds,readout=readout_seconds),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save(scene/"metadata.json",record); check()
            # Scanned geometry is first opened only after both decision artifacts are sealed.
            from scipy.spatial import cKDTree
            from plyfile import PlyData
            gt=results/f"20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"
            gt_sha=sha256(gt)
            if gt_sha != meta["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]: raise ValueError("GT SHA mismatch")
            vertex=PlyData.read(str(gt))["vertex"]; tree=cKDTree(np.column_stack([np.asarray(vertex[k],float) for k in ("x","y","z")]))
            rays=np.column_stack((xy,np.ones(len(E))))@np.linalg.inv(ci["K"]).T; nn={}
            for name,depth in (("E",E),("teacher",decision["teacher"]),("bandprojection",decision["selected"])):
                check(); world=(rays*depth[:,None])@ci["c2w"][:3,:3].T+ci["c2w"][:3,3]; mm=world@transform[:3,:3].T+transform[:3,3]
                nn[name]=tree.query(mm,k=1,workers=1)[0]
            record["nn_proxy"]={k:dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in nn.items()}
            take=decision["accept"]; record.update(accepted=int(take.sum()),accepted_delta_mean_mm=float((nn["bandprojection"]-nn["E"])[take].mean()) if take.any() else None,gt_sha256=gt_sha)
            if sha256(frame)!=frame_sha or sha256(args.checkpoint)!=args.checkpoint_sha256: raise ValueError("Read-only input changed")
            for name,digest in sealed.items():
                if sha256(scene/name)!=digest: raise ValueError("Sealed artifact changed")
            record["status"]="complete"; save(scene/"metadata.json",record); state["completed"].append(record); save(out/"metadata.json",state)
        check(); dependency_info(args.dependency_root)
        if sha256(args.checkpoint)!=args.checkpoint_sha256: raise ValueError("Checkpoint changed")
        state.update(status="complete",seconds=time.monotonic()-started,checkpoint_final_sha256=sha256(args.checkpoint)); save(out/"metadata.json",state)
    except BaseException as exc:
        state.update(status="failed_or_stopped",error=repr(exc)); save(out/"metadata.json",state); raise


def synthetic_check():
    x=np.linspace(.1,2,100); y=.3*x+.1; fit=np.arange(100)%2==0
    exact=calibrate(x,1/y,fit); assert exact["valid"] and abs(exact["a"]-.3)<1e-10 and abs(exact["b"]-.1)<1e-10
    noisy=y.copy(); noisy[np.flatnonzero(fit)[::7]]+=2
    robust=calibrate(x,1/noisy,fit); assert robust["valid"] and abs(robust["a"]-.3)<1e-3
    invalid=calibrate(np.ones(100),1/y,fit); assert not invalid["valid"]
    E=np.array([1.,2.,3.]); d=readout(E,[.1,np.nan,2],dict(valid=True,a=1.,b=0.,q=.2))
    assert np.array_equal(d["selected"][:2],E[:2]) and d["accept"][2]
    assert np.array_equal(readout(E,E,invalid)["selected"],E)
    print(json.dumps(dict(check="PASS",exact=exact,robust=robust,limitation="Empirical residual q is not a physical uncertainty guarantee")))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--check",action="store_true")
    for option in ("observation-root","dependency-root","checkpoint","output"): p.add_argument("--"+option,type=Path)
    p.add_argument("--checkpoint-sha256"); p.add_argument("--reference",default="0000"); p.add_argument("--max-seconds",type=int,default=900)
    args=p.parse_args()
    if args.check: synthetic_check()
    else:
        if any(getattr(args,k) is None for k in ("observation_root","dependency_root","checkpoint","checkpoint_sha256","output")): p.error("All input/output arguments required")
        if not 0<args.max_seconds<=900: p.error("max-seconds must be in (0,900]")
        run(args)
