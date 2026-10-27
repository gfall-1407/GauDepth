"""Diagnose fixed gap teachers against other frozen views, then sealed scan NN.

This is a diagnostic, not a deployed selector or physical confidence guarantee.
The neighbor E maps come from the same trained model. No scan GT enters scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

import consensus_depth as cd
from photo_objective import bilinear
from photo_surface_probe import HERE, save, sha256, source_info


def world(xy, depth, camera):
    h, w = camera["image_size"]
    fx, fy = camera["focal"]
    ray = np.column_stack(((xy[:, 0]-(w-1)/2)/fx,
                           (xy[:, 1]-(h-1)/2)/fy, np.ones(len(xy))))
    local = ray * depth[:, None]
    c = camera["c2w"]
    return local @ c[:3, :3].T + c[:3, 3]


def project(points, camera):
    c = camera["w2c"]
    local = points @ c[:3, :3].T + c[:3, 3]
    z = local[:, 2]
    h, w = camera["image_size"]
    fx, fy = camera["focal"]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = local[:, :2]/z[:, None] * [fx, fy] + [(w-1)/2, (h-1)/2]
    uv[~(np.isfinite(local).all(1) & (z > 0))] = np.nan
    return uv, z


def residuals(xy, depths, source, neighbor, expected_map, scale):
    uv, z = project(world(xy, depths, source), neighbor)
    observed, inside = bilinear(expected_map, uv)
    valid = inside & np.isfinite(observed) & (observed > 0) & (z > 0)
    signed = (z-observed)*scale
    back, _ = project(world(uv, observed, neighbor), source)
    roundtrip = np.linalg.norm(back-xy, axis=1)
    signed[~valid] = np.nan
    roundtrip[~valid] = np.nan
    return signed, roundtrip


def checked_scan(scan, args):
    cache = args.cache_root / f"scan{scan}_v1" / f"cache{scan}"
    meta = json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "cache_complete" or meta["h_mm"] != 1:
        raise ValueError("Require complete, fixed physical 1mm teacher cache")
    refs = json.loads((HERE/"G1_Huber全场景输入参考.json").read_text(encoding="utf-8"))["scans"][str(scan)]
    if (meta["model_sha256"], meta["cfg_sha256"]) != (refs["model_sha256"], refs["cfg_sha256"]):
        raise ValueError("Teacher model provenance differs")
    manifest = json.loads((args.input_root.parent/"input_manifest.json").read_text(encoding="utf-8"))
    checked = {}
    for name in ("training_cameras.json", "cameras.npz"):
        relative = f"scan{scan}/{name}"
        record = [r for r in manifest["files"] if r["relative"] == relative]
        f = args.input_root / relative
        if len(record) != 1 or sha256(f) != record[0]["sha256"] or f.stat().st_size != record[0]["bytes"]:
            raise ValueError("Camera input manifest mismatch: "+relative)
        checked[relative] = record[0]["sha256"]
    rows = json.loads((args.input_root/f"scan{scan}/training_cameras.json").read_text(encoding="utf-8"))
    by_name = {r["img_name"]: r for r in rows}
    if set(by_name) != set(meta["camera_names"]):
        raise ValueError("Exact camera inventory differs")
    cameras, frames = {}, {}
    alignment = np.array(meta["alignment"]["transform"], dtype=float)
    for v in meta["views"]:
        name = v["name"]; row = by_name[name]
        c2w = np.eye(4); c2w[:3, :3] = row["rotation"]; c2w[:3, 3] = row["position"]
        np.testing.assert_allclose(c2w[:3, :3].T@c2w[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        w2c = np.linalg.inv(c2w)
        cd._check_camera_projection(args.input_root/f"scan{scan}/cameras.npz",
                                    dict(image_name=name, world_view_transform=w2c.T.tolist()), alignment)
        h, w = v["image_size"]
        if [row["height"], row["width"]] != [2*h, 2*w]:
            raise ValueError("Camera resolution differs")
        np.testing.assert_allclose(np.array(v["focal"]), [row["fx"]/2, row["fy"]/2], atol=3e-4, rtol=0)
        cameras[name] = dict(c2w=c2w, w2c=w2c, image_size=(h, w), focal=v["focal"])
        f = cache/v["file"]
        if sha256(f) != v["sha256"]:
            raise ValueError("Teacher frame SHA differs: "+str(f))
        with np.load(f) as z:
            mask = z["gt_mask"].astype(bool).squeeze()
            e = z["expected"].squeeze().copy(); target = z["gap_huber"].squeeze().copy()
        if not np.all(np.isfinite(e[mask]) & (e[mask] > 0) & np.isfinite(target[mask]) & (target[mask] > 0)):
            raise ValueError("Nonfinite foreground depth")
        e[~mask] = np.nan
        frames[name] = (e, target, mask)
    neighbors = {r["ref_name"]: r["nearest_name"] for r in
                 (json.loads(line) for line in (cache/"multi_view.json").read_text().splitlines())}
    expected_sha = json.loads((args.input_root.parent/"neighbor_manifest.json").read_text())[str(scan)]
    if sha256(cache/"multi_view.json") != expected_sha or set(neighbors) != set(cameras):
        raise ValueError("Neighbor source differs")
    return meta, cameras, frames, neighbors, dict(camera_inputs=checked,
        metadata_sha256=sha256(cache/"metadata.json"), neighbor_sha256=expected_sha)


def diagnose(args):
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("New absolute independent output required")
    args.output.mkdir(parents=True)
    state = dict(status="partial", source=source_info(), completed=[],
        design="All foreground gap changes and a fixed equal-size unchanged random control per view; 8 original neighbors, same-neighbor E/H comparisons. Minimum 3 common valid neighbors follows existing fixed MV probe. No prediction changes or GT-driven parameter choice.",
        limitation="Same-model other-view native expected depths are a diagnostic, not independent physical GT. No mesh improvement is inferred from scan-point NN.")
    save(args.output/"metadata.json", state); start = time.monotonic()
    def boundary():
        if time.monotonic()-start > args.max_seconds or (args.output/"stop_after_view").exists():
            raise InterruptedError("Finite deadline/request at full view boundary")
    try:
        for scan in args.scans:
            boundary(); meta, cameras, frames, neighbors, provenance = checked_scan(scan, args)
            scale = meta["alignment"]["scale_to_dtu_mm"]
            for name in meta["camera_names"]:
                boundary(); e, target, mask = frames[name]; h, w = mask.shape
                changed = np.flatnonzero(mask & (target != e))
                unchanged = np.flatnonzero(mask & (target == e))
                rng = np.random.default_rng(20261027+scan*100+int(name))
                control = np.sort(rng.choice(unchanged, min(len(changed), len(unchanged)), replace=False))
                index = np.concatenate((changed, control)); group = np.arange(len(index)) < len(changed)
                y, x = np.divmod(index, w); xy = np.column_stack((x, y)).astype(float)
                depths = np.column_stack((e.ravel()[index], target.ravel()[index]))
                identity, identity_z = project(world(xy, depths[:, 0], cameras[name]), cameras[name])
                np.testing.assert_allclose(identity, xy, atol=1e-8, rtol=0)
                np.testing.assert_allclose(identity_z, depths[:, 0], atol=1e-10, rtol=0)
                errors, rounds = [], []
                for neighbor in neighbors[name]:
                    signed, rt = zip(*(residuals(xy, depths[:, k], cameras[name], cameras[neighbor], frames[neighbor][0], scale) for k in range(2)))
                    signed = np.array(signed).T; rt = np.array(rt).T
                    common = np.isfinite(signed).all(1) & np.isfinite(rt).all(1)
                    signed[~common] = np.nan; rt[~common] = np.nan
                    errors.append(signed); rounds.append(rt)
                errors = np.stack(errors, axis=1); rounds = np.stack(rounds, axis=1)
                count = np.isfinite(errors[:, :, 0]).sum(1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    score = np.nanmedian(np.abs(errors), axis=1)
                    rt_score = np.nanmedian(rounds, axis=1)
                score[count < 3] = np.nan; rt_score[count < 3] = np.nan
                if not np.array_equal(score[~group, 0], score[~group, 1], equal_nan=True):
                    raise AssertionError("Unchanged-control pair scores differ")
                f = args.output/f"scan{scan}_{name}.npz"
                np.savez_compressed(f, index=index, changed=group, depth=depths, signed_neighbor_mm=errors,
                                    roundtrip_neighbor_px=rounds, common_neighbors=count,
                                    median_abs_depth_mm=score, median_roundtrip_px=rt_score)
                state["completed"].append(dict(scan=scan, view=name, file=f.name, sha256=sha256(f),
                    changed=len(changed), control=len(control), available_changed=int(np.sum(group & (count >= 3))),
                    input=provenance))
                save(args.output/"metadata.json", state)
            del frames
        state.update(status="predictions_sealed", elapsed_seconds=time.monotonic()-start)
        save(args.output/"metadata.json", state)
    except BaseException as exc:
        state.update(status="partial_or_failed", error=repr(exc)); save(args.output/"metadata.json", state); raise


def evaluate(args):
    state = json.loads((args.output/"metadata.json").read_text())
    if state["status"] != "predictions_sealed":
        raise ValueError("Only fully sealed diagnostic predictions may read scanned geometry")
    # No prediction function receives a scanned-geometry array or its error.
    summary = []; started = time.monotonic(); evaluation_source = source_info()
    legacy_path = args.output/"nn_census_partial.json"
    legacy = {r["file"]: r for r in json.loads(legacy_path.read_text())["files"]} if legacy_path.exists() else {}
    evaluated = []
    for scan in args.scans:
        meta, cameras, _, _, _ = checked_scan(scan, args)
        gt = args.input_root/f"scan{scan}/stl{scan:03d}_total.ply"
        refs = json.loads((HERE/"G1_Huber全场景输入参考.json").read_text())["scans"][str(scan)]
        expected = refs["gt_inputs"][f"Points/stl/stl{scan:03d}_total.ply"]
        if sha256(gt) != expected:
            raise ValueError("Scanned point-cloud SHA differs")
        tree = cd._make_gt_tree(cd._load_gt(gt)); joined = []
        transform = np.array(meta["alignment"]["transform"], dtype=float)
        for row in [r for r in state["completed"] if r["scan"] == scan]:
            if time.monotonic()-started > args.max_seconds or (args.output/"stop_after_view").exists():
                raise InterruptedError("Finite NN phase deadline/request at score boundary")
            f = args.output/row["file"]
            if sha256(f) != row["sha256"]: raise ValueError("Sealed prediction changed")
            with np.load(f) as z: d = {k: z[k] for k in z.files}
            # Fixed selection uses only view/group/index and seed, never scanned error.
            rng = np.random.default_rng(20261027+10000*scan+int(row["view"]))
            selection = np.concatenate([np.sort(rng.choice(np.flatnonzero(d["changed"] == g),
                min(args.nn_per_group, np.sum(d["changed"] == g)), replace=False)) for g in (True, False)])
            previous = args.output/f"scan{scan}_{row['view']}_nn.npz"
            cached_nn = None
            if previous.exists():
                record = legacy.get(previous.name)
                if not record or sha256(previous) != record["sha256"] or record["prediction_sha256"] != row["sha256"]:
                    raise ValueError("Unverified prior complete NN file")
                with np.load(previous) as z:
                    if z["nn_mm"].shape != (len(d["index"]), 2) or not np.isfinite(z["nn_mm"]).all():
                        raise ValueError("Prior NN array shape/finite check failed")
                    cached_nn = z["nn_mm"][selection]
            d = {k: v[selection] for k, v in d.items()}
            h, w = cameras[row["view"]]["image_size"]; y, x = np.divmod(d["index"], w)
            xy = np.column_stack((x, y)); errors = []
            if cached_nn is None:
                for k in range(2):
                    points = world(xy, d["depth"][:, k], cameras[row["view"]])
                    points = points@transform[:3, :3].T+transform[:3, 3]
                    errors.append(cd._query_exact(points, tree))
                cached_nn = np.array(errors).T
            d["nn_mm"] = cached_nn; joined.append(d)
            score_file = args.output/f"scan{scan}_{row['view']}_nn_sampled.npz"
            np.savez_compressed(score_file, selection=selection, index=d["index"], nn_mm=d["nn_mm"])
            evaluated.append(dict(scan=scan, view=row["view"], file=score_file.name, sha256=sha256(score_file),
                prediction_sha256=row["sha256"], reused_previous=previous.exists(), sampled_n=len(selection)))
            save(args.output/"nn_sample_progress.json", dict(status="partial", source=evaluation_source,
                max_per_view_group=args.nn_per_group, completed=evaluated))
        for changed in (True, False):
            nn = np.concatenate([x["nn_mm"][x["changed"] == changed] for x in joined])
            geo = np.concatenate([x["median_abs_depth_mm"][x["changed"] == changed] for x in joined])
            valid = np.isfinite(geo).all(1); choose_h = valid & (geo[:, 1] < geo[:, 0])
            for tag, keep in (("all", np.ones(len(nn), bool)), ("prefer_H", choose_h), ("prefer_E_or_tie", valid & ~choose_h), ("unknown", ~valid)):
                a = nn[keep]; selected = np.where(choose_h[keep], a[:, 1], a[:, 0])
                summary.append(dict(scan=scan, group="gap_changed" if changed else "unchanged_control", stratum=tag,
                    n=len(a), E_NN_mm=float(a[:, 0].mean()) if len(a) else None,
                    H_NN_mm=float(a[:, 1].mean()) if len(a) else None,
                    diagnostic_selected_NN_mm=float(selected.mean()) if len(a) else None,
                    H_better=int(np.sum(a[:, 1] < a[:, 0])), H_worse=int(np.sum(a[:, 1] > a[:, 0]))))
    save(args.output/"summary.json", dict(status="complete", rows=summary,
        limitation=state["limitation"],
        denominator=f"Per scan: fixed seed maximum {args.nn_per_group} per view and changed/control stratum; pooled diagnostic sample, not full-mask or scene-equal Overall.",
        evaluation_source=evaluation_source, completed_scores=evaluated))
    save(args.output/"nn_sample_progress.json", dict(status="complete", source=evaluation_source,
        max_per_view_group=args.nn_per_group, completed=evaluated))
    state["status"] = "complete"; save(args.output/"metadata.json", state)
    print(json.dumps(summary))


def truncation_audit(args):
    """Audit the benchmark's fixed20mm cutoff on the SAME sampled ray NN pairs."""
    source=source_info();report=json.loads((args.output/'summary.json').read_text())
    if report['status']!='complete':raise ValueError('Complete sealed paired NN scores required')
    destination=args.output/'truncation_diagnostic.json'
    if destination.exists():raise ValueError('Do not overwrite completed cutoff diagnostic')
    pairs={}
    for row in report['completed_scores']:
        f=args.output/row['file'];prediction=args.output/f"scan{row['scan']}_{row['view']}.npz"
        if sha256(f)!=row['sha256'] or sha256(prediction)!=row['prediction_sha256']:
            raise ValueError('Sealed sample/prediction changed')
        with np.load(f) as z:selection=z['selection'];index=z['index'];nn=z['nn_mm']
        with np.load(prediction) as z:
            np.testing.assert_array_equal(index,z['index'][selection]);group=z['changed'][selection]
        if not np.isfinite(nn).all() or np.any(nn<0):raise ValueError('Invalid fixed NN pairs')
        for changed in (True,False):pairs.setdefault((row['scan'],changed),[]).append(nn[group==changed])
    rows=[];cutoff=20. # Existing native dtu_eval --max_dist; never fitted to results.
    for (scan,changed),values in pairs.items():
        a=np.concatenate(values);inside=a<cutoff;common=inside.all(1)
        def mean(x):return float(x.mean()) if len(x) else None
        rows.append(dict(scan=scan,group='gap_changed' if changed else 'unchanged_control',n=len(a),
            untruncated_E_mm=mean(a[:,0]),untruncated_H_mm=mean(a[:,1]),
            independent_cutoff_E_mm=mean(a[inside[:,0],0]),independent_cutoff_H_mm=mean(a[inside[:,1],1]),
            independent_E_n=int(inside[:,0].sum()),independent_H_n=int(inside[:,1].sum()),
            common_n=int(common.sum()),common_E_mm=mean(a[common,0]),common_H_mm=mean(a[common,1]),
            capped_E_mm=mean(np.minimum(a[:,0],cutoff)),capped_H_mm=mean(np.minimum(a[:,1],cutoff)),
            entered_H_support=int(np.sum(~inside[:,0]&inside[:,1])),
            left_H_support=int(np.sum(inside[:,0]&~inside[:,1]))))
    save(destination,dict(status='complete',source=source,cutoff_mm=cutoff,rows=rows,
        limitation='Same fixed per-view/group ray-NN sample only. Native mesh scoring additionally changes area samples, voxel thinning, ObsMask and geometry; cutoff behavior here does not prove the actual cause of mesh regression and is not a new evaluation protocol.'))
    print(json.dumps(rows))


def native_alignment_audit(args):
    """Recover the already applied native camera alignment from exact mesh IDs.

    No GT fitting/ICP: source and destination are the SAME preserved mesh before
    and after native Calibration transformation. Refuse differing face indices.
    """
    import trimesh
    source=source_info();records=[]
    for scan in args.scans:
        root=args.cache_root/f'scan{scan}_v1';meta=json.loads((root/f'cache{scan}/metadata.json').read_text())
        archive=np.asarray(meta['alignment']['transform'],dtype=float);transforms=[]
        for method in ('native_expected','gap_huber'):
            p=root/f'native{scan}'/method
            before=trimesh.load(p/'recon_post.ply',process=True)
            after=trimesh.load(p/'recon_aligned.ply',process=False)
            if before.vertices.shape!=after.vertices.shape or not np.array_equal(before.faces,after.faces):
                raise ValueError('Exact native vertex/face correspondence unavailable; no ICP fallback')
            ids=np.linspace(0,len(before.vertices)-1,min(512,len(before.vertices)),dtype=int)
            x=np.column_stack((np.asarray(before.vertices)[ids],np.ones(len(ids))))
            coefficients=np.linalg.lstsq(x,np.asarray(after.vertices)[ids],rcond=None)[0]
            transform=np.eye(4);transform[:3]=coefficients.T
            residual=np.linalg.norm(np.asarray(before.vertices)@transform[:3,:3].T+transform[:3,3]-np.asarray(after.vertices),axis=1)
            if float(residual.max())>5e-4:
                raise ValueError('Recovered native transformation exceeds fixed float32 PLY rounding budget')
            singular=np.linalg.svd(transform[:3,:3],compute_uv=False)
            if np.linalg.det(transform[:3,:3])<=0 or singular.max()/singular.min()>1.00001:
                raise ValueError('Expected the native camera-based positive similarity')
            transforms.append(transform)
            records.append(dict(scan=scan,method=method,n_vertices=len(before.vertices),
                native_camera_transform=transform.tolist(),archive_transform=archive.tolist(),
                transform_max_abs_difference=float(np.abs(transform-archive).max()),
                correspondence_fit_max_mm=float(residual.max()),source_mesh_sha256=sha256(p/'recon_post.ply'),
                aligned_mesh_sha256=sha256(p/'recon_aligned.ply')))
            del before,after
        if not np.allclose(transforms[0],transforms[1],atol=5e-4,rtol=0):
            raise ValueError('Native camera transform should be method independent within PLY rounding')
    path=args.output/'native_alignment_diagnostic.json'
    if path.exists():raise ValueError('Do not overwrite completed alignment diagnostic')
    save(path,dict(status='complete',source=source,rows=records,
        limitation='Transformation is recovered only from exact native mesh correspondences, not GT. It does not make area-weighted mesh scoring and ray-weighted NN equivalent.'))
    print(json.dumps(records))


def clipped_location_field(z,w,t,h):
    """Symmetric location-mixture toy, NOT the unverified native TSDF operator."""
    z,w,t=np.asarray(z,dtype=float),np.asarray(w,dtype=float),np.asarray(t,dtype=float)
    if z.ndim!=1 or z.shape!=w.shape or not len(z) or not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite locations and nonnegative nonzero weights required')
    if not np.isfinite(t).all() or not np.isfinite(h) or h<=0:raise ValueError('Finite queries/positive bandwidth required')
    d=(z-t[...,None])/h
    phi=(w*np.clip(d,-1.,1.)).sum(axis=-1)/w.sum()
    negative_derivative=(w*(np.abs(d)<1.)).sum(axis=-1)/(h*w.sum())
    return phi,negative_derivative


def mixture_operator_check(args):
    destination=args.output/'mixture_operator_check.json'
    if not args.output.is_absolute() or not args.output.is_dir() or destination.exists():
        raise ValueError('Existing absolute result directory and new toy output required')
    source=source_info();h=1.;offset=2.
    z=[offset,10.+offset,2.+offset];w=[.51,.49,1.]
    root_a=offset+49/51;root_b=offset+2.;point_root=offset+151/102;mixture_root=offset+1.98
    def field(t):return clipped_location_field(z,w,t,h)
    assert abs(float(clipped_location_field(z[:2],w[:2],root_a,h)[0]))<1e-12
    assert abs(float(field(mixture_root)[0]))<1e-12
    assert abs(float(clipped_location_field([root_a,root_b],[1.,1.],point_root,h)[0]))<1e-12
    assert abs(point_root-mixture_root)>.49 and abs(float(field(point_root)[0]))>.2
    grid=np.linspace(offset-2,offset+14,65)
    delta,_=clipped_location_field([offset+2],[.37],grid,h)
    np.testing.assert_allclose(delta,np.clip((offset+2-grid)/h,-1,1),atol=1e-15,rtol=0)
    linear_z=np.array([.9,1.1,1.2])+offset;linear_w=np.array([.2,.3,.5]);t=np.array([.8,1.,1.3])+offset
    phi,slope=clipped_location_field(linear_z,linear_w,t,2.)
    np.testing.assert_allclose(phi,((linear_z@linear_w)-t)/2,atol=1e-15,rtol=0)
    np.testing.assert_allclose(slope,.5,atol=1e-15,rtol=0)
    plateau,plateau_slope=clipped_location_field([offset,offset+10],[.5,.5],np.array([2,4,6,8])+offset,h)
    np.testing.assert_array_equal(plateau,np.zeros(4));np.testing.assert_array_equal(plateau_slope,np.zeros(4))
    probes=np.array([.3,.8,1.8,2.3])+offset;eps=1e-6
    finite_difference=(field(probes+eps)[0]-field(probes-eps)[0])/(2*eps)
    derivative_error=float(np.max(np.abs(finite_difference+field(probes)[1])))
    if derivative_error>1e-8:raise ValueError('Mixture-field derivative formula failed')
    report=dict(status='mathematical_checks_PASS_native_TSDF_bridge_PENDING',source=source,
        assumptions='One-dimensional same-axis symmetric clipping; equal total view weights; arbitrary toy h1, not asserted to be native TSDF truncation.',
        distributions=[dict(z=z[:2],w=w[:2]),dict(z=[z[2]],w=[1.])],
        separate_view_roots=[root_a,root_b],root_after_point_compression=point_root,
        root_after_distribution_pooling=mixture_root,root_difference=mixture_root-point_root,
        derivative_max_error=derivative_error,delta_and_linear_regime='PASS',zero_interval='PASS_nonunique_not_surface_guarantee',
        limitation='This proves noncommutation and lost local support slope only. Same-model rendering weights are not calibrated physical probabilities. Native pixel sampling, visibility/negative-truncation skipping, weight update and delta-E TSDF/weight/update-mask agreement remain unverified. Detached reader has no trainable distribution gradient. No scene, mesh, efficiency, novelty or stable-algorithm claim.')
    save(destination,report);print(json.dumps(report))


def native_operator_check(args):
    """Tiny localCPU0.19 delta bridge; NOT the remote/real-scene bridge."""
    import open3d as o3d
    if o3d.__version__!='0.19.0':raise ValueError('This source-verified check pins local Open3D0.19.0')
    destination=args.output/'native_operator_check.json'
    if not args.output.is_absolute() or not args.output.is_dir() or destination.exists():
        raise ValueError('Existing absolute result directory/new output required')
    source=source_info();voxel=np.float32(.125);trunc=np.float32(voxel*8);depth_max=np.float32(8.)
    keys=np.array([(x,y,z) for x in (-1,0) for y in (-1,0) for z in range(-1,20)],np.int32)
    intrinsic=o3d.core.Tensor([[4.,0.,2.],[0.,4.,2.],[0.,0.,1.]],o3d.core.Dtype.Float64)
    extrinsic=o3d.core.Tensor.eye(4,o3d.core.Dtype.Float64)
    block_keys=o3d.core.Tensor(keys);cases=[]
    for name,depths in [('plane2',[2.]),('invalid_zero',[0.]),('invalid_above_depthmax',[9.]),('two_frames2then4',[2.,4.])]:
        grid=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight'),
            attr_dtypes=(o3d.core.Dtype.Float32,o3d.core.Dtype.Float32),attr_channels=((1),(1)),
            voxel_size=float(voxel),block_resolution=4,block_count=128,device=o3d.core.Device('CPU:0'))
        grid.hashmap().activate(block_keys)
        grid.attribute('tsdf').fill_(0);grid.attribute('weight').fill_(0)
        coordinates,indices=grid.voxel_coordinates_and_flattened_indices()
        xyz=coordinates.numpy();flat=indices.numpy().ravel();z=xyz[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            u=xyz[:,0]*np.float32(4)/z+np.float32(2);v=xyz[:,1]*np.float32(4)/z+np.float32(2)
        projected=np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<5)&(v>=0)&(v<5)&(z>0)
        tsdf=np.zeros(len(z),np.float32);weight=tsdf.copy();maximum_error=0.;behind_skipped=0
        for depth in depths:
            d=np.float32(depth);sdf=d-z
            valid=projected&(d>0)&(d<=depth_max)&(sdf>=-trunc)
            observation=np.minimum(sdf,trunc)/trunc
            inv=np.float32(1)/(weight[valid]+np.float32(1))
            tsdf[valid]=(weight[valid]*tsdf[valid]+observation[valid])*inv
            weight[valid]+=np.float32(1)
            image=o3d.t.geometry.Image(o3d.core.Tensor(np.full((5,5),d,np.float32)))
            grid.integrate(block_keys,image,intrinsic,extrinsic,1.,float(depth_max),8.)
            actual_w=grid.attribute('weight').numpy().reshape(-1)[flat]
            actual_phi=grid.attribute('tsdf').numpy().reshape(-1)[flat]
            np.testing.assert_array_equal(actual_w,weight)
            np.testing.assert_array_equal(actual_w>0,weight>0)
            error=float(np.abs(actual_phi-tsdf).max());maximum_error=max(maximum_error,error)
            if error>8*np.finfo(np.float32).eps:raise ValueError('Fixed float32 TSDF budget exceeded; inspect operator, not GT')
            if 0<depth<=depth_max:behind_skipped+=int(np.sum(projected&(sdf<-trunc)))
        cases.append(dict(case=name,voxels=len(z),updated_voxels=int((weight>0).sum()),
            update_mask_and_weight='PASS_exact',tsdf_max_abs_error=maximum_error,
            behind_truncation_skipped_n=behind_skipped))
    report=dict(status='PASS_localCPU_constant_image_identity_camera_delta_bridge_only',source=source,
        open3d=o3d.__version__,voxel_size=float(voxel),trunc_voxel_multiplier=8.,sdf_trunc=float(trunc),depth_scale=1.,depth_max=float(depth_max),cases=cases,
        source_reference='https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h',
        limitation='Native skip mask and observation weight are separate from symmetric clipped-location toy. No remote package version proof, arbitrary-camera/nonconstant-pixel sampling bridge, full ray-distribution integration, model gradient, real scene or reconstruction evidence. All remain pending.')
    save(destination,report);print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-root", type=Path, default=HERE/"结果/20261026_夜间深度方法/inputs")
    p.add_argument("--cache-root", type=Path, default=HERE/"结果/20261027_Gap原生验证")
    p.add_argument("--scans", type=int, nargs="+", default=[40, 63])
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--nn-per-group", type=int, default=256)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument('--truncation-audit',action='store_true')
    p.add_argument('--native-alignment-audit',action='store_true')
    p.add_argument('--mixture-operator-check',action='store_true')
    p.add_argument('--native-operator-check',action='store_true')
    a = p.parse_args()
    if a.native_operator_check:native_operator_check(a)
    elif a.mixture_operator_check:mixture_operator_check(a)
    elif a.native_alignment_audit:native_alignment_audit(a)
    elif a.truncation_audit:truncation_audit(a)
    elif a.evaluate: evaluate(a)
    else: diagnose(a)
