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
    summary = []
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
            f = args.output/row["file"]
            if sha256(f) != row["sha256"]: raise ValueError("Sealed prediction changed")
            with np.load(f) as z: d = {k: z[k] for k in z.files}
            h, w = cameras[row["view"]]["image_size"]; y, x = np.divmod(d["index"], w)
            xy = np.column_stack((x, y)); errors = []
            for k in range(2):
                points = world(xy, d["depth"][:, k], cameras[row["view"]])
                points = points@transform[:3, :3].T+transform[:3, 3]
                errors.append(cd._query_exact(points, tree))
            d["nn_mm"] = np.array(errors).T; joined.append(d)
            np.savez_compressed(args.output/f"scan{scan}_{row['view']}_nn.npz", nn_mm=d["nn_mm"])
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
        limitation=state["limitation"], denominator="Per scan/changed stratum; all foreground changed pixels plus fixed equal-size unchanged controls, never mixed as full-mask or scene-equal Overall."))
    state["status"] = "complete"; save(args.output/"metadata.json", state)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-root", type=Path, default=HERE/"结果/20261026_夜间深度方法/inputs")
    p.add_argument("--cache-root", type=Path, default=HERE/"结果/20261027_Gap原生验证")
    p.add_argument("--scans", type=int, nargs="+", default=[40, 63])
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--evaluate", action="store_true")
    a = p.parse_args()
    if a.evaluate: evaluate(a)
    else: diagnose(a)
