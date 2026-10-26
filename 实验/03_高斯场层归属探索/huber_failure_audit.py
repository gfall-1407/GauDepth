"""CPU-only fixed diagnostics of frozen cached depths; no fitting or rule selection."""
from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from plyfile import PlyData

from sample import atomic_json, git, sha256
from consensus_depth import _check_camera_projection
from multiview_readout import project

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "结果"
NIGHT = RESULTS / "20261026_夜间深度方法"
CURRENT = RESULTS / "20261026_Huber全场景"
BAD = (40, 63, 83, 97, 105)
STOP = False


def check(out, deadline):
    if STOP or time.monotonic() > deadline or (out / "stop_after_view").exists():
        raise InterruptedError("Fixed time budget or requested completed-view boundary")


def describe(prefix, values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {prefix+"_n": len(a), prefix+"_mean": float(a.mean()) if len(a) else None,
            prefix+"_p50": float(np.median(a)) if len(a) else None,
            prefix+"_p95": float(np.quantile(a, .95)) if len(a) else None}


def points(index, depth, camera):
    h, w = camera["image_size"]
    y, x = np.divmod(index, w)
    fx, fy = camera["focal"]
    local = np.column_stack(((x-(w-1)/2)/fx*depth, (y-(h-1)/2)/fy*depth, depth))
    pose = camera["c2w"]
    return local @ pose[:3, :3].T + pose[:3, 3]


def interpolate(depth, mask, uv):
    h, w = depth.shape
    good = np.isfinite(uv).all(1)
    safe = np.where(np.isfinite(uv), uv, -100.)
    x, y = np.floor(safe).astype(np.int64).T
    good &= (x >= 0) & (y >= 0) & (x+1 < w) & (y+1 < h)
    ids = np.flatnonzero(good)
    result, span = np.full(len(uv), np.nan), np.full(len(uv), np.nan)
    if not len(ids):
        return result, span
    xx, yy = x[ids], y[ids]
    q = np.array([depth[yy, xx], depth[yy, xx+1], depth[yy+1, xx], depth[yy+1, xx+1]])
    valid = np.isfinite(q).all(0) & (q > 0).all(0)
    valid &= mask[yy, xx] & mask[yy, xx+1] & mask[yy+1, xx] & mask[yy+1, xx+1]
    dx, dy = safe[ids, 0]-xx, safe[ids, 1]-yy
    value = (1-dy)*((1-dx)*q[0]+dx*q[1])+dy*((1-dx)*q[2]+dx*q[3])
    result[ids[valid]] = value[valid]
    span[ids[valid]] = (q.max(0)-q.min(0))[valid]
    return result, span


def inputs(scan, reference):
    path = (NIGHT / ("dense_pilot_v1" if scan == 24 else f"dense_confirm_v1/scan{scan}_v1")
            if scan in (24, 37, 106) else CURRENT / f"scan{scan}_v1")
    m = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert m["status"] == "complete" and m["original_model_unchanged"] and m["original_cfg_unchanged"]
    root = NIGHT / f"inputs/scan{scan}"
    if not root.exists():
        root = RESULTS / f"20261023_一致深度支持/inputs/scan{scan}"
    archive = root / "cameras.npz"
    assert sha256(archive) == m["cameras_npz_sha256"]
    training = None
    if "focal" not in reference["views"][0]:
        training = {v["img_name"]: v for v in json.loads((root / "training_cameras.json").read_text(encoding="utf-8"))}
    cameras = []
    for vm, rv in zip(m["views"], reference["views"]):
        assert vm["name"] == rv["name"] and vm["mask_sha256"] == rv["mask_sha256"]
        if training is None:
            w2c = np.array(rv["world_view_transform"], dtype=float).T
        else:
            row = training[vm["name"]]
            c2w = np.eye(4); c2w[:3, :3] = row["rotation"]; c2w[:3, 3] = row["position"]
            w2c = np.linalg.inv(c2w)
        camera = dict(image_size=vm["image_size"], focal=vm["focal"], w2c=w2c, c2w=np.linalg.inv(w2c))
        _check_camera_projection(archive, dict(image_name=vm["name"], image_size=vm["image_size"],
            focal=vm["focal"], world_view_transform=w2c.T.tolist()), np.array(m["alignment"]["transform"]))
        test = np.array([100, 1000, 2000]); p = points(test, np.ones(3), camera); uv, z = project(p, camera)
        h, w = vm["image_size"]; y, x = np.divmod(test, w)
        np.testing.assert_allclose(uv, np.column_stack((x, y)), rtol=0, atol=1e-8)
        np.testing.assert_allclose(z, 1., rtol=0, atol=1e-10)
        cameras.append(camera)
    neighbors = {v["ref_name"]: v["nearest_name"] for v in
                 (json.loads(line) for line in (path / "multi_view.json").read_text().splitlines())}
    gt = root / f"stl{scan:03d}_total.ply"
    assert sha256(gt) == m["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]
    vertex = PlyData.read(str(gt))["vertex"]
    xyz = np.column_stack([np.asarray(vertex[k], dtype=float) for k in ("x", "y", "z")])
    return path, m, cameras, neighbors, cKDTree(xyz, leafsize=32)


def audit(scan, args, reference, state, deadline):
    path, m, cameras, neighbors, tree = inputs(scan, reference)
    frames = []
    for vm in m["views"]:
        check(args.output, deadline)
        f = path / vm["file"]; assert sha256(f) == vm["sha256"]
        with np.load(f) as z:
            frames.append({k: z[k].squeeze() for k in ("expected", "huber", "alpha", "gt_mask")})
    scale = float(m["alignment"]["scale_to_dtu_mm"]); transform = np.array(m["alignment"]["transform"])
    name_to_i = {v["name"]: i for i, v in enumerate(m["views"])}
    full, regions, sample = [], {}, {k: [] for k in ("view_index", "pixel_index", "delta", "nn_e", "nn_h", "mv_e", "mv_h", "mv_smooth_e", "mv_smooth_h")}
    for i, (frame, camera) in enumerate(zip(frames, cameras)):
        e, h, mask = frame["expected"], frame["huber"], frame["gt_mask"].astype(bool)
        keep = mask & np.isfinite(e) & np.isfinite(h) & (e > 0) & (h > 0)
        delta = (h.astype(float)-e)*scale; full.append(delta[keep])
        jump = np.zeros_like(e)
        for axis in (0, 1):
            d = np.abs(np.diff(e, axis=axis))*scale
            pair = np.diff(keep.astype(np.int8), axis=axis) == 0
            d *= pair
            a, b = [slice(None)]*2, [slice(None)]*2; a[axis] = slice(1, None); b[axis] = slice(None, -1)
            jump[tuple(a)] = np.maximum(jump[tuple(a)], d); jump[tuple(b)] = np.maximum(jump[tuple(b)], d)
        groups = {"mask_edge_3px": keep & ~binary_erosion(mask, iterations=3),
                  "depth_jump_ge1mm": keep & (jump >= 1.), "smooth_interior": keep & binary_erosion(mask, iterations=3) & (jump < 1.),
                  "alpha_lt0.9": keep & (frame["alpha"] < .9), "alpha_ge0.99": keep & (frame["alpha"] >= .99)}
        for label, group in groups.items():
            values = delta[group]; r = regions.setdefault(label, [0, 0., 0., 0]); r[0] += len(values); r[1] += values.sum(); r[2] += np.abs(values).sum(); r[3] += int((np.abs(values) >= 1.).sum())
        rng = np.random.default_rng(20261026 + scan*1000+i)
        index = rng.choice(np.flatnonzero(keep.ravel()), size=min(2048, int(keep.sum())), replace=False)
        sample["view_index"].append(np.full(len(index), i, dtype=np.int32)); sample["pixel_index"].append(index.astype(np.int32))
        pe = points(index, e.ravel()[index], camera); ph = points(index, h.ravel()[index], camera)
        nn = []
        for p in (pe, ph):
            world_mm = p @ transform[:3, :3].T + transform[:3, 3]
            nn.append(tree.query(world_mm, k=1, eps=0, workers=4)[0])
        sample["delta"].append(delta.ravel()[index]); sample["nn_e"].append(nn[0]); sample["nn_h"].append(nn[1])
        residual, smooth = [[], []], [[], []]
        for name in neighbors[m["views"][i]["name"]]:
            j = name_to_i[name]; values, spans = [], []
            for p, key in ((pe, "expected"), (ph, "huber")):
                uv, depth = project(p, cameras[j]); estimate, span = interpolate(frames[j][key], frames[j]["gt_mask"].astype(bool), uv)
                values.append((depth-estimate)*scale); spans.append(span*scale)
            common = np.isfinite(values[0]) & np.isfinite(values[1])
            common_smooth = common & (spans[0] <= 1.) & (spans[1] <= 1.)
            for k in (0, 1):
                residual[k].append(np.where(common, np.abs(values[k]), np.nan))
                smooth[k].append(np.where(common_smooth, np.abs(values[k]), np.nan))
        for matrices, keys in ((residual, ("mv_e", "mv_h")), (smooth, ("mv_smooth_e", "mv_smooth_h"))):
            a, b = map(np.array, matrices); valid = np.isfinite(a).sum(0) >= 3
            for x, key in ((a, keys[0]), (b, keys[1])): sample[key].append(np.nanmedian(x[:, valid], axis=0))
        state["active"] = dict(scan=scan, completed_views=i+1, views=len(frames)); atomic_json(args.output / "metadata.json", state)
        check(args.output, deadline)
    sample = {k: np.concatenate(v) for k, v in sample.items()}; delta = np.concatenate(full)
    with (args.output / f"scan{scan}_samples.npz").open("wb") as f: np.savez_compressed(f, **sample)
    nn_delta = sample["nn_h"]-sample["nn_e"]; core = sample["nn_e"] <= .5
    row = dict(scan=scan, mesh_worse=int(scan in BAD), mask_n=len(delta), signed_shift_mean_mm=float(delta.mean()),
        abs_shift_mean_mm=float(np.abs(delta).mean()), toward_camera_ge0p1_fraction=float((delta <= -.1).mean()),
        away_camera_ge0p1_fraction=float((delta >= .1).mean()), abs_shift_ge1_fraction=float((np.abs(delta) >= 1.).mean()),
        nn_worsened_fraction=float((nn_delta > 1e-4).mean()), nn_core_n=int(core.sum()),
        nn_core_delta_mm=float(nn_delta[core].mean()), nn_core_worsened_fraction=float((nn_delta[core] > 1e-4).mean()),
        voxel_mm=.002*scale, minimizer_width_ge1mm_pixels=sum(v["minimizer_width_ge1mm_mask_pixels"] for v in m["views"]))
    for key in ("nn_e", "nn_h", "mv_e", "mv_h", "mv_smooth_e", "mv_smooth_h"): row.update(describe(key, sample[key]))
    baseline_coherent = sample["mv_e"] <= 1.
    row.update(baseline_coherent_n=int(baseline_coherent.sum()),
               baseline_coherent_H_over1_fraction=float((sample["mv_h"][baseline_coherent] > 1.).mean()))
    for name, (n, signed, absolute, shifted) in regions.items():
        row[name+"_n"] = n; row[name+"_abs_shift_mm"] = float(absolute/n) if n else None
        row[name+"_ge1_fraction"] = shifted/n if n else None
    return row


def main(args):
    global STOP
    if git("status", "--short") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require clean committed approved source")
    if not args.output.is_absolute() or args.output.exists(): raise ValueError("Require new absolute output")
    if len(set(args.scans)) != len(args.scans): raise ValueError("Do not repeat scenes")
    args.output.mkdir(parents=True); signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    reference = json.loads((HERE / "G1_Huber全场景输入参考.json").read_text(encoding="utf-8"))["scans"]
    state = dict(status="partial", source_commit=git("rev-parse", "HEAD"), rows=[], scans=args.scans,
        protocol="No reconstruction/training/parameter fitting. Original cached masks, exact poses/focals QA, camera center(W-1)/2. Fixed2048 samples/view seed20261026+scan*1000+index. NN is one-way to scanned points, not pixel GT. MV is self-method symmetric map agreement on common valid neighbors (>=3 of original8), all4 bilinear pixels positive and mask-valid; smooth restricts both neighbor2x2 spans<=1mm. Occlusion remains possible; these are descriptive proxies, not causal attribution or a new metric replacing DTU.")
    deadline = time.monotonic()+args.max_seconds; atomic_json(args.output / "metadata.json", state)
    try:
        for scan in args.scans:
            row = audit(scan, args, reference[str(scan)], state, deadline); state["rows"].append(row)
            atomic_json(args.output / "metadata.json", state); print(json.dumps(row), flush=True)
        with (args.output / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(state["rows"][0])); w.writeheader(); w.writerows(state["rows"])
        state["status"] = "complete"; atomic_json(args.output / "metadata.json", state)
    except BaseException as exc:
        state["error"] = repr(exc); atomic_json(args.output / "metadata.json", state); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scans", nargs="+", type=int, default=[40,63,83,97,105,24,37,55,65,69,106,110,114,118,122])
    parser.add_argument("--max-seconds", type=float, default=1800.)
    main(parser.parse_args())
