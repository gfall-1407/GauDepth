"""CPU probe of continuous contribution-weighted depth support on saved rays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

SCANS = (24, 65, 105, 110)
MODE_BANDWIDTHS_MM = (0.5, 1.0, 2.0)
METHODS = ("expected", "native_median", "weighted_median", "top10", "top50",
           "cover90", "huber_1mm", "kernel_mode_0p5mm", "kernel_mode_1mm",
           "kernel_mode_2mm")
TIE_MM = 1e-4
MAX_ITER = 64
TOL_MM = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _weighted_median(z: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(z, kind="stable")
    j = int(np.searchsorted(np.cumsum(w[order], dtype=np.float64), .5 * w.sum(), side="left"))
    return float(z[order[min(j, len(order) - 1)]])


def _fixed_subsets(z: np.ndarray, w: np.ndarray) -> dict[str, float]:
    n = len(z)
    if not n or not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w <= 0):
        return {name: float("nan") for name in ("expected", "weighted_median", "top10", "top50", "cover90")}
    total = float(w.sum(dtype=np.float64))
    mean = float(np.dot(w, z) / total)
    order = np.argsort(-w, kind="stable")
    k10, k50 = math.ceil(.1 * n), math.ceil(.5 * n)
    csum = np.cumsum(w[order], dtype=np.float64)
    k90 = min(n, int(np.searchsorted(csum, (.9 - 1e-12) * total, side="left") + 1))

    def avg(ix):
        sw = float(w[ix].sum(dtype=np.float64))
        return float(np.dot(w[ix], z[ix]) / sw) if sw > 0 else float("nan")

    return dict(expected=mean, weighted_median=_weighted_median(z, w),
        top10=avg(order[:k10]), top50=avg(order[:k50]), cover90=avg(order[:k90]))


def _huber_location(z: np.ndarray, w: np.ndarray, start: float, h_mm: float = 1.0):
    if not len(z) or not np.isfinite(z).all() or not np.isfinite(w).all() or not np.isfinite(start):
        return float("nan"), "unknown", 0
    center = float(start)
    for iteration in range(1, MAX_ITER + 1):
        residual = z - center
        factor = np.ones_like(residual)
        far = np.abs(residual) > h_mm
        factor[far] = h_mm / np.abs(residual[far])
        ww = w * factor
        nxt = float(np.dot(ww, z) / ww.sum(dtype=np.float64))
        if abs(nxt - center) <= TOL_MM:
            return nxt, "converged", iteration
        center = nxt
    return float("nan"), "nonconverged", MAX_ITER


def _support(z: np.ndarray, w: np.ndarray, d: float, h_mm: float):
    p = w / w.sum(dtype=np.float64)
    kernel = np.exp(-0.5 * ((z - d) / h_mm) ** 2)
    contrib = p * kernel
    score = float(contrib.sum(dtype=np.float64))
    peak_share = float(contrib.max() / score) if score > 0 else float("nan")
    return score, peak_share


def _kernel_mode(z: np.ndarray, w: np.ndarray, h_mm: float):
    if not len(z) or not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w <= 0):
        return dict(value=float("nan"), status="unknown", starts=0, converged=0,
                    solution_clusters=0, score=float("nan"), score_gap=float("nan"), peak_share=float("nan"),
                    second_value=float("nan"), second_score=float("nan"), relative_second=float("nan"), iterations=0)
    p = w / w.sum(dtype=np.float64)
    starts, seen = [], set()
    for value in z:
        key = float(value)
        if key not in seen:
            starts.append(key)
            seen.add(key)
    solutions = []
    failed = 0
    max_iterations = 0
    for start_index, start in enumerate(starts):
        center = start
        converged = False
        for iteration in range(1, MAX_ITER + 1):
            kernel = np.exp(-0.5 * ((z - center) / h_mm) ** 2)
            mass = p * kernel
            denom = float(mass.sum(dtype=np.float64))
            if not (denom > 0 and math.isfinite(denom)):
                break
            nxt = float(np.dot(mass, z) / denom)
            if abs(nxt - center) <= TOL_MM:
                center = nxt
                score, peak = _support(z, w, center, h_mm)
                solutions.append((score, start_index, center, peak, iteration))
                max_iterations = max(max_iterations, iteration)
                converged = True
                break
            center = nxt
        if not converged:
            failed += 1
    if not solutions:
        return dict(value=float("nan"), status="unknown", starts=len(starts), converged=0,
                    solution_clusters=0, score=float("nan"), score_gap=float("nan"), peak_share=float("nan"),
                    second_value=float("nan"), second_score=float("nan"), relative_second=float("nan"),
                    iterations=max_iterations)
    # Keep the first accepted-order seed when multiple seeds converge to the same mode.
    unique = []
    for item in sorted(solutions, key=lambda x: x[1]):
        if not any(abs(item[2] - prev[2]) <= 5 * TOL_MM for prev in unique):
            unique.append(item)
    ranked = sorted(unique, key=lambda x: (-x[0], x[1]))
    best = ranked[0]
    # A second seed can converge to the same basin or nearby duplicate solution.
    # Only call it a competing support peak when it is separated from the best by >=2h.
    second = next((item for item in ranked[1:] if abs(item[2] - best[2]) >= 2.0 * h_mm), None)
    status = "partial" if failed else "converged"
    return dict(value=best[2], status=status, starts=len(starts), converged=len(solutions),
        solution_clusters=len(unique), score=best[0],
        score_gap=(best[0] - second[0]) if second else float("nan"),
        peak_share=best[3], second_value=second[2] if second else float("nan"),
        second_score=second[0] if second else float("nan"),
        relative_second=(second[0] / best[0]) if second and best[0] > 0 else float("nan"),
        iterations=max_iterations)


def _ray_estimates(z_mm: np.ndarray, w: np.ndarray, host_target_mm: np.ndarray):
    if len(z_mm) == 0:
        vals = {name: float("nan") for name in METHODS}
        vals["native_median"] = float(host_target_mm[1])
        return vals, dict(status="empty", members=0, weighted_sd_mm=float("nan"),
            huber_status="unknown", huber_iterations=0, mode={}, support={})
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Saved accepted-ray weights must be finite and positive")
    if not np.isfinite(z_mm).all():
        vals = {name: float("nan") for name in METHODS}
        vals["native_median"] = float(host_target_mm[1])
        return vals, dict(status="nonfinite_depth", members=len(z_mm), weighted_sd_mm=float("nan"),
            huber_status="unknown", huber_iterations=0, mode={}, support={})
    fixed = _fixed_subsets(z_mm, w)
    expected_scene = fixed["expected"]
    if np.isfinite(host_target_mm[0]):
        # Caller compares the unscaled value against the original host expected output.
        pass
    huber, huber_status, huber_iterations = _huber_location(z_mm, w, expected_scene, 1.0)
    modes = {h: _kernel_mode(z_mm, w, h) for h in MODE_BANDWIDTHS_MM}
    total = float(w.sum(dtype=np.float64))
    mu = expected_scene
    weighted_sd = float(np.sqrt(np.dot(w, (z_mm - mu) ** 2) / total))
    support = {}
    for h in MODE_BANDWIDTHS_MM:
        s0, p0 = _support(z_mm, w, mu, h)
        m = modes[h]
        support[h] = dict(expected=s0, expected_peak_share=p0, mode=m["score"],
                          mode_peak_share=m["peak_share"])
    vals = dict(expected=mu, native_median=float(host_target_mm[1]),
        weighted_median=fixed["weighted_median"], top10=fixed["top10"],
        top50=fixed["top50"], cover90=fixed["cover90"], huber_1mm=huber,
        kernel_mode_0p5mm=modes[0.5]["value"], kernel_mode_1mm=modes[1.0]["value"],
        kernel_mode_2mm=modes[2.0]["value"])
    details = dict(status="finite", members=len(z_mm), weighted_sd_mm=weighted_sd,
        huber_status=huber_status, huber_iterations=huber_iterations,
        mode=modes, support=support)
    return vals, details


def _points_mm(xy: np.ndarray, depth_scene: np.ndarray, camera_meta: dict, alignment: np.ndarray):
    h, width = map(int, camera_meta["image_size"])
    if not len(xy):
        return np.empty((0, 3), dtype=np.float64)
    fx, fy = map(float, camera_meta["focal"])
    cx, cy = (width - 1) / 2.0, (h - 1) / 2.0
    z = np.asarray(depth_scene, dtype=np.float64)
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    camera_points = np.column_stack(((x - cx) / fx * z, (y - cy) / fy * z, z))
    w2c = np.asarray(camera_meta["world_view_transform"], dtype=np.float64).T
    c2w = np.linalg.inv(w2c)
    world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    return world @ alignment[:3, :3].T + alignment[:3, 3]


def _load_gt(path: Path):
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"]
    points = np.column_stack([np.asarray(vertex[k], dtype=np.float64) for k in ("x", "y", "z")])
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Invalid independent DTU point cloud: {path}")
    return points


def _make_gt_tree(gt_points: np.ndarray):
    from scipy.spatial import cKDTree
    return cKDTree(gt_points, leafsize=32)


def _query_exact(points: np.ndarray, tree):
    return tree.query(points, k=1, eps=0, workers=1)[0]


def _load_scan(scan: int, ray_root: Path, input_root: Path, influence_root: Path):
    from numpy.lib.npyio import NpzFile
    scan_in = input_root / f"scan{scan}"
    model = scan_in / "point_cloud.ply"
    cameras = scan_in / "cameras.npz"
    gt = scan_in / f"stl{scan:03d}_total.ply"
    for path in (model, cameras, gt):
        if not path.is_file():
            raise FileNotFoundError(path)
    inf_meta_path = influence_root / f"scan{scan}" / "metadata.json"
    inf_meta = json.loads(inf_meta_path.read_text(encoding="utf-8"))
    if inf_meta.get("status") != "complete" or inf_meta.get("model_group") != "G1_MV":
        raise ValueError(f"Expected complete G1_MV influence metadata: {inf_meta_path}")
    old_dirs = (ray_root / f"scan{scan}", ray_root / f"scan{scan}_more")
    views = []
    for old_dir in old_dirs:
        meta_path = old_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("model_sha256") != inf_meta.get("model_sha256"):
            raise ValueError(f"Ray sample and complete G1 influence model differ for scan{scan}")
        if old_meta.get("git_remote") != inf_meta.get("git_remote"):
            raise ValueError(f"Unexpected source remote in ray metadata: {meta_path}")
        for vm in old_meta.get("views", []):
            if int(vm.get("random_samples", 0)) != 256 or int(vm.get("qualitative_samples", -1)) != 0:
                raise ValueError(f"Expected the original 256 random rays only: {meta_path}")
            ray_file = old_dir / vm["file"]
            if sha256(ray_file) != vm["npz_sha256"]:
                raise ValueError(f"Ray sample SHA256 mismatch: {ray_file}")
            views.append((old_meta, vm, ray_file))
    if len(views) != 3:
        raise ValueError(f"Expected exactly three saved views for scan{scan}; found {len(views)}")
    if sha256(model) != inf_meta.get("model_sha256"):
        raise ValueError(f"G1 model SHA256 mismatch: {model}")
    if sha256(cameras) != inf_meta.get("cameras_npz_sha256"):
        raise ValueError(f"Camera archive SHA256 mismatch: {cameras}")
    if sha256(gt) != inf_meta.get("gt_sha256"):
        raise ValueError(f"Independent GT SHA256 mismatch: {gt}")
    alignment_meta = inf_meta["alignment"]
    alignment = np.asarray(alignment_meta["transform"], dtype=np.float64)
    with np.load(cameras) as cam_npz:
        scale_mat = cam_npz["scale_mat_0"].astype(np.float64)
        np.testing.assert_allclose(scale_mat, alignment, rtol=0, atol=1e-6)
        scale = float(alignment_meta["scale_to_dtu_mm"])
        if not (scale > 0 and np.isfinite(scale)):
            raise ValueError(f"Invalid original shared DTU scale for scan{scan}")
        for _, vm, _ in views:
            idx = int(Path(vm["image_name"]).stem)
            np.testing.assert_allclose(cam_npz[f"scale_mat_{idx}"], scale_mat, rtol=0, atol=1e-6)
            check = next((x for x in alignment_meta["camera_checks"] if x["name"] == vm["image_name"]), None)
            if check is None or check["rotation_max_abs"] >= 1e-4 or check["translation_max_abs_scene_units"] >= 1e-4:
                raise ValueError(f"Existing camera-coordinate check missing/failed for {vm['image_name']}")
    return dict(model=model, cameras=cameras, gt=gt, influence_meta_path=inf_meta_path,
        inf_meta=inf_meta, alignment=alignment, scale=scale, views=views)


def _ray_metadata_records(ray_root: Path, scan: int):
    records = []
    for directory in (ray_root / f"scan{scan}", ray_root / f"scan{scan}_more"):
        meta_path = directory / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records.append(dict(path=str(meta_path.resolve()), sha256=sha256(meta_path),
            git_commit=meta.get("git_commit"), views=[dict(name=v["image_name"], file=v["file"],
                sha256=v["npz_sha256"]) for v in meta.get("views", [])]))
    return records


def _rq_decompose_projection(matrix: np.ndarray):
    """Small NumPy RQ decomposition matching OpenCV's projection decomposition."""
    q, r = np.linalg.qr(np.flipud(matrix).T)
    k = np.flipud(r.T)[:, ::-1]
    rotation = q.T[::-1, :]
    signs = np.where(np.diag(k) < 0, -1.0, 1.0)
    k = k @ np.diag(signs)
    rotation = np.diag(signs) @ rotation
    k /= k[2, 2]
    return k, rotation


def _check_camera_projection(camera_archive: Path, vm: dict, alignment: np.ndarray):
    idx = int(Path(vm["image_name"]).stem)
    with np.load(camera_archive) as archive:
        projection = np.asarray(archive[f"world_mat_{idx}"], dtype=np.float64) @ alignment
    p = projection[:3, :4]
    # P=K[R|t], so camera center is -R.T t and world-to-camera translation is t.
    intrinsics, rotation = _rq_decompose_projection(p[:, :3])
    kr = intrinsics @ rotation
    scale_factor = float(np.sum(kr * p[:, :3]) / np.sum(kr * kr))
    translation = np.linalg.solve(intrinsics, p[:, 3] / scale_factor)
    actual = np.asarray(vm["world_view_transform"], dtype=np.float64).T
    re = float(np.max(np.abs(rotation - actual[:3, :3])))
    te = float(np.max(np.abs(translation - actual[:3, 3])))
    if re >= 1e-4 or te >= 1e-4:
        raise ValueError(f"Projection/camera mismatch for {vm['image_name']}: R={re}, t={te}")
    return re, te


def _crosscheck_accuracy_map(scan: int, name: str, xy: np.ndarray, image_size: tuple[int, int], offsets: np.ndarray,
                             z: np.ndarray, w: np.ndarray, ray_errors: dict,
                             accuracy_root: Path, scan_data: dict):
    root = accuracy_root / f"scan{scan}"
    metadata_path = root / "metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    imeta = scan_data["inf_meta"]
    for key in ("model_sha256", "cameras_npz_sha256", "gt_sha256"):
        if meta.get(key) != imeta.get(key):
            raise ValueError(f"accuracy_v1 {key} mismatch for scan{scan}")
    if meta.get("status") != "complete" or meta.get("camera_names") != imeta.get("camera_names"):
        raise ValueError(f"Incomplete or different camera inventory in {metadata_path}")
    if meta.get("methods") != ["all", "top10pct_count", "top50pct_count", "cover90pct_weight"]:
        raise ValueError(f"Unexpected full-map method order in {metadata_path}")
    np.testing.assert_allclose(meta["alignment"]["transform"], imeta["alignment"]["transform"],
                               rtol=0, atol=1e-8)
    vm = next((v for v in meta["views"] if v["name"] == name), None)
    if vm is None:
        raise ValueError(f"Full-map view missing: scan{scan}/{name}")
    map_path = root / vm["file"]
    if sha256(map_path) != vm["sha256"]:
        raise ValueError(f"Full-map SHA256 mismatch: {map_path}")
    with np.load(map_path) as amap:
        pixels = amap["pixel_index"].astype(np.int64)
        if np.any(pixels[1:] <= pixels[:-1]):
            raise ValueError(f"Full-map pixel_index must be strictly increasing: {map_path}")
        map_image_size = tuple(map(int, amap["image_size"]))
        if map_image_size != image_size:
            raise ValueError(f"Full-map image dimensions mismatch for scan{scan}/{name}")
        wanted = xy[:, 1].astype(np.int64) * int(amap["image_size"][1]) + xy[:, 0]
        lookup = np.searchsorted(pixels, wanted)
        safe = np.minimum(lookup, max(0, len(pixels) - 1))
        if not len(pixels) or np.any(lookup >= len(pixels)) or np.any(pixels[safe] != wanted):
            raise ValueError(f"Sampled ray pixels absent from full-map archive: scan{scan}/{name}")
        idx = lookup
        full_means = amap["weighted_mean_depth"][idx].astype(np.float64)
        full_counts = amap["selected_counts"][idx].astype(np.int64)
        full_errors = amap["gt_point_distance_mm"][idx].astype(np.float64)
    expected_counts = np.zeros((len(xy), 4), dtype=np.int64)
    expected_means = np.full((len(xy), 4), np.nan)
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        zi, wi = z[a:b].astype(np.float64), w[a:b].astype(np.float64)
        fixed = _fixed_subsets(zi, wi)
        expected_means[i] = [fixed[k] for k in ("expected", "top10", "top50", "cover90")]
        expected_counts[i] = [len(zi), math.ceil(.1*len(zi)), math.ceil(.5*len(zi)),
            (min(len(zi), int(np.searchsorted(np.cumsum(wi[np.argsort(-wi, kind="stable")]),
                (.9-1e-12)*wi.sum(dtype=np.float64), side="left")+1)) if len(zi) else 0)]
    if not np.array_equal(full_counts, expected_counts):
        raise ValueError(f"Full-map selected counts disagree for scan{scan}/{name}")
    valid = np.isfinite(expected_means)
    if not np.array_equal(valid, np.isfinite(full_means)):
        raise ValueError(f"Full-map mean finite-mask mismatch at scan{scan}/{name}")
    max_depth_err = float(np.max(np.abs(full_means[valid] - expected_means[valid]))) if valid.any() else 0.0
    if max_depth_err > 2e-6:
        raise ValueError(f"Full-map float32 depth replay mismatch {max_depth_err:g} scene units at scan{scan}/{name}")
    method_keys = ("expected", "top10", "top50", "cover90")
    errors = np.asarray([[ray_errors[i][f"nn_{m}_mm"] for m in method_keys] for i in range(len(xy))])
    valid_err = np.isfinite(errors) & np.isfinite(full_errors)
    if not np.array_equal(np.isfinite(errors), np.isfinite(full_errors)):
        raise ValueError(f"Full-map GT-NN finite-mask mismatch at scan{scan}/{name}")
    max_nn_err = float(np.max(np.abs(full_errors[valid_err] - errors[valid_err]))) if valid_err.any() else 0.0
    if max_nn_err > 3e-4:
        raise ValueError(f"Full-map float32 GT-NN crosscheck mismatch {max_nn_err:g} mm at scan{scan}/{name}")
    return dict(map_path=str(map_path.resolve()), map_sha256=vm["sha256"],
        sampled_rays=len(xy), max_depth_abs_scene_units=max_depth_err,
        max_gt_nn_abs_mm=max_nn_err, count_match=True)


def _process_view(scan: int, old_meta: dict, vm: dict, ray_file: Path, scan_data: dict,
                  gt_tree, accuracy_root: Path):
    from numpy.lib.npyio import NpzFile
    name = vm["image_name"]
    camera_check = next((x for x in scan_data["inf_meta"]["alignment"]["camera_checks"] if x["name"] == name), None)
    if camera_check is None:
        raise ValueError(f"View absent from verified camera inventory: scan{scan}/{name}")
    with np.load(ray_file) as data:
        offsets = data["offsets"].astype(np.int64)
        depth = data["depth"].astype(np.float64)
        weight = data["weight"].astype(np.float64)
        xy = data["pixel_xy"].astype(np.int32)
        targets = data["host_target"].astype(np.float64)
        image_size = tuple(map(int, data["image_size"]))
        plane_degenerate = data["plane_degenerate"].astype(bool)
    if image_size != tuple(vm["image_size"]) or targets.shape != (len(xy), 3) or len(xy) != 256:
        raise ValueError(f"Ray archive and metadata dimensions disagree: {ray_file}")
    camera_names = scan_data["inf_meta"]["camera_names"]
    if name not in camera_names or name != f"{int(Path(name).stem):04d}":
        raise ValueError(f"Unexpected camera identity {name!r}")
    h, width = image_size
    if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= h):
        raise ValueError(f"Sampled pixel lies outside its recorded image: {ray_file}")
    camera = vm
    w2c = np.asarray(camera["world_view_transform"], dtype=np.float64).T
    if not np.isfinite(w2c).all() or not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError(f"Invalid saved view transform: scan{scan}/{name}")
    c2w = np.linalg.inv(w2c)
    center = np.asarray(camera["camera_center"], dtype=np.float64)
    np.testing.assert_allclose(c2w[:3, 3], center, rtol=0, atol=2e-5)
    rows, expected_replay = [], []
    all_values, all_details = [], []
    projection_check = _check_camera_projection(scan_data["cameras"], vm, scan_data["alignment"])
    for ray in range(len(xy)):
        first, last = int(offsets[ray]), int(offsets[ray + 1])
        z = depth[first:last]
        w = weight[first:last]
        if len(z) and (not np.isfinite(w).all() or np.any(w <= 0)):
            raise ValueError(f"Invalid accepted weights at scan{scan}/{name} ray {ray}")
        z_mm = z * scan_data["scale"]
        vals, detail = _ray_estimates(z_mm, w, targets[ray] * scan_data["scale"])
        np.testing.assert_allclose(vals["native_median"] / scan_data["scale"], targets[ray, 1],
            rtol=0, atol=1e-12, equal_nan=True,
            err_msg=f"Native median unit roundtrip mismatch: scan{scan}/{name} ray{ray}")
        expected_replay.append((vals["expected"] / scan_data["scale"] if np.isfinite(vals["expected"]) else np.nan,
                                targets[ray, 0]))
        for method in METHODS:
            if np.isfinite(vals[method]) and vals[method] > 0:
                point = _points_mm(xy[ray:ray+1], np.array([vals[method] / scan_data["scale"]]),
                                   camera, scan_data["alignment"])
                error = float(_query_exact(point, gt_tree)[0])
            else:
                error = float("nan")
            vals[f"nn_{method}_mm"] = error
            vals[f"shift_{method}_mm"] = vals[method] - vals["expected"] if np.isfinite(vals[method]) and np.isfinite(vals["expected"]) else float("nan")
        vals["native_minus_weighted_median_mm"] = vals["native_median"] - vals["weighted_median"] if np.isfinite(vals["native_median"]) and np.isfinite(vals["weighted_median"]) else float("nan")
        for hband in MODE_BANDWIDTHS_MM:
            vals[f"support_expected_h{hband:g}mm"] = detail["support"].get(hband, {}).get("expected", float("nan"))
            vals[f"support_mode_h{hband:g}mm"] = detail["support"].get(hband, {}).get("mode", float("nan"))
            vals[f"mode_peak_share_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("peak_share", float("nan"))
            vals[f"mode_status_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("status", "unknown")
            vals[f"mode_score_gap_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("score_gap", float("nan"))
            vals[f"mode_solution_clusters_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("solution_clusters", 0)
            vals[f"mode_second_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("second_value", float("nan"))
            vals[f"mode_relative_second_h{hband:g}mm"] = detail["mode"].get(hband, {}).get("relative_second", float("nan"))
        row = dict(scan=scan, view=name, ray=ray, x=int(xy[ray, 0]), y=int(xy[ray, 1]),
            accepted=len(z), ray_status=detail["status"], weighted_sd_mm=detail["weighted_sd_mm"],
            contribution_mass=float(np.sum(w, dtype=np.float64)),
            plane_degenerate_count=int(np.count_nonzero(plane_degenerate[first:last])),
            nonpositive_depth_count=int(np.count_nonzero(~(z > 0))),
            invalid_member_count=int(np.count_nonzero(~np.isfinite(z))),
            huber_status=detail["huber_status"], huber_iterations=detail["huber_iterations"])
        row.update(vals)
        rows.append(row)
        all_values.append(vals)
        all_details.append(detail)
    if expected_replay:
        pair = np.asarray(expected_replay, dtype=np.float64)
        tolerance = old_meta["replay_tolerance"]
        np.testing.assert_allclose(pair[:, 0], pair[:, 1], rtol=tolerance["rtol"],
                                   atol=tolerance["atol"], equal_nan=True,
                                   err_msg=f"Original host expected-depth replay mismatch: scan{scan}/{name}")
    crosscheck = _crosscheck_accuracy_map(scan, name, xy, image_size, offsets, depth, weight,
        all_values, accuracy_root, scan_data)
    for row in rows:
        row["coordinate_rotation_max_abs"] = projection_check[0]
        row["coordinate_translation_max_abs"] = projection_check[1]
        row["accuracy_map_depth_max_abs_scene"] = crosscheck["max_depth_abs_scene_units"]
        row["accuracy_map_nn_max_abs_mm"] = crosscheck["max_gt_nn_abs_mm"]
        # Main pre-registered support diagnostic: weak support at expected, with a competing peak >=2h away.
        row["support_ratio_R_h1mm"] = (row["support_expected_h1mm"] / row["support_mode_h1mm"]
            if row["support_mode_h1mm"] > 0 else float("nan"))
        row["weak_expected_support_R_lt_0p5"] = int(np.isfinite(row["support_ratio_R_h1mm"])
            and row["support_ratio_R_h1mm"] < .5)
        # Each ray's mode metadata is copied explicitly; no aggregate inference.
        row["competing_peak_ge_2h"] = int(np.isfinite(row["mode_second_h1mm"]) and
            abs(row["mode_second_h1mm"] - row["kernel_mode_1mm"]) >= 2.0)
        row["competing_peak_separation_mm"] = (abs(row["mode_second_h1mm"] - row["kernel_mode_1mm"])
            if np.isfinite(row["mode_second_h1mm"]) else float("nan"))
        row["competing_peak_relative_support_h1mm"] = row["mode_relative_second_h1mm"]
        row["weak_support_and_competing_peak"] = int(row["weak_expected_support_R_lt_0p5"] and row["competing_peak_ge_2h"])
    return rows, _summarize(scan, name, rows), all_details, crosscheck


def _summarize(scan: int, view: str, rows: list[dict]):
    summary = []
    for method in METHODS:
        errors = np.array([r[f"nn_{method}_mm"] for r in rows], dtype=np.float64)
        baseline = np.array([r["nn_expected_mm"] for r in rows], dtype=np.float64)
        paired = np.isfinite(errors) & np.isfinite(baseline)
        delta = errors[paired] - baseline[paired]
        summary.append(dict(scan=scan, view=view, method=method,
            rays=len(rows), finite_n=int(np.isfinite(errors).sum()), paired_n=int(paired.sum()),
            mean_nn_mm=float(np.mean(errors[np.isfinite(errors)])) if np.isfinite(errors).any() else None,
            mean_delta_mm=float(delta.mean()) if len(delta) else None,
            improved=int((delta < -TIE_MM).sum()), worsened=int((delta > TIE_MM).sum()),
            tied=int((np.abs(delta) <= TIE_MM).sum()),
            mean_shift_mm=float(np.nanmean([r[f"shift_{method}_mm"] for r in rows]))
                if np.isfinite([r[f"shift_{method}_mm"] for r in rows]).any() else None))
    for hband in MODE_BANDWIDTHS_MM:
        status = [r[f"mode_status_h{hband:g}mm"] for r in rows]
        summary.append(dict(scan=scan, view=view, method=f"mode_convergence_h{hband:g}mm",
            rays=len(rows), finite_n=sum(r[f"mode_solution_clusters_h{hband:g}mm"] > 0 for r in rows),
            paired_n=None, mean_nn_mm=None, mean_delta_mm=None, improved=None, worsened=None, tied=None,
            mean_shift_mm=None, converged=sum(x == "converged" for x in status),
            partial=sum(x == "partial" for x in status), unknown=sum(x == "unknown" for x in status)))
    base_errors = np.array([r["nn_expected_mm"] for r in rows], dtype=np.float64)
    rvals = np.array([r["support_ratio_R_h1mm"] for r in rows], dtype=np.float64)
    comp = np.array([r["competing_peak_ge_2h"] == 1 for r in rows], dtype=bool)
    for method in ("top10", "kernel_mode_1mm"):
        method_errors = np.array([r[f"nn_{method}_mm"] for r in rows], dtype=np.float64)
        for weak in (False, True):
            for has_competitor in (False, True):
                selected = np.isfinite(rvals) & (((rvals < .5) == weak))
                selected &= comp == has_competitor
                paired = selected & np.isfinite(method_errors) & np.isfinite(base_errors)
                delta = method_errors[paired] - base_errors[paired]
                summary.append(dict(scan=scan, view=view,
                    method=f"{method}_Rlt0p5_{int(weak)}_competitor_ge2mm_{int(has_competitor)}",
                    rays=int(selected.sum()), finite_n=int(np.isfinite(method_errors[selected]).sum()),
                    paired_n=int(paired.sum()),
                    mean_nn_mm=float(np.mean(method_errors[paired])) if paired.any() else None,
                    mean_delta_mm=float(delta.mean()) if len(delta) else None,
                    improved=int((delta < -TIE_MM).sum()), worsened=int((delta > TIE_MM).sum()),
                    tied=int((np.abs(delta) <= TIE_MM).sum()), mean_shift_mm=None))
    return summary


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _source_provenance():
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError(f"Source checkout must be clean before analysis: {dirty}")
    from importlib.metadata import version
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        remote=git("remote", "get-url", "origin"), clean=True,
        python=sys.version, platform=platform.platform(),
        packages={name: version(name) for name in ("numpy", "scipy", "plyfile")})


def _peak_diagnostics(expected: float, best: float, second: float, expected_nn: float,
                      best_nn: float, second_nn: float, relative_support: float, h_mm: float):
    finite = np.isfinite([expected, best, second]).all()
    separation = abs(second - best) if finite else float("nan")
    between = bool(finite and min(best, second) <= expected <= max(best, second))
    separated = bool(finite and separation >= 2.0 * h_mm)
    geometry_candidate = bool(separated and between and np.isfinite([expected_nn, best_nn, second_nn]).all()
        and best_nn <= 1.0 and second_nn <= 1.0 and expected_nn >= 2.0)
    return dict(second_finite=bool(np.isfinite(second)), separation_mm=separation,
        expected_between=between, separated_2h=separated,
        two_position_scan_geometry_candidate=geometry_candidate,
        relative_support_ge_0p25=bool(np.isfinite(relative_support) and relative_support >= .25))


def _number(row: dict, key: str):
    value = row.get(key, "")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _summarize_peak_rows(rows: list[dict], scan, view):
    out = []
    for h in MODE_BANDWIDTHS_MM:
        tag = f"{h:g}"
        n = len(rows)
        status = [r[f"mode_status_h{tag}mm"] for r in rows]
        rvals = np.array([_number(r, "support_ratio_R_h1mm") for r in rows])
        candidates = np.array([r[f"two_position_scan_geometry_candidate_h{tag}mm"] == 1 for r in rows])
        rel = np.array([_number(r, f"second_relative_support_{tag}mm") for r in rows])
        rel_ge = np.array([r[f"second_relative_support_ge_0p25_h{tag}mm"] == 1 for r in rows])
        second_finite = np.array([r[f"second_finite_h{tag}mm"] == 1 for r in rows])
        sep = np.array([r[f"separated_2h_h{tag}mm"] == 1 for r in rows])
        between = np.array([r[f"expected_between_h{tag}mm"] == 1 for r in rows])
        best_err = np.array([_number(r, f"nn_best_{tag}mm") for r in rows])
        second_err = np.array([_number(r, f"nn_second_{tag}mm") for r in rows])
        expected_err = np.array([_number(r, "nn_expected_mm") for r in rows])
        row = dict(scan=scan, view=view, bandwidth_mm=h, denominator_rays=n,
            mode_converged=sum(x == "converged" for x in status),
            mode_partial=sum(x == "partial" for x in status), mode_unknown=sum(x == "unknown" for x in status),
            R_finite=int(np.isfinite(rvals).sum()), R_gt_1=int((rvals[np.isfinite(rvals)] > 1).sum()),
            R_gt_1_converged=int(sum(np.isfinite(rvals[i]) and rvals[i] > 1 and status[i] == "converged" for i in range(n))),
            R_gt_1_partial=int(sum(np.isfinite(rvals[i]) and rvals[i] > 1 and status[i] == "partial" for i in range(n))),
            R_gt_1_unknown=int(sum(np.isfinite(rvals[i]) and rvals[i] > 1 and status[i] == "unknown" for i in range(n))),
            second_finite=int(second_finite.sum()), separated_ge_2h=int(sep.sum()),
            expected_between=int(between.sum()), candidate_n=int(candidates.sum()),
            second_relative_support_ge_0p25=int(rel_ge.sum()),
            second_relative_support_lt_0p25=int(np.sum(second_finite & np.isfinite(rel) & (rel < .25))),
            candidate_relative_support_ge_0p25=int(np.sum(candidates & np.isfinite(rel) & (rel >= .25))),
            candidate_relative_support_lt_0p25=int(np.sum(candidates & np.isfinite(rel) & (rel < .25))),
            relative_support_finite=int(np.isfinite(rel).sum()),
            mean_expected_nn_mm=float(np.nanmean(expected_err)) if np.isfinite(expected_err).any() else None,
            mean_best_nn_mm=float(np.nanmean(best_err)) if np.isfinite(best_err).any() else None,
            mean_second_nn_mm=float(np.nanmean(second_err)) if np.isfinite(second_err).any() else None)
        out.append(row)
    return out


def audit_peak_geometry(args):
    ray_root, input_root, influence_root, csv_path, out = map(Path,
        (args.ray_root, args.input_root, args.influence_root, args.rays_csv, args.output))
    if not out.is_absolute():
        raise ValueError("--output must be an absolute independent directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty peak-geometry output: {out}")
    v2_meta_path = csv_path.parent / "run_metadata.json"
    v2_meta = json.loads(v2_meta_path.read_text(encoding="utf-8"))
    if v2_meta.get("status") != "complete" or v2_meta.get("ray_count") != 3072 or \
            v2_meta.get("source", {}).get("commit") != "7aba27a72929dd47ddf20dd7a91e9f6116062745":
        raise ValueError("Expected the validated ray_support_v2 output and source commit")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    keyed = {}
    for r in source_rows:
        key = (int(r["scan"]), r["view"], int(r["ray"]))
        if key in keyed:
            raise ValueError(f"Duplicate ray row in v2 CSV: {key}")
        keyed[key] = r
    if len(keyed) != 3072:
        raise ValueError(f"Expected all 3072 v2 rows, got {len(keyed)}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ray_rows, view_rows, scan_rows, input_hashes, examples = [], [], [], [], {"positive": [], "counterexample": []}
    for scan in SCANS:
        scan_data = _load_scan(scan, ray_root, input_root, influence_root)
        gt = _load_gt(scan_data["gt"])
        tree = _make_gt_tree(gt)
        input_hashes.append(dict(scan=scan, model_sha256=sha256(scan_data["model"]),
            cameras_sha256=sha256(scan_data["cameras"]), gt_sha256=sha256(scan_data["gt"]),
            influence_metadata_sha256=sha256(scan_data["influence_meta_path"])))
        scan_group_rows = []
        for old_meta, vm, ray_file in scan_data["views"]:
            name = vm["image_name"]
            checks = _check_camera_projection(scan_data["cameras"], vm, scan_data["alignment"])
            with np.load(ray_file) as ray_data:
                xy = ray_data["pixel_xy"].astype(np.int32)
                image_size = tuple(map(int, ray_data["image_size"]))
            if len(xy) != 256 or image_size != tuple(vm["image_size"]):
                raise ValueError(f"Unexpected saved rays or dimensions for scan{scan}/{name}")
            items = [keyed.get((scan, name, i)) for i in range(len(xy))]
            if any(x is None for x in items):
                raise ValueError(f"Missing v2 rows for scan{scan}/{name}")
            if any(int(r["x"]) != int(xy[i, 0]) or int(r["y"]) != int(xy[i, 1]) for i, r in enumerate(items)):
                raise ValueError(f"v2 pixel identity mismatch for scan{scan}/{name}")
            depths = {"expected": np.array([_number(r, "expected") for r in items])}
            for h in MODE_BANDWIDTHS_MM:
                tag = f"{h:g}"
                method_tag = "0p5" if h == 0.5 else tag
                depths[f"mode_{tag}"] = np.array([_number(r, f"kernel_mode_{method_tag}mm") for r in items])
                depths[f"second_{tag}"] = np.array([_number(r, f"mode_second_h{tag}mm") for r in items])
            errors = {}
            for label, dmm in depths.items():
                valid = np.isfinite(dmm) & (dmm > 0)
                result = np.full(len(xy), np.nan)
                if valid.any():
                    points = _points_mm(xy[valid], dmm[valid] / scan_data["scale"], vm, scan_data["alignment"])
                    result[valid] = _query_exact(points, tree)
                errors[label] = result
            # Confirm the repeated expected and best-mode NN values use precisely the v2 camera/GT coordinates.
            for i, r in enumerate(items):
                for label, key in (("expected", "nn_expected_mm"), ("mode_1", "nn_kernel_mode_1mm_mm")):
                    saved, replay = _number(r, key), errors[label][i]
                    if np.isfinite(saved) != np.isfinite(replay) or (np.isfinite(saved) and abs(saved-replay) > 1e-6):
                        raise ValueError(f"NN re-projection mismatch scan{scan}/{name}/ray{i}/{label}")
            local_rows = []
            for i, base in enumerate(items):
                row = dict(scan=scan, view=name, ray=i, x=int(xy[i, 0]), y=int(xy[i, 1]),
                    expected_mm=depths["expected"][i], nn_expected_mm=errors["expected"][i],
                    support_ratio_R_h1mm=_number(base, "support_ratio_R_h1mm"),
                    source_ray_status=base["ray_status"],
                    camera_rotation_max_abs=checks[0], camera_translation_max_abs=checks[1])
                for h in MODE_BANDWIDTHS_MM:
                    tag = f"{h:g}"
                    best, second = depths[f"mode_{tag}"], depths[f"second_{tag}"]
                    row[f"mode_status_h{tag}mm"] = base[f"mode_status_h{tag}mm"]
                    rel = _number(base, f"mode_relative_second_h{tag}mm")
                    d = _peak_diagnostics(depths["expected"][i], best[i], second[i],
                        errors["expected"][i], errors[f"mode_{tag}"][i], errors[f"second_{tag}"][i], rel, h)
                    row[f"best_{tag}mm"] = best[i]
                    row[f"second_{tag}mm"] = second[i]
                    row[f"nn_best_{tag}mm"] = errors[f"mode_{tag}"][i]
                    row[f"nn_second_{tag}mm"] = errors[f"second_{tag}"][i]
                    row[f"second_relative_support_{tag}mm"] = rel
                    row[f"second_relative_support_ge_0p25_h{tag}mm"] = int(d["relative_support_ge_0p25"])
                    row[f"second_finite_h{tag}mm"] = int(d["second_finite"])
                    row[f"separation_h{tag}mm"] = d["separation_mm"]
                    row[f"expected_between_h{tag}mm"] = int(d["expected_between"])
                    row[f"separated_2h_h{tag}mm"] = int(d["separated_2h"])
                    row[f"two_position_scan_geometry_candidate_h{tag}mm"] = int(d["two_position_scan_geometry_candidate"])
                    if h == 1.0:
                        row["candidate_relative_support_ge_0p25"] = int(d["two_position_scan_geometry_candidate"] and d["relative_support_ge_0p25"])
                        row["candidate_relative_support_lt_0p25"] = int(d["two_position_scan_geometry_candidate"] and np.isfinite(rel) and rel < .25)
                local_rows.append(row)
            ray_rows.extend(local_rows)
            scan_group_rows.extend(local_rows)
            view_rows.extend(_summarize_peak_rows(local_rows, scan, name))
            positives = [r for r in local_rows if r["two_position_scan_geometry_candidate_h1mm"] == 1]
            negatives = [r for r in local_rows if r["separated_2h_h1mm"] == 1 and r["expected_between_h1mm"] == 1 and r["two_position_scan_geometry_candidate_h1mm"] == 0]
            examples["positive"].extend(positives)
            examples["counterexample"].extend(negatives)
        scan_rows.extend(_summarize_peak_rows(scan_group_rows, scan, "scan_pooled_sample"))
    scan_rows.extend(_summarize_peak_rows(ray_rows, "pooled", "all_3072_sampled_rays"))
    if len(ray_rows) != 3072:
        raise ValueError(f"Expected 3072 audited rays, got {len(ray_rows)}")
    _write_csv(out / "rays.csv", ray_rows)
    _write_csv(out / "view_summary.csv", view_rows)
    _write_csv(out / "scan_summary.csv", scan_rows)
    # Deterministic post-hoc examples are kept separate from the complete population summaries.
    examples["positive"] = sorted(examples["positive"], key=lambda r: (-r["second_relative_support_1mm"], r["scan"], r["view"], r["ray"]))[:3]
    examples["counterexample"] = sorted(examples["counterexample"], key=lambda r: (r["scan"], r["view"], r["ray"]))[:3]
    (out / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    metadata = dict(status="complete", source=_source_provenance(), source_ray_csv=str(csv_path.resolve()),
        source_ray_csv_sha256=sha256(csv_path), source_run_metadata_sha256=sha256(v2_meta_path),
        ray_count=len(ray_rows), views=len(view_rows)//len(MODE_BANDWIDTHS_MM), scans=list(SCANS),
        input_sha256=input_hashes, fixed_thresholds=dict(main_h_mm=1.0, separation_ge_2h=True,
            peak_nn_le_mm=1.0, expected_nn_ge_mm=2.0, expected_between_peaks=True, relative_support_reporting_threshold=.25),
        interpretation="Descriptive candidate for two scan-geometry-supported positions with a mean-depth gap; not proof of true surface, physical layer, visibility, or pixelwise ground truth. Sparse GT misses can affect nearest-neighbor distances. S(best) is selected to be high among converged search results by construction and is not correctness evidence. Partial search results are not called global maxima.",
        convergence_counts={h: {s: sum(r[f"mode_status_h{h:g}mm"]==s for r in ray_rows) for s in ("converged","partial","unknown")} for h in MODE_BANDWIDTHS_MM},
        R_gt_1_count=int(sum(bool(np.isfinite(_number(r,"support_ratio_R_h1mm")) and _number(r,"support_ratio_R_h1mm")>1) for r in ray_rows)),
        R_finite_count=int(sum(bool(np.isfinite(_number(r,"support_ratio_R_h1mm"))) for r in ray_rows)),
        elapsed_seconds=time.time()-started, output=str(out.resolve()))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def _peak_composition(z_mm: np.ndarray, weight: np.ndarray, peak_mm: float, top10: np.ndarray,
                      half_width_mm: float = 2.0, bandwidth_mm: float = 1.0):
    n = len(z_mm)
    total_w = float(weight.sum(dtype=np.float64))
    result = dict(window_q=float("nan"), window_n=None, window_neff=float("nan"),
        window_max_member_share=float("nan"), outside_mass=float("nan"),
        support_s=float("nan"), support_top10_s=float("nan"), support_rest_s=float("nan"),
        support_top10_fraction=float("nan"))
    if not np.isfinite(peak_mm) or n == 0:
        return result, None
    if not np.isfinite(z_mm).all():
        # Do not silently remove accepted members with unknown depth.
        result["window_n"] = None
        return result, None
    if not np.isfinite(weight).all() or np.any(weight <= 0) or not total_w > 0:
        raise ValueError("Accepted weights must be finite and positive")
    p = weight / total_w
    in_window = np.abs(z_mm - peak_mm) <= half_width_mm
    q = float(p[in_window].sum(dtype=np.float64))
    n_window = int(in_window.sum())
    squared = float(np.square(p[in_window]).sum(dtype=np.float64))
    neff = q * q / squared if squared > 0 else 0.0
    max_share = float(p[in_window].max() / q) if q > 0 else float("nan")
    kernel = np.exp(-0.5 * np.square((z_mm - peak_mm) / bandwidth_mm))
    contributions = p * kernel
    total_s = float(contributions.sum(dtype=np.float64))
    top_s = float(contributions[top10].sum(dtype=np.float64))
    rest_s = float(contributions[~top10].sum(dtype=np.float64))
    if not math.isclose(top_s + rest_s, total_s, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError("Top10/rest support contributions do not sum to total S")
    result.update(window_q=q, window_n=n_window, window_neff=neff,
        window_max_member_share=max_share, outside_mass=1.0-q,
        support_s=total_s, support_top10_s=top_s, support_rest_s=rest_s,
        support_top10_fraction=top_s / total_s if total_s > 0 else float("nan"))
    return result, in_window


def _composition_mean(rows, field):
    vals = np.asarray([_number(r, field) for r in rows], dtype=np.float64)
    finite = np.isfinite(vals)
    return int(finite.sum()), float(vals[finite].mean()) if finite.any() else None


def audit_peak_composition(args):
    ray_root, input_root, influence_root, csv_path, out = map(Path,
        (args.ray_root, args.input_root, args.influence_root, args.rays_csv, args.output))
    geometry_csv = Path(args.geometry_csv)
    if not out.is_absolute():
        raise ValueError("--output must be an absolute independent directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty peak-composition output: {out}")
    meta_path = csv_path.parent / "run_metadata.json"
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if source_meta.get("status") != "complete" or source_meta.get("ray_count") != 3072 or \
            source_meta.get("source", {}).get("commit") != "7aba27a72929dd47ddf20dd7a91e9f6116062745":
        raise ValueError("Expected the validated ray_support_v2 output and source commit")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    keyed = {(int(r["scan"]), r["view"], int(r["ray"])): r for r in source_rows}
    if len(keyed) != 3072 or len(source_rows) != 3072:
        raise ValueError("Expected 3072 unique v2 ray rows")
    geometry_meta_path = geometry_csv.parent / "run_metadata.json"
    geometry_meta = json.loads(geometry_meta_path.read_text(encoding="utf-8"))
    if geometry_meta.get("status") != "complete" or geometry_meta.get("ray_count") != 3072 or \
            geometry_meta.get("source", {}).get("commit") != "90c92402b3cdb86f8b548c5f02931ae5f8858829":
        raise ValueError("Expected the validated peak_geometry_v1_retry2 audit")
    with geometry_csv.open("r", newline="", encoding="utf-8-sig") as f:
        geometry_rows = list(csv.DictReader(f))
    geometry_keyed = {(int(r["scan"]), r["view"], int(r["ray"])): r for r in geometry_rows}
    if len(geometry_keyed) != 3072 or len(geometry_rows) != 3072:
        raise ValueError("Expected all 3072 unique geometry rows")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ray_rows, weak_summary, candidate_rows, input_hashes = [], [], [], []
    for scan in SCANS:
        data = _load_scan(scan, ray_root, input_root, influence_root)
        input_hashes.append(dict(scan=scan, model_sha256=sha256(data["model"]),
            cameras_sha256=sha256(data["cameras"]), gt_sha256=sha256(data["gt"]),
            influence_metadata_sha256=sha256(data["influence_meta_path"])))
        scan_rows = []
        for old_meta, vm, ray_file in data["views"]:
            with np.load(ray_file) as archive:
                offsets = archive["offsets"].astype(np.int64)
                depth = archive["depth"].astype(np.float64)
                weight = archive["weight"].astype(np.float64)
                plane = archive["plane_degenerate"].astype(bool)
                targets = archive["host_target"].astype(np.float64)
                xy = archive["pixel_xy"].astype(np.int32)
            if len(offsets) != 257 or offsets[0] != 0 or offsets[-1] != len(depth) or len(weight) != len(depth):
                raise ValueError(f"Invalid ragged ray offsets in {ray_file}")
            if len(plane) != len(depth) or targets.shape != (256, 3) or xy.shape != (256, 2):
                raise ValueError(f"Invalid ray metadata dimensions in {ray_file}")
            for ray in range(256):
                base = keyed.get((scan, vm["image_name"], ray))
                geometry = geometry_keyed.get((scan, vm["image_name"], ray))
                if (base is None or geometry is None or int(base["x"]) != int(xy[ray, 0]) or
                        int(base["y"]) != int(xy[ray, 1]) or int(geometry["x"]) != int(xy[ray, 0]) or
                        int(geometry["y"]) != int(xy[ray, 1])):
                    raise ValueError(f"v2 ray identity mismatch scan{scan}/{vm['image_name']}/{ray}")
                first, last = int(offsets[ray]), int(offsets[ray + 1])
                z_mm = depth[first:last] * data["scale"]
                w = weight[first:last]
                if not np.isfinite(w).all() or np.any(w <= 0):
                    raise ValueError(f"Invalid accepted weight at scan{scan}/{vm['image_name']}/{ray}")
                n = len(w)
                total_w = float(w.sum(dtype=np.float64))
                k10 = int(math.ceil(.1*n)) if n else 0
                order = np.argsort(-w, kind="stable")
                top10 = np.zeros(n, dtype=bool)
                top10[order[:k10]] = True
                best = _number(base, "kernel_mode_1mm")
                second = _number(base, "mode_second_h1mm")
                expected = _number(base, "expected")
                weak = int(np.isfinite(_number(base, "support_ratio_R_h1mm")) and _number(base, "support_ratio_R_h1mm") < .5)
                ratio_finite = bool(np.isfinite(_number(base, "support_ratio_R_h1mm")))
                nonweak = int(ratio_finite and _number(base, "support_ratio_R_h1mm") >= .5)
                candidate = int(geometry["two_position_scan_geometry_candidate_h1mm"] == "1")
                # The fixed candidate is copied from the already completed geometry audit.
                # This audit does not recompute that decision or select peaks.
                comp_best, mask_best = _peak_composition(z_mm, w, best, top10)
                comp_second, mask_second = _peak_composition(z_mm, w, second, top10)
                valid_depths = bool(np.isfinite(z_mm).all())
                overlap = None if mask_best is None or mask_second is None else bool(np.any(mask_best & mask_second))
                interval_overlap = (None if not (np.isfinite(best) and np.isfinite(second)) else
                                    bool(abs(best-second) <= 4.0))
                row = dict(scan=scan, view=vm["image_name"], ray=ray, x=int(xy[ray, 0]), y=int(xy[ray, 1]),
                    accepted=n, total_weight=total_w, top10_n=k10, nonfinite_depth_n=int((~np.isfinite(z_mm)).sum()),
                    plane_degenerate_n=int(plane[first:last].sum()),
                    plane_degenerate_any=int(plane[first:last].any()),
                    source_status=base["ray_status"], mode_status_h1mm=base["mode_status_h1mm"],
                    weak_R_lt_0p5=weak, geometry_candidate_h1mm=candidate,
                    R_finite=int(ratio_finite), R_unknown=int(not ratio_finite), nonweak_R_ge_0p5=nonweak,
                    support_ratio_R_h1mm=_number(base, "support_ratio_R_h1mm"), expected_mm=expected,
                    best_peak_mm=best, second_peak_mm=second, relative_second_support=_number(base, "mode_relative_second_h1mm"),
                    second_finite=int(np.isfinite(second)), peak_windows_member_overlap=overlap,
                    peak_windows_interval_overlap=interval_overlap, all_depths_finite=int(valid_depths))
                for prefix, comp in (("best", comp_best), ("second", comp_second)):
                    for key, val in comp.items():
                        row[f"{prefix}_{key}"] = val
                if candidate:
                    candidate_rows.append(row.copy())
                ray_rows.append(row)
                scan_rows.append(row)
        weak_rows = [r for r in scan_rows if r["weak_R_lt_0p5"] == 1]
        summary = dict(scan=scan, all_rays=len(scan_rows), weak_rays=len(weak_rows),
            R_finite=sum(r["R_finite"] for r in scan_rows), R_unknown=sum(r["R_unknown"] for r in scan_rows),
            nonweak_R_ge_0p5=sum(r["nonweak_R_ge_0p5"] for r in scan_rows),
            weak_second_missing=sum(r["second_finite"] == 0 for r in weak_rows),
            weak_partial_mode=sum(r["mode_status_h1mm"] == "partial" for r in weak_rows),
            weak_unknown_mode=sum(r["mode_status_h1mm"] == "unknown" for r in weak_rows))
        for peak in ("best", "second"):
            for field in ("window_q", "window_n", "window_neff", "window_max_member_share", "outside_mass",
                          "support_s", "support_top10_s", "support_rest_s", "support_top10_fraction"):
                count, mean = _composition_mean(weak_rows, f"{peak}_{field}")
                summary[f"weak_{peak}_{field}_finite_n"] = count
                summary[f"weak_{peak}_{field}_mean"] = mean
        weak_summary.append(summary)
    weak_rows = [r for r in ray_rows if r["weak_R_lt_0p5"] == 1]
    pooled = dict(scan="pooled", all_rays=len(ray_rows), weak_rays=len(weak_rows),
        R_finite=sum(r["R_finite"] for r in ray_rows), R_unknown=sum(r["R_unknown"] for r in ray_rows),
        nonweak_R_ge_0p5=sum(r["nonweak_R_ge_0p5"] for r in ray_rows),
        weak_second_missing=sum(r["second_finite"] == 0 for r in weak_rows),
        weak_partial_mode=sum(r["mode_status_h1mm"] == "partial" for r in weak_rows),
        weak_unknown_mode=sum(r["mode_status_h1mm"] == "unknown" for r in weak_rows))
    for peak in ("best", "second"):
        for field in ("window_q", "window_n", "window_neff", "window_max_member_share", "outside_mass",
                      "support_s", "support_top10_s", "support_rest_s", "support_top10_fraction"):
            count, mean = _composition_mean(weak_rows, f"{peak}_{field}")
            pooled[f"weak_{peak}_{field}_finite_n"] = count
            pooled[f"weak_{peak}_{field}_mean"] = mean
    weak_summary.append(pooled)
    if len(ray_rows) != 3072:
        raise ValueError(f"Expected all 3072 rays, got {len(ray_rows)}")
    if sum(r["weak_R_lt_0p5"] for r in ray_rows) != 83:
        raise ValueError("Expected the validated 83 weak-support rays in v2; source changed or audit mismatch")
    if len(candidate_rows) != 7:
        raise ValueError(f"Expected all seven fixed geometry candidates, got {len(candidate_rows)}")
    _write_csv(out / "rays.csv", ray_rows)
    _write_csv(out / "weak_scan_summary.csv", weak_summary)
    _write_csv(out / "geometry_candidates.csv", candidate_rows or [dict(note="no fixed-threshold geometry candidates")])
    metadata = dict(status="complete", source=_source_provenance(), ray_count=len(ray_rows),
        weak_R_lt_0p5_count=len(weak_rows), R_finite_count=sum(r["R_finite"] for r in ray_rows),
        R_unknown_count=sum(r["R_unknown"] for r in ray_rows), nonweak_R_ge_0p5_count=sum(r["nonweak_R_ge_0p5"] for r in ray_rows),
        geometry_candidate_count=len(candidate_rows),
        source_rays_csv=str(csv_path.resolve()), source_rays_sha256=sha256(csv_path),
        source_run_metadata_sha256=sha256(meta_path), source_geometry_csv=str(geometry_csv.resolve()),
        source_geometry_csv_sha256=sha256(geometry_csv), source_geometry_metadata_sha256=sha256(geometry_meta_path),
        input_sha256=input_hashes,
        fixed_parameters=dict(bandwidth_mm=1.0, peak_window_half_width_mm=2.0,
            top_members="stable descending original accepted weight; ceil(10% N)",
            weak_support="saved v2 R_h1mm < 0.5", geometry_candidate="copied fixed h1mm flag from validated geometry CSV; no recomputation"),
        interpretation="Window mass/member summaries and exact support-score contributions describe numerical peak composition, not physical layers or probabilities. All accepted weights are retained. Nonfinite accepted depths make depth-dependent composition unknown, not silently filtered.",
        elapsed_seconds=time.time()-started, output=str(out.resolve()))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


GATE_METHODS = {
    "A_all_mode1": "Select the saved 1 mm best mode regardless of converged/partial status; partial is a coarse-search result, not a global optimum.",
    "A_converged_only": "Select the saved 1 mm mode only when status is converged; partial/unknown remains unknown.",
    "B_weak_converged": "Replace expected only when R<0.5 and the saved mode status is converged.",
    "C_weak_converged_dominant": "B plus no saved second peak or relative second support<0.5.",
    "D_unknown_without_gate": "Expected for R>=0.5; mode for C-qualified weak rays; unknown otherwise.",
}


def _gate_one(name, expected, mode, second, ratio, rel, status, expected_nn, mode_nn):
    finite_ratio = np.isfinite(ratio)
    weak = bool(finite_ratio and ratio < .5)
    dominant = bool(not np.isfinite(second) or (np.isfinite(rel) and rel < .5))
    can_use_mode = bool(weak and status == "converged" and dominant)
    use_mode = name == "A_all_mode1" or (name == "A_converged_only" and status == "converged") or \
        (name in ("B_weak_converged", "C_weak_converged_dominant") and
        weak and status == "converged" and (name == "B_weak_converged" or dominant)) or \
        (name == "D_unknown_without_gate" and can_use_mode)
    force_unknown = bool((name == "A_converged_only" and status != "converged") or
        (name == "D_unknown_without_gate" and not (finite_ratio and ratio >= .5) and not can_use_mode))
    if use_mode:
        depth, nn, decision = mode, mode_nn, "mode"
    elif force_unknown:
        depth, nn, decision = float("nan"), float("nan"), "unknown"
    else:
        depth, nn, decision = expected, expected_nn, "expected"
    switched = bool(np.isfinite(depth) != np.isfinite(expected) or
        (np.isfinite(depth) and np.isfinite(expected) and abs(depth - expected) > TIE_MM))
    return dict(depth=depth, nn=nn, decision=decision, switched=int(switched),
        mode_branch=int(use_mode), unknown=int(not np.isfinite(depth)))


def _summarize_gate(rows, scope, scan, view):
    result = []
    for method, label in GATE_METHODS.items():
        n = len(rows)
        baseline = np.asarray([_number(r, "expected_nn_mm") for r in rows], dtype=np.float64)
        candidate = np.asarray([_number(r, f"{method}_nn_mm") for r in rows], dtype=np.float64)
        switched = np.asarray([r[f"{method}_switched"] == 1 for r in rows])
        mode_branch = np.asarray([r[f"{method}_mode_branch"] == 1 for r in rows])
        depth_finite = np.asarray([np.isfinite(_number(r, f"{method}_depth_mm")) for r in rows])
        pair = np.isfinite(baseline) & np.isfinite(candidate)
        delta = candidate[pair] - baseline[pair]
        switch_pair = pair & switched
        delta_sw = candidate[switch_pair] - baseline[switch_pair]
        status = [r["mode_status_h1mm"] for r in rows]
        ratios = np.asarray([_number(r, "support_ratio_R_h1mm") for r in rows])
        out = dict(scope=scope, scan=scan, view=view, method=method, rule=label,
            denominator_rays=n, finite_n=int(depth_finite.sum()), unknown_n=int((~depth_finite).sum()),
            mode_branch_n=int(mode_branch.sum()), actual_switch_n=int(switched.sum()),
            mode_converged_n=sum(x == "converged" for x in status),
            mode_partial_n=sum(x == "partial" for x in status), mode_unknown_n=sum(x == "unknown" for x in status),
            R_finite_n=int(np.isfinite(ratios).sum()), R_unknown_n=int((~np.isfinite(ratios)).sum()),
            R_gt_1_n=int(np.sum(np.isfinite(ratios) & (ratios > 1))), paired_n=int(pair.sum()),
            expected_nn_mean_paired_mm=float(baseline[pair].mean()) if pair.any() else None,
            candidate_nn_mean_paired_mm=float(candidate[pair].mean()) if pair.any() else None,
            delta_candidate_minus_expected_mean_mm=float(delta.mean()) if len(delta) else None,
            delta_candidate_minus_expected_median_mm=float(np.median(delta)) if len(delta) else None,
            improved=int(np.sum(delta < -TIE_MM)), worsened=int(np.sum(delta > TIE_MM)),
            tied=int(np.sum(np.abs(delta) <= TIE_MM)),
            switched_paired_n=int(switch_pair.sum()),
            switched_expected_nn_mean_mm=float(baseline[switch_pair].mean()) if switch_pair.any() else None,
            switched_candidate_nn_mean_mm=float(candidate[switch_pair].mean()) if switch_pair.any() else None,
            switched_delta_mean_mm=float(delta_sw.mean()) if len(delta_sw) else None,
            switched_delta_median_mm=float(np.median(delta_sw)) if len(delta_sw) else None,
            switched_improved=int(np.sum(delta_sw < -TIE_MM)), switched_worsened=int(np.sum(delta_sw > TIE_MM)),
            switched_tied=int(np.sum(np.abs(delta_sw) <= TIE_MM)))
        result.append(out)
    return result


def audit_support_gates(args):
    csv_path, out = Path(args.rays_csv), Path(args.output)
    started = time.time()
    if not out.is_absolute():
        raise ValueError("--output must be an absolute independent directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty support-gate output: {out}")
    meta_path = csv_path.parent / "run_metadata.json"
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if source_meta.get("status") != "complete" or source_meta.get("ray_count") != 3072 or \
            source_meta.get("source", {}).get("commit") != "7aba27a72929dd47ddf20dd7a91e9f6116062745":
        raise ValueError("Expected the validated ray_support_v2 output and source commit")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    if len(source_rows) != 3072 or len({(r["scan"], r["view"], r["ray"]) for r in source_rows}) != 3072:
        raise ValueError("Expected all 3072 unique ray_support_v2 rows")
    ray_rows = []
    for base in source_rows:
        expected = _number(base, "expected")
        mode = _number(base, "kernel_mode_1mm")
        second = _number(base, "mode_second_h1mm")
        ratio = _number(base, "support_ratio_R_h1mm")
        rel = _number(base, "mode_relative_second_h1mm")
        expected_nn = _number(base, "nn_expected_mm")
        mode_nn = _number(base, "nn_kernel_mode_1mm_mm")
        row = dict(scan=int(base["scan"]), view=base["view"], ray=int(base["ray"]),
            ray_status=base["ray_status"], mode_status_h1mm=base["mode_status_h1mm"],
            expected_depth_mm=expected, mode_depth_mm=mode, second_depth_mm=second,
            expected_nn_mm=expected_nn, support_ratio_R_h1mm=ratio,
            relative_second_support=rel, weak_R_lt_0p5=int(np.isfinite(ratio) and ratio < .5),
            R_finite=int(np.isfinite(ratio)), R_gt_1=int(np.isfinite(ratio) and ratio > 1))
        for method in GATE_METHODS:
            result = _gate_one(method, expected, mode, second, ratio, rel,
                base["mode_status_h1mm"], expected_nn, mode_nn)
            row[f"{method}_depth_mm"] = result["depth"]
            row[f"{method}_nn_mm"] = result["nn"]
            row[f"{method}_decision"] = result["decision"]
            row[f"{method}_mode_branch"] = result["mode_branch"]
            row[f"{method}_switched"] = result["switched"]
        ray_rows.append(row)
    if sum(r["weak_R_lt_0p5"] for r in ray_rows) != 83:
        raise ValueError("Expected the validated 83 weak R<0.5 rays")
    if sum(not r["R_finite"] for r in ray_rows) != 1:
        raise ValueError("Expected the validated single unknown-R ray; it must remain distinct from R>=0.5")
    summary = []
    summary.extend(_summarize_gate(ray_rows, "pooled", "pooled", "all_3072_sampled_rays"))
    for scan in SCANS:
        subset = [r for r in ray_rows if r["scan"] == scan]
        summary.extend(_summarize_gate(subset, "scan", scan, "scan_pooled_sample"))
        views = sorted({r["view"] for r in subset})
        for view in views:
            summary.extend(_summarize_gate([r for r in subset if r["view"] == view], "view", scan, view))
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "rays.csv", ray_rows)
    _write_csv(out / "summary.csv", summary)
    metadata = dict(status="complete", source=_source_provenance(), source_rays_csv=str(csv_path.resolve()),
        source_rays_sha256=sha256(csv_path), source_run_metadata_sha256=sha256(meta_path),
        ray_count=len(ray_rows), weak_count=sum(r["weak_R_lt_0p5"] for r in ray_rows),
        R_finite_count=sum(r["R_finite"] for r in ray_rows), R_unknown_count=sum(not r["R_finite"] for r in ray_rows),
        fixed_rules=GATE_METHODS, tie_tolerance_mm=TIE_MM,
        paired_delta_definition="candidate NN minus expected NN on identical finite-NN rows; negative is lower candidate NN. Switched-subset risk uses only rows where candidate differs from expected by more than the existing 1e-4 mm tie tolerance (or finite/unknown changes).",
        interpretation="Exploratory gates evaluated on the same four-scan sample used to inspect these mechanisms; not independent validation or a general improvement claim. A_all_mode1 includes partial bounded-search results and is a coarse-search comparison, not evidence about an intrinsic mode. A_converged_only leaves partial/unknown results unknown. R>1 counts are descriptive and do not establish a reliable support gate. Unknown R is not treated as R>=0.5.",
        elapsed_seconds=time.time()-started, output=str(out.resolve()))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run(args):
    ray_root, input_root, influence_root, accuracy_root, out = map(Path,
        (args.ray_root, args.input_root, args.influence_root, args.accuracy_root, args.output))
    if not out.is_absolute():
        raise ValueError("--output must be an absolute independent directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ray_rows, view_rows = [], []
    input_meta, crosschecks = [], []
    for scan in SCANS:
        data = _load_scan(scan, ray_root, input_root, influence_root)
        gt = _load_gt(data["gt"])
        gt_tree = _make_gt_tree(gt)
        input_meta.append(dict(scan=scan, model=str(data["model"].resolve()), model_sha256=sha256(data["model"]),
            cameras=str(data["cameras"].resolve()), cameras_sha256=sha256(data["cameras"]),
            gt=str(data["gt"].resolve()), gt_sha256=sha256(data["gt"]),
            influence_metadata_sha256=sha256(data["influence_meta_path"]),
            sampled_ray_metadata=_ray_metadata_records(ray_root, scan),
            alignment=data["alignment"].tolist(), scale_to_dtu_mm=data["scale"]))
        for old_meta, vm, rayfile in data["views"]:
            rows, summaries, _, crosscheck = _process_view(scan, old_meta, vm, rayfile, data, gt_tree, accuracy_root)
            ray_rows.extend(rows)
            view_rows.extend(summaries)
            crosschecks.append(dict(scan=scan, view=vm["image_name"], **crosscheck))
    if len(ray_rows) != 3072 or len(view_rows) != 12 * (len(METHODS) + len(MODE_BANDWIDTHS_MM) + 8):
        raise ValueError(f"Unexpected probe size: {len(ray_rows)} rays, {len(view_rows)} summary rows")
    scan_rows = []
    for scan in SCANS:
        subset = [r for r in ray_rows if r["scan"] == scan]
        scan_rows.extend(_summarize(scan, "scan_pooled_equal_view_sample", subset))
    _write_csv(out / "rays.csv", ray_rows)
    _write_csv(out / "view_summary.csv", view_rows)
    _write_csv(out / "scan_summary.csv", scan_rows)
    metadata = dict(status="complete", scans=list(SCANS), ray_count=len(ray_rows), views=12,
        source=_source_provenance(),
        methods=list(METHODS), fixed_parameters=dict(huber_h_mm=1.0, kernel_h_mm=list(MODE_BANDWIDTHS_MM),
            mean_shift_tol_mm=TOL_MM, mean_shift_max_iter=MAX_ITER, tie_mm=TIE_MM),
        support_definition="S(d)=sum_i p_i exp(-0.5*((z_i-d)/h)^2), p_i=original accepted w_i/sum(w); this is a weighted kernel support score, not a physical support probability. Mean-shift starts at each distinct accepted z; converged solutions are clustered as numerical solution basins (not physical layers or K); choose max score, ties by first accepted-order start. The second support peak is the highest-scoring converged cluster at least 2h from the best; it is NaN if absent.",
        robust_definition="Weighted Huber location with fixed 1 mm threshold, initialized at expected and iterated to 1e-7 mm or 64 iterations.",
        width_basis="No Gaussian covariance is used. Bandwidths are fixed in DTU millimeters before evaluation; no labels/GT select them.",
        coordinate_definition="Use original per-view focal, saved world_view_transform and pixel centers, then the verified shared influence-accuracy scale_mat transform. DTU GT is an independent scan point cloud; nearest-neighbor distance is one-way geometry, never pixel-depth truth.",
        input_sha256=input_meta, ray_root=str(ray_root.resolve()), influence_root=str(influence_root.resolve()),
        output=str(out.resolve()), elapsed_seconds=time.time()-started,
        dependence_note="Rays are paired within view; 3072 rays are not independent physical samples. Summaries are descriptive by view and scan; no pixel-level significance claims.")
    metadata["native_median_units"] = "Original saved host_target median in camera-z scene units, multiplied by the same verified scale_to_dtu_mm before comparison."
    metadata["depth_units"] = "All predicted depth, depth shift, weighted SD and support-mode coordinates in rays.csv are DTU mm; accepted input depth and host_target are camera-z scene units. NN errors are DTU mm."
    metadata["full_accuracy_map_crosschecks"] = crosschecks
    metadata["mechanism_thresholds"] = dict(main_bandwidth_mm=1.0, weak_expected_support_R_below=0.5,
        competing_peak_separation_at_least_mm=2.0,
        note="R is S(expected)/S(best-scoring converged cluster found by the bounded search); partial searches are not global maxima. The selected mode score is high by construction and is not correctness evidence.")
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def self_check():
    # Empty/singleton, equal single peak, separated two-peak, scale equivariance,
    # and exact contribution-preserving member-split invariance.
    empty = _fixed_subsets(np.array([]), np.array([]))
    assert all(np.isnan(x) for x in empty.values())
    one = _kernel_mode(np.array([2.0]), np.array([1.0]), 1.0)
    assert one["status"] == "converged" and abs(one["value"] - 2.0) < TOL_MM
    peak = _kernel_mode(np.array([0., .1, -.1]), np.array([.3, .4, .3]), 1.0)
    assert peak["status"] == "converged" and abs(peak["value"]) < .02
    z = np.array([0., .1, 5., 5.1])
    w = np.array([.4, .4, .1, .1])
    base = _kernel_mode(z, w, 1.0)
    assert base["solution_clusters"] >= 2 and np.isfinite(base["second_value"]) and 0 < base["relative_second"] < 1
    single_mode = _kernel_mode(np.array([0., .1, -.1]), np.array([.3, .4, .3]), 1.0)
    assert not np.isfinite(single_mode["second_value"])
    scaled = _kernel_mode(z * 3.0, w, 3.0)
    assert abs(scaled["value"] / 3.0 - base["value"]) < 5e-7
    split_z = np.repeat(z, 2)
    split_w = np.repeat(w / 2, 2)
    split = _kernel_mode(split_z, split_w, 1.0)
    np.testing.assert_allclose([split["value"], split["score"]], [base["value"], base["score"]],
                               rtol=0, atol=5e-7)
    base_s, _ = _support(z, w, base["value"], 1.0)
    split_s, _ = _support(split_z, split_w, split["value"], 1.0)
    assert abs(base_s - split_s) < 1e-14
    huber, state, _ = _huber_location(np.array([0., .1, 5., 5.1]), np.array([.4,.4,.1,.1]), .9)
    assert state == "converged" and np.isfinite(huber)
    assert np.isnan(_kernel_mode(np.array([0., np.nan]), np.array([.5,.5]), 1.)["value"])
    scale, native_scene = 324.6551818847656, 1.9148331
    native_vals, _ = _ray_estimates(np.array([1., 2.]) * scale, np.array([.4, .6]),
                                    np.array([1., native_scene, 1.]) * scale)
    np.testing.assert_allclose(native_vals["native_median"] / scale, native_scene, rtol=0, atol=1e-12)
    candidate = _peak_diagnostics(expected=5.0, best=0.0, second=5.0,
        expected_nn=3.0, best_nn=.5, second_nn=1.0, relative_support=.3, h_mm=1.0)
    assert candidate["two_position_scan_geometry_candidate"]
    assert candidate["relative_support_ge_0p25"]
    negative = _peak_diagnostics(expected=1.0, best=0.0, second=5.0,
        expected_nn=1.0, best_nn=.5, second_nn=.5, relative_support=.2, h_mm=1.0)
    assert not negative["two_position_scan_geometry_candidate"]
    assert not negative["relative_support_ge_0p25"]
    summary_row = dict(support_ratio_R_h1mm=1.1, nn_expected_mm=3., source_ray_status="finite")
    for h in MODE_BANDWIDTHS_MM:
        tag = f"{h:g}"
        summary_row.update({f"mode_status_h{tag}mm":"partial", f"two_position_scan_geometry_candidate_h{tag}mm":0,
            f"second_relative_support_{tag}mm":.3, f"second_relative_support_ge_0p25_h{tag}mm":1,
            f"second_finite_h{tag}mm":1, f"separated_2h_h{tag}mm":0, f"expected_between_h{tag}mm":0,
            f"nn_best_{tag}mm":1., f"nn_second_{tag}mm":2.})
    sample_summary = _summarize_peak_rows([summary_row], "synthetic", "one")
    assert len(sample_summary) == 3 and sample_summary[0]["R_gt_1_partial"] == 1
    assert sample_summary[0]["relative_support_finite"] == 1
    wz = np.array([0., 1., 4., 10.])
    ww = np.array([.4, .3, .2, .1])
    top = np.array([True, False, False, False])
    comp, mask = _peak_composition(wz, ww, 0., top)
    assert np.array_equal(mask, [True, True, False, False])
    assert abs(comp["window_q"] - .7) < 1e-14 and comp["window_n"] == 2
    assert abs(comp["window_neff"] - (.7**2 / (.4**2 + .3**2))) < 1e-14
    assert abs(comp["support_top10_s"] + comp["support_rest_s"] - comp["support_s"]) < 1e-14
    unknown, unknown_mask = _peak_composition(np.array([0., np.nan]), np.array([.5, .5]), 0., np.array([True, False]))
    assert unknown_mask is None and np.isnan(unknown["window_q"])
    g = _gate_one("A_all_mode1", 5., 2., 3.5, .3, .6, "converged", 4., 1.)
    assert g["decision"] == "mode" and g["switched"] == 1
    assert _gate_one("B_weak_converged", 5., 2., 3.5, .3, .6, "converged", 4., 1.)["decision"] == "mode"
    assert _gate_one("C_weak_converged_dominant", 5., 2., 3.5, .3, .6, "converged", 4., 1.)["decision"] == "expected"
    assert _gate_one("D_unknown_without_gate", 5., 2., 3.5, .3, .6, "converged", 4., 1.)["decision"] == "unknown"
    assert _gate_one("A_converged_only", 5., 2., 3.5, .3, .6, "partial", 4., 1.)["decision"] == "unknown"
    assert _gate_one("A_converged_only", 5., 2., 3.5, .3, .6, "converged", 4., 1.)["decision"] == "mode"
    assert _gate_one("C_weak_converged_dominant", 5., 2., np.nan, .3, np.nan, "converged", 4., 1.)["decision"] == "mode"
    assert _gate_one("B_weak_converged", 5., 2., np.nan, .3, np.nan, "partial", 4., 1.)["decision"] == "expected"
    assert _gate_one("D_unknown_without_gate", 5., 2., np.nan, np.nan, np.nan, "converged", 4., 1.)["decision"] == "unknown"
    print("CPU self-check passed: empty/single/one-peak/two-peak/scale/split/finite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run NumPy-only synthetic self-check")
    parser.add_argument("--ray-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--influence-root", type=Path)
    parser.add_argument("--accuracy-root", type=Path)
    parser.add_argument("--audit-peak-geometry", action="store_true",
                        help="Reproject saved v2 second/best peaks and audit their GT-nearest-neighbor geometry")
    parser.add_argument("--rays-csv", type=Path, help="Saved ray_support_v2/rays.csv")
    parser.add_argument("--geometry-csv", type=Path, help="Saved peak_geometry_v1_retry2/rays.csv")
    parser.add_argument("--audit-peak-composition", action="store_true",
                        help="Summarize raw accepted-ray mass and support contributions at saved 1mm peaks")
    parser.add_argument("--audit-support-gates", action="store_true",
                        help="Compare four fixed expected/mode gates using only saved v2 values")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    if args.audit_peak_geometry:
        if not all((args.ray_root, args.input_root, args.influence_root, args.rays_csv, args.output)):
            parser.error("peak-geometry audit requires --ray-root --input-root --influence-root --rays-csv --output")
        result = audit_peak_geometry(args)
        print(json.dumps(dict(status=result["status"], ray_count=result["ray_count"], views=result["views"],
                              elapsed_seconds=result["elapsed_seconds"], output=result["output"]), indent=2))
        return
    if args.audit_peak_composition:
        if not all((args.ray_root, args.input_root, args.influence_root, args.rays_csv,
                    args.geometry_csv, args.output)):
            parser.error("peak-composition audit requires --ray-root --input-root --influence-root --rays-csv --geometry-csv --output")
        result = audit_peak_composition(args)
        print(json.dumps(dict(status=result["status"], ray_count=result["ray_count"],
                              weak_R_lt_0p5_count=result["weak_R_lt_0p5_count"],
                              geometry_candidate_count=result["geometry_candidate_count"],
                              elapsed_seconds=result["elapsed_seconds"], output=result["output"]), indent=2))
        return
    if args.audit_support_gates:
        if not all((args.rays_csv, args.output)):
            parser.error("support-gate audit requires --rays-csv --output")
        result = audit_support_gates(args)
        print(json.dumps(dict(status=result["status"], ray_count=result["ray_count"],
                              weak_count=result["weak_count"], R_unknown_count=result["R_unknown_count"],
                              elapsed_seconds=result["elapsed_seconds"], output=result["output"]), indent=2))
        return
    if not all((args.ray_root, args.input_root, args.influence_root, args.accuracy_root, args.output)):
        parser.error("real-data mode requires --ray-root --input-root --influence-root --accuracy-root --output")
    result = run(args)
    print(json.dumps(dict(status=result["status"], ray_count=result["ray_count"], views=result["views"],
                          elapsed_seconds=result["elapsed_seconds"], output=result["output"]), indent=2))


if __name__ == "__main__":
    main()
