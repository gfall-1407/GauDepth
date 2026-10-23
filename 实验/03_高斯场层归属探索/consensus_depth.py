"""CPU probe of continuous contribution-weighted depth support on saved rays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
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


def _ray_estimates(z_mm: np.ndarray, w: np.ndarray, host_target: np.ndarray):
    if len(z_mm) == 0:
        vals = {name: float("nan") for name in METHODS}
        vals["native_median"] = float(host_target[1])
        return vals, dict(status="empty", members=0, weighted_sd_mm=float("nan"),
            huber_status="unknown", huber_iterations=0, mode={}, support={})
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Saved accepted-ray weights must be finite and positive")
    if not np.isfinite(z_mm).all():
        vals = {name: float("nan") for name in METHODS}
        vals["native_median"] = float(host_target[1])
        return vals, dict(status="nonfinite_depth", members=len(z_mm), weighted_sd_mm=float("nan"),
            huber_status="unknown", huber_iterations=0, mode={}, support={})
    fixed = _fixed_subsets(z_mm, w)
    expected_scene = fixed["expected"]
    if np.isfinite(host_target[0]):
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
    vals = dict(expected=mu, native_median=float(host_target[1]),
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
    np.testing.assert_allclose(c2w[3, :3], center, rtol=0, atol=2e-5)
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
        vals, detail = _ray_estimates(z_mm, w, targets[ray])
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
        methods=list(METHODS), fixed_parameters=dict(huber_h_mm=1.0, kernel_h_mm=list(MODE_BANDWIDTHS_MM),
            mean_shift_tol_mm=TOL_MM, mean_shift_max_iter=MAX_ITER, tie_mm=TIE_MM),
        support_definition="S(d)=sum_i p_i exp(-0.5*((z_i-d)/h)^2), p_i=original accepted w_i/sum(w); this is a weighted kernel support score, not a physical support probability. Mean-shift starts at each distinct accepted z; converged solutions are clustered as numerical solution basins (not physical layers or K); choose max score, ties by first accepted-order start. The second support peak is the highest-scoring converged cluster at least 2h from the best; it is NaN if absent.",
        robust_definition="Weighted Huber location with fixed 1 mm threshold, initialized at expected and iterated to 1e-7 mm or 64 iterations.",
        width_basis="No Gaussian covariance is used. Bandwidths are fixed in DTU millimeters before evaluation; no labels/GT select them.",
        coordinate_definition="Use original per-view focal, saved world_view_transform and pixel centers, then the verified shared influence-accuracy scale_mat transform. DTU GT is an independent scan point cloud; nearest-neighbor distance is one-way geometry, never pixel-depth truth.",
        input_sha256=input_meta, ray_root=str(ray_root.resolve()), influence_root=str(influence_root.resolve()),
        output=str(out.resolve()), elapsed_seconds=time.time()-started,
        dependence_note="Rays are paired within view; 3072 rays are not independent physical samples. Summaries are descriptive by view and scan; no pixel-level significance claims.")
    metadata["full_accuracy_map_crosschecks"] = crosschecks
    metadata["mechanism_thresholds"] = dict(main_bandwidth_mm=1.0, weak_expected_support_R_below=0.5,
        competing_peak_separation_at_least_mm=2.0,
        note="R is S(expected)/S(global mode); the mode score is maximized by construction, so S(mode) alone is not evidence of correctness.")
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
    print("CPU self-check passed: empty/single/one-peak/two-peak/scale/split/finite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run NumPy-only synthetic self-check")
    parser.add_argument("--ray-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--influence-root", type=Path)
    parser.add_argument("--accuracy-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    if not all((args.ray_root, args.input_root, args.influence_root, args.accuracy_root, args.output)):
        parser.error("real-data mode requires --ray-root --input-root --influence-root --accuracy-root --output")
    result = run(args)
    print(json.dumps(dict(status=result["status"], ray_count=result["ray_count"], views=result["views"],
                          elapsed_seconds=result["elapsed_seconds"], output=result["output"]), indent=2))


if __name__ == "__main__":
    main()
