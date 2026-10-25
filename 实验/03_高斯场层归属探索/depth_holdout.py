"""Fixed frozen-ray holdout comparisons; predictions never use scan GT."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

import consensus_depth as cd
import support_bound_audit as bound

METHODS = ("expected", "native_median", "weighted_median", "top10", "top50",
           "cover90", "huber_irls_1mm", "huber_1mm", "normal_weighted_median",
           "C_original", "bounded_mode_1mm")
REFERENCES = ("expected", "native_median", "weighted_median")
REMOTE = "git@github.com:gfall-1407/GauDepth.git"
STOP = False


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_stop(start, seconds):
    if STOP or time.monotonic() - start > seconds:
        raise InterruptedError("Signal or fixed batch time limit; completed views remain committed")


def huber_exact(z, w, h=1.):
    """Midpoint of the monotone derivative's zero interval; fixed 1mm Huber."""
    def derivative(d):
        return math.fsum(map(float, w * np.clip(d-z, -h, h)))
    left_lo = right_lo = float(z.min())
    left_hi = right_hi = float(z.max())
    for _ in range(80):
        mid = left_lo + (left_hi-left_lo)*.5
        if mid in (left_lo, left_hi):
            break
        if derivative(mid) >= 0:
            left_hi = mid
        else:
            left_lo = mid
    for _ in range(80):
        mid = right_lo + (right_hi-right_lo)*.5
        if mid in (right_lo, right_hi):
            break
        if derivative(mid) <= 0:
            right_lo = mid
        else:
            right_hi = mid
    left, right = left_hi, right_lo
    if right < left:  # A unique root can be bracketed by adjacent floats.
        left, right = min(left, right), max(left, right)
    return left + (right-left)*.5, right-left


def estimate(z, w, host, normals=None, normal_valid=None, ray_direction=None):
    """All arguments are camera-z mm except unmodified original weights."""
    vals = {m: float("nan") for m in METHODS}
    vals["native_median"] = float(host[1])
    info = dict(ray_status="empty", huber_status="unknown", coarse_status="unknown",
        coarse_starts=0, coarse_converged=0, coarse_mode_mm=float("nan"),
        coarse_second_mm=float("nan"), coarse_R=float("nan"),
        coarse_relative_second=float("nan"), C_decision="expected",
        bounded_status="unknown", bounded_reason="empty", bounded_seconds=0.,
        bounded_lower_mm=float("nan"), bounded_lower_score=float("nan"),
        bounded_upper_score=float("nan"), bounded_score_gap=float("nan"),
        bounded_intervals=0, bounded_fallback=1, peak_location_certified=0,
        geometry_fallback=1, geometry_invalid_members=0, geometry_q_mass=float("nan"),
        huber_minimizer_width_mm=float("nan"))
    if not len(z):
        return vals, info
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Accepted weights must be finite and strictly positive")
    if not np.isfinite(z).all():
        info.update(ray_status="nonfinite_depth", bounded_reason="nonfinite_depth")
        return vals, info
    fixed = cd._fixed_subsets(z, w)
    vals.update(fixed)
    vals["huber_irls_1mm"], hs, _ = cd._huber_location(z, w, vals["expected"], 1.)
    vals["huber_1mm"], info["huber_minimizer_width_mm"] = huber_exact(z, w)
    vals["normal_weighted_median"] = vals["weighted_median"]
    if normals is not None and normal_valid is not None and ray_direction is not None:
        lengths = np.linalg.norm(normals, axis=1)
        valid = np.asarray(normal_valid, dtype=bool) & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
        info["geometry_invalid_members"] = int((~valid).sum())
        if valid.all():
            unit = normals / lengths[:, None]
            direction = ray_direction / np.linalg.norm(ray_direction)
            q = w * (unit @ direction)**2
            info["geometry_q_mass"] = float(q.sum(dtype=np.float64))
            if np.isfinite(q).all() and q.sum(dtype=np.float64) > 0:
                vals["normal_weighted_median"] = cd._weighted_median(z, q)
                info["geometry_fallback"] = 0
    # The original C rule uses precisely the original 64-step search/status.
    coarse = cd._kernel_mode(z, w, 1.)
    s_expected = cd._support(z, w, vals["expected"], 1.)[0]
    ratio = s_expected / coarse["score"] if coarse["score"] > 0 else float("nan")
    gate = cd._gate_one("C_weak_converged_dominant", vals["expected"],
        coarse["value"], coarse["second_value"], ratio,
        coarse["relative_second"], coarse["status"], float("nan"), float("nan"))
    vals["C_original"] = gate["depth"]
    p = w / w.sum(dtype=np.float64)
    seeds = {k: float(v) for k, v in vals.items() if math.isfinite(v)}
    seeds["coarse_mode"] = coarse["value"]
    # Individual member depths are cheap valid lower-bound candidates. No GT.
    scores = np.array([bound.support(z, p, float(d), 1.) for d in z])
    seeds["best_member"] = float(z[int(np.argmax(scores))])
    result = bound.search(z, p, 1., seeds)
    certified = result["status"] == "certified_score"
    vals["bounded_mode_1mm"] = result["lower_depth_mm"] if certified else vals["expected"]
    info.update(ray_status="finite", huber_status=hs,
        coarse_status=coarse["status"], coarse_starts=coarse["starts"],
        coarse_converged=coarse["converged"], coarse_mode_mm=coarse["value"],
        coarse_second_mm=coarse["second_value"], coarse_R=ratio,
        coarse_relative_second=coarse["relative_second"], C_decision=gate["decision"],
        bounded_status=result["status"], bounded_reason=result["reason"],
        bounded_seconds=result["elapsed_seconds"], bounded_lower_mm=result["lower_depth_mm"],
        bounded_lower_score=result["lower_score"], bounded_upper_score=result["upper_score"],
        bounded_score_gap=result["certified_score_gap"],
        bounded_intervals=result["interval_count"], bounded_fallback=int(not certified))
    return vals, info


def load_scan(scan, args):
    root = args.input_root / f"scan{scan}"
    files = dict(model=root / "point_cloud.ply", cameras=root / "cameras.npz",
                 gt=root / f"stl{scan:03d}_total.ply")
    impath = args.influence_root / f"scan{scan}" / "metadata.json"
    imeta = json.loads(impath.read_text(encoding="utf-8"))
    if imeta.get("status") != "complete" or imeta.get("model_group") != "G1_MV":
        raise ValueError(f"Expected complete G1_MV metadata: {impath}")
    keys = dict(model="model_sha256", cameras="cameras_npz_sha256", gt="gt_sha256")
    digests = {key: cd.sha256(path) for key, path in files.items()}
    for key, digest in digests.items():
        if digest != imeta.get(keys[key]):
            raise ValueError(f"{key} SHA mismatch for scan{scan}")
    alignment = np.array(imeta["alignment"]["transform"], dtype=np.float64)
    scale = float(imeta["alignment"]["scale_to_dtu_mm"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid DTU scale")
    with np.load(files["cameras"]) as cams:
        np.testing.assert_allclose(cams["scale_mat_0"], alignment, rtol=0, atol=1e-6)
    views, lineage = [], []
    for directory in (args.ray_root / f"scan{scan}", args.ray_root / f"scan{scan}_more"):
        mp = directory / "metadata.json"
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("model_sha256") != digests["model"] or meta.get("git_remote") != REMOTE:
            raise ValueError(f"Ray model/origin mismatch: {mp}")
        if meta.get("git_status"):
            raise ValueError(f"Ray sample recorded dirty source: {mp}")
        lineage.append(dict(path=str(mp.resolve()), sha256=cd.sha256(mp),
            computed_commit=meta["git_commit"], cfg_sha256=meta["cfg_sha256"],
            host_binary_sha256=meta["host_binary_sha256"], reader_binary_sha256=meta["reader_binary_sha256"]))
        for vm in meta["views"]:
            if vm["random_samples"] != args.samples_per_view or vm["qualitative_samples"] != 0:
                raise ValueError(f"Expected only {args.samples_per_view} random rays: {mp}")
            if vm["seed"] != args.expected_seed:
                raise ValueError(f"Unexpected sampling seed: {mp}")
            if "GT mask" not in vm["sampling"] or not vm.get("mask_source"):
                raise ValueError("Opacity-selected rays cannot substitute for fixed DTU mask")
            rp = directory / vm["file"]
            if cd.sha256(rp) != vm["npz_sha256"]:
                raise ValueError(f"Raw NPZ SHA mismatch: {rp}")
            name = vm["image_name"]
            check = next((v for v in imeta["alignment"]["camera_checks"] if v["name"] == name), None)
            if check is None or check["rotation_max_abs"] >= 1e-4 or check["translation_max_abs_scene_units"] >= 1e-4:
                raise ValueError(f"Camera outside verified common inventory: scan{scan}/{name}")
            with np.load(files["cameras"]) as cams:
                np.testing.assert_allclose(cams[f"scale_mat_{int(name)}"], alignment, rtol=0, atol=1e-6)
            views.append((meta, vm, rp))
    if len(views) != args.views_per_scan or len({v[1]["image_name"] for v in views}) != len(views):
        raise ValueError(f"Expected exactly {args.views_per_scan} distinct views, got {len(views)} for scan{scan}")
    return dict(**files, inf_meta=imeta, inf_meta_path=impath, alignment=alignment, scale=scale,
        views=views, lineage=lineage, input_sha256=digests)


def process_view(scan, old, vm, rayfile, data, tree, args, started):
    with np.load(rayfile) as raw:
        offsets = raw["offsets"].astype(np.int64)
        z = raw["depth"].astype(np.float64)
        w = raw["weight"].astype(np.float64)
        xy = raw["pixel_xy"].astype(np.int32)
        target = raw["host_target"].astype(np.float64)
        replay = raw["replay"].astype(np.float64)
        image_size = tuple(map(int, raw["image_size"]))
        qualitative = raw["qualitative"].astype(bool)
        gids = raw["gaussian_id"].astype(np.int64)
        normals = raw["normal"].astype(np.float64)
        normal_valid = raw["normal_valid"].astype(bool)
    if image_size != tuple(vm["image_size"]) or len(xy) != args.samples_per_view or target.shape != (len(xy), 3):
        raise ValueError("Ray dimensions differ from sampling metadata")
    if len(offsets) != len(xy)+1 or offsets[0] != 0 or offsets[-1] != len(z) or len(z) != len(w) or len(gids) != len(z) or normals.shape != (len(z), 3) or normal_valid.shape != (len(z),) or np.any(np.diff(offsets) < 0):
        raise ValueError("Invalid complete-ray offsets/record sizes")
    if qualitative.any() or len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("Qualitative or duplicate pixels in statistical sample")
    if np.any(xy < 0) or np.any(xy[:, 0] >= image_size[1]) or np.any(xy[:, 1] >= image_size[0]):
        raise ValueError("Pixel outside image")
    tol = old["replay_tolerance"]
    np.testing.assert_allclose(replay, target, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True)
    camera_check = cd._check_camera_projection(data["cameras"], vm, data["alignment"])
    w2c = np.asarray(vm["world_view_transform"], dtype=np.float64).T
    np.testing.assert_allclose(np.linalg.inv(w2c)[:3, 3], vm["camera_center"], rtol=0, atol=2e-5)
    count = min(args.max_rays_per_view or len(xy), len(xy))
    xy, target = xy[:count], target[:count]
    offsets = offsets[:count+1]
    z, w = z[:offsets[-1]], w[:offsets[-1]]
    rows, predictions = [], []
    for ray, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        check_stop(started, args.max_seconds)
        fx, fy = map(float, vm["focal"])
        # This is the renderer plane-incidence ray, whose convention uses W/2,H/2.
        # Evaluation points retain the existing verified (W-1)/2,(H-1)/2 convention.
        direction = np.array([(xy[ray, 0]-image_size[1]/2.)/fx,
                              (xy[ray, 1]-image_size[0]/2.)/fy, 1.])
        vals, details = estimate(z[a:b] * data["scale"], w[a:b], target[ray] * data["scale"],
            normals[a:b], normal_valid[a:b], direction)
        expected_scene = vals["expected"] / data["scale"]
        if math.isfinite(expected_scene):
            np.testing.assert_allclose(expected_scene, target[ray, 0], rtol=tol["rtol"], atol=tol["atol"])
        elif b > a and np.isfinite(z[a:b]).all():
            raise ValueError("Finite nonempty ray unexpectedly has no expected depth")
        for method in METHODS:
            depth = vals[method]
            point = cd._points_mm(xy[ray:ray+1], np.array([depth / data["scale"]]), vm, data["alignment"]) if math.isfinite(depth) and depth > 0 else np.empty((0, 3))
            vals[f"nn_{method}_mm"] = float(cd._query_exact(point, tree)[0]) if len(point) else float("nan")
        rows.append(dict(scan=scan, view=vm["image_name"], ray=ray, x=int(xy[ray, 0]), y=int(xy[ray, 1]),
            accepted=int(b-a), contribution_mass=float(w[a:b].sum()),
            sample_commit=old["git_commit"], raw_npz_sha256=vm["npz_sha256"], **details, **vals))
        predictions.append(vals)
    qa = cd._crosscheck_accuracy_map(scan, vm["image_name"], xy, image_size, offsets, z, w,
                                    predictions, args.accuracy_root, data)
    qa.update(scan=scan, view=vm["image_name"], rotation_max_abs=camera_check[0], translation_max_abs=camera_check[1])
    return rows, qa


def summaries(rows, scope, scan="all", view="all"):
    result = []
    errors = {m: np.array([r[f"nn_{m}_mm"] for r in rows], dtype=np.float64) for m in METHODS}
    common = np.logical_and.reduce([np.isfinite(v) for v in errors.values()])
    for method in METHODS:
        for reference in REFERENCES:
            a, b = errors[reference], errors[method]
            mask = np.isfinite(a) & np.isfinite(b)
            delta = b[mask] - a[mask]
            switched = np.array([math.isfinite(r[method]) and math.isfinite(r[reference]) and abs(r[method]-r[reference]) > cd.TIE_MM for r in rows])[mask]
            def mean(v):
                return float(np.mean(v)) if len(v) else None
            result.append(dict(scope=scope, scan=scan, view=view, method=method, reference=reference,
                total_n=len(rows), candidate_finite_n=int(np.isfinite(b).sum()), reference_finite_n=int(np.isfinite(a).sum()),
                paired_n=int(mask.sum()), candidate_unknown_n=int((~np.isfinite(b)).sum()),
                bounded_solver_unknown_n=sum(r["bounded_status"] != "certified_score" for r in rows),
                geometry_fallback_n=sum(r["geometry_fallback"] for r in rows),
                reference_mean_nn_mm=mean(a[mask]), candidate_mean_nn_mm=mean(b[mask]),
                delta_mean_nn_mm=mean(delta), improved_n=int((delta < -1e-12).sum()), worsened_n=int((delta > 1e-12).sum()),
                worsened_ge1mm_n=int((delta >= 1.).sum()), worsened_ge5mm_n=int((delta >= 5.).sum()),
                delta_p95_mm=float(np.quantile(delta, .95)) if len(delta) else None,
                delta_p99_mm=float(np.quantile(delta, .99)) if len(delta) else None,
                delta_max_mm=float(np.max(delta)) if len(delta) else None,
                switched_n=int(switched.sum()), switched_improved_n=int((delta[switched] < -1e-12).sum()),
                switched_worsened_n=int((delta[switched] > 1e-12).sum()),
                common_all_methods_n=int(common.sum()), common_reference_mean_nn_mm=mean(a[common]),
                common_candidate_mean_nn_mm=mean(b[common]), common_delta_mean_nn_mm=mean(b[common]-a[common])))
    return result


def scan_equal(scan_rows):
    out = []
    # Only means are averaged across scans; quantiles and counts are never averaged.
    for method in METHODS:
        for reference in REFERENCES:
            subset = [r for r in scan_rows if r["method"] == method and r["reference"] == reference]
            item = dict(subset[0], scope="scan_equal", scan="all", view="all")
            for key in item:
                if key.endswith("_n"):
                    item[key] = sum(r[key] for r in subset)
                elif key.endswith("_mm"):
                    if "mean" in key and all(r[key] is not None for r in subset):
                        item[key] = float(np.mean([r[key] for r in subset]))
                    else:
                        item[key] = None
            out.append(item)
    return out


def run(args):
    if not args.output.is_absolute() or (args.output.exists() and any(args.output.iterdir())):
        raise ValueError("Output must be an absolute empty/new independent directory")
    if len(args.scans) != len(set(args.scans)):
        raise ValueError("Duplicate scans")
    args.output.mkdir(parents=True, exist_ok=True)
    source = cd._source_provenance()
    if not source["clean"] or source["remote"] != REMOTE:
        raise ValueError("Execute only committed clean source with the approved origin")
    bound.MAX_SECONDS_PER_RAY = args.bound_seconds
    metadata = dict(status="partial", args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source=source, methods=list(METHODS), references=list(REFERENCES), completed_views=[], inputs=[],
        sample_role=args.sample_role,
        parameters=dict(h_mm=1., support_gap=bound.TOL_SCORE, bound_seconds=args.bound_seconds,
            max_intervals=bound.MAX_INTERVALS, original_C_mean_shift_iterations=cd.MAX_ITER,
            original_C_mean_shift_tol_mm=cd.TOL_MM, peak_location_certified=False),
        huber_definition="huber_1mm: monotone derivative sum(w*clip(d-z,-1,1)) zero interval via two 80-step bisections, choose interval midpoint. huber_irls_1mm retains old 64-step IRLS, nonconverged is unknown; the objective is unchanged, this is a numerical repair, not a new estimator.",
        geometry_definition="q=w*(unit_camera_normal dot unit_ray)^2; weighted median under q. Ray uses [(x-W/2)/fx,(y-H/2)/fy,1], matching renderer incidence convention. Any invalid accepted normal or nonpositive total q falls back to original weighted median for the entire ray. Equal normal-displacement noise gives a possible inverse-depth-variance motivation, an assumption rather than evidence of physical correctness. No sign-based filtering or GT selection.",
        interpretation="Predictions use frozen accepted camera-z depths, original Talpha and saved normals only, never GT. S is a fixed 1mm Gaussian-kernel support score, not physical probability or Gaussian covariance. Bounded score certification does not certify a unique peak location. NN is exact one-way DTU scan-point nearest-neighbor distance, not pixel GT depth MAE or Chamfer. Every comparison row uses its own explicitly paired finite denominator; common-all-method means are also included. Scan-equal rows average only per-scan means, never quantiles. Unknown numerical solves fall back to expected in bounded_mode_1mm, and remain separately counted. The original C preserves the original 64-step coarse convergence semantics.")
    atomic_json(args.output / "metadata.json", metadata)
    start = time.monotonic()
    all_rows, all_summaries = [], []
    try:
        for scan in args.scans:
            data = load_scan(scan, args)
            metadata["inputs"].append(dict(scan=scan, input_sha256=data["input_sha256"],
                influence_metadata_sha256=cd.sha256(data["inf_meta_path"]), ray_metadata=data["lineage"],
                scale_to_dtu_mm=data["scale"]))
            tree = cd._make_gt_tree(cd._load_gt(data["gt"]))
            scan_rows = []
            for old, vm, path in data["views"]:
                rows, qa = process_view(scan, old, vm, path, data, tree, args, start)
                viewfile = args.output / f"scan{scan}_view{vm['image_name']}_rays.csv"
                write_csv(viewfile, rows)
                all_rows.extend(rows)
                scan_rows.extend(rows)
                all_summaries.extend(summaries(rows, "view", scan, vm["image_name"]))
                metadata["completed_views"].append(dict(scan=scan, view=vm["image_name"], rays=len(rows),
                    file=viewfile.name, sha256=cd.sha256(viewfile), full_accuracy_crosscheck=qa))
                metadata["elapsed_seconds"] = time.monotonic() - start
                write_csv(args.output / "summary.csv", all_summaries)
                atomic_json(args.output / "metadata.json", metadata)
                print(json.dumps(dict(scan=scan, view=vm["image_name"], rays=len(rows), elapsed_seconds=metadata["elapsed_seconds"])), flush=True)
            all_summaries.extend(summaries(scan_rows, "scan", scan))
        all_summaries.extend(summaries(all_rows, "pooled"))
        all_summaries.extend(scan_equal([r for r in all_summaries if r["scope"] == "scan"]))
        write_csv(args.output / "summary.csv", all_summaries)
        metadata.update(status="complete", ray_count=len(all_rows), elapsed_seconds=time.monotonic()-start,
            summary_sha256=cd.sha256(args.output / "summary.csv"))
        atomic_json(args.output / "metadata.json", metadata)
    except BaseException as exc:
        metadata.update(status="partial", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-start)
        atomic_json(args.output / "metadata.json", metadata)
        raise


def self_check():
    bound.MAX_SECONDS_PER_RAY = 2.
    for z, w in ((np.array([2.]), np.array([1.])),
                 (np.array([0., .1, 10., 10.1]), np.array([.4, .4, .1, .1]))):
        fixed = cd._fixed_subsets(z, w)
        vals, info = estimate(z, w, np.array([fixed["expected"], 2., 1.]))
        assert info["bounded_status"] == "certified_score" and info["bounded_score_gap"] <= 1e-6
        assert not info["peak_location_certified"]
        assert vals["expected"] == fixed["expected"]
        assert abs(vals["native_median"] - 2.) < 1e-12
    vals, info = estimate(np.array([]), np.array([]), np.array([0., 0., 0.]))
    assert info["bounded_status"] == "unknown" and math.isnan(vals["bounded_mode_1mm"])
    rows = []
    for i in range(3):
        row = {m: float(i) for m in METHODS}
        row.update({f"nn_{m}_mm": float(i+1) for m in METHODS})
        row["bounded_status"] = "certified_score"
        row["geometry_fallback"] = 0
        rows.append(row)
    rows[0]["nn_huber_1mm_mm"] = float("nan")
    sr = summaries(rows, "test")
    hs = next(r for r in sr if r["method"] == "huber_1mm" and r["reference"] == "expected")
    assert hs["paired_n"] == 2 and hs["candidate_unknown_n"] == 1 and hs["reference_mean_nn_mm"] == 2.5
    assert all(r["common_all_methods_n"] == 2 for r in sr)
    h, width = huber_exact(np.array([0., 10.]), np.array([.5, .5]))
    np.testing.assert_allclose([h, width], [5., 8.], rtol=0, atol=1e-12)
    # Incidence weighting and its required sign invariance/fallback.
    z, w = np.array([0., 10.]), np.array([.4, .6])
    ns = np.array([[0., 0., 1.], [1., 0., 0.]])
    a, ai = estimate(z, w, np.array([6., 3., 1.]), ns, np.array([True, True]), np.array([0., 0., 1.]))
    b, bi = estimate(z, w, np.array([6., 3., 1.]), -ns, np.array([True, True]), np.array([0., 0., 1.]))
    assert a["normal_weighted_median"] == b["normal_weighted_median"] == 0. and ai["geometry_fallback"] == 0
    c, ci = estimate(z, w, np.array([6., 3., 1.]), ns, np.array([False, True]), np.array([0., 0., 1.]))
    assert c["normal_weighted_median"] == 10. and ci["geometry_fallback"] == 1
    print("Fixed estimates, score bounds, empty rays and paired finite denominators: PASS", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--ray-root", type=Path)
    p.add_argument("--input-root", type=Path)
    p.add_argument("--influence-root", type=Path)
    p.add_argument("--accuracy-root", type=Path)
    p.add_argument("--scans", type=int, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--samples-per-view", type=int, default=512)
    p.add_argument("--views-per-scan", type=int, default=3)
    p.add_argument("--expected-seed", type=int, default=20261026)
    p.add_argument("--sample-role", choices=("exploration", "holdout", "smoke"), default="exploration")
    p.add_argument("--max-rays-per-view", type=int, default=0, help="Debug prefix only; 0 uses the entire fixed random sample")
    p.add_argument("--max-seconds", type=float, default=1800.)
    p.add_argument("--bound-seconds", type=float, default=2.)
    args = p.parse_args()
    if args.check:
        self_check()
        return
    for key in ("ray_root", "input_root", "influence_root", "accuracy_root", "scans", "output"):
        if not getattr(args, key):
            p.error(f"--{key.replace('_', '-')} is required")
    if min(args.samples_per_view, args.views_per_scan, args.bound_seconds, args.max_seconds) <= 0 or args.max_rays_per_view < 0:
        p.error("Counts/time limits must be positive; debug ray limit may be zero")
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(args)


if __name__ == "__main__":
    main()
