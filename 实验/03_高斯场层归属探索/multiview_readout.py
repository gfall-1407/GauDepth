"""Fixed GT-free multiview candidate selection, then separately mapped NN evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

import consensus_depth as cd
from depth_holdout import atomic_json, write_csv

POOL = ("huber_1mm", "weighted_median", "expected", "bounded_mode_1mm")
REFERENCES = ("huber_1mm", "expected", "native_median", "weighted_median")
TIE_MM = 1e-9
MIN_COMMON = 3
STOP = False


def check_time(start, limit):
    if STOP or time.monotonic() - start > limit:
        raise InterruptedError("Fixed batch deadline or stop signal; completed views remain saved")


def world_points(pixel, depths_scene, camera):
    """Camera convention is exactly Camera.Cx=(W-1)/2, Cy=(H-1)/2."""
    h, w = camera["image_size"]
    fx, fy = camera["focal"]
    directions = np.array([(pixel[0]-(w-1)/2.)/fx, (pixel[1]-(h-1)/2.)/fy, 1.])
    local = np.asarray(depths_scene)[:, None] * directions
    c2w = camera["c2w"]
    return local @ c2w[:3, :3].T + c2w[:3, 3]


def project(points, camera):
    local = points @ camera["w2c"][:3, :3].T + camera["w2c"][:3, 3]
    z = local[:, 2]
    h, w = camera["image_size"]
    fx, fy = camera["focal"]
    uv = np.full((len(points), 2), np.nan)
    good = np.isfinite(local).all(axis=1) & (z > 0)
    uv[good, 0] = local[good, 0]/z[good]*fx + (w-1)/2.
    uv[good, 1] = local[good, 1]/z[good]*fy + (h-1)/2.
    return uv, z


def bilinear(depth_map, uv):
    """All four integer pixels must have finite positive expected depths."""
    height, width = depth_map.shape
    out = np.full(len(uv), np.nan)
    finite = np.isfinite(uv).all(axis=1)
    for index in np.flatnonzero(finite):
        x, y = map(float, uv[index])
        x0, y0 = math.floor(x), math.floor(y)
        if x0 < 0 or y0 < 0 or x0+1 >= width or y0+1 >= height:
            continue
        values = depth_map[y0:y0+2, x0:x0+2]
        if not np.isfinite(values).all() or np.any(values <= 0):
            continue
        dx, dy = x-x0, y-y0
        out[index] = ((1-dy)*((1-dx)*values[0, 0]+dx*values[0, 1]) +
                     dy*((1-dx)*values[1, 0]+dx*values[1, 1]))
    return out


def select_candidate(pixel, depths_mm, source_camera, neighbors, scale_to_mm):
    """No GT/NN inputs. All candidates are compared on identical neighbors."""
    depths = np.asarray(depths_mm, dtype=np.float64)
    if depths.shape != (len(POOL),) or not math.isfinite(scale_to_mm) or scale_to_mm <= 0:
        raise ValueError("Unexpected fixed candidate pool or invalid scale")
    points = world_points(pixel, depths/scale_to_mm, source_camera)
    errors, names = [], []
    for name, camera, depth_map in neighbors:
        uv, z = project(points, camera)
        expected = bilinear(depth_map, uv)
        valid = np.isfinite(depths) & (depths > 0) & np.isfinite(z) & (z > 0) & np.isfinite(expected)
        if valid.all():
            errors.append(np.abs(z-expected)*scale_to_mm)
            names.append(name)
    scores = np.median(np.asarray(errors), axis=0) if len(errors) >= MIN_COMMON else np.full(len(POOL), np.nan)
    selected = 0
    fallback = len(errors) < MIN_COMMON
    if not fallback:
        minimum = float(np.min(scores))
        selected = next(i for i, score in enumerate(scores) if score <= minimum+TIE_MM)
    return dict(selected_method=POOL[selected], selected_depth_mm=float(depths[selected]),
        fallback=int(fallback), common_neighbors=len(errors), common_neighbor_names=";".join(names),
        **{f"score_{method}_mm": float(scores[i]) for i, method in enumerate(POOL)})


def checked_input_manifest(input_root, scan):
    path = input_root.parent / "input_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("instance_id") != "d6a7459179-693ab6ee":
        raise ValueError("Expected the explicitly verified G1 input manifest")
    needed = ("training_cameras.json", "cameras.npz", "cfg_args", "point_cloud.ply")
    digests = {}
    for name in needed:
        relative = f"scan{scan}/{name}"
        source = [item for item in manifest["files"] if item["relative"] == relative]
        if len(source) != 1:
            raise ValueError(f"Missing or duplicate source SHA: {relative}")
        local = input_root / relative
        digest = cd.sha256(local)
        if digest != source[0]["sha256"] or local.stat().st_size != source[0]["bytes"]:
            raise ValueError(f"Source/destination SHA or size mismatch: {local}")
        digests[name] = digest
    return digests, dict(path=str(path.resolve()), sha256=cd.sha256(path))


def load_scan(scan, args, neighbor_manifest):
    root = args.input_root / f"scan{scan}"
    sample_path = args.sample_root / f"scan{scan}" / "metadata.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if sample.get("git_status") or sample.get("git_remote") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Sample source must be committed and use approved origin")
    digests, manifest_record = checked_input_manifest(args.input_root, scan)
    if sample["model_sha256"] != digests["point_cloud.ply"] or sample["cfg_sha256"] != digests["cfg_args"]:
        raise ValueError("Sample model or cfg differs from checked inputs")
    accuracy_path = args.accuracy_root / f"scan{scan}" / "metadata.json"
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    if accuracy.get("status") != "complete" or accuracy.get("model_group") != "G1_MV":
        raise ValueError("Expected complete G1_MV full-mask expected maps")
    for key, name in (("model_sha256", "point_cloud.ply"), ("cfg_sha256", "cfg_args"), ("cameras_npz_sha256", "cameras.npz")):
        if accuracy.get(key) != digests[name]:
            raise ValueError(f"Accuracy {key} differs from sampled model/cameras")
    scale = float(accuracy["alignment"]["scale_to_dtu_mm"])
    alignment = np.array(accuracy["alignment"]["transform"], dtype=np.float64)
    if sample["effective_model_args"]["resolution"] != 2:
        raise ValueError("This fixed experiment requires the recorded resolution=2")
    sizes = {tuple(v["image_size"]) for v in sample["views"]}
    if sizes != {(581, 777)}:
        raise ValueError(f"Unexpected saved sample dimensions: {sizes}")
    h, w = next(iter(sizes))
    cameras, camera_checks = {}, []
    training = json.loads((root / "training_cameras.json").read_text(encoding="utf-8"))
    with np.load(root / "cameras.npz") as archive:
        for row in training:
            name = row["img_name"]
            if name in cameras or row["width"] != 2*w or row["height"] != 2*h:
                raise ValueError("Duplicate camera or training dimensions do not match resolution=2")
            c2w = np.eye(4)
            c2w[:3, :3] = np.array(row["rotation"], dtype=np.float64)
            c2w[:3, 3] = np.array(row["position"], dtype=np.float64)
            np.testing.assert_allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            if not np.isclose(np.linalg.det(c2w[:3, :3]), 1., rtol=0, atol=1e-6):
                raise ValueError("Improper camera rotation")
            w2c = np.linalg.inv(c2w)
            focal = (float(row["fx"])/2., float(row["fy"])/2.)
            if not np.isfinite(focal).all() or min(focal) <= 0:
                raise ValueError("Invalid training focal")
            vm = dict(image_name=name, world_view_transform=w2c.T.tolist(), image_size=[h, w], focal=focal)
            re, te = cd._check_camera_projection(root / "cameras.npz", vm, alignment)
            np.testing.assert_allclose(archive[f"scale_mat_{int(name)}"], alignment, rtol=0, atol=1e-6)
            cameras[name] = dict(image_size=(h, w), focal=focal, c2w=c2w, w2c=w2c)
            camera_checks.append(dict(name=name, archive_R_max_abs=re, archive_t_max_abs_scene=te))
    if set(cameras) != set(accuracy["camera_names"]):
        raise ValueError("Training camera names differ from verified full-mask inventory")
    for vm in sample["views"]:
        camera = cameras[vm["image_name"]]
        np.testing.assert_allclose(camera["w2c"], np.array(vm["world_view_transform"]).T, rtol=0, atol=2e-5)
        np.testing.assert_allclose(camera["c2w"][:3, 3], vm["camera_center"], rtol=0, atol=2e-5)
        np.testing.assert_allclose(camera["focal"], vm["focal"], rtol=0, atol=1e-3)
    neighbor_path = args.sample_root / f"scan{scan}" / "multi_view.json"
    expected_sha = neighbor_manifest.get(str(scan))
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or cd.sha256(neighbor_path) != expected_sha:
        raise ValueError(f"Missing or mismatched independently fetched multi_view source SHA for scan{scan}")
    neighbors = {}
    with neighbor_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            name, nearest = row["ref_name"], row["nearest_name"]
            if name in neighbors or name not in cameras or len(nearest) != 8 or len(set(nearest)) != 8 or name in nearest or not set(nearest) <= set(cameras):
                raise ValueError("Expected original eight unique non-self training neighbors")
            neighbors[name] = nearest
    if set(neighbors) != set(cameras):
        raise ValueError("Neighbor inventory differs from original training cameras")
    return dict(scale=scale, sample=sample, cameras=cameras, neighbors=neighbors, accuracy=accuracy,
        provenance=dict(scan=scan, input_sha256=digests, input_manifest=manifest_record,
            sample_metadata_sha256=cd.sha256(sample_path), sample_commit=sample["git_commit"],
            accuracy_metadata_sha256=cd.sha256(accuracy_path), neighbor_sha256=expected_sha,
            camera_checks=camera_checks, principal_point="Camera.Cx=(W-1)/2,Cy=(H-1)/2; archive principal point is not substituted"))


def load_map(scan, name, data, args):
    records = [v for v in data["accuracy"]["views"] if v["name"] == name]
    if len(records) != 1:
        raise ValueError("Missing/duplicate original accuracy view")
    vm = records[0]
    path = args.accuracy_root / f"scan{scan}" / vm["file"]
    if cd.sha256(path) != vm["sha256"]:
        raise ValueError(f"Accuracy-map SHA mismatch: {path}")
    with np.load(path) as raw:
        image_size = tuple(map(int, raw["image_size"]))
        pixel = raw["pixel_index"].astype(np.int64)
        depth = raw["weighted_mean_depth"][:, 0].astype(np.float64)
    if image_size != data["cameras"][name]["image_size"] or len(pixel) != len(depth) or np.any(np.diff(pixel) <= 0):
        raise ValueError("Invalid expected-map dimensions/strict sparse pixel inventory")
    if len(pixel) and (pixel[0] < 0 or pixel[-1] >= math.prod(image_size)):
        raise ValueError("Expected-map pixel index outside image")
    dense = np.full(image_size, np.nan, dtype=np.float64)
    dense.ravel()[pixel] = depth
    return dense, dict(scan=scan, view=name, path=str(path.resolve()), sha256=vm["sha256"])


def prediction_rows(path, expected_sha):
    """Strip evaluation columns at load; downstream decision functions never see NN."""
    if cd.sha256(path) != expected_sha:
        raise ValueError(f"Estimate CSV SHA mismatch: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            yield dict(scan=int(row["scan"]), view=row["view"], ray=int(row["ray"]), x=int(row["x"]), y=int(row["y"]),
                raw_npz_sha256=row["raw_npz_sha256"], **{method: float(row[method]) for method in POOL})


def evaluate(decisions, estimate_path, estimate_sha):
    """Called only after every view's decisions have been flushed and hashed."""
    if cd.sha256(estimate_path) != estimate_sha:
        raise ValueError("Source estimates changed after decisions")
    with estimate_path.open(encoding="utf-8-sig", newline="") as f:
        source = {int(r["ray"]): r for r in csv.DictReader(f)}
    output = []
    for decision in decisions:
        row = source[decision["ray"]]
        if (int(row["scan"]), row["view"], int(row["x"]), int(row["y"])) != (decision["scan"], decision["view"], decision["x"], decision["y"]):
            raise ValueError("Decision/evaluation identity mismatch")
        selected = decision["selected_method"]
        if abs(float(row[selected])-decision["selected_depth_mm"]) > 1e-9:
            raise ValueError("Selected depth was not copied from the fixed candidate")
        output.append(dict(scan=decision["scan"], view=decision["view"], ray=decision["ray"],
            selected_method=selected, selected_nn_mm=float(row[f"nn_{selected}_mm"]),
            fallback=decision["fallback"], **{f"nn_{method}_mm": float(row[f"nn_{method}_mm"]) for method in REFERENCES}))
    return output


def summarize(rows, scope, scan="all", view="all"):
    summaries = []
    for reference in REFERENCES:
        a = np.array([r[f"nn_{reference}_mm"] for r in rows])
        b = np.array([r["selected_nn_mm"] for r in rows])
        mask = np.isfinite(a) & np.isfinite(b)
        delta = b[mask]-a[mask]
        def avg(values):
            return float(np.mean(values)) if len(values) else None
        def percentile(values, q):
            return float(np.quantile(values, q)) if len(values) else None
        summaries.append(dict(scope=scope, scan=scan, view=view, method="multiview_fixed_pool", reference=reference,
            total_n=len(rows), paired_n=int(mask.sum()), candidate_unknown_n=int((~np.isfinite(b)).sum()),
            reference_unknown_n=int((~np.isfinite(a)).sum()), fallback_n=sum(r["fallback"] for r in rows),
            reference_mean_nn_mm=avg(a[mask]), candidate_mean_nn_mm=avg(b[mask]), delta_mean_nn_mm=avg(delta),
            reference_p95_nn_mm=percentile(a[mask], .95), reference_p99_nn_mm=percentile(a[mask], .99),
            candidate_p95_nn_mm=percentile(b[mask], .95), candidate_p99_nn_mm=percentile(b[mask], .99),
            reference_ge20mm_n=int((a[mask] >= 20).sum()), candidate_ge20mm_n=int((b[mask] >= 20).sum()),
            improved_n=int((delta < -1e-12).sum()), worsened_n=int((delta > 1e-12).sum()),
            worsened_ge1mm_n=int((delta >= 1).sum()), worsened_ge5mm_n=int((delta >= 5).sum()),
            delta_p95_mm=percentile(delta, .95), delta_p99_mm=percentile(delta, .99)))
    return summaries


def run(args):
    if not args.output.is_absolute() or (args.output.exists() and any(args.output.iterdir())):
        raise ValueError("Output must be an absolute empty/new independent directory")
    if len(set(args.scans)) != len(args.scans):
        raise ValueError("Duplicate requested scans")
    source = cd._source_provenance()
    if not source["clean"] or source["remote"] != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Run only committed clean approved source")
    neighbor_manifest = json.loads(args.neighbor_manifest.read_text(encoding="utf-8"))
    estimates_meta_path = args.estimates_dir / "metadata.json"
    estimates_meta = json.loads(estimates_meta_path.read_text(encoding="utf-8"))
    if estimates_meta.get("status") != "complete":
        raise ValueError("Candidate estimates must be complete before fixed MV selection")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(status="partial_decisions", source=source, inputs=[], decision_files=[], evaluation_files=[],
        estimates_metadata_sha256=cd.sha256(estimates_meta_path), estimates_source=estimates_meta["source"],
        neighbor_manifest_sha256=cd.sha256(args.neighbor_manifest),
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        fixed_rule=dict(pool=list(POOL), tie_mm=TIE_MM, minimum_common_neighbors=MIN_COMMON,
            neighbors="Original multi_view.json eight neighbors, no self or reselection",
            score="Median abs(projected camera-z minus bilinear neighbor expected depth), converted to DTU mm",
            validity="All four bilinear pixels finite positive; a neighbor is used only if all four pool candidates are valid",
            fallback="Huber1mm when fewer than three common neighbors; same first-pool preference for <=1e-9mm score ties"),
        interpretation="GT-free fixed candidate consistency ranking against frozen expected maps, with unmodeled occlusion and correlated-model bias. It does not prove visibility, geometry correctness, a pixelwise GT, or new independent training evidence. Source archive principal-point/crop provenance remains incomplete; this explicitly uses the existing Camera center convention validated against sample poses/focal. Decisions are flushed/hashed before separate evaluation accesses candidate NN columns. NN is exact one-way scan-point distance, never depth MAE or Chamfer.")
    atomic_json(args.output / "metadata.json", metadata)
    start = time.monotonic()
    decision_jobs = []
    try:
        for scan in args.scans:
            check_time(start, args.max_seconds)
            data = load_scan(scan, args, neighbor_manifest)
            metadata["inputs"].append(data["provenance"])
            cache = {}
            for vm in data["sample"]["views"]:
                check_time(start, args.max_seconds)
                name = vm["image_name"]
                candidates = [v for v in estimates_meta["completed_views"] if int(v["scan"]) == scan and v["view"] == name]
                if len(candidates) != 1:
                    raise ValueError("Missing or duplicate completed estimate view")
                record = candidates[0]
                estimate_path = args.estimates_dir / record["file"]
                raw_path = args.sample_root / f"scan{scan}" / vm["file"]
                if cd.sha256(raw_path) != vm["npz_sha256"]:
                    raise ValueError("Sample NPZ SHA mismatch")
                with np.load(raw_path) as raw:
                    xy = raw["pixel_xy"].astype(np.int64)
                neighbors = []
                for nearest in data["neighbors"][name]:
                    if nearest not in cache:
                        cache[nearest], provenance = load_map(scan, nearest, data, args)
                        data["provenance"].setdefault("depth_maps", []).append(provenance)
                    neighbors.append((nearest, data["cameras"][nearest], cache[nearest]))
                decisions = []
                for row in prediction_rows(estimate_path, record["sha256"]):
                    check_time(start, args.max_seconds)
                    if row["scan"] != scan or row["view"] != name or row["ray"] != len(decisions) or row["raw_npz_sha256"] != vm["npz_sha256"]:
                        raise ValueError("Estimate/raw ray identity mismatch")
                    if row["ray"] >= len(xy) or tuple(xy[row["ray"]]) != (row["x"], row["y"]):
                        raise ValueError("Estimate pixel differs from source random ray")
                    decision = select_candidate((row["x"], row["y"]), [row[m] for m in POOL], data["cameras"][name], neighbors, data["scale"])
                    decisions.append(dict(scan=scan, view=name, ray=row["ray"], x=row["x"], y=row["y"],
                        **decision, **{f"candidate_{m}_mm": row[m] for m in POOL}))
                if len(decisions) != vm["random_samples"] or vm["qualitative_samples"] != 0:
                    raise ValueError("Estimate sample count differs from original random rays")
                output = args.output / f"scan{scan}_view{name}_decisions.csv"
                write_csv(output, decisions)
                metadata["decision_files"].append(dict(scan=scan, view=name, file=output.name, sha256=cd.sha256(output), rays=len(decisions)))
                decision_jobs.append((decisions, estimate_path, record["sha256"]))
                metadata["elapsed_seconds"] = time.monotonic()-start
                atomic_json(args.output / "metadata.json", metadata)
                print(json.dumps(dict(phase="decisions", scan=scan, view=name, rays=len(decisions))), flush=True)
        # This explicit phase boundary is saved before any NN column is read.
        metadata["status"] = "decisions_complete_evaluation_pending"
        atomic_json(args.output / "metadata.json", metadata)
        evaluation, summary = [], []
        for decisions, estimate_path, estimate_sha in decision_jobs:
            check_time(start, args.max_seconds)
            rows = evaluate(decisions, estimate_path, estimate_sha)
            scan, name = rows[0]["scan"], rows[0]["view"]
            path = args.output / f"scan{scan}_view{name}_evaluation.csv"
            write_csv(path, rows)
            metadata["evaluation_files"].append(dict(scan=scan, view=name, file=path.name, sha256=cd.sha256(path)))
            evaluation.extend(rows)
            summary.extend(summarize(rows, "view", scan, name))
        for scan in args.scans:
            summary.extend(summarize([row for row in evaluation if row["scan"] == scan], "scan", scan))
        summary.extend(summarize(evaluation, "pooled"))
        write_csv(args.output / "summary.csv", summary)
        metadata.update(status="complete", rays=len(evaluation), elapsed_seconds=time.monotonic()-start,
            summary_sha256=cd.sha256(args.output / "summary.csv"))
        atomic_json(args.output / "metadata.json", metadata)
    except BaseException as exc:
        metadata.update(error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-start)
        atomic_json(args.output / "metadata.json", metadata)
        raise


def self_check():
    camera = dict(image_size=(9, 11), focal=(10., 12.), c2w=np.eye(4), w2c=np.eye(4))
    point = world_points((3., 4.), np.array([2.]), camera)
    uv, z = project(point, camera)
    np.testing.assert_allclose(uv, [[3., 4.]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(z, [2.], rtol=0, atol=0)
    dense = np.full((9, 11), 2.)
    np.testing.assert_allclose(bilinear(dense, np.array([[3.2, 4.7]])), [2.], rtol=0, atol=1e-14)
    invalid = dense.copy()
    invalid[5, 4] = np.nan
    assert math.isnan(bilinear(invalid, np.array([[3.2, 4.7]]))[0])
    assert math.isnan(bilinear(dense, np.array([[10., 4.]]))[0])
    ns = [(str(i), camera, dense) for i in range(8)]
    selected = select_candidate((5., 4.), np.array([2., 2.2, 2.4, 2.6]), camera, ns, 1.)
    assert selected["selected_method"] == "huber_1mm" and selected["common_neighbors"] == 8 and not selected["fallback"]
    tied = select_candidate((5., 4.), np.array([2., 2., 2., 2.]), camera, ns, 1.)
    assert tied["selected_method"] == "huber_1mm"
    fallback = select_candidate((5., 4.), np.array([2., 2., 2., 2.]), camera, ns[:2], 1.)
    assert fallback["fallback"] == 1 and fallback["selected_method"] == "huber_1mm"
    missing = select_candidate((5., 4.), np.array([2., np.nan, 2., 2.]), camera, ns, 1.)
    assert missing["common_neighbors"] == 0 and missing["fallback"] == 1
    # Nontrivial shared-neighbor requirement: only one candidate's pixel hits a hole.
    other = dict(camera, c2w=np.eye(4), w2c=np.eye(4))
    other["c2w"][0, 3] = .3
    other["w2c"][0, 3] = -.3
    points = world_points((5., 4.), np.array([1., 2., 3., 4.]), camera)
    projected, _ = project(points, other)
    sparse = dense.copy()
    sparse[4, 2] = np.nan
    assert np.isfinite(bilinear(sparse, projected)).sum() == 3
    bad = select_candidate((5., 4.), np.array([1., 2., 3., 4.]), camera, [(str(i), other, sparse) for i in range(8)], 1.)
    assert bad["common_neighbors"] == 0 and bad["fallback"] == 1
    # Selection's signature contains geometry/depth only, never a GT/NN argument.
    import inspect
    assert tuple(inspect.signature(select_candidate).parameters) == ("pixel", "depths_mm", "source_camera", "neighbors", "scale_to_mm")
    print("Camera roundtrip, plane/bilinear masks, common-neighbor fairness, tie/fallback and GT-free decision API: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    for key in ("estimates-dir", "sample-root", "input-root", "accuracy-root", "neighbor-manifest", "output"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--scans", type=int, nargs="+")
    parser.add_argument("--max-seconds", type=float, default=600.)
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    for key in ("estimates_dir", "sample_root", "input_root", "accuracy_root", "neighbor_manifest", "output", "scans"):
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} is required")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
