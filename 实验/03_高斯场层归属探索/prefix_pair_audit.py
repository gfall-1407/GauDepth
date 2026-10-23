#!/usr/bin/env python3
"""Pair frozen all-member and contribution-prefix maps on the same masked pixels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


SCANS = (24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122)
METHODS = ((1, "top10pct_count"), (2, "top50pct_count"), (3, "cover90pct_weight"))
BIN_EDGES_MM = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, math.inf)
BIN_LABELS = ("0-0.1", "0.1-0.5", "0.5-1", "1-2", "2-5", "5-inf")
EPS_MM = 1e-4
EXPECTED_COMMIT = "d72b008b07c85ca78b7de84574d8c8d379a47591"
EXPECTED_INSTANCE = "d6a7459179-693ab6ee"
EXPECTED_HOST = "autodl-container-d6a7459179-693ab6ee"
EXPECTED_METHODS = ["all", "top10pct_count", "top50pct_count", "cover90pct_weight"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _self_check() -> None:
    # Stored camera-z maps are scene units; the metadata scale converts them to DTU mm.
    scene_depth = np.array([1.25, 1.2509765625], dtype=np.float32)
    scale_to_mm = 1024.0
    depth_mm = scene_depth.astype(np.float64) * scale_to_mm
    assert np.array_equal(depth_mm, [1280.0, 1281.0])
    assert math.isclose(float(depth_mm[1] - depth_mm[0]), 1.0, rel_tol=0, abs_tol=1e-12)
    # Signed axial shifts: negative is nearer, positive is farther.
    shift = np.array([-5.0, -0.1, -0.00005, 0.0, 0.00005, 0.1, 5.0])
    direction = np.where(shift < 0, "near", np.where(shift > 0, "far", "zero"))
    assert direction.tolist() == ["near", "near", "near", "zero", "far", "far", "far"]
    mag_bin = np.digitize(np.abs(shift), BIN_EDGES_MM[1:-1], right=False)
    assert mag_bin.tolist() == [5, 1, 0, 0, 0, 1, 5]
    delta_error = np.array([-2 * EPS_MM, -EPS_MM, 0.0, EPS_MM, 2 * EPS_MM])
    improved = delta_error < -EPS_MM
    worsened = delta_error > EPS_MM
    tied = np.abs(delta_error) <= EPS_MM
    assert improved.tolist() == [True, False, False, False, False]
    assert worsened.tolist() == [False, False, False, False, True]
    assert tied.tolist() == [False, True, True, True, False]
    empty = np.array([], dtype=np.float64)
    assert int(np.isfinite(empty).sum()) == 0 and float(empty.sum()) == 0.0
    invalid = np.array([np.nan, np.inf, -np.inf])
    assert not np.isfinite(invalid).any()


def _stats(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "sum": 0.0}
    return {"n": int(len(a)), "mean": float(a.mean()), "sum": float(a.sum(dtype=np.float64))}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_and_views(input_root: Path, retrieval: dict):
    assert retrieval["status"] == "complete"
    assert retrieval["source_instance_id"] == EXPECTED_INSTANCE
    assert retrieval["source_hostname"] == EXPECTED_HOST
    assert retrieval["run_commit"] == EXPECTED_COMMIT
    assert retrieval["verified_scans"] == list(SCANS)
    assert retrieval["scan_count"] == 15 and retrieval["view_count"] == 855
    assert retrieval["bad_view_hashes"] == 0
    assert retrieval["mask_pixels"] == 152505785
    for scan in SCANS:
        root = input_root / f"scan{scan}"
        meta_path = root / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["git_commit"] == EXPECTED_COMMIT, (scan, meta.get("git_commit"))
        assert meta["scan"] == scan and meta["status"] == "complete"
        assert meta["methods"] == EXPECTED_METHODS
        assert meta["hostname"] == EXPECTED_HOST
        assert meta["git_remote"] == "git@github.com:gfall-1407/GauDepth.git"
        names = meta["camera_names"]
        expected_cameras = 49 if scan in (24, 37, 40, 55, 63, 65, 69) else 64
        assert len(names) == expected_cameras and len(set(names)) == expected_cameras
        assert [v["name"] for v in meta["views"]] == names
        assert len(meta["views"]) == expected_cameras
        for key in ("model_sha256", "cfg_sha256", "cameras_npz_sha256", "gt_sha256", "obs_sha256",
                    "host_binary_sha256", "reader_binary_sha256", "reader_source_sha256"):
            value = meta[key]
            assert len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower()), (scan, key)
        yield scan, meta, meta_path, root


def _verify_reference_sample(input_root: Path, influence_root: Path) -> dict:
    scan_root = input_root / "scan24"
    accuracy_meta = json.loads((scan_root / "metadata.json").read_text(encoding="utf-8"))
    influence_scan = influence_root / "scan24"
    influence_meta = json.loads((influence_scan / "metadata.json").read_text(encoding="utf-8"))
    assert influence_meta["status"] == "complete" and influence_meta["scan"] == 24
    assert influence_meta["git_commit"] == "c3b8b28f1fcd96656b285a66820f1007dd422caf"
    for key in ("model_sha256", "cfg_sha256", "cameras_npz_sha256", "gt_sha256", "obs_sha256"):
        assert influence_meta[key] == accuracy_meta[key], key
    scale = float(accuracy_meta["alignment"]["scale_to_dtu_mm"])
    assert math.isclose(scale, float(influence_meta["alignment"]["scale_to_dtu_mm"]), rel_tol=0, abs_tol=1e-12)
    view = next(v for v in accuracy_meta["views"] if v["name"] == "0000")
    influence_view = next(v for v in influence_meta["views"] if v["name"] == "0000")
    assert view["sha256"] and sha256(scan_root / view["file"]) == view["sha256"]
    influence_files = {item["name"]: item["sha256"] for item in influence_view["files"]}
    summary_path = influence_scan / "view_000.json"
    sample_path = influence_scan / "view_000_sample.npz"
    assert sha256(summary_path) == influence_files[summary_path.name]
    assert sha256(sample_path) == influence_files[sample_path.name]
    with np.load(scan_root / view["file"], allow_pickle=False) as maps, \
            np.load(sample_path, allow_pickle=False) as samples:
        map_index = maps["pixel_index"]
        sample_index = samples["pixel_index"]
        positions = np.searchsorted(map_index, sample_index)
        assert np.all(positions < len(map_index)) and np.array_equal(map_index[positions], sample_index)
        depth = maps["weighted_mean_depth"].astype(np.float64)
        sample_mean = samples["mean"]
        assert sample_mean.shape == (len(sample_index), 6)
        checks = []
        for col, name in zip((1, 2, 3), ("top10pct_count", "top50pct_count", "cover90pct_weight")):
            sample_shift_mm = (depth[positions, col] - depth[positions, 0]) * scale
            reference_sample_shift_mm = (sample_mean[:, col] - sample_mean[:, 0]) * scale
            sample_error = np.abs(sample_shift_mm - reference_sample_shift_mm)
            assert float(sample_error.max()) <= 1e-4, (name, float(sample_error.max()))
            full_view_mean_mm = float(((depth[:, col] - depth[:, 0]) * scale).mean())
            ref_stats = json.loads(summary_path.read_text(encoding="utf-8"))["comparison"]["all_mask"][name]["axial_shift_mm"]
            reference_view_mean_mm = float(ref_stats["sum"] / ref_stats["n"])
            assert abs(full_view_mean_mm - reference_view_mean_mm) <= 5e-6, (name, full_view_mean_mm, reference_view_mean_mm)
            checks.append({"method": name, "sample_count": len(sample_index),
                "sample_max_abs_difference_mm": float(sample_error.max()),
                "sample_mean_abs_difference_mm": float(sample_error.mean()),
                "map_view_mean_shift_mm": full_view_mean_mm,
                "influence_view_mean_shift_mm": reference_view_mean_mm,
                "view_mean_abs_difference_mm": abs(full_view_mean_mm - reference_view_mean_mm)})
    return {"status": "passed", "scan": 24, "camera": "0000", "scale_to_dtu_mm": scale,
        "accuracy_view_sha256": view["sha256"], "influence_summary_sha256": influence_files[summary_path.name],
        "influence_sample_sha256": influence_files[sample_path.name], "checks": checks,
        "tolerance": {"per_sample_max_abs_mm": 1e-4, "full_view_mean_abs_mm": 5e-6}}


def run_transition_audit(input_root: Path, output_root: Path) -> None:
    _self_check()
    started = time.monotonic()
    retrieval_path = input_root / "retrieval_status.json"
    validation_path = input_root / "local_validation_complete.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval.update(json.loads(validation_path.read_text(encoding="utf-8")))
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing transition output: {output_root}")

    thresholds = ((1.0, 5.0), (2.0, 10.0))
    categories = ("near_to_far", "far_to_near", "both_near", "both_far", "other")
    methods = ((0, "all"),) + tuple((column, name) for column, name in METHODS)
    scan_states = {}
    pooled = {}
    pooled_shift_parts = defaultdict(list)
    examples = defaultdict(list)
    source_rows = []
    total_mask = total_views = 0

    for scan, meta, meta_path, scan_root in _metadata_and_views(input_root, retrieval):
        scale = float(meta["alignment"]["scale_to_dtu_mm"])
        assert math.isfinite(scale) and scale > 0
        scan_state = {}
        scan_mask = 0
        for view in meta["views"]:
            assert time.monotonic() - started <= 600, "transition audit exceeded 600 seconds"
            view_path = scan_root / view["file"]
            digest = sha256(view_path)
            assert digest == view["sha256"], (scan, view["name"], digest, view["sha256"])
            with np.load(view_path, allow_pickle=False) as archive:
                index = archive["pixel_index"]
                depth = archive["weighted_mean_depth"].astype(np.float64, copy=False) * scale
                error = archive["gt_point_distance_mm"].astype(np.float64, copy=False)
                counts = archive["selected_counts"]
                image_size = archive["image_size"]
                assert depth.shape == error.shape == (len(index), 4)
                assert counts.shape == (len(index), 4) and image_size.shape == (2,)
                assert int(view["pixels"]) == len(index)
                width = int(image_size[1])
                assert width > 0
                base_valid = np.isfinite(depth[:, 0]) & np.isfinite(error[:, 0])
                scan_mask += len(index)
                for col, method in methods:
                    candidate_valid = np.isfinite(depth[:, col]) & np.isfinite(error[:, col])
                    pair = base_valid & candidate_valid
                    pair_n = int(pair.sum())
                    delta_z = depth[:, col] - depth[:, 0]
                    for near_mm, far_mm in thresholds:
                        key = (scan, method, near_mm, far_mm)
                        state = scan_state.setdefault(key, {"mask": 0, "pair": 0, "invalid": 0,
                            "base_nn_finite": 0, "candidate_nn_finite": 0,
                            "base_depth_finite": 0, "candidate_depth_finite": 0,
                            "counts": {category: 0 for category in categories}, "n2f_shift": [],
                            "n2f_dir": {"near": 0, "far": 0, "zero": 0}})
                        state["mask"] += len(index)
                        state["pair"] += pair_n
                        state["invalid"] += len(index) - pair_n
                        state["base_nn_finite"] += int(np.isfinite(error[:, 0]).sum())
                        state["candidate_nn_finite"] += int(np.isfinite(error[:, col]).sum())
                        state["base_depth_finite"] += int(np.isfinite(depth[:, 0]).sum())
                        state["candidate_depth_finite"] += int(np.isfinite(depth[:, col]).sum())
                        base_near = error[:, 0] <= near_mm
                        cand_near = error[:, col] <= near_mm
                        base_far = error[:, 0] >= far_mm
                        cand_far = error[:, col] >= far_mm
                        masks = {
                            "near_to_far": pair & base_near & cand_far,
                            "far_to_near": pair & base_far & cand_near,
                            "both_near": pair & base_near & cand_near,
                            "both_far": pair & base_far & cand_far,
                        }
                        classified = np.zeros(len(index), dtype=bool)
                        for category, selected in masks.items():
                            classified |= selected
                            n = int(selected.sum())
                            state["counts"][category] += n
                            pool_key = (method, near_mm, far_mm)
                            pool = pooled.setdefault(pool_key, {"mask": 0, "pair": 0, "invalid": 0,
                                "base_nn_finite": 0, "candidate_nn_finite": 0,
                                "base_depth_finite": 0, "candidate_depth_finite": 0,
                                "counts": {c: 0 for c in categories}, "n2f_dir": {"near": 0, "far": 0, "zero": 0}})
                            pool["counts"][category] += n
                            if n and len(examples[(method, near_mm, far_mm, category)]) < 3:
                                remaining = 3 - len(examples[(method, near_mm, far_mm, category)])
                                positions = np.flatnonzero(selected)[:remaining]
                                for pos in positions:
                                    pixel = int(index[pos])
                                    examples[(method, near_mm, far_mm, category)].append({
                                        "scan": scan, "camera": view["name"], "pixel_index": pixel,
                                        "row": pixel // width, "column": pixel % width,
                                        "image_size": [int(x) for x in image_size],
                                        "baseline_depth_mm": float(depth[pos, 0]),
                                        "candidate_depth_mm": float(depth[pos, col]),
                                        "baseline_scan_point_nn_mm": float(error[pos, 0]),
                                        "candidate_scan_point_nn_mm": float(error[pos, col]),
                                        "axial_shift_mm": float(delta_z[pos]),
                                        "selected_counts_all_and_candidate": [int(counts[pos, 0]), int(counts[pos, col])],
                                        "quality_mass_fraction_available": False,
                                    })
                        masks["other"] = pair & ~classified
                        other_n = int(masks["other"].sum())
                        state["counts"]["other"] += other_n
                        pool_key = (method, near_mm, far_mm)
                        pool = pooled.setdefault(pool_key, {"mask": 0, "pair": 0, "invalid": 0,
                            "base_nn_finite": 0, "candidate_nn_finite": 0,
                            "base_depth_finite": 0, "candidate_depth_finite": 0,
                            "counts": {c: 0 for c in categories}, "n2f_dir": {"near": 0, "far": 0, "zero": 0}})
                        pool["counts"]["other"] += other_n
                        n2f = masks["near_to_far"]
                        shifts = delta_z[n2f]
                        if shifts.size:
                            state["n2f_shift"].append(np.abs(shifts))
                            pooled_shift_parts[pool_key].append(np.abs(shifts))
                            state["n2f_dir"]["near"] += int((shifts < 0).sum())
                            state["n2f_dir"]["far"] += int((shifts > 0).sum())
                            state["n2f_dir"]["zero"] += int((shifts == 0).sum())
                            pool["n2f_dir"]["near"] += int((shifts < 0).sum())
                            pool["n2f_dir"]["far"] += int((shifts > 0).sum())
                            pool["n2f_dir"]["zero"] += int((shifts == 0).sum())
                        pool["mask"] += len(index)
                        pool["pair"] += pair_n
                        pool["invalid"] += len(index) - pair_n
                        pool["base_nn_finite"] += int(np.isfinite(error[:, 0]).sum())
                        pool["candidate_nn_finite"] += int(np.isfinite(error[:, col]).sum())
                        pool["base_depth_finite"] += int(np.isfinite(depth[:, 0]).sum())
                        pool["candidate_depth_finite"] += int(np.isfinite(depth[:, col]).sum())
            total_views += 1
        total_mask += scan_mask
        for key, state in scan_state.items():
            state["n2f_shift"] = np.concatenate(state["n2f_shift"]) if state["n2f_shift"] else np.array([], dtype=float)
            scan_states[key] = state
        source_rows.append({"scan": scan, "metadata_sha256": sha256(meta_path), "scale_to_dtu_mm": scale,
            "camera_count": len(meta["camera_names"]), "views": len(meta["views"]), "mask_pixels": scan_mask,
            "model_sha256": meta["model_sha256"], "cfg_sha256": meta["cfg_sha256"],
            "cameras_npz_sha256": meta["cameras_npz_sha256"], "gt_sha256": meta["gt_sha256"],
            "obs_sha256": meta["obs_sha256"]})
    assert total_views == 855 and total_mask == retrieval["mask_pixels"] == 152505785
    summary_rows = []
    for _, method in methods:
        for near_mm, far_mm in thresholds:
            pool = pooled[(method, near_mm, far_mm)]
            for aggregation in ("scan", "pixel_pool", "scan_equal"):
                if aggregation == "scan":
                    scan_rows = []
                    for scan in SCANS:
                        k = (scan, method, near_mm, far_mm)
                        # Scan state is retained below in scan_states.
                        state = scan_states[k]
                        row = _transition_row(aggregation, scan, method, near_mm, far_mm, state,
                            np.concatenate(state["n2f_shift"]) if state["n2f_shift"] else np.array([], dtype=float), categories)
                        summary_rows.append(row)
                        scan_rows.append(row)
                    continue
                if aggregation == "pixel_pool":
                    shifts = np.concatenate(pooled_shift_parts[(method, near_mm, far_mm)]) if pooled_shift_parts[(method, near_mm, far_mm)] else np.array([], dtype=float)
                    state = pool
                else:
                    scan_rows = [r for r in summary_rows if r["aggregation"] == "scan" and r["method"] == method and
                                 r["near_threshold_mm"] == near_mm and r["far_threshold_mm"] == far_mm]
                    pair_scans = [r for r in scan_rows if int(r["n_finite_pair"]) > 0]
                    shifts = np.array([], dtype=float)
                    state = {"mask": sum(int(r["n_mask"]) for r in scan_rows),
                        "pair": sum(int(r["n_finite_pair"]) for r in scan_rows),
                        "invalid": sum(int(r["n_invalid_pair"]) for r in scan_rows),
                        "base_nn_finite": sum(int(r["n_base_nn_finite"]) for r in scan_rows),
                        "candidate_nn_finite": sum(int(r["n_candidate_nn_finite"]) for r in scan_rows),
                        "base_depth_finite": sum(int(r["n_base_depth_finite"]) for r in scan_rows),
                        "candidate_depth_finite": sum(int(r["n_candidate_depth_finite"]) for r in scan_rows),
                        "counts": {c: None for c in categories},
                        "n2f_dir": {d: None for d in ("near", "far", "zero")}}
                    for c in categories:
                        vals = [float(r[f"fraction_{c}"]) for r in pair_scans]
                        state["counts"][c] = None
                        state.setdefault("fraction_overrides", {})[c] = float(np.mean(vals)) if vals else None
                    for d in ("near", "far", "zero"):
                        vals = [float(r[f"near_to_far_shift_direction_{d}_fraction"]) for r in pair_scans]
                        state["fraction_overrides"][f"near_to_far_shift_direction_{d}"] = float(np.mean(vals)) if vals else None
                    for field in ("mean_near_to_far_abs_shift_mm", "median_near_to_far_abs_shift_mm"):
                        vals = [float(r[field]) for r in pair_scans if r[field] is not None]
                        state.setdefault("value_overrides", {})[field] = float(np.mean(vals)) if vals else None
                summary_rows.append(_transition_row(aggregation, "ALL" if aggregation=="pixel_pool" else "MEAN_OF_SCAN_MEANS",
                    method, near_mm, far_mm, state, shifts, categories))

    output_root.mkdir(parents=True)
    _write_csv(output_root / "prefix_transition_summary.csv", summary_rows)
    example_rows = []
    for key, values in sorted(examples.items()):
        method, near_mm, far_mm, category = key
        for rank, value in enumerate(values, 1):
            example_rows.append({"method": method, "near_threshold_mm": near_mm, "far_threshold_mm": far_mm,
                "category": category, "display_rank_after_analysis": rank, **value})
    _write_csv(output_root / "prefix_transition_examples.csv", example_rows)
    audit = {"status": "complete", "run_commit": EXPECTED_COMMIT, "input_root": str(input_root),
        "scan_count": len(source_rows), "view_count": total_views, "mask_pixels": total_mask,
        "invalid_denominator_definition": "mask pixels lacking finite baseline/candidate camera-z or scan-point NN; pair counts are finite in all four values",
        "transition_definition": "NN is one-way distance to the scan point cloud, not pixelwise GT depth or a correctness label",
        "near_far_thresholds_mm": [{"near_le_mm": n, "far_ge_mm": f} for n,f in thresholds],
        "categories": list(categories), "direction_definition": "candidate camera-z minus all-member camera-z; negative=nearer along camera axis, positive=farther",
        "example_selection": "first three matching flattened pixel indices in scan/view traversal order; selected after aggregate rule fixed; quality mass fraction is not present in accuracy_v1 maps",
        "methods": [name for _, name in methods], "source_provenance": source_rows,
        "aggregate_rule": "pixel_pool sums counts once across all scans; scan_equal averages per-scan proportions and per-scan near-to-far shift summaries",
        "elapsed_seconds": time.monotonic()-started, "outputs": ["prefix_transition_summary.csv", "prefix_transition_examples.csv"]}
    (output_root / "prefix_transition_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def _transition_row(aggregation, scan, method, near_mm, far_mm, state, shifts, categories):
    pair_n=int(state["pair"]); mask_n=int(state["mask"]); invalid=int(state["invalid"])
    row={"aggregation":aggregation,"scan":scan,"method":method,"near_threshold_mm":near_mm,
        "far_threshold_mm":far_mm,"n_mask":mask_n,"n_finite_pair":pair_n,"n_invalid_pair":invalid,
        "n_base_nn_finite":state["base_nn_finite"],"n_candidate_nn_finite":state["candidate_nn_finite"],
        "n_base_depth_finite":state["base_depth_finite"],"n_candidate_depth_finite":state["candidate_depth_finite"],
        "n_near_to_far_shift_direction_near":state["n2f_dir"]["near"],
        "n_near_to_far_shift_direction_far":state["n2f_dir"]["far"],
        "n_near_to_far_shift_direction_zero":state["n2f_dir"]["zero"],
        "mean_near_to_far_abs_shift_mm":float(shifts.mean()) if len(shifts) else None,
        "median_near_to_far_abs_shift_mm":float(np.median(shifts)) if len(shifts) else None}
    overrides=state.get("fraction_overrides",{})
    for category in categories:
        count=state["counts"][category]
        row[f"n_{category}"]=count
        row[f"fraction_{category}"]=overrides.get(category, count/pair_n if pair_n and count is not None else None)
    for direction in ("near","far","zero"):
        count=state["n2f_dir"][direction]
        row[f"near_to_far_shift_direction_{direction}_fraction"]=(
            overrides.get(f"near_to_far_shift_direction_{direction}", count/state["counts"]["near_to_far"]
                if state["counts"]["near_to_far"] else None))
    for field,value in state.get("value_overrides",{}).items():
        row[field]=value
    return row


def run(input_root: Path, output_root: Path, reference_influence_root: Path | None = None) -> None:
    _self_check()
    reference_sample_check = (_verify_reference_sample(input_root, reference_influence_root)
                              if reference_influence_root is not None else None)
    retrieval_path = input_root / "retrieval_status.json"
    validation_path = input_root / "local_validation_complete.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    retrieval.update(validation)
    expected_output_names = {"prefix_pair_shift_bins.csv", "prefix_pair_scan_summary.csv",
                             "input_provenance.csv", "prefix_pair_audit.json"}
    if output_root.exists():
        existing = {p.name for p in output_root.iterdir()}
        assert existing == expected_output_names, f"Refusing to replace unknown output contents: {existing}"
        old_summary = json.loads((output_root / "prefix_pair_audit.json").read_text(encoding="utf-8"))
        assert old_summary.get("status") == "complete"
        assert old_summary.get("input_root") == str(input_root)
        assert old_summary.get("run_commit") == EXPECTED_COMMIT

    source_rows = []
    view_rows = []
    scan_rows = []
    bin_records: dict[tuple, dict] = {}
    totals: dict[tuple, dict] = defaultdict(lambda: {
        "n": 0, "shift_sum": 0.0, "abs_shift_sum": 0.0,
        "base_error_sum": 0.0, "candidate_error_sum": 0.0,
        "delta_error_sum": 0.0, "improved": 0, "worsened": 0, "tied": 0,
        "cameras": set(),
    })
    scan_totals: dict[tuple, list[dict]] = defaultdict(list)
    denominator_totals: dict[tuple, dict] = defaultdict(lambda: {"mask": 0, "pair": 0})
    input_view_hashes = []
    total_pixels = 0
    import time
    total_started = time.monotonic()

    for scan, meta, meta_path, scan_root in _metadata_and_views(input_root, retrieval):
        scale_to_dtu_mm = float(meta["alignment"]["scale_to_dtu_mm"])
        assert math.isfinite(scale_to_dtu_mm) and scale_to_dtu_mm > 0, (scan, scale_to_dtu_mm)
        scan_started = time.monotonic()
        scan_pixels = 0
        for view in meta["views"]:
            assert time.monotonic() - scan_started <= 120, f"scan {scan} exceeded 120s"
            assert time.monotonic() - total_started <= 600, "audit exceeded 600s"
            view_path = scan_root / view["file"]
            digest = sha256(view_path)
            assert digest == view["sha256"], (scan, view["name"], digest, view["sha256"])
            input_view_hashes.append({"scan": scan, "camera": view["name"], "file": view["file"], "sha256": digest})
            with np.load(view_path, allow_pickle=False) as archive:
                index = archive["pixel_index"]
                depth = archive["weighted_mean_depth"].astype(np.float64, copy=False)
                error = archive["gt_point_distance_mm"].astype(np.float64, copy=False)
                obs = archive["in_observation"]
                image_size = archive["image_size"]
                assert depth.shape == error.shape == obs.shape == (len(index), 4)
                assert index.ndim == 1 and len(np.unique(index)) == len(index)
                assert int(view["pixels"]) == len(index)
                assert image_size.shape == (2,) and int(image_size[0]) > 0 and int(image_size[1]) > 0
                assert not len(index) or (int(index.min()) >= 0 and int(index.max()) < int(np.prod(image_size)))
                assert np.isfinite(depth[:, 0]).all() and np.isfinite(error[:, 0]).all()
                scan_pixels += len(index)

                for col, method in METHODS:
                    pair_valid = (np.isfinite(depth[:, 0]) & np.isfinite(depth[:, col]) &
                                  np.isfinite(error[:, 0]) & np.isfinite(error[:, col]))
                    base_z = depth[pair_valid, 0] * scale_to_dtu_mm
                    cand_z = depth[pair_valid, col] * scale_to_dtu_mm
                    base_e = error[pair_valid, 0]
                    cand_e = error[pair_valid, col]
                    shift = cand_z - base_z
                    delta_e = cand_e - base_e
                    direction = np.where(shift < 0.0, "near", np.where(shift > 0.0, "far", "zero"))
                    bin_id = np.digitize(np.abs(shift), BIN_EDGES_MM[1:-1], right=False)
                    improved = delta_e < -EPS_MM
                    worsened = delta_e > EPS_MM
                    tied = np.abs(delta_e) <= EPS_MM

                    for d in ("near", "far", "zero"):
                        for b, label in enumerate(BIN_LABELS):
                            take = (direction == d) & (bin_id == b)
                            n = int(take.sum())
                            row = {
                                "aggregation": "scan", "scan": scan, "method": method,
                                "direction": d, "abs_shift_bin_mm": label, "n": n,
                                "camera_count": int(n > 0), "camera_names": view["name"] if n else "",
                                "mean_axial_shift_mm": float(shift[take].mean()) if n else None,
                                "mean_abs_shift_mm": float(np.abs(shift[take]).mean()) if n else None,
                                "mean_baseline_nn_mm": float(base_e[take].mean()) if n else None,
                                "mean_candidate_nn_mm": float(cand_e[take].mean()) if n else None,
                                "mean_delta_nn_mm": float(delta_e[take].mean()) if n else None,
                                "improved": int((improved & take).sum()),
                                "worsened": int((worsened & take).sum()),
                                "tied": int((tied & take).sum()),
                            }
                            key = (scan, method, d, label)
                            rec = bin_records.setdefault(key, {k: 0 for k in (
                                "n", "shift_sum", "abs_shift_sum", "base_error_sum", "candidate_error_sum",
                                "delta_error_sum", "improved", "worsened", "tied")})
                            rec["n"] += n
                            if n:
                                rec["shift_sum"] += float(shift[take].sum(dtype=np.float64))
                                rec["abs_shift_sum"] += float(np.abs(shift[take]).sum(dtype=np.float64))
                                rec["base_error_sum"] += float(base_e[take].sum(dtype=np.float64))
                                rec["candidate_error_sum"] += float(cand_e[take].sum(dtype=np.float64))
                                rec["delta_error_sum"] += float(delta_e[take].sum(dtype=np.float64))
                                rec["improved"] += row["improved"]
                                rec["worsened"] += row["worsened"]
                                rec["tied"] += row["tied"]
                                totals[(method, d, label)]["n"] += n
                                totals[(method, d, label)]["shift_sum"] += rec["shift_sum"] - rec.get("_last_shift_sum", 0.0)
                                totals[(method, d, label)]["abs_shift_sum"] += rec["abs_shift_sum"] - rec.get("_last_abs_shift_sum", 0.0)
                                totals[(method, d, label)]["base_error_sum"] += rec["base_error_sum"] - rec.get("_last_base_error_sum", 0.0)
                                totals[(method, d, label)]["candidate_error_sum"] += rec["candidate_error_sum"] - rec.get("_last_candidate_error_sum", 0.0)
                                totals[(method, d, label)]["delta_error_sum"] += rec["delta_error_sum"] - rec.get("_last_delta_error_sum", 0.0)
                                totals[(method, d, label)]["improved"] += row["improved"]
                                totals[(method, d, label)]["worsened"] += row["worsened"]
                                totals[(method, d, label)]["tied"] += row["tied"]
                                totals[(method, d, label)]["cameras"].add(view["name"])
                                scan_totals[(method, d, label)].append(row)
                                for suffix in ("shift_sum", "abs_shift_sum", "base_error_sum", "candidate_error_sum", "delta_error_sum"):
                                    rec[f"_last_{suffix}"] = rec[suffix]

                    pair_n = int(pair_valid.sum())
                    denominator_totals[(scan, method)]["mask"] += len(index)
                    denominator_totals[(scan, method)]["pair"] += pair_n
                    positive_harm = delta_e[delta_e > EPS_MM]
                    negative_gain = -delta_e[delta_e < -EPS_MM]
                    high2 = np.abs(shift) >= 2.0
                    high5 = np.abs(shift) >= 5.0
                    top10_row = {
                        "scan": scan, "camera": view["name"], "method": method,
                        "n_mask_pixels": len(index), "n_pair": pair_n, "n_missing": len(index) - pair_n,
                        "mean_base_depth_mm": float(base_z.mean()) if pair_n else None,
                        "mean_candidate_depth_mm": float(cand_z.mean()) if pair_n else None,
                        "mean_axial_shift_mm": float(shift.mean()) if pair_n else None,
                        "mean_abs_shift_mm": float(np.abs(shift).mean()) if pair_n else None,
                        "mean_base_nn_mm": float(base_e.mean()) if pair_n else None,
                        "mean_candidate_nn_mm": float(cand_e.mean()) if pair_n else None,
                        "mean_delta_nn_mm": float(delta_e.mean()) if pair_n else None,
                        "improved": int(improved.sum()), "worsened": int(worsened.sum()), "tied": int(tied.sum()),
                        "harm_pixels_fraction": float(worsened.mean()) if pair_n else None,
                        "positive_harm_sum_mm": float(positive_harm.sum(dtype=np.float64)),
                        "negative_gain_sum_mm": float(negative_gain.sum(dtype=np.float64)),
                        "harm_share_of_absolute_net_motion": (float(positive_harm.sum() /
                            (positive_harm.sum() + negative_gain.sum())) if len(positive_harm) or len(negative_gain) else None),
                        "abs_shift_ge2_n": int(high2.sum()),
                        "abs_shift_ge2_nonimprove_fraction": float(np.mean(delta_e[high2] >= -EPS_MM)) if high2.any() else None,
                        "abs_shift_ge2_nonimprove": int((delta_e[high2] >= -EPS_MM).sum()),
                        "abs_shift_ge2_worsened": int((worsened & high2).sum()),
                        "abs_shift_ge2_delta_error_sum_mm": float(delta_e[high2].sum(dtype=np.float64)),
                        "abs_shift_ge2_mean_delta_nn_mm": float(delta_e[high2].mean()) if high2.any() else None,
                        "abs_shift_ge5_n": int(high5.sum()),
                        "abs_shift_ge5_nonimprove_fraction": float(np.mean(delta_e[high5] >= -EPS_MM)) if high5.any() else None,
                        "abs_shift_ge5_nonimprove": int((delta_e[high5] >= -EPS_MM).sum()),
                        "abs_shift_ge5_worsened": int((worsened & high5).sum()),
                        "abs_shift_ge5_delta_error_sum_mm": float(delta_e[high5].sum(dtype=np.float64)),
                        "abs_shift_ge5_mean_delta_nn_mm": float(delta_e[high5].mean()) if high5.any() else None,
                    }
                    view_rows.append(top10_row)
        assert scan_pixels == sum(int(v["pixels"]) for v in meta["views"])
        total_pixels += scan_pixels
        source_rows.append({"scan": scan, "git_commit": meta["git_commit"], "model_sha256": meta["model_sha256"],
            "scale_to_dtu_mm": scale_to_dtu_mm,
            "cfg_sha256": meta["cfg_sha256"], "cameras_npz_sha256": meta["cameras_npz_sha256"],
            "gt_sha256": meta["gt_sha256"], "obs_sha256": meta["obs_sha256"],
            "host_binary_sha256": meta["host_binary_sha256"], "reader_binary_sha256": meta["reader_binary_sha256"],
            "reader_source_sha256": meta["reader_source_sha256"],
            "camera_count": len(meta["camera_names"]), "views": len(meta["views"]), "mask_pixels": scan_pixels,
            "metadata_sha256": sha256(meta_path)})
    assert len(source_rows) == 15 and sum(r["views"] for r in source_rows) == 855
    assert total_pixels == retrieval["mask_pixels"]

    # Reduce view rows to per-scan rows, then pooled pixels and scan-equal means.
    bin_rows = []
    for (scan, method, direction, label), r in sorted(bin_records.items()):
        n = r["n"]
        bin_rows.append({"aggregation": "scan", "scan": scan, "method": method, "direction": direction,
            "abs_shift_bin_mm": label, "n": n,
            "camera_count": int(sum(1 for v in scan_totals[(method, direction, label)] if v["scan"] == scan and v["n"])),
            "camera_names": ";".join(sorted({v["camera_names"] for v in scan_totals[(method, direction, label)]
                                                   if v["scan"] == scan and v["n"]})),
            "mean_axial_shift_mm": r["shift_sum"] / n if n else None,
            "mean_abs_shift_mm": r["abs_shift_sum"] / n if n else None,
            "mean_baseline_nn_mm": r["base_error_sum"] / n if n else None,
            "mean_candidate_nn_mm": r["candidate_error_sum"] / n if n else None,
            "mean_delta_nn_mm": r["delta_error_sum"] / n if n else None,
            "improved": r["improved"], "worsened": r["worsened"], "tied": r["tied"]})
    for _, method in METHODS:
        for d in ("near", "far", "zero"):
            for label in BIN_LABELS:
                total = totals[(method, d, label)]
                n = total["n"]
                pool = {"aggregation": "pixel_pool", "scan": "ALL", "method": method, "direction": d,
                    "abs_shift_bin_mm": label, "n": n, "camera_count": len(total["cameras"]),
                    "camera_names": ";".join(sorted(total["cameras"])),
                    "mean_axial_shift_mm": total["shift_sum"] / n if n else None,
                    "mean_abs_shift_mm": total["abs_shift_sum"] / n if n else None,
                    "mean_baseline_nn_mm": total["base_error_sum"] / n if n else None,
                    "mean_candidate_nn_mm": total["candidate_error_sum"] / n if n else None,
                    "mean_delta_nn_mm": total["delta_error_sum"] / n if n else None,
                    "improved": total["improved"], "worsened": total["worsened"], "tied": total["tied"]}
                bin_rows.append(pool)
                candidates = [r for r in bin_rows if r["aggregation"] == "scan" and r["method"] == method and
                              r["direction"] == d and r["abs_shift_bin_mm"] == label and r["n"] > 0]
                equal = {"aggregation": "scan_equal", "scan": "MEAN_OF_SCAN_MEANS", "method": method,
                    "direction": d, "abs_shift_bin_mm": label, "n": n, "camera_count": len(total["cameras"]),
                    "camera_names": ";".join(sorted(total["cameras"])),
                    "mean_axial_shift_mm": float(np.mean([x["mean_axial_shift_mm"] for x in candidates])) if candidates else None,
                    "mean_abs_shift_mm": float(np.mean([x["mean_abs_shift_mm"] for x in candidates])) if candidates else None,
                    "mean_baseline_nn_mm": float(np.mean([x["mean_baseline_nn_mm"] for x in candidates])) if candidates else None,
                    "mean_candidate_nn_mm": float(np.mean([x["mean_candidate_nn_mm"] for x in candidates])) if candidates else None,
                    "mean_delta_nn_mm": float(np.mean([x["mean_delta_nn_mm"] for x in candidates])) if candidates else None,
                    "improved": total["improved"], "worsened": total["worsened"], "tied": total["tied"],
                    "scans_with_pixels": len(candidates)}
                bin_rows.append(equal)

    sum_fields = ("n_pair", "improved", "worsened", "tied", "positive_harm_sum_mm", "negative_gain_sum_mm",
        "abs_shift_ge2_n", "abs_shift_ge2_nonimprove", "abs_shift_ge2_worsened", "abs_shift_ge2_delta_error_sum_mm",
        "abs_shift_ge5_n", "abs_shift_ge5_nonimprove", "abs_shift_ge5_worsened", "abs_shift_ge5_delta_error_sum_mm")
    mean_fields = ("mean_base_depth_mm", "mean_candidate_depth_mm", "mean_axial_shift_mm", "mean_abs_shift_mm",
        "mean_base_nn_mm", "mean_candidate_nn_mm", "mean_delta_nn_mm")
    for scan in SCANS:
        for _, method in METHODS:
            rows = [r for r in view_rows if r["scan"] == scan and r["method"] == method]
            n = sum(r["n_pair"] for r in rows)
            agg = {"aggregation": "scan", "scan": scan, "camera": "ALL", "method": method}
            den = denominator_totals[(scan, method)]
            agg["n_mask_pixels"], agg["n_missing"] = den["mask"], den["mask"] - den["pair"]
            for field in sum_fields:
                agg[field] = sum(r[field] for r in rows)
            for field in mean_fields:
                agg[field] = sum((r[field] or 0.0) * r["n_pair"] for r in rows) / n if n else None
            agg["harm_pixels_fraction"] = agg["worsened"] / n if n else None
            agg["improved_pixels_fraction"] = agg["improved"] / n if n else None
            harm, gain = agg["positive_harm_sum_mm"], agg["negative_gain_sum_mm"]
            agg["harm_share_of_absolute_net_motion"] = harm / (harm + gain) if harm + gain else None
            for threshold in (2, 5):
                count = agg[f"abs_shift_ge{threshold}_n"]
                agg[f"abs_shift_ge{threshold}_nonimprove_fraction"] = (
                    agg[f"abs_shift_ge{threshold}_nonimprove"] / count if count else None)
                agg[f"abs_shift_ge{threshold}_mean_delta_nn_mm"] = (
                    agg[f"abs_shift_ge{threshold}_delta_error_sum_mm"] / count if count else None)
            scan_rows.append(agg)
    for _, method in METHODS:
        rows = [r for r in scan_rows if r["method"] == method]
        n = sum(r["n_pair"] for r in rows)
        for aggregation in ("pixel_pool", "scan_equal"):
            agg = {"aggregation": aggregation, "scan": "ALL" if aggregation == "pixel_pool" else "MEAN_OF_SCAN_MEANS",
                   "camera": "ALL", "method": method}
            agg["n_mask_pixels"] = sum(r["n_mask_pixels"] for r in rows)
            agg["n_missing"] = sum(r["n_missing"] for r in rows)
            for field in sum_fields:
                agg[field] = sum(r[field] for r in rows)
            for field in mean_fields:
                vals = [r[field] for r in rows if r[field] is not None]
                if aggregation == "pixel_pool":
                    agg[field] = sum(r[field] * r["n_pair"] for r in rows if r[field] is not None) / n if n else None
                else:
                    agg[field] = float(np.mean(vals)) if vals else None
            if aggregation == "pixel_pool":
                agg["harm_pixels_fraction"] = agg["worsened"] / n if n else None
                agg["improved_pixels_fraction"] = agg["improved"] / n if n else None
            else:
                harm_fracs = [r["harm_pixels_fraction"] for r in rows if r["harm_pixels_fraction"] is not None]
                improve_fracs = [r["improved_pixels_fraction"] for r in rows if r["improved_pixels_fraction"] is not None]
                agg["harm_pixels_fraction"] = float(np.mean(harm_fracs)) if harm_fracs else None
                agg["improved_pixels_fraction"] = float(np.mean(improve_fracs)) if improve_fracs else None
            harm, gain = agg["positive_harm_sum_mm"], agg["negative_gain_sum_mm"]
            if aggregation == "pixel_pool":
                agg["harm_share_of_absolute_net_motion"] = harm / (harm + gain) if harm + gain else None
            else:
                shares = [r["harm_share_of_absolute_net_motion"] for r in rows
                          if r["harm_share_of_absolute_net_motion"] is not None]
                agg["harm_share_of_absolute_net_motion"] = float(np.mean(shares)) if shares else None
            for threshold in (2, 5):
                count = agg[f"abs_shift_ge{threshold}_n"]
                if aggregation == "pixel_pool":
                    agg[f"abs_shift_ge{threshold}_nonimprove_fraction"] = (
                        agg[f"abs_shift_ge{threshold}_nonimprove"] / count if count else None)
                    agg[f"abs_shift_ge{threshold}_mean_delta_nn_mm"] = (
                        agg[f"abs_shift_ge{threshold}_delta_error_sum_mm"] / count if count else None)
                else:
                    frac_field = f"abs_shift_ge{threshold}_nonimprove_fraction"
                    mean_field = f"abs_shift_ge{threshold}_mean_delta_nn_mm"
                    agg[frac_field] = float(np.mean([r[frac_field] for r in rows if r[frac_field] is not None])) \
                        if any(r[frac_field] is not None for r in rows) else None
                    agg[mean_field] = float(np.mean([r[mean_field] for r in rows if r[mean_field] is not None])) \
                        if any(r[mean_field] is not None for r in rows) else None
            scan_rows.append(agg)
    output_root.mkdir(parents=True)
    _write_csv(output_root / "prefix_pair_shift_bins.csv", bin_rows)
    _write_csv(output_root / "prefix_pair_scan_summary.csv", scan_rows)
    _write_csv(output_root / "input_provenance.csv", source_rows)
    summary = {
        "status": "complete", "run_commit": EXPECTED_COMMIT,
        "instance_id": EXPECTED_INSTANCE, "hostname": EXPECTED_HOST,
        "input_root": str(input_root), "retrieval_status_sha256": sha256(retrieval_path),
        "local_validation_sha256": sha256(validation_path),
        "scan_count": len(source_rows), "view_count": sum(r["views"] for r in source_rows),
        "mask_pixels": total_pixels, "source_provenance": source_rows,
        "input_view_hashes": input_view_hashes,
        "methods": [name for _, name in METHODS],
        "abs_axial_shift_bins_mm": list(BIN_LABELS),
        "direction_definition": "candidate weighted-mean camera-z minus all-member weighted-mean camera-z; negative=near, positive=far",
        "error_definition": "candidate minus all-member one-way nearest DTU point distance (mm); not pixelwise GT depth MAE or surface distance",
        "epsilon_tie_mm": EPS_MM,
        "top10_tail_definition": "Counts and sums of per-pixel delta-NN > epsilon vs < -epsilon; large-shift non-improvement reported separately for |axial shift|>=2 and >=5mm. No quantiles are averaged.",
        "aggregation": "Pixel pooled rows sum sufficient counts/sums; scan_equal rows average per-scan means only. Quantiles are not computed or averaged.",
        "depth_units": "weighted_mean_depth maps are scene units; converted to DTU mm using each scan metadata alignment.scale_to_dtu_mm; NN errors are already mm and are not rescaled",
        "reference_sample_check": reference_sample_check,
        "self_checks": ["scene-to-mm conversion", "empty", "invalid", "signed near/far/zero", "fixed bins", "epsilon=1e-4mm"],
        "outputs": ["prefix_pair_shift_bins.csv", "prefix_pair_scan_summary.csv", "input_provenance.csv"],
    }
    (output_root / "prefix_pair_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "scans": len(source_rows), "views": summary["view_count"],
                      "mask_pixels": total_pixels, "output": str(output_root)}, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-influence", type=Path)
    p.add_argument("--transition-audit", action="store_true")
    a = p.parse_args()
    if a.transition_audit:
        run_transition_audit(a.input, a.output)
    else:
        run(a.input, a.output, a.reference_influence)


if __name__ == "__main__":
    main()
