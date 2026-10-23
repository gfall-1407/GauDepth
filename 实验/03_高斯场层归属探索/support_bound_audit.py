"""Bound support maxima for a fixed set of saved rays; no GT or renderer access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

TOL_SCORE = 1e-6
MAX_INTERVALS = 200_000
MAX_SECONDS_PER_RAY = 10.0
MAX_ITER = 64
TOL_MM = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def support(z: np.ndarray, p: np.ndarray, d: float, h: float) -> float:
    return float(np.sum(p * np.exp(-0.5 * ((z - d) / h) ** 2), dtype=np.float64))


def upper_bound(z: np.ndarray, p: np.ndarray, a: float, b: float, h: float) -> float:
    """Componentwise interval supremum, with outward floating-point padding."""
    distance = np.maximum(np.maximum(a - z, z - b), 0.0)
    terms = p * np.exp(-0.5 * (distance / h) ** 2)
    total = math.fsum(map(float, terms))
    pad = 32.0 * np.finfo(np.float64).eps * (len(terms) + 1) * max(1.0, total)
    return float(np.nextafter(total + pad, math.inf))


def self_test() -> None:
    # Degenerate single-mode interval and a hand-checkable symmetric bimodal bound.
    z1 = np.array([2.0]); p1 = np.array([1.0])
    assert upper_bound(z1, p1, 2.0, 2.0, 1.0) >= support(z1, p1, 2.0, 1.0)
    z2 = np.array([-2.0, 2.0]); p2 = np.array([0.5, 0.5])
    u = upper_bound(z2, p2, -1.0, 1.0, 1.0)
    assert u >= math.exp(-0.5)
    for d in np.linspace(-1.0, 1.0, 1001):
        assert support(z2, p2, float(d), 1.0) <= u
    # Outside-domain interval also obeys the same bound.
    u_out = upper_bound(z2, p2, 3.0, 4.0, 1.0)
    assert all(support(z2, p2, float(d), 1.0) <= u_out for d in np.linspace(3, 4, 101))


def weighted_median(z: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(z, kind="stable")
    cumulative = np.cumsum(w[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, 0.5 * float(w.sum(dtype=np.float64)), side="left"))
    return float(z[order[min(index, len(order) - 1)]])


def find_modes(z: np.ndarray, p: np.ndarray, h: float):
    starts = list(dict.fromkeys(map(float, z)))
    solutions = []
    failed = 0
    for seed_index, start in enumerate(starts):
        center = start
        for iteration in range(1, MAX_ITER + 1):
            mass = p * np.exp(-0.5 * ((z - center) / h) ** 2)
            denominator = float(mass.sum(dtype=np.float64))
            if not (denominator > 0 and math.isfinite(denominator)):
                break
            nxt = float(np.dot(mass, z) / denominator)
            if abs(nxt - center) <= TOL_MM:
                solutions.append((support(z, p, nxt, h), seed_index, nxt, iteration))
                break
            center = nxt
        else:
            failed += 1
            continue
        if not solutions or solutions[-1][1] != seed_index:
            failed += 1
    unique = []
    for item in sorted(solutions, key=lambda row: row[1]):
        if not any(abs(item[2] - prior[2]) <= 5 * TOL_MM for prior in unique):
            unique.append(item)
    ranked = sorted(unique, key=lambda row: (-row[0], row[1]))
    best = ranked[0] if ranked else None
    second = next((row for row in ranked[1:] if abs(row[2] - best[2]) >= 2.0 * h), None) if best else None
    return best, second, failed, len(solutions), len(unique), len(starts)


def search(z: np.ndarray, p: np.ndarray, h: float, candidates: dict[str, float]):
    start_time = time.monotonic()
    lo, hi = float(z.min()), float(z.max())
    scores = {name: support(z, p, depth, h) for name, depth in candidates.items() if math.isfinite(depth)}
    lower_name = max(scores, key=scores.get)
    lower = scores[lower_name]
    lower_depth = candidates[lower_name]
    midpoint = (lo + hi) / 2.0
    mid_score = support(z, p, midpoint, h)
    if mid_score > lower:
        lower_name, lower, lower_depth = "domain_midpoint", mid_score, midpoint
    intervals = 1
    root_u = upper_bound(z, p, lo, hi, h)
    heap = [(-root_u, lo, hi)]
    status = "unknown"
    reason = ""
    while heap:
        upper = -heap[0][0]
        gap = upper - lower
        if gap <= TOL_SCORE:
            status, reason = "certified_score", "upper_minus_lower<=1e-6"
            break
        if time.monotonic() - start_time >= MAX_SECONDS_PER_RAY:
            reason = "10s_limit"
            break
        if intervals + 2 > MAX_INTERVALS:
            reason = "200000_interval_limit"
            break
        _, a, b = heap[0]
        mid = a + (b - a) / 2.0
        if not (a < mid < b):
            reason = "degenerate_interval"
            break
        heapq.heappop(heap)
        value = support(z, p, mid, h)
        if value > lower:
            lower_name, lower, lower_depth = "subinterval_midpoint", value, mid
        heapq.heappush(heap, (-upper_bound(z, p, a, mid, h), a, mid))
        heapq.heappush(heap, (-upper_bound(z, p, mid, b, h), mid, b))
        intervals += 2
    upper = -heap[0][0] if heap else lower
    gap = max(0.0, upper - lower)
    return dict(status=status, reason=reason, lower_score=lower, lower_depth_mm=lower_depth,
                lower_source=lower_name, upper_score=upper, certified_score_gap=gap, interval_count=intervals,
                elapsed_seconds=time.monotonic() - start_time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="20261022_夜间 directory")
    parser.add_argument("--inputs-root", type=Path, required=True, help="一致深度支持/inputs directory")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent result directory")
    args = parser.parse_args()
    self_test()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    with args.support_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    h1 = [r for r in rows if float(r["support_ratio_R_h1mm"]) > 1.0 and r["mode_status_h1mm"] == "partial"]
    fixed = {("65", "0032", "202")}
    h2 = [r for r in rows if (r["scan"], r["view"], r["ray"]) in fixed]
    if len(h1) != 13 or len(h2) != 1 or h2[0]["mode_status_h2mm"] != "partial":
        raise ValueError(f"Fixed selection mismatch: h1={len(h1)}, h2={len(h2)}")
    tasks = [(r, 1.0) for r in h1] + [(h2[0], 2.0)]
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "support_bound.csv"
    fields = ("scan", "view", "ray", "pixel_x", "pixel_y", "h_mm", "members", "scale_to_mm", "expected_mm", "weighted_median_mm",
              "saved_mode_mm", "saved_mode_status", "saved_mode_score", "saved_mode_score_le_final_lower", "lower_depth_mm", "lower_source",
              "lower_score", "upper_score", "certified_score_gap", "status", "reason", "interval_count", "elapsed_seconds",
              "original_expected_support", "original_mode_support", "original_relative_support_R", "peak_location_certified")
    results = []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); f.flush()
        for source_row, h in tasks:
            scan, view, ray = int(source_row["scan"]), source_row["view"], int(source_row["ray"])
            npz_path = args.raw_root / f"scan{scan}" / f"view_{int(view):03d}.npz"
            meta_path = args.raw_root / f"scan{scan}" / "metadata.json"
            camera_path = args.inputs_root / f"scan{scan}" / "cameras.npz"
            with np.load(npz_path) as data:
                left, right = map(int, data["offsets"][ray:ray + 2])
                z_scene = data["depth"][left:right].astype(np.float64)
                w = data["weight"][left:right].astype(np.float64)
                px, py = map(int, data["pixel_xy"][ray])
            if (px, py) != (int(source_row["x"]), int(source_row["y"])):
                raise ValueError(f"Pixel identity mismatch scan{scan}/{view}/ray{ray}")
            view_meta = json.loads(meta_path.read_text(encoding="utf-8"))["views"]
            saved_view = next(v for v in view_meta if v["image_name"] == view)
            if sha256(npz_path) != saved_view["npz_sha256"]:
                raise ValueError(f"Raw NPZ SHA mismatch scan{scan}/{view}")
            with np.load(camera_path) as camera:
                transform = camera["scale_mat_0"].astype(np.float64)
            scale = float(transform[0, 0])
            if scale <= 0 or not np.allclose(transform[:3, :3], np.eye(3) * scale, atol=1e-6, rtol=0):
                raise ValueError(f"Unexpected scale_mat for scan{scan}")
            if not len(w) or not np.isfinite(w).all() or np.any(w <= 0) or not np.isfinite(z_scene).all():
                raise ValueError(f"Invalid raw ray scan{scan}/{view}/ray{ray}")
            z = z_scene * scale
            p = w / float(w.sum(dtype=np.float64))
            expected = float(np.dot(w, z) / w.sum(dtype=np.float64))
            if abs(expected - float(source_row["expected"])) > 1e-8:
                raise ValueError(f"Raw expected mismatch scan{scan}/{view}/ray{ray}")
            median = weighted_median(z, w)
            mode_field = f"kernel_mode_{int(h)}mm"
            status_field = f"mode_status_h{int(h)}mm"
            support_expected_field = f"support_expected_h{int(h)}mm"
            support_mode_field = f"support_mode_h{int(h)}mm"
            mode = float(source_row[mode_field])
            best, second, failed, converged_starts, clusters, starts = find_modes(z, p, h)
            if best is None:
                raise ValueError(f"No saved raw mode scan{scan}/{view}/ray{ray}/h{h}")
            candidates = {"expected": expected, "weighted_median": median, "saved_mode": mode,
                          "domain_midpoint": (float(z.min()) + float(z.max())) / 2.0}
            result = search(z, p, h, candidates)
            saved_score = support(z, p, mode, h)
            if abs(saved_score - float(source_row[support_mode_field])) > 1e-10:
                raise ValueError(f"Saved mode score mismatch scan{scan}/{view}/ray{ray}/h{h}")
            result.update(scan=scan, view=view, ray=ray, pixel_x=px, pixel_y=py, h_mm=h, members=len(z), scale_to_mm=scale,
                          expected_mm=expected, weighted_median_mm=median, saved_mode_mm=mode,
                          saved_mode_status=source_row[status_field], saved_mode_score=saved_score,
                          saved_mode_score_le_final_lower=bool(saved_score <= result["lower_score"] + 1e-15),
                          original_expected_support=float(source_row[support_expected_field]),
                          original_mode_support=float(source_row[support_mode_field]),
                          original_relative_support_R=float(source_row[f"support_ratio_R_h{int(h)}mm"]),
                          interval_count=result["interval_count"], converged_starts=converged_starts,
                          failed_starts=failed, solution_clusters=clusters, starts=starts,
                          input_npz_sha256=sha256(npz_path), input_cameras_sha256=sha256(camera_path))
            results.append(result)
            writer.writerow({k: result.get(k, "") for k in fields}); f.flush()
            print(f"scan{scan}/{view}/ray{ray} h={h:g}: {result['status']} gap={result['certified_score_gap']:.3g} intervals={result['interval_count']} elapsed={result['elapsed_seconds']:.2f}s", flush=True)
    manifest_results = []
    for result in results:
        item = {k: v for k, v in result.items() if k not in ("scan", "view", "ray")}
        item.update(scan=result["scan"], view=result["view"], ray=result["ray"])
        manifest_results.append(item)
    manifest = {
        "schema": "support_bound_audit_v1", "status": "complete", "self_test": "passed",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                        check=True, capture_output=True, text=True).stdout.strip(),
        "script_sha256": sha256(Path(__file__)),
        "selection": {"h1": "support_ratio_R_h1mm>1 and mode_status_h1mm=partial (fixed source CSV; exactly 13)",
                      "h2": "scan65/view0032/ray202 (fixed; partial)", "count": len(results)},
        "method": {"support": "sum((w/sum(w))*exp(-0.5*((z_mm-d)/h_mm)^2))",
                    "upper": "sum(p_i*exp(-dist(z_i,[a,b])^2/(2*h^2))) with outward floating-point padding",
                    "lower_candidates": ["expected", "weighted_median", "saved_mode", "interval_midpoint"],
                    "split": "maximum-U interval first; binary midpoint split",
                    "stop": {"score_gap": TOL_SCORE, "max_intervals_per_ray": MAX_INTERVALS, "max_seconds_per_ray": MAX_SECONDS_PER_RAY},
                    "peak_position_certified": False, "GT_used": False},
        "inputs": {"support_csv": str(args.support_csv.resolve()), "support_csv_sha256": sha256(args.support_csv),
                   "raw_root": str(args.raw_root.resolve()), "inputs_root": str(args.inputs_root.resolve())},
        "results": manifest_results,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
