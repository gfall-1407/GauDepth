"""Pool completed per-view influence summaries without treating views as independent scans."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METHODS = ("all", "top10pct_count", "top50pct_count", "cover90pct_weight",
           "without_far_high_group", "without_far_high_single")
SCOPES = ("all_mask", "common_observation")
MASS_EDGES = (0.05, 0.1, 0.25, 0.5)
SEPARATION_EDGES = (0.5, 1.0, 2.0, 4.0)
ACCURACY_FIELDS = ("error_mm", "delta_error_mm", "axial_shift_mm", "euclidean_shift_mm")
COUNTS = ("improved", "worsened", "tied", "over20mm", "within1mm", "missing")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merge_moments(items):
    """Merge sufficient moments exactly by n; never average view means."""
    n = sum(int(m["n"]) for m in items)
    if n == 0:
        return dict(n=0, sum=0.0, sum_square=0.0, min=None, max=None)
    return dict(n=n, sum=math.fsum(float(m["sum"]) for m in items),
                sum_square=math.fsum(float(m["sum_square"]) for m in items),
                min=min(float(m["min"]) for m in items if int(m["n"])),
                max=max(float(m["max"]) for m in items if int(m["n"])))


def moment_stats(m):
    n = m["n"]
    if not n:
        return None, None, None, None
    mean = m["sum"] / n
    second = m["sum_square"] / n
    variance = second - mean * mean
    tol = 1e-12 * max(abs(second), abs(mean * mean), 1e-300)
    if variance < -tol:
        raise ValueError(f"materially negative variance from moments: {variance}")
    return mean, math.sqrt(max(0.0, variance)), m["min"], m["max"]


def _read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def discover(root, verify_all=False):
    scans, skipped = [], []
    for scan_dir in sorted(Path(root).glob("scan*")):
        meta_path = scan_dir / "metadata.json"
        if not meta_path.is_file():
            skipped.append(dict(scan=scan_dir.name, status="missing_metadata"))
            continue
        meta = _read_json(meta_path)
        status = meta.get("status")
        if status != "complete":
            skipped.append(dict(scan=scan_dir.name, status=status or "missing_status"))
            continue
        names = meta.get("camera_names")
        views = meta.get("views")
        if not isinstance(names, list) or not isinstance(views, list) or len(names) != len(views):
            raise ValueError(f"{scan_dir}: complete metadata camera_names/views mismatch")
        if len(set(names)) != len(names):
            raise ValueError(f"{scan_dir}: duplicate camera names")
        by_name = {v.get("name"): v for v in views}
        if len(by_name) != len(views) or set(by_name) != set(names):
            raise ValueError(f"{scan_dir}: completed views do not match full camera list")
        loaded = []
        for index, name in enumerate(names):
            view = by_name[name]
            if int(view.get("index", -1)) != index:
                raise ValueError(f"{scan_dir}/{name}: view index/order mismatch")
            summary_name = view.get("summary")
            if not summary_name:
                raise ValueError(f"{scan_dir}/{name}: missing summary path")
            summary_path = scan_dir / summary_name
            registry = {f["name"]: f["sha256"] for f in view.get("files", [])}
            expected = registry.get(summary_name)
            if expected is None or sha256(summary_path) != expected:
                raise ValueError(f"{scan_dir}/{name}: summary hash mismatch")
            if verify_all:
                for filename, digest in registry.items():
                    path = scan_dir / filename
                    if not path.is_file() or sha256(path) != digest:
                        raise ValueError(f"{scan_dir}/{name}: file hash mismatch: {filename}")
            summary = _read_json(summary_path)
            if summary.get("name") != name:
                raise ValueError(f"{summary_path}: camera name mismatch")
            loaded.append(summary)
        scans.append(dict(name=scan_dir.name, metadata=meta, views=loaded))
    if not scans:
        raise ValueError("No scan with metadata status=complete was found")
    return scans, skipped


def _empty_acc():
    return {field: [] for field in ACCURACY_FIELDS}, {field: 0 for field in COUNTS}


def _add_accuracy(acc, scope, method, view):
    comp = view.get("comparison", {}).get(scope, {}).get(method)
    if comp is None:
        raise ValueError(f"view {view.get('name')}: missing {scope}/{method}")
    mlist, counts = acc.setdefault((scope, method), _empty_acc())
    for field in ACCURACY_FIELDS:
        mlist[field].append(comp[field])
    for field in COUNTS:
        counts[field] += int(comp[field])


def _acc_row(scan, scope, method, acc):
    moments, counts = acc
    merged = {k: merge_moments(v) for k, v in moments.items()}
    error_mean = moment_stats(merged["error_mm"])[0]
    delta_mean = moment_stats(merged["delta_error_mm"])[0]
    axial_mean = moment_stats(merged["axial_shift_mm"])[0]
    euclidean_mean = moment_stats(merged["euclidean_shift_mm"])[0]
    return dict(scan=scan, scope=scope, method=method,
        finite_n=merged["error_mm"]["n"], paired_n=merged["delta_error_mm"]["n"],
        mean_mm=error_mean, mean_delta_mm=delta_mean,
        improved=counts["improved"], worsened=counts["worsened"], tied=counts["tied"],
        over20mm=counts["over20mm"], within1mm=counts["within1mm"], missing=counts["missing"],
        axial_shift_mean_mm=axial_mean, euclidean_shift_mean_mm=euclidean_mean)


def _collect_accuracy(views):
    acc = {}
    for view in views:
        for scope in SCOPES:
            for method in METHODS:
                _add_accuracy(acc, scope, method, view)
    return acc


def _collect_mechanism(views):
    cells = {}
    for view in views:
        for item in view.get("comparison", {}).get("mechanism_bins", []):
            key = (item["group"], int(item["mass_bin"]), int(item["separation_bin"]))
            c = cells.setdefault(key, dict(moments=[], improved=0, worsened=0, tied=0))
            c["moments"].append(item["delta_error_mm"])
            for label in ("improved", "worsened", "tied"):
                c[label] += int(item[label])
    return cells


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(input_root, output_root, verify_all=False):
    scans, skipped = discover(input_root, verify_all)
    methods_by_scan = {}
    statistics, accuracy, mechanisms = [], [], []
    summary_scans = []
    pooled_stat_items, pooled_acc_views, pooled_mech_views = {}, [], []
    for scan in scans:
        views = scan["views"]
        meta_methods = tuple(scan["metadata"].get("methods", METHODS))
        if meta_methods != METHODS:
            raise ValueError(f"{scan['name']}: unexpected method schema {meta_methods}")
        methods_by_scan[scan["name"]] = meta_methods
        stat_names = sorted(set(k for v in views for k in v.get("moments", {})))
        for stat in stat_names:
            items = [v.get("moments", {}).get(stat, dict(n=0,sum=0.,sum_square=0.,min=None,max=None)) for v in views]
            pooled_stat_items.setdefault(stat, []).append(merge_moments(items))
            m = merge_moments(items)
            mean, sd, low, high = moment_stats(m)
            statistics.append(dict(scan=scan["name"], stat=stat, n=m["n"], mean=mean,
                                   population_sd=sd, min=low, max=high))
        acc = _collect_accuracy(views)
        for (scope, method), values in sorted(acc.items()):
            accuracy.append(_acc_row(scan["name"], scope, method, values))
        cells = _collect_mechanism(views)
        for (group, mb, sb), cell in sorted(cells.items()):
            m = merge_moments(cell["moments"])
            mechanisms.append(dict(scan=scan["name"], group=group, mass_bin=mb,
                separation_bin=sb, n=m["n"], mean_delta_mm=moment_stats(m)[0],
                improved=cell["improved"], worsened=cell["worsened"], tied=cell["tied"]))
        pooled_acc_views.extend(views)
        pooled_mech_views.extend(views)
        per_method = {}
        for method in METHODS:
            row = next(r for r in accuracy if r["scan"] == scan["name"] and r["scope"] == "all_mask" and r["method"] == method)
            per_method[method] = dict(mean_delta_mm=row["mean_delta_mm"], improved=row["improved"],
                                      worsened=row["worsened"], tied=row["tied"], paired_n=row["paired_n"])
        summary_scans.append(dict(scan=scan["name"], views=len(views), cameras=list(scan["metadata"]["camera_names"]),
                                  per_method_all_mask=per_method))
    for stat, by_scan in sorted(pooled_stat_items.items()):
        m = merge_moments(by_scan); mean, sd, low, high = moment_stats(m)
        statistics.append(dict(scan="pooled", stat=stat, n=m["n"], mean=mean,
                               population_sd=sd, min=low, max=high))
    pooled_acc = _collect_accuracy(pooled_acc_views)
    for (scope, method), values in sorted(pooled_acc.items()):
        accuracy.append(_acc_row("pooled", scope, method, values))
    pooled_cells = _collect_mechanism(pooled_mech_views)
    for (group, mb, sb), cell in sorted(pooled_cells.items()):
        m = merge_moments(cell["moments"])
        mechanisms.append(dict(scan="pooled", group=group, mass_bin=mb,
            separation_bin=sb, n=m["n"], mean_delta_mm=moment_stats(m)[0],
            improved=cell["improved"], worsened=cell["worsened"], tied=cell["tied"]))
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "statistics.csv", statistics)
    _write_csv(output_root / "accuracy.csv", accuracy)
    _write_csv(output_root / "mechanism_bins.csv", mechanisms)
    payload = dict(input_root=str(Path(input_root).resolve()), scans=[s["name"] for s in scans],
        skiplist=skipped, partial=bool(skipped), view_count=sum(len(s["views"]) for s in scans),
        scan_count=len(scans), verify_all=bool(verify_all), methods=list(METHODS),
        scopes=list(SCOPES), mechanism_bins=dict(mass_edges=list(MASS_EDGES),
            separation_edges=list(SEPARATION_EDGES), mass_rule="numpy.digitize(right=False)",
            separation_rule="numpy.digitize(right=False)",
            separation="abs(group mean - all mean) / all-member weighted SD; zero SD is unknown"),
        pooling="View sufficient moments are pooled by their finite n; scan-pooled rows are pixel weighted. Views/pixels are not independent scans.",
        accuracy_interpretation="Nearest distance to DTU reference points, not pixelwise GT depth MAE; negative and missing values are retained. Paired n may differ by method.",
        per_scan_all_mask={s["scan"]:s["per_method_all_mask"] for s in summary_scans},
        summary_scans=summary_scans)
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Root with scan*/metadata.json")
    p.add_argument("--output", type=Path, required=True, help="New or empty aggregate output root")
    p.add_argument("--verify-all", action="store_true", help="Hash every sample/map file as well as summaries")
    a = p.parse_args()
    result = aggregate(a.input, a.output, a.verify_all)
    print(json.dumps({k: result[k] for k in ("scans", "skiplist", "partial", "view_count", "scan_count")}, indent=2))


if __name__ == "__main__":
    main()
