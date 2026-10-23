"""Reproduce one frozen model's native median TSDF and original DTU evaluation."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from depth_accuracy import SCANS, load_scene
from sample import forward, protect_scene, sha256


def _sha(path):
    return sha256(Path(path))


def _git_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Research repository must be clean before starting the pilot")
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        remote=git("remote", "get-url", "origin"))


def _read_cfg(path):
    node = ast.parse(Path(path).read_text(), mode="eval").body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "Namespace" and not node.args):
        raise ValueError(f"Unexpected cfg_args format: {path}")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def _inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _preflight(scan, model_root, source_root, gt_root, output, protected_roots=()):
    model = model_root / f"scan{scan}"
    source = source_root / f"scan{scan}"
    required = [model / "cfg_args", model / "point_cloud/iteration_30000/point_cloud.ply",
        source / "cameras.npz", gt_root / "Points/stl" / f"stl{scan:03d}_total.ply",
        gt_root / "ObsMask" / f"ObsMask{scan}_10.mat", gt_root / "ObsMask" / f"Plane{scan}.mat"]
    required.extend(gt_root / f"Calibration/cal18/pos_{i:03d}.txt" for i in range(1, 65))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required model/source/GT inputs are missing: " + ", ".join(missing))
    if not Path(output).is_absolute():
        raise ValueError("--output must be an absolute independent directory")
    for protected in (model_root, source_root, gt_root, *protected_roots):
        if protected is None:
            continue
        if _inside(output, protected):
            raise ValueError(f"Output must stay outside protected input root: {protected}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return model, source


def _history_paths(reference_root, scan):
    root = Path(reference_root) / f"scan{scan}"
    result_candidates = (root / "vis/results.json", root / "results.json")
    result = next((p for p in result_candidates if p.is_file()), result_candidates[0])
    mesh = root / "recon_post.ply"
    hash_candidates = (root / "recon_post.ply.sha256", root / "recon_post.sha256", root / "mesh_sha256.txt")
    hash_file = next((p for p in hash_candidates if p.is_file()), None)
    return result, mesh if mesh.is_file() else None, hash_file


def _compare_meshes(old_path, old_hash, new_path):
    import open3d as o3d
    new_hash = _sha(new_path)
    if old_path is None:
        return dict(historical_sha256=old_hash, new_sha256=new_hash,
            sha256_identical=old_hash == new_hash,
            historical_vertices=None, new_vertices=int(len(o3d.io.read_triangle_mesh(str(new_path)).vertices)),
            historical_triangles=None, new_triangles=int(len(o3d.io.read_triangle_mesh(str(new_path)).triangles)),
            same_vertex_array=None, max_abs_same_index_vertex_delta=None,
            same_triangle_array=None, vertex_comparison_available=False)
    actual_old_hash = _sha(old_path)
    if old_hash is not None and old_hash != actual_old_hash:
        raise ValueError(f"Historical mesh hash sidecar mismatch: {old_path}")
    old_hash = actual_old_hash
    old = o3d.io.read_triangle_mesh(str(old_path))
    new = o3d.io.read_triangle_mesh(str(new_path))
    ov, nv = np.asarray(old.vertices), np.asarray(new.vertices)
    of, nf = np.asarray(old.triangles), np.asarray(new.triangles)
    same_count = len(ov) == len(nv)
    same_vertex_array = bool(same_count and np.array_equal(ov, nv))
    same_faces = bool(of.shape == nf.shape and np.array_equal(of, nf))
    max_delta = float(np.max(np.abs(ov - nv))) if same_count and len(ov) else (0.0 if same_count else None)
    return dict(historical_sha256=old_hash, new_sha256=new_hash,
        sha256_identical=old_hash == new_hash,
        historical_vertices=int(len(ov)), new_vertices=int(len(nv)),
        historical_triangles=int(len(of)), new_triangles=int(len(nf)),
        same_vertex_array=same_vertex_array, max_abs_same_index_vertex_delta=max_delta,
        same_triangle_array=same_faces, vertex_comparison_available=True)


def _validate_expected_reference(root, scan, model_sha, camera_names):
    if root is None:
        return None
    scan_root = Path(root) / f"scan{scan}"
    meta_path = scan_root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"Expected-statistics scan is not complete: {meta_path}")
    if meta.get("model_sha256") != model_sha:
        raise ValueError("Expected-statistics model hash does not match the loaded model")
    if meta.get("camera_names") != camera_names:
        raise ValueError("Expected-statistics camera list does not match loaded model cameras")
    views = meta.get("views", [])
    if len(views) != len(camera_names):
        raise ValueError("Expected-statistics view count is incomplete")
    summary_hashes = []
    for view in views:
        summary_name = view.get("summary")
        files = {f["name"]: f["sha256"] for f in view.get("files", [])}
        if not summary_name or summary_name not in files:
            raise ValueError("Expected-statistics view lacks a hashed summary")
        summary_path = scan_root / summary_name
        digest = _sha(summary_path)
        if digest != files[summary_name]:
            raise ValueError(f"Expected-statistics summary hash mismatch: {summary_path}")
        summary_hashes.append(dict(name=view["name"], file=summary_name, sha256=digest))
    return dict(root=str(Path(root).resolve()), metadata=str(meta_path.resolve()),
        metadata_sha256=_sha(meta_path), status=meta["status"], git_commit=meta.get("git_commit"),
        model_sha256=model_sha, view_count=len(views), summary_hashes=summary_hashes,
        use="Reference only; no 855-view statistics were recomputed by this pilot.")


def _check_native_render(scan, cameras, gaussians, pipe, dataset, output):
    import torch
    from diff_gaussian_rasterization import _C
    from gaussian_renderer import render
    train = cameras
    if not train:
        raise ValueError("Native mesh extractor has no training cameras")
    camera = train[0]
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        raw, _, _ = forward(camera, gaussians, pipe, background, dataset.kernel_size)
        native = render(camera, gaussians, pipe, background, dataset.kernel_size)
        raw_expected, raw_median = raw[4], raw[5]
        torch.testing.assert_close(raw_expected, native["expected_depth"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(raw_median, native["median_depth"], rtol=2e-5, atol=2e-6)
        pair = torch.isfinite(raw_expected) & torch.isfinite(raw_median)
        delta = (raw_expected - raw_median).abs()[pair]
        distinct = int((delta > 0).sum().item())
        if distinct == 0:
            raise AssertionError("Fixed pilot view has no finite expected/median differences")
        return dict(camera_name=camera.image_name,
            finite_pair_pixels=int(pair.sum().item()), distinct_pixels=distinct,
            max_abs_expected_median=float(delta.max().item()),
            mean_abs_expected_median=float(delta.mean().item()),
            raw_expected_matches_native=True, raw_median_matches_native=True,
            rasterizer_C_path=str(Path(_C.__file__).resolve()),
            rasterizer_C_sha256=_sha(_C.__file__))


def _native_extract(mesh_extract, dataset, pipe, iteration, num_cluster, gaussians, scene):
    body = inspect.getsource(mesh_extract.extract_mesh)
    edits = {
        "gaussians = GaussianModel(dataset.sh_degree)": "gaussians = frozen_gaussians",
        "scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)": "scene = frozen_scene",
    }
    for old, new in edits.items():
        if body.count(old) != 1:
            raise AssertionError(f"Native extractor source mismatch: {old}")
        body = body.replace(old, new)
    scope = vars(mesh_extract).copy()
    scope.update(frozen_gaussians=gaussians, frozen_scene=scene)
    exec(compile(body, str(Path(mesh_extract.__file__).resolve()), "exec"), scope)
    scope["extract_mesh"](dataset, pipe, iteration, num_cluster)


def _native_evaluate(evaluator, dataset, iteration, gt_root, gaussians, scene):
    import torch
    body = inspect.getsource(evaluator.evaluate_mesh)
    runner = """import multiprocessing as mp
import open3d as o3d
import sys
from pathlib import Path

base_pool = mp.Pool
def bounded_pool(*args, **kwargs):
    if not args and "processes" not in kwargs:
        kwargs["processes"] = 8
    return base_pool(*args, **kwargs)
mp.Pool = bounded_pool
o3d.io.write_point_cloud = lambda *args, **kwargs: True

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
main_globals = sys.modules["__main__"].__dict__
main_globals["__file__"] = script
exec(compile(Path(script).read_text(), script, "exec"), main_globals)
"""
    command = "subprocess.run([sys.executable, '-c', " + repr(runner) + ", str(ROOT / 'dtu_eval' / 'eval.py'), '--data', aligned_mesh_file, '--scan', str(scan), '--mode', 'mesh', '--dataset_dir', str(DTU_PATH), '--vis_out_dir', out_dir], check=True, timeout=1200, cwd=ROOT)"
    if body.count("os.system(cmd)") != 1:
        raise AssertionError("Original evaluate_mesh command site changed")
    body = body.replace("os.system(cmd)", command)
    scope = vars(evaluator).copy()
    scope.update(args=SimpleNamespace(DTU=str(gt_root)),
        GaussianModel=lambda _degree: gaussians,
        Scene=lambda *_args, **_kwargs: scene,
        subprocess=subprocess, sys=sys, ROOT=ROOT)
    exec(compile(body, str(Path(evaluator.__file__).resolve()), "exec"), scope)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    scope["evaluate_mesh"](dataset, iteration, str(gt_root))


def _remove_pilot_intermediates(output):
    out = Path(output).resolve()
    removed = []
    for name in ("recon.ply", "recon_aligned.ply"):
        path = out / name
        if path.resolve().parent != out or not _inside(path, out):
            raise RuntimeError(f"Refusing to remove a path outside this pilot output: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Expected this scan's generated intermediate: {path}")
        record = dict(name=name, sha256=_sha(path), size_bytes=path.stat().st_size)
        path.unlink()
        record["removed"] = True
        removed.append(record)
    return removed


def run(args):
    repo = _git_provenance()
    import torch
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C
    import mesh_extract
    import evaluate_dtu_mesh
    from utils.general_utils import safe_state

    model_root, source_root, gt_root = args.model_root, args.source_root, args.gt_root
    model_path, source_path = _preflight(args.scan, model_root, source_root, gt_root, args.output,
        (args.reference_root, args.expected_statistics_root))
    out = args.output.resolve()
    config = _read_cfg(model_path / "cfg_args")
    cfg_ratio = float(config.get("depth_ratio", float("nan")))
    if not np.isfinite(cfg_ratio) or abs(cfg_ratio - 0.6) > 1e-12:
        raise ValueError(f"Expected original G2 depth_ratio=0.6, got {config.get('depth_ratio')}")
    old_results_path, old_mesh_path, old_hash_path = _history_paths(args.reference_root or model_root, args.scan)
    if not old_results_path.is_file():
        raise FileNotFoundError("Historical vis/results.json is required via --reference-root or --model-root/scanN")
    if old_mesh_path is None and old_hash_path is None:
        raise FileNotFoundError("Historical recon_post.ply or its SHA256 sidecar is required")
    historical_results = json.loads(old_results_path.read_text(encoding="utf-8"))
    historical_mesh_sha = _sha(old_mesh_path) if old_mesh_path is not None else old_hash_path.read_text().split()[0]
    if len(historical_mesh_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in historical_mesh_sha):
        raise ValueError("Historical mesh reference must contain a SHA256 digest")

    # Preserve the native seed setup from mesh_extract.py; model and Scene writes
    # are guarded by load_scene/protect_scene and all outputs use the new root.
    safe_state(False)
    dataset, pipe, gaussians, cameras, ply, model_sha = load_scene(
        args.scan, out, model_root, source_root)
    loaded_ratio = float(dataset.depth_ratio)
    if not np.isfinite(loaded_ratio) or abs(loaded_ratio - 0.6) > 1e-12:
        raise ValueError(f"Loaded dataset did not retain depth_ratio=0.6: {dataset.depth_ratio}")
    camera_names = [c.image_name for c in cameras]
    expected_reference = _validate_expected_reference(
        args.expected_statistics_root, args.scan, model_sha, camera_names)
    # Native extraction consumes getTrainCameras only. The loader's eval setting
    # remains exactly as stored in cfg_args; test cameras are never promoted.
    from scene import Scene
    from sample import protect_scene
    with protect_scene(out):
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    train_cameras = scene.getTrainCameras()
    scene_camera_names = [c.image_name for c in train_cameras + scene.getTestCameras()]
    if scene_camera_names != camera_names:
        raise AssertionError("Scene camera order changed between protected loads")
    if not train_cameras:
        raise ValueError("Loaded Scene has no training cameras")
    with open(source_path / "cameras.npz", "rb") as f:
        source_camera_sha = hashlib.sha256(f.read()).hexdigest()
    gt_hashes = {str(p.relative_to(gt_root)): _sha(p) for p in (
        gt_root / "Points/stl" / f"stl{args.scan:03d}_total.ply",
        gt_root / "ObsMask" / f"ObsMask{args.scan}_10.mat",
        gt_root / "ObsMask" / f"Plane{args.scan}.mat")}
    gt_hashes.update({f"Calibration/cal18/pos_{i:03d}.txt": _sha(gt_root / f"Calibration/cal18/pos_{i:03d}.txt") for i in range(1,65)})
    # Verify the retained raw _C outputs and the renderer wrapper on one fixed view.
    check = _check_native_render(args.scan, train_cameras, gaussians, pipe, dataset, out)
    # The output path is changed only through the dataset proxy. Native TSDF
    # rendering, including the cfg-selected median_depth, remains untouched.
    dataset_fields = vars(dataset).copy()
    dataset_fields.update(model_path=str(out), source_path=str(source_path))
    native_dataset = SimpleNamespace(**dataset_fields)
    with torch.no_grad():
        _native_extract(mesh_extract, native_dataset, pipe, 30000, 1, gaussians, scene)
    new_mesh = out / "recon_post.ply"
    if not new_mesh.is_file():
        raise FileNotFoundError("Native extractor did not produce recon_post.ply")
    mesh_comparison = _compare_meshes(old_mesh_path, historical_mesh_sha, new_mesh)
    with torch.no_grad():
        _native_evaluate(evaluate_dtu_mesh, native_dataset, 30000, gt_root, gaussians, scene)
    new_results_path = out / "vis" / "results.json"
    if not new_results_path.is_file():
        raise FileNotFoundError("Original DTU evaluator did not write vis/results.json")
    new_results = json.loads(new_results_path.read_text(encoding="utf-8"))
    if _sha(ply) != model_sha:
        raise RuntimeError("Original frozen model PLY changed during the pilot")
    removed_intermediates = _remove_pilot_intermediates(out)
    meta = dict(status="complete", scan=args.scan, expected_instance_id=args.expected_instance_id,
        actual_hostname=os.uname().nodename, git=repo,
        model_path=str(model_path.resolve()), model_sha256=model_sha,
        cfg_path=str((model_path / "cfg_args").resolve()), cfg_sha256=_sha(model_path / "cfg_args"),
        cfg_depth_ratio=config["depth_ratio"], cfg_eval=bool(config.get("eval", False)),
        source_path=str(source_path.resolve()), source_cameras_npz_sha256=source_camera_sha,
        gt_root=str(gt_root.resolve()), gt_inputs_sha256=gt_hashes,
        installed_rasterizer_C_path=str(Path(_C.__file__).resolve()),
        installed_rasterizer_C_sha256=_sha(_C.__file__),
        host_mesh_extract_sha256=_sha(mesh_extract.__file__),
        host_evaluate_dtu_mesh_sha256=_sha(evaluate_dtu_mesh.__file__),
        host_dtu_eval_sha256=_sha(ROOT / "dtu_eval/eval.py"),
        camera_names=camera_names, train_camera_names=[c.image_name for c in train_cameras],
        test_camera_names=[c.image_name for c in scene.getTestCameras()],
        native_depth_name="median_depth", fixed_view_check=check,
        historical=dict(results_path=str(old_results_path.resolve()), results=historical_results,
            mesh_path=str(old_mesh_path.resolve()) if old_mesh_path else None,
            mesh_sha256=historical_mesh_sha,
            mesh_sha256_sidecar=str(old_hash_path.resolve()) if old_hash_path else None),
        native=dict(results_path=str(new_results_path.resolve()), results=new_results,
            mesh_path=str(new_mesh.resolve()), mesh_sha256=_sha(new_mesh)),
        mesh_comparison=mesh_comparison, expected_statistics_reference=expected_reference,
        evaluation_protocol="Original evaluate_dtu_mesh.evaluate_mesh Calibration camera-center alignment and unchanged dtu_eval/eval.py scoring defaults; subprocess timeout 1200s; temporary multiprocessing.Pool wrapper limits default worker count to 8; default_rng remains unseeded; evaluator visualization point-cloud writes are disabled (write_point_cloud returns True).",
        evaluator_visualization_point_cloud_writes=False,
        removed_pilot_intermediates=removed_intermediates,
        writes="recon_post.ply and scores are retained; evaluator point-cloud visualizations are not written; after successful comparison and scoring, this scan's recon.ply and recon_aligned.ply are recorded by SHA256/size and removed.")
    (out / "native_median_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=int, choices=SCANS, required=True)
    p.add_argument("--model-root", type=Path, default=Path("/root/autodl-tmp/RaDe-GS/outtest"))
    p.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/DTU"))
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-root", type=Path,
        help="Read-only history root with scanN/vis/results.json and scanN/recon_post.ply; defaults to --model-root")
    p.add_argument("--expected-statistics-root", type=Path,
        help="Optional completed C3 scan root, referenced by model/camera/hash only")
    p.add_argument("--expected-instance-id", default="8209458d70-897b4075",
        help="Full expected AutoDL instance ID; the external guard must verify it")
    args = p.parse_args()
    result = run(args)
    print(json.dumps(dict(scan=result["scan"], model_sha256=result["model_sha256"],
        historical_overall=result["historical"]["results"].get("overall"),
        native_overall=result["native"]["results"].get("overall"),
        mesh_sha256_identical=result["mesh_comparison"]["sha256_identical"]), indent=2))


if __name__ == "__main__":
    main()
