"""Fixed G1 dense 1mm Huber QA and paired tracked-native TSDF/Calibration checks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from sample import atomic_json, forward, git, sha256
from depth_accuracy import load_scene, coordinate_check
from depth_holdout import huber_exact
from pixel_contribution import retained_mask
from night_depth_batch import FIXED

HOST_SHA = "f97dcd66f55d9ddbf673c1d467238fe6f983b9f0bf062645a87bb12d20aa91df"
SEED = 20261026
CHUNK = 32768
CUDA_FLAGS = ["-O3", "--use_fast_math"]
METHODS = ("native_expected", "huber_1mm")


def cfg_values(path):
    node = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace" or node.args:
        raise ValueError("Expected literal Namespace cfg_args")
    return {item.arg: ast.literal_eval(item.value) for item in node.keywords}


def source_provenance():
    if git("status", "--short") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require committed clean source and the approved origin")
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                remote=git("remote", "get-url", "origin"))


def verify_originals(metadata, ply, cfg):
    metadata.update(original_model_unchanged=sha256(ply) == metadata["model_sha256"],
                    original_cfg_unchanged=sha256(cfg) == metadata["cfg_sha256"])
    if not metadata["original_model_unchanged"] or not metadata["original_cfg_unchanged"]:
        raise ValueError("Protected original model/cfg changed")


def verify_input_reference(args):
    path = HERE / "G1_Huber全场景输入参考.json"
    reference = json.loads(path.read_text(encoding="utf-8"))["scans"][str(args.scan)]
    if (reference["model_sha256"], reference["cfg_sha256"]) != FIXED[args.scan][:2]:
        raise ValueError("Reference is not for the fixed G1 model")
    for root, inputs in ((args.source_root, reference["source_inputs"]),
                         (args.gt_root, reference["gt_inputs"])):
        for relative, digest in inputs.items():
            if sha256(root / relative) != digest:
                raise ValueError(f"Original G1/canonical DTU input mismatch: {relative}")
    return reference, sha256(path)


def verify_camera_reference(cameras, reference):
    if [c.image_name for c in cameras] != [v["name"] for v in reference["views"]]:
        raise ValueError("Loaded camera names/order differ from original G1")
    for camera, prior in zip(cameras, reference["views"]):
        size = [camera.image_height, camera.image_width]
        if size != prior["image_size"] or camera.gt_mask is None:
            raise ValueError("Original image resolution/mask missing or changed")
        mask = retained_mask(camera.gt_mask.cpu().numpy(), tuple(size))
        if hashlib.sha256(mask.tobytes()).hexdigest() != prior["mask_sha256"]:
            raise ValueError(f"Original G1 boolean mask changed at {camera.image_name}")
        if "focal" in prior:
            tx, ty = math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5)
            focal = [float(np.float32(np.float32(size[1])/(np.float32(2)*np.float32(tx)))),
                     float(np.float32(np.float32(size[0])/(np.float32(2)*np.float32(ty))))]
            np.testing.assert_array_equal(focal, prior["focal"])
            np.testing.assert_allclose(camera.world_view_transform.cpu().numpy(),
                                       prior["world_view_transform"], rtol=0, atol=1e-6)
    return dict(status="PASS", views=len(cameras), mask="exact bool bytes",
                focal="old4 exact loaded float32 focal; new11 exact source COLMAP camera SHA")


def extract_source(source):
    edits = {
        "gaussians = GaussianModel(dataset.sh_degree)": "gaussians = frozen_gaussians",
        "scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)": "scene = frozen_scene",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f"Tracked extract_mesh changed at {old}")
        source = source.replace(old, new)
    return source


EVALUATION_RUNNER = r'''
import multiprocessing as mp
import random
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
base_pool = mp.Pool
def bounded_pool(*args, **kwargs):
    if not args and "processes" not in kwargs:
        kwargs["processes"] = 8
    return base_pool(*args, **kwargs)
mp.Pool = bounded_pool
o3d.io.write_point_cloud = lambda *args, **kwargs: True
random.seed(20261026)
np.random.seed(20261026)
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
source = Path(script).read_text(encoding="utf-8")
if source.count("np.random.default_rng()") != 1:
    raise ValueError("Expected one original evaluation shuffle RNG")
source = source.replace("np.random.default_rng()", "np.random.default_rng(20261026)")
scope = sys.modules["__main__"].__dict__
scope["__file__"] = script
exec(compile(source, script, "exec"), scope)
'''


def evaluate_source(source):
    if source.count("os.system(cmd)") != 1:
        raise ValueError("Tracked evaluate_mesh subprocess site changed")
    command = ("subprocess.run([sys.executable, '-c', EVALUATION_RUNNER, "
               "str(ROOT / 'dtu_eval/eval.py'), '--data', aligned_mesh_file, '--scan', str(scan), "
               "'--mode', 'mesh', '--dataset_dir', str(DTU_PATH), '--vis_out_dir', out_dir], "
               "check=True, timeout=1200, cwd=ROOT)")
    return source.replace("os.system(cmd)", command)


def gpu_synthetic_check(extension):
    import torch
    records = torch.zeros((5, 2, 8), dtype=torch.float32, device="cuda")
    records[:, :, 0] = torch.tensor([[2., 0.], [0., 10.], [0., 0.],
                                    [float("nan"), 2.], [1., 2.]], device="cuda")
    records[:, :, 1] = torch.tensor([[1., 0.], [.5, .5], [0., 0.], [.5, .5], [0., .5]], device="cuda")
    counts = torch.tensor([1, 2, 0, 2, 2], dtype=torch.int32, device="cuda")
    depth, status, width = extension.read_huber(records, counts, 1.)
    torch.testing.assert_close(depth[:3], torch.tensor([2., 5., 0.], dtype=torch.float64, device="cuda"), rtol=0, atol=1e-12)
    torch.testing.assert_close(width[:3], torch.tensor([0., 8., 0.], dtype=torch.float64, device="cuda"), rtol=0, atol=1e-12)
    if status.cpu().tolist() != [1, 1, 0, 2, 2] or not bool(torch.isnan(depth[3:]).all()):
        raise AssertionError("CUDA Huber invalid/empty status semantics changed")
    try:
        extension.read_huber(records[:1], torch.tensor([3], dtype=torch.int32, device="cuda"), 1.)
    except RuntimeError:
        pass
    else:
        raise AssertionError("CUDA reader silently accepted counts>K")
    scaled = torch.zeros((1, 3, 8), dtype=torch.float32, device="cuda")
    scaled[0, :, 0] = torch.tensor([0., 0., 10.], device="cuda")
    scaled[0, :, 1] = torch.tensor([.125, .25, .375], device="cuda")
    value, _, interval = extension.read_huber(scaled, torch.tensor([3], dtype=torch.int32, device="cuda"), 1./37.)
    torch.testing.assert_close(value, torch.tensor([5.], dtype=torch.float64, device="cuda"), rtol=0, atol=1e-12)
    torch.testing.assert_close(interval, torch.tensor([10.-2./37.], dtype=torch.float64, device="cuda"), rtol=0, atol=1e-12)
    return dict(status="PASS", tests=["single_point", "nonunique_zero_interval_midpoint",
        "empty_zero_status", "nonfinite_depth_unknown", "zero_weight_unknown", "overflow_rejected",
        "balanced_zero_interval_nonunit_bandwidth"])


def dense_view(camera, gaussians, pipe, dataset, extension, scale, qa_first):
    import torch
    from gaussian_renderer import render

    bg = torch.ones(3, dtype=torch.float32, device="cuda")
    started = time.time()
    raw, fx, fy = forward(camera, gaussians, pipe, bg, dataset.kernel_size)
    _, rgb, alpha, normal, expected, median, _, geom, binning, image = raw
    h, w = camera.image_height, camera.image_width
    if camera.gt_mask is None:
        raise ValueError("Original DTU mask required; opacity is not a replacement")
    mask = retained_mask(camera.gt_mask.cpu().numpy(), (h, w))
    qa = dict(name=camera.image_name, image_size=[h, w], focal=[fx, fy],
        mask_pixels=int(mask.sum()), pixels=h*w, full_frame_replay_max_abs=[0., 0., 0.],
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        capacity_max=256, overflow_retries=0, empty_pixels=0, invalid_pixels=0)
    if qa_first:
        native = render(camera, gaussians, pipe, bg, dataset.kernel_size)
        pairs = ((rgb, native["render"]), (alpha, native["mask"]),
                 (expected, native["expected_depth"]), (median, native["median_depth"]))
        for a, b in pairs:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, equal_nan=True)
        qa["raw_vs_native_max_abs"] = [float((a-b).abs().nan_to_num().max().item()) for a, b in pairs]
        del native
    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(mask.ravel())
    if qa_first and len(eligible) < 64:
        raise ValueError("Expected at least 64 fixed random mask QA pixels")
    qa_pixels = set(map(int, rng.choice(eligible, size=64, replace=False))) if qa_first else set()
    cpu_rows = []
    huber = np.empty(h*w, dtype=np.float64)
    statuses = np.empty(h*w, dtype=np.int32)
    widths = np.empty(h*w, dtype=np.float64)
    targets = torch.stack((expected.reshape(-1), median.reshape(-1), alpha.reshape(-1)), dim=1)
    for start in range(0, h*w, CHUNK):
        stop = min(h*w, start+CHUNK)
        pixels = torch.arange(start, stop, device="cuda", dtype=torch.int32)
        capacity = 256
        while True:
            records, ids, counts, replay = extension.read_rays(geom, binning, image,
                len(gaussians.get_xyz), w, h, fx, fy, pixels, capacity)
            largest = int(counts.max().item())
            if largest <= capacity:
                break
            del records, ids, counts, replay
            capacity *= 2
            qa["overflow_retries"] += 1
            if capacity > 1024:
                raise ValueError(f"Complete-ray capacity exceeded 1024 at {camera.image_name}:{start}")
        qa["capacity_max"] = max(qa["capacity_max"], capacity)
        torch.testing.assert_close(replay, targets[start:stop], rtol=2e-5, atol=2e-6, equal_nan=True)
        discrepancy = (replay-targets[start:stop]).abs().nan_to_num().amax(0).cpu().numpy()
        qa["full_frame_replay_max_abs"] = np.maximum(qa["full_frame_replay_max_abs"], discrepancy).tolist()
        result, status, width = extension.read_huber(records, counts, 1./scale)
        if bool((status == 2).any()) or not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite/invalid accepted records; do not hide them as zero depth")
        huber[start:stop], statuses[start:stop], widths[start:stop] = result.cpu().numpy(), status.cpu().numpy(), width.cpu().numpy()
        for pixel in sorted(qa_pixels.intersection(range(start, stop))):
            row = pixel-start
            n = int(counts[row].item())
            rec = records[row, :n].cpu().numpy().astype(np.float64)
            if not n:
                if result[row].item() != 0 or status[row].item() != 0:
                    raise ValueError("Empty ray output must match native zero depth")
                cpu_rows.append(dict(pixel=pixel, members=0, cpu_mm=None, gpu_mm=0., abs_error_mm=0.))
            else:
                value, _ = huber_exact(rec[:, 0]*scale, rec[:, 1], 1.)
                gpu = float(result[row].item())*scale
                error = abs(gpu-value)
                if error > 1e-4:
                    raise ValueError(f"CPU/GPU Huber discrepancy {error}mm at pixel{pixel}")
                cpu_rows.append(dict(pixel=pixel, members=n, cpu_mm=value, gpu_mm=gpu, abs_error_mm=error))
        del records, ids, counts, replay, result, status, width, pixels
    qa["empty_pixels"] = int((statuses == 0).sum())
    qa["invalid_pixels"] = int((statuses == 2).sum())
    qa["nonpositive_nonempty_mask_pixels"] = int((mask.ravel() & (statuses == 1) & (huber <= 0)).sum())
    # No extra method-dependent validity mask. The original extractor alone applies gt_mask<.5.
    huber32 = huber.astype(np.float32)
    qa["float32_storage_max_error_mm"] = float(np.max(np.abs(huber32.astype(np.float64)-huber))*scale)
    qa["minimizer_width_max_mm"] = float(widths.max()*scale)
    qa["minimizer_width_ge1mm_mask_pixels"] = int((mask.ravel() & (widths*scale >= 1.)).sum())
    qa["cpu_qa"] = cpu_rows
    if qa_first and not any(row["members"] > 0 for row in cpu_rows):
        raise ValueError("All CPU QA rays were empty; this does not validate the estimator")
    qa["cpu_qa_max_error_mm"] = max((r["abs_error_mm"] for r in cpu_rows), default=None)
    qa["seconds"] = time.time()-started
    frame = dict(rgb=rgb.cpu().numpy(), expected=expected.cpu().numpy(),
        median=median.cpu().numpy(), alpha=alpha.cpu().numpy(), huber=huber32.reshape(1, h, w), gt_mask=mask)
    return frame, qa


def native_extract(host, dataset, pipe, gaussians, scene, render):
    body = extract_source(inspect.getsource(host.extract_mesh))
    scope = vars(host).copy()
    scope.update(frozen_gaussians=gaussians, frozen_scene=scene, render=render)
    exec(compile(body, str(Path(host.__file__).resolve()), "exec"), scope)
    scope["extract_mesh"](dataset, pipe, 30000, 1)


def native_evaluate(host, dataset, gt_root, gaussians, scene):
    import torch
    body = evaluate_source(inspect.getsource(host.evaluate_mesh))
    scope = vars(host).copy()
    scope.update(args=SimpleNamespace(DTU=str(gt_root)), GaussianModel=lambda _degree: gaussians,
        Scene=lambda *_a, **_kw: scene, subprocess=subprocess, sys=sys, ROOT=ROOT,
        EVALUATION_RUNNER=EVALUATION_RUNNER)
    exec(compile(body, str(Path(host.__file__).resolve()), "exec"), scope)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scope["evaluate_mesh"](dataset, 30000, str(gt_root))


def run(args):
    provenance = source_provenance()
    scan = args.scan
    model_sha, cfg_sha, camera_count, _ = FIXED[scan]
    expected_host = "autodl-container-" + args.instance_id
    if socket.gethostname() != expected_host:
        raise ValueError("Execution hostname must match the explicitly approved instance; models remain fixed G1")
    output = args.output.resolve()
    if not args.output.is_absolute() or output.exists():
        raise ValueError("Require a new absolute independent output directory")
    for root in (args.model_root, args.source_root, args.gt_root):
        if output == root.resolve() or root.resolve() in output.parents:
            raise ValueError("Output cannot be inside protected input roots")
    model = args.model_root / f"scan{scan}"
    cfg_path = model / "cfg_args"
    ply = model / "point_cloud/iteration_30000/point_cloud.ply"
    config = cfg_values(cfg_path)
    if sha256(ply) != model_sha or sha256(cfg_path) != cfg_sha:
        raise ValueError(f"Fixed G1 scan{scan} model/cfg differ from verified baseline")
    if float(config["depth_ratio"]) != 0. or bool(config.get("eval", False)):
        raise ValueError("This pilot retains the original G1 ratio0 and training cameras")
    reference, reference_sha = verify_input_reference(args)
    required = [args.gt_root / f"Calibration/cal18/pos_{i:03d}.txt" for i in range(1, 65)]
    required += [args.gt_root / f"Points/stl/stl{scan:03d}_total.ply", args.gt_root / f"ObsMask/ObsMask{scan}_10.mat", args.gt_root / f"ObsMask/Plane{scan}.mat"]
    if not args.pilot_only:
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
    output.mkdir(parents=True)
    metadata = dict(status="partial", source=provenance, scan=scan, methods=list(METHODS), views=[], results={},
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        instance_id=args.instance_id, hostname=socket.gethostname(),
        model_sha256=model_sha, cfg_sha256=cfg_sha, depth_ratio=0., h_mm=1.,
        native_entry="tracked mesh_extract.py expected_depth path; not a strict historical mesh_extract_old.py reproduction",
        policy="Original GT mask and tracked native TSDF/Calibration are identical for both methods; only expected_depth is replaced by Huber for huber_1mm, native median remains separate and unchanged. No new alpha cutoff or method-dependent common-coverage mask.",
        interpretation="Frozen readout pilot only. Huber is a classical robust estimator, not a novelty claim. Mesh scores cannot be inferred from earlier sampled one-way NN results.")
    atomic_json(output / "metadata.json", metadata)
    try:
        import torch
        from torch.utils.cpp_extension import load
        from diff_gaussian_rasterization import _C
        import mesh_extract
        import evaluate_dtu_mesh
        from utils.general_utils import safe_state
        metadata["disk_free_before_bytes"] = shutil.disk_usage(output).free
        if not args.pilot_only and metadata["disk_free_before_bytes"] < 1500*1024*1024:
            raise ValueError("Need at least1500MiB free for this pilot's frame cache and two native meshes")
        if sha256(_C.__file__) != HOST_SHA:
            raise ValueError("Installed original rasterizer binary changed; provenance must be reviewed")
        safe_state(False)
        dataset, pipe, gaussians, cameras, loaded_ply, digest = load_scene(scan, output, args.model_root, args.source_root)
        if digest != model_sha or len(cameras) != camera_count or [c.image_name for c in cameras] != [f"{i:04d}" for i in range(camera_count)]:
            raise ValueError(f"Fixed scan{scan} model or exact{camera_count} training camera inventory changed")
        metadata["input_reference_sha256"] = reference_sha
        metadata["camera_input_reference_check"] = verify_camera_reference(cameras, reference)
        transform, alignment = coordinate_check(dataset, cameras)
        scale = float(alignment["scale_to_dtu_mm"])
        build = HERE / "构建_huber_v1"
        build.mkdir(exist_ok=True)
        os.environ.setdefault("MAX_JOBS", "2")
        extension = load(name="gaudepth_dense_huber_reader_v1", sources=[str(HERE / "reader.cu")],
            build_directory=str(build), extra_cuda_cflags=CUDA_FLAGS, verbose=True)
        if not hasattr(extension, "read_huber"):
            raise ValueError("Exact new read_huber CUDA interface was not merged")
        metadata["gpu_synthetic_check"] = gpu_synthetic_check(extension)
        metadata.update(alignment=alignment, camera_names=[c.image_name for c in cameras],
            cameras_npz_sha256=sha256(Path(dataset.source_path) / "cameras.npz"),
            original_C_path=_C.__file__, original_C_sha256=HOST_SHA,
            original_C_provenance="Preserved installed binary; original build source gap remains. Replay confirms compatibility, not source identity.",
            python=sys.version, numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            wrapper_sha256=sha256(Path(__file__)), cpu_huber_reference_sha256=sha256(HERE / "depth_holdout.py"),
            forward_reference_sha256=sha256(HERE / "sample.py"),
            nonunique_minimizer_policy="Keep the fixed midpoint of the zero-derivative interval; a balanced separated bimodal field can return an unsupported midpoint. Width>=1mm is only a numerical objective diagnostic, not physical layer detection. No GT-selected endpoint.",
            reader_source_sha256=sha256(HERE / "reader.cu"), reader_binary=extension.__file__,
            reader_binary_sha256=sha256(extension.__file__), build=str(build), cuda_flags=CUDA_FLAGS,
            host_mesh_sha256=sha256(Path(mesh_extract.__file__)), host_evaluate_sha256=sha256(Path(evaluate_dtu_mesh.__file__)),
            dtu_eval_sha256=sha256(ROOT / "dtu_eval/eval.py"),
            gt_inputs_sha256={str(p.relative_to(args.gt_root)): sha256(p) for p in required if p.is_file()})
        atomic_json(output / "metadata.json", metadata)
        cache = output / "frames"
        cache.mkdir()
        with torch.no_grad():
            for vi, camera in enumerate(cameras[:1] if args.pilot_only else cameras):
                frame, qa = dense_view(camera, gaussians, pipe, dataset, extension, scale, vi == 0)
                path = cache / f"view_{vi:03d}.npz"
                with path.open("wb") as f:
                    np.savez_compressed(f, **frame)
                    f.flush()
                    os.fsync(f.fileno())
                qa.update(file=str(path.relative_to(output)), sha256=sha256(path))
                metadata["views"].append(qa)
                atomic_json(output / "metadata.json", metadata)
                print(json.dumps(dict(stage="dense", view=camera.image_name, seconds=qa["seconds"], cpu_qa_max_error_mm=qa["cpu_qa_max_error_mm"])), flush=True)
                if (output / "stop_after_stage").exists():
                    verify_originals(metadata, loaded_ply, cfg_path)
                    metadata["status"] = "stopped_after_view"
                    atomic_json(output / "metadata.json", metadata)
                    return metadata
        if args.pilot_only:
            verify_originals(metadata, loaded_ply, cfg_path)
            metadata["status"] = "pilot_complete"
            atomic_json(output / "metadata.json", metadata)
            return metadata
        metadata["status"] = "dense_complete"
        metadata["evaluation_protocol"] = "Original evaluate_dtu_mesh Calibration camera-center alignment and original dtu_eval score definitions. Both methods use fixed shuffle default_rng seed20261026 (explicit departure from unseeded history), default Pool limited to8, child timeout1200s, no visualization point-cloud writes; no scale_mat substitution or new ICP."
        atomic_json(output / "metadata.json", metadata)
        scene = SimpleNamespace(getTrainCameras=lambda: cameras)
        by_name = {v["name"]: v for v in metadata["views"]}
        for method in METHODS:
            destination = output / method
            destination.mkdir()
            proxy = SimpleNamespace(**dict(vars(dataset), model_path=str(destination)))
            def cached_render(camera, *_a, **_kw):
                item = by_name[camera.image_name]
                path = output / item["file"]
                if sha256(path) != item["sha256"]:
                    raise ValueError("Frozen frame cache changed")
                with np.load(path) as frame:
                    result = {"render": torch.from_numpy(frame["rgb"]).cuda(),
                        "expected_depth": torch.from_numpy(frame["expected" if method == "native_expected" else "huber"]).cuda(),
                        "median_depth": torch.from_numpy(frame["median"]).cuda(),
                        "mask": torch.from_numpy(frame["alpha"]).cuda()}
                return result
            started = time.time()
            with torch.no_grad():
                native_extract(mesh_extract, proxy, pipe, gaussians, scene, cached_render)
                native_evaluate(evaluate_dtu_mesh, proxy, args.gt_root, gaussians, scene)
            mesh = destination / "recon_post.ply"
            results = destination / "vis/results.json"
            scores = json.loads(results.read_text(encoding="utf-8"))
            if not all(np.isfinite(scores[k]) for k in ("mean_d2s", "mean_s2d", "overall")):
                raise ValueError("Nonfinite original evaluator scores")
            metadata["results"][method] = dict(scores=scores, mesh_sha256=sha256(mesh),
                aligned_sha256=sha256(destination / "recon_aligned.ply"), results_sha256=sha256(results), seconds=time.time()-started)
            atomic_json(output / "metadata.json", metadata)
            print(json.dumps(dict(stage="mesh", method=method, scores=scores)), flush=True)
            if (output / "stop_after_stage").exists():
                verify_originals(metadata, loaded_ply, cfg_path)
                metadata["status"] = "stopped_after_method"
                atomic_json(output / "metadata.json", metadata)
                return metadata
        verify_originals(metadata, loaded_ply, cfg_path)
        metadata.update(status="complete", original_model_unchanged=True, original_cfg_unchanged=True)
        atomic_json(output / "metadata.json", metadata)
        return metadata
    except BaseException as exc:
        metadata.update(status="failed_or_stopped", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "metadata.json", metadata)
        raise


def self_check():
    def function_source(path, name):
        text = path.read_text(encoding="utf-8")
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(text, node)
    compile(extract_source(function_source(ROOT / "mesh_extract.py", "extract_mesh")), "extract_mesh", "exec")
    compile(evaluate_source(function_source(ROOT / "evaluate_dtu_mesh.py", "evaluate_mesh")), "evaluate_mesh", "exec")
    compile(EVALUATION_RUNNER, "evaluation_child", "exec")
    z = np.array([0., 10.]); w = np.array([.5, .5])
    value, width = huber_exact(z, w, 1.)
    np.testing.assert_allclose([value, width], [5., 8.], rtol=0, atol=1e-12)
    print("Native source sites, controlled evaluator syntax and CPU Huber plateau: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan", type=int, choices=tuple(FIXED), default=24,
                        help="One of the 15 SHA-pinned G1 frozen scenes; unchanged fixed1mm protocol")
    parser.add_argument("--instance-id", choices=("d6a7459179-693ab6ee", "26a34fbbce-406e74f8"),
                        default="d6a7459179-693ab6ee", help="Approved execution instance, not model origin")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-root", type=Path, default=Path("/root/autodl-tmp/GauDepth/outtest"))
    parser.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/DTU"))
    parser.add_argument("--gt-root", type=Path, default=Path("/root/autodl-tmp/DTU_MVS_Data"))
    args = parser.parse_args()
    if args.check:
        self_check()
    elif args.output is None:
        parser.error("--output is required")
    else:
        result = run(args)
        print(json.dumps(dict(status=result["status"], results=result["results"])), flush=True)


if __name__ == "__main__":
    main()
