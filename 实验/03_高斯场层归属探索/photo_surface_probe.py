"""Observation-only surface identifiability probe; frozen models and original data stay read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
INSTANCE = "5d5147b55f-67549ee6"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def source_info():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    if git("status", "--porcelain") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require clean committed source and approved origin")
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"))


def export_observations(args):
    if socket.gethostname() != "autodl-container-" + INSTANCE:
        raise ValueError("Only the explicitly authorized G0")
    source = source_info()
    root = args.source_root.resolve()
    out = args.output.resolve()
    if not args.output.is_absolute() or out.exists() or out == root or root in out.parents:
        raise ValueError("Require a new independent output outside original observations")
    out.mkdir(parents=True)
    ref_path = HERE / "G1_Huber全场景输入参考.json"
    reference = json.loads(ref_path.read_text(encoding="utf-8"))["scans"]
    manifest = dict(status="partial", source=source, hostname=socket.gethostname(), instance_id=INSTANCE,
        source_root=str(root), reference_sha256=sha256(ref_path), scans={}, files=[],
        limitation="These are original observed DTU RGB files on G0, not rendered RGB. Camera/source reference SHA is checked. No previous G1 observed-RGB SHA was recorded, so G0 RGB equality to G1 training RGB is NOT asserted.")
    save(out / "metadata.json", manifest)
    archive = out / "observations.tar"
    try:
        with tarfile.open(archive, "w") as tar:
            for scan in args.scans:
                ref = reference[str(scan)]
                for relative, digest in ref["source_inputs"].items():
                    if sha256(root / relative) != digest:
                        raise ValueError("Camera/source mismatch: " + relative)
                    p = root / relative
                    tar.add(p, arcname=relative, recursive=False)
                    manifest["files"].append(dict(relative=relative, bytes=p.stat().st_size, sha256=digest))
                images = root / f"scan{scan}/images"
                paths = {}
                for p in images.iterdir():
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        if p.stem in paths:
                            raise ValueError("Ambiguous observed image name: " + p.stem)
                        paths[p.stem] = p
                names = [v["name"] for v in ref["views"]]
                if set(paths) != set(names):
                    raise ValueError(f"Exact observed image inventory differs for scan{scan}")
                manifest["scans"][str(scan)] = dict(views=len(names), camera_source_sha256=ref["source_inputs"])
                for name in names:
                    p = paths[name]
                    relative = str(p.relative_to(root)).replace("\\", "/")
                    digest = sha256(p)
                    tar.add(p, arcname=relative, recursive=False)
                    if sha256(p) != digest:
                        raise ValueError("Original observation changed during read")
                    manifest["files"].append(dict(relative=relative, bytes=p.stat().st_size, sha256=digest))
        manifest.update(status="complete_waiting_transfer", archive_bytes=archive.stat().st_size,
                        archive_sha256=sha256(archive), completed_at=time.time())
        save(out / "metadata.json", manifest)
        while not (out / "transfer_verified").exists():
            time.sleep(2)
        manifest["status"] = "complete"
        save(out / "metadata.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed_or_stopped", error=repr(exc))
        save(out / "metadata.json", manifest)
        raise


def observation_probe(args):
    """Fixed image-only profile fit, independent inference-view check, then NN diagnostics."""
    import importlib.util
    import sys
    import numpy as np
    from PIL import Image
    from photo_objective import photo_cost, bilinear
    from consensus_depth import _check_camera_projection

    info = source_info()
    output = args.output.resolve()
    if not args.output.is_absolute() or output.exists():
        raise ValueError("Require a new independent output")
    output.mkdir(parents=True)
    started = time.monotonic()
    state = dict(status="partial", source=info, completed=[], parameters=dict(scans=[37, 40],
        reference="0000", samples=512, seed=20261026, patch_radius=2, profile_steps=17,
        profile_halfwidth_pixels=1., fit_neighbors=[0, 2, 4, 6], gate_neighbors=[1, 3], audit_neighbors=[5, 7]),
        limitation="Exploratory inference with already-trained-on RGB. Photo/profile and scan-point NN are proxies, not pixel GT, new training holdout, calibrated uncertainty, or native reconstruction evidence. Unknown returns E; no GT enters the decision.")
    save(output / "metadata.json", state)
    selected = json.loads((args.observation_root.parent / "selected_local_verified.json").read_text())
    if selected["status"] != "selected_subset_complete":
        raise ValueError("Selected observation source SHA not verified")
    for rel, record in selected["files"].items():
        if sha256(args.observation_root / rel) != record["sha256"]:
            raise ValueError("Observation changed: " + rel)
    spec = importlib.util.spec_from_file_location("photo_colmap_loader", REPO / "scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    results = HERE / "结果"
    frozen = json.loads((HERE / "G1_Huber原缓存SHA.json").read_text(encoding="utf-8"))

    def check():
        if time.monotonic() - started > args.max_seconds or (output / "stop_after_scene").exists():
            raise InterruptedError("Finite local probe deadline/stop")

    try:
        for scan in (37, 40):
            check()
            cache = (results / "20261026_夜间深度方法/dense_confirm_v1/scan37_v1" if scan == 37
                     else results / "20261026_Huber全场景/scan40_v1")
            meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
            assert meta["status"] == "complete"
            scale = float(meta["alignment"]["scale_to_dtu_mm"])
            transform = np.asarray(meta["alignment"]["transform"], dtype=float)
            neighbor_rows = [json.loads(line) for line in (cache / "multi_view.json").read_text().splitlines()]
            names = next(row["nearest_name"] for row in neighbor_rows if row["ref_name"] == "0000")
            assert len(names) == 8
            fit, hold, audit = [0, 2, 4, 6], [1, 3], [5, 7]
            files = {vm["name"]: vm for vm in meta["views"]}
            intrinsics = loader.read_intrinsics_binary(args.observation_root / f"scan{scan}/sparse/0/cameras.bin")
            extrinsics = {Path(row.name).stem: row for row in loader.read_extrinsics_binary(
                args.observation_root / f"scan{scan}/sparse/0/images.bin").values()}
            images, frames, cameras, photo_paths = {}, {}, {}, {}
            source_checks = []
            for name in ["0000", *names]:
                check()
                vm = files[name]
                assert sha256(cache / vm["file"]) == frozen[str(scan)][vm["file"]] == vm["sha256"]
                with np.load(cache / vm["file"]) as frame:
                    frames[name] = {key: frame[key].copy() for key in ("expected", "huber", "gt_mask")}
                image_row = extrinsics[name]
                camera_row = intrinsics[image_row.camera_id]
                assert camera_row.model == "PINHOLE"
                fx, fy, cx, cy = camera_row.params
                height, width = vm["image_size"]
                sx, sy = width / camera_row.width, height / camera_row.height
                K = np.array([[sx * fx, 0., sx * cx - .5], [0., sy * fy, sy * cy - .5], [0., 0., 1.]])
                np.testing.assert_allclose(K[:2, 2], [(width-1)/2., (height-1)/2.], rtol=0, atol=1e-8)
                np.testing.assert_allclose([K[0, 0], K[1, 1]], vm["focal"], rtol=0, atol=3e-4)
                w2c = np.eye(4)
                w2c[:3, :3] = image_row.qvec2rotmat()
                w2c[:3, 3] = image_row.tvec
                w2c = w2c.astype(np.float32).astype(float)
                re, te = _check_camera_projection(args.observation_root / f"scan{scan}/cameras.npz",
                    dict(image_name=name, world_view_transform=w2c.T.tolist()), transform)
                cameras[name] = dict(K=K, w2c=w2c, c2w=np.linalg.inv(w2c), image_size=(height, width))
                image_paths = [p for p in (args.observation_root / f"scan{scan}/images").iterdir() if p.stem == name]
                assert len(image_paths) == 1
                photo_paths[name] = image_paths[0]
                with Image.open(image_paths[0]) as photo:
                    assert photo.size == (camera_row.width, camera_row.height)
                    if name == "0000" or name in [names[j] for j in fit+hold]:
                        rgb = np.asarray(photo.convert("RGB").resize((width, height)), dtype=float) / 255.
                        gray = rgb @ np.array([.299, .587, .114])
                        gray[~frames[name]["gt_mask"].astype(bool)] = np.nan
                        images[name] = gray
                source_checks.append(dict(name=name, camera_id=image_row.camera_id, K=K.tolist(),
                    pose_rotation_max_abs=re, pose_translation_max_abs_scene=te,
                    image_sha256=sha256(image_paths[0]), frame_sha256=vm["sha256"]))
            ci = cameras["0000"]
            h, w = ci["image_size"]
            e_map = frames["0000"]["expected"].reshape(h, w)
            h_map = frames["0000"]["huber"].reshape(h, w)
            mask = frames["0000"]["gt_mask"].astype(bool)
            interior = mask.copy(); interior[:3] = False; interior[-3:] = False; interior[:, :3] = False; interior[:, -3:] = False
            eligible = np.flatnonzero(interior & np.isfinite(e_map) & (e_map > 0))
            rng = np.random.default_rng(20261026 + scan)
            index = np.sort(rng.choice(eligible, size=512, replace=False))
            y, x = np.divmod(index, w); xy = np.column_stack((x, y)).astype(float)
            E, H = e_map.ravel()[index], h_map.ravel()[index]
            delta_mm = (H.astype(float)-E.astype(float))*scale
            soft = (E.astype(float)+np.sign(delta_mm)*np.maximum(np.abs(delta_mm)-1., 0.)/scale).astype(np.float32)
            Kinv = np.linalg.inv(ci["K"])
            def ray_at(xx, yy):
                return np.column_stack((xx, yy, np.ones_like(xx))) @ Kinv.T
            ray = ray_at(x, y)
            dx = ray_at(x+1, y)*e_map[y, x+1, None] - ray_at(x-1, y)*e_map[y, x-1, None]
            dy = ray_at(x, y+1)*e_map[y+1, x, None] - ray_at(x, y-1)*e_map[y-1, x, None]
            normals = np.cross(dx, dy)
            norm = np.linalg.norm(normals, axis=1)
            normals /= np.maximum(norm[:, None], 1e-20)
            normals[norm < 1e-12] = np.nan
            points = ray * E[:, None]
            max_jac = np.zeros(len(index))
            eh_displacements = []
            world_ray = ray @ ci["c2w"][:3, :3].T
            center = ci["c2w"][:3, 3]
            for name in names:
                cj = cameras[name]
                a = world_ray @ cj["w2c"][:3, :3].T
                b = center @ cj["w2c"][:3, :3].T + cj["w2c"][:3, 3]
                q = a * E[:, None] + b
                jac = (a[:, :2]*b[2] - b[:2]*a[:, 2, None]) / np.maximum(q[:, 2, None]**2, 1e-20)
                jac *= np.array([cj["K"][0, 0], cj["K"][1, 1]])
                max_jac = np.maximum(max_jac, np.linalg.norm(jac, axis=1))
                qh = a * H[:, None] + b
                displacement = (qh[:, :2]/qh[:, 2, None] - q[:, :2]/q[:, 2, None]) * np.array([cj["K"][0, 0], cj["K"][1, 1]])
                eh_displacements.append(np.linalg.norm(displacement, axis=1))
            span = 1. / np.maximum(max_jac, 1e-20)
            profile = E[:, None] + span[:, None] * np.linspace(-1., 1., 17)
            profile[profile <= 0] = np.nan
            candidates = np.column_stack((E, H, soft, profile))
            costs = np.full((8, len(index), candidates.shape[1]), np.nan)
            for j in fit+hold:
                check()
                name = names[j]
                costs[j] = photo_cost(images["0000"], images[name], xy, candidates, normals, ci, cameras[name], radius=2)
            pcost = costs[:, :, 3:]
            common = np.isfinite(pcost[fit+hold]).all(axis=(0, 2)) & np.isfinite(costs[fit+hold, :, 0]).all(axis=0)
            fit_loss = np.mean(np.where(np.isfinite(pcost[fit]), pcost[fit], np.inf), axis=0)
            best = np.argmin(fit_loss, axis=1)
            best_score = fit_loss[np.arange(len(index)), best]
            tied = np.sum(np.isfinite(fit_loss) & np.isclose(fit_loss, best_score[:, None], rtol=0, atol=1e-8), axis=1) > 1
            held_best = pcost[hold][:, np.arange(len(index)), best]
            held_E = costs[hold, :, 0]
            gain = held_E - held_best
            accept = common & (best > 0) & (best < 16) & (best != 8) & ~tied & (gain > 1e-8).all(axis=0)
            D = E.astype(np.float64, copy=True); D[accept] = profile[np.arange(len(index)), best][accept]
            scene = output / f"scan{scan}_view0000"; scene.mkdir()
            # Decisions are persisted BEFORE any independent scan cloud is loaded.
            np.savez_compressed(scene / "decisions.npz", index=index, xy=xy, E=E, H=H, soft=soft,
                profile=profile, best=best, common=common, accept=accept, selected=D, costs=costs,
                max_neighbor_EH_displacement_px=np.max(eh_displacements, axis=0))
            sealed_decision_sha = sha256(scene/"decisions.npz")
            # Audit RGB is decoded and scored only after D and its SHA are sealed.
            for j in audit:
                check()
                name = names[j]
                with Image.open(photo_paths[name]) as photo:
                    rgb = np.asarray(photo.convert("RGB").resize((w, h)), dtype=float) / 255.
                gray = rgb @ np.array([.299, .587, .114])
                gray[~frames[name]["gt_mask"].astype(bool)] = np.nan
                costs[j] = photo_cost(images["0000"], gray, xy, candidates, normals, ci, cameras[name], radius=2)
            assert sha256(scene/"decisions.npz") == sealed_decision_sha
            audit_best = pcost[audit][:, np.arange(len(index)), best]
            audit_E = costs[audit, :, 0]
            audit_take = accept[None, :] & np.isfinite(audit_best) & np.isfinite(audit_E)
            audit_delta = audit_best - audit_E
            scene_record = dict(scan=scan, view="0000", samples=len(index), common6_profile=int(common.sum()),
                accepted=int(accept.sum()), unknown_or_unchanged=int((~accept).sum()),
                independent_inference_audit_pairs=int(audit_take.sum()),
                independent_inference_audit_coverage_fraction=float(audit_take.sum()/(2*accept.sum())) if accept.any() else None,
                independent_inference_audit_delta_mean=float(audit_delta[audit_take].mean()) if audit_take.any() else None,
                independent_inference_audit_worse_fraction=float((audit_delta[audit_take]>0).mean()) if audit_take.any() else None,
                EH_max_neighbor_displacement_median_px=float(np.median(np.max(eh_displacements, axis=0))),
                EH_max_neighbor_displacement_lt0p02_fraction=float((np.max(eh_displacements, axis=0) < .02).mean()),
                source_checks=source_checks, decisions_sha256=sealed_decision_sha,
                decision_dtype=str(D.dtype), audit_read_only_after_decision=True)
            np.savez_compressed(scene/"audit.npz", costs=costs[audit], valid_pairs=audit_take, selected_delta=audit_delta)
            save(scene / "metadata.json", scene_record)
            from scipy.spatial import cKDTree
            from plyfile import PlyData
            gt_path = results / f"20261026_夜间深度方法/inputs/scan{scan}/stl{scan:03d}_total.ply"
            assert sha256(gt_path) == meta["gt_inputs_sha256"][f"Points/stl/stl{scan:03d}_total.ply"]
            vertex = PlyData.read(str(gt_path))["vertex"]
            tree = cKDTree(np.column_stack([np.asarray(vertex[k], dtype=float) for k in ("x", "y", "z")]))
            nn = {}
            for label, depth in (("expected", E), ("huber", H), ("soft", soft), ("photo_profile", D)):
                world = (ray*depth[:, None]) @ ci["c2w"][:3, :3].T + ci["c2w"][:3, 3]
                mm = world @ transform[:3, :3].T + transform[:3, 3]
                nn[label] = tree.query(mm, k=1, eps=0, workers=1)[0]
            scene_record.update(nn_proxy={k:dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), p95=float(np.quantile(v, .95))) for k, v in nn.items()},
                accepted_nn_delta_mean_mm=float((nn["photo_profile"]-nn["expected"])[accept].mean()) if accept.any() else None,
                accepted_nn_worse_fraction=float((nn["photo_profile"][accept]>nn["expected"][accept]).mean()) if accept.any() else None,
                gt_loaded_only_after_decisions=True)
            save(scene / "metadata.json", scene_record)
            state["completed"].append(scene_record); save(output / "metadata.json", state)
            print(json.dumps({k:v for k,v in scene_record.items() if k not in ("source_checks",)}), flush=True)
        state.update(status="complete", seconds=time.monotonic()-started)
        save(output / "metadata.json", state)
    except BaseException as exc:
        state.update(status="failed_or_stopped", error=repr(exc));save(output/"metadata.json",state);raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="Read-only original RGB archive on guarded G0; no model or CUDA execution")
    e.add_argument("--scans", nargs="+", type=int, choices=(37, 40), default=[37, 40])
    e.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/DTU"))
    e.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("probe", help="Local fixed observation-only profile and inference-view holdout probe")
    q.add_argument("--observation-root", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--max-seconds", type=int, default=900)
    args = p.parse_args()
    export_observations(args) if args.command == "export" else observation_probe(args)


if __name__ == "__main__":
    main()
