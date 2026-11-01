"""Run frozen 32-window local plane training reference on CPU only."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from local_plane_training_cpu import (ANCHOR_OFFSETS, CAPTURE_SHA, InvalidWindow,
                                      frozen_objective_torch, geometry, fit,
                                      member_plane, sha256)


def camera_roster(camera_dir, width, height):
    """COLMAP path and scaling of reviewed 13bc90b camera_roster."""
    loader_path = Path(__file__).parent / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("local_plane_colmap_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    intr = module.read_intrinsics_binary(str(camera_dir / "cameras.bin"))
    extr = module.read_extrinsics_binary(str(camera_dir / "images.bin"))
    roster = {}
    for row in extr.values():
        lens = intr[row.camera_id]
        if lens.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError("unsupported_distortion")
        fx0 = lens.params[0]
        fy0 = lens.params[1] if lens.model == "PINHOLE" else fx0
        Rwc = module.qvec2rotmat(row.qvec)
        t = np.asarray(row.tvec)
        roster[Path(row.name).stem] = dict(R=Rwc, t=t, center=-Rwc.T@t,
            fx=fx0*width/lens.width, fy=fy0*height/lens.height,
            cx=(width-1)/2, cy=(height-1)/2)
    return roster


def camera_matrix(c):
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.]])


def verify_reference_camera(cap, ref):
    w2c = np.eye(4)
    w2c[:3, :3] = ref["R"]
    w2c[:3, 3] = ref["t"]
    pairs = ((cap["world_view_transform"], w2c.T),
             (cap["camera_R"], ref["R"].T),
             (cap["camera_T"], ref["t"]),
             (cap["intrinsics"], [ref[k] for k in ("fx", "fy", "cx", "cy")]))
    if not all(np.allclose(a, b, rtol=2e-5, atol=2e-5) for a, b in pairs):
        raise ValueError("reference_camera_mismatch")
    H, W = map(int, cap["image_size"])
    projection = np.zeros((4, 4))
    projection[0, 0] = 2*ref["fx"]/W
    projection[1, 1] = 2*ref["fy"]/H
    projection[2, 2] = 100/(100-.01)
    projection[2, 3] = -100*.01/(100-.01)
    projection[3, 2] = 1
    if not np.allclose(cap["full_proj_transform"], w2c.T@projection.T,
                       rtol=2e-5, atol=2e-5):
        raise ValueError("reference_projection_mismatch")


def window_members(cap, index, id_to_row):
    mass = {}
    for query in range(9*index, 9*index+9):
        count = int(cap["count"][query])
        if count < 0 or count > cap["ids"].shape[1]:
            raise InvalidWindow("invalid_count")
        if bool(cap["overflow"][query]):
            raise InvalidWindow("capture_overflow")
        if len(set(map(int, cap["ids"][query, :count]))) != count:
            raise InvalidWindow("duplicate_id_within_ray")
        for gid, w in zip(cap["ids"][query, :count], cap["weight"][query, :count]):
            gid = int(gid)
            if gid not in id_to_row or not np.isfinite(w) or w < 0:
                raise InvalidWindow("invalid_member_or_weight")
            mass.setdefault(gid, []).append(float(w)/9)
    ids = sorted(mass)
    return ids, np.array([sum(mass[i]) for i in ids], dtype=np.float64)


def gradient_check(mu, quat, axes, q, latent, assignment, Rref, tref, geom):
    x = torch.tensor(mu, dtype=torch.float64, requires_grad=True)
    r = torch.tensor(quat, dtype=torch.float64, requires_grad=True)
    value = frozen_objective_torch(x, r, axes, q, latent, assignment, Rref, tref, geom)
    gx, gr = torch.autograd.grad(value, (x, r))
    result = {"loss": float(value), "groups": {}}
    for name, source, grad in (("center", mu, gx.detach().numpy()),
                               ("raw_quaternion", quat, gr.detach().numpy())):
        if not np.isfinite(grad).all():
            raise InvalidWindow("nonfinite_gradient_"+name)
        flat = int(np.argmax(np.abs(grad)))
        pos = np.unravel_index(flat, grad.shape)
        ad = float(grad[pos])
        if ad == 0:
            result["groups"][name] = {"status": "all_zero_unproven"}
            continue
        h = 1e-6*max(1., abs(float(source[pos])))
        plus, minus = source.copy(), source.copy()
        plus[pos] += h
        minus[pos] -= h
        def evaluate(mm, rr):
            with torch.no_grad():
                return float(frozen_objective_torch(torch.as_tensor(mm, dtype=torch.float64),
                    torch.as_tensor(rr, dtype=torch.float64), axes, q, latent, assignment,
                    Rref, tref, geom))
        fd = (evaluate(plus, quat)-evaluate(minus, quat))/(2*h) if name == "center" else (evaluate(mu, plus)-evaluate(mu, minus))/(2*h)
        tol = 1e-6+1e-4*max(abs(ad), abs(fd))
        result["groups"][name] = dict(status="pass" if abs(ad-fd) <= tol else "fail",
                                        index=[int(v) for v in pos], ad=ad, fd=fd, error=abs(ad-fd), tolerance=tol)
    return result


def run(capture_path, camera_dir, output):
    start = time.monotonic()
    capture_path, camera_dir, output = map(Path, (capture_path, camera_dir, output))
    if output.exists():
        raise ValueError("output_already_exists")
    if sha256(capture_path) != CAPTURE_SHA:
        raise ValueError("capture_sha_mismatch")
    with np.load(capture_path, allow_pickle=False) as z:
        cap = {k: z[k].copy() for k in z.files}
    if cap["xy"].shape != (288, 2) or tuple(cap["image_size"]) != (581, 777):
        raise ValueError("capture_shape_mismatch")
    for name, expected in (("cameras.bin", "9dd012bebe2dea0514af3f784dbb8fff23354fd7e9476ae6f9c668c40162d06b"),
                           ("images.bin", "433bda84b16ae7961619442c5bc74c70ddbb090e38ff61eaf601ff3719bb2794")):
        if sha256(camera_dir/name) != expected:
            raise ValueError("colmap_sha_mismatch_"+name)
    small = camera_roster(camera_dir, 777, 581)
    verify_reference_camera(cap, small["0049"])
    raw = camera_roster(camera_dir, 1554, 1162)
    ref, dst = raw["0049"], raw["0050"]
    Rrel = dst["R"]@ref["R"].T
    trel = dst["t"]-Rrel@ref["t"]
    ids_all = list(map(int, cap["unique_id"]))
    if len(ids_all) != len(set(ids_all)):
        raise ValueError("duplicate_global_id")
    id_to_row = {gid: i for i, gid in enumerate(ids_all)}
    rows = []
    for p in range(32):
        entry = dict(index=p, status="invalid")
        try:
            ids, mass = window_members(cap, p, id_to_row)
            entry.update(ids=ids, mass=mass.tolist(), Z=float(sum(mass)))
            if not ids or not np.any(mass > 0):
                raise InvalidWindow("zero_mass")
            selected = np.array([id_to_row[gid] for gid in ids])
            mu = cap["center"][selected].astype(np.float64)
            quat = cap["raw_quaternion"][selected].astype(np.float64)
            scales = cap["raw_scale"][selected].astype(np.float64)
            if not np.isfinite(mu).all() or not np.isfinite(quat).all() or not np.isfinite(scales).all():
                raise InvalidWindow("nonfinite_raw_parameter")
            axes = np.argmin(scales, axis=1)
            gaps = np.diff(np.sort(scales, axis=1), axis=1)[:, 0]
            planes = np.zeros((len(ids), 3), dtype=np.float64)
            for j in np.flatnonzero(mass > 0):
                planes[j] = member_plane(mu[j], quat[j], int(axes[j]), ref["R"], ref["t"])[0]
            center_raw = 2*cap["xy"][9*p+4].astype(np.float64)+.5
            uv = center_raw + ANCHOR_OFFSETS
            geom = geometry(camera_matrix(ref), camera_matrix(dst), Rrel, trel, uv)
            best, models = fit(planes, mass, geom)
            q = mass/best["Z"]
            grad = gradient_check(mu, quat, axes, q, best["latent"], best["assignment"],
                                  ref["R"], ref["t"], geom)
            ray = np.linalg.inv(camera_matrix(ref)) @ np.array([*center_raw, 1.])
            positive = sum(bool(float(plane@ray)>0) for plane in planes[mass > 0])
            entry.update(status="valid", selected_M=best["M"], models=models, best=best,
                raw_center=center_raw.tolist(), axes=axes.tolist(), shortest_log_scale_gap=gaps.tolist(),
                positive_center_depth_fraction=positive/int(np.count_nonzero(mass)), gradient=grad)
        except (InvalidWindow, np.linalg.LinAlgError, ValueError) as exc:
            entry["reason"] = str(exc)
        rows.append(entry)
        print(json.dumps({"window":p,"status":entry["status"],"reason":entry.get("reason"),
                          "selected_M":entry.get("selected_M")}), flush=True)
    valid = [r for r in rows if r["status"] == "valid"]
    result = dict(protocol="local_plane_training_cpu_v1", capture_sha256=CAPTURE_SHA,
        camera_sha256={n:sha256(camera_dir/n) for n in ("cameras.bin", "images.bin")},
        camera_source_commit="13bc90ba91e52634760cd70f803117860609a37e",
        reference="0049", target="0050", original_resolution=[1554,1162],
        windows=rows, valid_count=len(valid), invalid_count=32-len(valid),
        invalid_indices=[r["index"] for r in rows if r["status"] != "valid"],
        R=(sum(r["best"]["numerator"] for r in valid)/sum(r["best"]["Z"] for r in valid)) if len(valid)==32 else None,
        partial_valid_R=(sum(r["best"]["numerator"] for r in valid)/sum(r["best"]["Z"] for r in valid)) if valid else None,
        elapsed_sec=time.monotonic()-start)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--camera-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.capture, args.camera_dir, args.output)
