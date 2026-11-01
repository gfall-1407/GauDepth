"""Offline fresh scan110 capture runner. Requires numpy and Pillow, no Torch/GPU.

This evaluates frozen depth hypotheses and heldout image patch costs. It does
not implement a Gaussian parameter gradient or an optimizer update direction.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from visible_branch_adapter import evaluate_fixed_32
from visible_branch_reference import Camera, Member, patch_cost, pixel_ray

EXPECTED_CAPTURE = {
    "git_commit": "faa9294b58bd91323dc42ee4e81ecd2297c4918b",
    "ply_sha256": "733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7",
    "cfg_sha256": "f41faab818f1d64b35f83acd71981a4170eff2384fa8543688790fa4f3010b85",
    "binary_sha256": "e42925112ccf32fad4185c63c56f196ca5a529338c4bcbf40380c8bd7c092a1d",
    "result_sha256": "74fbd05c3a2c10549adca00f885cdb39c525d003ce120d1a3d593c7feb78d6ef",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def colmap_module():
    path = Path(__file__).parent / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("visible_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_roster(camera_dir, width, height):
    loader = colmap_module()
    intr = loader.read_intrinsics_binary(str(camera_dir / "cameras.bin"))
    extr = loader.read_extrinsics_binary(str(camera_dir / "images.bin"))
    roster = {}
    for row in extr.values():
        lens = intr[row.camera_id]
        if lens.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"unsupported distortion model: {lens.model}")
        fx0 = lens.params[0]
        fy0 = lens.params[1] if lens.model == "PINHOLE" else fx0
        Rwc = loader.qvec2rotmat(row.qvec)
        t = np.asarray(row.tvec)
        center = -Rwc.T @ t
        camera = Camera(tuple(center.tolist()), tuple(map(tuple, Rwc.T.tolist())),
                        fx0*width/lens.width, fy0*height/lens.height,
                        (width-1)/2, (height-1)/2)
        roster[Path(row.name).stem] = (camera, Rwc, t)
    return roster


def verify_reference_camera(capture, roster, view):
    cam, Rwc, t = roster[view]
    w2c = np.eye(4)
    w2c[:3, :3] = Rwc
    w2c[:3, 3] = t
    if not np.allclose(capture["world_view_transform"], w2c.T,
                       rtol=2e-5, atol=2e-5):
        raise ValueError("reference world-view matrix disagrees with COLMAP")
    if not np.allclose(capture["camera_R"], Rwc.T,
                       rtol=2e-5, atol=2e-5):
        raise ValueError("reference camera rotation disagrees with COLMAP")
    if not np.allclose(capture["camera_T"], t, rtol=2e-5, atol=2e-5):
        raise ValueError("reference camera translation disagrees with COLMAP")
    if not np.allclose(capture["intrinsics"],
                       [cam.fx, cam.fy, cam.cx, cam.cy], rtol=2e-5, atol=2e-5):
        raise ValueError("reference intrinsics disagree with COLMAP resize")
    H, W = map(int, capture["image_size"])
    projection = np.zeros((4, 4))
    projection[0, 0] = 2*cam.fx/W
    projection[1, 1] = 2*cam.fy/H
    projection[2, 2] = 100/(100-.01)
    projection[2, 3] = -100*.01/(100-.01)
    projection[3, 2] = 1
    if not np.allclose(capture["full_proj_transform"], w2c.T @ projection.T,
                       rtol=2e-5, atol=2e-5):
        raise ValueError("reference full projection matrix disagrees with COLMAP")


def image_gray(path, width, height):
    image = Image.open(path)
    if len(image.getbands()) < 3:
        raise ValueError("expected RGB image channels")
    channels = [np.asarray(channel.resize((width, height)), dtype=np.float32)/255.
                for channel in image.split()[:3]]
    gray = .299*channels[0]+.587*channels[1]+.114*channels[2]
    return gray.tolist()


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 2e-3:
        raise ValueError("invalid activated quaternion")
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def make_member(capture, row, alpha, T, reference, pixel):
    scale = np.asarray(capture["scale_render"][row], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.min(scale) <= 0:
        raise ValueError("invalid effective Gaussian scale")
    axes = rotation_matrix(capture["quaternion"][row])
    short = int(np.argmin(scale))
    normal = axes[:, short]
    mu = np.asarray(capture["center"][row], dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError("invalid Gaussian center")
    ray = np.asarray(pixel_ray(reference, *pixel))
    denom = float(normal @ ray)
    if abs(denom) <= 1e-8:
        footprint = 0.
    else:
        depth = float(normal @ (mu-np.asarray(reference.center)) / denom)
        point = np.asarray(reference.center)+depth*ray
        local = axes.T @ (point-mu)
        energy = sum((local[k]/scale[k])**2 for k in range(3) if k != short)
        footprint = float(np.exp(-.5*energy))
    return Member(int(capture["unique_id"][row]), tuple(mu.tolist()),
                  tuple(normal.tolist()), float(alpha), float(T), footprint)


def heldout_cost(depth, reference, reference_image, heldout, pixel):
    cam, image = heldout
    ray = pixel_ray(reference, *pixel)
    mu = tuple(o+depth*v for o, v in zip(reference.center, ray))
    normal = tuple(reference.camera_to_world[r][2] for r in range(3))
    plane = Member(-1, mu, normal, .5, 1., 1.)
    return patch_cost(plane, reference, reference_image,
                      cam, image, *pixel)


def run(capture_dir, image_dir, camera_dir, image_manifest):
    capture_dir, image_dir, camera_dir = map(Path, (capture_dir, image_dir, camera_dir))
    meta = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    image_meta = json.loads(Path(image_manifest).read_text(encoding="utf-8"))
    for field, expected in EXPECTED_CAPTURE.items():
        if meta.get(field) != expected:
            raise ValueError(f"capture identity mismatch: {field}")
    if meta.get("detached") is not True or meta.get("ply_row_count") != 142892:
        raise ValueError("unexpected model or native capture provenance")
    if meta["seed"] != 20261101 or meta["rays"] != 32 or meta["neighborhood"] != "3x3":
        raise ValueError("capture is not frozen 32-ray scan110 contract")
    if Path(meta["image_path"]).stem != "0049" or meta["view"] != "0049":
        raise ValueError("reference view is not fixed 0049")
    npz_path = capture_dir / "native_members.npz"
    require_hash(npz_path, meta["result_sha256"])
    for name in ("cameras.bin", "images.bin"):
        expected = next((value for key, value in meta["colmap_sha256"].items()
                         if Path(key).name == name), None)
        if expected is None:
            raise ValueError(f"capture lacks {name} hash")
        require_hash(camera_dir / name, expected)
    for view in ("0049", "0012", "0050", "0058"):
        expected = image_meta["targets"][view]["image_sha256"]
        require_hash(image_dir / f"{view}.png", expected)
    with np.load(npz_path, allow_pickle=False) as data:
        cap = {key: data[key] for key in data.files}
    if cap["xy"].shape != (288, 2) or cap["ids"].shape[0] != 288:
        raise ValueError("capture lacks 32 x 3x3 native records")
    if np.isnan(cap["opacity_logit"]).any():
        raise ValueError("NaN raw opacity logit")
    # Signed infinite raw logits are valid sigmoid endpoints; activated fields
    # and native accepted contributions must remain finite.
    for field in ("opacity_unaa", "opacity_render", "alpha", "T_before",
                  "center", "scale_render", "quaternion"):
        if not np.isfinite(cap[field]).all():
            raise ValueError(f"nonfinite effective capture field: {field}")
    if np.any(cap["overflow"]) or np.max(cap["count"]) > cap["ids"].shape[1]:
        raise ValueError("incomplete native accepted record")
    H, W = map(int, cap["image_size"])
    capture_roster = camera_roster(camera_dir, W, H)
    verify_reference_camera(cap, capture_roster, "0049")
    raw_sizes = {view: Image.open(image_dir/f"{view}.png").size
                 for view in ("0049", "0012", "0050", "0058")}
    if set(raw_sizes.values()) != {(2*W, 2*H)}:
        raise ValueError("expected native raw images at exactly twice capture resolution")
    raw_W, raw_H = 2*W, 2*H
    raw_roster = camera_roster(camera_dir, raw_W, raw_H)
    cams = {view: raw_roster[view][0] for view in raw_sizes}
    imgs = {view: image_gray(image_dir/f"{view}.png", raw_W, raw_H)
            for view in cams}
    witnesses = [(cams[view], imgs[view]) for view in ("0012", "0050")]
    heldout = (cams["0058"], imgs["0058"])
    unique = cap["unique_id"]
    if not np.array_equal(unique, np.sort(unique)) or len(np.unique(unique)) != len(unique):
        raise ValueError("unique_id is not sorted and unique")
    rows = []
    for q in range(32):
        slot = q*9+4
        capture_pixel = tuple(map(float, cap["xy"][slot]))
        if int(capture_pixel[1])*W+int(capture_pixel[0]) != int(cap["seed_flat"][q]):
            raise ValueError("frozen center pixel disagrees with seed_flat")
        actual_neighbors = cap["xy"][q*9:(q+1)*9]
        expected_neighbors = np.array([(capture_pixel[0]+dx, capture_pixel[1]+dy)
                                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        if not np.array_equal(actual_neighbors, expected_neighbors):
            raise ValueError("3x3 normal neighborhood ordering mismatch")
        pixel = ((capture_pixel[0]+.5)*2-.5, (capture_pixel[1]+.5)*2-.5)
        if not np.allclose(pixel_ray(cams["0049"], *pixel),
                           pixel_ray(capture_roster["0049"][0], *capture_pixel),
                           rtol=0, atol=1e-12):
            raise ValueError("raw and captured center rays disagree")
        count = int(cap["count"][slot])
        ids = cap["ids"][slot, :count]
        if count < 0 or np.any(ids < 0) or bool(cap["overflow"][slot]):
            raise ValueError("incomplete native accepted center record")
        indexes = np.searchsorted(unique, ids)
        if np.any(indexes >= len(unique)) or not np.array_equal(unique[indexes], ids):
            raise ValueError("native ID missing original PLY row")
        members = [make_member(cap, int(row), cap["alpha"][slot, j],
                               cap["T_before"][slot, j], cams["0049"], pixel)
                   for j, row in enumerate(indexes)]
        rows.append(dict(members=members, reference_camera=cams["0049"],
                         reference_image=imgs["0049"], witnesses=witnesses,
                         pixel=pixel, expected_depth=float(cap["replay_depth"][slot])))
    # Freeze evaluable indices from the native Expected depth before inspecting
    # any candidate; candidate failures never shrink the denominator.
    base_costs = [heldout_cost(row["expected_depth"], cams["0049"], imgs["0049"],
                               heldout, row["pixel"]) for row in rows]
    fixed_eval_indices = [i for i, cost in enumerate(base_costs) if cost is not None]
    results = evaluate_fixed_32(rows)
    table = []
    for i, result in enumerate(results):
        base = base_costs[i]
        candidate = heldout_cost(result.candidate_depth, cams["0049"], imgs["0049"],
                                 heldout, result.pixel)
        table.append(dict(pixel_raw=result.pixel,
                          pixel_capture=tuple(map(float, cap["xy"][i*9+4])),
                          expected=result.expected_depth,
                          candidate=result.candidate_depth, valid=result.valid,
                          reason=result.reason, accepted_count=result.accepted_count,
                          branch_ids=result.branch_ids, witness_costs=result.costs,
                          perturb_costs={str(k): v for k, v in result.perturb_costs.items()},
                          cost_failure_counts=result.cost_failure_counts,
                          heldout_expected=base, heldout_candidate=candidate,
                          heldout_delta=None if base is None or candidate is None
                          else candidate-base))
    missing_candidate_indices = [i for i in fixed_eval_indices
                                 if table[i]["heldout_candidate"] is None]
    fixed_deltas = [table[i]["heldout_delta"] for i in fixed_eval_indices]
    fixed_mean = (float(np.mean(fixed_deltas))
                  if fixed_eval_indices and not missing_candidate_indices else None)
    valid_indices = [i for i, row in enumerate(table) if row["valid"]]
    valid_complete = all(i in fixed_eval_indices and table[i]["heldout_delta"] is not None
                         for i in valid_indices)
    valid_median = (float(np.median([table[i]["heldout_delta"] for i in valid_indices]))
                    if valid_indices and valid_complete else None)
    valid_count = sum(r["valid"] for r in table)
    return dict(capture_sha256=meta["result_sha256"], fixed_rays=32,
                valid=valid_count, coverage_gate_met=valid_count >= 8,
                heldout_fixed_eval_indices=fixed_eval_indices,
                heldout_fixed_eval_count=len(fixed_eval_indices),
                heldout_candidate_missing_on_fixed_indices=missing_candidate_indices,
                heldout_full32_mean_delta=fixed_mean if len(fixed_eval_indices) == 32 else None,
                heldout_fixed_eval_mean_delta=fixed_mean,
                valid_heldout_median_delta=valid_median,
                direction_gate="not_evaluated_no_parameter_gradient",
                rows=table)


def main():
    parser = argparse.ArgumentParser()
    for name in ("capture", "images", "cameras", "image-manifest", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    result = run(args.capture, args.images, args.cameras, args.image_manifest)
    output = Path(args.output)
    if output.exists():
        raise ValueError("output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
