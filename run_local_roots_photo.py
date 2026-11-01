"""Run frozen local_roots_photo_v1 on the fresh scan110 CPU capture."""

import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path
import time

import numpy as np
from PIL import Image

from local_roots_photo import OFFSETS, decide, finalize_root, validated_root
from visible_branch_reference import Member, patch_cost_diagnostic, pixel_ray, project
from visible_branch_runner import (EXPECTED_CAPTURE, camera_roster, digest,
                                   image_gray, require_hash, verify_reference_camera)


GEOMETRY_SHA = "0b3a600725142558e69fde7aa45d3f2eaa0b715d487d59d784207a086e4a73bf"
GEOMETRY_COMMIT = "63f2beface9d37203058550b4a238f54cb2d5323"
OLD_EVAL_SHA = "a21849ef97fb865d9d324009f39d58d4907a42b10129102edfea048cdc89156f"
PROTOCOL = "local_roots_photo_v1"
S = tuple(i for i in range(32) if i != 23)


def plane_cost(normal, depth, reference, reference_image, target, pixel):
    ray = pixel_ray(reference, *pixel)
    center = tuple(o+depth*v for o, v in zip(reference.center, ray))
    plane = Member(-1, center, tuple(normal), .5, 1., 1.)
    camera, image = target
    return patch_cost_diagnostic(plane, reference, reference_image,
                                 camera, image, *pixel)


def heldout_summary(rays):
    """Missing formal deltas retain the frozen 31-index denominator."""
    known, missing = [], []
    for i in S:
        row = rays[i]
        if row["selected_root"] is None:
            known.append(0.)
        elif row["heldout"]["position_delta"] is None:
            missing.append(i)
        else:
            known.append(row["heldout"]["position_delta"])
    total = sum(known)
    return dict(fixed_indices=list(S), denominator=31,
                known_count=len(known), missing_indices=missing,
                formal_mean=total/31 if not missing else None,
                formal_mean_bounds=[(total-len(missing))/31,
                                    (total+len(missing))/31])


def run(geometry_path, capture_dir, image_dir, camera_dir, image_manifest,
        old_eval_path, selection_path, output_path):
    started = time.monotonic()
    geometry_path, capture_dir, image_dir, camera_dir = map(
        Path, (geometry_path, capture_dir, image_dir, camera_dir))
    selection_path, output_path = Path(selection_path), Path(output_path)
    journal_path = selection_path.with_suffix(".jsonl")
    if selection_path.exists() or output_path.exists() or journal_path.exists():
        raise ValueError("selection and output paths must be new")
    require_hash(geometry_path, GEOMETRY_SHA)
    require_hash(old_eval_path, OLD_EVAL_SHA)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    old_eval = json.loads(Path(old_eval_path).read_text(encoding="utf-8"))
    if (len(geometry["rays"]) != 32 or geometry["git_commit"] !=
            "39160bfaab4090b985026884d6869391cfec3a08" or
            geometry["enrichment"]["git_commit"] !=
            "e32ce8f9e1c22cd8e04021f062dbd63d8d7a7b1c" or
            geometry["capture_sha256"] != EXPECTED_CAPTURE["result_sha256"]):
        raise ValueError("geometry source identity mismatch")
    if sum(len(row["grid_512"]["roots"]) for row in geometry["rays"]) != 870:
        raise ValueError("full source root universe changed")
    if tuple(old_eval["heldout_fixed_eval_indices"]) != S:
        raise ValueError("native fixed heldout evaluable set mismatch")
    meta = json.loads((capture_dir/"metadata.json").read_text(encoding="utf-8"))
    for field, value in EXPECTED_CAPTURE.items():
        if meta.get(field) != value:
            raise ValueError(f"capture identity mismatch: {field}")
    npz_path = capture_dir/"native_members.npz"
    require_hash(npz_path, meta["result_sha256"])
    with np.load(npz_path, allow_pickle=False) as data:
        cap = {key: data[key] for key in data.files}
    H, W = map(int, cap["image_size"])
    if (H, W) != (581, 777):
        raise ValueError("unexpected capture resolution")
    for name in ("cameras.bin", "images.bin"):
        expected = next(v for k, v in meta["colmap_sha256"].items()
                        if Path(k).name == name)
        require_hash(camera_dir/name, expected)
    image_meta = json.loads(Path(image_manifest).read_text(encoding="utf-8"))
    for view in ("0049", "0012", "0050", "0058"):
        require_hash(image_dir/f"{view}.png",
                     image_meta["targets"][view]["image_sha256"])
        with Image.open(image_dir/f"{view}.png") as image:
            if image.size != (1554, 1162):
                raise ValueError("unexpected raw image size")
    capture_roster = camera_roster(camera_dir, W, H)
    verify_reference_camera(cap, capture_roster, "0049")
    raw_roster = camera_roster(camera_dir, 1554, 1162)
    cameras = {v: raw_roster[v][0] for v in ("0049", "0012", "0050", "0058")}
    if (cameras["0049"].cx, cameras["0049"].cy) != (776.5, 580.5):
        raise ValueError("incorrect raw host principal point")
    images = {v: image_gray(image_dir/f"{v}.png", 1554, 1162)
              for v in ("0049", "0012", "0050")}
    ref = cameras["0049"]
    witnesses = {v: (cameras[v], images[v]) for v in ("0012", "0050")}
    selection = dict(protocol=PROTOCOL, geometry_sha256=GEOMETRY_SHA,
                     geometry_commit=GEOMETRY_COMMIT,
                     geometry_source_commit=geometry["git_commit"],
                     capture_sha256=meta["result_sha256"],
                     native_eval_sha256=OLD_EVAL_SHA,
                     reference="0049", witnesses=["0012", "0050"],
                     heldout="0058", color_space="gamma_rgb_luma",
                     fixed_ray_count=32, source_root_count=870,
                     rays=[])
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal = journal_path.open("x", encoding="utf-8")
    for source_ray in geometry["rays"]:
        if time.monotonic()-started > 600:
            raise TimeoutError("local roots photo CPU deadline 600 s")
        i = source_ray["center_index"]
        if i != len(selection["rays"]):
            raise ValueError("geometry ray order changed")
        q = i*9+4
        pixel_capture = tuple(map(float, cap["xy"][q]))
        if source_ray["query_index"] != q or source_ray["flat_pixel"] != int(cap["seed_flat"][i]):
            raise ValueError("geometry/capture ray mismatch")
        pixel_raw = (2*pixel_capture[0]+.5, 2*pixel_capture[1]+.5)
        if not np.allclose(pixel_ray(ref, *pixel_raw),
                           pixel_ray(capture_roster["0049"][0], *pixel_capture),
                           rtol=0, atol=1e-12):
            raise ValueError("raw/capture ray mismatch")
        ray = pixel_ray(ref, *pixel_raw)
        ray_length = sqrt(sum(v*v for v in ray))
        expected = float(cap["replay_depth"][q])
        expected_z = expected/ray_length
        if abs(expected-source_ray["native_expected"]) > 2e-6:
            raise ValueError("geometry native Expected mismatch")
        candidates = []
        for j, source_root in enumerate(source_ray["grid_512"]["roots"]):
            if not source_root["matched_admissible"]:
                continue
            root_id = validated_root(source_root, i, j)
            z = source_root["z"]
            normal = source_root["surface_normal_world"]
            point = tuple(o+z*v for o, v in zip(ref.center, ray))
            plane = Member(-1, point, tuple(normal), .5, 1., 1.)
            evidence = {}
            for view, (camera, image) in witnesses.items():
                center = project(camera, point)
                samples = []
                for dx, dy in OFFSETS:
                    cost, status = patch_cost_diagnostic(
                        plane, ref, images["0049"], camera, image,
                        pixel_raw[0]+dx, pixel_raw[1]+dy)
                    samples.append(dict(offset=[dx, dy], cost=cost,
                                        reason=status, projected_center=center,
                                        visibility="unverified"))
                evidence[view] = samples
            candidates.append(finalize_root(root_id, z, point, normal,
                                            bool(source_root["outside_all_3sigma"]),
                                            evidence))
        decision = decide(candidates, expected_z)
        decision["output_depth_native_metric"] = decision["output_depth"]*ray_length
        selection["rays"].append(dict(center_index=i, query_index=q,
                                      pixel_capture=pixel_capture, pixel_raw=pixel_raw,
                                      native_expected=expected,
                                      native_expected_camera_z=expected_z,
                                      ray_length=ray_length,
                                      roots=candidates, **decision))
        journal.write(json.dumps(selection["rays"][-1], allow_nan=False)+"\n")
        journal.flush()
    journal.close()
    if sum(len(r["roots"]) for r in selection["rays"]) != 199:
        raise ValueError("199 matched_admissible roots not preserved")
    flags = [root["root_id"] for row in selection["rays"] for root in row["roots"]
             if root["outside_all_3sigma"]]
    if flags != [[8, 22], [21, 20]]:
        raise ValueError("outside-3sigma root flags changed")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False),
                              encoding="utf-8")
    selection_sha = digest(selection_path)
    # Only now open 0058: heldout is barred from all prior decisions.
    heldout = (cameras["0058"], image_gray(image_dir/"0058.png", 1554, 1162))
    for row in selection["rays"]:
        best_id = row["selected_root"]
        info = dict(fixed_native_eligible=row["center_index"] in S,
                    selected_plane_cost=None, same_normal_expected_cost=None,
                    position_delta=None, missing_reason=None)
        if best_id is not None:
            best = next(r for r in row["roots"] if r["root_id"] == best_id)
            normal = best["surface_normal_world"]
            candidate, a_reason = plane_cost(normal, best["z"], ref,
                                             images["0049"], heldout, row["pixel_raw"])
            baseline, b_reason = plane_cost(normal, row["native_expected_camera_z"], ref,
                                            images["0049"], heldout, row["pixel_raw"])
            info.update(selected_plane_cost=candidate,
                        same_normal_expected_cost=baseline,
                        position_delta=None if candidate is None or baseline is None
                        else candidate-baseline,
                        missing_reason=None if candidate is not None and baseline is not None
                        else dict(candidate=a_reason, baseline=b_reason))
        row["heldout"] = info
    counts = Counter(r["decision"] for r in selection["rays"])
    result = dict(protocol=PROTOCOL, selection_sha256=selection_sha,
                  geometry_sha256=GEOMETRY_SHA, capture_sha256=meta["result_sha256"],
                  native_eval_sha256=OLD_EVAL_SHA, fixed_ray_count=32,
                  matched_candidate_count=199, decision_counts=dict(counts),
                  selected_count=counts["selected"],
                  heldout=heldout_summary(selection["rays"]),
                  seconds=time.monotonic()-started,
                  limitation="frozen forward only; no Gaussian gradient or training; 0058 was selection-heldout but may have trained original 7k model",
                  rays=selection["rays"])
    if result["seconds"] > 600:
        raise TimeoutError("local roots photo CPU deadline 600 s")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False),
                           encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("geometry", "capture", "images", "cameras", "image-manifest",
                 "old-eval", "selection", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    result = run(a.geometry, a.capture, a.images, a.cameras, a.image_manifest,
                 a.old_eval, a.selection, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rays"},
                     allow_nan=False))


if __name__ == "__main__":
    main()
