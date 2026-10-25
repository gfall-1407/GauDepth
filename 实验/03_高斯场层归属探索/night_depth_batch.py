"""Fixed frozen-model samples with verified inputs; run only under night_guard.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from sample import atomic_json, git, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXED = {
    # (model SHA256, cfg SHA256, camera count, selected image names).
    # Read only these fields from influence_G1_v1 metadata, without GT metrics.
    24: ("928670a2d4571cc9db9e221e77670d00c4ccf0add2526e87c839601de5b4388a",
         "b6def520fe83e1bf8179d537e5983260375f92d3238df2aa89401a0f8ad62b04", 49,
         ("0000", "0024", "0048")),
    37: ("34fe9d7243208432410bcaf41e44f390e41009ef3730a5f5dfd63137bbe325c1",
         "d81c37f61020b0f39d465860921229fd8e8095dc1ab894b0e4360fbf7aef8d71", 49,
         ("0000", "0024", "0048")),
    40: ("d1df2e5ac7048dc2ba3b76197bb252d59134faf4220242bfb3e39238f6e5e0d8",
         "70b2a1135b1df09d51ad979f02efeb6d153fb12787e84c0052f350dbff7a1dcb", 49,
         ("0000", "0024", "0048")),
    55: ("b0a02337236760c6eca8229491f15b834f4d3ad2bb2598801bd37aad84cf34aa",
         "25094871b264ee00e562b41d623e7376b131829531c6094c7736881b8f1270e2", 49,
         ("0000", "0024", "0048")),
    63: ("7693148889edcc6bd4377f2b2d8e9277dde0e6701fe89e97c8a146e9b08189d4",
         "f9c5bc4ce3094bdf656cb4a8a886a23e07a082b27a2e55696ec02d666a5519d4", 49,
         ("0000", "0024", "0048")),
    65: ("3e86c5d09ec4cb2cde3e915e469b0b7a6915c344f313c1a0de8c61bfa21d1c6a",
         "aaded6a2ba2c91a195a4dafbf90adab930b204f15565a89f60c9981e34d39aff", 49,
         ("0000", "0024", "0048")),
    69: ("3aaf29e9954419c1b1aebefdd36f2ae1b6503c2146f56a255c2f98887ce0e6bc",
         "0af5d1877f51d73f98d141610fb6c531d0c6f1994d56e95c64aefd598ca3e5bd", 49,
         ("0000", "0024", "0048")),
    83: ("0c32a97c920e7eef1f0b11cc0a7e74b029222246d347026f5ed2e8c3df83cc00",
         "471ce635fcc93368525d4d90f8510e94dc1497db1c7998ac5cf73223e3607d78", 64,
         ("0000", "0032", "0063")),
    97: ("19e5416fca39d187774569f446ef76dd274882662b59c88c6bb8409dd65c02e7",
         "eabe9d527759f0552d725676cb167d38c93adb607049fa2921afe11aa634d444", 64,
         ("0000", "0032", "0063")),
    105: ("01f69740d720c80db84f17b6d866ba8ea7f844fcc1b9e309a5a1d604d0f97316",
          "77a3518a8cbf5bf858c58483524d13a38acfc63e43b3a1d5f4bb451396a05f78", 64,
          ("0000", "0032", "0063")),
    106: ("1901bf477fabda932c19088e3801f95b590ed5dd8832f840fceb623c03026e76",
          "6c8fdf2529673758548496b4985f085fca6e45647cad1ffa1fa559784b5eab2d", 64,
          ("0000", "0032", "0063")),
    110: ("057aa57fae1d52c4de2e3611075d3b4ef80929fd6edad3e1d2a32a040c5a2a1e",
          "f41faab818f1d64b35f83acd71981a4170eff2384fa8543688790fa4f3010b85", 64,
          ("0000", "0032", "0063")),
    114: ("ef991dca22c5b63b2eac583be423d58cb7c485d4ff5c8710169593d7cd3350c0",
          "6c17b0e4854b506feb614c1eb4d30c527ce5ad1dbde901883e61dcfbc6f560cc", 64,
          ("0000", "0032", "0063")),
    118: ("df0d2ae37fbf718b99338453a44eb31b3f8a197e41a7b619c7562927314fcada",
          "4d88b5c778f336e315a34d7c47a648758e9abe95bbfedc7fbd1f70129430c114", 64,
          ("0000", "0032", "0063")),
    122: ("9d423236b5bf8efe5c6e30a1eb6b8572fc4a157145472fbfb8c225a19fbf5eaa",
          "c3110fc15e397fd68531daf91b3f5118df0f7ffa4e93d53e905125666e5ceec0", 64,
          ("0000", "0032", "0063")),
}
CONFIRMATION_SCANS = (37, 40, 55, 63, 69, 83, 97, 118, 122)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scans", nargs="+", type=int, choices=tuple(FIXED), default=[106, 114],
                        help="Explicit nonrepeating frozen scan IDs; defaults to the original challenge")
    parser.add_argument("--samples", type=int, default=512, help="Random rays per fixed view, 1..512")
    args = parser.parse_args()
    if len(set(args.scans)) != len(args.scans):
        parser.error("--scans must not repeat an ID")
    if not 0 < args.samples <= 512:
        parser.error("--samples must be between 1 and 512")
    if socket.gethostname() != "autodl-container-d6a7459179-693ab6ee":
        raise ValueError("Only the verified G1 is authorized")
    if git("status", "--short") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require clean committed source and approved origin")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    views_by_scan = {str(scan): [0, FIXED[scan][2] // 2, FIXED[scan][2] - 1] for scan in args.scans}
    shared_views = views_by_scan[str(args.scans[0])]
    if any(v != shared_views for v in views_by_scan.values()):
        shared_views = None
    role = ("new_ray_challenge_selected_from_prior_risk_not_blind_scene_selection"
            if args.scans == [106, 114] else
            "predeclared_remaining_scene_confirmation_no_result_based_scene_removal"
            if set(args.scans).issubset(CONFIRMATION_SCANS) else
            "explicit_frozen_scan_subset_roles_require_parent_context")
    metadata = dict(status="partial", source_commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"), hostname=socket.gethostname(),
        instance_id="d6a7459179-693ab6ee", python=sys.executable,
        sample_role=role, views=shared_views, views_by_scan=views_by_scan,
        samples_per_view=args.samples, seed=20261026, scans=args.scans,
        inputs=[], completed_scans=[],
        scan_status={str(scan): "pending" for scan in args.scans})
    atomic_json(output / "batch.json", metadata)
    try:
        for scan in args.scans:
            model_hash, cfg_hash, camera_count, expected_names = FIXED[scan]
            selected_views = views_by_scan[str(scan)]
            metadata["active_scan"] = scan
            metadata["scan_status"][str(scan)] = "checking_inputs"
            atomic_json(output / "batch.json", metadata)
            model = Path(f"/root/autodl-tmp/GauDepth/outtest/scan{scan}")
            sources = dict(model=model / "point_cloud/iteration_30000/point_cloud.ply",
                cfg=model / "cfg_args",
                cameras=Path(f"/root/autodl-tmp/DTU/scan{scan}/cameras.npz"),
                gt=Path(f"/root/autodl-tmp/DTU_MVS_Data/Points/stl/stl{scan:03d}_total.ply"))
            digests = {name: sha256(path) for name, path in sources.items()}
            if digests["model"] != model_hash or digests["cfg"] != cfg_hash:
                raise ValueError(f"Original model/config differ from verified G1 scan{scan}")
            metadata["inputs"].append(dict(scan=scan, files=[dict(kind=name, path=str(path),
                bytes=path.stat().st_size, sha256=digests[name]) for name, path in sources.items()]))
            command = [sys.executable, str(HERE / "sample.py"), "--model-path", str(model),
                "--source-path", f"/root/autodl-tmp/DTU/scan{scan}", "--iteration", "30000",
                "--split", "train", "--views", ",".join(map(str, selected_views)),
                "--samples", str(args.samples), "--seed", "20261026",
                "--output", str(output / f"scan{scan}"), "--host-build-provenance",
                "Installed original host _C retained; binary SHA checked against G1 metadata. Original build source gap remains; exact accepted-ray replay verifies compatibility, not source identity."]
            metadata["active_command"] = command
            metadata["scan_status"][str(scan)] = "sampling"
            atomic_json(output / "batch.json", metadata)
            with (output / f"scan{scan}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
            saved = json.loads((output / f"scan{scan}/metadata.json").read_text())
            if saved.get("status") != "complete" or not saved.get("original_model_unchanged"):
                raise ValueError(f"Incomplete sample scan{scan}")
            inventory = saved.get("camera_inventory", [])
            if len(inventory) != camera_count:
                raise ValueError(f"Unexpected camera count for scan{scan}: {len(inventory)} != {camera_count}")
            if ([inventory[i]["image_name"] for i in selected_views] != list(expected_names) or
                    [v["view_index"] for v in saved["views"]] != selected_views or
                    [v["image_name"] for v in saved["views"]] != list(expected_names)):
                raise ValueError("Unexpected camera inventory; do not reinterpret list indices")
            if any(v["random_samples"] != args.samples or v["qualitative_samples"] != 0
                   for v in saved["views"]):
                raise ValueError(f"Unexpected random sample count in scan{scan}")
            if sha256(sources["model"]) != model_hash or sha256(sources["cfg"]) != cfg_hash:
                raise ValueError(f"Original model/config changed during scan{scan}")
            metadata["completed_scans"].append(scan)
            metadata["scan_status"][str(scan)] = "verified_complete"
            atomic_json(output / "batch.json", metadata)
        metadata["status"] = "complete_waiting_transfer"
        atomic_json(output / "batch.json", metadata)
        # Guard's independent deadline remains in force throughout transfer.
        while not (output / "transfer_done").exists():
            time.sleep(2)
        metadata["status"] = "complete"
        atomic_json(output / "batch.json", metadata)
    except BaseException as exc:
        active_scan = metadata.get("active_scan")
        if active_scan is not None and active_scan not in metadata["completed_scans"]:
            metadata["scan_status"][str(active_scan)] = "failed_or_stopped"
        metadata.update(status="failed_or_stopped", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "batch.json", metadata)
        raise


if __name__ == "__main__":
    main()
