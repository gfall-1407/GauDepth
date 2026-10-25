"""One fixed frozen-model challenge sample; run only under night_guard.py."""
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
    106: ("1901bf477fabda932c19088e3801f95b590ed5dd8832f840fceb623c03026e76",
          "6c8fdf2529673758548496b4985f085fca6e45647cad1ffa1fa559784b5eab2d"),
    114: ("ef991dca22c5b63b2eac583be423d58cb7c485d4ff5c8710169593d7cd3350c0",
          "6c17b0e4854b506feb614c1eb4d30c527ce5ad1dbde901883e61dcfbc6f560cc"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "autodl-container-d6a7459179-693ab6ee":
        raise ValueError("Only the verified G1 is authorized")
    if git("status", "--short") or git("remote", "get-url", "origin") != "git@github.com:gfall-1407/GauDepth.git":
        raise ValueError("Require clean committed source and approved origin")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    metadata = dict(status="partial", source_commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"), hostname=socket.gethostname(),
        instance_id="d6a7459179-693ab6ee", python=sys.executable,
        sample_role="new_ray_challenge_selected_from_prior_risk_not_blind_scene_selection",
        views=[0, 32, 63], samples_per_view=512, seed=20261026,
        scans=list(FIXED), inputs=[], completed_scans=[])
    atomic_json(output / "batch.json", metadata)
    try:
        for scan, (model_hash, cfg_hash) in FIXED.items():
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
                "--split", "train", "--views", "0,32,63", "--samples", "512", "--seed", "20261026",
                "--output", str(output / f"scan{scan}"), "--host-build-provenance",
                "Installed original host _C retained; binary SHA checked against G1 metadata. Original build source gap remains; exact accepted-ray replay verifies compatibility, not source identity."]
            metadata["active_command"] = command
            atomic_json(output / "batch.json", metadata)
            with (output / f"scan{scan}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
            saved = json.loads((output / f"scan{scan}/metadata.json").read_text())
            if saved.get("status") != "complete" or not saved.get("original_model_unchanged"):
                raise ValueError(f"Incomplete sample scan{scan}")
            if [v["image_name"] for v in saved["views"]] != ["0000", "0032", "0063"]:
                raise ValueError("Unexpected camera inventory; do not reinterpret list indices")
            metadata["completed_scans"].append(scan)
            atomic_json(output / "batch.json", metadata)
        metadata["status"] = "complete_waiting_transfer"
        atomic_json(output / "batch.json", metadata)
        # Guard's independent deadline remains in force throughout transfer.
        while not (output / "transfer_done").exists():
            time.sleep(2)
        metadata["status"] = "complete"
        atomic_json(output / "batch.json", metadata)
    except BaseException as exc:
        metadata.update(status="failed_or_stopped", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "batch.json", metadata)
        raise


if __name__ == "__main__":
    main()
