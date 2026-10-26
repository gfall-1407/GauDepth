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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="Read-only original RGB archive on guarded G0; no model or CUDA execution")
    e.add_argument("--scans", nargs="+", type=int, choices=(37, 40), default=[37, 40])
    e.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/DTU"))
    e.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export_observations(args)


if __name__ == "__main__":
    main()
