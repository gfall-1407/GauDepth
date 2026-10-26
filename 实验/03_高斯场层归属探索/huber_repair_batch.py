#!/usr/bin/env python3
"""One finite, predeclared G0 frozen-cache repair batch, managed by night_guard."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
INSTANCE = "5d5147b55f-67549ee6"
BAD = (40, 63, 83, 97, 105)
OTHER = (24, 37, 55, 65, 69, 106, 110, 114, 118, 122)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--counterexamples-only", action="store_true",
                   help="Only fixed new counterexamples37/110, complementary region controls; no new formula")
    p.add_argument("--tail-control-only", action="store_true",
                   help="Only37/110 diagnostic: omit large-tail shrink, keep same1mm small-delta restoration")
    args = p.parse_args()
    if args.counterexamples_only and args.tail_control_only:
        p.error("Choose one finite diagnostic batch")
    root = args.root.resolve()
    if not args.root.is_absolute() or socket.gethostname() != "autodl-container-" + INSTANCE:
        raise ValueError("Require the verified G0 and an absolute independent result root")
    controller = root / ("tail_control_batch.json" if args.tail_control_only else
                         "counterexamples_batch.json" if args.counterexamples_only else "batch.json")
    if controller.exists():
        raise ValueError("Do not replay a previous batch")
    state = dict(status="waiting_inputs", pid=os.getpid(), started=time.time(), completed=[])

    def save(status, **extra):
        state.update(status=status, updated=time.time(), **extra)
        tmp = controller.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(controller)

    def stopped():
        return (root / "stop_batch").exists()

    signal.signal(signal.SIGTERM, lambda *_: (root / "stop_batch").touch())

    def wait_marker(name):
        while not (root / name).exists():
            if stopped():
                raise InterruptedError("Requested stop at saved task boundary")
            time.sleep(2)

    def task(scan, group, methods=None):
        if stopped():
            raise InterruptedError("Requested stop at saved task boundary")
        output = root / group / f"scan{scan}_v1"
        command = [sys.executable, str(HERE / "dense_huber_mesh.py"), "--scan", str(scan),
                   "--instance-id", INSTANCE, "--model-root", str(root / "inputs_G1/outtest"),
                   "--output", str(output)]
        if methods is None:
            command += ["--cache-only"]
        else:
            command += ["--cache-root", str(root / "cache" / f"scan{scan}_v1"), "--methods", *methods]
        save("running", scan=scan, group=group, methods=methods, command=command)
        log = root / "logs" / f"{group}_{scan}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("xb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        expected = "cache_complete" if methods is None else "complete"
        if result.returncode != 0 or metadata["status"] != expected:
            raise RuntimeError(f"{group}/{scan} failed or partial: {result.returncode}/{metadata['status']}")
        state["completed"].append(dict(scan=scan, group=group, methods=methods, metadata=str(output / "metadata.json")))
        save("task_complete")

    try:
        save("waiting_inputs")
        wait_marker("inputs_ready")
        if args.tail_control_only:
            if json.loads((root / "counterexamples_batch.json").read_text())["status"] != "complete_waiting_transfer":
                raise ValueError("Region counterexample controls must finish first")
            for scan in (37, 110):
                task(scan, "tail_control", ["tail_unshrunk_control"])
            save("complete_waiting_transfer")
            wait_marker("tail_control_transfer_verified")
            save("complete")
            return
        if args.counterexamples_only:
            if json.loads((root / "batch.json").read_text())["status"] != "complete_waiting_transfer":
                raise ValueError("Original full15 batch must finish before extra controls")
            if json.loads((root / "bridge_review.json").read_text())["status"] != "PASS_exact":
                raise ValueError("Original bridge did not pass")
            for scan in (37, 110):
                task(scan, "counterexamples", ["edge_restore", "interior_restore"])
            save("complete_waiting_transfer")
            wait_marker("counterexamples_transfer_verified")
            save("complete")
            return
        task(40, "cache")
        task(40, "bridge", ["native_expected", "huber_1mm"])
        save("awaiting_bridge_review")
        wait_marker("bridge_verified")
        for scan in BAD:
            if scan != 40:
                task(scan, "cache")
            task(scan, "causal", ["edge_restore", "interior_restore", "soft_tail_1mm"])
        save("awaiting_five_review")
        wait_marker("five_verified")
        for scan in OTHER:
            task(scan, "cache")
            task(scan, "confirmation", ["soft_tail_1mm"])
        save("complete_waiting_transfer")
        wait_marker("transfer_verified")
        save("complete")
    except BaseException as exc:
        save("stopped_or_failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
