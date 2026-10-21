#!/usr/bin/env python3
"""Small Linux-only deadline guard for one explicitly managed child process."""

import argparse
import datetime as dt
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zoneinfo import ZoneInfo


EXPECTED_HOSTNAME = "autodl-container-d6a7459179-693ab6ee"
EXPECTED_INSTANCE_ID = "d6a7459179-693ab6ee"
POLL_SECONDS = 2
CHILD_EXIT_LINGER = 120
SHUTDOWN_RESERVE = 15


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
        if os.name == "posix":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_identity(expected_hostname, instance_id, dry_run=False, observed=None):
    if expected_hostname != EXPECTED_HOSTNAME or instance_id != EXPECTED_INSTANCE_ID:
        raise ValueError("仅允许指定的 GauDepth1 hostname 与完整实例 ID")
    if "gaudepth2" in expected_hostname.lower() or "gaudepth2" in instance_id.lower():
        raise ValueError("GauDepth2 被禁止")
    actual = socket.gethostname() if observed is None else observed
    if not dry_run and not sys.platform.startswith("linux"):
        raise ValueError("真实模式仅支持 Linux")
    if actual != expected_hostname:
        raise ValueError(f"hostname 不匹配：实际 {actual!r}，要求 {expected_hostname!r}")


def parse_deadline(value, hard_stop, require_future=True):
    try:
        deadline = float(value)
    except (TypeError, ValueError):
        raise ValueError("deadline 必须是 Unix 时间戳")
    if not (math.isfinite(deadline) and math.isfinite(hard_stop) and deadline <= hard_stop and (deadline > time.time() or not require_future)):
        raise ValueError("deadline 必须有限、晚于当前时间且不得越过本批硬截止")
    return deadline


def latest_safe_deadline(grace,hard_stop):
    return hard_stop - grace - SHUTDOWN_RESERVE


def guard_is_alive(state):
    """Check PID liveness and, on Linux, confirm it is running this guard's run mode."""
    pid = state.get("guard_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if sys.platform.startswith("linux"):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        except OSError:
            return False
        return any(arg.endswith(b"night_guard.py") for arg in argv) and b"run" in argv
    return True


def log_event(state_path, event, **fields):
    state_path = Path(state_path)
    record = read_json(state_path) if state_path.exists() else {}
    record.update(fields)
    record["last_event"] = event
    record["updated_at"] = dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    atomic_json(state_path, record)


def find_supervisors(proc_root="/proc"):
    """Return PIDs whose argv exactly identifies the approved platform daemon."""
    matches = []
    try:
        entries = os.scandir(proc_root)
    except OSError:
        return matches
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = Path(entry.path, "cmdline").read_bytes()
            except OSError:
                continue
            argv = raw.rstrip(b"\0").split(b"\0")
            if argv == [b"/bin/supervisord", b"-c", b"/init/supervisor/supervisor.ini"]:
                matches.append(int(entry.name))
    return matches


def stop_managed_child(child, grace, state_path):
    if child.poll() is not None and not process_group_exists(child.pid):
        return
    log_event(state_path, "term_sent", child_pid=child.pid)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    until = time.monotonic() + grace
    while process_group_exists(child.pid) and time.monotonic() < until:
        time.sleep(0.2)
    if process_group_exists(child.pid):
        log_event(state_path, "kill_sent", child_pid=child.pid)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()
    log_event(state_path, "child_stopped", child_returncode=child.returncode)


def process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def command_run(args):
    state_dir = Path(args.state_dir)
    deadline_path = state_dir / "deadline.json"
    state_path = state_dir / "status.json"
    if deadline_path.exists() or state_path.exists():
        raise ValueError("状态目录已有 deadline.json 或 status.json，拒绝覆盖")
    if args.simulate_hostname and not args.dry_run:
        raise ValueError("--simulate-hostname 只允许 dry-run")
    observed = args.simulate_hostname or None
    validate_identity(args.expected_hostname, args.instance_id, args.dry_run, observed)
    if not args.command:
        raise ValueError("run 后必须提供受控子进程命令")
    if args.grace < 0 or args.grace > 3600:
        raise ValueError("保存宽限须在 0 到 3600 秒之间")
    if not (0 < args.minutes <= args.max_minutes <= 1440):
        raise ValueError("须满足 0 < 初始分钟数 <= 本批最长分钟数 <= 1440")
    started=time.time()
    hard_stop=started+args.max_minutes*60+args.grace+SHUTDOWN_RESERVE
    deadline = parse_deadline(args.deadline_at if args.deadline_at is not None else started + args.minutes * 60,hard_stop)
    deadline = min(deadline, latest_safe_deadline(args.grace,hard_stop))
    if deadline <= time.time():
        raise ValueError("已无足够时间留出保存宽限与停机操作")
    if args.shutdown_command:
        raise ValueError("不调用平台 shutdown 脚本；仅允许核验后停止唯一 supervisord")

    atomic_json(deadline_path, {"deadline": deadline, "hard_stop": hard_stop})
    atomic_json(state_path, {
        "mode": "dry-run" if args.dry_run else "real",
        "guard_pid": os.getpid(),
        "hostname": observed or socket.gethostname(),
        "instance_id": args.instance_id,
        "child_pid": None,
        "command": args.command,
        "shutdown_method": "verified-supervisord-sigterm",
        "grace_seconds": args.grace,
        "deadline": deadline,
        "hard_stop": hard_stop,
        "max_minutes":args.max_minutes,
        "last_event": "armed",
        "updated_at": dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    })
    child = None
    if not args.dry_run:
        child = subprocess.Popen(args.command, start_new_session=True)
        try:
            log_event(state_path, "child_started", child_pid=child.pid)
        except OSError:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
    print(f"armed; deadline={dt.datetime.fromtimestamp(deadline, ZoneInfo('Asia/Shanghai')).isoformat()}", flush=True)

    last_valid_deadline = deadline
    invalid_deadline_logged = False
    child_exit_at = None
    while True:
        try:
            current = read_json(deadline_path)
            if not isinstance(current,dict):
                raise ValueError('deadline 文件必须为 JSON 对象')
            observed_deadline = parse_deadline(current.get("deadline"), hard_stop, require_future=False)
            last_valid_deadline = observed_deadline
            invalid_deadline_logged = False
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if not invalid_deadline_logged:
                log_event(state_path, "deadline_file_invalid_using_last_valid", deadline_error=str(exc))
                invalid_deadline_logged = True
        now = time.time()
        if child is not None and child_exit_at is None and child.poll() is not None:
            child_exit_at = now
            log_event(state_path, "child_exited", child_returncode=child.returncode,
                      child_exit_linger_seconds=CHILD_EXIT_LINGER)
        stop_at = min(last_valid_deadline, latest_safe_deadline(args.grace,hard_stop))
        if child_exit_at is not None:
            stop_at = min(stop_at, child_exit_at + CHILD_EXIT_LINGER)
        if now >= stop_at:
            break
        time.sleep(min(POLL_SECONDS, max(0.1, stop_at - now)))

    log_event(state_path, "deadline_reached")
    if child is not None and (child.poll() is None or process_group_exists(child.pid)):
        remaining = max(0, hard_stop - time.time() - SHUTDOWN_RESERVE)
        stop_managed_child(child, min(args.grace, remaining), state_path)
    elif args.dry_run:
        log_event(state_path, "dry_run_would_stop_child")

    if args.dry_run:
        log_event(state_path, "dry_run_would_sync_and_shutdown")
        print("dry-run complete; no signal, sync, or shutdown executed", flush=True)
        return 0

    log_event(state_path, "sync_started")
    os.sync()
    try:
        validate_identity(args.expected_hostname, args.instance_id)
        matches = find_supervisors()
        if len(matches) != 1:
            log_event(state_path, "shutdown_failed", shutdown_error=f"expected one approved supervisord, found {len(matches)}")
            print("refused: approved supervisord was not unique", file=sys.stderr, flush=True)
            return 1
        # Re-read the selected PID's argv immediately before signaling it.
        pid = matches[0]
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if argv != [b"/bin/supervisord", b"-c", b"/init/supervisor/supervisor.ini"]:
            log_event(state_path, "shutdown_failed", shutdown_error="supervisord identity changed before signal")
            return 1
        log_event(state_path, "shutdown_term_sent", supervisor_pid=pid)
        os.kill(pid, signal.SIGTERM)
        log_event(state_path, "shutdown_signal_sent", supervisor_pid=pid)
        print("SIGTERM sent to the uniquely verified supervisord; control-panel shutdown not yet verified", flush=True)
        return 0
    except (OSError, ValueError) as exc:
        log_event(state_path, "shutdown_failed", shutdown_error=str(exc))
        print(f"shutdown failed: {exc}", file=sys.stderr, flush=True)
        return 1


def command_renew(args):
    validate_identity(args.expected_hostname, args.instance_id, args.dry_run)
    if args.minutes <= 0 or args.minutes > 1440:
        raise ValueError("续期须为 1 到 1440 分钟")
    state_path = Path(args.state_dir) / "status.json"
    state = read_json(state_path)
    hard_stop=float(state['hard_stop'])
    if not math.isfinite(hard_stop):raise ValueError('本批硬截止无效')
    if not guard_is_alive(state):
        raise ValueError("没有匹配的活动 guard 进程，拒绝续期")
    requested_deadline = time.time() + args.minutes * 60
    deadline = min(requested_deadline, latest_safe_deadline(state.get("grace_seconds", 120),hard_stop))
    if deadline <= time.time():
        raise ValueError("已无足够时间留出保存宽限与停机操作，拒绝续期")
    parse_deadline(deadline,hard_stop)
    atomic_json(Path(args.state_dir) / "deadline.json", {"deadline": deadline, "hard_stop": hard_stop})
    log_event(state_path, "renewed", deadline=deadline)
    print(f"deadline={dt.datetime.fromtimestamp(deadline, ZoneInfo('Asia/Shanghai')).isoformat()}")
    return 0


def command_status(args):
    state_dir = Path(args.state_dir)
    print(json.dumps({"status": read_json(state_dir / "status.json"),
                      "deadline": read_json(state_dir / "deadline.json")}, ensure_ascii=False, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="启动并监护一个受控子进程")
    run.add_argument("--state-dir", required=True)
    run.add_argument("--minutes", type=int, default=30)
    run.add_argument("--max-minutes",type=int,default=60,help="本批最长计算分钟数；硬截止含保存宽限与停机余量")
    run.add_argument("--grace", type=int, default=120)
    run.add_argument("--expected-hostname", required=True)
    run.add_argument("--instance-id", required=True)
    run.add_argument("--shutdown-command", nargs="+", help=argparse.SUPPRESS)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--simulate-hostname", help="仅供本地 dry-run 的身份模拟")
    run.add_argument("--deadline-at", type=float, help="显式 Unix 截止时间；仍受硬截止约束")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=command_run)
    renew = sub.add_parser("renew", help="原子更新期限，受硬截止限制")
    renew.add_argument("--state-dir", required=True)
    renew.add_argument("--minutes", type=int, default=60)
    renew.add_argument("--expected-hostname", required=True)
    renew.add_argument("--instance-id", required=True)
    renew.add_argument("--dry-run", action="store_true")
    renew.set_defaults(func=command_renew)
    status = sub.add_parser("status", help="读取期限和最近动作")
    status.add_argument("--state-dir", required=True)
    status.set_defaults(func=command_status)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "run" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return args.func(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"拒绝或失败：{exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
