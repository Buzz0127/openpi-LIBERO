#!/usr/bin/env python3
"""Keep one fixed SSH reverse proxy tunnel alive with bounded backoff."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", type=Path, default=Path("/usr/bin/ssh"))
    parser.add_argument("--host", default="openpi-libero")
    parser.add_argument("--remote-port", type=int, default=17890)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-port", type=int, default=7890)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--overall-timeout-seconds", type=float, default=90_000)
    parser.add_argument("--initial-backoff-seconds", type=float, default=5)
    parser.add_argument("--max-backoff-seconds", type=float, default=60)
    args = parser.parse_args()
    if not args.ssh.is_file():
        parser.error(f"ssh executable does not exist: {args.ssh}")
    for name in ("remote_port", "local_port"):
        value = getattr(args, name)
        if not 1 <= value <= 65_535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    if args.evidence_dir.exists():
        parser.error(f"evidence directory already exists: {args.evidence_dir}")
    if args.overall_timeout_seconds <= 0:
        parser.error("--overall-timeout-seconds must be > 0")
    if args.initial_backoff_seconds < 0 or args.max_backoff_seconds < 0:
        parser.error("backoff values must be >= 0")
    if args.initial_backoff_seconds > args.max_backoff_seconds:
        parser.error("initial backoff cannot exceed maximum backoff")
    return args


def main() -> int:
    args = parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    events_path = args.evidence_dir / "events.jsonl"
    stopping = False
    child: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        append_event(events_path, {"event": "signal_received", "signal": signum, "time_utc": utc_now()})
        if child is not None and child.poll() is None:
            child.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handle_signal)

    command = [
        str(args.ssh),
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ClearAllForwardings=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-R",
        f"127.0.0.1:{args.remote_port}:{args.local_host}:{args.local_port}",
        args.host,
    ]
    append_event(
        events_path,
        {"command": command, "event": "supervisor_start", "pid": os.getpid(), "time_utc": utc_now()},
    )
    started = time.monotonic()
    attempt = 0
    while not stopping:
        elapsed = time.monotonic() - started
        if elapsed >= args.overall_timeout_seconds:
            append_event(events_path, {"elapsed_seconds": elapsed, "event": "overall_timeout", "time_utc": utc_now()})
            return 124
        attempt += 1
        stdout_path = args.evidence_dir / f"attempt-{attempt:04d}.stdout.log"
        stderr_path = args.evidence_dir / f"attempt-{attempt:04d}.stderr.log"
        append_event(events_path, {"attempt": attempt, "event": "tunnel_start", "time_utc": utc_now()})
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=False,
            )
            returncode = child.wait()
        child = None
        append_event(
            events_path,
            {"attempt": attempt, "event": "tunnel_exit", "returncode": returncode, "time_utc": utc_now()},
        )
        if stopping:
            return 0
        delay = min(args.max_backoff_seconds, args.initial_backoff_seconds * (2 ** min(attempt - 1, 20)))
        append_event(events_path, {"attempt": attempt, "delay_seconds": delay, "event": "retry_scheduled", "time_utc": utc_now()})
        deadline = time.monotonic() + delay
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(0.25, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
