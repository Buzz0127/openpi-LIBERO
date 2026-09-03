#!/usr/bin/env python3
"""Retry one pinned Hugging Face download without deleting partial files.

This wrapper is intended to run *inside* the project's resource/storage guard.
It never creates a new session or process group: the guard therefore retains
ownership of the wrapper and every downloader process it starts.
"""

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


TRANSIENT_MARKERS = (
    "proxyerror",
    "remotedisconnected",
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "maxretryerror",
    "connection reset",
    "connection aborted",
    "temporary failure",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "tls",
    "ssl",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)

FATAL_MARKERS = (
    "no space left on device",
    "permission denied",
    "401 client error",
    "403 client error",
    "404 client error",
    "revision not found",
    "repository not found",
    "entry not found",
    "invalid revision",
    "checksum mismatch",
    "hash mismatch",
    "unrecognized arguments",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tail_text(path: Path, limit: int = 131_072) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def classify_failure(stderr: str) -> tuple[str, str | None]:
    lowered = stderr.lower()
    for marker in FATAL_MARKERS:
        if marker in lowered:
            return "fatal", marker
    for marker in TRANSIENT_MARKERS:
        if marker in lowered:
            return "transient", marker
    return "unknown", None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--downloader", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--overall-timeout-seconds", type=float, default=84_600)
    parser.add_argument("--initial-backoff-seconds", type=float, default=10)
    parser.add_argument("--max-backoff-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.overall_timeout_seconds <= 0:
        parser.error("--overall-timeout-seconds must be > 0")
    if args.initial_backoff_seconds < 0 or args.max_backoff_seconds < 0:
        parser.error("backoff values must be >= 0")
    if args.initial_backoff_seconds > args.max_backoff_seconds:
        parser.error("initial backoff cannot exceed maximum backoff")
    if not args.downloader.is_file():
        parser.error(f"downloader does not exist: {args.downloader}")
    if args.evidence_dir.exists():
        parser.error(f"evidence directory already exists: {args.evidence_dir}")
    return args


def main() -> int:
    args = parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    events_path = args.evidence_dir / "events.jsonl"
    stopping = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        append_event(
            events_path,
            {"event": "signal_received", "signal": signum, "time_utc": utc_now()},
        )

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handle_signal)

    command = [
        str(args.downloader),
        "download",
        args.repo_id,
        "--repo-type",
        args.repo_type,
        "--revision",
        args.revision,
        "--local-dir",
        str(args.local_dir),
        "--max-workers",
        str(args.max_workers),
    ]
    append_event(
        events_path,
        {
            "command": command,
            "event": "wrapper_start",
            "max_attempts": args.max_attempts,
            "overall_timeout_seconds": args.overall_timeout_seconds,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "time_utc": utc_now(),
        },
    )
    started = time.monotonic()

    for attempt in range(1, args.max_attempts + 1):
        elapsed = time.monotonic() - started
        if stopping:
            return 128 + signal.SIGTERM
        if elapsed >= args.overall_timeout_seconds:
            append_event(
                events_path,
                {"attempt": attempt, "elapsed_seconds": elapsed, "event": "overall_timeout", "time_utc": utc_now()},
            )
            return 124

        stdout_path = args.evidence_dir / f"attempt-{attempt:04d}.stdout.log"
        stderr_path = args.evidence_dir / f"attempt-{attempt:04d}.stderr.log"
        append_event(
            events_path,
            {"attempt": attempt, "elapsed_seconds": elapsed, "event": "attempt_start", "time_utc": utc_now()},
        )
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=False,
            )
            while child.poll() is None:
                if stopping:
                    # The enclosing guard signals the entire owned process group,
                    # including this child. Wait briefly so it can be reaped.
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                    return 128 + signal.SIGTERM
                time.sleep(0.25)
            returncode = child.wait()

        elapsed = time.monotonic() - started
        if returncode == 0:
            append_event(
                events_path,
                {"attempt": attempt, "elapsed_seconds": elapsed, "event": "completed", "returncode": 0, "time_utc": utc_now()},
            )
            return 0

        classification, marker = classify_failure(tail_text(stderr_path))
        append_event(
            events_path,
            {
                "attempt": attempt,
                "classification": classification,
                "elapsed_seconds": elapsed,
                "event": "attempt_failed",
                "matched_marker": marker,
                "returncode": returncode,
                "time_utc": utc_now(),
            },
        )
        if classification != "transient" or attempt == args.max_attempts:
            return returncode if returncode != 0 else 1

        delay = min(
            args.max_backoff_seconds,
            args.initial_backoff_seconds * (2 ** min(attempt - 1, 20)),
        )
        remaining = args.overall_timeout_seconds - elapsed
        if delay >= remaining:
            append_event(
                events_path,
                {"attempt": attempt, "event": "insufficient_time_for_retry", "remaining_seconds": remaining, "time_utc": utc_now()},
            )
            return 124
        append_event(
            events_path,
            {"attempt": attempt, "delay_seconds": delay, "event": "retry_scheduled", "time_utc": utc_now()},
        )
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if stopping:
                return 128 + signal.SIGTERM
            time.sleep(min(0.25, deadline - time.monotonic()))

    return 1


if __name__ == "__main__":
    sys.exit(main())
