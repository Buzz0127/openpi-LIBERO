#!/usr/bin/env python3
"""Run one bounded CPU stage in its own process group and reap it safely.

All resource thresholds are disabled unless explicitly supplied.  G1a uses
only short synthetic commands to verify the state machine; it does not enable
new shared-host thresholds for real workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXIT_TIMEOUT = 124
EXIT_GUARD_STOP = 125
EXIT_SPAWN_FAILURE = 126


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence file: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_mem_available_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _read_child_rss_bytes(pid: int) -> int:
    with Path(f"/proc/{pid}/status").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                fields = line.split()
                return int(fields[1]) * 1024
    raise RuntimeError(f"VmRSS not found for child pid {pid}")


def _sample(pid: int) -> dict[str, Any]:
    load1 = os.getloadavg()[0]
    cpu_count = os.cpu_count() or 1
    return {
        "monotonic_seconds": time.monotonic(),
        "wall_time_epoch_seconds": time.time(),
        "child_rss_bytes": _read_child_rss_bytes(pid),
        "mem_available_bytes": _read_mem_available_bytes(),
        "load1": load1,
        "load1_per_cpu": load1 / cpu_count,
        "logical_cpu_count": cpu_count,
    }


def _verified_group_alive(proc: subprocess.Popen[Any], expected_pgid: int) -> bool:
    if proc.poll() is not None:
        return False
    try:
        return os.getpgid(proc.pid) == expected_pgid == proc.pid
    except ProcessLookupError:
        return False


def _terminate_own_group(
    proc: subprocess.Popen[Any], expected_pgid: int, term_grace: float, kill_grace: float
) -> dict[str, Any]:
    result = {
        "ownership_verified": False,
        "term_sent": False,
        "kill_sent": False,
        "wait_reaped": False,
    }
    if _verified_group_alive(proc, expected_pgid):
        result["ownership_verified"] = True
        os.killpg(expected_pgid, signal.SIGTERM)
        result["term_sent"] = True
    try:
        proc.wait(timeout=term_grace)
        result["wait_reaped"] = True
        return result
    except subprocess.TimeoutExpired:
        pass

    if _verified_group_alive(proc, expected_pgid):
        os.killpg(expected_pgid, signal.SIGKILL)
        result["kill_sent"] = True
    try:
        proc.wait(timeout=kill_grace)
        result["wait_reaped"] = True
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("owned child process group did not exit after SIGKILL") from error
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--term-grace-seconds", type=float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=5.0)
    parser.add_argument("--max-child-rss-bytes", type=int)
    parser.add_argument("--min-mem-available-bytes", type=int)
    parser.add_argument("--max-load1-per-cpu", type=float)
    parser.add_argument("--resource-consecutive-samples", type=int, default=3)
    parser.add_argument("--monitor-failure-consecutive-samples", type=int, default=2)
    parser.add_argument("--simulate-monitor-failure-after", type=int, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    for name in ("timeout_seconds", "sample_interval_seconds", "term_grace_seconds", "kill_grace_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.resource_consecutive_samples < 1 or args.monitor_failure_consecutive_samples < 1:
        parser.error("consecutive-sample counts must be positive")
    return args


def main() -> int:
    args = _parse_args()
    args.attempt_dir.mkdir(parents=True, exist_ok=False)
    temporary_dir = args.attempt_dir / "tmp"
    cache_dir = args.attempt_dir / "cache"
    pycache_dir = args.attempt_dir / "pycache"
    temporary_dir.mkdir()
    cache_dir.mkdir()
    pycache_dir.mkdir()

    child_environment = os.environ.copy()
    child_environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(pycache_dir),
            "TMPDIR": str(temporary_dir),
            "TMP": str(temporary_dir),
            "TEMP": str(temporary_dir),
            "XDG_CACHE_HOME": str(cache_dir),
        }
    )
    thresholds = {
        "max_child_rss_bytes": args.max_child_rss_bytes,
        "min_mem_available_bytes": args.min_mem_available_bytes,
        "max_load1_per_cpu": args.max_load1_per_cpu,
        "resource_consecutive_samples": args.resource_consecutive_samples,
        "monitor_failure_consecutive_samples": args.monitor_failure_consecutive_samples,
    }
    start_wall = time.time()
    start_monotonic = time.monotonic()
    stop_signal: list[int] = []

    def _record_signal(signum: int, _frame: Any) -> None:
        if not stop_signal:
            stop_signal.append(signum)

    previous_handlers = {
        signum: signal.signal(signum, _record_signal)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }

    stdout_path = args.attempt_dir / "child.stdout.log"
    stderr_path = args.attempt_dir / "child.stderr.log"
    samples_path = args.attempt_dir / "resource_samples.jsonl"
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout_stream, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_stream:
            try:
                proc = subprocess.Popen(
                    args.command,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=child_environment,
                    start_new_session=True,
                    text=True,
                )
            except Exception as error:
                _atomic_json(
                    args.attempt_dir / "exit_status.json",
                    {"reason_code": "spawn_failure", "error": repr(error), "wait_reaped": True},
                )
                return EXIT_SPAWN_FAILURE

            pgid = proc.pid
            _atomic_json(
                args.attempt_dir / "run_manifest.json",
                {
                    "schema_version": 1,
                    "command": args.command,
                    "child_pid": proc.pid,
                    "expected_child_pgid": pgid,
                    "start_new_session": True,
                    "start_time_epoch_seconds": start_wall,
                    "timeout_seconds": args.timeout_seconds,
                    "sample_interval_seconds": args.sample_interval_seconds,
                    "thresholds": thresholds,
                    "threshold_activation": "disabled_when_null",
                    "reason_codes": [
                        "completed",
                        "child_exit_nonzero",
                        "timeout",
                        "monitor_failure",
                        "resource_max_child_rss",
                        "resource_min_mem_available",
                        "resource_max_load1_per_cpu",
                        "external_signal",
                        "spawn_failure",
                    ],
                    "child_environment_overrides": {
                        key: child_environment[key]
                        for key in (
                            "PYTHONDONTWRITEBYTECODE",
                            "PYTHONPYCACHEPREFIX",
                            "TMPDIR",
                            "TMP",
                            "TEMP",
                            "XDG_CACHE_HOME",
                        )
                    },
                },
            )

            violation_counts = {
                "resource_max_child_rss": 0,
                "resource_min_mem_available": 0,
                "resource_max_load1_per_cpu": 0,
            }
            monitor_failures = 0
            sample_index = 0
            reason_code: str | None = None
            with samples_path.open("x", encoding="utf-8") as samples_stream:
                while True:
                    returncode = proc.poll()
                    if returncode is not None:
                        reason_code = "completed" if returncode == 0 else "child_exit_nonzero"
                        break
                    if stop_signal:
                        reason_code = "external_signal"
                        break
                    if time.monotonic() - start_monotonic >= args.timeout_seconds:
                        reason_code = "timeout"
                        break

                    try:
                        if (
                            args.simulate_monitor_failure_after is not None
                            and sample_index >= args.simulate_monitor_failure_after
                        ):
                            raise RuntimeError("simulated monitor failure")
                        sample = _sample(proc.pid)
                        monitor_failures = 0
                        sample["sample_index"] = sample_index
                        sample["monitor_ok"] = True
                        checks = {
                            "resource_max_child_rss": args.max_child_rss_bytes is not None
                            and sample["child_rss_bytes"] > args.max_child_rss_bytes,
                            "resource_min_mem_available": args.min_mem_available_bytes is not None
                            and sample["mem_available_bytes"] < args.min_mem_available_bytes,
                            "resource_max_load1_per_cpu": args.max_load1_per_cpu is not None
                            and sample["load1_per_cpu"] > args.max_load1_per_cpu,
                        }
                        for check_reason, violated in checks.items():
                            violation_counts[check_reason] = (
                                violation_counts[check_reason] + 1 if violated else 0
                            )
                        sample["checks"] = checks
                        sample["consecutive_violation_counts"] = dict(violation_counts)
                        samples_stream.write(json.dumps(sample, sort_keys=True) + "\n")
                        samples_stream.flush()
                        for check_reason, count in violation_counts.items():
                            if count >= args.resource_consecutive_samples:
                                reason_code = check_reason
                                break
                    except Exception as error:
                        returncode = proc.poll()
                        if returncode is not None:
                            reason_code = "completed" if returncode == 0 else "child_exit_nonzero"
                            break
                        monitor_failures += 1
                        samples_stream.write(
                            json.dumps(
                                {
                                    "sample_index": sample_index,
                                    "wall_time_epoch_seconds": time.time(),
                                    "monitor_ok": False,
                                    "monitor_error": repr(error),
                                    "consecutive_monitor_failures": monitor_failures,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        samples_stream.flush()
                        if monitor_failures >= args.monitor_failure_consecutive_samples:
                            reason_code = "monitor_failure"
                    if reason_code is not None:
                        break
                    sample_index += 1
                    time.sleep(args.sample_interval_seconds)

            termination = {
                "ownership_verified": False,
                "term_sent": False,
                "kill_sent": False,
                "wait_reaped": proc.poll() is not None,
            }
            if proc.poll() is None:
                termination = _terminate_own_group(
                    proc, pgid, args.term_grace_seconds, args.kill_grace_seconds
                )
            else:
                proc.wait()
                termination["wait_reaped"] = True

            child_returncode = proc.returncode
            end_wall = time.time()
            status = {
                "schema_version": 1,
                "reason_code": reason_code,
                "child_pid": proc.pid,
                "expected_child_pgid": pgid,
                "child_returncode": child_returncode,
                "external_signal": stop_signal[0] if stop_signal else None,
                "start_time_epoch_seconds": start_wall,
                "end_time_epoch_seconds": end_wall,
                "elapsed_seconds": time.monotonic() - start_monotonic,
                **termination,
            }
            _atomic_json(args.attempt_dir / "exit_status.json", status)

            if reason_code == "completed":
                return 0
            if reason_code == "child_exit_nonzero":
                return child_returncode if 1 <= child_returncode <= 123 else 1
            if reason_code == "timeout":
                return EXIT_TIMEOUT
            return EXIT_GUARD_STOP
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
