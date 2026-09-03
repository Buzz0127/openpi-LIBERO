#!/usr/bin/env python3
"""Run one command in an owned process group under storage and host-resource limits."""

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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_mem_available_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def _read_child_rss_bytes(pid: int) -> int:
    with Path(f"/proc/{pid}/status").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"VmRSS not found for child {pid}")


def _usage_bytes(root: Path, seen: set[tuple[int, int]]) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    allocated_total = 0
    apparent_total = 0
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            stat = path.lstat()
        except FileNotFoundError:
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        allocated_total += stat.st_blocks * 512
        apparent_total += stat.st_size
        if path.is_dir() and not path.is_symlink():
            try:
                with os.scandir(path) as entries:
                    stack.extend(Path(entry.path) for entry in entries)
            except FileNotFoundError:
                continue
    return allocated_total, apparent_total


def _validate_non_overlapping_roots(roots: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in roots]
    if len(set(resolved)) != len(resolved):
        raise ValueError("monitored roots must be unique")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError(f"monitored roots overlap: {left} and {right}")


def _storage_sample(
    roots: list[Path], baselines: dict[str, dict[str, int]]
) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    positive_delta_total = 0
    for root in roots:
        current_allocated, current_apparent = _usage_bytes(root, seen)
        baseline = baselines[str(root)]
        positive_allocated_delta = max(0, current_allocated - baseline["allocated_bytes"])
        positive_apparent_delta = max(0, current_apparent - baseline["apparent_bytes"])
        billed_delta = max(positive_allocated_delta, positive_apparent_delta)
        positive_delta_total += billed_delta
        records.append(
            {
                "path": str(root),
                "baseline_allocated_bytes": baseline["allocated_bytes"],
                "baseline_apparent_bytes": baseline["apparent_bytes"],
                "current_allocated_bytes": current_allocated,
                "current_apparent_bytes": current_apparent,
                "positive_allocated_delta_bytes": positive_allocated_delta,
                "positive_apparent_delta_bytes": positive_apparent_delta,
                "billed_delta_bytes": billed_delta,
            }
        )
    return records, positive_delta_total


def _verified_group_alive(proc: subprocess.Popen[Any], expected_pgid: int) -> bool:
    if proc.poll() is not None:
        return False
    try:
        return os.getpgid(proc.pid) == expected_pgid == proc.pid
    except ProcessLookupError:
        return False


def _terminate_own_group(
    proc: subprocess.Popen[Any], expected_pgid: int, term_grace: float, kill_grace: float
) -> dict[str, bool]:
    result = {"ownership_verified": False, "term_sent": False, "kill_sent": False, "wait_reaped": False}
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
    proc.wait(timeout=kill_grace)
    result["wait_reaped"] = True
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--monitor-root", action="append", required=True, type=Path)
    parser.add_argument("--existing-billed-bytes", required=True, type=int)
    parser.add_argument("--soft-limit-bytes", required=True, type=int)
    parser.add_argument("--hard-limit-bytes", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--near-soft-margin-bytes", type=int, default=5 * (1 << 30))
    parser.add_argument("--near-sample-interval-seconds", type=float, default=0.25)
    parser.add_argument("--term-grace-seconds", type=float, default=15.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=5.0)
    parser.add_argument("--min-mem-available-bytes", type=int)
    parser.add_argument("--max-child-rss-bytes", type=int)
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
    if not 0 <= args.existing_billed_bytes < args.soft_limit_bytes < args.hard_limit_bytes:
        parser.error("limits must satisfy 0 <= existing < soft < hard")
    if args.timeout_seconds <= 0 or args.sample_interval_seconds <= 0:
        parser.error("timeout and sampling intervals must be positive")
    if args.near_sample_interval_seconds <= 0 or args.near_soft_margin_bytes < 0:
        parser.error("near-soft sampling values are invalid")
    if args.resource_consecutive_samples < 1 or args.monitor_failure_consecutive_samples < 1:
        parser.error("consecutive-sample counts must be positive")
    _validate_non_overlapping_roots(args.monitor_root)
    return args


def main() -> int:
    args = _parse_args()
    args.attempt_dir.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "cache", "pycache"):
        (args.attempt_dir / name).mkdir()

    roots = [path.resolve(strict=False) for path in args.monitor_root]
    baselines: dict[str, dict[str, int]] = {}
    seen: set[tuple[int, int]] = set()
    for root in roots:
        allocated, apparent = _usage_bytes(root, seen)
        baselines[str(root)] = {"allocated_bytes": allocated, "apparent_bytes": apparent}

    child_env = os.environ.copy()
    child_env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(args.attempt_dir / "pycache"),
            "TMPDIR": str(args.attempt_dir / "tmp"),
            "TMP": str(args.attempt_dir / "tmp"),
            "TEMP": str(args.attempt_dir / "tmp"),
            "XDG_CACHE_HOME": str(args.attempt_dir / "cache"),
        }
    )
    start_wall = time.time()
    start_monotonic = time.monotonic()
    received_signal: list[int] = []

    def _record_signal(signum: int, _frame: Any) -> None:
        if not received_signal:
            received_signal.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _record_signal)

    stdout_path = args.attempt_dir / "child.stdout.log"
    stderr_path = args.attempt_dir / "child.stderr.log"
    samples_path = args.attempt_dir / "samples.jsonl"
    with stdout_path.open("x", encoding="utf-8") as stdout_stream, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr_stream:
        try:
            proc = subprocess.Popen(
                args.command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                env=child_env,
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
                "storage": {
                    "roots": [str(path) for path in roots],
                    "baselines": baselines,
                    "existing_billed_bytes": args.existing_billed_bytes,
                    "soft_limit_bytes": args.soft_limit_bytes,
                    "hard_limit_bytes": args.hard_limit_bytes,
                    "billing_per_root": "max(positive allocated delta, positive apparent delta)",
                    "positive_per_root_deltas_prevent_decrease_masking": True,
                    "hardlinks_deduplicated_per_sample": True,
                },
                "resources": {
                    "min_mem_available_bytes": args.min_mem_available_bytes,
                    "max_child_rss_bytes": args.max_child_rss_bytes,
                    "max_load1_per_cpu": args.max_load1_per_cpu,
                    "consecutive_samples": args.resource_consecutive_samples,
                },
            },
        )

        violations = {"min_mem": 0, "max_rss": 0, "max_load": 0}
        monitor_failures = 0
        sample_index = 0
        last_sample: dict[str, Any] | None = None
        reason_code: str | None = None
        with samples_path.open("x", encoding="utf-8") as samples_stream:
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    reason_code = "completed" if returncode == 0 else "child_exit_nonzero"
                    break
                if received_signal:
                    reason_code = "external_signal"
                    break
                if time.monotonic() - start_monotonic >= args.timeout_seconds:
                    reason_code = "timeout"
                    break
                try:
                    if args.simulate_monitor_failure_after is not None and sample_index >= args.simulate_monitor_failure_after:
                        raise RuntimeError("simulated monitor failure")
                    root_records, positive_delta = _storage_sample(roots, baselines)
                    billed = args.existing_billed_bytes + positive_delta
                    load1 = os.getloadavg()[0]
                    cpu_count = os.cpu_count() or 1
                    sample = {
                        "sample_index": sample_index,
                        "wall_time_epoch_seconds": time.time(),
                        "monotonic_seconds": time.monotonic(),
                        "roots": root_records,
                        "positive_stage_delta_bytes": positive_delta,
                        "billed_lora_bytes": billed,
                        "soft_headroom_bytes": args.soft_limit_bytes - billed,
                        "hard_headroom_bytes": args.hard_limit_bytes - billed,
                        "child_rss_bytes": _read_child_rss_bytes(proc.pid),
                        "mem_available_bytes": _read_mem_available_bytes(),
                        "load1": load1,
                        "logical_cpu_count": cpu_count,
                        "load1_per_cpu": load1 / cpu_count,
                        "monitor_ok": True,
                    }
                    monitor_failures = 0
                    last_sample = sample
                    if billed >= args.hard_limit_bytes:
                        reason_code = "storage_hard_limit"
                    elif billed >= args.soft_limit_bytes:
                        reason_code = "storage_soft_limit"

                    checks = {
                        "min_mem": args.min_mem_available_bytes is not None
                        and sample["mem_available_bytes"] < args.min_mem_available_bytes,
                        "max_rss": args.max_child_rss_bytes is not None
                        and sample["child_rss_bytes"] > args.max_child_rss_bytes,
                        "max_load": args.max_load1_per_cpu is not None
                        and sample["load1_per_cpu"] > args.max_load1_per_cpu,
                    }
                    for name, violated in checks.items():
                        violations[name] = violations[name] + 1 if violated else 0
                    if violations["min_mem"] >= args.resource_consecutive_samples:
                        reason_code = "resource_min_mem_available"
                    elif violations["max_rss"] >= args.resource_consecutive_samples:
                        reason_code = "resource_max_child_rss"
                    elif violations["max_load"] >= args.resource_consecutive_samples:
                        reason_code = "resource_max_load1_per_cpu"
                    sample["violation_counts"] = violations.copy()
                    samples_stream.write(json.dumps(sample, sort_keys=True) + "\n")
                    samples_stream.flush()
                    os.fsync(samples_stream.fileno())
                except Exception as error:
                    monitor_failures += 1
                    samples_stream.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "wall_time_epoch_seconds": time.time(),
                                "monitor_ok": False,
                                "monitor_failure_count": monitor_failures,
                                "error": repr(error),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    samples_stream.flush()
                    os.fsync(samples_stream.fileno())
                    if monitor_failures >= args.monitor_failure_consecutive_samples:
                        reason_code = "monitor_failure"

                sample_index += 1
                if reason_code is not None:
                    break
                interval = args.sample_interval_seconds
                if last_sample is not None and last_sample["soft_headroom_bytes"] <= args.near_soft_margin_bytes:
                    interval = args.near_sample_interval_seconds
                time.sleep(interval)

        termination = {"ownership_verified": False, "term_sent": False, "kill_sent": False, "wait_reaped": False}
        returncode = proc.poll()
        if returncode is None:
            termination = _terminate_own_group(proc, pgid, args.term_grace_seconds, args.kill_grace_seconds)
            returncode = proc.returncode
        else:
            proc.wait()
            termination["wait_reaped"] = True

        final_billed = last_sample["billed_lora_bytes"] if last_sample else None
        _atomic_json(
            args.attempt_dir / "exit_status.json",
            {
                "reason_code": reason_code,
                "child_pid": proc.pid,
                "expected_child_pgid": pgid,
                "child_returncode": returncode,
                "external_signal": received_signal[0] if received_signal else None,
                "elapsed_seconds": time.monotonic() - start_monotonic,
                "sample_count": sample_index,
                "final_billed_lora_bytes": final_billed,
                "observed_soft_limit_overshoot_bytes": (
                    max(0, final_billed - args.soft_limit_bytes) if final_billed is not None else None
                ),
                "final_sample": last_sample,
                **termination,
            },
        )

    if reason_code == "completed":
        return 0
    if reason_code == "child_exit_nonzero":
        return int(proc.returncode or 1)
    if reason_code == "timeout":
        return EXIT_TIMEOUT
    return EXIT_GUARD_STOP


if __name__ == "__main__":
    sys.exit(main())
