#!/usr/bin/env python3
"""Run one task under a fail-safe NVIDIA GPU utilization pause guard.

The guard creates a new process group for the command it launches and signals
only that group. It never discovers or signals unrelated processes. When total
utilization on the selected physical GPU reaches the pause threshold, the guard
sends SIGSTOP to its child group. It resumes with SIGCONT only after utilization
has stayed below the lower resume threshold for a configured number of samples.
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any, List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class GpuStatus:
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float

    @property
    def free_memory_percent(self) -> float:
        if self.memory_total_mib <= 0:
            raise RuntimeError("GPU reported non-positive total memory")
        return 100.0 * (self.memory_total_mib - self.memory_used_mib) / self.memory_total_mib


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a command and pause only its process group at high GPU utilization."
    )
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--pause-at", type=float, default=95.0)
    parser.add_argument("--resume-at", type=float, default=85.0)
    parser.add_argument("--min-free-memory-percent", type=float, default=15.0)
    parser.add_argument("--resume-free-memory-percent", type=float, default=20.0)
    parser.add_argument("--terminate-free-memory-percent", type=float, default=10.0)
    parser.add_argument("--resume-samples", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--monitor-error-limit", type=int, default=3)
    parser.add_argument("--max-prelaunch-wait-seconds", type=float, default=300.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=15.0)
    parser.add_argument("--log", type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if not 0.0 < args.resume_at < args.pause_at <= 100.0:
        parser.error("thresholds must satisfy 0 < resume-at < pause-at <= 100")
    if not (
        0.0
        < args.terminate_free_memory_percent
        < args.min_free_memory_percent
        < args.resume_free_memory_percent
        <= 100.0
    ):
        parser.error(
            "memory thresholds must satisfy 0 < terminate < min-free < resume-free <= 100"
        )
    if args.resume_samples <= 0:
        parser.error("--resume-samples must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.monitor_error_limit <= 0:
        parser.error("--monitor-error-limit must be positive")
    if args.max_prelaunch_wait_seconds <= 0:
        parser.error("--max-prelaunch-wait-seconds must be positive")
    if args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.terminate_grace_seconds <= 0:
        parser.error("--terminate-grace-seconds must be positive")
    return args


def query_gpu_status(physical_gpu: int) -> GpuStatus:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=10)
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and int(fields[0]) == physical_gpu:
            return GpuStatus(
                utilization_percent=float(fields[1]),
                memory_used_mib=float(fields[2]),
                memory_total_mib=float(fields[3]),
            )
    raise RuntimeError("physical GPU {} not found".format(physical_gpu))


class EventLogger:
    def __init__(self, path: Optional[pathlib.Path]) -> None:
        self.path = path
        if self.path is not None:
            self.path = self.path.expanduser().resolve()
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"timestamp_utc": utc_now(), "event": event}
        record.update(fields)
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def signal_child_group(child: subprocess.Popen[Any], signum: int) -> bool:
    if child.poll() is not None:
        return False
    os.killpg(child.pid, signum)
    return True


def terminate_child(
    child: subprocess.Popen[Any],
    paused: bool,
    grace_seconds: float,
    logger: EventLogger,
) -> None:
    if child.poll() is not None:
        return
    if paused:
        signal_child_group(child, signal.SIGCONT)
        logger.emit("resumed_for_termination", child_pid=child.pid)
    signal_child_group(child, signal.SIGTERM)
    logger.emit("termination_requested", child_pid=child.pid)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signal_child_group(child, signal.SIGKILL)
        logger.emit("termination_forced", child_pid=child.pid)
        child.wait()


def wait_until_launch_safe(args: argparse.Namespace, logger: EventLogger) -> None:
    errors = 0
    deadline = time.monotonic() + args.max_prelaunch_wait_seconds
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU did not fall below the pause threshold before launch")
        try:
            status = query_gpu_status(args.physical_gpu)
            errors = 0
            launch_allowed = (
                status.utilization_percent < args.pause_at
                and status.free_memory_percent > args.min_free_memory_percent
            )
            logger.emit(
                "prelaunch_sample",
                physical_gpu=args.physical_gpu,
                utilization_percent=status.utilization_percent,
                free_memory_percent=status.free_memory_percent,
                memory_used_mib=status.memory_used_mib,
                memory_total_mib=status.memory_total_mib,
                launch_allowed=launch_allowed,
            )
            if launch_allowed:
                return
        except Exception as exc:
            errors += 1
            logger.emit(
                "monitor_error",
                phase="prelaunch",
                consecutive_errors=errors,
                error="{}: {}".format(type(exc).__name__, exc),
            )
            if errors >= args.monitor_error_limit:
                raise RuntimeError("GPU monitoring unavailable before launch") from exc
        time.sleep(args.interval_seconds)


def run_guarded(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log)
    logger.emit(
        "guard_started",
        physical_gpu=args.physical_gpu,
        pause_at=args.pause_at,
        resume_at=args.resume_at,
        min_free_memory_percent=args.min_free_memory_percent,
        resume_free_memory_percent=args.resume_free_memory_percent,
        terminate_free_memory_percent=args.terminate_free_memory_percent,
        resume_samples=args.resume_samples,
        max_prelaunch_wait_seconds=args.max_prelaunch_wait_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        command=args.command,
    )
    wait_until_launch_safe(args, logger)

    child = subprocess.Popen(args.command, start_new_session=True)
    logger.emit("child_started", child_pid=child.pid)
    started = time.monotonic()
    paused = False
    safe_samples = 0
    monitor_errors = 0
    shutdown_signal = []  # type: List[int]

    def request_shutdown(signum: int, _frame: Any) -> None:
        shutdown_signal.append(signum)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_shutdown)

    try:
        while child.poll() is None:
            if shutdown_signal:
                logger.emit("guard_shutdown_requested", signal=shutdown_signal[-1])
                terminate_child(child, paused, args.terminate_grace_seconds, logger)
                return 128 + shutdown_signal[-1]
            if time.monotonic() - started >= args.max_runtime_seconds:
                logger.emit("runtime_limit_reached")
                terminate_child(child, paused, args.terminate_grace_seconds, logger)
                return 124

            try:
                status = query_gpu_status(args.physical_gpu)
                monitor_errors = 0
                action = "none"
                if status.free_memory_percent <= args.terminate_free_memory_percent:
                    logger.emit(
                        "memory_emergency",
                        child_pid=child.pid,
                        free_memory_percent=status.free_memory_percent,
                    )
                    terminate_child(child, paused, args.terminate_grace_seconds, logger)
                    return 125
                pressure = (
                    status.utilization_percent >= args.pause_at
                    or status.free_memory_percent <= args.min_free_memory_percent
                )
                if not paused and pressure:
                    if signal_child_group(child, signal.SIGSTOP):
                        paused = True
                        safe_samples = 0
                        action = "paused"
                elif paused:
                    safe = (
                        status.utilization_percent <= args.resume_at
                        and status.free_memory_percent >= args.resume_free_memory_percent
                    )
                    safe_samples = safe_samples + 1 if safe else 0
                    if safe_samples >= args.resume_samples:
                        if signal_child_group(child, signal.SIGCONT):
                            paused = False
                            safe_samples = 0
                            action = "resumed"
                logger.emit(
                    "gpu_sample",
                    physical_gpu=args.physical_gpu,
                    utilization_percent=status.utilization_percent,
                    free_memory_percent=status.free_memory_percent,
                    memory_used_mib=status.memory_used_mib,
                    memory_total_mib=status.memory_total_mib,
                    child_pid=child.pid,
                    paused=paused,
                    safe_samples=safe_samples,
                    action=action,
                )
            except Exception as exc:
                monitor_errors += 1
                logger.emit(
                    "monitor_error",
                    phase="runtime",
                    consecutive_errors=monitor_errors,
                    error="{}: {}".format(type(exc).__name__, exc),
                )
                if monitor_errors >= args.monitor_error_limit and not paused:
                    if signal_child_group(child, signal.SIGSTOP):
                        paused = True
                        safe_samples = 0
                        logger.emit("paused_fail_safe", child_pid=child.pid)
            time.sleep(args.interval_seconds)
    except BaseException:
        terminate_child(child, paused, args.terminate_grace_seconds, logger)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return_code = child.wait()
    logger.emit("child_exited", child_pid=child.pid, return_code=return_code)
    return return_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_guarded(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
