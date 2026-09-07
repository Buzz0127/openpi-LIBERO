#!/usr/bin/env python3
"""Run one authorized stage autonomously with bounded retries and atomic status."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO

import experiment_identity
import write_evidence_manifest


EXIT_TIMEOUT = 124
EXIT_HEARTBEAT_TIMEOUT = 125
EXIT_IDENTITY_FAILURE = 126
REQUIRED_IDENTITIES = {"model", "dataset", "norm", "golden", "config", "split"}
ACTIVE_STATES = {"starting", "running", "retrying", "terminating"}
PROGRESS_KEYS = {
    "current_step",
    "last_committed_step",
    "recent_metrics",
    "pause_count",
    "resume_count",
    "resource_peaks",
}
OFFLINE_OVERRIDES = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_replace_json(path: Path, value: object) -> None:
    _atomic_replace(path, _canonical_json(value))


def _atomic_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_new_json(path: Path, value: object) -> None:
    _atomic_new(path, _canonical_json(value))


def _append_event(path: Path, value: object) -> None:
    with path.open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    return experiment_identity.sha256_file(path)


def _git(root: Path, *args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip() if result.returncode == 0 else "NONE"


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        "root": str(root.resolve()),
        "branch": _git(root, "branch", "--show-current"),
        "head": _git(root, "rev-parse", "HEAD"),
        "upstream": _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", allow_failure=True),
        "status": _git(root, "status", "--porcelain=v1"),
    }


def _parse_identities(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or key in result:
            raise ValueError(f"identity must be a unique KEY=SHA256 pair: {raw}")
        result[key] = experiment_identity.validate_sha256(value, key)
    if set(result) != REQUIRED_IDENTITIES:
        raise ValueError(f"identities must be exactly {sorted(REQUIRED_IDENTITIES)}")
    return result


def _validate_stage_plan(path: Path, args: argparse.Namespace, identities: dict[str, str]) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("execution_authorized") is not False:
        raise ValueError("stage plan must be an explicitly non-authorizing JSON object")
    claimed = plan.get("plan_identity_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_identity_sha256", None)
    if claimed is None or experiment_identity.canonical_sha256(unsigned) != claimed:
        raise ValueError("stage plan internal identity mismatch")
    expected_if_present = {
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "model_identity_sha256": identities["model"],
        "dataset_manifest_sha256": identities["dataset"],
    }
    mismatched = {
        key: {"plan": plan[key], "runtime": expected}
        for key, expected in expected_if_present.items()
        if key in plan and plan[key] != expected
    }
    if mismatched:
        raise ValueError(f"stage plan/runtime mismatch: {mismatched}")
    return plan


def _safe_current_target(path: Path, attempt_id: str) -> None:
    if not path.exists():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("status") in ACTIVE_STATES and current.get("attempt_id") != attempt_id:
        raise RuntimeError("another attempt is recorded active in current.json")


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_progress(path: Path, defaults: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    if not path.exists():
        return dict(defaults), None
    stat_before = path.stat()
    value = json.loads(path.read_text(encoding="utf-8"))
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError("progress file changed while reading")
    if not isinstance(value, dict) or not set(value) <= PROGRESS_KEYS or not _finite_tree(value):
        raise ValueError("progress JSON has unexpected keys or non-finite values")
    merged = dict(defaults)
    merged.update(value)
    return merged, stat_after.st_mtime_ns


def _group_alive(proc: subprocess.Popen[bytes], pgid: int) -> bool:
    if proc.poll() is not None:
        return False
    try:
        return os.getpgid(proc.pid) == proc.pid == pgid
    except ProcessLookupError:
        return False


def _stop_owned_group(
    proc: subprocess.Popen[bytes], pgid: int, term_grace: float, kill_grace: float
) -> dict[str, bool]:
    result = {"ownership_verified": False, "term_sent": False, "kill_sent": False, "wait_reaped": False}
    if _group_alive(proc, pgid):
        result["ownership_verified"] = True
        os.killpg(pgid, signal.SIGTERM)
        result["term_sent"] = True
    try:
        proc.wait(timeout=term_grace)
        result["wait_reaped"] = True
        return result
    except subprocess.TimeoutExpired:
        pass
    if _group_alive(proc, pgid):
        os.killpg(pgid, signal.SIGKILL)
        result["kill_sent"] = True
    proc.wait(timeout=kill_grace)
    result["wait_reaped"] = True
    return result


class _RotatingCapture(threading.Thread):
    def __init__(self, source: BinaryIO, destination: Path, max_bytes: int, backups: int):
        super().__init__(daemon=True)
        self.source = source
        self.destination = destination
        self.max_bytes = max_bytes
        self.backups = backups
        self.error: BaseException | None = None

    def _rotate(self) -> None:
        oldest = self.destination.with_name(f"{self.destination.name}.{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backups - 1, 0, -1):
            source = self.destination.with_name(f"{self.destination.name}.{index}")
            if source.exists():
                os.replace(source, self.destination.with_name(f"{self.destination.name}.{index + 1}"))
        if self.destination.exists():
            os.replace(self.destination, self.destination.with_name(f"{self.destination.name}.1"))

    def run(self) -> None:
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            stream = self.destination.open("xb")
            size = 0
            try:
                while True:
                    chunk = self.source.read(64 << 10)
                    if not chunk:
                        break
                    if size and size + len(chunk) > self.max_bytes:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                        self._rotate()
                        stream = self.destination.open("xb")
                        size = 0
                    while len(chunk) > self.max_bytes:
                        stream.write(chunk[: self.max_bytes])
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                        self._rotate()
                        chunk = chunk[self.max_bytes :]
                        stream = self.destination.open("xb")
                    stream.write(chunk)
                    stream.flush()
                    size += len(chunk)
            finally:
                if not stream.closed:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
        except BaseException as error:
            self.error = error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--current-json", required=True, type=Path)
    parser.add_argument("--stage-plan", required=True, type=Path)
    parser.add_argument("--expected-stage-plan-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-upstream", required=True)
    parser.add_argument("--identity", action="append", default=[])
    parser.add_argument("--stage", required=True)
    parser.add_argument("--segment-start", required=True, type=int)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--expected-final-committed-step", required=True, type=int)
    parser.add_argument("--required-output", action="append", default=[], type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-timeout-seconds", required=True, type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--term-grace-seconds", type=float, default=30.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-return-code", action="append", default=[], type=int)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--max-log-bytes", type=int, default=10_000_000)
    parser.add_argument("--log-backups", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("child command required after --")
    if args.segment_start < 0 or args.segment_end <= args.segment_start:
        parser.error("segment must satisfy 0 <= start < end")
    if args.train_seed == args.eval_seed:
        parser.error("training and evaluation seeds must differ")
    if args.expected_final_committed_step != args.segment_end:
        parser.error("expected final committed step must equal segment end")
    for name in (
        "timeout_seconds", "heartbeat_timeout_seconds", "sample_interval_seconds",
        "term_grace_seconds", "kill_grace_seconds", "retry_delay_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_retries < 0 or args.max_log_bytes < 1024 or args.log_backups < 1:
        parser.error("retry and log bounds are invalid")
    return args


def main() -> int:
    args = _parse_args()
    attempt = args.attempt_dir.resolve()
    current_json = args.current_json.resolve()
    if attempt.exists():
        raise FileExistsError(attempt)
    if current_json.parent != attempt.parent:
        raise ValueError("current.json and attempt must share one stage parent")
    progress = args.progress_json.resolve()
    if progress != attempt and attempt not in progress.parents:
        raise ValueError("progress JSON must stay inside the collision-safe attempt")
    if progress.exists() or any(path.exists() for path in args.required_output):
        raise FileExistsError("progress and required completion outputs must be new")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    _safe_current_target(current_json, attempt.name)
    identities = _parse_identities(args.identity)
    if _sha256(args.stage_plan) != args.expected_stage_plan_sha256:
        raise RuntimeError("stage plan SHA-256 mismatch")
    stage_plan = _validate_stage_plan(args.stage_plan, args, identities)
    source = _source_snapshot(args.source_root)
    expected_source = {
        "branch": args.expected_branch,
        "head": args.expected_head,
        "upstream": args.expected_upstream,
        "status": "",
    }
    if any(source[key] != value for key, value in expected_source.items()):
        raise RuntimeError(f"source identity mismatch: {source}")
    runtime_bounds = {
        "timeout_seconds": args.timeout_seconds,
        "heartbeat_timeout_seconds": args.heartbeat_timeout_seconds,
        "sample_interval_seconds": args.sample_interval_seconds,
        "term_grace_seconds": args.term_grace_seconds,
        "kill_grace_seconds": args.kill_grace_seconds,
        "max_retries": args.max_retries,
        "retry_return_codes": sorted(set(args.retry_return_code)),
        "retry_delay_seconds": args.retry_delay_seconds,
        "max_log_bytes": args.max_log_bytes,
        "log_backups": args.log_backups,
    }
    exact_plan_fields = {
        "stage": args.stage,
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "expected_final_committed_step": args.expected_final_committed_step,
        "source": source,
        "identities": identities,
        "attempt_dir": str(attempt),
        "current_json": str(current_json),
        "progress_json": str(progress),
        "required_outputs": [str(path.resolve()) for path in args.required_output],
        "bounds": runtime_bounds,
        "command": args.command,
        "next_stage_auto_start": False,
    }
    mismatched_plan_fields = {
        key: {"plan": stage_plan.get(key), "runtime": value}
        for key, value in exact_plan_fields.items()
        if stage_plan.get(key) != value
    }
    if mismatched_plan_fields:
        raise RuntimeError(f"autonomous plan/runtime mismatch: {mismatched_plan_fields}")
    if stage_plan.get("tools", {}).get("orchestrator_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("orchestrator source identity differs from stage plan")

    attempt.mkdir()
    status_path = attempt / "status.json"
    heartbeat_path = attempt / "heartbeat.json"
    events_path = attempt / "events.jsonl"
    start_epoch = time.time()
    start_mono = time.monotonic()
    stop_signal: list[int] = []
    previous_handlers = {
        signum: signal.signal(signum, lambda received, _frame: stop_signal.append(received) if not stop_signal else None)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    defaults = {
        "current_step": args.segment_start,
        "last_committed_step": args.segment_start,
        "recent_metrics": {},
        "pause_count": 0,
        "resume_count": 0,
        "resource_peaks": {},
    }
    base_status = {
        "schema_version": 1,
        "attempt_id": attempt.name,
        "stage": args.stage,
        "segment": {"start": args.segment_start, "end": args.segment_end},
        "orchestrator_pid": os.getpid(),
        "start_time_epoch_seconds": start_epoch,
        "start_time_utc": _utc_now(),
        "max_retries": args.max_retries,
        "source": source,
        "stage_plan_sha256": args.expected_stage_plan_sha256,
        "stage_plan_identity_sha256": stage_plan["plan_identity_sha256"],
        "identities": identities,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
    }
    manifest = {
        **base_status,
        "command": args.command,
        "current_json": str(current_json),
        "progress_json": str(args.progress_json.resolve()),
        "required_outputs": [str(path.resolve()) for path in args.required_output],
        "bounds": runtime_bounds,
        "next_stage_auto_start": False,
        "child_environment_overrides": OFFLINE_OVERRIDES,
        "child_environment_unset": list(PROXY_VARIABLES),
    }
    _atomic_new_json(attempt / "run_manifest.json", manifest)
    _append_event(events_path, {"event": "orchestrator_started", "time_utc": _utc_now()})

    sequence = 0
    latest_progress = dict(defaults)
    final_reason = "internal_error"
    final_returncode = 1
    retry_index = 0
    child_records: list[dict[str, Any]] = []

    def publish(state: str, *, child_pid: int | None, child_pgid: int | None, reason: str | None) -> None:
        nonlocal sequence
        sequence += 1
        now_epoch = time.time()
        status = {
            **base_status,
            **latest_progress,
            "status": state,
            "child_pid": child_pid,
            "child_pgid": child_pgid,
            "retry_index": retry_index,
            "update_time_epoch_seconds": now_epoch,
            "update_time_utc": _utc_now(),
            "heartbeat_sequence": sequence,
            "exit_reason": reason,
        }
        heartbeat = {
            "schema_version": 1,
            "attempt_id": attempt.name,
            "stage": args.stage,
            "status": state,
            "sequence": sequence,
            "update_time_epoch_seconds": now_epoch,
        }
        _atomic_replace_json(status_path, status)
        _atomic_replace_json(heartbeat_path, heartbeat)
        _atomic_replace_json(current_json, status)

    publish("starting", child_pid=None, child_pgid=None, reason=None)
    try:
        while True:
            stdout_path = attempt / f"child-{retry_index:02d}.stdout.log"
            stderr_path = attempt / f"child-{retry_index:02d}.stderr.log"
            proc = subprocess.Popen(
                args.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **{key: value for key, value in os.environ.items() if key not in PROXY_VARIABLES},
                    **OFFLINE_OVERRIDES,
                },
                start_new_session=True,
            )
            assert proc.stdout is not None and proc.stderr is not None
            pgid = proc.pid
            stdout_capture = _RotatingCapture(proc.stdout, stdout_path, args.max_log_bytes, args.log_backups)
            stderr_capture = _RotatingCapture(proc.stderr, stderr_path, args.max_log_bytes, args.log_backups)
            stdout_capture.start()
            stderr_capture.start()
            _append_event(events_path, {
                "event": "child_started", "retry_index": retry_index, "pid": proc.pid,
                "pgid": pgid, "time_utc": _utc_now(),
            })
            publish("running", child_pid=proc.pid, child_pgid=pgid, reason=None)
            child_start = time.monotonic()
            last_progress = child_start
            last_progress_mtime: int | None = None
            child_reason: str | None = None
            termination = {"ownership_verified": False, "term_sent": False, "kill_sent": False, "wait_reaped": False}
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    child_reason = "completed" if returncode == 0 else "child_exit_nonzero"
                    break
                if stop_signal:
                    child_reason = "external_signal"
                    break
                now = time.monotonic()
                if now - start_mono >= args.timeout_seconds:
                    child_reason = "timeout"
                    break
                try:
                    progress, progress_mtime = _read_progress(args.progress_json, defaults)
                    if progress_mtime is not None and progress_mtime != last_progress_mtime:
                        last_progress = now
                        last_progress_mtime = progress_mtime
                    latest_progress = progress
                except Exception as error:
                    child_reason = "invalid_progress"
                    _append_event(events_path, {"event": child_reason, "error": repr(error), "time_utc": _utc_now()})
                    break
                if now - last_progress >= args.heartbeat_timeout_seconds:
                    child_reason = "heartbeat_timeout"
                    break
                publish("running", child_pid=proc.pid, child_pgid=pgid, reason=None)
                time.sleep(args.sample_interval_seconds)

            if proc.poll() is None:
                publish("terminating", child_pid=proc.pid, child_pgid=pgid, reason=child_reason)
                termination = _stop_owned_group(proc, pgid, args.term_grace_seconds, args.kill_grace_seconds)
            else:
                proc.wait()
                termination["wait_reaped"] = True
            stdout_capture.join(timeout=args.kill_grace_seconds)
            stderr_capture.join(timeout=args.kill_grace_seconds)
            if stdout_capture.is_alive() or stderr_capture.is_alive() or stdout_capture.error or stderr_capture.error:
                child_reason = "log_capture_failure"
            child_record = {
                "retry_index": retry_index,
                "child_pid": proc.pid,
                "child_pgid": pgid,
                "returncode": proc.returncode,
                "reason": child_reason,
                "elapsed_seconds": time.monotonic() - child_start,
                **termination,
            }
            child_records.append(child_record)
            _append_event(events_path, {"event": "child_finished", **child_record, "time_utc": _utc_now()})

            retryable = (
                child_reason == "child_exit_nonzero"
                and proc.returncode in set(args.retry_return_code)
                and retry_index < args.max_retries
            )
            if retryable:
                retry_index += 1
                publish("retrying", child_pid=None, child_pgid=None, reason="retryable_child_exit")
                _append_event(events_path, {
                    "event": "retry_scheduled", "retry_index": retry_index,
                    "delay_seconds": args.retry_delay_seconds, "time_utc": _utc_now(),
                })
                time.sleep(args.retry_delay_seconds)
                continue

            if child_reason == "completed":
                latest_progress, _ = _read_progress(args.progress_json, defaults)
                missing = [str(path) for path in args.required_output if not path.is_file()]
                if latest_progress.get("last_committed_step") != args.expected_final_committed_step:
                    child_reason = "committed_step_mismatch"
                elif missing:
                    child_reason = "required_output_missing"
                else:
                    final_reason = "completed"
                    final_returncode = 0
                    break
            final_reason = child_reason or "unknown_failure"
            final_returncode = (
                EXIT_TIMEOUT if final_reason == "timeout"
                else EXIT_HEARTBEAT_TIMEOUT if final_reason == "heartbeat_timeout"
                else proc.returncode if proc.returncode and 1 <= proc.returncode <= 123
                else 1
            )
            break
    except BaseException as error:
        final_reason = "orchestrator_exception"
        final_returncode = 1
        _append_event(events_path, {"event": final_reason, "error": repr(error), "time_utc": _utc_now()})
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    status_name = "pass" if final_returncode == 0 else "fail"
    output_hashes = {
        str(path.resolve()): _sha256(path)
        for path in args.required_output
        if path.is_file()
    }
    publish(status_name, child_pid=None, child_pgid=None, reason=final_reason)
    summary = {
        "schema_version": 1,
        "status": status_name,
        "reason_code": final_reason,
        "attempt_id": attempt.name,
        "stage": args.stage,
        "segment": {"start": args.segment_start, "end": args.segment_end},
        "elapsed_seconds": time.monotonic() - start_mono,
        "retry_count": retry_index,
        "last_progress": latest_progress,
        "child_runs": child_records,
        "output_sha256": output_hashes,
        "next_stage_started": False,
    }
    summary["summary_identity_sha256"] = experiment_identity.canonical_sha256(summary)
    _atomic_new_json(attempt / "summary.json", summary)
    _atomic_new(attempt / "exit_code.txt", f"{final_returncode}\n".encode("ascii"))
    manifest_payload = write_evidence_manifest.build_manifest(attempt, attempt / "output-files.sha256")
    _atomic_new(attempt / "output-files.sha256", manifest_payload.encode("utf-8"))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return final_returncode


if __name__ == "__main__":
    sys.exit(main())
