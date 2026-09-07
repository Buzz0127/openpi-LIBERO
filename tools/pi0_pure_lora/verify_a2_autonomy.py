#!/usr/bin/env python3
"""Fail-closed verification for one completed A2 autonomy probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(args: argparse.Namespace) -> dict[str, object]:
    attempt = args.attempt_dir.resolve()
    plan_path = args.stage_plan.resolve()
    _require(attempt.is_dir(), "attempt directory is missing")
    _require(_sha256(plan_path) == args.expected_stage_plan_sha256, "stage plan SHA-256 mismatch")

    plan = _load(plan_path)
    status = _load(attempt / "status.json")
    summary = _load(attempt / "summary.json")
    run_manifest = _load(attempt / "run_manifest.json")
    progress = _load(attempt / "progress.json")
    probe = _load(attempt / "probe-result.json")
    launcher = _load(attempt / "launcher_receipt.json")

    attempt_id = attempt.name
    for name, value in (("status", status), ("summary", summary), ("run manifest", run_manifest)):
        _require(value.get("attempt_id") == attempt_id, f"{name} attempt identity mismatch")
    _require(status.get("status") == "pass" and status.get("exit_reason") == "completed", "status is not a completed pass")
    _require(summary.get("status") == "pass" and summary.get("reason_code") == "completed", "summary is not a completed pass")
    _require((attempt / "exit_code.txt").read_text(encoding="utf-8").strip() == "0", "exit code is not zero")
    _require(summary.get("retry_count") == 0, "autonomy probe retried unexpectedly")
    _require(summary.get("next_stage_started") is False, "a next stage was started")
    _require(run_manifest.get("next_stage_auto_start") is False, "run manifest permits next-stage auto-start")
    _require(probe.get("next_stage_started") is False, "probe reports a next stage")
    _require(probe.get("offline_environment_verified") is True, "offline environment was not verified")
    _require(probe.get("proxies_unset") is True, "proxy environment was not removed")

    expected_step = plan.get("expected_final_committed_step")
    _require(expected_step == args.expected_final_committed_step, "CLI/plan committed-step mismatch")
    for name, value in (("status", status), ("progress", progress)):
        _require(value.get("current_step") == expected_step, f"{name} current step mismatch")
        _require(value.get("last_committed_step") == expected_step, f"{name} committed step mismatch")
    _require(summary.get("last_progress", {}).get("last_committed_step") == expected_step, "summary committed step mismatch")
    _require(probe.get("steps") == expected_step, "probe step count mismatch")

    child_runs = summary.get("child_runs")
    _require(isinstance(child_runs, list) and len(child_runs) == 1, "expected exactly one child run")
    child = child_runs[0]
    _require(isinstance(child, dict), "invalid child run record")
    _require(child.get("child_pid") == child.get("child_pgid"), "child was not its own process-group leader")
    _require(child.get("returncode") == 0 and child.get("wait_reaped") is True, "child did not exit cleanly and get reaped")
    _require(child.get("term_sent") is False and child.get("kill_sent") is False, "successful child received a stop signal")
    _require(probe.get("pid") == probe.get("pgid") == child.get("child_pid"), "probe/summary process identity mismatch")

    _require(launcher.get("attempt_dir") == str(attempt), "launcher attempt binding mismatch")
    sequences = launcher.get("heartbeat_sequences")
    _require(isinstance(sequences, list) and len(sequences) >= 2, "launcher did not observe enough heartbeats")
    _require(all(isinstance(item, int) for item in sequences), "launcher heartbeat sequence is invalid")
    _require(all(right > left for left, right in zip(sequences, sequences[1:])), "launcher heartbeats did not advance")
    _require(launcher.get("tmux_session_alive") is True, "tmux was not alive when launcher detached")
    _require(launcher.get("launcher_detaches_after_verification") is True, "launcher did not declare detachment")
    session = subprocess.run([args.tmux, "has-session", "-t", args.session_name], capture_output=True, check=False)
    _require(session.returncode != 0, "tmux session still exists after terminal completion")

    manifest_lines = (attempt / "output-files.sha256").read_text(encoding="utf-8").splitlines()
    _require(bool(manifest_lines), "output hash manifest is empty")
    verified_outputs: list[str] = []
    for line in manifest_lines:
        claimed, separator, relative = line.partition("  ")
        _require(bool(separator) and len(claimed) == 64, f"invalid output manifest line: {line}")
        candidate = attempt / relative
        _require(candidate.is_file() and candidate.resolve().is_relative_to(attempt), f"invalid output path: {relative}")
        _require(_sha256(candidate) == claimed, f"output SHA-256 mismatch: {relative}")
        verified_outputs.append(relative)
    expected_required = {str(Path(item).resolve()) for item in plan.get("required_outputs", [])}
    _require(expected_required == {str((attempt / "probe-result.json").resolve())}, "unexpected required output set")
    _require((attempt / "probe-result.json").name in verified_outputs, "required output is absent from hash manifest")

    max_log_bytes = int(run_manifest.get("bounds", {}).get("max_log_bytes", -1))
    log_backups = int(run_manifest.get("bounds", {}).get("log_backups", -1))
    _require(max_log_bytes == args.max_log_bytes and log_backups == args.log_backups, "log bounds mismatch")
    logs = sorted(path for path in attempt.iterdir() if ".log" in path.name)
    _require(bool(logs), "no captured logs found")
    _require(all(path.stat().st_size <= max_log_bytes for path in logs), "a captured log exceeds its bound")
    _require(len([path for path in logs if path.name.startswith("child-00.stdout.log")]) <= log_backups + 1, "too many stdout log backups")
    _require(not list(attempt.rglob("*.tmp")) and not list(attempt.rglob("*.partial")), "temporary or partial files remain")

    plan_identity = plan.get("plan_identity_sha256")
    for name, value in (("status", status), ("summary", summary), ("run manifest", run_manifest)):
        key = "stage_plan_identity_sha256" if name != "summary" else None
        if key is not None:
            _require(value.get(key) == plan_identity, f"{name} plan identity mismatch")
    _require(status.get("stage_plan_sha256") == args.expected_stage_plan_sha256, "status plan file hash mismatch")
    _require(run_manifest.get("stage_plan_sha256") == args.expected_stage_plan_sha256, "run manifest plan file hash mismatch")

    return {
        "schema_version": 1,
        "status": "pass",
        "attempt_id": attempt_id,
        "stage_plan_sha256": args.expected_stage_plan_sha256,
        "stage_plan_identity_sha256": plan_identity,
        "final_committed_step": expected_step,
        "retry_count": 0,
        "child_pid_equals_pgid": True,
        "child_wait_reaped": True,
        "offline_environment_verified": True,
        "proxies_unset": True,
        "launcher_observed_advancing_heartbeats": True,
        "tmux_session_absent_after_completion": True,
        "next_stage_started": False,
        "verified_output_files": verified_outputs,
        "max_log_bytes": max_log_bytes,
        "log_backups": log_backups,
        "temporary_or_partial_files": 0,
        "verified_time_epoch_seconds": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--stage-plan", type=Path, required=True)
    parser.add_argument("--expected-stage-plan-sha256", required=True)
    parser.add_argument("--expected-final-committed-step", type=int, required=True)
    parser.add_argument("--tmux", default="/usr/bin/tmux")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--max-log-bytes", type=int, required=True)
    parser.add_argument("--log-backups", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if args.output.exists():
        temporary.unlink()
        raise FileExistsError(args.output)
    os.replace(temporary, args.output)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
