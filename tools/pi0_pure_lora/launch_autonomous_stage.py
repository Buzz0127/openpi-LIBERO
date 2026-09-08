#!/usr/bin/env python3
"""Launch an autonomous orchestrator in tmux and verify initial heartbeat progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def _atomic_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _tmux_has(tmux: Path, session: str) -> bool:
    return subprocess.run(
        [str(tmux), "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmux", required=True, type=Path)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--orchestrator", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--required-heartbeat-observations", type=int, default=2)
    parser.add_argument("orchestrator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.orchestrator_args and args.orchestrator_args[0] == "--":
        args.orchestrator_args = args.orchestrator_args[1:]
    if not args.orchestrator_args:
        parser.error("orchestrator arguments required after --")
    if not SESSION_RE.fullmatch(args.session_name):
        parser.error("unsafe tmux session name")
    if args.startup_timeout_seconds <= 0 or args.required_heartbeat_observations < 2:
        parser.error("startup bounds are invalid")
    return args


def _bound_attempt(orchestrator_args: list[str]) -> Path:
    # The orchestrator's child command follows the first ``--`` separator and
    # legitimately has its own nested --attempt-dir flags (storage guard and
    # runner).  Only the outer orchestrator option binds this launcher.
    try:
        boundary = orchestrator_args.index("--")
    except ValueError:
        boundary = len(orchestrator_args)
    return Path(_argument_value(orchestrator_args[:boundary], "--attempt-dir")).resolve()


def _argument_value(arguments: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"orchestrator command must contain exactly one {name}")
    return arguments[positions[0] + 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    for path in (args.tmux, args.python, args.orchestrator):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _bound_attempt(args.orchestrator_args) != args.attempt_dir.resolve():
        raise ValueError("launcher and orchestrator attempt paths differ")
    plan_path = Path(_argument_value(args.orchestrator_args, "--stage-plan"))
    expected_plan_sha256 = _argument_value(args.orchestrator_args, "--expected-stage-plan-sha256")
    if _sha256(plan_path) != expected_plan_sha256:
        raise RuntimeError("launcher observed a stage plan SHA-256 mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("tools", {}).get("launcher_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("launcher source identity differs from stage plan")
    if plan.get("tools", {}).get("orchestrator_sha256") != _sha256(args.orchestrator):
        raise RuntimeError("orchestrator source identity differs from stage plan")
    if plan.get("stage") == "T1-engineering-100-200":
        import t1_execution_contract
        t1_execution_contract.validate_launch_bindings(plan)
    if args.attempt_dir.exists():
        raise FileExistsError(args.attempt_dir)
    if _tmux_has(args.tmux, args.session_name):
        raise RuntimeError("tmux session already exists")
    command = [str(args.python), str(args.orchestrator), *args.orchestrator_args]
    launch = subprocess.run(
        [str(args.tmux), "new-session", "-d", "-s", args.session_name, shlex.join(command)],
        text=True,
        capture_output=True,
        check=False,
    )
    if launch.returncode != 0:
        raise RuntimeError(f"tmux launch failed: {launch.stderr.strip()}")

    heartbeat = args.attempt_dir / "heartbeat.json"
    deadline = time.monotonic() + args.startup_timeout_seconds
    observed: list[int] = []
    try:
        while time.monotonic() < deadline:
            if heartbeat.is_file():
                value = json.loads(heartbeat.read_text(encoding="utf-8"))
                if value.get("attempt_id") != args.attempt_dir.name or value.get("status") not in {
                    "starting", "running", "retrying", "terminating"
                }:
                    raise RuntimeError("heartbeat identity or state mismatch")
                sequence = int(value["sequence"])
                if not observed or sequence > observed[-1]:
                    observed.append(sequence)
                if len(observed) >= args.required_heartbeat_observations:
                    if not _tmux_has(args.tmux, args.session_name):
                        raise RuntimeError("tmux session exited during startup verification")
                    receipt = {
                        "schema_version": 1,
                        "session_name": args.session_name,
                        "attempt_dir": str(args.attempt_dir.resolve()),
                        "orchestrator_command": command,
                        "heartbeat_sequences": observed,
                        "tmux_session_alive": True,
                        "launcher_detaches_after_verification": True,
                        "launcher_pid": os.getpid(),
                        "verified_time_epoch_seconds": time.time(),
                    }
                    _atomic_new_json(args.attempt_dir / "launcher_receipt.json", receipt)
                    print(json.dumps(receipt, sort_keys=True))
                    return 0
            if not _tmux_has(args.tmux, args.session_name):
                raise RuntimeError("tmux session exited before heartbeat verification")
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for advancing autonomous heartbeat")
    except BaseException:
        if _tmux_has(args.tmux, args.session_name):
            subprocess.run([str(args.tmux), "kill-session", "-t", args.session_name], check=False)
        raise


if __name__ == "__main__":
    sys.exit(main())
