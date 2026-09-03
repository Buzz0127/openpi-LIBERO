#!/usr/bin/env python3
"""Run the guarded lean N1b R3 stage and compare the published stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import automate_n1b_norm_stats as common


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--common-helper", required=True, type=Path)
    parser.add_argument("--common-helper-sha256", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--canonical-target", required=True, type=Path)
    parser.add_argument("--official-stats", required=True, type=Path)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument("--identity", action="append", required=True)
    parser.add_argument("--monitor-root", action="append", required=True, type=Path)
    parser.add_argument("--expected-dataset-length", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    attempt = args.attempt_dir.resolve()
    if not attempt.is_dir():
        raise FileNotFoundError(f"attempt directory does not exist: {attempt}")
    if args.canonical_target.exists():
        raise FileExistsError(f"canonical target already exists: {args.canonical_target}")
    for path, expected, label in (
        (args.guard, args.guard_sha256, "guard"),
        (args.runner, args.runner_sha256, "runner"),
        (args.common_helper, args.common_helper_sha256, "common helper"),
    ):
        actual = common.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")

    start = time.time()
    guard_run = attempt / "guard-run"
    r3_report = guard_run / "r3_report.json"
    child = [
        str(args.python),
        str(args.runner),
        "--config-name",
        args.config_name,
        "--expected-target",
        str(args.canonical_target),
        "--expected-dataset-length",
        str(args.expected_dataset_length),
        "--batch-size",
        str(args.batch_size),
        "--sample-every",
        "100",
        "--report",
        str(r3_report),
    ]
    for identity in args.identity:
        child.extend(["--identity", identity])
    guard_command = [str(args.python), str(args.guard), "--attempt-dir", str(guard_run)]
    for root in args.monitor_root:
        guard_command.extend(["--monitor-root", str(root)])
    guard_command.extend(
        [
            "--existing-billed-bytes", "0",
            "--soft-limit-bytes", "500000000",
            "--hard-limit-bytes", "1000000000",
            "--timeout-seconds", str(args.timeout_seconds),
            "--sample-interval-seconds", "5",
            "--near-soft-margin-bytes", "100000000",
            "--near-sample-interval-seconds", "1",
            "--min-mem-available-bytes", str(64 * (1 << 30)),
            "--max-child-rss-bytes", str(6 * (1 << 30)),
            "--max-load1-per-cpu", "0.9",
            "--resource-consecutive-samples", "3",
            "--monitor-failure-consecutive-samples", "2",
            "--",
            *child,
        ]
    )
    common.atomic_json(
        attempt / "automation_manifest.json",
        {
            "status": "started",
            "phase": "N1b-R3",
            "start_time_epoch_seconds": start,
            "guard_command": guard_command,
            "environment_overrides": common._offline_overrides(),
            "explicitly_unset_environment": [
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"
            ],
        },
    )
    status_path = attempt / "automation_status.json"
    try:
        result = subprocess.run(guard_command, env=common._offline_environment(), check=False)
        if result.returncode != 0:
            raise RuntimeError(f"guard returned {result.returncode}")
        if not r3_report.is_file():
            raise RuntimeError("guard succeeded without R3 report")
        report = json.loads(r3_report.read_text())
        if report.get("status") != "pass":
            raise RuntimeError("R3 report is not pass")
        comparison = common.compare_stats(
            args.canonical_target / "norm_stats.json", args.official_stats, args.official_sha256
        )
        common.atomic_json(attempt / "comparison_report.json", comparison)
        common.atomic_json(
            status_path,
            {
                "status": "pass",
                "reason_code": "completed",
                "elapsed_seconds": time.time() - start,
                "attempt_dir": str(attempt),
                "canonical_target": str(args.canonical_target),
                "comparison": comparison,
            },
        )
        return 0
    except BaseException as error:
        common.atomic_json(
            status_path,
            {
                "status": "fail",
                "reason_code": type(error).__name__,
                "error": repr(error),
                "elapsed_seconds": time.time() - start,
                "attempt_dir": str(attempt),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
