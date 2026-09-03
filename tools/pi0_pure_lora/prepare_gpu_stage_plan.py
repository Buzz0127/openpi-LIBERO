"""Seal a GPU stage command after a valid 30-sample dual-GPU preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import experiment_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--stage-plan", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--max-runtime-seconds", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("stage command required after --")
    preflight = json.loads(args.preflight_report.read_text())
    saved_hash = preflight.pop("preflight_identity_sha256", None)
    if len(preflight.get("samples", [])) < 30 or preflight.get("sample_count") < 30:
        raise ValueError("GPU stage requires at least 30 dual-GPU samples")
    if experiment_identity.canonical_sha256(preflight) != saved_hash:
        raise ValueError("preflight identity mismatch")
    if experiment_identity.sha256_file(args.guard) != args.expected_guard_sha256:
        raise ValueError("GPU guard SHA-256 mismatch")
    stage = json.loads(args.stage_plan.read_text())
    if stage.get("execution_authorized") is not False:
        raise ValueError("stage plan must remain explicitly non-authorizing")
    selected = int(preflight["selected_physical_gpu"])
    guard_command = [
        str(args.python), str(args.guard), "--physical-gpu", str(selected),
        "--pause-at", "95", "--resume-at", "85", "--min-free-memory-percent", "15",
        "--resume-free-memory-percent", "20", "--terminate-free-memory-percent", "10",
        "--resume-samples", "5", "--interval-seconds", "1", "--monitor-error-limit", "3",
        "--max-prelaunch-wait-seconds", "300", "--max-runtime-seconds", str(args.max_runtime_seconds),
        "--terminate-grace-seconds", "15", "--", *command,
    ]
    result = {
        "schema_version": 1,
        "stage_plan_identity_sha256": stage["plan_identity_sha256"],
        "preflight_identity_sha256": saved_hash,
        "selected_physical_gpu": selected,
        "selected_gpu_uuid": preflight["selected_gpu_uuid"],
        "environment": {"CUDA_VISIBLE_DEVICES": str(selected), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        "guard_command": guard_command,
        "guard_controls_only_child_process_group": True,
        "external_monitor_outside_child_process_group": True,
        "mapping_verification_required_before_workload": True,
        "execution_authorized": False,
    }
    result["gpu_stage_identity_sha256"] = experiment_identity.canonical_sha256(result)
    experiment_identity.atomic_write_new(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
