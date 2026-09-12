#!/usr/bin/env python3
"""Run one future formal-segment runner, then require terminal acceptance.

The interpreter is intentionally ``sys.executable``: launch plans bind the
fixed OpenPI virtualenv path and must not resolve it to a UV base interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import experiment_identity


def _validate_fresh_preflight(path: Path, expected_sha256: str, selected_gpu: int) -> None:
    """Bind the guarded child to a fresh, complete, selected-card preflight."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("preflight file SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(report); claimed = unsigned.pop("preflight_identity_sha256", None)
    if claimed != experiment_identity.canonical_sha256(unsigned):
        raise RuntimeError("preflight internal identity mismatch")
    samples = report.get("samples", [])
    if report.get("sample_count") != len(samples) or len(samples) < 30:
        raise RuntimeError("preflight lacks 30 complete samples")
    if report.get("selected_physical_gpu") != selected_gpu:
        raise RuntimeError("preflight selected GPU differs from launch binding")
    finished = report.get("collection_finished_epoch_seconds")
    if not isinstance(finished, (int, float)) or not 0 <= time.time() - finished <= 120:
        raise RuntimeError("preflight is stale at formal driver launch")
    latest = samples[-1]
    matches = [gpu for gpu in latest.get("gpus", []) if gpu.get("index") == selected_gpu]
    if len(matches) != 1 or matches[0].get("free_memory_percent", 0) <= 15 or matches[0].get("utilization_percent", 100) >= 95:
        raise RuntimeError("selected GPU no longer meets preflight launch gate")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(selected_gpu) or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("formal driver environment does not preserve selected single-GPU mapping")


def _validate_c_ft2_decision(path: Path, prior_acceptance: Path, prior_result: Path, segment_start: int) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value); claimed = unsigned.pop("decision_identity_sha256", None)
    if claimed != experiment_identity.canonical_sha256(unsigned):
        raise RuntimeError("C-FT2 decision identity mismatch")
    allowed = value.get("allowed_next_input", {})
    if (value.get("stage"), value.get("decision"), value.get("execution_authorized"), value.get("next_stage_auto_start")) != (
        "C-FT2-controlled-exception", "accept-existing-ft1-ft2-chain", False, False,
    ):
        raise RuntimeError("C-FT2 decision scope or authorization mismatch")
    if (allowed.get("segment_start") != segment_start
            or allowed.get("prior_acceptance_sha256") != hashlib.sha256(prior_acceptance.read_bytes()).hexdigest()
            or allowed.get("prior_result_sha256") != hashlib.sha256(prior_result.read_bytes()).hexdigest()):
        raise RuntimeError("C-FT2 decision does not bind this FT3 prior chain")


def _validate_support_package_importable() -> None:
    if importlib.util.find_spec("pi0_pure_lora") is None:
        raise RuntimeError("PYTHONPATH must expose the pi0_pure_lora package parent before GPU work")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--prior-acceptance", required=True, type=Path)
    parser.add_argument("--prior-result", required=True, type=Path)
    parser.add_argument("--prior-identity-schema", required=True, choices=("ft2-newline-v1", "formal-canonical-v1"))
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--acceptance-output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--selected-physical-gpu", required=True, type=int)
    parser.add_argument("--c-ft2-decision", type=Path)
    parser.add_argument("--segment-start", required=True, type=int)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    if args.segment_end <= args.segment_start or not runner_args or args.acceptance_output.exists():
        raise ValueError("new acceptance output, valid segment, and runner arguments are required")
    _validate_fresh_preflight(args.preflight_report, args.expected_preflight_sha256, args.selected_physical_gpu)
    if args.prior_identity_schema == "ft2-newline-v1":
        if args.c_ft2_decision is None:
            raise RuntimeError("FT3 requires an explicit C-FT2 decision record")
        _validate_c_ft2_decision(args.c_ft2_decision, args.prior_acceptance, args.prior_result, args.segment_start)
    elif args.c_ft2_decision is not None:
        raise ValueError("C-FT2 decision is only valid for the FT3 historical transition")
    _validate_support_package_importable()
    subprocess.run([sys.executable, str(args.runner), *runner_args], check=True)
    try:
        progress = json.loads(args.progress.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("runner did not publish readable scalar progress") from error
    if progress != {"current_step": args.segment_end, "last_committed_step": args.segment_end}:
        raise RuntimeError("runner did not commit exact final scalar progress before terminal acceptance")
    subprocess.run([
        sys.executable, str(args.verifier), "--prior-acceptance", str(args.prior_acceptance),
        "--prior-result", str(args.prior_result), "--prior-identity-schema", args.prior_identity_schema,
        "--result", str(args.result), "--segment-start", str(args.segment_start),
        "--segment-end", str(args.segment_end), "--output", str(args.acceptance_output),
    ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
