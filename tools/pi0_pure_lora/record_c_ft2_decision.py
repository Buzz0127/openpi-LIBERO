#!/usr/bin/env python3
"""Record the user-approved, narrowly scoped C-FT2 historical exception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import experiment_identity
import verify_ft2_result


def _summary(path: Path, expected_reason: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "fail" or value.get("reason_code") != expected_reason:
        raise RuntimeError(f"historical outer summary is not the expected {expected_reason} failure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ft1-result", required=True, type=Path)
    parser.add_argument("--ft1-acceptance", required=True, type=Path)
    parser.add_argument("--ft1-outer-summary", required=True, type=Path)
    parser.add_argument("--ft2-result", required=True, type=Path)
    parser.add_argument("--ft2-acceptance", required=True, type=Path)
    parser.add_argument("--ft2-outer-summary", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    expected_ft2 = verify_ft2_result.verify(args.ft1_acceptance, args.ft1_result, args.ft2_result)
    observed_ft2 = json.loads(args.ft2_acceptance.read_text(encoding="utf-8"))
    if observed_ft2 != expected_ft2:
        raise RuntimeError("historical FT2 acceptance differs from its independent verifier result")
    _summary(args.ft1_outer_summary, "child_exit_nonzero")
    _summary(args.ft2_outer_summary, "committed_step_mismatch")
    decision = {
        "schema_version": 1,
        "stage": "C-FT2-controlled-exception",
        "decision": "accept-existing-ft1-ft2-chain",
        "user_decision": "accept-narrow-historical-outer-exception",
        "accepted_inputs": {
            "ft1_result_sha256": experiment_identity.sha256_file(args.ft1_result),
            "ft1_acceptance_sha256": experiment_identity.sha256_file(args.ft1_acceptance),
            "ft1_outer_summary_sha256": experiment_identity.sha256_file(args.ft1_outer_summary),
            "ft2_result_sha256": experiment_identity.sha256_file(args.ft2_result),
            "ft2_acceptance_sha256": experiment_identity.sha256_file(args.ft2_acceptance),
            "ft2_outer_summary_sha256": experiment_identity.sha256_file(args.ft2_outer_summary),
        },
        "exception_scope": {
            "covers_only": {"FT1": "child_exit_nonzero", "FT2": "committed_step_mismatch"},
            "does_not_cover": ["identity_drift", "frozen_parameter_violation", "restore_failure", "future_outer_terminal_discrepancy"],
            "historical_outer_summaries_remain_failed": True,
        },
        "allowed_next_input": {"segment_start": 5000, "prior_result_sha256": experiment_identity.sha256_file(args.ft2_result),
                               "prior_acceptance_sha256": experiment_identity.sha256_file(args.ft2_acceptance)},
        "execution_authorized": False,
        "next_stage_auto_start": False,
    }
    decision["decision_identity_sha256"] = experiment_identity.canonical_sha256(decision)
    experiment_identity.atomic_write_new(args.output, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
