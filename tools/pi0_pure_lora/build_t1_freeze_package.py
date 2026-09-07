#!/usr/bin/env python3
"""Build a non-executing, fail-closed T1 training freeze package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")
IDENTITY_KEYS = {"model", "dataset", "norm", "golden", "config", "split"}


def _sha(value: str, label: str) -> str:
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _git_oid(value: str) -> str:
    if not GIT_OID.fullmatch(value):
        raise ValueError("source_head must be a lowercase full Git object ID")
    return value


def _pairs(values: list[str], required: set[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or key in result:
            raise ValueError(f"{label} must contain unique KEY=SHA256 pairs")
        result[key] = _sha(value, f"{label}.{key}")
    if set(result) != required:
        raise ValueError(f"{label} keys must be exactly {sorted(required)}")
    return result


def _canonical_identity(value: dict[str, object]) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def build(args: argparse.Namespace) -> dict[str, object]:
    identities = _pairs(args.identity, IDENTITY_KEYS, "identity")
    tools = _pairs(args.tool_sha256, set(args.required_tool), "tool_sha256")
    if args.source_status != "clean":
        raise ValueError("OpenPI source worktree must be clean")
    if args.source_upstream != "NONE":
        raise ValueError("remote OpenPI worktree upstream must remain NONE")
    if args.train_seed == args.eval_seed:
        raise ValueError("training and evaluation seeds must differ")
    if (args.engineering_start, args.engineering_end) != (100, 200):
        raise ValueError("first autonomy/resume engineering segment is frozen to 100->200")
    candidates = args.candidate_step
    if candidates != sorted(set(candidates)) or not candidates or candidates[-1] != args.training_target_step:
        raise ValueError("candidate steps must be unique, sorted, and end at the training target")
    expected_candidates = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
    if candidates != expected_candidates:
        raise ValueError(f"candidate steps must be exactly {expected_candidates}")
    if args.checkpoint_bytes <= 0 or args.adapter_bytes <= 0 or args.atomic_margin_bytes <= 0:
        raise ValueError("artifact size estimates must be positive")

    retained_new_milestones = 1 + len(candidates)  # engineering step 200 plus formal candidates.
    steady_peak = args.current_billed_bytes + retained_new_milestones * (
        args.checkpoint_bytes + args.adapter_bytes
    )
    worst_peak = steady_peak + args.checkpoint_bytes + args.atomic_margin_bytes
    if worst_peak >= args.review_line_bytes:
        raise ValueError("worst-case peak reaches the mandatory review line")
    if args.hard_limit_bytes - worst_peak < args.minimum_uncommitted_bytes:
        raise ValueError("worst-case peak leaves insufficient uncommitted headroom")
    if not (
        args.review_line_bytes < args.soft_stop_bytes < args.hard_limit_bytes
        and args.minimum_uncommitted_bytes > 0
    ):
        raise ValueError("invalid storage thresholds")

    formal_segments = []
    start = 0
    for end in candidates:
        formal_segments.append({"start": start, "end": end, "candidate": True})
        start = end
    package: dict[str, object] = {
        "schema_version": 1,
        "stage": "T1-preflight-freeze",
        "execution_authorized": False,
        "execution_ready": False,
        "source_snapshot_kind": "fixed_head_plus_uncommitted_file_hashes",
        "source": {
            "root": args.source_root,
            "branch": args.source_branch,
            "head": _git_oid(args.source_head),
            "status": args.source_status,
            "upstream": args.source_upstream,
        },
        "identities": identities,
        "tool_sha256": tools,
        "seeds": {"training": args.train_seed, "evaluation": args.eval_seed},
        "resume_input": {
            "checkpoint_root": args.resume_checkpoint_root,
            "checkpoint_step": 100,
            "checkpoint_tree_sha256": _sha(args.resume_checkpoint_tree_sha256, "checkpoint_tree_sha256"),
            "adapter_identity_sha256": _sha(args.resume_adapter_identity_sha256, "adapter_identity_sha256"),
            "restore_receipt_sha256": _sha(args.resume_receipt_sha256, "restore_receipt_sha256"),
            "acceptance_report_identity_sha256": _sha(
                args.resume_acceptance_identity_sha256, "acceptance_report_identity_sha256"
            ),
            "all_parameter_and_optimizer_values_restored": True,
            "automatic_pruning_enabled": False,
            "old_checkpoint_deletion_authorized": False,
        },
        "engineering_segment": {
            "purpose": "autonomous resume and deterministic loader-position validation only",
            "start": args.engineering_start,
            "end": args.engineering_end,
            "candidate": False,
            "batch_size": 1,
            "num_workers": 0,
            "expected_new_checkpoint_step": args.engineering_end,
            "expected_new_adapter_step": args.engineering_end,
            "delete_step_100_after_success": False,
        },
        "formal_training": {
            "starts_from_base_in_a_new_run_root": True,
            "target_step": args.training_target_step,
            "segments": formal_segments,
            "candidate_steps": candidates,
            "development_selection_protocol": "E0 outcome-blind preregistration; dev-40 only; main-200 never selects",
            "tie_break": ["higher_dev_success", "earlier_train_step", "lexicographically_smaller_adapter_identity"],
            "append_candidates_after_first_dev_episode": False,
        },
        "storage": {
            "current_billed_bytes": args.current_billed_bytes,
            "measured_full_state_bytes": args.checkpoint_bytes,
            "measured_adapter_bytes": args.adapter_bytes,
            "atomic_write_margin_bytes": args.atomic_margin_bytes,
            "retained_new_milestones_for_no_deletion_bound": retained_new_milestones,
            "steady_peak_if_no_deletion_bytes": steady_peak,
            "worst_peak_with_one_extra_full_state_and_margin_bytes": worst_peak,
            "review_line_bytes": args.review_line_bytes,
            "soft_stop_bytes": args.soft_stop_bytes,
            "hard_limit_bytes": args.hard_limit_bytes,
            "minimum_uncommitted_bytes": args.minimum_uncommitted_bytes,
            "hard_headroom_at_worst_peak_bytes": args.hard_limit_bytes - worst_peak,
            "soft_headroom_at_worst_peak_bytes": args.soft_stop_bytes - worst_peak,
            "automatic_pruning_allowed": False,
            "deletion_authorized": False,
        },
        "required_before_execution": [
            "implement and unit-test exact resume data-loader position handling",
            "bind the concrete runner and both guards by SHA-256",
            "create a collision-safe A2 stage plan with exact command and output paths",
            "repeat at least 30 seconds of dual-GPU plus CPU/RAM preflight",
            "dynamically select and pin one physical GPU with more than 15 percent free VRAM",
            "verify XLA_PYTHON_CLIENT_PREALLOCATE=false and single-card mapping",
            "obtain separate user authorization for the GPU engineering segment",
        ],
        "blocking_facts": [
            {
                "code": "resume_loader_position_unverified",
                "evidence": "OpenPI restore_state discards data_loader and train.py recreates its iterator from the beginning",
            },
            {
                "code": "gpu_execution_not_authorized",
                "evidence": "this package is a CPU-only non-executing freeze artifact",
            },
        ],
        "automatic_next_stage": False,
        "execution_command": None,
    }
    package["package_identity_sha256"] = _canonical_identity(package)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-status", default="clean")
    parser.add_argument("--source-upstream", default="NONE")
    parser.add_argument("--identity", action="append", default=[], required=True)
    parser.add_argument("--required-tool", action="append", default=[], required=True)
    parser.add_argument("--tool-sha256", action="append", default=[], required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--resume-checkpoint-root", required=True)
    parser.add_argument("--resume-checkpoint-tree-sha256", required=True)
    parser.add_argument("--resume-adapter-identity-sha256", required=True)
    parser.add_argument("--resume-receipt-sha256", required=True)
    parser.add_argument("--resume-acceptance-identity-sha256", required=True)
    parser.add_argument("--engineering-start", type=int, required=True)
    parser.add_argument("--engineering-end", type=int, required=True)
    parser.add_argument("--training-target-step", type=int, required=True)
    parser.add_argument("--candidate-step", type=int, action="append", default=[], required=True)
    parser.add_argument("--current-billed-bytes", type=int, required=True)
    parser.add_argument("--checkpoint-bytes", type=int, required=True)
    parser.add_argument("--adapter-bytes", type=int, required=True)
    parser.add_argument("--atomic-margin-bytes", type=int, required=True)
    parser.add_argument("--review-line-bytes", type=int, required=True)
    parser.add_argument("--soft-stop-bytes", type=int, required=True)
    parser.add_argument("--hard-limit-bytes", type=int, required=True)
    parser.add_argument("--minimum-uncommitted-bytes", type=int, required=True)
    args = parser.parse_args()
    package = build(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(package, indent=2, sort_keys=True) + "\n").encode()
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
