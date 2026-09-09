#!/usr/bin/env python3
"""Build the CPU-only, non-executing FT0 formal-training freeze package.

This freezes the experiment contract for a fresh pi0_base trajectory.  It does
not import OpenPI/JAX, inspect a checkpoint, or produce an executable command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")
IDENTITY_KEYS = {"model", "dataset", "norm", "golden", "config", "split"}
CANDIDATE_STEPS = [1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
TRAINING_SEED = 42
EVALUATION_SEED = 7
FORMAL_BATCH_SIZE = 1
FORMAL_NUM_WORKERS = 0
TARGET_STEP = 30_000


def _sha256(value: str, label: str) -> str:
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _pairs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key or key in result:
            raise ValueError("identity must contain unique KEY=SHA256 pairs")
        result[key] = _sha256(value, f"identity.{key}")
    if set(result) != IDENTITY_KEYS:
        raise ValueError(f"identity keys must be exactly {sorted(IDENTITY_KEYS)}")
    return result


def _canonical(value: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def _new_absolute(path: str, label: str) -> str:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts or value.name in {"", ".", "/"}:
        raise ValueError(f"{label} must be a normalized, non-root absolute path")
    return str(value)


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.source_status != "clean" or args.source_upstream != "NONE":
        raise ValueError("the fixed OpenPI source must be clean and have no upstream")
    if not GIT_OID.fullmatch(args.source_head):
        raise ValueError("source_head must be a full lowercase Git object ID")
    if args.train_seed != TRAINING_SEED or args.eval_seed != EVALUATION_SEED:
        raise ValueError("FT0 freezes training/evaluation seeds to 42/7")
    if args.batch_size != FORMAL_BATCH_SIZE or args.num_workers != FORMAL_NUM_WORKERS:
        raise ValueError("FT0 freezes the shared-host loader to batch size 1 and zero workers")
    if args.target_step != TARGET_STEP or args.candidate_step != CANDIDATE_STEPS:
        raise ValueError(f"FT0 candidate steps must be exactly {CANDIDATE_STEPS}")
    if not args.run_name.startswith("pi0-libero-pure-lora-ft0-seed42-"):
        raise ValueError("run_name must identify the fresh FT0 seed-42 trajectory")
    if any(value <= 0 for value in (args.current_billed_bytes, args.full_state_bytes, args.adapter_bytes, args.atomic_margin_bytes)):
        raise ValueError("storage inputs must be positive")
    if not 0 < args.review_line_bytes < args.soft_stop_bytes < args.hard_limit_bytes:
        raise ValueError("storage thresholds are invalid")
    if args.minimum_uncommitted_bytes <= 0:
        raise ValueError("minimum uncommitted storage reserve must be positive")

    identities = _pairs(args.identity)
    source_files = {
        "training_config": {"path": _new_absolute(args.config_source, "config_source"), "sha256": _sha256(args.config_source_sha256, "config_source_sha256")},
        "optimizer": {"path": _new_absolute(args.optimizer_source, "optimizer_source"), "sha256": _sha256(args.optimizer_source_sha256, "optimizer_source_sha256")},
    }
    checkpoint_root = _new_absolute(args.checkpoint_root, "checkpoint_root")
    adapter_root = _new_absolute(args.adapter_root, "adapter_root")
    checkpoint_run_root = str(Path(checkpoint_root) / args.run_name)
    adapter_run_root = str(Path(adapter_root) / args.run_name)

    segments = []
    start = 0
    for end in CANDIDATE_STEPS:
        segments.append(
            {
                "start": start,
                "end": end,
                "candidate_adapter_publish": True,
                "full_state_required_for_resume": True,
                "user_authorization_required": True,
                "auto_start_next_segment": False,
            }
        )
        start = end

    # Do not assume any old full state can be deleted.  The extra state is the
    # transient newly written state while the preceding known-good state still
    # exists, and the margin covers atomic metadata/evidence publication.
    retained_candidate_states = len(CANDIDATE_STEPS)
    retained_candidate_adapters = len(CANDIDATE_STEPS)
    steady_peak = args.current_billed_bytes + retained_candidate_states * args.full_state_bytes + retained_candidate_adapters * args.adapter_bytes
    worst_peak = steady_peak + args.full_state_bytes + args.atomic_margin_bytes
    if worst_peak >= args.review_line_bytes:
        raise ValueError("worst-case formal-training peak reaches the review line")
    if args.hard_limit_bytes - worst_peak < args.minimum_uncommitted_bytes:
        raise ValueError("worst-case formal-training peak leaves insufficient hard-limit reserve")

    package: dict[str, object] = {
        "schema_version": 1,
        "stage": "FT0-formal-training-freeze",
        "execution_authorized": False,
        "execution_ready": False,
        "candidate": False,
        "source": {
            "root": _new_absolute(args.source_root, "source_root"),
            "branch": args.source_branch,
            "head": args.source_head,
            "status": args.source_status,
            "upstream": args.source_upstream,
        },
        "source_files": source_files,
        "identities": identities,
        "formal_training": {
            "starts_from": "pi0_base",
            "starts_from_engineering_step_200": False,
            "run_name": args.run_name,
            "checkpoint_run_root": checkpoint_run_root,
            "adapter_run_root": adapter_run_root,
            "training_seed": TRAINING_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "batch_size": FORMAL_BATCH_SIZE,
            "num_workers": FORMAL_NUM_WORKERS,
            "target_step": TARGET_STEP,
            "candidate_steps": CANDIDATE_STEPS,
            "segments": segments,
            "checkpoint_schedule": {
                "save_only_at_segment_end": True,
                "automatic_pruning_enabled": False,
                "checkpoint_max_to_keep": None,
                "keep_period": None,
            },
            "adapter_policy": {
                "publication": "adapter_only_at_candidate_steps",
                "full_state_role": "resume_only",
                "adapter_release_requires_verified_full_restore": True,
            },
            "optimizer": {
                "name": "AdamW",
                "b1": 0.9,
                "b2": 0.95,
                "eps": 1e-8,
                "weight_decay": 1e-10,
                "clip_gradient_norm": 1.0,
            },
            "lr_schedule": {
                "name": "CosineDecaySchedule",
                "warmup_steps": 1_000,
                "peak_lr": 2.5e-5,
                "decay_steps": TARGET_STEP,
                "decay_lr": 2.5e-6,
            },
            "logging": {"log_interval_steps": 100, "wandb_enabled": False},
            "offline_required": True,
        },
        "storage": {
            "current_billed_bytes": args.current_billed_bytes,
            "full_state_bytes_upper_bound": args.full_state_bytes,
            "adapter_bytes_upper_bound": args.adapter_bytes,
            "atomic_write_margin_bytes": args.atomic_margin_bytes,
            "retained_candidate_full_states": retained_candidate_states,
            "retained_candidate_adapters": retained_candidate_adapters,
            "steady_peak_without_deletion_bytes": steady_peak,
            "worst_peak_old_plus_new_full_state_bytes": worst_peak,
            "review_line_bytes": args.review_line_bytes,
            "soft_stop_bytes": args.soft_stop_bytes,
            "hard_limit_bytes": args.hard_limit_bytes,
            "minimum_uncommitted_bytes": args.minimum_uncommitted_bytes,
            "review_headroom_bytes": args.review_line_bytes - worst_peak,
            "soft_headroom_bytes": args.soft_stop_bytes - worst_peak,
            "hard_headroom_bytes": args.hard_limit_bytes - worst_peak,
            "deletion_authorized": False,
            "automatic_pruning_allowed": False,
        },
        "selection_protocol": {
            "development_only": True,
            "main_never_selects": True,
            "tie_break": ["higher_dev_success", "earlier_train_step", "lexicographically_smaller_adapter_identity"],
            "candidate_addition_after_first_dev_episode": False,
        },
        "required_before_first_segment": [
            "implement and CPU-test a formal-segment runner for 0->1000 and later exact resumes",
            "bind the formal runner, launcher, orchestrator, guards, and verifier by committed tool snapshot SHA-256",
            "validate a collision-safe FT1 plan with a fresh 30-second dual-GPU plus CPU/RAM preflight",
            "obtain separate user authorization for FT1 0->1000",
        ],
        "automatic_next_stage": False,
        "execution_command": None,
    }
    package["package_identity_sha256"] = _canonical(package)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-status", default="clean")
    parser.add_argument("--source-upstream", default="NONE")
    parser.add_argument("--identity", action="append", required=True)
    parser.add_argument("--config-source", required=True)
    parser.add_argument("--config-source-sha256", required=True)
    parser.add_argument("--optimizer-source", required=True)
    parser.add_argument("--optimizer-source-sha256", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--target-step", required=True, type=int)
    parser.add_argument("--candidate-step", required=True, type=int, action="append")
    parser.add_argument("--current-billed-bytes", required=True, type=int)
    parser.add_argument("--full-state-bytes", required=True, type=int)
    parser.add_argument("--adapter-bytes", required=True, type=int)
    parser.add_argument("--atomic-margin-bytes", required=True, type=int)
    parser.add_argument("--review-line-bytes", required=True, type=int)
    parser.add_argument("--soft-stop-bytes", required=True, type=int)
    parser.add_argument("--hard-limit-bytes", required=True, type=int)
    parser.add_argument("--minimum-uncommitted-bytes", required=True, type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(package, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
