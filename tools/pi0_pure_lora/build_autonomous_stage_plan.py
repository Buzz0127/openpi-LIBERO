#!/usr/bin/env python3
"""Build an immutable, non-authorizing plan for one autonomous stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import autonomous_stage_orchestrator as orchestrator
import experiment_identity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--current-json", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-upstream", required=True)
    parser.add_argument("--orchestrator", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
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
    parser.add_argument("--sample-interval-seconds", required=True, type=float)
    parser.add_argument("--term-grace-seconds", required=True, type=float)
    parser.add_argument("--kill-grace-seconds", required=True, type=float)
    parser.add_argument("--max-retries", required=True, type=int)
    parser.add_argument("--retry-return-code", action="append", default=[], type=int)
    parser.add_argument("--retry-delay-seconds", required=True, type=float)
    parser.add_argument("--max-log-bytes", required=True, type=int)
    parser.add_argument("--log-backups", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("child command required after --")
    if args.segment_start < 0 or args.segment_end <= args.segment_start:
        parser.error("invalid segment")
    if args.expected_final_committed_step != args.segment_end or args.train_seed == args.eval_seed:
        parser.error("invalid step or seed identity")
    if args.max_retries < 0 or args.max_log_bytes < 1024 or args.log_backups < 1:
        parser.error("invalid retry or log bounds")
    return args


def main() -> int:
    args = _parse_args()
    if args.output.exists() or args.attempt_dir.exists():
        raise FileExistsError("plan output and attempt must be new")
    if args.current_json.resolve().parent != args.attempt_dir.resolve().parent:
        raise ValueError("current.json and attempt must share one stage parent")
    identities = orchestrator._parse_identities(args.identity)
    source = orchestrator._source_snapshot(args.source_root)
    expected = {
        "branch": args.expected_branch,
        "head": args.expected_head,
        "upstream": args.expected_upstream,
        "status": "",
    }
    if any(source[key] != value for key, value in expected.items()):
        raise RuntimeError(f"source identity mismatch: {source}")
    for tool in (args.orchestrator, args.launcher, Path(__file__)):
        if not tool.is_file():
            raise FileNotFoundError(tool)
    bounds = {
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
    plan = {
        "schema_version": 1,
        "stage": args.stage,
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "expected_final_committed_step": args.expected_final_committed_step,
        "source": source,
        "identities": identities,
        "attempt_dir": str(args.attempt_dir.resolve()),
        "current_json": str(args.current_json.resolve()),
        "progress_json": str(args.progress_json.resolve()),
        "required_outputs": [str(path.resolve()) for path in args.required_output],
        "bounds": bounds,
        "command": args.command,
        "tools": {
            "planner_sha256": experiment_identity.sha256_file(Path(__file__)),
            "orchestrator_sha256": experiment_identity.sha256_file(args.orchestrator),
            "launcher_sha256": experiment_identity.sha256_file(args.launcher),
        },
        "execution_authorized": False,
        "next_stage_auto_start": False,
    }
    plan["plan_identity_sha256"] = experiment_identity.canonical_sha256(plan)
    experiment_identity.atomic_write_new(args.output, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
