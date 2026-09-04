#!/usr/bin/env python3
"""Fail-closed verifier for a completed S1c ten-step evidence attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import experiment_identity


def _storage(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw_bytes, raw_path = line.split(maxsplit=1)
        values[raw_path] = int(raw_bytes)
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--expected-model-identity", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--max-stage-increment-bytes", default=1_000_000_000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    attempt = args.attempt.resolve()
    preflight = json.loads((attempt / "run/preflight.json").read_text())
    claimed_preflight_identity = preflight.pop("preflight_identity_sha256")
    if experiment_identity.canonical_sha256(preflight) != claimed_preflight_identity:
        raise RuntimeError("preflight identity mismatch")
    if preflight["sample_count"] < 30 or len(preflight["samples"]) < 30:
        raise RuntimeError("fewer than 30 preflight samples")
    selected = int(preflight["selected_physical_gpu"])

    input_hashes = (attempt / "static/inputs.sha256").read_text()
    if args.expected_tool_sha256 not in input_hashes or args.expected_guard_sha256 not in input_hashes:
        raise RuntimeError("tool or guard identity absent from static inputs")
    if (attempt / "run/exit_code.txt").read_text().strip() != "0":
        raise RuntimeError("guard exit code was not zero")
    result = json.loads((attempt / "run/result.json").read_text())
    checks = {
        "stage_is_s1c": result.get("stage") == "S1c",
        "model_identity_matches": result.get("model_identity_sha256") == args.expected_model_identity,
        "exactly_ten_steps": result.get("training_steps_executed") == result.get("final_step") == 10,
        "ten_metric_records": len(result.get("metrics_trace", [])) == 10,
        "all_metrics_finite": result.get("all_metrics_finite") is True and all(
            math.isfinite(value) for metrics in result.get("metrics_trace", []) for value in metrics.values()
        ),
        "all_golden_leaves_changed": result.get("golden_leaf_count") == result.get("changed_golden_leaf_count") == 20,
        "all_non_golden_leaves_frozen": result.get("non_golden_leaf_count") == 50
        and result.get("changed_non_golden_leaf_count") == 0
        and result.get("non_golden_hashes_unchanged") is True,
        "no_checkpoint_written": result.get("checkpoint_written") is False,
        "no_adapter_written": result.get("adapter_written") is False,
        "openpi_worktree_clean_before": not (attempt / "static/git-status-before.txt").read_text().strip(),
        "openpi_worktree_clean_after": not (attempt / "static/git-status-after.txt").read_text().strip(),
        "checkpoint_root_absent": not args.checkpoint_root.exists(),
        "adapter_root_absent": not args.adapter_root.exists(),
    }

    guard_events = [json.loads(line) for line in (attempt / "run/guard.jsonl").read_text().splitlines() if line]
    gpu_samples = [event for event in guard_events if event["event"] == "gpu_sample"]
    checks.update(
        {
            "guard_child_exit_zero": bool(guard_events)
            and guard_events[-1]["event"] == "child_exited"
            and guard_events[-1]["return_code"] == 0,
            "guard_selected_same_gpu": guard_events[0].get("physical_gpu") == selected,
            "guard_no_monitor_error": not any(event["event"] == "monitor_error" for event in guard_events),
            "guard_no_memory_emergency": not any(event["event"] == "memory_emergency" for event in guard_events),
        }
    )

    before = _storage(attempt / "static/storage-before.txt")
    after = _storage(attempt / "static/storage-after.txt")
    deltas = {path: after[path] - before[path] for path in before if path in after}
    positive_delta = sum(max(0, value) for value in deltas.values())
    checks["storage_increment_within_bound"] = positive_delta <= args.max_stage_increment_bytes
    for unchanged in (
        "/home/wengzr/projects/openpi-lora-cache",
        "/home/wengzr/.cache/openpi",
        "/home/wengzr/.cache/uv",
    ):
        checks[f"unchanged:{unchanged}"] = deltas.get(unchanged) == 0

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"S1c acceptance failed: {failed}")
    report = {
        "schema_version": 1,
        "stage": "S1c",
        "status": "pass",
        "checks": checks,
        "preflight_identity_sha256": claimed_preflight_identity,
        "selected_physical_gpu": selected,
        "selected_gpu_uuid": preflight["selected_gpu_uuid"],
        "result_sha256": _sha256(attempt / "run/result.json"),
        "guard_sha256": _sha256(attempt / "run/guard.jsonl"),
        "guard_summary": {
            "runtime_samples": len(gpu_samples),
            "max_utilization_percent": max(sample["utilization_percent"] for sample in gpu_samples),
            "minimum_free_memory_percent": min(sample["free_memory_percent"] for sample in gpu_samples),
            "pause_events": sum(sample.get("action") == "paused" for sample in gpu_samples),
            "resume_events": sum(sample.get("action") == "resumed" for sample in gpu_samples),
        },
        "storage_deltas_bytes": deltas,
        "positive_storage_increment_bytes": positive_delta,
        "result_summary": {
            "elapsed_seconds": result["elapsed_seconds"],
            "training_steps_executed": result["training_steps_executed"],
            "changed_golden_leaf_count": result["changed_golden_leaf_count"],
            "changed_non_golden_leaf_count": result["changed_non_golden_leaf_count"],
            "checkpoint_written": result["checkpoint_written"],
            "adapter_written": result["adapter_written"],
        },
    }
    report["report_identity_sha256"] = experiment_identity.canonical_sha256(report)
    experiment_identity.atomic_write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
