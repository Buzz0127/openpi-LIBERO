#!/usr/bin/env python3
"""Fail-closed verification for the completed S1d 100-step checkpoint attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import experiment_identity


EXPECTED_STEP = 100
EXPECTED_SEED = 42
EXPECTED_GOLDEN_LEAVES = 20
EXPECTED_NON_GOLDEN_LEAVES = 50
EXPECTED_PARAMETER_LEAVES = 70
EXPECTED_OPTIMIZER_LEAVES = 42
SOFT_LIMIT_BYTES = 240_000_000_000
HARD_LIMIT_BYTES = 250_000_000_000
UNCHANGED_CACHE_ROOTS = {
    "/home/wengzr/projects/openpi-lora-cache",
    "/home/wengzr/.cache/openpi",
    "/home/wengzr/.cache/uv",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_regular_file(path: Path) -> dict[str, Any]:
    before = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact entry is not a regular file: {path}")
    digest = _sha256(path)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return {"bytes": before.st_size, "sha256": digest}


def _artifact_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"artifact root is not a real directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains a symlink: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = _hash_regular_file(path)
    if not files:
        raise RuntimeError(f"artifact contains no files: {root}")
    stable = {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files.values()),
        "files": files,
    }
    return {**stable, "artifact_tree_sha256": experiment_identity.canonical_sha256(stable)}


def _adapter_identity(manifest: dict[str, Any]) -> str:
    keys = ("schema_version", "artifact_type", "identities", "train_step", "train_seed", "entries")
    if not all(key in manifest for key in keys):
        raise RuntimeError("adapter manifest is missing stable identity fields")
    return experiment_identity.canonical_sha256({key: manifest[key] for key in keys})


def _has_partial_or_temporary_path(root: Path) -> bool:
    markers = (".partial", ".tmp", "tmp-")
    return any(any(marker in path.name for marker in markers) for path in root.rglob("*"))


def _metric_values(metrics_trace: object) -> list[float]:
    if not isinstance(metrics_trace, list):
        return []
    values: list[float] = []
    for record in metrics_trace:
        if not isinstance(record, dict):
            return []
        for value in record.values():
            if not isinstance(value, (int, float)):
                return []
            values.append(float(value))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--expected-model-identity", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-gpu-guard-sha256", required=True)
    parser.add_argument("--expected-storage-guard-sha256", required=True)
    parser.add_argument("--max-stage-increment-bytes", default=30_000_000_000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    attempt = args.attempt.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    adapter_root = args.adapter_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)

    preflight = _load_json(attempt / "run/preflight.json")
    claimed_preflight_identity = preflight.get("preflight_identity_sha256")
    stable_preflight = dict(preflight)
    stable_preflight.pop("preflight_identity_sha256", None)
    preflight_identity_valid = (
        claimed_preflight_identity is not None
        and experiment_identity.canonical_sha256(stable_preflight) == claimed_preflight_identity
    )
    selected_gpu = int(preflight["selected_physical_gpu"])
    selected_uuid = preflight["selected_gpu_uuid"]
    samples = preflight.get("samples", [])

    input_hashes = (attempt / "static/inputs.sha256").read_text(encoding="utf-8")
    result = _load_json(attempt / "run/result.json")
    receipt = result.get("receipt", {})
    if not isinstance(receipt, dict):
        raise RuntimeError("embedded checkpoint receipt is not an object")
    receipt_path = adapter_root / f"step-{EXPECTED_STEP:08d}.verified.json"
    disk_receipt = _load_json(receipt_path)
    adapter_dir = adapter_root / f"step-{EXPECTED_STEP:08d}"
    adapter_manifest = _load_json(adapter_dir / "manifest.json")
    entries = adapter_manifest.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("adapter entries are not a list")

    checkpoint_tree = _artifact_manifest(checkpoint_dir)
    adapter_tree = _artifact_manifest(adapter_root)
    adapter_files = {str(path.relative_to(adapter_dir)) for path in adapter_dir.rglob("*") if path.is_file()}
    declared_adapter_files = {"manifest.json"} | {
        entry.get("file") for entry in entries if isinstance(entry, dict) and isinstance(entry.get("file"), str)
    }
    adapter_file_hashes_valid = all(
        isinstance(entry, dict)
        and isinstance(entry.get("file"), str)
        and (adapter_dir / entry["file"]).is_file()
        and _sha256(adapter_dir / entry["file"]) == entry.get("file_sha256")
        for entry in entries
    )

    gpu_events = [
        json.loads(line)
        for line in (attempt / "run/gpu-guard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gpu_samples = [event for event in gpu_events if event.get("event") == "gpu_sample"]
    storage_status = _load_json(attempt / "run/storage-guard/exit_status.json")
    storage_manifest = _load_json(attempt / "run/storage-guard/run_manifest.json")
    final_sample = storage_status.get("final_sample", {})
    roots = final_sample.get("roots", []) if isinstance(final_sample, dict) else []
    root_deltas = {
        record["path"]: int(record["billed_delta_bytes"])
        for record in roots
        if isinstance(record, dict) and "path" in record and "billed_delta_bytes" in record
    }
    metric_values = _metric_values(result.get("metrics_trace"))

    checkpoint_top = {path.name for path in checkpoint_dir.iterdir()}
    adapter_top = {path.name for path in adapter_root.iterdir()}
    checks = {
        "preflight_identity_valid": preflight_identity_valid,
        "preflight_has_at_least_30_samples": preflight.get("sample_count") == len(samples) and len(samples) >= 30,
        "preflight_selected_gpu_uuid_present": all(
            any(
                gpu.get("index") == selected_gpu and gpu.get("uuid") == selected_uuid
                for gpu in sample.get("gpus", [])
            )
            for sample in samples
        ),
        "preflight_selected_gpu_above_15_percent_free": all(
            any(
                gpu.get("index") == selected_gpu
                and float(gpu.get("free_memory_percent", -1)) > 15.0
                for gpu in sample.get("gpus", [])
            )
            for sample in samples
        ),
        "tool_hash_pinned": args.expected_tool_sha256 in input_hashes,
        "gpu_guard_hash_pinned": args.expected_gpu_guard_sha256 in input_hashes,
        "storage_guard_hash_pinned": args.expected_storage_guard_sha256 in input_hashes,
        "outer_exit_zero": (attempt / "run/exit_code.txt").read_text(encoding="utf-8").strip() == "0",
        "stage_is_s1d": result.get("stage") == "S1d",
        "model_identity_matches": result.get("model_identity_sha256") == args.expected_model_identity,
        "exactly_100_steps": result.get("initial_step") == 0
        and result.get("training_steps_executed") == result.get("final_step") == EXPECTED_STEP,
        "fixed_training_seed": result.get("training_seed") == EXPECTED_SEED,
        "bounded_loader": result.get("batch_size") == 1 and result.get("num_workers") == 0 and result.get("shuffle") is True,
        "100_finite_metric_records": len(result.get("metrics_trace", [])) == EXPECTED_STEP
        and bool(metric_values)
        and all(math.isfinite(value) for value in metric_values)
        and result.get("all_metrics_finite") is True,
        "all_golden_changed": result.get("golden_leaf_count") == EXPECTED_GOLDEN_LEAVES
        and result.get("changed_golden_leaf_count") == EXPECTED_GOLDEN_LEAVES,
        "all_non_golden_frozen": result.get("non_golden_leaf_count") == EXPECTED_NON_GOLDEN_LEAVES
        and result.get("changed_non_golden_leaf_count") == 0
        and result.get("non_golden_hashes_unchanged") is True,
        "checkpoint_written_at_step_100": result.get("checkpoint_written") is True
        and result.get("checkpoint_steps") == [EXPECTED_STEP]
        and checkpoint_top == {str(EXPECTED_STEP)},
        "checkpoint_size_matches": result.get("checkpoint_total_bytes") == checkpoint_tree["total_bytes"],
        "checkpoint_restore_verified": result.get("checkpoint_restore_verified") is True,
        "adapter_written_at_step_100": result.get("adapter_written") is True
        and adapter_top == {f"step-{EXPECTED_STEP:08d}", f"step-{EXPECTED_STEP:08d}.verified.json"},
        "adapter_size_matches": result.get("adapter_total_bytes") == adapter_tree["total_bytes"],
        "base_plus_adapter_verified": result.get("base_plus_adapter_tree_shape_hash_verified") is True,
        "receipt_matches_disk": receipt == disk_receipt,
        "receipt_full_value_restore": all(
            receipt.get(key) is True
            for key in (
                "checkpoint_restore_succeeded",
                "parameter_tree_shape_dtype_equal",
                "optimizer_tree_shape_dtype_equal",
                "all_parameter_values_equal_after_restore",
                "all_optimizer_values_equal_after_restore",
                "adapter_values_equal_after_restore",
            )
        ),
        "receipt_leaf_counts": receipt.get("parameter_leaf_count") == EXPECTED_PARAMETER_LEAVES
        and receipt.get("optimizer_leaf_count") == EXPECTED_OPTIMIZER_LEAVES,
        "receipt_no_prune_or_delete": receipt.get("save_step") == EXPECTED_STEP
        and receipt.get("automatic_pruning_enabled") is False
        and receipt.get("old_checkpoint_deletion_performed") is False,
        "adapter_manifest_exact_entries": len(entries) == EXPECTED_GOLDEN_LEAVES
        and len({entry.get("path") for entry in entries if isinstance(entry, dict)}) == EXPECTED_GOLDEN_LEAVES
        and adapter_files == declared_adapter_files,
        "adapter_manifest_identity_valid": _adapter_identity(adapter_manifest)
        == adapter_manifest.get("adapter_identity_sha256")
        == result.get("adapter_identity_sha256")
        == receipt.get("adapter_identity_sha256"),
        "adapter_manifest_training_identity": adapter_manifest.get("train_step") == EXPECTED_STEP
        and adapter_manifest.get("train_seed") == EXPECTED_SEED,
        "adapter_file_hashes_valid": adapter_file_hashes_valid,
        "no_partial_or_temporary_artifacts": not _has_partial_or_temporary_path(checkpoint_dir)
        and not _has_partial_or_temporary_path(adapter_root),
        "gpu_guard_selected_preflight_gpu": bool(gpu_events)
        and gpu_events[0].get("event") == "guard_started"
        and gpu_events[0].get("physical_gpu") == selected_gpu,
        "gpu_guard_child_exit_zero": bool(gpu_events)
        and gpu_events[-1].get("event") == "child_exited"
        and gpu_events[-1].get("return_code") == 0,
        "gpu_guard_no_monitor_or_memory_emergency": not any(
            event.get("event") in {"monitor_error", "memory_emergency"} for event in gpu_events
        ),
        "storage_guard_completed_and_reaped": storage_status.get("reason_code") == "completed"
        and storage_status.get("child_returncode") == 0
        and storage_status.get("wait_reaped") is True
        and storage_status.get("term_sent") is False
        and storage_status.get("kill_sent") is False
        and storage_status.get("external_signal") is None,
        "storage_guard_new_session": storage_manifest.get("start_new_session") is True
        and storage_manifest.get("child_pid") == storage_manifest.get("expected_child_pgid"),
        "storage_below_soft_and_hard_limits": storage_manifest.get("storage", {}).get("soft_limit_bytes")
        == SOFT_LIMIT_BYTES
        and storage_manifest.get("storage", {}).get("hard_limit_bytes") == HARD_LIMIT_BYTES
        and int(storage_status.get("final_billed_lora_bytes", HARD_LIMIT_BYTES)) < SOFT_LIMIT_BYTES,
        "storage_increment_within_bound": int(final_sample.get("positive_stage_delta_bytes", -1)) >= 0
        and int(final_sample.get("positive_stage_delta_bytes", args.max_stage_increment_bytes + 1))
        <= args.max_stage_increment_bytes,
        "shared_caches_unchanged": all(root_deltas.get(root) == 0 for root in UNCHANGED_CACHE_ROOTS),
        "openpi_worktree_clean_before": not (attempt / "static/git-status-before.txt").read_text(encoding="utf-8").strip(),
        "openpi_worktree_clean_after": not (attempt / "static/git-status-after.txt").read_text(encoding="utf-8").strip(),
        "no_task_gpu_process_after": not (attempt / "static/nvidia-compute-apps-after.txt").read_text(encoding="utf-8").strip(),
    }

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"S1d acceptance failed: {failed}")

    loss_values = [float(record["loss"]) for record in result["metrics_trace"]]
    report = {
        "schema_version": 1,
        "stage": "S1d",
        "status": "pass",
        "checks": checks,
        "preflight_identity_sha256": claimed_preflight_identity,
        "selected_physical_gpu": selected_gpu,
        "selected_gpu_uuid": selected_uuid,
        "result_sha256": _sha256(attempt / "run/result.json"),
        "checkpoint_artifact": checkpoint_tree,
        "adapter_artifact": adapter_tree,
        "adapter_identity_sha256": result["adapter_identity_sha256"],
        "guard_summary": {
            "runtime_samples": len(gpu_samples),
            "max_utilization_percent": max(sample["utilization_percent"] for sample in gpu_samples),
            "minimum_free_memory_percent": min(sample["free_memory_percent"] for sample in gpu_samples),
            "pause_events": sum(sample.get("action") == "paused" for sample in gpu_samples),
            "resume_events": sum(sample.get("action") == "resumed" for sample in gpu_samples),
        },
        "storage_summary": {
            "sample_count": storage_status["sample_count"],
            "final_billed_lora_bytes": storage_status["final_billed_lora_bytes"],
            "positive_stage_delta_bytes": final_sample["positive_stage_delta_bytes"],
            "soft_headroom_bytes": final_sample["soft_headroom_bytes"],
            "hard_headroom_bytes": final_sample["hard_headroom_bytes"],
            "root_deltas_bytes": root_deltas,
        },
        "result_summary": {
            "elapsed_seconds": result["elapsed_seconds"],
            "training_steps_executed": result["training_steps_executed"],
            "loss_first": loss_values[0],
            "loss_last": loss_values[-1],
            "loss_min": min(loss_values),
            "loss_max": max(loss_values),
            "changed_golden_leaf_count": result["changed_golden_leaf_count"],
            "changed_non_golden_leaf_count": result["changed_non_golden_leaf_count"],
            "checkpoint_total_bytes": result["checkpoint_total_bytes"],
            "adapter_total_bytes": result["adapter_total_bytes"],
        },
    }
    report["report_identity_sha256"] = experiment_identity.canonical_sha256(report)
    experiment_identity.atomic_write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
