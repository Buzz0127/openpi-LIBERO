#!/usr/bin/env python3
"""Run the bounded N1b normalization stage to completion without an interactive client."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_numbers(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_flatten_numbers(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"non-finite numeric value at {prefix}")
        result[prefix] = number
    return result


def compare_stats(canonical: Path, official: Path, expected_official_sha: str) -> dict[str, Any]:
    official_sha = sha256_file(official)
    if official_sha != expected_official_sha:
        raise RuntimeError(f"official stats identity changed: {official_sha}")
    canonical_sha = sha256_file(canonical)
    canonical_json = json.loads(canonical.read_text())
    official_json = json.loads(official.read_text())
    canonical_numbers = _flatten_numbers(canonical_json)
    official_numbers = _flatten_numbers(official_json)
    canonical_paths = set(canonical_numbers)
    official_paths = set(official_numbers)
    shared = sorted(canonical_paths & official_paths)
    differences = [abs(canonical_numbers[path] - official_numbers[path]) for path in shared]
    return {
        "status": "pass",
        "canonical_path": str(canonical),
        "canonical_sha256": canonical_sha,
        "canonical_bytes": canonical.stat().st_size,
        "official_path": str(official),
        "official_sha256": official_sha,
        "official_bytes": official.stat().st_size,
        "file_sha256_equal": canonical_sha == official_sha,
        "canonical_numeric_leaf_count": len(canonical_numbers),
        "official_numeric_leaf_count": len(official_numbers),
        "missing_from_canonical": sorted(official_paths - canonical_paths),
        "extra_in_canonical": sorted(canonical_paths - official_paths),
        "all_numeric_equal": canonical_paths == official_paths
        and all(difference == 0.0 for difference in differences),
        "max_abs_difference": max(differences, default=0.0),
        "mean_abs_difference": sum(differences) / len(differences) if differences else 0.0,
        "same_normalization_values": canonical_paths == official_paths
        and all(difference == 0.0 for difference in differences),
        "interpretation": (
            "A mismatch does not invalidate the controlled Base versus pure-LoRA comparison; "
            "it makes pi0_libero an external end-to-end reference with checkpoint-owned stats."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--compute-script", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--canonical-target", required=True, type=Path)
    parser.add_argument("--official-stats", required=True, type=Path)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument("--identity", action="append", required=True)
    parser.add_argument("--monitor-root", action="append", required=True, type=Path)
    parser.add_argument("--expected-dataset-length", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    return parser.parse_args()


def _offline_overrides() -> dict[str, str]:
    return {
            "HF_HOME": "/home/wengzr/projects/openpi-lora-cache/huggingface",
            "HF_DATASETS_CACHE": "/home/wengzr/projects/openpi-lora-cache/huggingface/datasets",
            "HF_LEROBOT_HOME": "/home/wengzr/projects/openpi-lora-cache/huggingface/lerobot",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora/src",
        }


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    environment.update(_offline_overrides())
    return environment


def main() -> int:
    args = _parse_args()
    attempt = args.attempt_dir.resolve()
    if not attempt.is_dir():
        raise FileNotFoundError(f"attempt directory does not exist: {attempt}")
    if args.canonical_target.exists():
        raise FileExistsError(f"canonical target already exists: {args.canonical_target}")
    if sha256_file(args.guard) != args.guard_sha256:
        raise RuntimeError("storage guard SHA-256 mismatch")
    if sha256_file(args.wrapper) != args.wrapper_sha256:
        raise RuntimeError("atomic wrapper SHA-256 mismatch")

    start = time.time()
    status_path = attempt / "automation_status.json"
    guard_run = attempt / "guard-run"
    wrapper_report = guard_run / "n1b_report.json"
    child_command = [
        str(args.python),
        str(args.wrapper),
        "--compute-script",
        str(args.compute_script),
        "--config-name",
        args.config_name,
        "--expected-target",
        str(args.canonical_target),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--expected-dataset-length",
        str(args.expected_dataset_length),
        "--report",
        str(wrapper_report),
    ]
    for identity in args.identity:
        child_command.extend(["--identity", identity])
    guard_command = [
        str(args.python),
        str(args.guard),
        "--attempt-dir",
        str(guard_run),
    ]
    for root in args.monitor_root:
        guard_command.extend(["--monitor-root", str(root)])
    guard_command.extend(
        [
            "--existing-billed-bytes",
            "0",
            "--soft-limit-bytes",
            "500000000",
            "--hard-limit-bytes",
            "1000000000",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--sample-interval-seconds",
            "5",
            "--near-soft-margin-bytes",
            "100000000",
            "--near-sample-interval-seconds",
            "1",
            "--min-mem-available-bytes",
            str(64 * (1 << 30)),
            "--max-child-rss-bytes",
            str(8 * (1 << 30)),
            "--max-load1-per-cpu",
            "0.9",
            "--resource-consecutive-samples",
            "3",
            "--monitor-failure-consecutive-samples",
            "2",
            "--",
            *child_command,
        ]
    )
    atomic_json(
        attempt / "automation_manifest.json",
        {
            "status": "started",
            "start_time_epoch_seconds": start,
            "guard_command": guard_command,
            "environment_overrides": _offline_overrides(),
            "explicitly_unset_environment": [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ],
            "source_drop_last_remainder_frames": args.expected_dataset_length % args.batch_size,
            "expected_processed_frames":
                (args.expected_dataset_length // args.batch_size) * args.batch_size,
        },
    )
    try:
        result = subprocess.run(guard_command, env=_offline_environment(), check=False)
        if result.returncode != 0:
            raise RuntimeError(f"guard returned {result.returncode}")
        if not wrapper_report.is_file():
            raise RuntimeError("guard succeeded without N1b report")
        wrapper_result = json.loads(wrapper_report.read_text())
        if wrapper_result.get("status") != "pass":
            raise RuntimeError("N1b wrapper report is not pass")
        comparison = compare_stats(
            args.canonical_target / "norm_stats.json", args.official_stats, args.official_sha256
        )
        atomic_json(attempt / "comparison_report.json", comparison)
        atomic_json(
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
        atomic_json(
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
