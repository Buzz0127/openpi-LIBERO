#!/usr/bin/env python3
"""Audit a pinned LeRobot LIBERO dataset revision without downloading samples."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_url


REPO_ID = "physical-intelligence/libero"
REPO_TYPE = "dataset"
ALLOWED_METADATA = (
    "meta/episodes.jsonl",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
)
MAX_METADATA_BYTES = 1 << 20
GIB = 1 << 30
LORA_HARD_LIMIT_BYTES = 100 * GIB
LORA_SOFT_STOP_BYTES = 95 * GIB
MIN_UNCOMMITTED_RESERVE_BYTES = 20 * GIB
# S0 measured only a few MiB. D1a deliberately rounds existing LoRA-owned
# additions up to 1 GiB so the D1b peak decision does not depend on a fragile
# apparent-size versus allocated-size comparison.
CURRENT_LORA_BILLED_UPPER_BOUND_BYTES = 1 * GIB
EXPECTED_STATS_KEYS = {
    "actions",
    "episode_index",
    "frame_index",
    "image",
    "index",
    "state",
    "task_index",
    "timestamp",
    "wrist_image",
}


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fetch_small_file(path: str, revision: str, timeout_s: int) -> bytes:
    if path not in ALLOWED_METADATA:
        raise ValueError(f"refusing non-metadata path: {path}")
    url = hf_hub_url(REPO_ID, path, repo_type=REPO_TYPE, revision=revision)
    with requests.get(url, stream=True, timeout=(10, timeout_s)) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared is not None and int(declared) > MAX_METADATA_BYTES:
            raise RuntimeError(f"metadata Content-Length exceeds limit: {path}: {declared}")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > MAX_METADATA_BYTES:
                raise RuntimeError(f"metadata body exceeds limit: {path}: >{MAX_METADATA_BYTES}")
            chunks.append(chunk)
    return b"".join(chunks)


def _file_record(sibling: Any) -> dict[str, Any]:
    raw = _jsonable(sibling)
    path = getattr(sibling, "rfilename", None) or raw.get("rfilename") or raw.get("path")
    size = getattr(sibling, "size", None)
    if size is None:
        size = raw.get("size")
    if path is None or size is None:
        raise RuntimeError(f"file metadata lacks path or exact size: {raw}")
    return {
        "path": str(path),
        "size": int(size),
        "blob_id": raw.get("blob_id") or raw.get("blobId"),
        "lfs": raw.get("lfs"),
    }


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def audit(
    revision: str,
    output_dir: pathlib.Path,
    timeout_s: int,
    offline_input_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir()

    if offline_input_dir is None:
        api = HfApi()
        info = api.repo_info(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            files_metadata=True,
            timeout=timeout_s,
        )
        resolved_revision = str(info.sha)
        siblings = info.siblings
        acquisition_mode = "online_huggingface_hub"
    else:
        repo_info_path = offline_input_dir / "repo_info.json"
        raw_repo_info = json.loads(repo_info_path.read_text())
        resolved_revision = str(raw_repo_info["sha"])
        siblings = raw_repo_info["siblings"]
        acquisition_mode = "offline_verified_inputs"
    if resolved_revision != revision:
        raise RuntimeError(f"revision mismatch: requested={revision}, resolved={resolved_revision}")

    files = sorted((_file_record(item) for item in siblings), key=lambda item: item["path"])
    manifest_path = output_dir / "repo_manifest.jsonl"
    manifest_lines = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in files]
    manifest_payload = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_payload)

    payloads: dict[str, bytes] = {}
    metadata_hashes: dict[str, dict[str, Any]] = {}
    for path in ALLOWED_METADATA:
        if offline_input_dir is None:
            payload = _fetch_small_file(path, revision, timeout_s)
        else:
            source = offline_input_dir / pathlib.PurePosixPath(path).name
            payload = source.read_bytes()
            if len(payload) > MAX_METADATA_BYTES:
                raise RuntimeError(f"offline metadata body exceeds limit: {source}: {len(payload)}")
        payloads[path] = payload
        target = metadata_dir / pathlib.PurePosixPath(path).name
        target.write_bytes(payload)
        metadata_hashes[path] = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}

    dataset_info = json.loads(payloads["meta/info.json"])
    episodes = _parse_jsonl(payloads["meta/episodes.jsonl"])
    tasks = _parse_jsonl(payloads["meta/tasks.jsonl"])
    dataset_stats = json.loads(payloads["meta/stats.json"])

    expected_episode_paths = {
        f"data/chunk-{index // int(dataset_info['chunks_size']):03d}/episode_{index:06d}.parquet"
        for index in range(int(dataset_info["total_episodes"]))
    }
    actual_episode_paths = {item["path"] for item in files if item["path"].endswith(".parquet")}
    repo_paths = {item["path"] for item in files}
    missing_metadata = sorted(set(ALLOWED_METADATA) - repo_paths)
    missing_episodes = sorted(expected_episode_paths - actual_episode_paths)
    unexpected_parquet = sorted(actual_episode_paths - expected_episode_paths)

    episode_indices = [int(row["episode_index"]) for row in episodes]
    task_indices = [int(row["task_index"]) for row in tasks]
    total_episode_frames = sum(int(row["length"]) for row in episodes)
    episode_task_strings = {task for row in episodes for task in row["tasks"]}

    checks = {
        "revision_exact": resolved_revision == revision,
        "metadata_allowlist_complete": not missing_metadata,
        "episode_count_matches_info": len(episodes) == int(dataset_info["total_episodes"]),
        "episode_indices_contiguous": episode_indices == list(range(len(episodes))),
        "episode_frames_match_info": total_episode_frames == int(dataset_info["total_frames"]),
        "episode_task_count_matches_info": len(episode_task_strings) == int(dataset_info["total_tasks"]),
        "task_count_matches_info": len(tasks) == int(dataset_info["total_tasks"]),
        "task_indices_contiguous": task_indices == list(range(len(tasks))),
        "task_strings_unique": len({row["task"] for row in tasks}) == len(tasks),
        "parquet_manifest_exact": not missing_episodes and not unexpected_parquet,
        "stats_keys_expected": set(dataset_stats) == EXPECTED_STATS_KEYS,
        "no_videos": int(dataset_info["total_videos"]) == 0,
        "fps_is_10": int(dataset_info["fps"]) == 10,
    }

    total_bytes = sum(item["size"] for item in files)
    parquet_bytes = sum(item["size"] for item in files if item["path"].endswith(".parquet"))
    dataset_download_peak_bytes = 2 * total_bytes
    d1b_committed_peak_bytes = CURRENT_LORA_BILLED_UPPER_BOUND_BYTES + dataset_download_peak_bytes
    d1b_remaining_hard_bytes = LORA_HARD_LIMIT_BYTES - d1b_committed_peak_bytes
    d1b_soft_headroom_bytes = LORA_SOFT_STOP_BYTES - d1b_committed_peak_bytes
    budget_checks = {
        "below_95_gib_soft_stop": d1b_committed_peak_bytes < LORA_SOFT_STOP_BYTES,
        "at_least_20_gib_uncommitted_reserve": d1b_remaining_hard_bytes >= MIN_UNCOMMITTED_RESERVE_BYTES,
    }
    checks.update({f"budget_{key}": value for key, value in budget_checks.items()})
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "acquisition_mode": acquisition_mode,
        "checks": checks,
        "identity": {
            "file_count": len(files),
            "repo_total_bytes": total_bytes,
            "parquet_file_count": len(actual_episode_paths),
            "parquet_total_bytes": parquet_bytes,
            "metadata_file_count": len(ALLOWED_METADATA),
            "repo_manifest_sha256": _sha256_bytes(manifest_payload),
            "metadata": metadata_hashes,
        },
        "d1b_budget": {
            "policy_hard_limit_bytes": LORA_HARD_LIMIT_BYTES,
            "policy_soft_stop_bytes": LORA_SOFT_STOP_BYTES,
            "policy_min_uncommitted_reserve_bytes": MIN_UNCOMMITTED_RESERVE_BYTES,
            "current_lora_billed_upper_bound_bytes": CURRENT_LORA_BILLED_UPPER_BOUND_BYTES,
            "download_peak_model": "current upper bound + two complete repository copies (partial plus final)",
            "dataset_download_peak_bytes": dataset_download_peak_bytes,
            "committed_peak_bytes": d1b_committed_peak_bytes,
            "remaining_hard_bytes": d1b_remaining_hard_bytes,
            "soft_stop_headroom_bytes": d1b_soft_headroom_bytes,
            "checks": budget_checks,
            "authorization": "not authorized; budget feasibility only",
        },
        "schema": {
            "codebase_version": dataset_info["codebase_version"],
            "robot_type": dataset_info["robot_type"],
            "total_episodes": dataset_info["total_episodes"],
            "total_frames": dataset_info["total_frames"],
            "total_tasks": dataset_info["total_tasks"],
            "total_chunks": dataset_info["total_chunks"],
            "chunks_size": dataset_info["chunks_size"],
            "fps": dataset_info["fps"],
            "splits": dataset_info["splits"],
            "data_path": dataset_info["data_path"],
            "features": dataset_info["features"],
            "episode_min_length": min(int(row["length"]) for row in episodes),
            "episode_max_length": max(int(row["length"]) for row in episodes),
        },
        "mismatches": {
            "missing_metadata": missing_metadata,
            "missing_episodes": missing_episodes,
            "unexpected_parquet": unexpected_parquet,
        },
        "notes": [
            "Only the four allowlisted metadata files were downloaded; no parquet, video, model, or checkpoint was fetched.",
            "Offline mode consumes a separately hashed Hugging Face repo-info response and four fixed-revision metadata files.",
            "meta/stats.json is upstream LeRobot dataset metadata, not the canonical OpenPI LIBERO normalization asset.",
        ],
    }
    (output_dir / "d1a_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--offline-input-dir", type=pathlib.Path)
    args = parser.parse_args()
    report = audit(args.revision, args.output_dir, args.timeout_seconds, args.offline_input_dir)
    print(json.dumps({"status": report["status"], **report["identity"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
