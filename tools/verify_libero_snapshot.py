#!/usr/bin/env python3
"""Verify a local LIBERO snapshot against the pinned D1a repository manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = str(record["path"])
        if name in records:
            raise RuntimeError(f"duplicate manifest path: {name}")
        records[name] = record
    return records


def verify(
    dataset_root: pathlib.Path,
    repo_manifest: pathlib.Path,
    d1a_report: pathlib.Path,
    output: pathlib.Path,
    allow_partial: bool,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    root = dataset_root.resolve(strict=True)
    expected = _load_manifest(repo_manifest)
    d1a = json.loads(d1a_report.read_text())
    if d1a["status"] != "pass" or d1a["resolved_revision"] != d1a["requested_revision"]:
        raise RuntimeError("D1a report is not a passing pinned-revision report")

    actual: dict[str, dict[str, Any]] = {}
    cache_artifacts: list[str] = []
    completed_zero_byte_locks: list[str] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = pathlib.Path(directory)
        relative_directory = directory_path.relative_to(root)
        if relative_directory == pathlib.Path(".") and ".cache" in dirnames:
            dirnames.remove(".cache")
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            stat = path.lstat()
            actual[relative] = {"size": stat.st_size, "is_symlink": path.is_symlink()}

    expected_paths = set(expected)
    actual_paths = set(actual)
    cache_root = root / ".cache"
    download_cache_root = cache_root / "huggingface" / "download"
    if cache_root.exists():
        for path in cache_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.name.endswith(".lock"):
                try:
                    locked_repo_path = path.relative_to(download_cache_root).as_posix()[: -len(".lock")]
                except ValueError:
                    cache_artifacts.append(relative)
                    continue
                matching_complete_file = (
                    path.stat().st_size == 0
                    and locked_repo_path in expected_paths
                    and locked_repo_path in actual_paths
                    and not actual[locked_repo_path]["is_symlink"]
                    and int(actual[locked_repo_path]["size"]) == int(expected[locked_repo_path]["size"])
                )
                if matching_complete_file:
                    completed_zero_byte_locks.append(relative)
                else:
                    cache_artifacts.append(relative)
            elif (
                path.name.endswith(".incomplete")
                or path.name.endswith(".partial")
                or path.name.endswith(".tmp")
                or path.name.endswith(".temp")
                or ".tmp" in path.name
            ):
                cache_artifacts.append(relative)

    cache_artifacts.sort()
    completed_zero_byte_locks.sort()
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    size_mismatches = [
        {"path": path, "expected": int(expected[path]["size"]), "actual": int(actual[path]["size"])}
        for path in sorted(expected_paths & actual_paths)
        if int(expected[path]["size"]) != int(actual[path]["size"])
    ]
    symlinks = sorted(path for path, record in actual.items() if record["is_symlink"])

    metadata_checks: dict[str, Any] = {}
    for repo_path, identity in d1a["identity"]["metadata"].items():
        local_path = root / repo_path
        metadata_checks[repo_path] = {
            "present": local_path.is_file(),
            "sha256": _sha256(local_path) if local_path.is_file() else None,
            "expected_sha256": identity["sha256"],
        }
        metadata_checks[repo_path]["matches"] = (
            metadata_checks[repo_path]["sha256"] == identity["sha256"]
        )

    checks = {
        "no_unexpected_repo_files": not unexpected,
        "completed_file_sizes_match": not size_mismatches,
        "no_symlinks": not symlinks,
        "all_expected_files_present": not missing,
        "no_incomplete_cache_artifacts": not cache_artifacts,
        "metadata_hashes_match_when_present": all(
            (not value["present"]) or value["matches"] for value in metadata_checks.values()
        ),
    }
    required = [
        checks["no_unexpected_repo_files"],
        checks["completed_file_sizes_match"],
        checks["no_symlinks"],
        checks["metadata_hashes_match_when_present"],
    ]
    if not allow_partial:
        required.extend([checks["all_expected_files_present"], checks["no_incomplete_cache_artifacts"]])

    report = {
        "status": "pass" if all(required) else "fail",
        "mode": "partial" if allow_partial else "complete",
        "dataset_root": str(root),
        "revision": d1a["resolved_revision"],
        "expected_file_count": len(expected),
        "actual_repo_file_count": len(actual),
        "expected_total_bytes": sum(int(record["size"]) for record in expected.values()),
        "actual_repo_bytes": sum(int(record["size"]) for record in actual.values()),
        "missing_count": len(missing),
        "unexpected": unexpected,
        "size_mismatches": size_mismatches,
        "symlinks": symlinks,
        "cache_artifact_count": len(cache_artifacts),
        "cache_artifacts": cache_artifacts,
        "completed_zero_byte_lock_count": len(completed_zero_byte_locks),
        "completed_zero_byte_lock_paths_sha256": hashlib.sha256(
            "".join(f"{path}\n" for path in completed_zero_byte_locks).encode()
        ).hexdigest(),
        "metadata": metadata_checks,
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--repo-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--d1a-report", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = verify(args.dataset_root, args.repo_manifest, args.d1a_report, args.output, args.allow_partial)
    print(json.dumps({key: report[key] for key in ("status", "mode", "actual_repo_file_count", "actual_repo_bytes", "missing_count", "cache_artifact_count")}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
