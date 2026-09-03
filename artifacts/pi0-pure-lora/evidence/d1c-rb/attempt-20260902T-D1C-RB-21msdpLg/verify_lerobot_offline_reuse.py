#!/usr/bin/env python3
"""Prove one pinned LIBERO snapshot and one retained Arrow cache are reused offline.

The real LeRobot import is intentionally deferred until after all static identity
and environment checks pass. Tests can inject a fake dataset factory without
importing LeRobot or touching the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from collections.abc import Callable, Mapping
from typing import Any


REPO_ID = "physical-intelligence/libero"
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _manifest_sha256(rows: list[str]) -> str:
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def _scan_tree(root: pathlib.Path) -> dict[str, Any]:
    """Return a metadata identity for a tree without following symlinks."""
    files: dict[str, dict[str, Any]] = {}
    if root.exists():
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            directory_path = pathlib.Path(directory)
            # os.walk lists symlinked directories in dirnames even when it does
            # not descend into them. Record them so symlinks cannot hide.
            for dirname in list(dirnames):
                path = directory_path / dirname
                if path.is_symlink():
                    stat = path.lstat()
                    relative = path.relative_to(root).as_posix()
                    files[relative] = {
                        "kind": "symlink",
                        "size": stat.st_size,
                        "allocated": stat.st_blocks * 512,
                        "mtime_ns": stat.st_mtime_ns,
                        "nlink": stat.st_nlink,
                        "target": os.readlink(path),
                    }
                    dirnames.remove(dirname)
            for filename in filenames:
                path = directory_path / filename
                stat = path.lstat()
                relative = path.relative_to(root).as_posix()
                files[relative] = {
                    "kind": "symlink" if path.is_symlink() else "file",
                    "size": stat.st_size,
                    "allocated": stat.st_blocks * 512,
                    "mtime_ns": stat.st_mtime_ns,
                    "nlink": stat.st_nlink,
                    "target": os.readlink(path) if path.is_symlink() else None,
                }
    rows = [
        f"{name}\0{record['kind']}\0{record['size']}\0{record['allocated']}\0"
        f"{record['mtime_ns']}\0{record['nlink']}\0{record['target'] or ''}\n"
        for name, record in sorted(files.items())
    ]
    return {
        "root": str(root.resolve(strict=False)),
        "files": files,
        "summary": {
            "file_entry_count": len(files),
            "apparent_bytes": sum(int(record["size"]) for record in files.values()),
            "allocated_bytes": sum(int(record["allocated"]) for record in files.values()),
            "symlink_count": sum(record["kind"] == "symlink" for record in files.values()),
            "multi_link_file_count": sum(
                record["kind"] == "file" and int(record["nlink"]) != 1
                for record in files.values()
            ),
            "metadata_manifest_sha256": _manifest_sha256(rows),
        },
    }


def _tree_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = before["files"]
    after_files = after["files"]
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = sorted(
        name for name in set(before_files) & set(after_files) if before_files[name] != after_files[name]
    )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "added_apparent_bytes": sum(int(after_files[name]["size"]) for name in added),
        "added_allocated_bytes": sum(int(after_files[name]["allocated"]) for name in added),
        "max_added_file_bytes": max((int(after_files[name]["size"]) for name in added), default=0),
    }


def _repo_roots(hf_lerobot_home: pathlib.Path) -> list[str]:
    matches: list[str] = []
    if hf_lerobot_home.exists():
        for info in hf_lerobot_home.rglob("meta/info.json"):
            candidate = info.parent.parent
            if candidate.name == "libero" and candidate.is_dir():
                matches.append(str(candidate.resolve()))
    return sorted(set(matches))


def _arrow_tree_roots(hf_datasets_cache: pathlib.Path) -> list[pathlib.Path]:
    parents: set[pathlib.Path] = set()
    if hf_datasets_cache.exists():
        for path in hf_datasets_cache.rglob("*.arrow"):
            if path.is_file() and not path.is_symlink():
                parents.add(path.parent.resolve())
    return sorted(parents, key=str)


def _arrow_cache_summary(
    hf_datasets_cache: pathlib.Path, arrow_root: pathlib.Path
) -> dict[str, Any]:
    cache = hf_datasets_cache.resolve(strict=True)
    root = arrow_root.resolve(strict=True)
    try:
        relative_root = root.relative_to(cache)
    except ValueError as error:
        raise ValueError(f"Arrow root {root} is outside HF datasets cache {cache}") from error
    parts = relative_root.parts
    if len(parts) < 4:
        raise ValueError(f"Arrow root does not expose builder/config/version/fingerprint: {root}")
    records: list[tuple[str, int]] = []
    symlinks: list[str] = []
    hardlinks: list[str] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = pathlib.Path(directory)
        for dirname in list(dirnames):
            path = directory_path / dirname
            if path.is_symlink():
                symlinks.append(path.relative_to(root).as_posix())
                dirnames.remove(dirname)
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
            else:
                stat = path.lstat()
                records.append((relative, stat.st_size))
                if stat.st_nlink != 1:
                    hardlinks.append(relative)
    records.sort()
    rows = [f"{name}\0{size}\n" for name, size in records]
    return {
        "root": str(root),
        "relative_root": relative_root.as_posix(),
        "builder": parts[-4],
        "config": parts[-3],
        "version": parts[-2],
        "fingerprint": parts[-1],
        "file_count": len(records),
        "arrow_file_count": sum(name.endswith(".arrow") for name, _ in records),
        "apparent_bytes": sum(size for _, size in records),
        "symlinks": sorted(symlinks),
        "hardlinks": sorted(hardlinks),
        "path_size_manifest_sha256": _manifest_sha256(rows),
    }


def _load_raw_snapshot_identity(
    report_path: pathlib.Path, dataset_root: pathlib.Path, revision: str
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    selected = {
        "status": report.get("status"),
        "mode": report.get("mode"),
        "dataset_root": report.get("dataset_root"),
        "revision": report.get("revision"),
        "expected_file_count": report.get("expected_file_count"),
        "actual_repo_file_count": report.get("actual_repo_file_count"),
        "expected_total_bytes": report.get("expected_total_bytes"),
        "actual_repo_bytes": report.get("actual_repo_bytes"),
        "missing_count": report.get("missing_count"),
        "unexpected": report.get("unexpected"),
        "size_mismatches": report.get("size_mismatches"),
        "symlinks": report.get("symlinks"),
        "cache_artifact_count": report.get("cache_artifact_count"),
        "metadata": report.get("metadata"),
    }
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    selected["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    selected["checks"] = {
        "report_passes_complete_mode": selected["status"] == "pass" and selected["mode"] == "complete",
        "dataset_root_exact": pathlib.Path(str(selected["dataset_root"])).resolve(strict=False)
        == dataset_root,
        "revision_exact": selected["revision"] == revision,
        "file_counts_match": selected["expected_file_count"] == selected["actual_repo_file_count"],
        "byte_counts_match": selected["expected_total_bytes"] == selected["actual_repo_bytes"],
        "positive_file_and_byte_counts": isinstance(selected["actual_repo_file_count"], int)
        and selected["actual_repo_file_count"] > 0
        and isinstance(selected["actual_repo_bytes"], int)
        and selected["actual_repo_bytes"] > 0,
        "no_missing_or_unexpected": selected["missing_count"] == 0 and selected["unexpected"] == [],
        "no_size_mismatches_or_symlinks": selected["size_mismatches"] == [] and selected["symlinks"] == [],
        "no_incomplete_cache_artifacts": selected["cache_artifact_count"] == 0,
        "metadata_hashes_match": isinstance(selected["metadata"], dict)
        and bool(selected["metadata"])
        and all(value.get("matches") is True for value in selected["metadata"].values()),
    }
    return selected


def _environment_checks(
    environment: Mapping[str, str],
    hf_home: pathlib.Path,
    hf_datasets_cache: pathlib.Path,
    hf_lerobot_home: pathlib.Path,
) -> dict[str, bool]:
    def exact(name: str, expected: pathlib.Path) -> bool:
        value = environment.get(name)
        return value is not None and pathlib.Path(value).resolve(strict=False) == expected

    return {
        "hf_home_exact": exact("HF_HOME", hf_home),
        "hf_datasets_cache_exact": exact("HF_DATASETS_CACHE", hf_datasets_cache),
        "hf_lerobot_home_exact": exact("HF_LEROBOT_HOME", hf_lerobot_home),
        "hf_hub_offline": environment.get("HF_HUB_OFFLINE") == "1",
        "hf_datasets_offline": environment.get("HF_DATASETS_OFFLINE") == "1",
        "transformers_offline": environment.get("TRANSFORMERS_OFFLINE") == "1",
        "all_proxy_variables_unset": all(not environment.get(name) for name in PROXY_VARIABLES),
    }


def verify(
    *,
    dataset_root: pathlib.Path,
    hf_home: pathlib.Path,
    hf_datasets_cache: pathlib.Path,
    hf_lerobot_home: pathlib.Path,
    raw_snapshot_report: pathlib.Path,
    revision: str,
    expected_arrow_root: pathlib.Path,
    expected_arrow_builder: str,
    expected_arrow_config: str,
    expected_arrow_version: str,
    expected_arrow_fingerprint: str,
    expected_arrow_file_count: int,
    expected_arrow_manifest_sha256: str,
    max_new_file_bytes: int,
    output: pathlib.Path,
    dataset_factory: Callable[..., Any],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    root = dataset_root.resolve(strict=True)
    home = hf_home.resolve(strict=True)
    datasets_cache = hf_datasets_cache.resolve(strict=True)
    lerobot_home = hf_lerobot_home.resolve(strict=True)
    arrow_root = expected_arrow_root.resolve(strict=True)
    if max_new_file_bytes < 0:
        raise ValueError("max_new_file_bytes must be non-negative")

    raw_identity = _load_raw_snapshot_identity(raw_snapshot_report, root, revision)
    env_checks = _environment_checks(
        os.environ if environment is None else environment, home, datasets_cache, lerobot_home
    )
    roots_before = _repo_roots(lerobot_home)
    arrow_roots_before = [str(path) for path in _arrow_tree_roots(datasets_cache)]
    arrow_before = _arrow_cache_summary(datasets_cache, arrow_root)
    hf_before = _scan_tree(home)

    static_checks = {
        **{f"raw_{key}": value for key, value in raw_identity["checks"].items()},
        **{f"env_{key}": value for key, value in env_checks.items()},
        "datasets_cache_is_inside_hf_home": home == datasets_cache or home in datasets_cache.parents,
        "lerobot_home_is_inside_hf_home": home == lerobot_home or home in lerobot_home.parents,
        "dataset_root_is_inside_lerobot_home": lerobot_home == root or lerobot_home in root.parents,
        "single_raw_repo_root_before": roots_before == [str(root)],
        "single_arrow_tree_before": arrow_roots_before == [str(arrow_root)],
        "arrow_builder_exact": arrow_before["builder"] == expected_arrow_builder,
        "arrow_config_exact": arrow_before["config"] == expected_arrow_config,
        "arrow_version_exact": arrow_before["version"] == expected_arrow_version,
        "arrow_fingerprint_exact": arrow_before["fingerprint"] == expected_arrow_fingerprint,
        "arrow_file_count_exact": arrow_before["arrow_file_count"] == expected_arrow_file_count,
        "arrow_manifest_exact": arrow_before["path_size_manifest_sha256"]
        == expected_arrow_manifest_sha256,
        "arrow_has_no_symlinks_before": not arrow_before["symlinks"],
        "arrow_has_no_hardlinks_before": not arrow_before["hardlinks"],
        "hf_home_has_no_symlinks_before": hf_before["summary"]["symlink_count"] == 0,
        "hf_home_has_no_hardlinks_before": hf_before["summary"]["multi_link_file_count"] == 0,
    }
    if not all(static_checks.values()):
        report = {
            "status": "fail",
            "phase": "preflight",
            "repo_id": REPO_ID,
            "revision": revision,
            "dataset_root": str(root),
            "hf_home": str(home),
            "hf_datasets_cache": str(datasets_cache),
            "hf_lerobot_home": str(lerobot_home),
            "raw_snapshot_identity": raw_identity,
            "arrow_before": arrow_before,
            "raw_repo_roots_before": roots_before,
            "arrow_tree_roots_before": arrow_roots_before,
            "hf_home_before": hf_before["summary"],
            "checks": static_checks,
        }
        _write_report(output, report)
        return report

    dataset = dataset_factory(
        REPO_ID,
        root=root,
        revision=revision,
        force_cache_sync=False,
        download_videos=False,
    )

    hf_after = _scan_tree(home)
    tree_diff = _tree_diff(hf_before, hf_after)
    roots_after = _repo_roots(lerobot_home)
    arrow_roots_after = [str(path) for path in _arrow_tree_roots(datasets_cache)]
    arrow_after = _arrow_cache_summary(datasets_cache, arrow_root)
    metadata: dict[str, Any] = {
        "total_episodes": getattr(dataset.meta, "total_episodes", None),
        "total_frames": getattr(dataset.meta, "total_frames", None),
        "total_tasks": getattr(dataset.meta, "total_tasks", None),
        "fps": getattr(dataset.meta, "fps", None),
    }
    runtime_checks = {
        "dataset_repo_id_exact": dataset.repo_id == REPO_ID,
        "dataset_revision_exact": dataset.revision == revision,
        "dataset_root_exact": pathlib.Path(dataset.root).resolve(strict=False) == root,
        "length_is_273465": len(dataset) == 273465,
        "total_episodes_is_1693": metadata["total_episodes"] == 1693,
        "total_frames_is_273465": metadata["total_frames"] == 273465,
        "total_tasks_is_40": metadata["total_tasks"] == 40,
        "fps_is_10": metadata["fps"] == 10,
        "single_raw_repo_root_after": roots_after == [str(root)],
        "single_arrow_tree_after": arrow_roots_after == [str(arrow_root)],
        "arrow_identity_stable": arrow_after == arrow_before,
        "full_hf_home_metadata_identity_stable": hf_after["summary"] == hf_before["summary"],
        "no_added_hf_home_entries": not tree_diff["added"],
        "no_removed_hf_home_entries": not tree_diff["removed"],
        "no_changed_hf_home_entries": not tree_diff["changed"],
        "no_new_file_over_limit": tree_diff["max_added_file_bytes"] <= max_new_file_bytes,
        "hf_home_has_no_symlinks_after": hf_after["summary"]["symlink_count"] == 0,
        "hf_home_has_no_hardlinks_after": hf_after["summary"]["multi_link_file_count"] == 0,
    }
    checks = {**static_checks, **runtime_checks}
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "phase": "runtime",
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "dataset_root": str(pathlib.Path(dataset.root).resolve(strict=False)),
        "hf_home": str(home),
        "hf_datasets_cache": str(datasets_cache),
        "hf_lerobot_home": str(lerobot_home),
        "length": len(dataset),
        "metadata": metadata,
        "max_new_file_bytes": max_new_file_bytes,
        "raw_snapshot_identity": raw_identity,
        "raw_repo_roots_before": roots_before,
        "raw_repo_roots_after": roots_after,
        "arrow_tree_roots_before": arrow_roots_before,
        "arrow_tree_roots_after": arrow_roots_after,
        "arrow_before": arrow_before,
        "arrow_after": arrow_after,
        "hf_home_before": hf_before["summary"],
        "hf_home_after": hf_after["summary"],
        "hf_home_diff": tree_diff,
        "checks": checks,
    }
    _write_report(output, report)
    return report


def _write_report(output: pathlib.Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link() is atomic and refuses an existing destination, unlike replace().
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--hf-home", required=True, type=pathlib.Path)
    parser.add_argument("--hf-datasets-cache", required=True, type=pathlib.Path)
    parser.add_argument("--hf-lerobot-home", required=True, type=pathlib.Path)
    parser.add_argument("--raw-snapshot-report", required=True, type=pathlib.Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-arrow-root", required=True, type=pathlib.Path)
    parser.add_argument("--expected-arrow-builder", required=True)
    parser.add_argument("--expected-arrow-config", required=True)
    parser.add_argument("--expected-arrow-version", required=True)
    parser.add_argument("--expected-arrow-fingerprint", required=True)
    parser.add_argument("--expected-arrow-file-count", required=True, type=int)
    parser.add_argument("--expected-arrow-manifest-sha256", required=True)
    parser.add_argument("--max-new-file-bytes", type=int, default=100_000_000)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    report = verify(
        dataset_root=args.dataset_root,
        hf_home=args.hf_home,
        hf_datasets_cache=args.hf_datasets_cache,
        hf_lerobot_home=args.hf_lerobot_home,
        raw_snapshot_report=args.raw_snapshot_report,
        revision=args.revision,
        expected_arrow_root=args.expected_arrow_root,
        expected_arrow_builder=args.expected_arrow_builder,
        expected_arrow_config=args.expected_arrow_config,
        expected_arrow_version=args.expected_arrow_version,
        expected_arrow_fingerprint=args.expected_arrow_fingerprint,
        expected_arrow_file_count=args.expected_arrow_file_count,
        expected_arrow_manifest_sha256=args.expected_arrow_manifest_sha256,
        max_new_file_bytes=args.max_new_file_bytes,
        output=args.output,
        dataset_factory=LeRobotDataset,
    )
    print(json.dumps({"status": report["status"], "phase": report["phase"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
