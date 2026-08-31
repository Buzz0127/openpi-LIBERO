#!/usr/bin/env python3
"""Verify the frozen G1b LoRA manifest against source and real-tree evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification evidence: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tree = _load_json(args.tree)
    manifest = _load_json(args.manifest)
    if manifest.get("status") != "g1b_golden_frozen":
        raise ValueError(f"manifest is not frozen G1b golden: {manifest.get('status')!r}")

    source_hashes = {}
    for relative_path, expected_hash in manifest["source_identity"].items():
        source_path = args.openpi_root / relative_path
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"source hash mismatch for {relative_path}: {actual_hash} != {expected_hash}"
            )
        source_hashes[relative_path] = actual_hash
    records = {record["path"]: record for record in tree["records"]}
    entries = manifest["entries"]
    manifest_paths = [entry["path"] for entry in entries]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("duplicate paths in proposed golden manifest")

    expected_tree_hash = manifest["evidence_identity"]["full_param_tree_sha256"]
    actual_tree_hash = _sha256(args.tree)
    if actual_tree_hash != expected_tree_hash:
        raise ValueError(
            f"full parameter tree hash mismatch: {actual_tree_hash} != {expected_tree_hash}"
        )

    missing_paths = sorted(set(manifest_paths) - set(records))
    if missing_paths:
        raise ValueError(f"manifest paths absent from real parameter tree: {missing_paths}")

    review = manifest["independent_review"]
    terminals = set(review["source_declared_terminals"])
    legal_parents = set(review["legal_parent_paths"])
    terminal_matches = {
        path for path, record in records.items() if record["terminal"] in terminals
    }
    illegal_parent_matches = sorted(
        path for path in terminal_matches if records[path]["parent_path"] not in legal_parents
    )
    if illegal_parent_matches:
        raise ValueError(
            f"LoRA terminal found outside independently approved parents: {illegal_parent_matches}"
        )
    independently_derived_candidate_paths = {
        path
        for path, record in records.items()
        if record["terminal"] in terminals and record["parent_path"] in legal_parents
    }
    manifest_path_set = set(manifest_paths)
    if manifest_path_set != independently_derived_candidate_paths:
        raise ValueError(
            "manually enumerated manifest differs from independently derived exact path set: "
            f"missing={sorted(independently_derived_candidate_paths - manifest_path_set)}, "
            f"extra={sorted(manifest_path_set - independently_derived_candidate_paths)}"
        )

    for entry in entries:
        record = records[entry["path"]]
        if entry["shape"] != record["shape"]:
            raise ValueError(f"shape mismatch for {entry['path']}")
        if entry["parameter_count"] != record["parameter_count"]:
            raise ValueError(f"parameter-count mismatch for {entry['path']}")
        if record["variable_type"] != manifest["review_invariants"]["variable_type"]:
            raise ValueError(f"variable-type mismatch for {entry['path']}")
        if record["dtype"] != manifest["review_invariants"]["dtype"]:
            raise ValueError(f"dtype mismatch for {entry['path']}")

    protected = tuple(manifest["protected_non_adapter_scopes"])
    protected_matches = sorted(
        path for path in manifest_paths if path.startswith(protected)
    )
    if protected_matches:
        raise ValueError(f"adapter manifest intersects protected scopes: {protected_matches}")

    adapter_parameter_count = sum(records[path]["parameter_count"] for path in manifest_paths)
    non_adapter_paths = set(records) - manifest_path_set
    non_adapter_parameter_count = sum(records[path]["parameter_count"] for path in non_adapter_paths)
    invariants = manifest["review_invariants"]
    actual_values = {
        "adapter_leaf_count": len(manifest_paths),
        "adapter_parameter_count": adapter_parameter_count,
        "total_param_leaf_count": len(records),
        "total_parameter_count": sum(record["parameter_count"] for record in records.values()),
        "non_adapter_leaf_count": len(non_adapter_paths),
        "non_adapter_parameter_count": non_adapter_parameter_count,
    }
    for name, actual in actual_values.items():
        if invariants[name] != actual:
            raise ValueError(f"manifest invariant mismatch for {name}: {invariants[name]} != {actual}")

    result = {
        "schema_version": 1,
        "status": "verified_g1b_golden_frozen",
        "tree": {"path": str(args.tree.resolve()), "sha256": actual_tree_hash},
        "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest)},
        "source_hashes": dict(sorted(source_hashes.items())),
        "actual_values": actual_values,
        "manifest_equals_independently_derived_exact_path_set": True,
        "illegal_parent_matches": [],
        "protected_scope_intersection": [],
        "all_manifest_shapes_counts_types_and_dtypes_match": True,
    }
    _write_json_exclusive(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
