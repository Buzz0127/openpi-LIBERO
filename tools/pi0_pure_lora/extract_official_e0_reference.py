#!/usr/bin/env python3
"""Extract E0 split success references from the existing official summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, Mapping, Sequence

from preregister_e0_splits import atomic_write_new, canonical_bytes, validate_entries, validate_official_protocol


def verify_manifest_identity(manifest: Mapping[str, Any]) -> None:
    claimed = manifest.get("manifest_identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_identity_sha256", None)
    actual = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if claimed != actual:
        raise ValueError(f"manifest identity mismatch: {claimed!r} != {actual!r}")


def extract_reference(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    verify_manifest_identity(manifest)
    validate_entries(manifest.get("entries", []))
    validate_official_protocol(summary)
    task_map = {(item["suite"], item["task_id"]): item for item in summary["tasks"]}
    rows = []
    for entry in manifest["entries"]:
        task = task_map[(entry["suite"], entry["task_id"])]
        failed = set(task["failed_initial_states"])
        rows.append({**entry, "success": entry["initial_state_index"] not in failed})
    aggregates: Dict[str, Dict[str, Any]] = {}
    for split in ("development", "main"):
        selected = [row for row in rows if row["split"] == split]
        successes = sum(bool(row["success"]) for row in selected)
        aggregates[split] = {
            "episodes": len(selected),
            "successes": successes,
            "failures": len(selected) - successes,
            "success_rate": successes / len(selected),
        }
    return {
        "schema_version": 1,
        "scope": "existing pi0_libero external reference projected onto preregistered E0 states",
        "normalization_protocol": "pi0_libero checkpoint-owned; external reference, not controlled Base/pure-LoRA normalization",
        "source_overall": summary["overall"],
        "aggregates": aggregates,
        "entries": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--official-summary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = json.loads(args.official_summary.read_text(encoding="utf-8"))
    report = extract_reference(manifest, summary)
    atomic_write_new(args.output, report)
    print(json.dumps({"output": str(args.output), "aggregates": report["aggregates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
