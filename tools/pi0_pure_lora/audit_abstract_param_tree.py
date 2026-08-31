#!/usr/bin/env python3
"""Audit a serialized abstract parameter tree without importing OpenPI or JAX.

G1a deliberately operates only on synthetic JSON records.  A later, separately
approved stage may serialize real ``nnx.eval_shape`` leaves into the same input
format.  Candidate LoRA leaves are selected by exact terminal names and
variable types; a substring such as ``lora`` elsewhere in the path is never
sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LORA_LEAF_NAMES = frozenset(
    {
        "lora_a",
        "lora_b",
        "gating_einsum_lora_a",
        "gating_einsum_lora_b",
        "linear_lora_a",
        "linear_lora_b",
    }
)
DEFAULT_TRAINABLE_VARIABLE_TYPES = frozenset({"Param", "LoRAParam"})
BASE_TERMINAL_NAMES = frozenset({"w", "weight", "kernel", "bias"})


def _shape_and_count(raw_shape: Any) -> tuple[list[int], int]:
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("shape must be a non-empty JSON list")
    shape: list[int] = []
    for dim in raw_shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid shape dimension: {dim!r}")
        shape.append(dim)
    return shape, math.prod(shape)


def _normalize_record(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each leaf must be a JSON object")
    path_parts = raw.get("path_parts")
    if not isinstance(path_parts, list) or not path_parts:
        raise ValueError("path_parts must be a non-empty JSON list")
    if any(not isinstance(part, str) or not part for part in path_parts):
        raise ValueError("every serialized PathPart must be a non-empty string")
    variable_type = raw.get("variable_type")
    if not isinstance(variable_type, str) or not variable_type:
        raise ValueError("variable_type must be a non-empty string")
    dtype = raw.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype must be a non-empty string")
    shape, parameter_count = _shape_and_count(raw.get("shape"))

    leaf_name = path_parts[-1]
    lower_parts = [part.lower() for part in path_parts]
    exact_lora_candidate = (
        leaf_name in DEFAULT_LORA_LEAF_NAMES
        and variable_type in DEFAULT_TRAINABLE_VARIABLE_TYPES
    )
    return {
        "path_parts": path_parts,
        "joined_path": "/".join(path_parts),
        "parent_path": "/".join(path_parts[:-1]),
        "leaf_name": leaf_name,
        "variable_type": variable_type,
        "shape": shape,
        "dtype": dtype,
        "parameter_count": parameter_count,
        "exact_lora_candidate": exact_lora_candidate,
        "lora_substring_only": any("lora" in part.lower() for part in path_parts)
        and not exact_lora_candidate,
        "base_terminal": leaf_name.lower() in BASE_TERMINAL_NAMES,
        "subsystem_hints": sorted(
            {
                hint
                for hint, needles in {
                    "siglip": ("siglip", "vision", "image"),
                    "state_projection": ("state_proj", "state_projection"),
                    "action_projection": ("action_proj", "action_projection"),
                    "time_projection": ("time_proj", "time_projection"),
                }.items()
                if any(any(needle in part for needle in needles) for part in lower_parts)
            }
        ),
    }


def audit_records(records: Iterable[Any]) -> dict[str, Any]:
    normalized = [_normalize_record(record) for record in records]
    paths = [record["joined_path"] for record in normalized]
    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate serialized leaf paths: {duplicates}")

    normalized.sort(key=lambda record: record["joined_path"])
    candidate_paths = [
        record["joined_path"] for record in normalized if record["exact_lora_candidate"]
    ]
    return {
        "schema_version": 1,
        "mode": "g1a_synthetic_records_only",
        "candidate_rule": {
            "terminal_names": sorted(DEFAULT_LORA_LEAF_NAMES),
            "variable_types": sorted(DEFAULT_TRAINABLE_VARIABLE_TYPES),
            "path_substring_matching": False,
        },
        "leaf_count": len(normalized),
        "parameter_count": sum(record["parameter_count"] for record in normalized),
        "exact_lora_candidate_count": len(candidate_paths),
        "exact_lora_candidate_parameter_count": sum(
            record["parameter_count"]
            for record in normalized
            if record["exact_lora_candidate"]
        ),
        "exact_lora_candidate_paths": candidate_paths,
        "terminal_name_counts": dict(sorted(Counter(record["leaf_name"] for record in normalized).items())),
        "variable_type_counts": dict(
            sorted(Counter(record["variable_type"] for record in normalized).items())
        ),
        "records": normalized,
    }


def _exclusive_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-tree-json",
        required=True,
        type=Path,
        help="Synthetic leaf-record JSON; real Pi0 model construction is intentionally unsupported in G1a.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.fake_tree_json.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list):
        raise ValueError("fake-tree JSON root must be a list")

    report = audit_records(raw)
    report["input"] = {
        "path": str(args.fake_tree_json.resolve()),
        "sha256": _sha256(args.fake_tree_json),
    }
    _exclusive_json_write(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
