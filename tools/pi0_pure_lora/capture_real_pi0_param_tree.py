#!/usr/bin/env python3
"""Capture the real Pi0 LoRA abstract parameter tree without loading weights.

This G1b tool constructs exactly the model architecture used by
``pi0_libero_low_mem_finetune`` through ``nnx.eval_shape``.  It never imports a
training data config, restores a checkpoint, or performs a forward pass.  The
output is descriptive evidence only: source-derived candidates still require
an independent path-by-path review before they can become the golden adapter
manifest used by a freeze filter.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import flax
from flax import nnx
import jax

from openpi.models import pi0_config


MODEL_CONFIG = {
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
}

# This list is independently derived from the parameter declarations in
# openpi/models/lora.py at the pinned checkout.  It is a discovery aid, not the
# final training filter or the G2 test oracle.
SOURCE_DECLARED_LORA_TERMINALS = frozenset(
    {
        "lora_a",
        "lora_b",
        "gating_einsum_lora_a",
        "gating_einsum_lora_b",
        "linear_lora_a",
        "linear_lora_b",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite G1b evidence: {path}")
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


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _serialize_path_part(part: Any) -> dict[str, str]:
    return {
        "type": _qualified_type(part),
        "str": str(part),
        "repr": repr(part),
    }


def _serialize_variable(path: tuple[Any, ...], variable_state: Any) -> dict[str, Any]:
    value = variable_state.value
    shape = [int(dimension) for dimension in value.shape]
    path_strings = [str(part) for part in path]
    terminal = path_strings[-1]
    return {
        "path": "/".join(path_strings),
        "path_parts": path_strings,
        "typed_path_parts": [_serialize_path_part(part) for part in path],
        "terminal": terminal,
        "parent_path": "/".join(path_strings[:-1]),
        "variable_state_type": _qualified_type(variable_state),
        "variable_type": f"{variable_state.type.__module__}.{variable_state.type.__qualname__}",
        "abstract_value_type": _qualified_type(value),
        "shape": shape,
        "dtype": str(value.dtype),
        "parameter_count": math.prod(shape),
        "source_declared_lora_terminal": terminal in SOURCE_DECLARED_LORA_TERMINALS,
        "contains_lora_substring": any("lora" in part.lower() for part in path_strings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-openpi-commit",
        required=True,
        help="Pinned identity recorded into the report; the wrapper verifies Git separately.",
    )
    parser.add_argument("--capture-script", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse G1b output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "cpu" or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(
            f"G1b requires CPU-only JAX, got backend={backend!r}, devices={devices!r}"
        )

    config = pi0_config.Pi0Config(**MODEL_CONFIG)
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    all_state = nnx.state(abstract_model).flat_state()
    param_state = nnx.state(abstract_model, nnx.Param).flat_state()
    records = [_serialize_variable(path, variable) for path, variable in param_state.items()]
    records.sort(key=lambda record: record["path"])
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate flattened parameter paths found")

    all_state_type_counts = Counter(
        f"{variable.type.__module__}.{variable.type.__qualname__}"
        for variable in all_state.values()
    )
    terminal_counts = Counter(record["terminal"] for record in records)
    candidate_records = [
        record for record in records if record["source_declared_lora_terminal"]
    ]
    substring_only_records = [
        record
        for record in records
        if record["contains_lora_substring"] and not record["source_declared_lora_terminal"]
    ]

    tree_report = {
        "schema_version": 1,
        "mode": "g1b_real_nnx_eval_shape_cpu_only",
        "model_config": MODEL_CONFIG,
        "openpi_commit": args.expected_openpi_commit,
        "jax_backend": backend,
        "jax_devices": [str(device) for device in devices],
        "flax_version": flax.__version__,
        "jax_version": jax.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "all_state_leaf_count": len(all_state),
        "all_state_type_counts": dict(sorted(all_state_type_counts.items())),
        "param_leaf_count": len(records),
        "total_parameter_count": sum(record["parameter_count"] for record in records),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "records": records,
    }
    candidate_report = {
        "schema_version": 1,
        "status": "source_derived_candidates_not_yet_golden",
        "source_declared_lora_terminals": sorted(SOURCE_DECLARED_LORA_TERMINALS),
        "candidate_leaf_count": len(candidate_records),
        "candidate_parameter_count": sum(
            record["parameter_count"] for record in candidate_records
        ),
        "candidate_records": candidate_records,
        "lora_substring_only_leaf_count": len(substring_only_records),
        "lora_substring_only_records": substring_only_records,
    }
    summary = {
        "schema_version": 1,
        "status": "captured_pending_manual_golden_review",
        "started_epoch_seconds": started,
        "finished_epoch_seconds": time.time(),
        "elapsed_seconds": time.time() - started,
        "capture_script": {
            "path": str(args.capture_script.resolve()),
            "sha256": _sha256(args.capture_script),
        },
        "outputs": {
            "full_param_tree": "full_param_tree.json",
            "source_derived_candidates": "source_derived_adapter_candidates.json",
        },
    }

    _write_json_exclusive(args.output_dir / "full_param_tree.json", tree_report)
    _write_json_exclusive(
        args.output_dir / "source_derived_adapter_candidates.json", candidate_report
    )
    _write_json_exclusive(args.output_dir / "capture_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
