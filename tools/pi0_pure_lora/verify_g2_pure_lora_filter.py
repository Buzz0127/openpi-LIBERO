#!/usr/bin/env python3
"""Verify the G2 production freeze filter against the frozen G1b manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from flax import nnx
import jax

from openpi.models import pi0_config


MODEL_VARIANTS = {
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite G2 evidence: {path}")
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


def _joined_paths(flat_state: dict[tuple[Any, ...], Any]) -> set[str]:
    return {"/".join(str(part) for part in path) for path in flat_state}


def _parameter_count(flat_state: dict[tuple[Any, ...], Any]) -> int:
    return sum(math.prod(variable.value.shape) for variable in flat_state.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--patch-sha256", required=True)
    args = parser.parse_args()

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"G2 audit must be CPU-only; devices={jax.devices()!r}")

    manifest = json.loads(args.golden_manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "g1b_golden_frozen":
        raise ValueError(f"manifest is not frozen: {manifest['status']!r}")
    golden_paths = {entry["path"] for entry in manifest["entries"]}
    if len(golden_paths) != len(manifest["entries"]):
        raise ValueError("duplicate paths in golden manifest")

    config = pi0_config.Pi0Config(**MODEL_VARIANTS)
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    all_params = nnx.state(abstract_model, nnx.Param).flat_state()
    freeze_filter = config.get_pure_lora_freeze_filter()
    trainable = nnx.state(
        abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))
    ).flat_state()
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()

    all_paths = _joined_paths(all_params)
    trainable_paths = _joined_paths(trainable)
    frozen_paths = _joined_paths(frozen)
    missing_golden = sorted(golden_paths - trainable_paths)
    unexpected_trainable = sorted(trainable_paths - golden_paths)
    overlap = sorted(trainable_paths & frozen_paths)
    partition_missing = sorted(all_paths - trainable_paths - frozen_paths)
    partition_extra = sorted((trainable_paths | frozen_paths) - all_paths)

    protected_scopes = tuple(manifest["protected_non_adapter_scopes"])
    protected_trainables = sorted(path for path in trainable_paths if path.startswith(protected_scopes))
    base_terminal_trainables = sorted(
        path
        for path in trainable_paths
        if path.rsplit("/", 1)[-1] in {"w", "weight", "kernel", "bias", "scale"}
    )

    invariants = manifest["review_invariants"]
    actual = {
        "total_leaf_count": len(all_params),
        "total_parameter_count": _parameter_count(all_params),
        "trainable_leaf_count": len(trainable),
        "trainable_parameter_count": _parameter_count(trainable),
        "frozen_leaf_count": len(frozen),
        "frozen_parameter_count": _parameter_count(frozen),
    }
    failures = {
        "missing_golden": missing_golden,
        "unexpected_trainable": unexpected_trainable,
        "trainable_frozen_overlap": overlap,
        "partition_missing": partition_missing,
        "partition_extra": partition_extra,
        "protected_trainables": protected_trainables,
        "base_terminal_trainables": base_terminal_trainables,
    }
    if any(failures.values()):
        raise AssertionError(json.dumps(failures, indent=2, sort_keys=True))
    expected = {
        "total_leaf_count": invariants["total_param_leaf_count"],
        "total_parameter_count": invariants["total_parameter_count"],
        "trainable_leaf_count": invariants["adapter_leaf_count"],
        "trainable_parameter_count": invariants["adapter_parameter_count"],
        "frozen_leaf_count": invariants["non_adapter_leaf_count"],
        "frozen_parameter_count": invariants["non_adapter_parameter_count"],
    }
    if actual != expected:
        raise AssertionError(f"G2 count mismatch: actual={actual}, expected={expected}")

    report = {
        "schema_version": 1,
        "status": "g2_pure_lora_filter_verified",
        "openpi_commit": args.openpi_commit,
        "patch_sha256": args.patch_sha256,
        "golden_manifest": {
            "path": str(args.golden_manifest.resolve()),
            "sha256": _sha256(args.golden_manifest),
        },
        "model_variants": MODEL_VARIANTS,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "actual": actual,
        "expected": expected,
        "failures": failures,
        "trainable_equals_golden": trainable_paths == golden_paths,
        "all_non_golden_params_frozen": frozen_paths == all_paths - golden_paths,
        "trainable_paths": sorted(trainable_paths),
    }
    _write_json_exclusive(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
