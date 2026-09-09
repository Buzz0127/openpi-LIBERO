"""CPU-only fail-closed contract for one FT0-frozen formal training segment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def _load_frozen_package(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    unsigned = dict(value)
    claimed = unsigned.pop("package_identity_sha256", None)
    if claimed != _canonical(unsigned):
        raise RuntimeError("FT0 package identity mismatch")
    if value.get("stage") != "FT0-formal-training-freeze":
        raise RuntimeError("unexpected formal training package stage")
    if value.get("execution_authorized") is not False or value.get("automatic_next_stage") is not False:
        raise RuntimeError("FT0 package authorization boundary changed")
    return value


def validate_segment(
    *,
    package_path: Path,
    segment_start: int,
    segment_end: int,
    environment: dict[str, str],
    attempt_dir: Path,
    outputs: dict[str, Path],
    checkpoint_root: Path,
    adapter_root: Path,
) -> dict[str, Any]:
    """Validate an FT segment before any OpenPI/JAX/model import.

    The caller remains responsible for the later GPU preflight and separately
    authorized launch.  This function only accepts a frozen candidate segment,
    a single-card mapping, and collision-safe task-owned output paths.
    """
    package = _load_frozen_package(package_path)
    training = package["formal_training"]
    matches = [row for row in training.get("segments", []) if (row.get("start"), row.get("end")) == (segment_start, segment_end)]
    if len(matches) != 1:
        raise ValueError("segment is not one pre-registered FT0 candidate boundary")
    segment = matches[0]
    if segment.get("user_authorization_required") is not True or segment.get("auto_start_next_segment") is not False:
        raise RuntimeError("segment authorization boundary changed")
    if training.get("starts_from") != "pi0_base" or training.get("starts_from_engineering_step_200") is not False:
        raise RuntimeError("formal trajectory must start from pi0_base, not engineering step 200")
    if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")
    if {"progress", "loader_receipt", "rng_receipt", "composition_receipt", "result"} != set(outputs):
        raise ValueError("formal runner outputs must have the exact required names")
    resolved = {name: path.resolve() for name, path in outputs.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("formal runner output paths must be distinct")
    for path in resolved.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if not path.is_relative_to(attempt_dir.resolve()):
            raise ValueError("formal runner outputs must stay inside the attempt")
    expected_checkpoint = Path(training["checkpoint_run_root"])
    expected_adapter = Path(training["adapter_run_root"])
    if checkpoint_root.resolve() != expected_checkpoint.resolve() or adapter_root.resolve() != expected_adapter.resolve():
        raise RuntimeError("formal artifact roots differ from the frozen new trajectory")
    if segment_start == 0:
        if checkpoint_root.exists() or adapter_root.exists():
            raise FileExistsError("fresh FT1 trajectory roots already exist")
        resume_mode = "fresh_pi0_base"
    else:
        if not (checkpoint_root / str(segment_start)).is_dir():
            raise FileNotFoundError("verified preceding full state is missing")
        if not (adapter_root / f"step-{segment_start:08d}.verified.json").is_file():
            raise FileNotFoundError("verified preceding adapter receipt is missing")
        resume_mode = "verified_prior_segment"
    return {
        "schema_version": 1,
        "ft0_package_sha256": _sha256(package_path),
        "ft0_package_identity_sha256": package["package_identity_sha256"],
        "segment_start": segment_start,
        "segment_end": segment_end,
        "resume_mode": resume_mode,
        "physical_gpu": int(visible),
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "train_seed": training["training_seed"],
        "eval_seed": training["evaluation_seed"],
        "execution_authorized": False,
        "next_stage_auto_start": False,
    }
