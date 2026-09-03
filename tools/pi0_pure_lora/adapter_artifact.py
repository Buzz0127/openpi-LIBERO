"""Fail-closed adapter-only artifacts for the pi0 LIBERO pure-LoRA experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import flax.traverse_util
import numpy as np


REQUIRED_IDENTITIES = {
    "base_manifest_sha256",
    "config_patch_sha256",
    "norm_stats_sha256",
    "golden_manifest_sha256",
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(_canonical({"dtype": str(array.dtype), "shape": list(array.shape)}))
    # memoryview.cast does not support ml_dtypes.bfloat16 (PEP 3118 code E).
    # ndarray.tobytes preserves the exact contiguous storage bytes for all
    # numeric parameter dtypes used by OpenPI, including bfloat16.
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def golden_entries(golden: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = {entry["path"]: entry for entry in golden["entries"]}
    expected = int(golden["review_invariants"]["adapter_leaf_count"])
    if len(entries) != expected or len(entries) != len(golden["entries"]):
        raise ValueError("Golden manifest paths are missing or duplicated")
    return entries


def _validate_identities(identities: Mapping[str, str]) -> dict[str, str]:
    if set(identities) != REQUIRED_IDENTITIES or any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in identities.values()
    ):
        raise ValueError("Artifact identities must be the four exact lowercase SHA-256 fields")
    return dict(identities)


def export_adapter(
    params: Mapping[str, Any],
    golden: Mapping[str, Any],
    output_dir: Path,
    *,
    identities: Mapping[str, str],
    train_step: int,
    train_seed: int,
) -> dict[str, Any]:
    """Atomically export exactly the Golden adapter leaves as individual NPY files."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    checked_identities = _validate_identities(identities)
    golden_map = golden_entries(golden)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    missing = sorted(set(golden_map) - set(flat))
    if missing:
        raise KeyError(f"Missing Golden adapter leaves: {missing}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    arrays_dir = partial / "arrays"
    arrays_dir.mkdir()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(golden_map)):
        array = np.asarray(flat[path])
        expected_shape = tuple(golden_map[path]["shape"])
        if array.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {path}: {array.shape} != {expected_shape}")
        relative = f"arrays/{index:04d}.npy"
        destination = partial / relative
        with destination.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        records.append({
            "path": path,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "parameter_count": int(array.size),
            "file": relative,
            "file_sha256": _sha256_file(destination),
            "array_sha256": array_sha256(array),
        })
    stable = {
        "schema_version": 1,
        "artifact_type": "pi0_pure_lora_adapter_only",
        "identities": checked_identities,
        "train_step": int(train_step),
        "train_seed": int(train_seed),
        "entries": records,
    }
    manifest = dict(stable)
    manifest["adapter_identity_sha256"] = hashlib.sha256(_canonical(stable)).hexdigest()
    manifest_path = partial / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output_dir)
    return manifest


def compose_adapter(
    base_params: Mapping[str, Any],
    reference_params: Mapping[str, Any],
    golden: Mapping[str, Any],
    artifact_dir: Path,
    *,
    expected_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Overlay a validated adapter and require the result to equal the reference tree."""
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    identities = _validate_identities(expected_identities)
    if manifest.get("identities") != identities:
        raise ValueError("Adapter identity mismatch")
    stable = {key: manifest[key] for key in ("schema_version", "artifact_type", "identities", "train_step", "train_seed", "entries")}
    if hashlib.sha256(_canonical(stable)).hexdigest() != manifest.get("adapter_identity_sha256"):
        raise ValueError("Adapter manifest identity hash mismatch")

    golden_map = golden_entries(golden)
    records = {entry["path"]: entry for entry in manifest["entries"]}
    if set(records) != set(golden_map) or len(records) != len(manifest["entries"]):
        raise ValueError("Adapter entries are not exactly the Golden leaf set")
    expected_files = {"manifest.json", *(entry["file"] for entry in records.values())}
    actual_files = {str(path.relative_to(artifact_dir)) for path in artifact_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Adapter artifact contains missing or unexpected files")

    flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(reference_params, sep="/")
    if set(flat_base) not in (set(flat_ref), set(flat_ref) - set(golden_map)):
        raise ValueError("Base parameter keys do not match the reference tree")
    result = dict(flat_base)
    for path in sorted(golden_map):
        record = records[path]
        source = artifact_dir / record["file"]
        if _sha256_file(source) != record["file_sha256"]:
            raise ValueError(f"Adapter file hash mismatch: {path}")
        array = np.load(source, allow_pickle=False)
        reference = flat_ref[path]
        reference_shape = tuple(reference.shape)
        reference_dtype = np.dtype(reference.dtype)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Adapter manifest shape/dtype mismatch: {path}")
        if array.shape != reference_shape or array.dtype != reference_dtype:
            raise ValueError(f"Adapter/reference shape/dtype mismatch: {path}")
        if array_sha256(array) != record["array_sha256"]:
            raise ValueError(f"Adapter array hash mismatch: {path}")
        result[path] = array
    if set(result) != set(flat_ref):
        raise ValueError("Composed parameter keys do not exactly match the reference tree")
    return flax.traverse_util.unflatten_dict(result, sep="/")
