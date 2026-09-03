"""Strict identities shared by pure-LoRA training, serving, and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


IDENTITY_KEYS = {
    "base_manifest_sha256",
    "config_patch_sha256",
    "golden_manifest_sha256",
    "norm_stats_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def validate_git_oid(value: str, field: str) -> str:
    """Validate a full SHA-1 or SHA-256 Git object ID, never an abbreviation."""
    if len(value) not in (40, 64) or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a full lowercase Git object ID")
    return value


def validate_model_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stable_keys = {
        "schema_version", "experiment", "model_mode", "openpi_commit", "identities",
        "adapter_identity_sha256", "training_seed", "artifact_purpose",
    }
    if set(manifest) != stable_keys | {"model_identity_sha256"}:
        raise ValueError("model manifest has missing or unexpected fields")
    if manifest["schema_version"] != 1 or manifest["experiment"] != "pi0_base_to_libero_pure_lora":
        raise ValueError("unsupported model manifest schema or experiment")
    if manifest["model_mode"] not in ("base", "base_plus_adapter"):
        raise ValueError("unsupported model mode")
    identities = manifest["identities"]
    if set(identities) != IDENTITY_KEYS:
        raise ValueError("model identities are not the exact required set")
    for key, value in identities.items():
        validate_sha256(value, key)
    adapter = manifest["adapter_identity_sha256"]
    if manifest["model_mode"] == "base" and adapter is not None:
        raise ValueError("base mode must not contain an adapter identity")
    if manifest["model_mode"] == "base_plus_adapter":
        validate_sha256(adapter, "adapter_identity_sha256")
    validate_git_oid(manifest["openpi_commit"], "openpi_commit")
    stable = {key: manifest[key] for key in stable_keys}
    expected = canonical_sha256(stable)
    if manifest["model_identity_sha256"] != expected:
        raise ValueError("model identity hash mismatch")
    return dict(manifest)


def build_model_manifest(
    *, model_mode: str, openpi_commit: str, identities: Mapping[str, str],
    adapter_identity_sha256: str | None, training_seed: int | None, artifact_purpose: str,
) -> dict[str, Any]:
    stable = {
        "schema_version": 1,
        "experiment": "pi0_base_to_libero_pure_lora",
        "model_mode": model_mode,
        "openpi_commit": openpi_commit,
        "identities": dict(identities),
        "adapter_identity_sha256": adapter_identity_sha256,
        "training_seed": training_seed,
        "artifact_purpose": artifact_purpose,
    }
    manifest = dict(stable)
    manifest["model_identity_sha256"] = canonical_sha256(stable)
    return validate_model_manifest(manifest)


def atomic_write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
