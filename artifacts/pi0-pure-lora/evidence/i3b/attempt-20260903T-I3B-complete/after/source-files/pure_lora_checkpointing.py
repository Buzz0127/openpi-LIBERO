"""Fail-closed checkpoint finalization for the pi0 LIBERO pure-LoRA run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jax

from openpi.training import checkpoints
from pi0_pure_lora import adapter_artifact
from pi0_pure_lora import experiment_identity


def _leaf_specs(tree: Any) -> list[tuple[str, tuple[int, ...], str]]:
    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(str(path), tuple(value.shape), str(value.dtype)) for path, value in leaves]


def _atomic_json_new(path: Path, value: object) -> None:
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


def save_restore_and_export(
    checkpoint_manager: Any,
    state: Any,
    data_loader: Any,
    *,
    save_step: int,
    model_manifest_path: Path,
    golden_manifest_path: Path,
    adapter_root: Path,
    train_seed: int,
) -> dict[str, Any]:
    """Save, wait, restore, verify tree/adapter values, then publish adapter."""
    model_manifest = experiment_identity.validate_model_manifest(
        json.loads(model_manifest_path.read_text(encoding="utf-8"))
    )
    if model_manifest.get("model_mode") != "base":
        raise ValueError("training initialization manifest must describe base mode")
    identities = model_manifest.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("model manifest identities missing")
    golden = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    if experiment_identity.sha256_file(golden_manifest_path) != identities["golden_manifest_sha256"]:
        raise ValueError("Golden manifest SHA-256 does not match model identity")
    adapter_dir = adapter_root / f"step-{save_step:08d}"
    receipt_path = adapter_root / f"step-{save_step:08d}.verified.json"
    if adapter_dir.exists() or receipt_path.exists():
        raise FileExistsError(f"segment artifacts already exist for step {save_step}")

    checkpoints.save_state(checkpoint_manager, state, data_loader, save_step)
    checkpoint_manager.wait_until_finished()
    restored = checkpoints.restore_state(checkpoint_manager, state, data_loader, step=save_step)
    if int(jax.device_get(restored.step)) != int(jax.device_get(state.step)):
        raise ValueError("restored training step mismatch")
    if _leaf_specs(restored.params) != _leaf_specs(state.params):
        raise ValueError("restored parameter tree/shape/dtype mismatch")
    if _leaf_specs(restored.opt_state) != _leaf_specs(state.opt_state):
        raise ValueError("restored optimizer tree/shape/dtype mismatch")

    source_params = state.params.to_pure_dict()
    restored_params = restored.params.to_pure_dict()
    source_flat = adapter_artifact.flax.traverse_util.flatten_dict(source_params, sep="/")
    restored_flat = adapter_artifact.flax.traverse_util.flatten_dict(restored_params, sep="/")
    golden_paths = set(adapter_artifact.golden_entries(golden))
    for path in golden_paths:
        if adapter_artifact.array_sha256(source_flat[path]) != adapter_artifact.array_sha256(restored_flat[path]):
            raise ValueError(f"restored adapter value mismatch: {path}")

    adapter_manifest = adapter_artifact.export_adapter(
        restored_params,
        golden,
        adapter_dir,
        identities=identities,
        train_step=save_step,
        train_seed=train_seed,
    )
    receipt = {
        "schema_version": 1,
        "save_step": save_step,
        "checkpoint_restore_succeeded": True,
        "parameter_tree_shape_dtype_equal": True,
        "optimizer_tree_shape_dtype_equal": True,
        "adapter_values_equal_after_restore": True,
        "adapter_identity_sha256": adapter_manifest["adapter_identity_sha256"],
        "automatic_pruning_enabled": False,
        "old_checkpoint_deletion_performed": False,
    }
    _atomic_json_new(receipt_path, receipt)
    return receipt
