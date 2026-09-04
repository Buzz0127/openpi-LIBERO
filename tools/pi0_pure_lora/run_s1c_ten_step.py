#!/usr/bin/env python3
"""Run exactly ten real pure-LoRA training steps; never save a checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


TRAINING_STEPS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _flat(tree: object) -> dict[str, object]:
    import flax.traverse_util

    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    return flax.traverse_util.flatten_dict(tree, sep="/")


def _hash_paths(flat: dict[str, object], paths: set[str], array_sha256) -> dict[str, str]:
    return {path: array_sha256(flat[path]) for path in sorted(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--golden-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")

    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root))
    import flax.nnx as nnx
    from flax.training import common_utils
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training import config as training_config
    from openpi.training import data_loader as training_data_loader
    from openpi.training import sharding
    from pi0_pure_lora import adapter_artifact
    from pi0_pure_lora import experiment_identity
    from scripts import train

    started = time.monotonic()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"expected exactly one visible JAX GPU, got {devices}")
    manifest = experiment_identity.validate_model_manifest(json.loads(args.model_manifest.read_text()))
    golden = json.loads(args.golden_manifest.read_text())
    if _sha256(args.golden_manifest) != manifest["identities"]["golden_manifest_sha256"]:
        raise RuntimeError("Golden manifest identity mismatch")
    golden_paths = set(adapter_artifact.golden_entries(golden))

    base_config = training_config.get_config("pi0_libero_pure_lora")
    config = dataclasses.replace(base_config, batch_size=1, num_workers=0)
    if str(config.pure_lora_model_manifest) != str(args.model_manifest):
        raise RuntimeError("TrainConfig model manifest path mismatch")
    if str(config.pure_lora_golden_manifest) != str(args.golden_manifest):
        raise RuntimeError("TrainConfig Golden manifest path mismatch")

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    timing: dict[str, object] = {}
    mark = time.monotonic()
    loader = training_data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)
    batches = iter(loader)
    timing["data_loader_create_seconds"] = time.monotonic() - mark

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mark = time.monotonic()
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    timing["train_state_init_seconds"] = time.monotonic() - mark

    before_flat = _flat(state.params)
    all_paths = set(before_flat)
    if not golden_paths <= all_paths:
        raise RuntimeError("Golden adapter paths missing from runtime state")
    non_golden_paths = all_paths - golden_paths
    mark = time.monotonic()
    adapter_before = _hash_paths(before_flat, golden_paths, adapter_artifact.array_sha256)
    frozen_before = _hash_paths(before_flat, non_golden_paths, adapter_artifact.array_sha256)
    timing["pre_step_hash_seconds"] = time.monotonic() - mark

    step_fn = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    metrics_trace: list[dict[str, float]] = []
    step_seconds: list[float] = []
    batch_seconds: list[float] = []
    for expected_step in range(1, TRAINING_STEPS + 1):
        mark = time.monotonic()
        batch = next(batches)
        jax.block_until_ready(batch)
        batch_seconds.append(time.monotonic() - mark)
        train_rng, step_rng = jax.random.split(train_rng)
        mark = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = step_fn(step_rng, state, batch)
        jax.block_until_ready((state, info))
        step_seconds.append(time.monotonic() - mark)
        reduced = jax.device_get(jax.tree.map(jnp.mean, info))
        metrics = {key: float(value) for key, value in reduced.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"non-finite training metrics at step {expected_step}: {metrics}")
        if int(jax.device_get(state.step)) != expected_step:
            raise RuntimeError(f"train-state step mismatch at {expected_step}")
        metrics_trace.append(metrics)
    timing["batch_seconds"] = batch_seconds
    timing["step_compile_and_execute_seconds"] = step_seconds

    after_flat = _flat(state.params)
    if set(after_flat) != all_paths:
        raise RuntimeError("parameter leaf set changed after ten steps")
    mark = time.monotonic()
    adapter_after = _hash_paths(after_flat, golden_paths, adapter_artifact.array_sha256)
    frozen_after = _hash_paths(after_flat, non_golden_paths, adapter_artifact.array_sha256)
    timing["post_step_hash_seconds"] = time.monotonic() - mark
    changed_adapters = sorted(path for path in golden_paths if adapter_before[path] != adapter_after[path])
    changed_frozen = sorted(path for path in non_golden_paths if frozen_before[path] != frozen_after[path])
    if set(changed_adapters) != golden_paths:
        raise RuntimeError(f"not every Golden adapter changed: {sorted(golden_paths - set(changed_adapters))}")
    if changed_frozen:
        raise RuntimeError(f"non-Golden parameters changed: {changed_frozen}")

    opt_leaves = jax.tree_util.tree_leaves(state.opt_state)
    optimizer_array_parameters = sum(int(np.prod(leaf.shape)) for leaf in opt_leaves if hasattr(leaf, "shape"))
    result = {
        "schema_version": 1,
        "stage": "S1c",
        "physical_gpu": int(visible),
        "jax_devices": [str(device) for device in devices],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "batch_size": 1,
        "num_workers": 0,
        "shuffle": False,
        "initial_step": 0,
        "final_step": TRAINING_STEPS,
        "metrics_trace": metrics_trace,
        "all_metrics_finite": True,
        "golden_leaf_count": len(golden_paths),
        "non_golden_leaf_count": len(non_golden_paths),
        "changed_golden_leaf_count": len(changed_adapters),
        "changed_golden_paths": changed_adapters,
        "changed_non_golden_leaf_count": 0,
        "non_golden_hashes_unchanged": True,
        "optimizer_leaf_count": len(opt_leaves),
        "optimizer_array_parameter_count": optimizer_array_parameters,
        "timing": timing,
        "elapsed_seconds": time.monotonic() - started,
        "training_steps_executed": TRAINING_STEPS,
        "checkpoint_written": False,
        "adapter_written": False,
    }
    _atomic_json_new(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
