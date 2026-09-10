#!/usr/bin/env python3
"""Run one separately authorized FT0-frozen formal pure-LoRA segment."""

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

import formal_segment_contract as contract
import resume_sequence
import run_s1d_hundred_step as s1d


def _new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_inputs(args: argparse.Namespace) -> dict[str, object]:
    for path in (args.openpi_root, args.model_manifest, args.golden_manifest, args.ft0_package):
        if not path.exists():
            raise FileNotFoundError(path)
    receipt = contract.validate_segment(
        package_path=args.ft0_package, segment_start=args.segment_start, segment_end=args.segment_end,
        environment=dict(os.environ), attempt_dir=args.attempt_dir,
        outputs={"progress": args.progress, "loader_receipt": args.loader_receipt, "rng_receipt": args.rng_receipt,
                 "composition_receipt": args.composition_receipt, "result": args.output},
        checkpoint_root=args.checkpoint_root, adapter_root=args.adapter_root,
    )
    if args.segment_start == 0:
        if args.previous_result is not None or args.expected_previous_result_sha256 is not None:
            raise ValueError("fresh FT1 must not accept a previous segment result")
    else:
        if args.previous_result is None or args.expected_previous_result_sha256 is None:
            raise ValueError("resumed formal segment requires its previous result identity")
        if not args.previous_result.is_file() or _sha256(args.previous_result) != args.expected_previous_result_sha256:
            raise RuntimeError("previous formal segment result identity mismatch")
        previous = json.loads(args.previous_result.read_text())
        if previous.get("status") != "pass" or previous.get("segment_end") != args.segment_start:
            raise RuntimeError("previous formal segment result does not certify this resume boundary")
    return receipt


def _run_gpu(args: argparse.Namespace, receipt: dict[str, object]) -> dict[str, object]:
    """Heavy path, deliberately entered only after all CPU-only gates pass."""
    sys.path[:0] = [str(args.openpi_root / "src"), str(args.openpi_root)]
    import flax.traverse_util
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training import checkpoints, config as training_config, data_loader, pure_lora_checkpointing, sharding
    from pi0_pure_lora import adapter_artifact, experiment_identity
    from scripts import train

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"expected exactly one visible JAX GPU, got {devices}")
    manifest = experiment_identity.validate_model_manifest(json.loads(args.model_manifest.read_text()))
    golden = json.loads(args.golden_manifest.read_text())
    if _sha256(args.golden_manifest) != manifest["identities"]["golden_manifest_sha256"]:
        raise RuntimeError("Golden manifest identity mismatch")
    golden_paths = set(adapter_artifact.golden_entries(golden))
    if len(golden_paths) != 20:
        raise RuntimeError("formal runner requires exactly 20 Golden adapter leaves")
    base_config = training_config.get_config("pi0_libero_pure_lora")
    config = dataclasses.replace(base_config, batch_size=1, num_workers=0, num_train_steps=args.segment_end,
                                 save_interval=args.segment_end, seed=42, exp_name=args.run_name,
                                 keep_period=None, checkpoint_max_to_keep=None)
    if config.checkpoint_dir.resolve() != args.checkpoint_root.resolve():
        raise RuntimeError("formal checkpoint root differs from frozen config")
    if (Path(config.pure_lora_adapter_base_dir) / args.run_name).resolve() != args.adapter_root.resolve():
        raise RuntimeError("formal adapter root differs from frozen config")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    def loader_factory(): return data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    def fingerprint(batch: object) -> str:
        digest = hashlib.sha256()
        for leaf in jax.tree.leaves(batch):
            array = np.ascontiguousarray(jax.device_get(leaf)); digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
    data_iter, batch, loader = resume_sequence.verify_position(loader_factory, args.segment_start, fingerprint)
    _new_json(args.loader_receipt, loader)
    manager = None
    try:
        manager, resuming = checkpoints.initialize_checkpoint_dir(args.checkpoint_root, keep_period=None, max_to_keep=None, overwrite=False, resume=args.segment_start > 0)
        prior_steps = tuple(row["end"] for row in json.loads(args.ft0_package.read_text())["formal_training"]["segments"] if row["end"] <= args.segment_start)
        if tuple(manager.all_steps()) != prior_steps or resuming != (args.segment_start > 0):
            raise RuntimeError("formal checkpoint history differs from the frozen segment boundary")
        rng = jax.random.key(config.seed); train_rng, init_rng = jax.random.split(rng)
        state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=args.segment_start > 0)
        if args.segment_start:
            state = checkpoints.restore_state(manager, state, loader_factory(), step=args.segment_start)
        jax.block_until_ready(state)
        if int(jax.device_get(state.step)) != args.segment_start: raise RuntimeError("formal restored step mismatch")
        for _ in range(args.segment_start): train_rng, _ = jax.random.split(train_rng)
        _new_json(args.rng_receipt, {"seed": 42, "restored_step": args.segment_start, "replayed_split_count": args.segment_start})
        before = s1d._flat(state.params); frozen = set(before) - golden_paths
        if len(frozen) != 50 or not golden_paths <= set(before): raise RuntimeError("formal parameter tree violates 20/50 contract")
        before_adapter = s1d._hash_paths(before, golden_paths, adapter_artifact.array_sha256); before_frozen = s1d._hash_paths(before, frozen, adapter_artifact.array_sha256)
        step_fn = jax.jit(functools.partial(train.train_step, config), in_shardings=(replicated, state_sharding, data_sharding), out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        metrics = []
        for step in range(args.segment_start + 1, args.segment_end + 1):
            train_rng, step_rng = jax.random.split(train_rng)
            with sharding.set_mesh(mesh): state, info = step_fn(step_rng, state, batch)
            jax.block_until_ready((state, info)); row = {key: float(value) for key, value in jax.device_get(jax.tree.map(jnp.mean, info)).items()}
            if not all(math.isfinite(value) for value in row.values()) or int(jax.device_get(state.step)) != step: raise RuntimeError("formal segment produced non-finite metrics or wrong step")
            metrics.append(row)
            if step < args.segment_end: batch = next(data_iter)
        after = s1d._flat(state.params); after_adapter = s1d._hash_paths(after, golden_paths, adapter_artifact.array_sha256); after_frozen = s1d._hash_paths(after, frozen, adapter_artifact.array_sha256)
        if any(before_adapter[p] == after_adapter[p] for p in golden_paths) or any(before_frozen[p] != after_frozen[p] for p in frozen): raise RuntimeError("formal pure-LoRA Golden/frozen invariant failed")
        saved = pure_lora_checkpointing.save_restore_and_export(manager, state, loader_factory(), save_step=args.segment_end, model_manifest_path=args.model_manifest, golden_manifest_path=args.golden_manifest, adapter_root=args.adapter_root, train_seed=42)
        if tuple(manager.all_steps()) != prior_steps + (args.segment_end,): raise RuntimeError("new full state did not coexist with preceding known-good state")
    finally:
        if manager is not None: manager.close()
    _new_json(args.composition_receipt, {"status": "pass", "save_step": args.segment_end, "adapter_identity_sha256": saved["adapter_identity_sha256"]})
    return {"schema_version": 1, "stage": "formal-pure-lora-segment", "status": "pass", "segment_start": args.segment_start, "segment_end": args.segment_end, "identities": manifest["identities"], "ft0_contract": receipt, "metrics_count": len(metrics), "all_metrics_finite": True, "changed_golden_leaf_count": 20, "changed_non_golden_leaf_count": 0, "checkpoint_steps": list(prior_steps + (args.segment_end,)), "checkpoint_restore_receipt": saved, "next_stage_started": False, "candidate": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openpi-root", "model-manifest", "golden-manifest", "ft0-package", "attempt-dir", "checkpoint-root", "adapter-root", "progress", "loader-receipt", "rng-receipt", "composition-receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--run-name", required=True); parser.add_argument("--segment-start", required=True, type=int); parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--previous-result", type=Path); parser.add_argument("--expected-previous-result-sha256")
    args = parser.parse_args(); receipt = _validate_local_inputs(args)
    # The autonomous orchestrator accepts only scalar progress state.  The
    # full FT0 contract remains independently bound in the final result.
    _new_json(args.progress, {"current_step": args.segment_start, "last_committed_step": args.segment_start})
    result = _run_gpu(args, receipt); _new_json(args.output, result); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
