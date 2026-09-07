#!/usr/bin/env python3
"""Resume one bounded pure-LoRA segment with deterministic loader/RNG positioning."""

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

import resume_sequence
import run_s1d_hundred_step as s1d


def _atomic_replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_static(args: argparse.Namespace, environment: dict[str, str]) -> None:
    if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")
    if (args.segment_start, args.segment_end) != (100, 200):
        raise ValueError("the first T1 engineering segment is frozen to 100->200")
    if args.train_seed != 42 or args.eval_seed != 7:
        raise ValueError("the frozen training/evaluation seeds are 42 and 7")
    if not args.checkpoint_dir.is_dir() or not (args.checkpoint_dir / str(args.segment_start)).is_dir():
        raise FileNotFoundError("verified starting checkpoint step is missing")
    if not args.adapter_root.is_dir() or not (args.adapter_root / f"step-{args.segment_start:08d}.verified.json").is_file():
        raise FileNotFoundError("verified starting adapter receipt is missing")
    for path in (args.model_manifest, args.golden_manifest, args.s1d_acceptance_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _file_sha256(args.s1d_acceptance_report) != args.expected_s1d_acceptance_sha256:
        raise RuntimeError("S1d acceptance report file identity mismatch")
    acceptance = json.loads(args.s1d_acceptance_report.read_text(encoding="utf-8"))
    if acceptance.get("status") != "pass" or acceptance.get("report_identity_sha256") != args.expected_s1d_report_identity:
        raise RuntimeError("S1d acceptance report is not the frozen pass identity")
    if acceptance.get("checkpoint_artifact", {}).get("artifact_tree_sha256") != args.expected_checkpoint_tree_sha256:
        raise RuntimeError("starting checkpoint tree identity mismatch")
    if acceptance.get("adapter_identity_sha256") != args.expected_adapter_identity_sha256:
        raise RuntimeError("starting adapter identity mismatch")
    for path in (args.progress, args.loader_receipt, args.output):
        if path.exists():
            raise FileExistsError(path)
        if not s1d._inside(path, args.attempt_dir):
            raise ValueError("progress, loader receipt, and output must stay inside the attempt")
    if (args.checkpoint_dir / str(args.segment_end)).exists():
        raise FileExistsError("target checkpoint step already exists")
    if (args.adapter_root / f"step-{args.segment_end:08d}").exists():
        raise FileExistsError("target adapter step already exists")


def _verify_resume_artifacts(args: argparse.Namespace) -> dict[str, object]:
    import verify_s1d_result

    acceptance = json.loads(args.s1d_acceptance_report.read_text(encoding="utf-8"))
    checkpoint_tree = verify_s1d_result._artifact_manifest(args.checkpoint_dir)
    adapter_tree = verify_s1d_result._artifact_manifest(args.adapter_root)
    if checkpoint_tree["artifact_tree_sha256"] != args.expected_checkpoint_tree_sha256:
        raise RuntimeError("starting checkpoint files no longer match the verified tree")
    if adapter_tree["artifact_tree_sha256"] != acceptance["adapter_artifact"]["artifact_tree_sha256"]:
        raise RuntimeError("starting adapter files no longer match the verified tree")
    return {"checkpoint": checkpoint_tree, "adapter": adapter_tree}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--golden-manifest", type=Path, required=True)
    parser.add_argument("--s1d-acceptance-report", type=Path, required=True)
    parser.add_argument("--expected-s1d-acceptance-sha256", required=True)
    parser.add_argument("--expected-s1d-report-identity", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--expected-adapter-identity-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--loader-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--segment-end", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    args = parser.parse_args()
    _validate_static(args, dict(os.environ))

    _atomic_replace_json(args.progress, {
        "current_step": args.segment_start,
        "last_committed_step": args.segment_start,
        "recent_metrics": {},
        "pause_count": 0,
        "resume_count": 1,
        "resource_peaks": {},
    })
    _verify_resume_artifacts(args)

    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root))
    import flax.traverse_util
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training import checkpoints
    from openpi.training import config as training_config
    from openpi.training import data_loader as training_data_loader
    from openpi.training import pure_lora_checkpointing
    from openpi.training import sharding
    from pi0_pure_lora import adapter_artifact
    from pi0_pure_lora import experiment_identity
    from scripts import train

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"expected exactly one visible JAX GPU, got {devices}")
    manifest = experiment_identity.validate_model_manifest(json.loads(args.model_manifest.read_text()))
    golden = json.loads(args.golden_manifest.read_text())
    if _file_sha256(args.golden_manifest) != manifest["identities"]["golden_manifest_sha256"]:
        raise RuntimeError("Golden manifest identity mismatch")
    golden_paths = set(adapter_artifact.golden_entries(golden))
    base_config = training_config.get_config("pi0_libero_pure_lora")
    config = dataclasses.replace(
        base_config,
        batch_size=1,
        num_workers=0,
        num_train_steps=args.segment_end,
        seed=args.train_seed,
        exp_name=args.exp_name,
    )
    if config.checkpoint_dir.resolve() != args.checkpoint_dir.resolve():
        raise RuntimeError("checkpoint path does not match frozen config/exp-name")
    if (Path(config.pure_lora_adapter_base_dir) / args.exp_name).resolve() != args.adapter_root.resolve():
        raise RuntimeError("adapter path does not match frozen config/exp-name")
    if config.keep_period is not None or config.checkpoint_max_to_keep is not None:
        raise RuntimeError("automatic checkpoint pruning must remain disabled")

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def loader_factory():
        return training_data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)

    def batch_fingerprint(batch: object) -> str:
        leaves = jax.tree.leaves(batch)
        digest = hashlib.sha256()
        for leaf in leaves:
            array = np.ascontiguousarray(jax.device_get(leaf))
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    started = time.monotonic()
    data_iter, batch, loader_receipt = resume_sequence.verify_position(
        loader_factory, args.segment_start, batch_fingerprint
    )
    s1d._atomic_json_new(args.loader_receipt, loader_receipt)

    manager = None
    try:
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            args.checkpoint_dir,
            keep_period=None,
            max_to_keep=None,
            overwrite=False,
            resume=True,
        )
        if not resuming or tuple(manager.all_steps()) != (args.segment_start,):
            raise RuntimeError(f"checkpoint steps before segment are not exactly ({args.segment_start},)")
        rng = jax.random.key(config.seed)
        train_rng, init_rng = jax.random.split(rng)
        state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=True)
        state = checkpoints.restore_state(manager, state, loader_factory(), step=args.segment_start)
        jax.block_until_ready(state)
        if int(jax.device_get(state.step)) != args.segment_start:
            raise RuntimeError("restored train-state step mismatch")
        for _ in range(args.segment_start):
            train_rng, _ = jax.random.split(train_rng)

        before_flat = s1d._flat(state.params)
        if not golden_paths <= set(before_flat):
            raise RuntimeError("Golden adapter paths missing from restored state")
        non_golden_paths = set(before_flat) - golden_paths
        adapter_before = s1d._hash_paths(before_flat, golden_paths, adapter_artifact.array_sha256)
        frozen_before = s1d._hash_paths(before_flat, non_golden_paths, adapter_artifact.array_sha256)
        step_fn = jax.jit(
            functools.partial(train.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        metrics_trace: list[dict[str, float]] = []
        for expected_step in range(args.segment_start + 1, args.segment_end + 1):
            train_rng, step_rng = jax.random.split(train_rng)
            with sharding.set_mesh(mesh):
                state, info = step_fn(step_rng, state, batch)
            jax.block_until_ready((state, info))
            reduced = jax.device_get(jax.tree.map(jnp.mean, info))
            metrics = {key: float(value) for key, value in reduced.items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"non-finite metrics at step {expected_step}")
            if int(jax.device_get(state.step)) != expected_step:
                raise RuntimeError(f"train-state step mismatch at {expected_step}")
            metrics_trace.append(metrics)
            _atomic_replace_json(args.progress, {
                "current_step": expected_step,
                "last_committed_step": args.segment_start,
                "recent_metrics": metrics,
                "pause_count": 0,
                "resume_count": 1,
                "resource_peaks": {},
            })
            if expected_step < args.segment_end:
                batch = next(data_iter)

        after_flat = s1d._flat(state.params)
        adapter_after = s1d._hash_paths(after_flat, golden_paths, adapter_artifact.array_sha256)
        frozen_after = s1d._hash_paths(after_flat, non_golden_paths, adapter_artifact.array_sha256)
        changed_adapters = sorted(path for path in golden_paths if adapter_before[path] != adapter_after[path])
        changed_frozen = sorted(path for path in non_golden_paths if frozen_before[path] != frozen_after[path])
        if set(changed_adapters) != golden_paths or changed_frozen:
            raise RuntimeError("pure-LoRA Golden/frozen post-segment invariant failed")
        receipt = pure_lora_checkpointing.save_restore_and_export(
            manager,
            state,
            loader_factory(),
            save_step=args.segment_end,
            model_manifest_path=args.model_manifest,
            golden_manifest_path=args.golden_manifest,
            adapter_root=args.adapter_root,
            train_seed=args.train_seed,
        )
        if tuple(manager.all_steps()) != (args.segment_start, args.segment_end):
            raise RuntimeError("old and new checkpoints did not coexist after verified save")
    finally:
        if manager is not None:
            manager.close()

    result = {
        "schema_version": 1,
        "status": "pass",
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "loader_position_verified": True,
        "starting_checkpoint_tree_verified": True,
        "starting_adapter_tree_verified": True,
        "rng_split_steps_replayed": args.segment_start,
        "metrics_count": len(metrics_trace),
        "all_metrics_finite": True,
        "changed_golden_leaf_count": len(changed_adapters),
        "changed_non_golden_leaf_count": len(changed_frozen),
        "checkpoint_steps": [args.segment_start, args.segment_end],
        "checkpoint_restore_receipt": receipt,
        "old_checkpoint_deleted": False,
        "next_stage_started": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    s1d._atomic_json_new(args.output, result)
    _atomic_replace_json(args.progress, {
        "current_step": args.segment_end,
        "last_committed_step": args.segment_end,
        "recent_metrics": metrics_trace[-1],
        "pause_count": 0,
        "resume_count": 1,
        "resource_peaks": {},
    })
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
