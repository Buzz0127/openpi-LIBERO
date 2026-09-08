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
import experiment_identity


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
    for path in (args.model_manifest, args.golden_manifest, args.s1d_acceptance_report, args.freeze_package):
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
    outputs = (args.progress, args.loader_receipt, args.rng_receipt, args.composition_receipt, args.output)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("runner output paths must be distinct")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if not s1d._inside(path, args.attempt_dir):
            raise ValueError("progress, loader receipt, and output must stay inside the attempt")
    if (args.checkpoint_dir / str(args.segment_end)).exists():
        raise FileExistsError("target checkpoint step already exists")
    if (args.adapter_root / f"step-{args.segment_end:08d}").exists():
        raise FileExistsError("target adapter step already exists")
    if (args.adapter_root / f"step-{args.segment_end:08d}.verified.json").exists():
        raise FileExistsError("target adapter receipt already exists")


def _validate_freeze(args: argparse.Namespace) -> dict[str, object]:
    freeze = json.loads(args.freeze_package.read_text())
    unsigned = dict(freeze)
    identity = unsigned.pop("package_identity_sha256", None)
    # The historical freeze builder includes a trailing newline in its identity.
    payload = (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if identity != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("freeze package identity mismatch")
    if freeze.get("execution_authorized") is not False or freeze.get("automatic_next_stage") is not False:
        raise RuntimeError("freeze package authorization boundary changed")
    segment = freeze.get("engineering_segment", {})
    if (segment.get("start"), segment.get("end"), segment.get("candidate")) != (100, 200, False):
        raise RuntimeError("freeze engineering segment mismatch")
    if freeze.get("seeds") != {"training": 42, "evaluation": 7}:
        raise RuntimeError("freeze seeds mismatch")
    model = experiment_identity.validate_model_manifest(json.loads(args.model_manifest.read_text()))
    expected = {
        "model": model["model_identity_sha256"],
        "golden": _file_sha256(args.golden_manifest),
        "norm": model["identities"]["norm_stats_sha256"],
        "config": model["identities"]["config_patch_sha256"],
    }
    if any(freeze.get("identities", {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError("freeze/model identity mismatch")
    if freeze["source"]["head"] != model["openpi_commit"]:
        raise RuntimeError("freeze/model source mismatch")
    resume = freeze.get("resume_input", {})
    if (resume.get("checkpoint_root") != str(args.checkpoint_dir)
            or resume.get("checkpoint_tree_sha256") != args.expected_checkpoint_tree_sha256
            or resume.get("adapter_identity_sha256") != args.expected_adapter_identity_sha256
            or resume.get("acceptance_report_identity_sha256") != args.expected_s1d_report_identity):
        raise RuntimeError("freeze resume input mismatch")
    return freeze


def verify_rng_replay(seed, restored_step, resumed_key, make_key, split_key, fingerprint):
    """Compare the resumed key with a separate replay of the original S1d sequence."""
    reference, _ = split_key(make_key(seed))
    for _ in range(restored_step):
        reference, _ = split_key(reference)
    _, expected_first = split_key(reference)
    _, actual_first = split_key(resumed_key)
    receipt = {
        "schema_version": 1, "seed": seed, "restored_step": restored_step,
        "replayed_split_count": restored_step,
        "reference_train_key_sha256": fingerprint(reference),
        "resumed_train_key_sha256": fingerprint(resumed_key),
        "first_step_key_sha256": fingerprint(actual_first),
        "reference_first_step_key_sha256": fingerprint(expected_first),
    }
    if (receipt["reference_train_key_sha256"] != receipt["resumed_train_key_sha256"]
            or receipt["first_step_key_sha256"] != receipt["reference_first_step_key_sha256"]):
        raise RuntimeError("RNG replay mismatch")
    return receipt


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
    parser.add_argument("--freeze-package", type=Path, required=True)
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
    parser.add_argument("--rng-receipt", type=Path, required=True)
    parser.add_argument("--composition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--segment-end", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    args = parser.parse_args()
    _validate_static(args, dict(os.environ))
    freeze = _validate_freeze(args)
    import autonomous_stage_orchestrator as orchestrator
    source = orchestrator._source_snapshot(args.openpi_root)
    expected_source = {**freeze["source"], "status": ""}
    if source != expected_source:
        raise RuntimeError("runtime source differs from frozen clean worktree")

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
    if (Path(config.pure_lora_model_manifest).resolve() != args.model_manifest.resolve()
            or Path(config.pure_lora_golden_manifest).resolve() != args.golden_manifest.resolve()):
        raise RuntimeError("config manifest paths differ from verified runner inputs")
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
        def key_hash(key):
            return hashlib.sha256(np.ascontiguousarray(jax.device_get(jax.random.key_data(key))).tobytes()).hexdigest()
        rng_receipt = verify_rng_replay(
            args.train_seed, args.segment_start, train_rng,
            jax.random.key, jax.random.split, key_hash,
        )
        s1d._atomic_json_new(args.rng_receipt, rng_receipt)

        before_flat = s1d._flat(state.params)
        if not golden_paths <= set(before_flat):
            raise RuntimeError("Golden adapter paths missing from restored state")
        non_golden_paths = set(before_flat) - golden_paths
        if len(golden_paths) != 20 or len(non_golden_paths) != 50:
            raise RuntimeError("restored parameter tree must contain 20 Golden and 50 frozen leaves")
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
        if set(after_flat) != set(before_flat):
            raise RuntimeError("parameter path set changed during segment")
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

    # Independently load the exported adapter and compare every composed parameter.
    reference_params = state.params.to_pure_dict()
    reference_flat = flax.traverse_util.flatten_dict(reference_params, sep="/")
    base_without_adapter = flax.traverse_util.unflatten_dict(
        {path: value for path, value in reference_flat.items() if path not in golden_paths}, sep="/"
    )
    composed = adapter_artifact.compose_adapter(
        base_without_adapter, reference_params, golden,
        args.adapter_root / f"step-{args.segment_end:08d}",
        expected_identities=manifest["identities"],
    )
    composed_flat = flax.traverse_util.flatten_dict(composed, sep="/")
    after_hashes = {**adapter_after, **frozen_after}
    composed_hashes = s1d._hash_paths(composed_flat, set(composed_flat), adapter_artifact.array_sha256)
    if composed_hashes != after_hashes:
        raise RuntimeError("base plus adapter composition differs from final parameters")
    s1d._atomic_json_new(args.composition_receipt, {
        "schema_version": 1, "status": "pass", "save_step": args.segment_end,
        "parameter_hashes": composed_hashes,
        "adapter_identity_sha256": receipt["adapter_identity_sha256"],
    })
    import verify_s1d_result
    checkpoint_after = verify_s1d_result._artifact_manifest(args.checkpoint_dir)
    adapter_artifact_after = verify_s1d_result._artifact_manifest(args.adapter_root)

    result = {
        "schema_version": 1,
        "stage": "T1-resume-engineering",
        "status": "pass",
        "identities": freeze["identities"],
        "source_head": source["head"],
        "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
        "jax_device_count": len(devices),
        "batch_size": 1, "num_workers": 0, "shuffle": True,
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "loader_position_verified": True,
        "starting_checkpoint_tree_verified": True,
        "starting_adapter_tree_verified": True,
        "rng_split_steps_replayed": args.segment_start,
        "metrics_count": len(metrics_trace),
        "metrics_trace": metrics_trace,
        "all_metrics_finite": True,
        "parameter_hashes_before": {**adapter_before, **frozen_before},
        "parameter_hashes_after": after_hashes,
        "loader_receipt_sha256": _file_sha256(args.loader_receipt),
        "rng_receipt_sha256": _file_sha256(args.rng_receipt),
        "composition_receipt_sha256": _file_sha256(args.composition_receipt),
        "changed_golden_leaf_count": len(changed_adapters),
        "changed_non_golden_leaf_count": len(changed_frozen),
        "checkpoint_steps": [args.segment_start, args.segment_end],
        "checkpoint_restore_receipt": receipt,
        "checkpoint_artifact_after": checkpoint_after,
        "adapter_artifact_after": adapter_artifact_after,
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
