"""Bounded S1a checkpoint-load or train-state-init smoke; never trains or saves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


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


def _flat_paths(tree: object) -> set[str]:
    import flax.traverse_util

    return set(flax.traverse_util.flatten_dict(tree, sep="/"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("load", "compile"), required=True)
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
    import flax.traverse_util
    import jax
    from openpi.shared import array_typing as at
    from openpi.training import config as training_config
    from openpi.training import sharding
    from scripts import train

    started = time.monotonic()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"expected exactly one visible JAX GPU, got {devices}")
    config = training_config.get_config("pi0_libero_pure_lora")
    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    golden = json.loads(args.golden_manifest.read_text(encoding="utf-8"))
    if _sha256(args.golden_manifest) != manifest["identities"]["golden_manifest_sha256"]:
        raise RuntimeError("Golden manifest identity mismatch")
    if str(config.pure_lora_model_manifest) != str(args.model_manifest):
        raise RuntimeError("TrainConfig model manifest path mismatch")
    if str(config.pure_lora_golden_manifest) != str(args.golden_manifest):
        raise RuntimeError("TrainConfig Golden manifest path mismatch")

    rng = jax.random.key(config.seed)
    abstract_model = nnx.eval_shape(config.model.create, rng)
    expected = nnx.state(abstract_model, nnx.Param).to_pure_dict()
    loaded = config.weight_loader.load(expected)
    at.check_pytree_equality(expected=expected, got=loaded, check_shapes=True, check_dtypes=True)
    loaded = flax.traverse_util.unflatten_dict(
        {key: value for key, value in flax.traverse_util.flatten_dict(loaded).items() if not isinstance(value, jax.ShapeDtypeStruct)}
    )
    jax.block_until_ready(loaded)
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "physical_gpu": int(visible),
        "jax_devices": [str(device) for device in devices],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "loaded_leaf_count": len(_flat_paths(loaded)),
        "checkpoint_load_completed": True,
        "training_performed": False,
        "checkpoint_written": False,
    }
    if args.mode == "compile":
        mesh = sharding.make_mesh(1)
        state, _ = train.init_train_state(config, rng, mesh, resume=False)
        jax.block_until_ready(state)
        model = nnx.merge(state.model_def, state.params)
        trainable = nnx.state(model, config.trainable_filter).to_pure_dict()
        trainable_paths = _flat_paths(trainable)
        golden_paths = {entry["path"] for entry in golden["entries"]}
        if trainable_paths != golden_paths:
            raise RuntimeError("runtime trainable paths differ from Golden manifest")
        result.update({
            "train_state_init_compiled": True,
            "runtime_trainable_equals_golden": True,
            "runtime_trainable_leaf_count": len(trainable_paths),
            "train_state_step": int(jax.device_get(state.step)),
        })
    result["elapsed_seconds"] = time.monotonic() - started
    _atomic_json_new(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
