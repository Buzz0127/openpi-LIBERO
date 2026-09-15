"""Serve the project adapter through the pinned official dense-pi0 JAX chain.

This is an O2-only entry point.  It never creates a derived checkpoint: the
validated FP32 base plus adapter are merged one target kernel at a time in RAM,
then converted once to bfloat16 and placed on the selected JAX GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-patch-sha256", required=True)
    parser.add_argument("--runtime-recipe", required=True, type=Path)
    parser.add_argument(
        "--runtime-mode", choices=("merged-dense", "unmerged-lora"), default="merged-dense",
        help="O2 compares the same canonical transform/Policy chain with either merged dense or original LoRA modules.",
    )
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--rng-seed", type=int, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_paths(args: argparse.Namespace) -> dict[str, str]:
    for path in (
        args.openpi_root, args.base_params, args.base_manifest, args.golden,
        args.adapter, args.model_manifest, args.norm_stats, args.runtime_recipe,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 1 <= args.port <= 65535:
        raise ValueError("port outside valid range")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit() or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    recipe = json.loads(args.runtime_recipe.read_text(encoding="utf-8"))
    if recipe.get("runtime_kind") != "pi0_pure_lora_official_dense_bf16":
        raise ValueError("runtime recipe is not the official dense-pi0 recipe")
    return {
        "base_manifest_file_sha256": _sha256(args.base_manifest),
        "golden_manifest_sha256": _sha256(args.golden),
        "adapter_manifest_sha256": _sha256(args.adapter / "manifest.json"),
        "norm_stats_sha256": _sha256(args.norm_stats),
        "model_manifest_sha256": _sha256(args.model_manifest),
        "runtime_recipe_sha256": _sha256(args.runtime_recipe),
    }


def inject_policy_rng(policy: Any, jax: Any, seed: int) -> dict[str, Any]:
    """Set the actual Policy RNG after construction without truth-testing a key.

    The pinned upstream constructor evaluates ``rng or jax.random.key(0)``.
    Passing a typed JAX key into that expression raises a boolean-conversion
    error.  The thin compatibility step preserves normal ``Policy.infer``
    split-on-each-request semantics while making the requested seed explicit.
    """
    if not isinstance(seed, int):
        raise TypeError("rng seed must be an integer")
    key = jax.random.key(seed)
    policy._rng = key  # Pinned Policy owns and advances this field in infer().
    return {
        "rng_seed": seed,
        "rng_injection": "post-construction-private-field; pinned constructor rejects typed-key truth testing",
    }


def _base_identity(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    value = manifest.get("identities", {}).get("base_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("base manifest lacks canonical identity")
    return value


def build_policy(args: argparse.Namespace, input_hashes: dict[str, str]):
    """Build official dense Pi0/Policy/transforms from project-owned inputs."""
    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root))
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    import jax.numpy as jnp
    import numpy as np
    import adapter_artifact
    import official_pi0_lora_merge as merge
    import openpi.models.gemma as gemma
    import openpi.models.model as model_lib
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.policy as policy_lib
    import openpi.training.config as config_lib
    import openpi.transforms as transforms

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    identities = {
        "base_manifest_sha256": _base_identity(args.base_manifest),
        "config_patch_sha256": args.config_patch_sha256,
        "norm_stats_sha256": input_hashes["norm_stats_sha256"],
        "golden_manifest_sha256": input_hashes["golden_manifest_sha256"],
    }
    if model_manifest.get("model_mode") != "base_plus_adapter" or model_manifest.get("identities") != identities:
        raise ValueError("project model manifest is not the locked pure-LoRA artifact")
    adapter_manifest = json.loads((args.adapter / "manifest.json").read_text(encoding="utf-8"))
    if adapter_manifest.get("adapter_identity_sha256") != model_manifest.get("adapter_identity_sha256"):
        raise ValueError("adapter/model manifest identity mismatch")
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    merge.validate_golden_mapping(golden)

    # Abstract trees validate both source adapter shapes and the ordinary dense
    # target before any released parameter array is changed in memory.
    pure_config = config_lib.get_config("pi0_libero_pure_lora")
    dense_model_config = pi0_config.Pi0Config()
    pure_model = nnx.eval_shape(pure_config.model.create, jax.random.key(0))
    _, pure_state = nnx.split(pure_model)
    pure_reference = pure_state.to_pure_dict()
    base = model_lib.restore_params(args.base_params, restore_type=np.ndarray)
    validated_pure = adapter_artifact.compose_adapter(
        base, pure_reference, golden, args.adapter, expected_identities=identities
    )
    flat_validated = flax.traverse_util.flatten_dict(validated_pure, sep="/")
    if args.runtime_mode == "unmerged-lora":
        runtime_flat = dict(flat_validated)
        runtime_model_config = pure_config.model
        dense_contract = {
            "leaf_count": len(runtime_flat),
            "parameter_count": sum(int(np.asarray(value).size) for value in runtime_flat.values()),
        }
        merge_audit = {
            "mode": "unmerged-lora",
            "source_adapter_validated": True,
            "merge_before_cast": False,
            "adapter_leaf_count": len(merge.expected_adapter_paths()),
        }
    else:
        dense_model = nnx.eval_shape(dense_model_config.create, jax.random.key(0))
        _, dense_state = nnx.split(dense_model)
        dense_reference = dense_state.to_pure_dict()
        flat_adapter = {path: flat_validated[path] for path in merge.expected_adapter_paths()}
        flat_base = flax.traverse_util.flatten_dict(base, sep="/")
        merged_flat, merge_audit = merge.merge_in_place(
            flat_base, flat_adapter, golden, attention_scales=merge.scaling_from_official_configs(gemma)
        )
        flat_reference = flax.traverse_util.flatten_dict(dense_reference, sep="/")
        dense_contract = merge.validate_dense_reference(merged_flat, flat_reference)
        runtime_flat = merged_flat
        runtime_model_config = dense_model_config
        del dense_model, dense_state, dense_reference, flat_reference, flat_adapter, merged_flat
    del validated_pure, flat_validated, pure_reference, base

    devices = tuple(jax.devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("official dense runtime requires exactly one selected JAX GPU")
    device = devices[0]
    device_flat: dict[str, Any] = {}
    cast_audit: list[dict[str, Any]] = []
    for path in sorted(runtime_flat):
        source = runtime_flat.pop(path)
        device_value = jax.device_put(jnp.asarray(source, dtype=jnp.bfloat16), device)
        device_flat[path] = device_value
        cast_audit.append({"path": path, "source_dtype": str(source.dtype), "runtime_dtype": "bfloat16", "shape": list(source.shape), "placement": str(device)})
        del source
    runtime_parameters = {
        "placement": str(device), "leaf_count": len(device_flat),
        "parameter_count": dense_contract["parameter_count"], "runtime_dtype": "bfloat16",
    }
    model = runtime_model_config.load(flax.traverse_util.unflatten_dict(device_flat, sep="/"))
    # Keep the project's canonical target-domain normalization; only the model
    # graph/parameter representation follows the official dense Pi0 path.
    data_config = pure_config.data.create(pure_config.assets_dirs, runtime_model_config)
    if data_config.norm_stats is None:
        raise ValueError("canonical normalization did not resolve")
    policy = policy_lib.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=pure_config.policy_metadata,
    )
    rng_audit = inject_policy_rng(policy, jax, args.rng_seed)
    runtime_manifest = {
        "schema_version": 1,
        "runtime_kind": "pi0_pure_lora_official_dense_bf16",
        "runtime_mode": args.runtime_mode,
        "source_identities": identities,
        "adapter_identity_sha256": adapter_manifest["adapter_identity_sha256"],
        "input_hashes": input_hashes,
        "merge_audit": merge_audit,
        "dense_contract": dense_contract,
        "cast_audit": cast_audit,
        "runtime_parameters": runtime_parameters,
        "rng_audit": rng_audit,
    }
    return policy, runtime_manifest


def main() -> int:
    args = parse_args()
    input_hashes = validate_paths(args)
    policy, manifest = build_policy(args, input_hashes)
    from openpi.serving import websocket_policy_server

    print(json.dumps({"event": "official_dense_policy_ready", "port": args.port, "runtime_manifest": manifest}, sort_keys=True), flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
