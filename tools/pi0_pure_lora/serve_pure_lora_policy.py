"""Serve one verified pi0_base + pure-LoRA adapter policy over WebSocket.

The upstream policy server accepts an Orbax checkpoint only.  This narrowly
scoped entry point is the explicit adapter-only bridge for E1.  It never creates
LIBERO environments, runs episodes, writes checkpoints, or downloads assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-patch-sha256", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> dict[str, str]:
    for path in (
        args.openpi_root, args.base_params, args.base_manifest, args.golden,
        args.norm_stats, args.model_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.adapter is not None and not args.adapter.exists():
        raise FileNotFoundError(args.adapter)
    if args.port < 1 or args.port > 65535:
        raise ValueError("port outside valid range")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    return {
        # Retain the serialization hash in evidence, but do not confuse it
        # with the stable C0 base-weight identity used by adapter artifacts.
        "base_manifest_file_sha256": sha256_file(args.base_manifest),
        "golden_manifest_sha256": sha256_file(args.golden),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "model_manifest_sha256": sha256_file(args.model_manifest),
    }


def base_identity_from_manifest(path: Path) -> str:
    """Return the canonical base identity embedded in the C0 manifest."""
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("identities", {}).get("base_manifest_sha256")
    if not isinstance(identity, str) or len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("base manifest lacks a canonical base_manifest_sha256")
    return identity


def compose_base_with_reference_lora(base_params, reference_params, golden):
    """Complete a released base tree with its zero-effect reference LoRA leaves."""
    flat_base = adapter_artifact.flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_reference = adapter_artifact.flax.traverse_util.flatten_dict(reference_params, sep="/")
    golden_paths = set(adapter_artifact.golden_entries(golden))
    if set(flat_base) not in (set(flat_reference), set(flat_reference) - golden_paths):
        raise ValueError("Base parameter keys do not match the reference tree")
    completed = dict(flat_base)
    for path in golden_paths:
        completed[path] = flat_reference[path]
    if set(completed) != set(flat_reference):
        raise ValueError("Base completion did not produce the reference tree")
    return adapter_artifact.flax.traverse_util.unflatten_dict(completed, sep="/")


def build_policy(args: argparse.Namespace, input_hashes: dict[str, str]):
    """Build either the canonical Base or its identity-bound pure-LoRA overlay."""
    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root))
    import flax.nnx as nnx
    import jax
    import numpy as np
    import adapter_artifact
    import openpi.models.model as model_lib
    import openpi.policies.policy as policy_lib
    import openpi.training.config as config_lib
    import openpi.transforms as transforms

    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    identities = {
        "base_manifest_sha256": base_identity_from_manifest(args.base_manifest),
        "config_patch_sha256": args.config_patch_sha256,
        "norm_stats_sha256": input_hashes["norm_stats_sha256"],
        "golden_manifest_sha256": input_hashes["golden_manifest_sha256"],
    }
    if manifest.get("identities") != identities:
        raise ValueError("model manifest identity mismatch")
    mode = manifest.get("model_mode")
    if mode not in {"base", "base_plus_adapter"}:
        raise ValueError("unsupported model manifest mode")
    if mode == "base_plus_adapter":
        if args.adapter is None:
            raise ValueError("base_plus_adapter requires --adapter")
        adapter_manifest = json.loads((args.adapter / "manifest.json").read_text(encoding="utf-8"))
        if adapter_manifest.get("adapter_identity_sha256") != manifest.get("adapter_identity_sha256"):
            raise ValueError("adapter/model manifest identity mismatch")
    elif args.adapter is not None:
        raise ValueError("base model must not receive --adapter")
    config = config_lib.get_config("pi0_libero_pure_lora")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    _, abstract_state = nnx.split(abstract_model)
    reference = abstract_state.to_pure_dict()
    base = model_lib.restore_params(args.base_params, restore_type=np.ndarray)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    composed = (
        adapter_artifact.compose_adapter(base, reference, golden, args.adapter, expected_identities=identities)
        if mode == "base_plus_adapter"
        else compose_base_with_reference_lora(base, reference, golden)
    )
    model = config.model.load(composed)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("canonical norm stats did not resolve")
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
        metadata=config.policy_metadata,
    )
    return policy, manifest


def main() -> int:
    args = parse_args()
    input_hashes = validate_paths(args)
    policy, manifest = build_policy(args, input_hashes)
    from openpi.serving import websocket_policy_server

    print(json.dumps({
        "event": "e1_policy_ready", "port": args.port,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "adapter_identity_sha256": manifest["adapter_identity_sha256"],
        **input_hashes,
    }, sort_keys=True), flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
