"""Build (but do not run) a CPU Policy from pi0_base plus an adapter-only artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import numpy as np

import adapter_artifact
import openpi.models.model as model_lib
import openpi.policies.policy as policy_lib
import openpi.training.config as config_lib
import openpi.transforms as transforms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-patch-sha256", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    identities = {
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "config_patch_sha256": args.config_patch_sha256,
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "golden_manifest_sha256": sha256_file(args.golden),
    }
    config = config_lib.get_config("pi0_libero_pure_lora")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    _, abstract_state = nnx.split(abstract_model)
    reference = abstract_state.to_pure_dict()
    base = model_lib.restore_params(args.base_params, restore_type=np.ndarray)
    composed = adapter_artifact.compose_adapter(
        base, reference, golden, args.adapter, expected_identities=identities
    )
    model = config.model.load(composed)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("Canonical norm stats were not resolved for Policy construction")
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
    report = {
        "schema_version": 1,
        "status": "passed",
        "policy_type": f"{type(policy).__module__}.{type(policy).__name__}",
        "config": config.name,
        "adapter_identity_sha256": json.loads((args.adapter / "manifest.json").read_text())["adapter_identity_sha256"],
        "identities": identities,
        "canonical_norm_stats_resolved": True,
        "input_transform_count": 2 + len(data_config.data_transforms.inputs) + len(data_config.model_transforms.inputs),
        "output_transform_count": len(data_config.model_transforms.outputs) + 1 + len(data_config.data_transforms.outputs),
        "inference_executed": False,
        "gpu_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
