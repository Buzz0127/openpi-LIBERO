"""CPU-only real pi0_base adapter export/composition/roundtrip validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np

import adapter_artifact
import openpi.models.model as model_lib
import openpi.training.config as config_lib


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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-step", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    flat_reference = flax.traverse_util.flatten_dict(reference, sep="/")
    golden_map = adapter_artifact.golden_entries(golden)
    if set(golden_map) - set(flat_reference):
        raise ValueError("Golden leaves missing from real reference tree")

    base = model_lib.restore_params(args.base_params, restore_type=np.ndarray)
    flat_base = flax.traverse_util.flatten_dict(base, sep="/")
    if set(flat_base) != set(flat_reference) - set(golden_map):
        raise ValueError("Released pi0_base keys do not equal reference minus Golden adapters")
    synthetic_adapter = {
        path: np.zeros(tuple(entry["shape"]), dtype=np.dtype(flat_reference[path].dtype))
        for path, entry in golden_map.items()
    }
    initial = dict(flat_base)
    initial.update(synthetic_adapter)
    initial_tree = flax.traverse_util.unflatten_dict(initial, sep="/")
    base_hashes_before = {path: adapter_artifact.array_sha256(value) for path, value in flat_base.items()}

    first = adapter_artifact.export_adapter(
        initial_tree, golden, args.output_dir / "adapter-a", identities=identities,
        train_step=args.train_step, train_seed=args.train_seed,
    )
    composed = adapter_artifact.compose_adapter(
        base, reference, golden, args.output_dir / "adapter-a", expected_identities=identities,
    )
    flat_composed = flax.traverse_util.flatten_dict(composed, sep="/")
    base_hashes_after = {
        path: adapter_artifact.array_sha256(flat_composed[path]) for path in flat_base
    }
    if base_hashes_before != base_hashes_after:
        raise ValueError("A non-adapter base leaf changed during composition")
    second = adapter_artifact.export_adapter(
        composed, golden, args.output_dir / "adapter-b", identities=identities,
        train_step=args.train_step, train_seed=args.train_seed,
    )
    if first["adapter_identity_sha256"] != second["adapter_identity_sha256"]:
        raise ValueError("Adapter export/composition/re-export identity changed")
    config.model.load(composed)
    report = {
        "schema_version": 1,
        "status": "passed",
        "config": config.name,
        "base_leaf_count": len(flat_base),
        "adapter_leaf_count": len(golden_map),
        "full_leaf_count": len(flat_composed),
        "adapter_parameter_count": sum(int(np.asarray(flat_composed[path]).size) for path in golden_map),
        "identities": identities,
        "adapter_identity_sha256": first["adapter_identity_sha256"],
        "non_adapter_hashes_unchanged": True,
        "roundtrip_adapter_identity_equal": True,
        "model_load_tree_shape_validation": True,
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
