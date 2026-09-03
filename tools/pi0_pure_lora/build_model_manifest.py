"""Build a strict Base or Base+Adapter model identity manifest from real artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import experiment_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["base", "base_plus_adapter"], required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--config-identity", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--artifact-purpose", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config_hash = args.config_identity.read_text(encoding="utf-8").strip()
    identities = {
        "base_manifest_sha256": experiment_identity.sha256_file(args.base_manifest),
        "config_patch_sha256": experiment_identity.validate_sha256(config_hash, "config_patch_sha256"),
        "golden_manifest_sha256": experiment_identity.sha256_file(args.golden),
        "norm_stats_sha256": experiment_identity.sha256_file(args.norm_stats),
    }
    adapter_identity = None
    if args.mode == "base_plus_adapter":
        if args.adapter is None:
            parser.error("base_plus_adapter requires --adapter")
        adapter_manifest = json.loads((args.adapter / "manifest.json").read_text(encoding="utf-8"))
        if adapter_manifest.get("identities") != identities:
            raise ValueError("adapter artifact identities do not match requested model inputs")
        adapter_identity = adapter_manifest["adapter_identity_sha256"]
    elif args.adapter is not None or args.training_seed is not None:
        parser.error("base mode must not specify adapter or training seed")
    manifest = experiment_identity.build_model_manifest(
        model_mode=args.mode,
        openpi_commit=args.openpi_commit,
        identities=identities,
        adapter_identity_sha256=adapter_identity,
        training_seed=args.training_seed,
        artifact_purpose=args.artifact_purpose,
    )
    experiment_identity.atomic_write_new(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
