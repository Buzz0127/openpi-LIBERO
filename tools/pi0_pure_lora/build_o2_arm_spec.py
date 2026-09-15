"""Seal the three sequential O2 serving commands without executing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARM_NAMES = ("official_pi0_libero_reference", "project_unmerged_bf16", "project_merged_dense_bf16")


def _absolute_existing(value: Path, label: str, *, directory: bool) -> Path:
    if not value.is_absolute() or not value.exists() or (not value.is_dir() if directory else not value.is_file()):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} must be an existing absolute {kind}")
    return value


def build(args: argparse.Namespace) -> list[dict[str, object]]:
    directories = {"openpi_root", "official_checkpoint", "base_params", "adapter"}
    paths = {name: _absolute_existing(getattr(args, name), name, directory=name in directories) for name in (
        "python", "openpi_root", "official_checkpoint", "server_source", "base_params",
        "base_manifest", "golden", "adapter", "model_manifest", "norm_stats", "runtime_recipe",
    )}
    if not (paths["official_checkpoint"] / "params").is_dir():
        raise ValueError("official_checkpoint lacks params directory")
    if not (paths["adapter"] / "manifest.json").is_file():
        raise ValueError("adapter lacks manifest.json")
    if args.output.exists() or len(set(args.ports)) != 3 or any(not 1024 <= port <= 65535 for port in args.ports):
        raise ValueError("output must be new and O2 needs three distinct unprivileged ports")
    common = [str(paths["python"]), str(paths["server_source"]), "--openpi-root", str(paths["openpi_root"]), "--base-params", str(paths["base_params"]), "--base-manifest", str(paths["base_manifest"]), "--golden", str(paths["golden"]), "--adapter", str(paths["adapter"]), "--model-manifest", str(paths["model_manifest"]), "--norm-stats", str(paths["norm_stats"]), "--config-patch-sha256", args.config_patch_sha256, "--runtime-recipe", str(paths["runtime_recipe"]), "--rng-seed", str(args.rng_seed)]
    return [
        {"name": ARM_NAMES[0], "port": args.ports[0], "command": [str(paths["python"]), str(paths["openpi_root"] / "scripts/serve_policy.py"), "--port", str(args.ports[0]), "policy:checkpoint", "--policy.config", "pi0_libero", "--policy.dir", str(paths["official_checkpoint"]) ]},
        {"name": ARM_NAMES[1], "port": args.ports[1], "command": [*common, "--runtime-mode", "unmerged-lora", "--port", str(args.ports[1])]},
        {"name": ARM_NAMES[2], "port": args.ports[2], "command": [*common, "--runtime-mode", "merged-dense", "--port", str(args.ports[2])]},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "openpi_root", "official_checkpoint", "server_source", "base_params", "base_manifest", "golden", "adapter", "model_manifest", "norm_stats", "runtime_recipe"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--config-patch-sha256", required=True)
    parser.add_argument("--rng-seed", required=True, type=int)
    parser.add_argument("--ports", type=int, nargs=3, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    arms = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(arms, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps({"status": "arm-spec-written", "arms": list(ARM_NAMES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
