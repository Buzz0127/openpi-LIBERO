"""CPU-only abstract-tree audit for the ordinary official dense Pi0Config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root))
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    import openpi.models.pi0_config as pi0_config
    import official_pi0_lora_merge as merge

    abstract_model = nnx.eval_shape(pi0_config.Pi0Config().create, jax.random.key(0))
    _, state = nnx.split(abstract_model)
    flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    paths = sorted(flat)
    parameter_count = sum(int(value.size) for value in flat.values())
    merge_targets = {rule.target for rule in merge.MERGE_RULES}
    missing_merge_targets = sorted(merge_targets - set(paths))
    result = {
        "schema_version": 1,
        "execution_kind": "abstract-shape-only",
        "leaf_count": len(paths),
        "parameter_count": parameter_count,
        "contains_lora_leaf": any("lora" in path for path in paths),
        "merge_target_count": len(merge_targets),
        "missing_merge_targets": missing_merge_targets,
        "expected_leaf_count": merge.EXPECTED_DENSE_LEAF_COUNT,
        "expected_parameter_count": merge.EXPECTED_DENSE_PARAMETER_COUNT,
        "status": "pass" if len(paths) == merge.EXPECTED_DENSE_LEAF_COUNT and parameter_count == merge.EXPECTED_DENSE_PARAMETER_COUNT and not any("lora" in path for path in paths) and not missing_merge_targets else "fail",
    }
    if result["status"] != "pass":
        raise ValueError(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
