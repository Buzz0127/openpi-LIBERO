"""Measure whether a tiny pinned LoRA layer survives bf16 dense fusion exactly.

This is a CPU-only diagnostic.  It deliberately does not set an acceptance
tolerance: strict O2 requires byte-identical outputs, so the report records
the exact result rather than reclassifying a nonzero difference as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _summary(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return {
        "exact_sha256_equal": hashlib.sha256(np.ascontiguousarray(left).tobytes()).hexdigest()
        == hashlib.sha256(np.ascontiguousarray(right).tobytes()).hexdigest(),
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "shape": list(left.shape),
        "dtype": str(left.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.openpi_root.is_absolute() or not (args.openpi_root / "src/openpi/models/lora.py").is_file():
        raise ValueError("--openpi-root must be the fixed OpenPI source tree")
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.openpi_root / "src"))
    import flax
    import jax
    import jax.numpy as jnp
    from openpi.models import lora

    if jax.default_backend() != "cpu" or len(jax.devices()) != 1:
        raise RuntimeError("O2 bf16 feasibility diagnostic must run on one CPU device")
    key = jax.random.key(7)
    x = (jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 13).astype(jnp.bfloat16)
    config = lora.LoRAConfig(rank=3, alpha=5.0)
    module = lora.Einsum((3, 5), lora_config=config)
    params = flax.core.unfreeze(module.init(key, "BSD,DF->BSF", x))
    params["params"]["w"] = jax.random.normal(jax.random.key(1), (3, 5), dtype=jnp.bfloat16)
    params["params"]["lora_a"] = jax.random.normal(jax.random.key(2), (3, 3), dtype=jnp.bfloat16)
    params["params"]["lora_b"] = jax.random.normal(jax.random.key(3), (3, 5), dtype=jnp.bfloat16)
    lora_params = flax.core.freeze(params)
    separate = module.apply(lora_params, "BSD,DF->BSF", x)
    dense_weight = (
        params["params"]["w"]
        + jnp.matmul(params["params"]["lora_a"], params["params"]["lora_b"]) * config.scaling_value
    ).astype(jnp.bfloat16)
    dense = lora.Einsum((3, 5))
    fused = dense.apply({"params": {"w": dense_weight}}, "BSD,DF->BSF", x)
    report = {
        "schema_version": 1,
        "stage": "O2-cpu-bf16-fusion-feasibility",
        "jax_backend": jax.default_backend(),
        "formula": "xW + (xA)B*(alpha/r) versus x*(W + A*B*(alpha/r))",
        "strict_equivalence": _summary(np.asarray(separate), np.asarray(fused)),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "measured", "exact": report["strict_equivalence"]["exact_sha256_equal"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
