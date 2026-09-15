"""Fail-closed in-memory pure-LoRA to official dense-pi0 merge primitives.

This module implements the mathematics of OpenPI commit
``15a9616a00943ada6c20a0f158e3adb39df2ccac`` without serializing a derived
checkpoint.  It is intentionally separate from training: callers must validate
the original base and adapter artifacts before calling :func:`merge_in_place`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


OFFICIAL_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
EXPECTED_DENSE_LEAF_COUNT = 50
EXPECTED_DENSE_PARAMETER_COUNT = 3_238_048_528


@dataclass(frozen=True)
class MergeRule:
    """One audited A/B pair and its dense kernel target."""

    name: str
    adapter_a: str
    adapter_b: str
    target: str
    operator: str  # ``einsum`` has alpha/r scaling; ``feedforward`` does not.
    variant: str  # ``gemma_2b`` or ``gemma_300m``.


def _rule(prefix: str, target_leaf: str, operator: str, variant: str) -> MergeRule:
    return MergeRule(
        name=prefix,
        adapter_a=prefix + "/lora_a",
        adapter_b=prefix + "/lora_b",
        target=prefix + "/" + target_leaf,
        operator=operator,
        variant=variant,
    )


def _feedforward_rule(prefix: str, target_leaf: str, variant: str) -> MergeRule:
    return MergeRule(
        name=prefix + "/" + target_leaf,
        adapter_a=prefix + "/" + target_leaf + "_lora_a",
        adapter_b=prefix + "/" + target_leaf + "_lora_b",
        target=prefix + "/" + target_leaf,
        operator="feedforward",
        variant=variant,
    )


MERGE_RULES = (
    _rule("PaliGemma/llm/layers/attn/attn_vec_einsum", "w", "einsum", "gemma_2b"),
    _rule("PaliGemma/llm/layers/attn/kv_einsum", "w", "einsum", "gemma_2b"),
    _rule("PaliGemma/llm/layers/attn/q_einsum", "w", "einsum", "gemma_2b"),
    _feedforward_rule("PaliGemma/llm/layers/mlp", "gating_einsum", "gemma_2b"),
    _feedforward_rule("PaliGemma/llm/layers/mlp", "linear", "gemma_2b"),
    _rule("PaliGemma/llm/layers/attn/attn_vec_einsum_1", "w", "einsum", "gemma_300m"),
    _rule("PaliGemma/llm/layers/attn/kv_einsum_1", "w", "einsum", "gemma_300m"),
    _rule("PaliGemma/llm/layers/attn/q_einsum_1", "w", "einsum", "gemma_300m"),
    _feedforward_rule("PaliGemma/llm/layers/mlp_1", "gating_einsum", "gemma_300m"),
    _feedforward_rule("PaliGemma/llm/layers/mlp_1", "linear", "gemma_300m"),
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def expected_adapter_paths() -> set[str]:
    return {path for rule in MERGE_RULES for path in (rule.adapter_a, rule.adapter_b)}


def validate_golden_mapping(golden: Mapping[str, Any]) -> None:
    entries = golden.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Golden manifest must contain entries")
    paths = [entry.get("path") for entry in entries]
    if any(not isinstance(path, str) for path in paths):
        raise ValueError("Golden entry path is invalid")
    expected = expected_adapter_paths()
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise ValueError("Golden adapter paths do not exactly match audited merge rules")
    invariant_count = golden.get("review_invariants", {}).get("adapter_leaf_count")
    if invariant_count != len(expected):
        raise ValueError("Golden adapter leaf count disagrees with audited merge rules")


def scaling_from_official_configs(gemma_module: Any) -> dict[str, float]:
    """Derive attention scales from the installed fixed OpenPI source.

    FeedForward's pinned implementation deliberately does *not* apply this
    scale.  We still record its declared scaling value for auditability.
    """
    variants = {"gemma_2b": "gemma_2b_lora", "gemma_300m": "gemma_300m_lora"}
    result: dict[str, float] = {}
    for dense_name, lora_name in variants.items():
        configs = gemma_module.get_config(lora_name).lora_configs
        attn = float(configs["attn"].scaling_value)
        ffn = float(configs["ffn"].scaling_value)
        if not np.isfinite(attn) or not np.isfinite(ffn) or attn <= 0 or ffn <= 0:
            raise ValueError("official LoRA scale must be finite and positive")
        result[dense_name + ".attention"] = attn
        result[dense_name + ".feedforward_declared"] = ffn
    return result


def _validate_triplet(rule: MergeRule, target: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[int, ...]:
    if target.ndim < 2 or a.ndim != target.ndim or b.ndim != target.ndim:
        raise ValueError(f"{rule.name}: arrays must share rank >= 2")
    leading = target.shape[:-2]
    if a.shape[:-2] != leading or b.shape[:-2] != leading:
        raise ValueError(f"{rule.name}: scan/head leading axes differ")
    if a.shape[-2] != target.shape[-2] or b.shape[-1] != target.shape[-1] or a.shape[-1] != b.shape[-2]:
        raise ValueError(f"{rule.name}: A/B/kernel matrix shapes do not compose")
    if not np.issubdtype(target.dtype, np.floating):
        raise TypeError(f"{rule.name}: target kernel must be floating point")
    if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(b.dtype, np.floating):
        raise TypeError(f"{rule.name}: LoRA leaves must be floating point")
    return leading


def merge_in_place(
    flat_base: Mapping[str, Any],
    flat_adapter: Mapping[str, Any],
    golden: Mapping[str, Any],
    *,
    attention_scales: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Merge all twenty audited LoRA leaves into ten dense kernels.

    Every leading scan/head coordinate is updated independently.  This avoids a
    full 4-D/5-D delta allocation; the largest temporary is one 2-D matrix. The
    returned dense dictionary owns the original base arrays and mutates only the
    ten targets in memory.  It never writes a checkpoint.
    """
    validate_golden_mapping(golden)
    if set(flat_adapter) != expected_adapter_paths():
        raise ValueError("adapter leaves are not exactly the audited Golden set")
    dense = {path: np.asarray(value) for path, value in flat_base.items()}
    targets = {rule.target for rule in MERGE_RULES}
    if not targets.issubset(dense):
        raise KeyError("base tree is missing one or more audited dense targets")
    audit_rules: list[dict[str, Any]] = []
    for rule in MERGE_RULES:
        target = dense[rule.target]
        a = np.asarray(flat_adapter[rule.adapter_a])
        b = np.asarray(flat_adapter[rule.adapter_b])
        leading = _validate_triplet(rule, target, a, b)
        if not target.flags.writeable:
            target = target.copy()
            dense[rule.target] = target
        scale = 1.0
        if rule.operator == "einsum":
            scale_key = rule.variant + ".attention"
            if scale_key not in attention_scales:
                raise KeyError(f"missing derived attention scale: {scale_key}")
            scale = float(attention_scales[scale_key])
        elif rule.operator != "feedforward":
            raise ValueError(f"unsupported merge operator: {rule.operator}")
        if not np.isfinite(scale):
            raise ValueError("merge scale must be finite")
        indices = np.ndindex(leading) if leading else ((),)
        for index in indices:
            delta = np.matmul(a[index], b[index])
            if scale != 1.0:
                delta *= scale
            np.add(target[index], delta, out=target[index], casting="same_kind")
        audit_rules.append({
            "rule": rule.name,
            "operator": rule.operator,
            "variant": rule.variant,
            "scale_applied": scale,
            "target": rule.target,
            "target_shape": list(target.shape),
            "target_dtype": str(target.dtype),
            "leading_chunk_count": int(np.prod(leading, dtype=np.int64)) if leading else 1,
        })
    return dense, {
        "schema_version": 1,
        "official_openpi_commit": OFFICIAL_OPENPI_COMMIT,
        "merge_before_cast": True,
        "target_count": len(targets),
        "adapter_leaf_count": len(expected_adapter_paths()),
        "non_target_leaf_count": len(dense) - len(targets),
        "rules": audit_rules,
        "attention_scales": dict(sorted(attention_scales.items())),
        "feedforward_scaling_policy": "Pinned FeedForward._dot adds xAB without scaling_value.",
    }


def validate_dense_reference(flat_dense: Mapping[str, Any], flat_reference: Mapping[str, Any]) -> dict[str, int]:
    """Require exact dense-tree key/shape agreement before model.load."""
    if set(flat_dense) != set(flat_reference):
        raise ValueError("merged dense tree keys do not exactly match ordinary Pi0 reference")
    count = 0
    for path, reference in flat_reference.items():
        value = np.asarray(flat_dense[path])
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"dense reference shape mismatch: {path}")
        count += int(value.size)
    if len(flat_dense) != EXPECTED_DENSE_LEAF_COUNT:
        raise ValueError("ordinary Pi0 leaf count differs from audited expectation")
    if count != EXPECTED_DENSE_PARAMETER_COUNT:
        raise ValueError("ordinary Pi0 parameter count differs from audited expectation")
    return {"leaf_count": len(flat_dense), "parameter_count": count}
