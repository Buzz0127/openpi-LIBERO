"""Run one explicit-noise direct Policy diagnostic for an authorized O2 arm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import o2_websocket_timing as timing


def action_summary(result: dict[str, Any], *, label: str = "action", expected_shape: tuple[int, int] = (50, 7)) -> tuple[np.ndarray, dict[str, Any]]:
    actions = np.asarray(result.get("actions"))
    if actions.shape != expected_shape or not np.isfinite(actions).all():
        raise ValueError(f"{label} returned invalid action chunk")
    return actions, {"sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(), "shape": list(actions.shape), "dtype": str(actions.dtype)}


def compare_actions(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("cannot compare invalid action arrays")
    difference = np.abs(left - right)
    result = {
        "exact_sha256_equal": hashlib.sha256(np.ascontiguousarray(left).tobytes()).hexdigest() == hashlib.sha256(np.ascontiguousarray(right).tobytes()).hexdigest(),
        "max_abs": float(np.max(difference)), "mean_abs": float(np.mean(difference)),
    }
    if left.shape == (50, 7):
        result.update({"component_layout": "physical-action-7d", "translation_max_abs": float(np.max(difference[:, :3])), "rotation_max_abs": float(np.max(difference[:, 3:6])), "gripper_max_abs": float(np.max(difference[:, 6:]))})
    else:
        result["component_layout"] = "model-latent"
    return result


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("o2_runtime_server", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_with_intermediates(policy: Any, observation: dict[str, Any], noise: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror pinned JAX ``Policy.infer`` and retain its pre-output-transform actions.

    The public inference API intentionally exposes only physical (unnormalized)
    actions.  O2 needs both spaces, so this narrow diagnostic copies the pinned
    JAX path verbatim up to ``_output_transform``.  It accepts explicit noise
    and advances the policy RNG by exactly one split, just like ``infer``.
    It is never used by the WebSocket serving path.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    if getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError("O2 direct diagnostic supports only the pinned JAX Policy")
    inputs = jax.tree.map(lambda value: value, observation)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
    policy._rng, sample_rng = jax.random.split(policy._rng)
    batched_noise = jnp.asarray(noise)
    if batched_noise.ndim == 2:
        batched_noise = batched_noise[None, ...]
    if batched_noise.ndim != 3:
        raise ValueError("explicit noise must have two or three dimensions")
    normalized_outputs = {
        "state": inputs["state"],
        "actions": policy._sample_actions(
            sample_rng, model_lib.Observation.from_dict(inputs), noise=batched_noise, **dict(policy._sample_kwargs)
        ),
    }
    normalized_outputs = jax.tree.map(lambda value: np.asarray(value[0, ...]), normalized_outputs)
    physical_outputs = policy._output_transform(dict(normalized_outputs))
    return normalized_outputs, physical_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-source", required=True, type=Path)
    parser.add_argument("--server-args", required=True, type=Path, help="JSON object matching the runtime server argparse namespace")
    parser.add_argument("--input-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat-output-dir", type=Path, help="optional same-instance replay after restoring the configured RNG seed")
    args = parser.parse_args()
    if args.output_dir.exists() or (args.repeat_output_dir is not None and args.repeat_output_dir.exists()):
        raise FileExistsError(args.output_dir)
    runtime = _load_module(args.server_source)
    raw_args = json.loads(args.server_args.read_text(encoding="utf-8"))
    for field in ("openpi_root", "base_params", "base_manifest", "golden", "adapter", "model_manifest", "norm_stats", "runtime_recipe"):
        raw_args[field] = Path(raw_args[field])
    namespace = SimpleNamespace(**raw_args)
    input_hashes = runtime.validate_paths(namespace)
    policy, runtime_manifest = runtime.build_policy(namespace, input_hashes)
    observation, bundle_manifest = timing._load_bundle(args.input_bundle)
    noise = np.load(args.input_bundle / "noise.npy", allow_pickle=False)
    def save(root: Path):
        normalized_result, result = infer_with_intermediates(policy, observation, noise)
        normalized_actions, normalized_summary = action_summary(normalized_result, label="normalized direct Policy action", expected_shape=(50, 32))
        actions, summary = action_summary(result, label="unnormalized direct Policy action")
        root.mkdir(parents=True)
        for name, value in (("actions.npy", actions), ("normalized_actions.npy", normalized_actions)):
            with (root / name).open("xb") as handle: np.save(handle, value, allow_pickle=False)
        output = {"schema_version": 1, "stage": "O2-direct-policy-explicit-noise", "input_bundle_identity_sha256": bundle_manifest["bundle_identity_sha256"], "noise_sha256": next(record["array_sha256"] for record in bundle_manifest["records"] if record["name"] == "noise"), "action": summary, "normalized_action": normalized_summary, "runtime_manifest": runtime_manifest}
        with (root / "result.json").open("x", encoding="utf-8") as handle: json.dump(output, handle, indent=2, sort_keys=True); handle.write("\n")
        return summary
    summary = save(args.output_dir)
    if args.repeat_output_dir is not None:
        import jax
        runtime.inject_policy_rng(policy, jax, namespace.rng_seed)
        save(args.repeat_output_dir)
    print(json.dumps({"status": "pass", "action_sha256": summary["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
