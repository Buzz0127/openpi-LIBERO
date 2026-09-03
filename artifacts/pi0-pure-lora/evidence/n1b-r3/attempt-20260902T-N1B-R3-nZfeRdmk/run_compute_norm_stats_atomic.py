#!/usr/bin/env python3
"""Run OpenPI normalization computation and atomically publish one canonical asset."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish_directory(
    target: Path,
    payload: bytes,
    validate: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Write, validate, and publish a new directory without exposing a partial target."""
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"canonical target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    output = temporary / "norm_stats.json"
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    validation = validate(output)
    _fsync_directory(temporary)
    if target.exists():
        raise FileExistsError(f"canonical target appeared before publish: {target}")
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    return validation


def _parse_identity(value: str) -> tuple[Path, str]:
    path_text, separator, expected = value.rpartition("=")
    if not separator or len(expected) != 64:
        raise argparse.ArgumentTypeError("identity must be PATH=64_HEX_SHA256")
    try:
        int(expected, 16)
    except ValueError as error:
        raise argparse.ArgumentTypeError("identity SHA-256 must be hexadecimal") from error
    return Path(path_text), expected.lower()


def _verify_identities(identities: list[tuple[Path, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, expected in identities:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"identity mismatch for {path}: expected {expected}, got {actual}")
        result[str(path.resolve())] = actual
    return result


def _assert_offline_environment() -> dict[str, str]:
    expected = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"environment mismatch: {key} must equal {value!r}")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(key):
            raise RuntimeError(f"proxy must be unset in offline stage: {key}")
    return expected


def _load_compute_module(path: Path):
    spec = importlib.util.spec_from_file_location("n1b_compute_norm_stats", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load compute script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_stats(stats: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    expected_shapes = {"state": (8,), "actions": (7,)}
    if set(stats) != set(expected_shapes):
        raise RuntimeError(f"unexpected normalization keys: {sorted(stats)}")
    summary: dict[str, Any] = {}
    for key, expected_shape in expected_shapes.items():
        fields: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for field in ("mean", "std", "q01", "q99"):
            array = np.asarray(getattr(stats[key], field))
            if array.shape != expected_shape:
                raise RuntimeError(f"{key}.{field} shape {array.shape} != {expected_shape}")
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"{key}.{field} contains non-finite values")
            arrays[field] = array
            fields[field] = array.tolist()
        if not np.all(arrays["std"] > 0):
            raise RuntimeError(f"{key}.std must be strictly positive")
        if not np.all(arrays["q01"] <= arrays["q99"]):
            raise RuntimeError(f"{key} quantiles are reversed")
        summary[key] = {"shape": list(expected_shape), **fields}
    return summary


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-script", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--expected-target", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--expected-dataset-length", required=True, type=int)
    parser.add_argument("--identity", action="append", required=True, type=_parse_identity)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start = time.time()
    if args.batch_size < 1 or args.num_workers < 0 or args.expected_dataset_length < 1:
        raise ValueError("batch size and dataset length must be positive; workers must be non-negative")
    environment = _assert_offline_environment()
    identities = _verify_identities(args.identity)
    compute_script = args.compute_script.resolve()
    if str(compute_script) not in identities:
        raise RuntimeError("compute script must be included in bound identities")
    target = args.expected_target.resolve()
    if target.exists():
        raise FileExistsError(f"canonical target already exists: {target}")

    module = _load_compute_module(compute_script)
    config = module._config.get_config(args.config_name)
    resolved_target = module.resolve_output_path(config).resolve()
    if resolved_target != target:
        raise RuntimeError(f"config target {resolved_target} != expected target {target}")

    publication: dict[str, Any] = {}
    original_save = module.normalize.save

    def atomic_save(directory: Path | str, norm_stats: dict[str, Any]) -> None:
        nonlocal publication
        requested = Path(directory).resolve()
        if requested != target:
            raise RuntimeError(f"save requested unexpected target: {requested}")
        if publication:
            raise RuntimeError("normalization save called more than once")
        serialized = module.normalize.serialize_json(norm_stats).encode()

        def validate(path: Path) -> dict[str, Any]:
            loaded = module.normalize.deserialize_json(path.read_text())
            return _validate_stats(loaded)

        summary = atomic_publish_directory(target, serialized, validate)
        published_file = target / "norm_stats.json"
        publication = {
            "path": str(published_file),
            "bytes": published_file.stat().st_size,
            "sha256": sha256_file(published_file),
            "stats": summary,
        }

    runtime_config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=args.num_workers)
    original_get_config = module._config.get_config

    def get_runtime_config(name: str):
        if name != args.config_name:
            raise RuntimeError(f"unexpected config request: {name}")
        return runtime_config

    module.normalize.save = atomic_save
    module._config.get_config = get_runtime_config
    try:
        module.main(args.config_name, max_frames=None)
    finally:
        module.normalize.save = original_save
        module._config.get_config = original_get_config
    if not publication:
        raise RuntimeError("compute script returned without publishing normalization stats")

    report = {
        "status": "pass",
        "phase": "N1b",
        "config_name": args.config_name,
        "elapsed_seconds": time.time() - start,
        "environment": environment,
        "identities": identities,
        "target": str(target),
        "runtime_loader": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "expected_dataset_length": args.expected_dataset_length,
            "expected_batches": args.expected_dataset_length // args.batch_size,
            "expected_processed_frames": (args.expected_dataset_length // args.batch_size) * args.batch_size,
            "source_drop_last_remainder_frames": args.expected_dataset_length % args.batch_size,
            "shuffle": False,
            "drop_last": True,
        },
        "publication": publication,
        "atomic_protocol": "same-filesystem temp directory, validated JSON, fsync file and directory, os.replace directory",
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
