#!/usr/bin/env python3
"""Diagnose N1b RSS growth with a state/action-only normalization input path."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


PROBES = (0, 106, 213, 101469, 101580, 153511, 153653, 220495, 220604, 273356, 273410, 273464)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_index_matrix(indices, episode_end_exclusive, horizon: int):
    indices = [int(value) for value in indices]
    ends = [int(value) for value in episode_end_exclusive]
    if len(indices) != len(ends):
        raise ValueError("indices and episode ends must have the same shape")
    return [[min(index + offset, end - 1) for offset in range(horizon)] for index, end in zip(indices, ends)]


def _rss_bytes() -> int:
    with Path("/proc/self/status").open() as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not found")


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


def _assert_offline() -> None:
    expected = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} must equal {value!r}")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(key):
            raise RuntimeError(f"proxy must be unset: {key}")


def _release_allocator_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _array_stack(values):
    import numpy as np

    return np.stack([np.asarray(value) for value in values])


def _slope(samples: list[dict[str, int]]) -> float:
    if len(samples) < 2:
        return 0.0
    first = samples[0]
    last = samples[-1]
    span = last["batch"] - first["batch"]
    return (last["rss_bytes"] - first["rss_bytes"]) / span if span else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--expected-target", required=True, type=Path)
    parser.add_argument("--expected-dataset-length", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", required=True, type=int)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--release-every", type=int, default=0)
    parser.add_argument("--identity", action="append", required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _assert_offline()
    if args.expected_target.exists():
        raise FileExistsError(f"canonical target exists: {args.expected_target}")
    if min(args.batch_size, args.max_batches, args.sample_every) < 1 or args.release_every < 0:
        raise ValueError("invalid batch or sampling arguments")

    identities: dict[str, str] = {}
    for value in args.identity:
        path_text, separator, expected = value.rpartition("=")
        if not separator:
            raise ValueError(f"invalid identity: {value}")
        path = Path(path_text).resolve(strict=True)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"identity mismatch for {path}: {actual} != {expected}")
        identities[str(path)] = actual

    import numpy as np
    import pyarrow as pa
    import openpi.shared.normalize as normalize
    import openpi.training.config as config
    import openpi.training.data_loader as data_loader
    import openpi.transforms as transforms

    start = time.time()
    train_config = config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    base = getattr(dataset, "_dataset", dataset)
    if len(dataset) != args.expected_dataset_length:
        raise RuntimeError(f"dataset length {len(dataset)} != {args.expected_dataset_length}")
    hf_dataset = base.hf_dataset
    if "state" not in hf_dataset.column_names or "actions" not in hf_dataset.column_names:
        raise RuntimeError(f"required columns missing: {hf_dataset.column_names}")

    end_by_frame = np.empty(len(dataset), dtype=np.int64)
    starts = np.asarray(base.episode_data_index["from"], dtype=np.int64)
    ends = np.asarray(base.episode_data_index["to"], dtype=np.int64)
    for episode_start, episode_end in zip(starts, ends, strict=True):
        end_by_frame[episode_start:episode_end] = episode_end

    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    probe_results = []
    for frame in PROBES:
        full = transform(copy.deepcopy(dataset[frame]))
        state = _array_stack(hf_dataset.select([frame])["state"])[0]
        query = query_index_matrix([frame], [end_by_frame[frame]], train_config.model.action_horizon)[0]
        actions = _array_stack(hf_dataset.select(query)["actions"])
        actions[..., :6] -= state[..., :6]
        state_equal = np.allclose(state, np.asarray(full["state"]), rtol=0.0, atol=0.0)
        actions_equal = np.allclose(actions, np.asarray(full["actions"]), rtol=0.0, atol=1e-6)
        probe_results.append({"frame": frame, "state_equal": bool(state_equal), "actions_equal": bool(actions_equal)})
    if not all(item["state_equal"] and item["actions_equal"] for item in probe_results):
        raise RuntimeError(f"lean path failed full-transform equivalence: {probe_results}")

    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    samples: list[dict[str, int]] = []
    limit = min(len(dataset), args.max_batches * args.batch_size)
    for batch_number, start_index in enumerate(range(0, limit, args.batch_size), start=1):
        indices = np.arange(start_index, min(start_index + args.batch_size, limit), dtype=np.int64)
        states = _array_stack(hf_dataset.select(indices.tolist())["state"])
        query = np.asarray(
            query_index_matrix(indices, end_by_frame[indices], train_config.model.action_horizon), dtype=np.int64
        )
        actions = _array_stack(hf_dataset.select(query.reshape(-1).tolist())["actions"])
        actions = actions.reshape(len(indices), train_config.model.action_horizon, -1)
        actions[..., :6] -= states[:, None, :6]
        running["state"].update(states)
        running["actions"].update(actions)
        del states, query, actions, indices
        if args.release_every and batch_number % args.release_every == 0:
            _release_allocator_memory()
        if batch_number == 1 or batch_number % args.sample_every == 0 or batch_number == args.max_batches:
            samples.append(
                {
                    "batch": batch_number,
                    "rss_bytes": _rss_bytes(),
                    "pyarrow_allocated_bytes": int(pa.total_allocated_bytes()),
                }
            )

    statistics = {key: value.get_statistics() for key, value in running.items()}
    report = {
        "status": "pass",
        "phase": "n1b_memory_diagnostic",
        "config_name": args.config_name,
        "mode": "state_action_only",
        "dataset_length": len(dataset),
        "batch_size": args.batch_size,
        "completed_batches": args.max_batches,
        "processed_frames": limit,
        "release_every": args.release_every,
        "elapsed_seconds": time.time() - start,
        "probe_equivalence": probe_results,
        "samples": samples,
        "rss_slope_bytes_per_batch": _slope(samples),
        "rss_growth_bytes": samples[-1]["rss_bytes"] - samples[0]["rss_bytes"],
        "final_stats_shapes": {
            key: {field: list(np.asarray(getattr(value, field)).shape) for field in ("mean", "std", "q01", "q99")}
            for key, value in statistics.items()
        },
        "identities": identities,
        "canonical_target_absent": not args.expected_target.exists(),
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
