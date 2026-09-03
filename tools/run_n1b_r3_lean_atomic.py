#!/usr/bin/env python3
"""Compute canonical LIBERO Pi0 norm stats through the validated lean input path."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

import diagnose_n1b_memory as diagnostic
import run_compute_norm_stats_atomic as atomic_helper


def _parse_identity(value: str) -> tuple[Path, str]:
    path_text, separator, expected = value.rpartition("=")
    if not separator or len(expected) != 64:
        raise argparse.ArgumentTypeError("identity must be PATH=64_HEX_SHA256")
    return Path(path_text), expected.lower()


def _verify_identities(values: list[tuple[Path, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, expected in values:
        resolved = path.resolve(strict=True)
        actual = diagnostic.sha256_file(resolved)
        if actual != expected:
            raise RuntimeError(f"identity mismatch for {resolved}: {actual} != {expected}")
        result[str(resolved)] = actual
    return result


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
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--expected-target", required=True, type=Path)
    parser.add_argument("--expected-dataset-length", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--identity", action="append", required=True, type=_parse_identity)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    diagnostic._assert_offline()
    target = args.expected_target.resolve()
    if target.exists():
        raise FileExistsError(f"canonical target already exists: {target}")
    if min(args.expected_dataset_length, args.batch_size, args.sample_every) < 1:
        raise ValueError("dataset length, batch size, and sampling interval must be positive")
    identities = _verify_identities(args.identity)

    import numpy as np
    import openpi.shared.normalize as normalize
    import openpi.training.config as config
    import openpi.training.data_loader as data_loader
    import openpi.transforms as transforms

    start = time.time()
    train_config = config.get_config(args.config_name)
    resolved_target = Path(str(train_config.data.resolve_asset_path(train_config.assets_dirs))).resolve()
    if resolved_target != target:
        raise RuntimeError(f"config target {resolved_target} != expected target {target}")
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
    probe_results: list[dict[str, Any]] = []
    for frame in diagnostic.PROBES:
        full = transform(copy.deepcopy(dataset[frame]))
        state = diagnostic._array_stack(hf_dataset.select([frame])["state"])[0]
        query = diagnostic.query_index_matrix(
            [frame], [end_by_frame[frame]], train_config.model.action_horizon
        )[0]
        actions = diagnostic._array_stack(hf_dataset.select(query)["actions"])
        actions[..., :6] -= state[..., :6]
        state_equal = np.allclose(state, np.asarray(full["state"]), rtol=0.0, atol=0.0)
        actions_equal = np.allclose(actions, np.asarray(full["actions"]), rtol=0.0, atol=1e-6)
        probe_results.append({"frame": frame, "state_equal": bool(state_equal), "actions_equal": bool(actions_equal)})
    if not all(item["state_equal"] and item["actions_equal"] for item in probe_results):
        raise RuntimeError(f"lean path failed full-transform equivalence: {probe_results}")

    full_batches = len(dataset) // args.batch_size
    processed_frames = full_batches * args.batch_size
    remainder_frames = len(dataset) - processed_frames
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    rss_samples: list[dict[str, int]] = []
    for batch_number, start_index in enumerate(range(0, processed_frames, args.batch_size), start=1):
        indices = np.arange(start_index, start_index + args.batch_size, dtype=np.int64)
        states = diagnostic._array_stack(hf_dataset.select(indices.tolist())["state"])
        query = np.asarray(
            diagnostic.query_index_matrix(indices, end_by_frame[indices], train_config.model.action_horizon),
            dtype=np.int64,
        )
        actions = diagnostic._array_stack(hf_dataset.select(query.reshape(-1).tolist())["actions"])
        actions = actions.reshape(args.batch_size, train_config.model.action_horizon, -1)
        actions[..., :6] -= states[:, None, :6]
        running["state"].update(states)
        running["actions"].update(actions)
        del states, query, actions, indices
        if batch_number == 1 or batch_number % args.sample_every == 0 or batch_number == full_batches:
            rss_samples.append({"batch": batch_number, "rss_bytes": diagnostic._rss_bytes()})

    norm_stats = {key: accumulator.get_statistics() for key, accumulator in running.items()}
    serialized = normalize.serialize_json(norm_stats).encode()

    def validate(path: Path) -> dict[str, Any]:
        loaded = normalize.deserialize_json(path.read_text())
        return atomic_helper._validate_stats(loaded)

    validation = atomic_helper.atomic_publish_directory(target, serialized, validate)
    published = target / "norm_stats.json"
    report = {
        "status": "pass",
        "phase": "N1b-R3",
        "config_name": args.config_name,
        "mode": "state_action_only",
        "elapsed_seconds": time.time() - start,
        "dataset_length": len(dataset),
        "batch_size": args.batch_size,
        "completed_batches": full_batches,
        "processed_frames": processed_frames,
        "source_drop_last_remainder_frames": remainder_frames,
        "action_horizon": train_config.model.action_horizon,
        "probe_equivalence": probe_results,
        "rss_samples": rss_samples,
        "rss_growth_bytes": rss_samples[-1]["rss_bytes"] - rss_samples[0]["rss_bytes"],
        "publication": {
            "target": str(target),
            "path": str(published),
            "bytes": published.stat().st_size,
            "sha256": diagnostic.sha256_file(published),
            "validation": validation,
            "atomic_protocol": "same-filesystem temporary directory, validated JSON, fsync, directory os.replace",
        },
        "identities": identities,
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
