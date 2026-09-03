#!/usr/bin/env python3
"""Audit LIBERO inputs to canonical normalization without writing norm stats."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


REPO_ID = "physical-intelligence/libero"
SUITE_BLOCKS = {
    "libero_10": 0,
    "libero_goal": 10,
    "libero_object": 20,
    "libero_spatial": 30,
}
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_query_indices(frame: int, episode_start: int, episode_end: int, horizon: int) -> list[int]:
    if not episode_start <= frame < episode_end:
        raise ValueError("frame is outside episode")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return [max(episode_start, min(episode_end - 1, frame + delta)) for delta in range(horizon)]


def expected_padding(frame: int, episode_start: int, episode_end: int, horizon: int) -> list[bool]:
    if not episode_start <= frame < episode_end:
        raise ValueError("frame is outside episode")
    return [frame + delta < episode_start or frame + delta >= episode_end for delta in range(horizon)]


def select_probe_frames(
    tasks: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    episode_ranges: Mapping[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    task_by_index = {int(record["task_index"]): str(record["task"]) for record in tasks}
    if sorted(task_by_index) != list(range(40)):
        raise ValueError("expected exactly task indices 0..39")
    task_index_by_text = {text: index for index, text in task_by_index.items()}
    if len(task_index_by_text) != len(task_by_index):
        raise ValueError("task descriptions must be unique")

    first_episode_for_task: dict[int, int] = {}
    episode_task_index: dict[int, int] = {}
    for record in episodes:
        episode_index = int(record["episode_index"])
        episode_tasks = record["tasks"]
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1:
            raise ValueError(f"episode {episode_index} must have exactly one task")
        task_text = str(episode_tasks[0])
        if task_text not in task_index_by_text:
            raise ValueError(f"episode {episode_index} references unknown task")
        task_index = task_index_by_text[task_text]
        episode_task_index[episode_index] = task_index
        first_episode_for_task.setdefault(task_index, episode_index)

    if sorted(episode_ranges) != list(range(len(episodes))):
        raise ValueError("episode ranges must be contiguous from zero")

    selections: dict[int, dict[str, Any]] = {}

    def add_frame(frame: int, episode_index: int, reason: str) -> None:
        start, end = episode_ranges[episode_index]
        if not start <= frame < end:
            raise ValueError("selected frame outside episode")
        record = selections.setdefault(
            frame,
            {
                "frame_index": frame,
                "episode_index": episode_index,
                "task_index": episode_task_index[episode_index],
                "episode_start": start,
                "episode_end": end,
                "reasons": [],
            },
        )
        record["reasons"].append(reason)

    for episode_index, label in ((0, "first_episode"), (len(episodes) - 1, "last_episode")):
        start, end = episode_ranges[episode_index]
        for frame, position in ((start, "start"), ((start + end - 1) // 2, "middle"), (end - 1, "end")):
            add_frame(frame, episode_index, f"{label}:{position}")

    for suite, task_index in SUITE_BLOCKS.items():
        if task_index not in first_episode_for_task:
            raise ValueError(f"no episode for representative {suite} task {task_index}")
        episode_index = first_episode_for_task[task_index]
        start, end = episode_ranges[episode_index]
        add_frame(start, episode_index, f"{suite}:representative_start")
        add_frame(end - 1, episode_index, f"{suite}:representative_end")

    return [selections[index] for index in sorted(selections)]


def _array_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    finite = bool(np.all(np.isfinite(array))) if numeric else None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "min": float(np.min(array)) if numeric and array.size else None,
        "max": float(np.max(array)) if numeric and array.size else None,
    }


def _transform_identity(transform_groups: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    payload = {
        name: [f"{type(item).__module__}.{type(item).__qualname__}:{item!r}" for item in items]
        for name, items in transform_groups.items()
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"groups": payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _environment_checks(
    environment: Mapping[str, str], hf_home: pathlib.Path, hf_datasets_cache: pathlib.Path, hf_lerobot_home: pathlib.Path
) -> dict[str, bool]:
    def exact(name: str, expected: pathlib.Path) -> bool:
        value = environment.get(name)
        return value is not None and pathlib.Path(value).resolve(strict=False) == expected

    return {
        "hf_home_exact": exact("HF_HOME", hf_home),
        "hf_datasets_cache_exact": exact("HF_DATASETS_CACHE", hf_datasets_cache),
        "hf_lerobot_home_exact": exact("HF_LEROBOT_HOME", hf_lerobot_home),
        "hf_hub_offline": environment.get("HF_HUB_OFFLINE") == "1",
        "hf_datasets_offline": environment.get("HF_DATASETS_OFFLINE") == "1",
        "transformers_offline": environment.get("TRANSFORMERS_OFFLINE") == "1",
        "proxies_unset": all(not environment.get(name) for name in PROXY_VARIABLES),
        "gpu_hidden": environment.get("CUDA_VISIBLE_DEVICES") == "",
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    dataset_root = args.dataset_root.resolve(strict=True)
    hf_home = args.hf_home.resolve(strict=True)
    hf_datasets_cache = args.hf_datasets_cache.resolve(strict=True)
    hf_lerobot_home = args.hf_lerobot_home.resolve(strict=True)
    openpi_worktree = args.openpi_worktree.resolve(strict=True)
    expected_asset_path = args.expected_asset_path.resolve(strict=False)
    if expected_asset_path.exists():
        raise FileExistsError(f"canonical asset target already exists: {expected_asset_path}")

    d1c_report = json.loads(args.d1c_report.read_text())
    environment_checks = _environment_checks(os.environ, hf_home, hf_datasets_cache, hf_lerobot_home)
    source_identities = {
        str(path.resolve(strict=True)): _sha256_file(path.resolve(strict=True)) for path in args.source_file
    }

    import numpy as np
    import openpi
    import openpi.shared.normalize as normalize
    import openpi.training.config as config
    import openpi.training.data_loader as data_loader
    import openpi.transforms as transforms

    openpi_file = pathlib.Path(openpi.__file__).resolve(strict=True)
    train_config = config.get_config(args.config_name)
    resolved_asset_path = pathlib.Path(
        str(train_config.data.resolve_asset_path(train_config.assets_dirs))
    ).resolve(strict=False)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    transform_identity = _transform_identity(
        {
            "repack": data_config.repack_transforms.inputs,
            "data": data_config.data_transforms.inputs,
        }
    )
    dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    base_dataset = getattr(dataset, "_dataset", dataset)

    tasks = _load_jsonl(dataset_root / "meta/tasks.jsonl")
    episodes = _load_jsonl(dataset_root / "meta/episodes.jsonl")
    episode_ranges = {
        episode_index: (
            int(np.asarray(base_dataset.episode_data_index["from"][episode_index]).item()),
            int(np.asarray(base_dataset.episode_data_index["to"][episode_index]).item()),
        )
        for episode_index in range(len(episodes))
    }
    probes = select_probe_frames(tasks, episodes, episode_ranges)
    transform = transforms.compose(
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    probe_results: list[dict[str, Any]] = []
    all_probe_checks: list[bool] = []

    for probe in probes:
        frame = int(probe["frame_index"])
        episode_index = int(probe["episode_index"])
        start_frame, end_frame = episode_ranges[episode_index]
        raw = dataset[frame]
        raw_episode_index = int(np.asarray(raw["episode_index"]).item())
        raw_task_index = int(np.asarray(raw["task_index"]).item())
        query_indices, padding = base_dataset._get_query_indices(frame, episode_index)
        action_query_indices = [int(value) for value in query_indices["actions"]]
        expected_indices = expected_query_indices(frame, start_frame, end_frame, args.action_horizon)
        expected_pad = expected_padding(frame, start_frame, end_frame, args.action_horizon)
        actual_pad = np.asarray(raw["actions_is_pad"], dtype=bool).tolist()
        raw_actions = np.asarray(raw["actions"])
        raw_state = np.asarray(raw["state"])
        selected_actions = base_dataset.hf_dataset.select(expected_indices)["actions"]
        expected_raw_actions = np.stack([np.asarray(value) for value in selected_actions])
        transformed = transform(copy.deepcopy(raw))
        transformed_state = np.asarray(transformed["state"])
        transformed_actions = np.asarray(transformed["actions"])
        expected_delta_actions = raw_actions.copy()
        expected_delta_actions[..., :6] -= raw_state[..., :6]
        prompt = transformed.get("prompt")
        prompt_text = str(prompt.item()) if hasattr(prompt, "item") else str(prompt)
        task_text = str(tasks[raw_task_index]["task"])
        image_summaries = {key: _array_summary(value) for key, value in transformed["image"].items()}
        padded_positions = [index for index, is_pad in enumerate(expected_pad) if is_pad]
        pad_repeats_endpoint = all(
            np.array_equal(raw_actions[position], expected_raw_actions[-1]) for position in padded_positions
        )
        checks = {
            "episode_index_exact": raw_episode_index == episode_index,
            "task_index_exact": raw_task_index == int(probe["task_index"]),
            "query_indices_exact": action_query_indices == expected_indices,
            "query_indices_within_episode": all(start_frame <= value < end_frame for value in action_query_indices),
            "padding_exact": actual_pad == expected_pad and padding["actions_is_pad"].tolist() == expected_pad,
            "padded_actions_repeat_episode_endpoint": pad_repeats_endpoint,
            "raw_actions_match_clamped_indices": np.array_equal(raw_actions, expected_raw_actions),
            "raw_state_shape_8": raw_state.shape == (8,),
            "raw_actions_shape_50x7": raw_actions.shape == (args.action_horizon, 7),
            "transformed_state_shape_8": transformed_state.shape == (8,),
            "transformed_actions_shape_50x7": transformed_actions.shape == (args.action_horizon, 7),
            "state_finite": bool(np.all(np.isfinite(transformed_state))),
            "actions_finite": bool(np.all(np.isfinite(transformed_actions))),
            "extra_delta_transform_exact": np.allclose(
                transformed_actions, expected_delta_actions, rtol=0.0, atol=1e-6
            ),
            "prompt_matches_task": prompt_text == task_text,
            "image_keys_exact": sorted(transformed["image"]) == [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
            ],
            "image_shapes_256x256x3": all(
                summary["shape"] == [256, 256, 3] for summary in image_summaries.values()
            ),
            "images_uint8_and_finite": all(
                summary["dtype"] == "uint8" and summary["finite"] is True
                for summary in image_summaries.values()
            ),
        }
        all_probe_checks.extend(checks.values())
        running["state"].update(transformed_state)
        running["actions"].update(transformed_actions)
        probe_results.append(
            {
                **probe,
                "task": task_text,
                "query_first": action_query_indices[0],
                "query_last": action_query_indices[-1],
                "padding_count": sum(expected_pad),
                "raw_state": _array_summary(raw_state),
                "raw_actions": _array_summary(raw_actions),
                "transformed_state": _array_summary(transformed_state),
                "transformed_actions": _array_summary(transformed_actions),
                "images": image_summaries,
                "checks": checks,
            }
        )

    smoke_stats = {}
    for key, accumulator in running.items():
        statistics = accumulator.get_statistics()
        smoke_stats[key] = {
            name: np.asarray(getattr(statistics, name)).tolist()
            for name in ("mean", "std", "q01", "q99")
        }

    global_checks = {
        **{f"env_{name}": value for name, value in environment_checks.items()},
        "d1c_report_pass": d1c_report.get("status") == "pass",
        "d1c_revision_exact": d1c_report.get("revision") == args.expected_revision,
        "d1c_dataset_root_exact": pathlib.Path(str(d1c_report.get("dataset_root"))).resolve(strict=False)
        == dataset_root,
        "openpi_import_from_worktree": openpi_worktree in openpi_file.parents,
        "canonical_asset_path_exact": resolved_asset_path == expected_asset_path,
        "canonical_asset_target_absent_before_and_after": not expected_asset_path.exists(),
        "repo_id_exact": data_config.repo_id == REPO_ID,
        "normalization_is_zscore": data_config.use_quantile_norm is False,
        "normalization_input_has_no_existing_stats": data_config.norm_stats is None,
        "action_horizon_is_50": train_config.model.action_horizon == args.action_horizon == 50,
        "extra_delta_transform_enabled": bool(train_config.data.extra_delta_transform),
        "dataset_root_exact": pathlib.Path(base_dataset.root).resolve(strict=False) == dataset_root,
        "dataset_length_is_273465": len(dataset) == 273465,
        "episodes_are_1693": len(episodes) == 1693,
        "tasks_are_40": len(tasks) == 40,
        "all_probe_checks_pass": all(all_probe_checks),
        "four_suite_blocks_represented": all(
            any(reason.startswith(f"{suite}:") for result in probe_results for reason in result["reasons"])
            for suite in SUITE_BLOCKS
        ),
        "first_and_last_episode_edges_represented": all(
            any(reason == target for result in probe_results for reason in result["reasons"])
            for target in ("first_episode:start", "first_episode:end", "last_episode:start", "last_episode:end")
        ),
    }
    report = {
        "status": "pass" if all(global_checks.values()) else "fail",
        "phase": "n1a_norm_input_smoke",
        "config_name": args.config_name,
        "repo_id": data_config.repo_id,
        "expected_revision": args.expected_revision,
        "dataset_root": str(dataset_root),
        "dataset_length": len(dataset),
        "canonical_asset_path": str(resolved_asset_path),
        "canonical_asset_created": expected_asset_path.exists(),
        "openpi_import": str(openpi_file),
        "openpi_worktree": str(openpi_worktree),
        "source_identities": source_identities,
        "d1c_report": {"path": str(args.d1c_report), "sha256": _sha256_file(args.d1c_report)},
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("lerobot", "datasets", "huggingface-hub", "pyarrow", "numpy")
        },
        "transform_identity": transform_identity,
        "probe_count": len(probe_results),
        "probes": probe_results,
        "small_sample_stats": smoke_stats,
        "small_sample_stats_are_not_canonical": True,
        "checks": global_checks,
        "elapsed_seconds": time.monotonic() - start,
    }
    _atomic_json(args.output, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi0_libero_pure_lora")
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--hf-home", required=True, type=pathlib.Path)
    parser.add_argument("--hf-datasets-cache", required=True, type=pathlib.Path)
    parser.add_argument("--hf-lerobot-home", required=True, type=pathlib.Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--d1c-report", required=True, type=pathlib.Path)
    parser.add_argument("--expected-asset-path", required=True, type=pathlib.Path)
    parser.add_argument("--openpi-worktree", required=True, type=pathlib.Path)
    parser.add_argument("--source-file", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    report = run_audit(args)
    print(json.dumps({"status": report["status"], "probe_count": report["probe_count"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
