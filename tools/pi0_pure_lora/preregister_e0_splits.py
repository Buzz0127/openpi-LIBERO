#!/usr/bin/env python3
"""Create the outcome-blind E0 LIBERO development/main split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
STATES_PER_TASK = 50
EVAL_SEED = 7
SALT = "pi0-base-libero-pure-lora-e0-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_protocol(summary: Mapping[str, Any]) -> None:
    protocol = summary.get("protocol", {})
    expected = {
        "episodes": 2000,
        "initial_states_per_task": STATES_PER_TASK,
        "seed": EVAL_SEED,
        "suites": list(SUITES),
        "tasks_per_suite": TASKS_PER_SUITE,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"official protocol mismatch for {key}: {protocol.get(key)!r} != {value!r}")

    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(SUITES) * TASKS_PER_SUITE:
        raise ValueError("official summary must contain exactly 40 task summaries")
    observed = {(item.get("suite"), item.get("task_id")) for item in tasks}
    expected_tasks = {(suite, task_id) for suite in SUITES for task_id in range(TASKS_PER_SUITE)}
    if observed != expected_tasks:
        raise ValueError("official summary task coverage mismatch")
    if any(item.get("episodes") != STATES_PER_TASK for item in tasks):
        raise ValueError("every official task summary must cover 50 initial states")
    for item in tasks:
        failures = item.get("failed_initial_states")
        if not isinstance(failures, list) or len(failures) != len(set(failures)):
            raise ValueError("failed_initial_states must be a unique list")
        if any(not isinstance(state, int) or not 0 <= state < STATES_PER_TASK for state in failures):
            raise ValueError("official failed state index is outside [0, 50)")
        if item.get("failures") != len(failures):
            raise ValueError("official failed-state list/count mismatch")
        if item.get("successes") + item.get("failures") != STATES_PER_TASK:
            raise ValueError("official task success/failure count mismatch")


def rank_states(suite: str, task_id: int) -> List[int]:
    """Rank states without consulting any evaluation outcome."""
    return sorted(
        range(STATES_PER_TASK),
        key=lambda state: hashlib.sha256(
            f"{SALT}|seed={EVAL_SEED}|{suite}|task={task_id}|state={state}".encode("utf-8")
        ).hexdigest(),
    )


def build_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            ranked = rank_states(suite, task_id)
            entries.append(
                {
                    "split": "development",
                    "suite": suite,
                    "task_id": task_id,
                    "initial_state_index": ranked[0],
                }
            )
            entries.extend(
                {
                    "split": "main",
                    "suite": suite,
                    "task_id": task_id,
                    "initial_state_index": state,
                }
                for state in ranked[1:6]
            )
    return entries


def validate_entries(entries: Iterable[Mapping[str, Any]]) -> None:
    rows = list(entries)
    if len(rows) != 240:
        raise ValueError(f"expected 240 entries, found {len(rows)}")
    seen = set()
    for row in rows:
        key = (row.get("split"), row.get("suite"), row.get("task_id"), row.get("initial_state_index"))
        if key in seen:
            raise ValueError(f"duplicate split entry: {key}")
        seen.add(key)
        if row.get("suite") not in SUITES or row.get("split") not in {"development", "main"}:
            raise ValueError(f"invalid entry: {row}")
        if not isinstance(row.get("task_id"), int) or not 0 <= row["task_id"] < TASKS_PER_SUITE:
            raise ValueError(f"invalid task id: {row}")
        if not isinstance(row.get("initial_state_index"), int) or not 0 <= row["initial_state_index"] < STATES_PER_TASK:
            raise ValueError(f"invalid state index: {row}")

    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            dev = {row["initial_state_index"] for row in rows if row["suite"] == suite and row["task_id"] == task_id and row["split"] == "development"}
            main = {row["initial_state_index"] for row in rows if row["suite"] == suite and row["task_id"] == task_id and row["split"] == "main"}
            if len(dev) != 1 or len(main) != 5 or dev & main:
                raise ValueError(f"invalid development/main partition for {suite} task {task_id}")


def build_manifest(
    official_summary_path: pathlib.Path,
    evaluator_path: pathlib.Path,
    c0_model_manifest_path: pathlib.Path,
    openpi_commit: str,
    libero_commit: str,
) -> Dict[str, Any]:
    summary = json.loads(official_summary_path.read_text(encoding="utf-8"))
    validate_official_protocol(summary)
    entries = build_entries()
    validate_entries(entries)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "stage": "E0",
        "scope": "pi0_base LIBERO pure-LoRA evaluation preregistration",
        "selection": {
            "algorithm": "ascending_sha256_rank_v1",
            "outcome_blind": True,
            "salt": SALT,
            "eval_seed": EVAL_SEED,
            "available_initial_states_per_task": STATES_PER_TASK,
            "development_states_per_task": 1,
            "main_states_per_task": 5,
        },
        "protocol": {
            "suites": list(SUITES),
            "tasks_per_suite": TASKS_PER_SUITE,
            "development_episodes": 40,
            "main_episodes": 200,
            "training_seed": 42,
            "evaluation_seed": EVAL_SEED,
        },
        "checkpoint_selection": {
            "eligible_candidates": "T1 segment-end adapter steps fixed in the T1 execution plan before T1 starts and passing identity/restore validation",
            "candidate_steps_must_be_frozen_before_t1": True,
            "candidate_registry_may_not_change_after_first_development_episode": True,
            "selection_data": "development split only",
            "primary_metric": "development successes out of 40",
            "tie_break_order": ["lowest_train_step", "lexicographically_smallest_adapter_identity_sha256"],
            "lock_before_main": True,
            "main_split_uses": "one final comparison only; never hyperparameter or checkpoint selection",
        },
        "expansion": {
            "full_2000_evaluation": "optional after main and requires separate user authorization",
            "main_results_may_not_be_used_to_reselect_checkpoint": True,
        },
        "comparison": {
            "controlled": "Base and pure-LoRA share canonical LIBERO normalization",
            "official_reference": "existing pi0_libero seed-7 results with checkpoint-owned normalization; no rerun",
        },
        "identities": {
            "openpi_commit": openpi_commit,
            "libero_commit": libero_commit,
            "evaluator_sha256": sha256_file(evaluator_path),
            "c0_model_manifest_sha256": sha256_file(c0_model_manifest_path),
            "official_summary_sha256": sha256_file(official_summary_path),
        },
        "entries": entries,
    }
    manifest["manifest_identity_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return manifest


def atomic_write_new(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing temporary output: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-summary", type=pathlib.Path, required=True)
    parser.add_argument("--evaluator", type=pathlib.Path, required=True)
    parser.add_argument("--c0-model-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--libero-commit", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.official_summary, args.evaluator, args.c0_model_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = build_manifest(
        args.official_summary,
        args.evaluator,
        args.c0_model_manifest,
        args.openpi_commit,
        args.libero_commit,
    )
    atomic_write_new(args.output, manifest)
    print(json.dumps({"output": str(args.output), "entries": len(manifest["entries"]), "manifest_identity_sha256": manifest["manifest_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
