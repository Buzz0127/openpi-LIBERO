#!/usr/bin/env python3
"""Independently accept a completed T1 100->200 attempt, without loading arrays.

This is a terminal, read-only verifier for a future authorized run. Unit tests
use tiny fake artifacts. It never imports JAX, OpenPI, numpy, or a data loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any


SHARED_CACHE_ROOTS = {
    "/home/wengzr/projects/openpi-lora-cache",
    "/home/wengzr/.cache/openpi", "/home/wengzr/.cache/uv",
}
IDENTITY_KEYS = {"model", "dataset", "norm", "golden", "config", "split"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"not a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _signed(value: dict[str, Any], key: str, *, newline: bool = False) -> str:
    stable = {name: item for name, item in value.items() if name != key}
    expected = hashlib.sha256((json.dumps(stable, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest() if newline else _canonical(stable)
    _require(_digest(value.get(key)) and expected == value[key], f"{key} mismatch")
    return value[key]


def _file_record(path: Path) -> dict[str, Any]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(all(getattr(before, key) == getattr(after, key) for key in fields), f"file changed while hashing: {path}")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _sha256(path: Path) -> str:
    return _file_record(path)["sha256"]


def _safe_relative(value: object) -> Path:
    _require(isinstance(value, str) and bool(value), "empty artifact relative path")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and str(path) == value, f"unsafe relative path: {value}")
    return path


def artifact_manifest(root: Path) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), f"invalid artifact root: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"symlink in artifact: {path}")
        _require(not any(mark in path.name for mark in (".partial", ".tmp", "tmp-", ".orbax-checkpoint-tmp")), f"temporary artifact: {path}")
        if path.is_dir():
            continue
        files[str(path.relative_to(root))] = _file_record(path)
    _require(bool(files), f"empty artifact: {root}")
    stable = {"root": str(root.resolve()), "files": files, "file_count": len(files),
              "total_bytes": sum(item["bytes"] for item in files.values())}
    return {**stable, "artifact_tree_sha256": _canonical(stable)}


def _old_files_retained(old: dict[str, Any], current: dict[str, Any], prefix: str) -> None:
    _signed(old, "artifact_tree_sha256")
    _require(old["root"] == current["root"], "old artifact root changed")
    _require(old["file_count"] == len(old["files"]) and old["total_bytes"] == sum(v["bytes"] for v in old["files"].values()), "old artifact manifest totals mismatch")
    _require(bool(old["files"]), "empty old artifact manifest")
    for name, record in old["files"].items():
        _safe_relative(name)
        _require(current["files"].get(name) == record, f"step-100 file changed or deleted: {name}")
    old_subtree = {name for name in old["files"] if name.startswith(prefix)}
    new_subtree = {name for name in current["files"] if name.startswith(prefix)}
    _require(old_subtree == new_subtree, "step-100 subtree gained unexpected files")


def _parameter_map(value: object) -> dict[str, str]:
    _require(isinstance(value, dict) and len(value) == 70, "parameter hash map must contain all 70 leaves")
    _require(all(isinstance(path, str) and path and _digest(digest) for path, digest in value.items()), "invalid parameter path/hash")
    return value


def _process_absent(identifier: int, *, group: bool = False) -> bool:
    _require(type(identifier) is int and identifier > 1, "invalid owned process identity")
    try:
        (os.killpg if group else os.kill)(identifier, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _terminal_process_audit(pids: list[int], tmux: str, session_name: str) -> dict[str, Any]:
    rows = []
    for pid in sorted(set(pids)):
        row = {"pid": pid, "pid_absent": _process_absent(pid), "pgid_absent": _process_absent(pid, group=True)}
        _require(row["pid_absent"] and row["pgid_absent"], f"owned process/group still exists: {pid}")
        rows.append(row)
    completed = subprocess.run([tmux, "has-session", "-t", session_name], capture_output=True, check=False, timeout=10)
    # A missing binary/permission failure must never masquerade as no session.
    _require(completed.returncode == 1, "tmux session remains or tmux query failed")
    receipt = {"schema_version": 1, "sample_time_epoch_seconds": time.time(), "owned_processes": rows,
               "tmux_session": session_name, "tmux_absent": True}
    return {**receipt, "process_audit_identity_sha256": _canonical(receipt)}


def _output_manifest(attempt: Path, required: set[Path]) -> dict[str, str]:
    values = {}
    for line in (attempt / "output-files.sha256").read_text().splitlines():
        claimed, separator, name = line.partition("  ")
        _require(bool(separator) and _digest(claimed), "invalid A2 output manifest line")
        path = _safe_relative(name)
        _require(name not in values and name != "output-files.sha256", "duplicate/self A2 output manifest entry")
        candidate = attempt / path
        _require(candidate.resolve().is_relative_to(attempt), "A2 output escaped attempt")
        _require(_sha256(candidate) == claimed, f"A2 output hash mismatch: {name}")
        values[name] = claimed
    _require(bool(values), "empty A2 output manifest")
    _require({str(path.relative_to(attempt)) for path in required} <= set(values), "A2 output manifest omits required evidence")
    return values


def _verify_preflight(plan: dict[str, Any], run: dict[str, Any], result: dict[str, Any], guard: dict[str, Any]) -> None:
    binding = plan["preflight"]
    _require(_sha256(Path(binding["path"])) == binding["sha256"], "preflight file hash mismatch")
    preflight = _load(Path(binding["path"]))
    _signed(preflight, "preflight_identity_sha256")
    selected = plan["selected_physical_gpu"]
    uuid = plan["selected_gpu_uuid"]
    _require(type(selected) is int and selected >= 0 and preflight.get("selected_physical_gpu") == selected and preflight.get("selected_gpu_uuid") == uuid, "preflight GPU identity mismatch")
    _require(plan.get("environment") == {"CUDA_VISIBLE_DEVICES": str(selected), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, "single-GPU environment mismatch")
    _require(result.get("physical_gpu") == selected and result.get("jax_device_count") == 1 and guard.get("physical_gpu") == selected, "runner/guard GPU mapping mismatch")
    samples = preflight.get("samples", [])
    _require(type(preflight.get("sample_count")) is int and preflight["sample_count"] == len(samples) and 30 <= len(samples) <= 120, "preflight requires 30 dual-GPU samples")
    previous_stamp = None
    gpu_ids = None
    for index, sample in enumerate(samples):
        stamp = sample.get("monotonic_seconds")
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and sample.get("sample_index") == index, "preflight sample timestamp/order invalid")
        _require(previous_stamp is None or .9 <= stamp - previous_stamp <= 10, "preflight sample spacing invalid")
        previous_stamp = stamp
        gpus = sample.get("gpus", [])
        _require(len(gpus) == 2, "preflight sample omitted a GPU")
        keys = {(gpu["index"], gpu["uuid"]) for gpu in gpus}
        _require(len(keys) == 2 and (gpu_ids is None or gpu_ids == keys) and (selected, uuid) in keys, "preflight physical GPU identities changed")
        gpu_ids = keys
        host = sample.get("host", {})
        cpus = host.get("logical_cpu_count")
        load = host.get("load1")
        ratio = host.get("load1_per_cpu")
        memory = host.get("mem_available_bytes")
        _require(type(cpus) is int and cpus > 0 and all(type(v) in (int, float) and math.isfinite(v) for v in (load, ratio, memory)) and load >= 0 and abs(ratio - load / cpus) < 1e-9 and ratio < .90 and memory > 64_000_000_000, "preflight CPU/RAM unsafe")
    selected_row = next(gpu for gpu in samples[-1]["gpus"] if gpu["index"] == selected)
    _require(selected_row["free_memory_percent"] > 15 and selected_row["utilization_percent"] < 95, "selected GPU does not meet strict safety gate")
    duration = samples[-1]["monotonic_seconds"] - samples[0]["monotonic_seconds"]
    _require(29 <= duration <= 180, "preflight was not a 30-second observation")
    finished = preflight.get("collection_finished_epoch_seconds")
    started = preflight.get("collection_started_epoch_seconds")
    launched = run.get("start_time_epoch_seconds")
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in (finished, started, launched)) and 29 <= finished - started <= 180 and 0 <= launched - finished <= 120, "preflight was stale at A2 launch")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    attempt = args.attempt_dir.resolve()
    plan_path = args.stage_plan.resolve()
    _require(_sha256(plan_path) == args.expected_stage_plan_sha256, "stage plan file hash mismatch")
    plan = _load(plan_path)
    _signed(plan, "plan_identity_sha256")
    freeze = _load(args.freeze_package)
    _require(_signed(freeze, "package_identity_sha256", newline=True) == args.expected_freeze_identity, "freeze identity mismatch")
    _require(set(freeze["identities"]) == IDENTITY_KEYS and all(_digest(v) for v in freeze["identities"].values()), "frozen identities missing")
    _require(plan["identities"] == freeze["identities"], "plan experiment identity mismatch")
    _require(plan["source"]["head"] == freeze["source"]["head"], "plan source HEAD mismatch")
    _require(plan["attempt_dir"] == str(attempt), "plan attempt binding mismatch")
    _require(plan.get("execution_authorized") is False and plan.get("next_stage_auto_start") is False, "plan authorization boundary changed")
    _require((plan["segment_start"], plan["segment_end"], plan["train_seed"], plan["eval_seed"], plan["expected_final_committed_step"]) == (100, 200, 42, 7, 200), "T1 segment/seeds changed")
    _require(plan.get("freeze_package_identity_sha256") == args.expected_freeze_identity, "plan freeze binding mismatch")
    for role, path in (("freeze_package", args.freeze_package), ("s1d_acceptance_report", args.s1d_acceptance_report), ("golden_manifest", args.golden_manifest)):
        _require(plan["input_files"][role] == {"path": str(path.resolve()), "sha256": _sha256(path)}, f"plan input file mismatch: {role}")
    for name, binding in {**plan["input_files"], **plan["tool_bindings"]}.items():
        path = Path(binding["path"])
        _require(path.is_absolute() and path.is_file() and path.stat().st_size <= 10_000_000 and _sha256(path) == binding["sha256"], f"pinned lightweight file changed: {name}")
    model = _load(Path(plan["input_files"]["model_manifest"]["path"]))
    _require(_signed(model, "model_identity_sha256") == freeze["identities"]["model"], "base model manifest identity mismatch")
    paths = plan["paths"]
    required_names = ("result", "loader_receipt", "rng_receipt", "composition_receipt", "progress", "gpu_events")
    evidence_paths = {name: Path(paths[name]) for name in required_names}
    for path in evidence_paths.values():
        _require(path.is_absolute() and path.resolve().is_relative_to(attempt), "plan evidence path escapes attempt")
    guard_dir = Path(paths["storage_guard_dir"])
    _require(guard_dir.is_absolute() and guard_dir.resolve().is_relative_to(attempt), "storage guard path escapes attempt")
    required = set(evidence_paths.values()) | {guard_dir / "exit_status.json", guard_dir / "run_manifest.json", guard_dir / "samples.jsonl"}
    _require(required <= {Path(path) for path in plan["required_outputs"]}, "plan omits required T1 evidence")
    required |= {attempt / name for name in ("status.json", "summary.json", "run_manifest.json", "exit_code.txt", "launcher_receipt.json")}
    output_files = _output_manifest(attempt, required)

    acceptance = _load(args.s1d_acceptance_report)
    resume_input = freeze["resume_input"]
    _require(_signed(acceptance, "report_identity_sha256") == resume_input["acceptance_report_identity_sha256"] and acceptance["status"] == "pass", "S1d acceptance identity mismatch")
    _require(acceptance["checkpoint_artifact"]["artifact_tree_sha256"] == resume_input["checkpoint_tree_sha256"], "frozen starting checkpoint identity mismatch")
    _require(acceptance["adapter_identity_sha256"] == resume_input["adapter_identity_sha256"], "frozen starting adapter identity mismatch")
    _require(str(args.checkpoint_dir.resolve()) == resume_input["checkpoint_root"], "frozen checkpoint root mismatch")
    _require(str(args.adapter_root.resolve()) == acceptance["adapter_artifact"]["root"], "frozen adapter root mismatch")
    _require(_sha256(args.golden_manifest) == freeze["identities"]["golden"], "independent Golden manifest identity mismatch")
    golden = _load(args.golden_manifest)
    golden_entries = golden["entries"]
    golden_map = {entry["path"]: entry for entry in golden_entries}
    _require(len(golden_entries) == len(golden_map) == 20 and golden["review_invariants"]["total_param_leaf_count"] == 70, "Golden manifest leaf invariants changed")
    golden_paths = set(golden_map)

    result = _load(evidence_paths["result"])
    _require(result.get("status") == "pass" and result.get("stage") == "T1-resume-engineering", "runner did not pass T1 engineering stage")
    _require(result.get("identities") == freeze["identities"] and result.get("source_head") == freeze["source"]["head"], "runner experiment/source identity mismatch")
    _require((result.get("segment_start"), result.get("segment_end"), result.get("train_seed"), result.get("eval_seed")) == (100, 200, 42, 7), "runner segment/seeds mismatch")
    _require((result.get("batch_size"), result.get("num_workers"), result.get("shuffle")) == (1, 0, True), "runner loader configuration mismatch")
    _require(result.get("next_stage_started") is False and result.get("old_checkpoint_deleted") is False, "runner deletion/next-stage violation")
    metrics = result.get("metrics_trace")
    _require(isinstance(metrics, list) and len(metrics) == 100, "expected 100 independent metric records")
    _require(all(isinstance(row, dict) and "loss" in row and all(type(v) in (int, float) and math.isfinite(v) for v in row.values()) for row in metrics), "non-finite or invalid metric record")
    before = _parameter_map(result.get("parameter_hashes_before"))
    after = _parameter_map(result.get("parameter_hashes_after"))
    _require(set(before) == set(after) and golden_paths <= set(before), "parameter path universe mismatch")
    changed = {path for path in before if before[path] != after[path]}
    _require(changed == golden_paths and len(set(before) - golden_paths) == 50, "Golden/non-Golden parameter changes violated")

    loader = _load(evidence_paths["loader_receipt"])
    _require(result.get("loader_receipt_sha256") == _sha256(evidence_paths["loader_receipt"]), "loader receipt hash mismatch")
    _require((loader.get("restored_step"), loader.get("skipped_batch_count"), loader.get("first_resumed_batch_index")) == (100, 100, 100), "loader continuity mismatch")
    _require(loader.get("position_verified") is True and _digest(loader.get("reference_batch_sha256")) and loader["reference_batch_sha256"] == loader.get("resumed_batch_sha256"), "loader fingerprint mismatch")
    rng = _load(evidence_paths["rng_receipt"])
    _require(result.get("rng_receipt_sha256") == _sha256(evidence_paths["rng_receipt"]), "RNG receipt hash mismatch")
    _require((rng.get("seed"), rng.get("restored_step"), rng.get("replayed_split_count")) == (42, 100, 100), "RNG continuity count mismatch")
    for reference, actual in (("reference_train_key_sha256", "resumed_train_key_sha256"), ("reference_first_step_key_sha256", "first_step_key_sha256")):
        _require(_digest(rng.get(reference)) and rng[reference] == rng.get(actual), "RNG reference/replay mismatch")

    checkpoint = artifact_manifest(args.checkpoint_dir)
    adapter = artifact_manifest(args.adapter_root)
    _old_files_retained(acceptance["checkpoint_artifact"], checkpoint, "100/")
    _old_files_retained(acceptance["adapter_artifact"], adapter, "step-00000100/")
    _require(adapter["files"]["step-00000100.verified.json"]["sha256"] == resume_input["restore_receipt_sha256"], "frozen step-100 restore receipt changed")
    _require({p.name for p in args.checkpoint_dir.iterdir()} == {"100", "200"}, "checkpoint steps must be exactly 100 and 200")
    _require({p.name for p in args.adapter_root.iterdir()} == {"step-00000100", "step-00000100.verified.json", "step-00000200", "step-00000200.verified.json"}, "unexpected adapter root entries")
    _require(checkpoint == result.get("checkpoint_artifact_after") and adapter == result.get("adapter_artifact_after"), "saved artifact tree changed after runner completion")
    commit = _load(args.checkpoint_dir / "200/_CHECKPOINT_METADATA")
    _require(type(commit.get("commit_timestamp_nsecs")) is int and commit["commit_timestamp_nsecs"] > 0, "step-200 atomic commit metadata missing")
    for group in ("params", "train_state"):
        for suffix in ("_METADATA", "manifest.ocdbt"):
            _require(checkpoint["files"].get(f"200/{group}/{suffix}", {}).get("bytes", 0) > 0, "step-200 incomplete checkpoint metadata")
        _require(any(name.startswith(f"200/{group}/") and "/d/" in name and row["bytes"] > 0 for name, row in checkpoint["files"].items()), "step-200 incomplete checkpoint chunks")
    norm_files = [row for name, row in checkpoint["files"].items() if name.startswith("200/assets/") and name.endswith("/norm_stats.json")]
    _require(len(norm_files) == 1 and norm_files[0]["sha256"] == freeze["identities"]["norm"], "step-200 canonical normalization mismatch")
    adapter_dir = args.adapter_root / "step-00000200"
    adapter_manifest = _load(adapter_dir / "manifest.json")
    adapter_id = _signed(adapter_manifest, "adapter_identity_sha256")
    _require(adapter_manifest.get("artifact_type") == "pi0_pure_lora_adapter_only" and (adapter_manifest.get("train_step"), adapter_manifest.get("train_seed")) == (200, 42), "adapter training identity mismatch")
    entries = adapter_manifest["entries"]
    _require(len(entries) == 20 and {entry["path"] for entry in entries} == golden_paths, "adapter paths differ from independent Golden")
    declared_files = {"manifest.json"}
    for entry in entries:
        file = _safe_relative(entry["file"])
        _require(str(file) not in declared_files, "duplicate adapter file")
        declared_files.add(str(file))
        _require(_sha256(adapter_dir / file) == entry["file_sha256"] and entry["array_sha256"] == after[entry["path"]], "adapter file/array hash mismatch")
        _require(entry["shape"] == golden_map[entry["path"]]["shape"] and entry["parameter_count"] == golden_map[entry["path"]]["parameter_count"] and entry["dtype"] == "float32", "adapter Golden shape/dtype mismatch")
    _require(declared_files == {str(p.relative_to(adapter_dir)) for p in adapter_dir.rglob("*") if p.is_file()}, "unexpected adapter files")
    adapter_ids = adapter_manifest["identities"]
    _require(adapter_ids == model["identities"], "adapter identities differ from frozen base model")
    _require(all(adapter_ids.get(key) == freeze["identities"][name] for key, name in (("golden_manifest_sha256", "golden"), ("norm_stats_sha256", "norm"), ("config_patch_sha256", "config"))), "adapter experiment identities mismatch")
    receipt_path = args.adapter_root / "step-00000200.verified.json"
    receipt = _load(receipt_path)
    _require(receipt == result.get("checkpoint_restore_receipt"), "restore receipt disk/result mismatch")
    for key in ("checkpoint_restore_succeeded", "parameter_tree_shape_dtype_equal", "optimizer_tree_shape_dtype_equal", "all_parameter_values_equal_after_restore", "all_optimizer_values_equal_after_restore", "adapter_values_equal_after_restore"):
        _require(receipt.get(key) is True, f"restore receipt failed: {key}")
    _require((receipt.get("parameter_leaf_count"), receipt.get("optimizer_leaf_count"), receipt.get("save_step")) == (70, 42, 200), "restore leaf/step mismatch")
    _require(receipt.get("automatic_pruning_enabled") is False and receipt.get("old_checkpoint_deletion_performed") is False, "pruning/deletion receipt violation")
    _require(receipt.get("adapter_identity_sha256") == adapter_id, "restore adapter identity mismatch")
    composition = _load(evidence_paths["composition_receipt"])
    _require(result.get("composition_receipt_sha256") == _sha256(evidence_paths["composition_receipt"]), "composition receipt hash mismatch")
    _require(composition.get("status") == "pass" and composition.get("save_step") == 200 and composition.get("adapter_identity_sha256") == adapter_id and composition.get("parameter_hashes") == after, "base plus adapter composition does not equal all final parameters")

    status = _load(attempt / "status.json")
    summary = _load(attempt / "summary.json")
    run = _load(attempt / "run_manifest.json")
    progress = _load(evidence_paths["progress"])
    _signed(summary, "summary_identity_sha256")
    _require((attempt / "exit_code.txt").read_text().strip() == "0", "A2 exit code not zero")
    for value in (status, summary, run):
        _require(value.get("attempt_id") == attempt.name, "A2 attempt identity mismatch")
    for value in (status, run):
        _require(value.get("stage_plan_sha256") == args.expected_stage_plan_sha256 and value.get("stage_plan_identity_sha256") == plan["plan_identity_sha256"], "A2 plan identity mismatch")
        _require(value.get("identities") == freeze["identities"] and value.get("source") == plan["source"], "A2 experiment/source mismatch")
    _require(status.get("status") == summary.get("status") == "pass" and status.get("exit_reason") == summary.get("reason_code") == "completed", "A2 not completed successfully")
    _require(summary.get("retry_count") == 0 and summary.get("next_stage_started") is False and run.get("next_stage_auto_start") is False, "A2 retry/next-stage violation")
    _require(run.get("bounds") == plan["bounds"] and run.get("command") == plan["command"], "A2 command/bounds mismatch")
    _require(0 < plan["bounds"]["timeout_seconds"] <= 7200 and plan["bounds"]["max_retries"] == 0, "A2 wall time/retry bounds violated")
    _require(all(run.get("child_environment_overrides", {}).get(key) == "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")), "A2 offline environment missing")
    _require({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"} <= set(run.get("child_environment_unset", [])), "A2 proxy removal missing")
    for value in (status, progress, summary.get("last_progress", {})):
        _require((value.get("current_step"), value.get("last_committed_step")) == (200, 200), "A2 committed-step mismatch")
    for path in plan["required_outputs"]:
        _require(summary.get("output_sha256", {}).get(path) == _sha256(Path(path)), "A2 required output hash mismatch")
    child_runs = summary.get("child_runs")
    _require(isinstance(child_runs, list) and len(child_runs) == 1, "A2 requires exactly one child run")
    child = child_runs[0]
    _require(child.get("child_pid") == child.get("child_pgid") and child.get("returncode") == 0 and child.get("wait_reaped") is True and child.get("term_sent") is False and child.get("kill_sent") is False, "A2 child did not exit and reap cleanly")
    _require(child.get("group_exit_confirmed") is True, "A2 owned group exit not confirmed")
    launcher = _load(attempt / "launcher_receipt.json")
    sequences = launcher.get("heartbeat_sequences", [])
    _require(launcher.get("attempt_dir") == str(attempt) and len(sequences) >= 2 and all(type(v) is int for v in sequences) and all(b > a for a, b in zip(sequences, sequences[1:])) and launcher.get("tmux_session_alive") is True and launcher.get("launcher_detaches_after_verification") is True, "launcher heartbeat/detachment evidence invalid")
    max_log = plan["bounds"]["max_log_bytes"]
    backups = plan["bounds"]["log_backups"]
    logs = [p for p in attempt.iterdir() if p.is_file() and ".log" in p.name]
    _require(bool(logs) and all(p.stat().st_size <= max_log for p in logs), "A2 log bounds failed")
    for stream in ("stdout", "stderr"):
        _require(len([p for p in logs if p.name.startswith(f"child-00.{stream}.log")]) <= backups + 1, "A2 log backup bound exceeded")

    storage_run = _load(guard_dir / "run_manifest.json")
    storage_exit = _load(guard_dir / "exit_status.json")
    _require(storage_exit.get("reason_code") == "completed" and storage_exit.get("child_returncode") == 0 and storage_exit.get("wait_reaped") is True and storage_exit.get("term_sent") is False and storage_exit.get("kill_sent") is False and storage_exit.get("external_signal") is None, "storage guard failed or did not reap")
    _require(storage_exit.get("group_exit_confirmed") is True, "storage guard owned group exit not confirmed")
    _require(storage_run.get("start_new_session") is True and storage_run.get("child_pid") == storage_run.get("expected_child_pgid"), "storage guard child ownership invalid")
    _require(storage_run.get("storage", {}).get("soft_limit_bytes") == 240_000_000_000 and storage_run["storage"].get("hard_limit_bytes") == 250_000_000_000, "storage limits changed")
    final_sample = storage_exit.get("final_sample", {})
    _require(type(storage_exit.get("final_billed_lora_bytes")) is int and 0 <= storage_exit["final_billed_lora_bytes"] < 240_000_000_000, "storage above soft stop")
    delta = final_sample.get("positive_stage_delta_bytes")
    _require(type(delta) is int and 0 <= delta <= args.max_stage_increment_bytes, "stage storage increment bound exceeded")
    deltas = {row["path"]: row["billed_delta_bytes"] for row in final_sample.get("roots", [])}
    _require(all(deltas.get(path) == 0 for path in SHARED_CACHE_ROOTS), "shared caches changed")
    gpu = [json.loads(line) for line in evidence_paths["gpu_events"].read_text().splitlines() if line.strip()]
    _require(bool(gpu) and gpu[0].get("event") == "guard_started" and gpu[-1].get("event") == "child_exited" and gpu[-1].get("return_code") == 0, "GPU guard terminal failure")
    _require(gpu[-1].get("wait_reaped") is True and gpu[-1].get("group_exit_confirmed") is True, "GPU guard owned group exit not confirmed")
    _require(not any(row.get("event") in {"monitor_error", "memory_emergency", "termination_requested", "termination_forced", "runtime_timeout"} for row in gpu), "GPU guard emergency/error")
    _require(any(row.get("event") == "gpu_sample" for row in gpu), "no GPU runtime samples")
    _verify_preflight(plan, run, result, gpu[0])
    gpu_children = [row["child_pid"] for row in gpu if row.get("event") == "child_started"]
    _require(len(gpu_children) == 1 and gpu[-1].get("child_pid") == gpu_children[0], "GPU guard child identity mismatch")
    process_audit = _terminal_process_audit([child["child_pid"], storage_run["child_pid"], gpu_children[0]], args.tmux, args.session_name)

    report = {"schema_version": 1, "stage": "T1-resume-engineering", "status": "pass", "candidate": False,
              "segment_start": 100, "segment_end": 200, "training_completed": False, "next_stage_started": False,
              "stage_plan_sha256": args.expected_stage_plan_sha256, "freeze_identity_sha256": args.expected_freeze_identity,
              "identities": freeze["identities"], "changed_golden_paths": sorted(changed), "unchanged_non_golden_leaf_count": 50,
              "checkpoint_artifact": checkpoint, "adapter_artifact": adapter, "adapter_identity_sha256": adapter_id,
              "restore_receipt_sha256": _sha256(receipt_path), "result_sha256": _sha256(evidence_paths["result"]),
              "receipt_sha256": {name: _sha256(evidence_paths[name]) for name in ("loader_receipt", "rng_receipt", "composition_receipt")},
              "output_manifest_sha256": _sha256(attempt / "output-files.sha256"),
              "verified_output_files": output_files, "process_audit": process_audit,
              "verification_scope": "file integrity and emitted full-value restore/composition evidence; no model or arrays loaded"}
    report["report_identity_sha256"] = _canonical(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attempt-dir", "stage-plan", "freeze-package", "s1d-acceptance-report", "golden-manifest", "checkpoint-dir", "adapter-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-stage-plan-sha256", "expected-freeze-identity", "session-name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tmux", default="/usr/bin/tmux")
    parser.add_argument("--max-stage-increment-bytes", type=int, default=7_000_000_000)
    args = parser.parse_args()
    _require(not args.output.exists(), "acceptance output already exists")
    report = verify(args)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # link publishes without replacing an existing output, including a raced one.
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink()
    print(json.dumps({"status": "pass", "report_identity_sha256": report["report_identity_sha256"], "report_file_sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
