"""CPU-only contracts for the non-authorizing T1 engineering execution package."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time

import experiment_identity as identity

STAGE = "T1-engineering-100-200"
IDENTITIES = {"model", "dataset", "norm", "golden", "config", "split"}
TOOL_ROLES = {"runner", "orchestrator", "launcher", "gpu_guard", "storage_guard", "preflight", "builder", "finalizer", "contract", "resume_sequence", "s1d_reference_runner", "verifier", "experiment_identity", "write_evidence_manifest", "s1d_verifier", "adapter_artifact"}
INPUT_ROLES = {"freeze_package", "s1d_acceptance_report", "model_manifest", "golden_manifest"}
BOUNDS = {"timeout_seconds": 6840, "heartbeat_timeout_seconds": 1800, "sample_interval_seconds": 1, "term_grace_seconds": 60, "kill_grace_seconds": 10, "max_retries": 0, "retry_return_codes": [], "retry_delay_seconds": 5, "max_log_bytes": 10_000_000, "log_backups": 2}
GUARD_BOUNDS = {"storage_timeout_seconds": 6600, "storage_term_grace_seconds": 45, "storage_kill_grace_seconds": 5, "gpu_prelaunch_seconds": 300, "gpu_runtime_seconds": 6000, "gpu_term_grace_seconds": 15, "gpu_kill_grace_seconds": 5}
PREFLIGHT_GATE = {"free_memory_percent_strictly_greater_than": 15.0, "utilization_percent_strictly_less_than": 95.0, "min_mem_available_bytes": 64_000_000_000, "max_load1_per_cpu": 0.90}
GUARD_THRESHOLDS = {"pause_utilization_percent": 95, "resume_utilization_percent": 85, "pause_free_memory_percent": 15, "resume_free_memory_percent": 20, "resume_consecutive_samples": 5, "terminate_free_memory_percent": 10}


def monitor_roots(source_root: str) -> list[str]:
    return [source_root, "/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora", "/home/wengzr/projects/openpi-lora-cache", "/home/wengzr/projects/openpi-lora-runs", "/home/wengzr/.cache/openpi", "/home/wengzr/.cache/uv"]


def expected_environment(runner_path: str, attempt_dir: str, selected: int) -> dict:
    return {
        "CUDA_VISIBLE_DEVICES": str(selected),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": str(Path(runner_path).parent.parent),
        "HF_HOME": "/home/wengzr/projects/openpi-lora-cache/huggingface",
        "HF_DATASETS_CACHE": "/home/wengzr/projects/openpi-lora-cache/huggingface/datasets",
        "HF_LEROBOT_HOME": "/home/wengzr/projects/openpi-lora-cache/huggingface/lerobot",
        "JAX_COMPILATION_CACHE_DIR": str(Path(attempt_dir) / "jax-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def signed(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = identity.canonical_sha256(result)
    return result


def verify_identity(value: dict, field: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if claimed != identity.canonical_sha256(unsigned):
        raise ValueError(f"{field} mismatch")


def absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"normalized absolute path required: {value}")
    return path


def new_path(path: Path) -> None:
    # lexists also rejects dangling symlinks; existing symlink parents are unsafe.
    if os.path.lexists(path):
        raise FileExistsError(path)
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"symlink parent forbidden: {path}")


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path: Path, value: dict) -> None:
    """Publish complete JSON without an overwrite race (link is exclusive)."""
    new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def current_host() -> dict:
    return {"hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid finite number: {name}")
    return float(value)


def validate_preflight(report: dict, *, now_epoch: float, now_monotonic: float, host_identity: dict) -> dict:
    verify_identity(report, "preflight_identity_sha256")
    if report.get("schema_version") != 1 or report.get("status", "pass") != "pass" or report.get("reasons") or report.get("errors"):
        raise ValueError("preflight schema or failure reasons are invalid")
    if report.get("host_identity") != host_identity or not host_identity.get("boot_id") or not host_identity.get("hostname"):
        raise ValueError("preflight is not from this host and boot")
    if report.get("launch_gate") != PREFLIGHT_GATE or report.get("guard_thresholds") != GUARD_THRESHOLDS or report.get("jax_preallocation_required") is not False:
        raise ValueError("preflight gate or guard policy changed")
    samples = report.get("samples", [])
    if type(report.get("sample_count")) is not int or report["sample_count"] != len(samples) or not 30 <= len(samples) <= 120:
        raise ValueError("preflight requires 30..120 complete samples")
    started = number(report.get("collection_started_epoch_seconds"), "start epoch")
    finished = number(report.get("collection_finished_epoch_seconds"), "finish epoch")
    if not 29 <= finished - started <= 180 or not 0 <= now_epoch - finished <= 120:
        raise ValueError("preflight wall-clock span or freshness failed")
    previous = None
    gpu_identity = None
    for ordinal, sample in enumerate(samples):
        stamp = number(sample.get("monotonic_seconds"), "sample monotonic")
        if sample.get("sample_index") != ordinal or (previous is not None and not 0.9 <= stamp - previous <= 10):
            raise ValueError("preflight sample order or spacing invalid")
        previous = stamp
        gpus = sample.get("gpus", [])
        if len(gpus) != 2:
            raise ValueError("every sample must contain both physical GPUs")
        keys = []
        for gpu in gpus:
            index = gpu.get("index")
            uuid = gpu.get("uuid")
            if type(index) is not int or index < 0 or not isinstance(uuid, str) or not uuid.startswith("GPU-"):
                raise ValueError("invalid physical GPU identity")
            keys.append((index, uuid))
            total = number(gpu.get("memory_total_mib"), "total VRAM")
            used = number(gpu.get("memory_used_mib"), "used VRAM")
            free = number(gpu.get("free_memory_percent"), "free VRAM")
            util = number(gpu.get("utilization_percent"), "utilization")
            if total <= 0 or not 0 <= used <= total or not 0 <= util <= 100 or abs(free - 100 * (total - used) / total) > 1e-6:
                raise ValueError("GPU numeric content is inconsistent")
        keys = sorted(keys)
        if len({key[0] for key in keys}) != 2 or len({key[1] for key in keys}) != 2 or (gpu_identity is not None and gpu_identity != keys):
            raise ValueError("GPU identities changed during preflight")
        gpu_identity = keys
        host = sample.get("host", {})
        cpu = host.get("logical_cpu_count")
        memory = number(host.get("mem_available_bytes"), "available RAM")
        load = number(host.get("load1"), "load1")
        ratio = number(host.get("load1_per_cpu"), "load per CPU")
        if type(cpu) is not int or cpu <= 0 or load < 0 or abs(ratio - load / cpu) > 1e-9 or memory <= 64_000_000_000 or ratio >= .90:
            raise ValueError("CPU/RAM sample is unsafe or inconsistent")
    first_mono, last_mono = samples[0]["monotonic_seconds"], samples[-1]["monotonic_seconds"]
    if not 29 <= last_mono - first_mono <= 180 or not 0 <= now_monotonic - last_mono <= 120:
        raise ValueError("preflight monotonic span or freshness failed")
    if abs((now_epoch - finished) - (now_monotonic - last_mono)) > 15 or abs((finished - started) - (last_mono - first_mono)) > 15:
        raise ValueError("preflight wall and monotonic clocks disagree")
    candidates = [gpu for gpu in samples[-1]["gpus"] if gpu["free_memory_percent"] > 15 and gpu["utilization_percent"] < 95]
    if not candidates:
        raise ValueError("no GPU clears the strict launch gate")
    selected = min(candidates, key=lambda gpu: (gpu["utilization_percent"], -gpu["free_memory_percent"], gpu["index"]))
    if type(report.get("selected_physical_gpu")) is not int or report["selected_physical_gpu"] != selected["index"] or report.get("selected_gpu_uuid") != selected["uuid"]:
        raise ValueError("preflight GPU selection differs from the validated policy")
    return selected


def validate_paths(paths: dict) -> None:
    required = {"python", "tmux", "attempt_dir", "current_json", "stage_plan", "launch_review", "model_manifest", "golden_manifest", "s1d_acceptance_report", "freeze_package", "checkpoint_root", "adapter_root", "preflight", "progress", "result", "loader_receipt", "rng_receipt", "composition_receipt", "gpu_events", "storage_guard_dir", "checkpoint_200", "adapter_200", "adapter_200_receipt"}
    if set(paths) != required:
        raise ValueError(f"path keys must be exactly {sorted(required)}")
    objects = {key: absolute(value) for key, value in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("bound paths alias each other")
    attempt = objects["attempt_dir"]
    if objects["current_json"].parent != attempt.parent:
        raise ValueError("current.json must share the stage parent")
    for key in ("progress", "result", "loader_receipt", "rng_receipt", "composition_receipt", "gpu_events", "storage_guard_dir"):
        if objects[key].parent != attempt:
            raise ValueError(f"{key} must be a direct child of attempt")
    for key in ("stage_plan", "launch_review", "preflight"):
        if attempt in objects[key].parents or objects[key] == attempt:
            raise ValueError("review inputs cannot pre-create the attempt directory")
    for key, expected in (("checkpoint_200", objects["checkpoint_root"] / "200"), ("adapter_200", objects["adapter_root"] / "step-00000200"), ("adapter_200_receipt", objects["adapter_root"] / "step-00000200.verified.json")):
        if objects[key] != expected:
            raise ValueError(f"invalid target path: {key}")


def validate_collisions(paths: dict) -> None:
    validate_paths(paths)
    for key in ("attempt_dir", "checkpoint_200", "adapter_200", "adapter_200_receipt"):
        new_path(Path(paths[key]))
    current = Path(paths["current_json"])
    if current.is_symlink():
        raise ValueError("current pointer cannot be a symlink")
    if current.exists() and json.loads(current.read_text()).get("status") in {"starting", "running", "retrying", "terminating"}:
        raise RuntimeError("another stage attempt is active")


def validate_file_bindings(bindings: dict) -> None:
    for name, record in bindings.items():
        path = absolute(record["path"])
        identity.validate_sha256(record["sha256"], name)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file() or path.stat().st_size > 10_000_000:
            raise ValueError(f"binding must be a regular lightweight file: {name}")
        if identity.sha256_file(path) != record["sha256"]:
            raise ValueError(f"runtime file identity mismatch: {name}")


def validate_launch_bindings(plan: dict) -> None:
    """Read only lightweight files and target existence; never launch or read tensors."""
    if plan.get("stage") != STAGE:
        return
    verify_identity(plan, "plan_identity_sha256")
    if plan.get("execution_authorized") is not False or plan.get("next_stage_auto_start") is not False:
        raise ValueError("T1 plans must remain non-authorizing")
    validate_collisions(plan["paths"])
    validate_file_bindings(plan["tool_bindings"])
    validate_file_bindings(plan["input_files"])
    preflight_path = Path(plan["preflight"]["path"])
    validate_file_bindings({"preflight": plan["preflight"]})
    report = json.loads(preflight_path.read_text())
    selected = validate_preflight(report, now_epoch=time.time(), now_monotonic=time.monotonic(), host_identity=current_host())
    expected = expected_environment(plan["tool_bindings"]["runner"]["path"], plan["paths"]["attempt_dir"], selected["index"])
    if plan.get("monitor_roots") != monitor_roots(plan["source"]["root"]):
        raise ValueError("runtime storage monitor roots changed")
    if plan["command"][:len(expected) + 1] != ["/usr/bin/env", *[f"{key}={value}" for key, value in expected.items()]]:
        raise ValueError("exact child command does not pin the required environment")
    if plan["environment"] != expected or plan["selected_gpu_uuid"] != selected["uuid"] or plan["selected_physical_gpu"] != selected["index"]:
        raise ValueError("launch GPU mapping differs from fresh preflight")
