"""Run the authorized frozen E2 main-200 comparison and stop before E3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import experiment_identity
import run_e1_development as common


AUTH_TOKEN = "authorized-e2-main-400"


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return experiment_identity.sha256_file(path)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--e0-manifest", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-openpi-root", required=True, type=Path)
    parser.add_argument("--model-python", required=True, type=Path)
    parser.add_argument("--libero-python", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--selected-model-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-identity", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--egl-vendor-file", required=True, type=Path)
    parser.add_argument("--expected-policy-openpi-commit", required=True)
    parser.add_argument("--expected-libero-openpi-commit", required=True)
    parser.add_argument("--expected-libero-commit", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-task-output-bytes", type=int, default=178_571)
    parser.add_argument("--max-stage-output-bytes", type=int, default=100_000_000)
    parser.add_argument("--server-wait-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.authorization != AUTH_TOKEN:
        parser.error("explicit E2 authorization token is required")
    return args


def _main_entries(manifest: dict) -> list[dict]:
    entries = sorted((row for row in manifest["entries"] if row.get("split") == "main"), key=lambda row: (row["suite"], row["task_id"], row["initial_state_index"]))
    if len(entries) != 200 or len({(row["suite"], row["task_id"], row["initial_state_index"]) for row in entries}) != 200:
        raise ValueError("E2 requires exactly 200 unique frozen main entries")
    return entries


def _models(args: argparse.Namespace, lock: dict) -> list[dict]:
    if lock.get("stage") != "E1-selection-lock" or lock.get("e1_completed") != 280 or lock.get("e2_authorized") is not False:
        raise ValueError("selection lock is not a valid pre-E2 lock")
    stable = dict(lock); claimed = stable.pop("lock_identity_sha256", None)
    if claimed != _canonical(stable):
        raise ValueError("selection lock identity mismatch")
    selected = json.loads(args.selected_model_manifest.read_text())
    if selected.get("model_mode") != "base_plus_adapter" or selected.get("adapter_identity_sha256") != lock.get("adapter_identity_sha256"):
        raise ValueError("selected model manifest does not match E1 lock")
    adapter = args.adapter_root / "step-{:08d}".format(lock["step"])
    if not (adapter / "manifest.json").is_file():
        raise FileNotFoundError(adapter)
    return [
        {"label": "base", "adapter": None, "model_manifest": args.base_manifest},
        {"label": "pure_lora_step_{:08d}".format(lock["step"]), "adapter": adapter, "model_manifest": args.selected_model_manifest},
    ]


def _validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("JAX preallocation must be disabled")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("E2 CUDA mapping differs from selected physical GPU")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.physical_gpu) or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("E2 EGL mapping mismatch")
    for value in vars(args).values():
        if isinstance(value, Path) and not value.exists() and value != args.attempt_dir:
            raise FileNotFoundError(value)


def _row(label: str, entry: dict, result: dict, path: Path) -> dict:
    return {"model_label": label, "suite": entry["suite"], "task_id": entry["task_id"], "initial_state_index": entry["initial_state_index"], "outcome": "success" if result.get("success") else "policy_failure", "result_path": str(path), "result_sha256": _sha256(path)}


def main() -> int:
    args = _parse(); _validate_environment(args)
    if args.attempt_dir.exists(): raise FileExistsError(args.attempt_dir)
    lock = json.loads(args.selection_lock.read_text()); entries = _main_entries(json.loads(args.e0_manifest.read_text())); models = _models(args, lock)
    if socket.socket().connect_ex(("127.0.0.1", args.port)) == 0: raise RuntimeError("E2 port already in use")
    common._atomic_new(args.attempt_dir / "run_manifest.json", {"schema_version": 1, "stage": "E2-main-400", "authorization": AUTH_TOKEN, "selection_lock_sha256": _sha256(args.selection_lock), "selection_lock_identity_sha256": lock["lock_identity_sha256"], "entries": entries, "models": [{"label": model["label"], "adapter": str(model["adapter"]) if model["adapter"] else None, "model_manifest_sha256": _sha256(model["model_manifest"])} for model in models], "next_stage_started": False})
    common._atomic_replace(args.attempt_dir / "status.json", {"status": "starting", "completed": 0, "total": 400, "next_stage_started": False})
    completed = 0
    server = None
    try:
        for model in models:
            directory = args.attempt_dir / model["label"]; directory.mkdir(parents=True)
            env = dict(os.environ); env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "PYTHONPATH": os.pathsep.join([str(args.server.parent), str(args.libero_openpi_root / "third_party/libero"), str(args.libero_openpi_root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])})
            server_command = [str(args.model_python), str(args.server), "--openpi-root", str(args.openpi_root), "--base-params", str(args.base_params), "--base-manifest", str(args.base_manifest), "--golden", str(args.golden), "--norm-stats", str(args.norm_stats), "--config-patch-sha256", args.config_identity.read_text().strip(), "--model-manifest", str(model["model_manifest"]), "--port", str(args.port)]
            if model["adapter"]: server_command += ["--adapter", str(model["adapter"])]
            with (directory / "server.log").open("xb") as log:
                server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, env=env)
                common._wait_port(args.port, server, args.server_wait_seconds)
                for entry in entries:
                    task_dir = directory / "{}-task-{:02d}-state-{:02d}".format(entry["suite"], entry["task_id"], entry["initial_state_index"])
                    result_path = task_dir / "task_{:02d}_init_{:02d}_result.json".format(entry["task_id"], entry["initial_state_index"])
                    command = [str(args.libero_python), str(args.evaluator), "--suite", entry["suite"], "--task-id", str(entry["task_id"]), "--initial-states", str(entry["initial_state_index"]), "--max-episodes", "1", "--seed", "7", "--host", "127.0.0.1", "--port", str(args.port), "--server-wait-seconds", "30", "--replan-steps", "5", "--physical-gpu", str(args.physical_gpu), "--mujoco-egl-device-id", str(args.physical_gpu), "--egl-vendor-file", str(args.egl_vendor_file), "--gpu-sample-interval", "1", "--max-gpu-memory-fraction", "0.90", "--max-baseline-gpu-utilization", "100", "--max-baseline-gpu-memory-fraction", "1", "--policy-config", "pi0_libero_pure_lora", "--openpi-root", str(args.libero_openpi_root), "--checkpoint-dir", str(args.base_params), "--base-manifest", str(args.base_manifest), "--norm-stats", str(args.norm_stats), "--model-manifest", str(model["model_manifest"]), "--task-state-manifest", str(args.e0_manifest), "--evaluation-split", "main", "--expected-openpi-commit", args.expected_libero_openpi_commit, "--expected-libero-commit", args.expected_libero_commit, "--output-dir", str(task_dir), "--max-output-bytes", str(args.max_task_output_bytes), "--no-save-video"]
                    if model["adapter"]: command += ["--adapter-dir", str(model["adapter"])]
                    done = subprocess.run(command, env=env, text=True, capture_output=True)
                    task_dir.mkdir(parents=True, exist_ok=True); (task_dir / "evaluator.stdout.log").write_text(done.stdout); (task_dir / "evaluator.stderr.log").write_text(done.stderr)
                    if done.returncode != 0 or not result_path.exists(): raise RuntimeError("infrastructure evaluator failure at {}".format(entry))
                    common._append_jsonl(args.attempt_dir / "main_results.jsonl", _row(model["label"], entry, json.loads(result_path.read_text()), result_path))
                    completed += 1
                    if common._directory_bytes(args.attempt_dir) > args.max_stage_output_bytes: raise RuntimeError("E2 stage output budget exceeded")
                    common._atomic_replace(args.attempt_dir / "status.json", {"status": "running", "completed": completed, "total": 400, "model_label": model["label"], "next_stage_started": False})
            common._append_jsonl(args.attempt_dir / "server_cleanup.jsonl", {"model_label": model["label"], **common._stop_owned_process(server)}); server = None
        summary = {"schema_version": 1, "stage": "E2-main-400", "status": "pass", "completed": completed, "selection_lock_sha256": _sha256(args.selection_lock), "next_stage_started": False}; summary["summary_identity_sha256"] = _canonical(summary)
        common._atomic_new(args.attempt_dir / "summary.json", summary); common._atomic_replace(args.attempt_dir / "status.json", {"status": "pass", "completed": 400, "total": 400, "next_stage_started": False}); return 0
    except BaseException as error:
        cleanup = common._stop_owned_process(server)
        common._atomic_replace(args.attempt_dir / "status.json", {"status": "fail", "completed": completed, "total": 400, "reason": repr(error), "next_stage_started": False}); raise


if __name__ == "__main__":
    raise SystemExit(main())
