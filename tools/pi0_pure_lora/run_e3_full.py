"""Run only the E1-locked pure-LoRA adapter on the preregistered full-2000 set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

import experiment_identity
import run_e1_development as common
import run_e2_main as e2


AUTH_TOKEN = "authorized-e3-full-2000"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--e3-manifest", required=True, type=Path)
    parser.add_argument("--e2-audit-summary", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
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
    parser.add_argument("--max-stage-output-bytes", type=int, default=250_000_000)
    parser.add_argument("--server-wait-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.authorization != AUTH_TOKEN:
        parser.error("explicit E3 authorization token is required")
    return args


def _entries(manifest: dict) -> list[dict]:
    entries = sorted((row for row in manifest["entries"] if row.get("split") == "full"), key=lambda row: (row["suite"], row["task_id"], row["initial_state_index"]))
    keys = {(row["suite"], row["task_id"], row["initial_state_index"]) for row in entries}
    if len(entries) != 2000 or len(keys) != 2000:
        raise ValueError("E3 requires exactly 2000 unique preregistered full entries")
    return entries


def _validate_selection(lock_path: Path, selected_path: Path, adapter_root: Path) -> tuple[dict, Path]:
    lock = json.loads(lock_path.read_text())
    stable = dict(lock)
    if stable.pop("lock_identity_sha256", None) != e2._canonical(stable):
        raise ValueError("selection lock identity mismatch")
    selected = json.loads(selected_path.read_text())
    if selected.get("model_mode") != "base_plus_adapter" or selected.get("adapter_identity_sha256") != lock.get("adapter_identity_sha256"):
        raise ValueError("E3 selected model differs from the E1 lock")
    adapter = adapter_root / "step-{:08d}".format(lock["step"])
    if not (adapter / "manifest.json").is_file():
        raise FileNotFoundError(adapter)
    return lock, adapter


def main() -> int:
    args = _parse()
    e2._validate_environment(args)
    if args.attempt_dir.exists():
        raise FileExistsError(args.attempt_dir)
    audit = json.loads(args.e2_audit_summary.read_text())
    if audit.get("stage") != "E2-main-paired-audit" or audit.get("status") != "pass" or audit.get("denominator") != 200:
        raise ValueError("E3 requires a completed paired E2 audit")
    entries = _entries(json.loads(args.e3_manifest.read_text()))
    lock, adapter = _validate_selection(args.selection_lock, args.selected_model_manifest, args.adapter_root)
    if socket.socket().connect_ex(("127.0.0.1", args.port)) == 0:
        raise RuntimeError("E3 port already in use")
    common._atomic_new(args.attempt_dir / "run_manifest.json", {"schema_version": 1, "stage": "E3-full-2000", "authorization": AUTH_TOKEN, "entries": entries, "model_label": "pure_lora_step_{:08d}".format(lock["step"]), "adapter": str(adapter), "selection_lock_sha256": experiment_identity.sha256_file(args.selection_lock), "selection_lock_identity_sha256": lock["lock_identity_sha256"], "e2_audit_summary_sha256": experiment_identity.sha256_file(args.e2_audit_summary), "e3_manifest_sha256": experiment_identity.sha256_file(args.e3_manifest), "next_stage_started": False})
    common._atomic_replace(args.attempt_dir / "status.json", {"status": "starting", "completed": 0, "total": 2000, "next_stage_started": False})
    server = None
    completed = 0
    try:
        directory = args.attempt_dir / "pure_lora_step_{:08d}".format(lock["step"])
        directory.mkdir(parents=True)
        env = dict(os.environ)
        env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "PYTHONPATH": os.pathsep.join([str(args.server.parent), str(args.libero_openpi_root / "third_party/libero"), str(args.libero_openpi_root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])})
        server_command = [str(args.model_python), str(args.server), "--openpi-root", str(args.openpi_root), "--base-params", str(args.base_params), "--base-manifest", str(args.base_manifest), "--golden", str(args.golden), "--norm-stats", str(args.norm_stats), "--config-patch-sha256", args.config_identity.read_text().strip(), "--adapter", str(adapter), "--model-manifest", str(args.selected_model_manifest), "--port", str(args.port)]
        with (directory / "server.log").open("xb") as log:
            server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, env=env)
            common._wait_port(args.port, server, args.server_wait_seconds)
            for entry in entries:
                task_dir = directory / "{}-task-{:02d}-state-{:02d}".format(entry["suite"], entry["task_id"], entry["initial_state_index"])
                result_path = task_dir / "task_{:02d}_init_{:02d}_result.json".format(entry["task_id"], entry["initial_state_index"])
                command = [str(args.libero_python), str(args.evaluator), "--suite", entry["suite"], "--task-id", str(entry["task_id"]), "--initial-states", str(entry["initial_state_index"]), "--max-episodes", "1", "--seed", "7", "--host", "127.0.0.1", "--port", str(args.port), "--server-wait-seconds", "30", "--replan-steps", "5", "--physical-gpu", str(args.physical_gpu), "--mujoco-egl-device-id", str(args.physical_gpu), "--egl-vendor-file", str(args.egl_vendor_file), "--gpu-sample-interval", "1", "--max-gpu-memory-fraction", "0.90", "--max-baseline-gpu-utilization", "100", "--max-baseline-gpu-memory-fraction", "1", "--policy-config", "pi0_libero_pure_lora", "--openpi-root", str(args.libero_openpi_root), "--checkpoint-dir", str(args.base_params), "--base-manifest", str(args.base_manifest), "--norm-stats", str(args.norm_stats), "--model-manifest", str(args.selected_model_manifest), "--adapter-dir", str(adapter), "--task-state-manifest", str(args.e3_manifest), "--evaluation-split", "full", "--expected-openpi-commit", args.expected_libero_openpi_commit, "--expected-libero-commit", args.expected_libero_commit, "--output-dir", str(task_dir), "--max-output-bytes", str(args.max_task_output_bytes), "--no-save-video"]
                done = subprocess.run(command, env=env, text=True, capture_output=True)
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "evaluator.stdout.log").write_text(done.stdout)
                (task_dir / "evaluator.stderr.log").write_text(done.stderr)
                if done.returncode != 0 or not result_path.exists():
                    raise RuntimeError("E3 infrastructure evaluator failure at {}".format(entry))
                common._append_jsonl(args.attempt_dir / "full_results.jsonl", e2._row("pure_lora_step_{:08d}".format(lock["step"]), entry, json.loads(result_path.read_text()), result_path))
                completed += 1
                if common._directory_bytes(args.attempt_dir) > args.max_stage_output_bytes:
                    raise RuntimeError("E3 stage output budget exceeded")
                common._atomic_replace(args.attempt_dir / "status.json", {"status": "running", "completed": completed, "total": 2000, "next_stage_started": False})
        common._append_jsonl(args.attempt_dir / "server_cleanup.jsonl", {"model_label": "pure_lora", **common._stop_owned_process(server)})
        server = None
        summary = {"schema_version": 1, "stage": "E3-full-2000", "status": "pass", "completed": completed, "selection_lock_sha256": experiment_identity.sha256_file(args.selection_lock), "next_stage_started": False}
        summary["summary_identity_sha256"] = e2._canonical(summary)
        common._atomic_new(args.attempt_dir / "summary.json", summary)
        common._atomic_replace(args.attempt_dir / "status.json", {"status": "pass", "completed": 2000, "total": 2000, "next_stage_started": False})
        return 0
    except BaseException as error:
        common._stop_owned_process(server)
        common._atomic_replace(args.attempt_dir / "status.json", {"status": "fail", "completed": completed, "total": 2000, "reason": repr(error), "next_stage_started": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
