"""Rerun only the invalid Base half of E2 on its frozen main-200 split.

This is a recovery controller, not a new model-selection or evaluation stage.
It preserves the completed LoRA half of the parent E2 attempt and records why
the parent Base half was invalid before producing a disjoint replacement.
"""
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


AUTH_TOKEN = "authorized-e2-base-recovery-200"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--prior-e2-attempt", required=True, type=Path)
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
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--egl-vendor-file", required=True, type=Path)
    parser.add_argument("--expected-policy-openpi-commit", required=True)
    parser.add_argument("--expected-libero-openpi-commit", required=True)
    parser.add_argument("--expected-libero-commit", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-task-output-bytes", type=int, default=178_571)
    parser.add_argument("--max-stage-output-bytes", type=int, default=50_000_000)
    parser.add_argument("--server-wait-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.authorization != AUTH_TOKEN:
        parser.error("explicit Base-recovery authorization token is required")
    return args


def _validate_parent(prior_attempt: Path) -> dict[str, object]:
    summary_path = prior_attempt / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "E2-main-400" or summary.get("status") != "pass" or summary.get("completed") != 400:
        raise ValueError("prior E2 attempt is not a completed E2-main-400 attempt")
    results = sorted((prior_attempt / "base").glob("**/*_result.json"))
    if len(results) != 200:
        raise ValueError("prior E2 Base half does not contain exactly 200 result files")
    parsed = [json.loads(path.read_text()) for path in results]
    if not all(not row.get("success") and row.get("policy_requests") == 0 for row in parsed):
        raise ValueError("prior E2 Base half is not the recorded zero-request invalid run")
    return {
        "prior_e2_summary_sha256": experiment_identity.sha256_file(summary_path),
        "prior_base_result_count": len(results),
        "prior_base_failure_count": sum(not row.get("success") for row in parsed),
        "prior_base_policy_request_count": sum(int(row.get("policy_requests", 0)) for row in parsed),
    }


def _validate_lock(selection_lock: Path, selected_model_manifest: Path) -> dict[str, object]:
    lock = json.loads(selection_lock.read_text())
    stable = dict(lock)
    claimed = stable.pop("lock_identity_sha256", None)
    if lock.get("stage") != "E1-selection-lock" or lock.get("e1_completed") != 280 or lock.get("e2_authorized") is not False or claimed != e2._canonical(stable):
        raise ValueError("selection lock is not the frozen pre-E2 lock")
    selected = json.loads(selected_model_manifest.read_text())
    if selected.get("model_mode") != "base_plus_adapter" or selected.get("adapter_identity_sha256") != lock.get("adapter_identity_sha256"):
        raise ValueError("selected model manifest no longer matches the E1 selection lock")
    return lock


def main() -> int:
    args = _parse()
    e2._validate_environment(args)
    if args.attempt_dir.exists():
        raise FileExistsError(args.attempt_dir)
    parent = _validate_parent(args.prior_e2_attempt)
    lock = _validate_lock(args.selection_lock, args.selected_model_manifest)
    entries = e2._main_entries(json.loads(args.e0_manifest.read_text()))
    if socket.socket().connect_ex(("127.0.0.1", args.port)) == 0:
        raise RuntimeError("Base-recovery port already in use")
    common._atomic_new(args.attempt_dir / "run_manifest.json", {
        "schema_version": 1,
        "stage": "E2-base-recovery-200",
        "authorization": AUTH_TOKEN,
        "recovery_reason": "prior Base server retained nnx.eval_shape placeholders",
        "prior_e2_attempt": str(args.prior_e2_attempt),
        "selection_lock_sha256": experiment_identity.sha256_file(args.selection_lock),
        "selection_lock_identity_sha256": lock["lock_identity_sha256"],
        "entries": entries,
        "model": {"label": "base", "adapter": None, "model_manifest_sha256": experiment_identity.sha256_file(args.base_manifest)},
        "next_stage_started": False,
        **parent,
    })
    common._atomic_replace(args.attempt_dir / "status.json", {"status": "starting", "completed": 0, "total": 200, "next_stage_started": False})
    server = None
    completed = 0
    try:
        directory = args.attempt_dir / "base"
        directory.mkdir(parents=True)
        env = dict(os.environ)
        env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "PYTHONPATH": os.pathsep.join([str(args.server.parent), str(args.libero_openpi_root / "third_party/libero"), str(args.libero_openpi_root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])})
        server_command = [str(args.model_python), str(args.server), "--openpi-root", str(args.openpi_root), "--base-params", str(args.base_params), "--base-manifest", str(args.base_manifest), "--golden", str(args.golden), "--norm-stats", str(args.norm_stats), "--config-patch-sha256", args.config_identity.read_text().strip(), "--model-manifest", str(args.base_manifest), "--port", str(args.port)]
        with (directory / "server.log").open("xb") as log:
            server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, env=env)
            common._wait_port(args.port, server, args.server_wait_seconds)
            for entry in entries:
                task_dir = directory / "{}-task-{:02d}-state-{:02d}".format(entry["suite"], entry["task_id"], entry["initial_state_index"])
                result_path = task_dir / "task_{:02d}_init_{:02d}_result.json".format(entry["task_id"], entry["initial_state_index"])
                command = [str(args.libero_python), str(args.evaluator), "--suite", entry["suite"], "--task-id", str(entry["task_id"]), "--initial-states", str(entry["initial_state_index"]), "--max-episodes", "1", "--seed", "7", "--host", "127.0.0.1", "--port", str(args.port), "--server-wait-seconds", "30", "--replan-steps", "5", "--physical-gpu", str(args.physical_gpu), "--mujoco-egl-device-id", str(args.physical_gpu), "--egl-vendor-file", str(args.egl_vendor_file), "--gpu-sample-interval", "1", "--max-gpu-memory-fraction", "0.90", "--max-baseline-gpu-utilization", "100", "--max-baseline-gpu-memory-fraction", "1", "--policy-config", "pi0_libero_pure_lora", "--openpi-root", str(args.libero_openpi_root), "--checkpoint-dir", str(args.base_params), "--base-manifest", str(args.base_manifest), "--norm-stats", str(args.norm_stats), "--model-manifest", str(args.base_manifest), "--task-state-manifest", str(args.e0_manifest), "--evaluation-split", "main", "--expected-openpi-commit", args.expected_libero_openpi_commit, "--expected-libero-commit", args.expected_libero_commit, "--output-dir", str(task_dir), "--max-output-bytes", str(args.max_task_output_bytes), "--no-save-video"]
                done = subprocess.run(command, env=env, text=True, capture_output=True)
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "evaluator.stdout.log").write_text(done.stdout)
                (task_dir / "evaluator.stderr.log").write_text(done.stderr)
                if done.returncode != 0 or not result_path.exists():
                    raise RuntimeError("infrastructure evaluator failure at {}".format(entry))
                common._append_jsonl(args.attempt_dir / "main_results.jsonl", e2._row("base", entry, json.loads(result_path.read_text()), result_path))
                completed += 1
                if common._directory_bytes(args.attempt_dir) > args.max_stage_output_bytes:
                    raise RuntimeError("Base-recovery stage output budget exceeded")
                common._atomic_replace(args.attempt_dir / "status.json", {"status": "running", "completed": completed, "total": 200, "model_label": "base", "next_stage_started": False})
        common._append_jsonl(args.attempt_dir / "server_cleanup.jsonl", {"model_label": "base", **common._stop_owned_process(server)})
        server = None
        summary = {"schema_version": 1, "stage": "E2-base-recovery-200", "status": "pass", "completed": completed, "selection_lock_sha256": experiment_identity.sha256_file(args.selection_lock), "next_stage_started": False}
        summary["summary_identity_sha256"] = e2._canonical(summary)
        common._atomic_new(args.attempt_dir / "summary.json", summary)
        common._atomic_replace(args.attempt_dir / "status.json", {"status": "pass", "completed": 200, "total": 200, "next_stage_started": False})
        return 0
    except BaseException as error:
        common._stop_owned_process(server)
        common._atomic_replace(args.attempt_dir / "status.json", {"status": "fail", "completed": completed, "total": 200, "reason": repr(error), "next_stage_started": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
