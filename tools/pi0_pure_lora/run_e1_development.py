"""Run the authorized, preregistered E1 development denominator only.

This runner intentionally stops after 7 x 40 development episodes.  It cannot
create a checkpoint-selection lock or begin E2.  Its parent is expected to be
the existing task-owned GPU guard; server and evaluator children inherit that
same guarded process group, while this runner reaps only PIDs it created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import evaluation_control
import experiment_identity

AUTH_TOKEN = "authorized-e1-dev-280"
EXPECTED_STEPS = (1000, 5000, 10000, 15000, 20000, 25000, 30000)


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    return experiment_identity.sha256_file(path)


def _base_identity_from_manifest(path: Path) -> str:
    """Read the C0-bound base identity, not the serialization's file hash.

    Adapter artifacts bind to the stable base-weight identity carried by the
    C0 model manifest.  Hashing the manifest JSON itself would instead bind to
    its presentation bytes and cannot compose a previously exported adapter.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("identities", {}).get("base_manifest_sha256")
    if not isinstance(identity, str) or len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("base manifest lacks a canonical base_manifest_sha256")
    return identity


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _wait_port(port: int, server: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"policy server exited before readiness: {server.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("policy server readiness timeout")


def _stop_owned_process(process: subprocess.Popen | None, grace_seconds: float = 15.0) -> dict:
    result = {"pid": process.pid if process else None, "term_sent": False, "kill_sent": False, "reaped": process is None}
    if process is None or process.poll() is not None:
        if process is not None:
            result["reaped"] = True
        return result
    process.send_signal(signal.SIGTERM); result["term_sent"] = True
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill(); result["kill_sent"] = True; process.wait(timeout=5)
    result["reaped"] = process.poll() is not None
    return result


def _registration_candidates(registration: dict) -> list[dict]:
    rows = []
    for raw in registration.get("candidates", []):
        adapter = raw.get("adapter_identity_sha256", raw.get("adapter_identity"))
        if not isinstance(raw.get("step"), int) or not isinstance(adapter, str) or len(adapter) != 64:
            raise ValueError("invalid registered candidate")
        rows.append({"step": raw["step"], "adapter_identity_sha256": adapter})
    if tuple(sorted(row["step"] for row in rows)) != EXPECTED_STEPS or len({row["adapter_identity_sha256"] for row in rows}) != 7:
        raise ValueError("candidate registry differs from frozen E1 set")
    return sorted(rows, key=lambda row: row["step"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--e1-plan", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-openpi-root", required=True, type=Path)
    parser.add_argument("--model-python", required=True, type=Path)
    parser.add_argument("--libero-python", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-identity", required=True, type=Path)
    parser.add_argument("--e0-manifest", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--egl-vendor-file", required=True, type=Path)
    parser.add_argument("--expected-openpi-commit", required=True)
    parser.add_argument("--expected-libero-openpi-commit", required=True)
    parser.add_argument("--expected-libero-commit", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-task-output-bytes", type=int, default=178_571)
    parser.add_argument("--max-stage-output-bytes", type=int, default=50_000_000)
    parser.add_argument("--server-wait-seconds", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.authorization != AUTH_TOKEN:
        parser.error("explicit E1 authorization token is required")
    if args.port < 1024 or args.port > 65535 or args.max_task_output_bytes <= 0 or args.max_stage_output_bytes <= 0:
        parser.error("invalid bounded runtime option")
    return args


def _validate_inputs(args: argparse.Namespace) -> tuple[dict, list[dict], list[dict], dict]:
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("JAX preallocation must be disabled")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("runner CUDA mapping differs from fixed physical GPU")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.physical_gpu) or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("runner EGL mapping differs from fixed physical GPU")
    for path in vars(args).values():
        if isinstance(path, Path) and not path.exists() and path != args.attempt_dir:
            raise FileNotFoundError(path)
    registration = json.loads(args.registration.read_text())
    index = json.loads(args.candidate_index.read_text())
    plan = json.loads(args.e1_plan.read_text())
    readiness = json.loads(args.readiness.read_text())
    candidates = _registration_candidates(registration)
    entries = sorted((entry for entry in plan["development_entries"]), key=lambda x: (x["suite"], x["task_id"], x["initial_state_index"]))
    if plan.get("execution_authorized") is not False or len(entries) != 40 or plan.get("total_episodes") != 280:
        raise ValueError("invalid frozen E1 plan")
    if readiness.get("status") != "prepared_not_authorized" or index.get("registration_identity_sha256") != registration.get("identity_sha256") or plan.get("registration_identity_sha256") != registration.get("identity_sha256"):
        raise ValueError("E-PREP identity binding mismatch")
    return registration, candidates, entries, plan


def _model_manifest(args: argparse.Namespace, candidate: dict, output: Path) -> dict:
    adapter = args.adapter_root / f"step-{candidate['step']:08d}"
    adapter_manifest = json.loads((adapter / "manifest.json").read_text())
    if adapter_manifest.get("adapter_identity_sha256") != candidate["adapter_identity_sha256"]:
        raise RuntimeError("candidate adapter artifact identity mismatch")
    config_hash = args.config_identity.read_text().strip()
    manifest = experiment_identity.build_model_manifest(
        model_mode="base_plus_adapter", openpi_commit=args.expected_openpi_commit,
        identities={
            "base_manifest_sha256": _base_identity_from_manifest(args.base_manifest),
            "config_patch_sha256": config_hash,
            "golden_manifest_sha256": _sha256(args.golden),
            "norm_stats_sha256": _sha256(args.norm_stats),
        }, adapter_identity_sha256=candidate["adapter_identity_sha256"], training_seed=42,
        artifact_purpose="e1_development_candidate",
    )
    _atomic_new(output, manifest)
    return manifest


def _result_row(candidate: dict, entry: dict, result: dict, result_path: Path) -> dict:
    outcome = "success" if result.get("success") else "policy_failure"
    if result.get("failure_reason") == "exception":
        outcome = "infrastructure_failure"
    return {
        "split": "development", "suite": entry["suite"], "task_id": entry["task_id"],
        "initial_state_index": entry["initial_state_index"], "candidate_step": candidate["step"],
        "outcome": outcome, "infrastructure_retry": False,
        "candidate_adapter_identity_sha256": candidate["adapter_identity_sha256"],
        "result_path": str(result_path), "result_sha256": _sha256(result_path),
    }


def main() -> int:
    args = _parse_args()
    registration, candidates, entries, plan = _validate_inputs(args)
    if args.attempt_dir.exists() and not args.resume:
        raise FileExistsError(args.attempt_dir)
    args.attempt_dir.mkdir(parents=True, exist_ok=args.resume)
    if (args.attempt_dir / "summary.json").exists():
        raise RuntimeError("completed attempt cannot be rerun")
    if socket.socket().connect_ex(("127.0.0.1", args.port)) == 0:
        raise RuntimeError("requested E1 port is already in use")
    source = {str(path): _sha256(path) for path in (args.evaluator, args.server)}
    _atomic_replace(args.attempt_dir / "run_manifest.json", {
        "schema_version": 1, "stage": "E1-development-280", "authorization": AUTH_TOKEN,
        "registration_identity_sha256": registration["identity_sha256"], "e1_plan_identity_sha256": plan["plan_identity_sha256"],
        "candidates": candidates, "development_entries": entries, "source_sha256": source,
        "physical_gpu": args.physical_gpu, "port": args.port, "video_enabled": False,
        "policy_openpi_root": str(args.openpi_root), "libero_openpi_root": str(args.libero_openpi_root),
        "next_stage_started": False,
    })
    _atomic_replace(args.attempt_dir / "status.json", {
        "status": "starting", "completed": 0, "total": 280,
        "physical_gpu": args.physical_gpu, "next_stage_started": False,
    })
    ledger: dict[str, dict] = {}
    server = None
    completed = 0
    try:
        for candidate in candidates:
            candidate_dir = args.attempt_dir / "candidates" / f"step-{candidate['step']:08d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = candidate_dir / "model_manifest.json"
            if manifest_path.exists():
                model_manifest = json.loads(manifest_path.read_text())
                if model_manifest.get("adapter_identity_sha256") != candidate["adapter_identity_sha256"]:
                    raise RuntimeError("existing model manifest identity mismatch")
            else:
                model_manifest = _model_manifest(args, candidate, manifest_path)
            adapter = args.adapter_root / f"step-{candidate['step']:08d}"
            env = dict(os.environ)
            env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "PYTHONPATH": os.pathsep.join([str(args.server.parent), str(args.libero_openpi_root / "third_party/libero"), str(args.libero_openpi_root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])})
            with (candidate_dir / "server.log").open("xb") as log:
                server = subprocess.Popen([str(args.model_python), str(args.server), "--openpi-root", str(args.openpi_root), "--base-params", str(args.base_params), "--base-manifest", str(args.base_manifest), "--golden", str(args.golden), "--norm-stats", str(args.norm_stats), "--config-patch-sha256", args.config_identity.read_text().strip(), "--adapter", str(adapter), "--model-manifest", str(manifest_path), "--port", str(args.port)], stdout=log, stderr=subprocess.STDOUT, env=env)
                _wait_port(args.port, server, args.server_wait_seconds)
                for entry in entries:
                    task_dir = candidate_dir / f"{entry['suite']}-task-{entry['task_id']:02d}-state-{entry['initial_state_index']:02d}"
                    result_path = task_dir / f"task_{entry['task_id']:02d}_init_{entry['initial_state_index']:02d}_result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                    else:
                        command = [str(args.libero_python), str(args.evaluator), "--suite", entry["suite"], "--task-id", str(entry["task_id"]), "--initial-states", str(entry["initial_state_index"]), "--max-episodes", "1", "--seed", "7", "--host", "127.0.0.1", "--port", str(args.port), "--server-wait-seconds", "30", "--replan-steps", "5", "--physical-gpu", str(args.physical_gpu), "--mujoco-egl-device-id", str(args.physical_gpu), "--egl-vendor-file", str(args.egl_vendor_file), "--gpu-sample-interval", "1", "--max-gpu-memory-fraction", "0.90", "--max-baseline-gpu-utilization", "100", "--max-baseline-gpu-memory-fraction", "1", "--policy-config", "pi0_libero_pure_lora", "--openpi-root", str(args.libero_openpi_root), "--checkpoint-dir", str(args.base_params), "--base-manifest", str(args.base_manifest), "--norm-stats", str(args.norm_stats), "--adapter-dir", str(adapter), "--model-manifest", str(manifest_path), "--task-state-manifest", str(args.e0_manifest), "--evaluation-split", "development", "--expected-openpi-commit", args.expected_libero_openpi_commit, "--expected-libero-commit", args.expected_libero_commit, "--output-dir", str(task_dir), "--max-output-bytes", str(args.max_task_output_bytes), "--no-save-video"]
                        completed_process = subprocess.run(command, env=env, text=True, capture_output=True)
                        (task_dir / "evaluator.stdout.log").parent.mkdir(parents=True, exist_ok=True)
                        (task_dir / "evaluator.stdout.log").write_text(completed_process.stdout)
                        (task_dir / "evaluator.stderr.log").write_text(completed_process.stderr)
                        if completed_process.returncode != 0 or not result_path.exists():
                            raise RuntimeError(f"infrastructure evaluator failure at {entry}: exit={completed_process.returncode}")
                        result = json.loads(result_path.read_text())
                    row = _result_row(candidate, entry, result, result_path)
                    key = evaluation_control.episode_key(row["split"], row["suite"], row["task_id"], row["initial_state_index"], row["candidate_step"])
                    if key not in ledger:
                        evaluation_control.record_episode(ledger, {key: row[key] for key in ("split", "suite", "task_id", "initial_state_index", "candidate_step", "outcome", "infrastructure_retry")})
                        _append_jsonl(args.attempt_dir / "development_results.jsonl", row)
                    if row["outcome"] == "infrastructure_failure":
                        raise RuntimeError(f"infrastructure result at {entry}")
                    completed = len(ledger)
                    if _directory_bytes(args.attempt_dir) > args.max_stage_output_bytes:
                        raise RuntimeError("E1 stage output budget exceeded")
                    _atomic_replace(args.attempt_dir / "status.json", {"status": "running", "completed": completed, "total": 280, "candidate_step": candidate["step"], "next_stage_started": False})
            cleanup = _stop_owned_process(server); server = None
            _append_jsonl(args.attempt_dir / "server_cleanup.jsonl", {"candidate_step": candidate["step"], **cleanup})
        if len(ledger) != 280:
            raise RuntimeError(f"incomplete E1 denominator: {len(ledger)}/280")
        summary = {"schema_version": 1, "stage": "E1-development-280", "status": "pass", "completed": 280, "successes": sum(row["outcome"] == "success" for row in ledger.values()), "policy_failures": sum(row["outcome"] == "policy_failure" for row in ledger.values()), "registration_identity_sha256": registration["identity_sha256"], "next_stage_started": False}
        summary["summary_identity_sha256"] = _canonical(summary)
        _atomic_new(args.attempt_dir / "summary.json", summary)
        _atomic_replace(args.attempt_dir / "status.json", {"status": "pass", "completed": 280, "total": 280, "next_stage_started": False})
        return 0
    except BaseException as error:
        cleanup = _stop_owned_process(server); server = None
        _atomic_replace(args.attempt_dir / "status.json", {"status": "fail", "completed": completed, "total": 280, "reason": repr(error), "cleanup": cleanup, "next_stage_started": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
