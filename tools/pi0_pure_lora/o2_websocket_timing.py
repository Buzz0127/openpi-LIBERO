"""Sequential, bounded normal-WebSocket timing for pre-registered O2 arms.

This deliberately does not send diagnostic noise: the pinned WebSocket protocol
accepts observations only.  Noise-based correctness is a separate direct-Policy
diagnostic and must not alter normal serving semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time
from typing import Any

import numpy as np

import build_o2_input_bundle as bundle_lib


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty timing sample")
    values = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "p95": values[round(.95 * (len(values) - 1))], "max": values[-1]}


def _load_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays: dict[str, Any] = {}
    for record in manifest["records"]:
        value = np.load(root / record["file"], allow_pickle=False)
        if str(value.dtype) != record["dtype"] or list(value.shape) != record["shape"] or bundle_lib.array_sha256(value) != record["array_sha256"]:
            raise ValueError("fixed input bundle identity mismatch: " + record["name"])
        arrays[record["name"]] = value
    observation = {
        "observation/image": arrays["image"], "observation/wrist_image": arrays["wrist_image"],
        "observation/state": arrays["state"], "prompt": manifest["prompt"],
    }
    return observation, manifest


def _action_record(response: dict[str, Any]) -> dict[str, Any]:
    actions = np.asarray(response.get("actions"))
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise ValueError("normal WebSocket policy returned invalid action chunk")
    return {"sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(), "shape": list(actions.shape), "dtype": str(actions.dtype)}


def _wait_port(port: int, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("arm server exited before readiness")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(.2)
    raise TimeoutError("arm server readiness timeout")


def _stop_owned(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    result = {"pid": process.pid, "term_sent": False, "kill_sent": False, "returncode": None}
    if process.poll() is None:
        process.terminate()
        result["term_sent"] = True
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            result["kill_sent"] = True
            process.wait(timeout=5)
    result["returncode"] = process.returncode
    return result


def _run_arm(spec: dict[str, Any], observation: dict[str, Any], attempt: Path, warmups: int, samples: int, ready_timeout: float) -> dict[str, Any]:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    name, port, command = spec["name"], int(spec["port"]), spec["command"]
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("arm command must be a string list")
    log_path = attempt / (name + ".server.log")
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_port(port, process, ready_timeout)
            client = WebsocketClientPolicy("127.0.0.1", port)
            warmup_ms, timing_ms, actions = [], [], []
            for index in range(warmups + samples):
                begin = time.monotonic()
                response = client.infer(observation)
                elapsed = (time.monotonic() - begin) * 1000
                actions.append(_action_record(response))
                (warmup_ms if index < warmups else timing_ms).append(elapsed)
            return {"name": name, "command": command, "server_ready_seconds": time.monotonic() - started, "first_request_ms": warmup_ms[0], "warmup_count": warmups, "stable_rpc_ms": _summary(timing_ms), "actions": actions}
        finally:
            cleanup = _stop_owned(process)
            with (attempt / (name + ".cleanup.json")).open("x", encoding="utf-8") as handle:
                json.dump(cleanup, handle, indent=2, sort_keys=True)
                handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--input-bundle", required=True, type=Path)
    parser.add_argument("--arm-spec", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--ready-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.attempt_dir.exists() or args.warmups <= 0 or args.samples <= 0:
        raise ValueError("attempt must be new and sample counts positive")
    observation, input_manifest = _load_bundle(args.input_bundle)
    specs = json.loads(args.arm_spec.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or [item.get("name") for item in specs] != ["official_pi0_libero_reference", "project_unmerged_bf16", "project_merged_dense_bf16"]:
        raise ValueError("arm spec must be the pre-registered sequential O2 arm order")
    if len({item.get("port") for item in specs}) != len(specs):
        raise ValueError("O2 arm ports must be distinct")
    args.attempt_dir.mkdir(parents=True)
    results = [_run_arm(spec, observation, args.attempt_dir, args.warmups, args.samples, args.ready_timeout_seconds) for spec in specs]
    output = {"schema_version": 1, "stage": "O2-normal-websocket-timing", "input_bundle_identity_sha256": input_manifest["bundle_identity_sha256"], "noise_used": False, "arms": results}
    with (args.attempt_dir / "timing.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "pass", "arms": [arm["name"] for arm in results]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
