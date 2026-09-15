"""Bounded, sequential host-vs-device parameter-residency RPC benchmark.

This diagnostic is deliberately separate from E1/E2/E3.  It starts one policy
server at a time, uses the same fixed observation and explicit policy RNG root,
and records synchronized WebSocket round-trip timing plus action hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timing sample is empty")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def _fixed_observation() -> dict[str, Any]:
    return {
        "observation/image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    }


def _wait_port(port: int, server: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("policy server exited before accepting connections")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("policy server port did not become ready")


def _stop_owned_server(server: subprocess.Popen[bytes]) -> dict[str, Any]:
    result = {"pid": server.pid, "term_sent": False, "kill_sent": False, "returncode": None}
    if server.poll() is None:
        server.terminate()
        result["term_sent"] = True
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            result["kill_sent"] = True
            server.wait(timeout=5)
    result["returncode"] = server.returncode
    return result


def _action_array(response: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(response["actions"])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("policy returned invalid actions")
    return actions


def _action_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(actions.tobytes(order="C")).hexdigest()


def _action_equivalence(host_actions: list[np.ndarray], device_actions: list[np.ndarray]) -> dict[str, Any]:
    """Produce a compact, durable diagnostic before enforcing exact equality."""
    if len(host_actions) != len(device_actions):
        raise ValueError("host/device action count differs")
    host_hashes = [_action_hash(actions) for actions in host_actions]
    device_hashes = [_action_hash(actions) for actions in device_actions]
    mismatches = [index for index, pair in enumerate(zip(host_hashes, device_hashes)) if pair[0] != pair[1]]
    if not mismatches:
        return {"exact_hashes_equal": True, "mismatch_count": 0, "first_mismatch": None}
    differences = [np.abs(host_actions[index] - device_actions[index]) for index in mismatches]
    return {
        "exact_hashes_equal": False,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0],
        "max_abs_difference": float(max(np.max(value) for value in differences)),
        "mean_abs_difference": float(np.mean(np.concatenate([value.reshape(-1) for value in differences]))),
        "host_first_mismatch_hash": host_hashes[mismatches[0]],
        "device_first_mismatch_hash": device_hashes[mismatches[0]],
    }


def _server_command(args: argparse.Namespace, residency: str, port: int) -> list[str]:
    command = [
        str(args.model_python), str(args.server), "--openpi-root", str(args.openpi_root),
        "--base-params", str(args.base_params), "--base-manifest", str(args.base_manifest),
        "--golden", str(args.golden), "--norm-stats", str(args.norm_stats),
        "--config-patch-sha256", args.config_patch_sha256, "--adapter", str(args.adapter),
        "--model-manifest", str(args.model_manifest), "--port", str(port), "--rng-seed",
        str(args.rng_seed), "--parameter-residency", residency,
    ]
    return command


def _run_variant(args: argparse.Namespace, residency: str, port: int) -> dict[str, Any]:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    log_path = args.attempt_dir / (residency + ".server.log")
    started = time.monotonic()
    with log_path.open("xb") as log:
        server = subprocess.Popen(_server_command(args, residency, port), stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_port(port, server, args.server_wait_seconds)
            client = WebsocketClientPolicy("127.0.0.1", port)
            observation = _fixed_observation()
            actions = []
            warmup_ms = []
            stable_ms = []
            for index in range(args.warmups + args.samples):
                begin = time.monotonic()
                response = client.infer(observation)
                elapsed = (time.monotonic() - begin) * 1000.0
                actions.append(_action_array(response))
                (warmup_ms if index < args.warmups else stable_ms).append(elapsed)
            return {
                "residency": residency,
                "server_command": _server_command(args, residency, port),
                "server_ready_seconds": time.monotonic() - started,
                "first_request_ms": warmup_ms[0],
                "warmup_count": args.warmups,
                "stable_rpc_ms": _summary(stable_ms),
                "action_hashes": [_action_hash(value) for value in actions],
                "_action_values": actions,
            }
        finally:
            cleanup = _stop_owned_server(server)
            with (args.attempt_dir / (residency + ".cleanup.json")).open("x", encoding="utf-8") as handle:
                json.dump(cleanup, handle, indent=2, sort_keys=True)
                handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--model-python", required=True, type=Path)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--config-patch-sha256", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--host-port", type=int, required=True)
    parser.add_argument("--device-port", type=int, required=True)
    parser.add_argument("--server-wait-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.attempt_dir.exists():
        raise FileExistsError(args.attempt_dir)
    if args.warmups <= 0 or args.samples <= 0:
        parser.error("warmups and samples must be positive")
    if args.host_port == args.device_port:
        parser.error("host and device ports must differ")
    return args


def main() -> int:
    args = parse_args()
    for port in (args.host_port, args.device_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("P1 benchmark port is already in use: {}".format(port))
    args.attempt_dir.mkdir(parents=True)
    host = _run_variant(args, "host", args.host_port)
    device = _run_variant(args, "device", args.device_port)
    equivalence = _action_equivalence(host.pop("_action_values"), device.pop("_action_values"))
    result = {
        "schema_version": 1,
        "stage": "P1-parameter-residency-ab",
        "rng_control": {"policy_rng_seed": args.rng_seed, "request_order": "fixed sequential", "fresh_policy_process_per_variant": True},
        "observation": {"identity": "fixed-constant-224px-state8", "count": args.warmups + args.samples},
        "host": host,
        "device": device,
        "action_equivalence": equivalence,
    }
    with (args.attempt_dir / "benchmark.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not equivalence["exact_hashes_equal"]:
        raise RuntimeError("host/device action hashes differ; see benchmark.json")
    print(json.dumps({"status": "pass", "action_hashes_equal": True, "host_stable_rpc_ms": host["stable_rpc_ms"], "device_stable_rpc_ms": device["stable_rpc_ms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
