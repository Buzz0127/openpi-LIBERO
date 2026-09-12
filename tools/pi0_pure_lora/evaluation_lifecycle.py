"""Static and fake-only lifecycle controls for the future E1 evaluator.

This module deliberately cannot launch a real policy server.  It records the
command/environment contract that E1 must bind, and its executable helper is
restricted to caller-supplied *fake* commands used by CPU tests.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Mapping, Sequence

FIXED_OPENPI_PYTHON = "/home/wengzr/projects/openpi/.venv/bin/python"


def static_wiring(*, openpi_root: str, evaluator: str, libero_python: str) -> dict:
    """Return an inert command template; placeholders are resolved only by E1."""
    if not all(isinstance(v, str) and v.startswith("/") for v in (openpi_root, evaluator, libero_python)):
        raise ValueError("all command roots must be absolute paths")
    return {
        "execution_authorized": False,
        "server": {
            "argv_template": [
                FIXED_OPENPI_PYTHON, f"{openpi_root}/scripts/serve_policy.py", "--port", "{port}",
                "policy:checkpoint", "--policy.config", "pi0_libero_pure_lora",
                "--policy.dir", "{materialized_policy_dir}",
            ],
            "required_environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "{physical_gpu}",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "identity_binding": ["base_manifest", "adapter_manifest", "norm_stats", "model_manifest"],
        },
        "websocket": {"host": "127.0.0.1", "port": "{port}", "readiness": "TCP readiness then finite-action evaluator evidence"},
        "evaluator": {
            "argv_template": [libero_python, evaluator, "--host", "127.0.0.1", "--port", "{port}",
                              "--evaluation-split", "development", "--task-state-manifest", "{e0_manifest}"],
            "identity_binding": ["model_manifest", "e0_manifest", "candidate_adapter_identity_sha256"],
        },
        "cleanup": {"ownership": "only child process groups created by the E1 supervisor", "order": ["evaluator", "server"], "signals": ["SIGTERM", "SIGKILL"], "reap_required": True},
    }


def _terminate_owned(proc: subprocess.Popen, grace_seconds: float) -> dict:
    """Terminate only the PGID created for ``proc`` and always reap it."""
    result = {"pid": proc.pid, "pgid": proc.pid, "term_sent": False, "kill_sent": False}
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        result["term_sent"] = True
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            result["kill_sent"] = True
    proc.wait(timeout=grace_seconds)
    result["returncode"] = proc.returncode
    result["reaped"] = proc.poll() is not None
    return result


def fake_lifecycle(server_command: Sequence[str], evaluator_command: Sequence[str], *, timeout_seconds: float = 1.0) -> dict:
    """CPU-test-only server/evaluator ordering and owned-PGID cleanup proof."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    server = subprocess.Popen(list(server_command), start_new_session=True)
    evaluator = None
    try:
        deadline = time.monotonic() + timeout_seconds
        while server.poll() is None and time.monotonic() < deadline:
            break  # Fake readiness is the caller's already-running sleep process.
        if server.poll() is not None:
            raise RuntimeError("fake server exited before evaluator start")
        evaluator = subprocess.Popen(list(evaluator_command), start_new_session=True)
        evaluator.wait(timeout=timeout_seconds)
        return {"server": _terminate_owned(server, timeout_seconds), "evaluator": {"pid": evaluator.pid, "pgid": evaluator.pid, "returncode": evaluator.returncode, "reaped": evaluator.poll() is not None}}
    except BaseException:
        if evaluator is not None and evaluator.poll() is None:
            _terminate_owned(evaluator, timeout_seconds)
        _terminate_owned(server, timeout_seconds)
        raise
