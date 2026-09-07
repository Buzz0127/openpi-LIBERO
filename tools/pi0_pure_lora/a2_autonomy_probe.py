#!/usr/bin/env python3
"""Short synthetic worker used only to prove A2 detached orchestration behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


REQUIRED_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _atomic_replace(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    _atomic_replace(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--interval-seconds", required=True, type=float)
    parser.add_argument("--stdout-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 3 or args.interval_seconds <= 0 or args.stdout_bytes < 0:
        parser.error("invalid bounded probe settings")
    if args.progress.exists() or args.output.exists():
        raise FileExistsError("probe outputs must be new")
    environment_ok = all(os.environ.get(key) == value for key, value in REQUIRED_ENVIRONMENT.items())
    proxies_unset = all(key not in os.environ for key in PROXY_VARIABLES)
    if not environment_ok or not proxies_unset:
        raise RuntimeError("orchestrator did not enforce offline environment")
    if os.getpgid(0) != os.getpid():
        raise RuntimeError("probe was not launched as its own process-group leader")
    started = time.time()
    for step in range(1, args.steps + 1):
        _atomic_replace(
            args.progress,
            {
                "current_step": step,
                "last_committed_step": args.steps if step == args.steps else 0,
                "recent_metrics": {"synthetic_loss": 1.0 / step},
                "pause_count": 0,
                "resume_count": 0,
                "resource_peaks": {"synthetic_rss_bytes": 1},
            },
        )
        print(f"A2_AUTONOMY_PROBE step={step} " + ("x" * args.stdout_bytes), flush=True)
        time.sleep(args.interval_seconds)
    _atomic_new(
        args.output,
        {
            "schema_version": 1,
            "status": "pass",
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "parent_pid": os.getppid(),
            "steps": args.steps,
            "elapsed_seconds": time.time() - started,
            "offline_environment_verified": True,
            "proxies_unset": True,
            "next_stage_started": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
