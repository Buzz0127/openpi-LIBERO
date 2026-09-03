"""Sample both GPUs plus CPU/RAM before selecting one physical GPU for a stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import experiment_identity


def gpu_samples(command: str) -> list[dict]:
    output = subprocess.check_output([
        command, "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    records = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        total = float(fields[4])
        records.append({
            "index": int(fields[0]), "uuid": fields[1], "utilization_percent": float(fields[2]),
            "memory_used_mib": float(fields[3]), "memory_total_mib": total,
            "free_memory_percent": 100.0 * (total - float(fields[3])) / total,
        })
    if len(records) < 2:
        raise ValueError("preflight requires both physical GPUs")
    return records


def cpu_ram_sample(proc_root: Path) -> dict:
    values = {}
    for line in (proc_root / "meminfo").read_text().splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    load1 = float((proc_root / "loadavg").read_text().split()[0])
    cpu_count = max(1, len([line for line in (proc_root / "cpuinfo").read_text().splitlines() if line.startswith("processor")]))
    return {"mem_available_bytes": values["MemAvailable"], "load1": load1, "logical_cpu_count": cpu_count, "load1_per_cpu": load1 / cpu_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--min-free-memory-percent", type=float, default=15.0)
    parser.add_argument("--max-utilization-percent", type=float, default=95.0)
    parser.add_argument("--min-mem-available-bytes", type=int, default=64_000_000_000)
    parser.add_argument("--max-load1-per-cpu", type=float, default=0.90)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--test-allow-short", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 30 and not args.test_allow_short:
        parser.error("production preflight requires at least 30 samples")
    if args.interval_seconds <= 0:
        parser.error("interval must be positive")
    samples = []
    for index in range(args.samples):
        record = {"sample_index": index, "monotonic_seconds": time.monotonic(), "gpus": gpu_samples(args.nvidia_smi), "host": cpu_ram_sample(args.proc_root)}
        samples.append(record)
        if index + 1 < args.samples:
            time.sleep(args.interval_seconds)
    latest = samples[-1]
    if latest["host"]["mem_available_bytes"] <= args.min_mem_available_bytes or latest["host"]["load1_per_cpu"] >= args.max_load1_per_cpu:
        raise RuntimeError("CPU/RAM launch gate failed")
    candidates = [gpu for gpu in latest["gpus"] if gpu["free_memory_percent"] > args.min_free_memory_percent and gpu["utilization_percent"] < args.max_utilization_percent]
    if not candidates:
        raise RuntimeError("no GPU satisfies >15% free VRAM and <95% utilization")
    # Once every candidate has cleared the strict free-memory threshold,
    # prefer the least busy shared GPU. Use free memory only as a tie-breaker.
    selected = min(candidates, key=lambda gpu: (gpu["utilization_percent"], -gpu["free_memory_percent"], gpu["index"]))
    report = {
        "schema_version": 1, "sample_count": len(samples), "samples": samples,
        "selected_physical_gpu": selected["index"], "selected_gpu_uuid": selected["uuid"],
        "launch_gate": {"free_memory_percent_strictly_greater_than": args.min_free_memory_percent, "utilization_percent_strictly_less_than": args.max_utilization_percent, "min_mem_available_bytes": args.min_mem_available_bytes, "max_load1_per_cpu": args.max_load1_per_cpu},
        "jax_preallocation_required": False,
        "guard_thresholds": {"pause_utilization_percent": 95, "resume_utilization_percent": 85, "pause_free_memory_percent": 15, "resume_free_memory_percent": 20, "resume_consecutive_samples": 5, "terminate_free_memory_percent": 10},
    }
    report["preflight_identity_sha256"] = experiment_identity.canonical_sha256(report)
    experiment_identity.atomic_write_new(args.output, report)
    print(json.dumps({"selected_physical_gpu": selected["index"], "preflight_identity_sha256": report["preflight_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
