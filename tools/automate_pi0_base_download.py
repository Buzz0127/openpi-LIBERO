#!/usr/bin/env python3
"""Run the guarded pi0_base params download as one auditable automation stage."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--downloader", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--monitor-root", action="append", required=True, type=Path)
    parser.add_argument("--existing-billed-bytes", required=True, type=int)
    parser.add_argument("--soft-limit-bytes", required=True, type=int)
    parser.add_argument("--hard-limit-bytes", required=True, type=int)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--proxy", help="Explicit HTTP(S) proxy; omit for direct server access")
    parser.add_argument("--timeout-seconds", type=int, default=86_400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.python, args.guard, args.downloader, args.source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-20260902T-B1-DOWNLOAD-", dir=args.evidence_root))
    manifest_copy = attempt / "source_manifest.json"
    shutil.copyfile(args.source_manifest, manifest_copy)
    manifest = json.loads(manifest_copy.read_text(encoding="utf-8"))
    if not 10_000_000_000 <= manifest["total_bytes"] <= 15_000_000_000:
        raise RuntimeError(f"unexpected pi0_base manifest size: {manifest['total_bytes']}")
    if args.final.exists():
        raise FileExistsError(f"final target already exists: {args.final}")
    command = [
        str(args.python), str(args.guard),
        "--attempt-dir", str(attempt / "guard"),
    ]
    for root in args.monitor_root:
        command += ["--monitor-root", str(root)]
    command += [
        "--existing-billed-bytes", str(args.existing_billed_bytes),
        "--soft-limit-bytes", str(args.soft_limit_bytes),
        "--hard-limit-bytes", str(args.hard_limit_bytes),
        "--timeout-seconds", str(args.timeout_seconds),
        "--sample-interval-seconds", "5",
        "--near-soft-margin-bytes", "5000000000",
        "--near-sample-interval-seconds", "0.25",
        "--min-mem-available-bytes", "64000000000",
        "--max-child-rss-bytes", "4000000000",
        "--max-load1-per-cpu", "0.90",
        "--resource-consecutive-samples", "3",
        "--", str(args.python), str(args.downloader), "download",
        "--manifest", str(manifest_copy),
        "--scratch", str(args.scratch),
        "--final", str(args.final),
        "--lock", str(args.lock),
        "--report", str(attempt / "download_report.json"),
        "--retries", "12",
    ]
    if args.proxy:
        command += ["--proxy", args.proxy]
    atomic_json(attempt / "automation_manifest.json", {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "command": command,
        "source_manifest_sha256": sha256(manifest_copy),
        "source_total_bytes": manifest["total_bytes"],
        "source_object_count": manifest["object_count"],
        "expected_peak_billed_bytes": args.existing_billed_bytes + manifest["total_bytes"],
        "remaining_to_hard_limit_bytes": args.hard_limit_bytes - args.existing_billed_bytes - manifest["total_bytes"],
    })
    print(f"ATTEMPT={attempt}", flush=True)
    result = subprocess.run(command, check=False)
    atomic_json(attempt / "automation_exit.json", {"ended_at_utc": utc_now(), "returncode": result.returncode})
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
