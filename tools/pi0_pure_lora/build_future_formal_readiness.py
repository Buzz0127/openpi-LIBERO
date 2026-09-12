#!/usr/bin/env python3
"""Build a non-executable FT3--FT7 readiness record from frozen inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import experiment_identity
import formal_future_schedule
import formal_segment_contract


REQUIRED_TOOLS = {"runner", "driver", "verifier", "orchestrator", "planner", "gpu_guard", "storage_guard"}


def _parse_tools(raw: list[str]) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for item in raw:
        key, separator, path = item.partition("=")
        if not separator or not key or key in values:
            raise ValueError("tools must be unique NAME=PATH entries")
        values[key] = Path(path)
    if set(values) != REQUIRED_TOOLS:
        raise ValueError("readiness record requires the exact future formal tool set")
    if any(not path.is_file() for path in values.values()):
        raise FileNotFoundError("a required future formal tool is missing")
    return values


def _decision(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value); claimed = unsigned.pop("decision_identity_sha256", None)
    if claimed != experiment_identity.canonical_sha256(unsigned):
        raise RuntimeError("C-FT2 decision identity mismatch")
    if value.get("stage") != "C-FT2-controlled-exception" or value.get("decision") != "accept-existing-ft1-ft2-chain":
        raise RuntimeError("future readiness requires the explicit accepted C-FT2 decision")
    if value.get("execution_authorized") is not False or value.get("next_stage_auto_start") is not False:
        raise RuntimeError("C-FT2 decision authorization boundary changed")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--c-ft2-decision", required=True, type=Path)
    parser.add_argument("--ft0-package", required=True, type=Path)
    parser.add_argument("--tool", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    decision = _decision(args.c_ft2_decision)
    package = formal_segment_contract._load_frozen_package(args.ft0_package)
    actual = tuple((row.get("start"), row.get("end")) for row in package["formal_training"].get("segments", []))
    if actual[-len(formal_future_schedule.FUTURE_SEGMENTS):] != formal_future_schedule.FUTURE_SEGMENTS:
        raise RuntimeError("FT0 package future boundaries differ from the registered FT3-FT7 schedule")
    tools = _parse_tools(args.tool)
    record = {
        "schema_version": 1,
        "stage": "R-FT-future-formal-readiness",
        "status": "pass",
        "c_ft2_decision_sha256": experiment_identity.sha256_file(args.c_ft2_decision),
        "c_ft2_decision_identity_sha256": decision["decision_identity_sha256"],
        "ft0_package_sha256": experiment_identity.sha256_file(args.ft0_package),
        "ft0_package_identity_sha256": package["package_identity_sha256"],
        "future_segments": [{"start": start, "end": end, "user_authorization_required": True,
                             "next_stage_auto_start": False} for start, end in formal_future_schedule.FUTURE_SEGMENTS],
        "tools": {name: {"path": str(path), "sha256": experiment_identity.sha256_file(path)} for name, path in sorted(tools.items())},
        "identity_schemas": {"historical_ft2": "ft2-newline-v1", "future_terminal": "formal-canonical-v1"},
        "execution_authorized": False,
        "next_stage_auto_start": False,
    }
    record["readiness_identity_sha256"] = experiment_identity.canonical_sha256(record)
    experiment_identity.atomic_write_new(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
