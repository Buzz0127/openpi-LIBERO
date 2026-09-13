"""Create the outcome-blind full-2000 E3 state manifest from frozen E0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(e0: dict[str, Any], e0_sha256: str) -> dict[str, Any]:
    protocol = e0["protocol"]
    selection = e0["selection"]
    suites = list(protocol["suites"])
    tasks_per_suite = int(protocol["tasks_per_suite"])
    states_per_task = int(selection["available_initial_states_per_task"])
    if suites != ["libero_spatial", "libero_object", "libero_goal", "libero_10"] or tasks_per_suite != 10 or states_per_task != 50:
        raise ValueError("E3 only accepts the fixed four-suite, 10-task, 50-state E0 protocol")
    entries = [{"split": "full", "suite": suite, "task_id": task, "initial_state_index": state} for suite in suites for task in range(tasks_per_suite) for state in range(states_per_task)]
    if len(entries) != 2000 or len({(row["suite"], row["task_id"], row["initial_state_index"]) for row in entries}) != 2000:
        raise ValueError("E3 full state construction is not exactly 2000 unique entries")
    full_keys = {(row["suite"], row["task_id"], row["initial_state_index"]) for row in entries}
    e0_keys = {(row["suite"], row["task_id"], row["initial_state_index"]) for row in e0["entries"]}
    if not e0_keys <= full_keys:
        raise ValueError("E0 development/main entries are not contained in full-2000")
    stable = {"schema_version": 1, "stage": "E3-full-2000-preregistration", "parent_e0_manifest_sha256": e0_sha256, "parent_e0_manifest_identity_sha256": e0["manifest_identity_sha256"], "evaluation_seed": protocol["evaluation_seed"], "entries": entries, "selection_lock_rule": "reuse the pre-E2 locked adapter; main/E3 outcomes cannot reselect it", "next_stage_started": False}
    manifest = dict(stable)
    manifest["manifest_identity_sha256"] = hashlib.sha256(_canonical(stable)).hexdigest()
    return manifest


def write_new(output: Path, manifest: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        path = partial / "e3_full_state_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            for child in partial.iterdir():
                child.unlink()
            partial.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(json.loads(args.e0_manifest.read_text()), _sha256(args.e0_manifest))
    write_new(args.output, manifest)
    print(json.dumps({"entries": len(manifest["entries"]), "manifest_identity_sha256": manifest["manifest_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
