"""Seal the completed E1 development ranking; it never launches E2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import evaluation_control


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary_path = args.run_dir / "summary.json"
    results_path = args.run_dir / "development_results.jsonl"
    summary = json.loads(summary_path.read_text())
    registration = json.loads(args.registration.read_text())
    if summary.get("status") != "pass" or summary.get("completed") != 280:
        raise ValueError("E1 run is not complete")
    if summary.get("registration_identity_sha256") != registration.get("identity_sha256"):
        raise ValueError("E1 summary/registration identity mismatch")
    ledger = {}
    fields = ("split", "suite", "task_id", "initial_state_index", "candidate_step", "outcome", "infrastructure_retry")
    for line in results_path.read_text().splitlines():
        row = json.loads(line)
        evaluation_control.record_episode(ledger, {field: row[field] for field in fields})
    if len(ledger) != 280:
        raise ValueError("E1 results ledger is incomplete")
    lock = evaluation_control.select(ledger, registration)
    lock.update({
        "e1_run_dir": str(args.run_dir),
        "e1_summary_sha256": sha256(summary_path),
        "e1_results_sha256": sha256(results_path),
        "e1_completed": 280,
        "next_stage_started": False,
        "e2_authorized": False,
    })
    evaluation_control.write_lock(args.output, lock)
    print(json.dumps(lock, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
