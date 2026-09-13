"""Fail-closed audit for the paired E2 Base and locked pure-LoRA main-200 results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _expected_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = sorted((row for row in manifest["entries"] if row.get("split") == "main"), key=lambda row: (row["suite"], row["task_id"], row["initial_state_index"]))
    keys = [(row["suite"], row["task_id"], row["initial_state_index"]) for row in entries]
    if len(entries) != 200 or len(set(keys)) != 200:
        raise ValueError("E2 audit requires exactly 200 unique frozen main entries")
    return entries


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["suite"]), int(row["task_id"]), int(row["initial_state_index"]))


def _collect(root: Path, expected: list[dict[str, Any]], label: str) -> dict[tuple[str, int, int], tuple[dict[str, Any], Path]]:
    expected_keys = {_key(row) for row in expected}
    collected: dict[tuple[str, int, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted(root.glob("**/*_result.json")):
        row = json.loads(path.read_text())
        key = _key(row)
        if key not in expected_keys or key in collected:
            raise ValueError(f"{label} result keys are missing, duplicated, or outside E0 main")
        if row.get("failure_reason") == "exception" or int(row.get("policy_requests", 0)) <= 0:
            raise ValueError(f"{label} has an invalid infrastructure result at {key}")
        collected[key] = (row, path)
    if set(collected) != expected_keys:
        raise ValueError(f"{label} does not contain exactly the frozen main-200 results")
    return collected


def _rate(successes: int, total: int) -> float:
    return 100.0 * successes / total


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(bool(row[key]) for row in rows)
    return {"successes": successes, "denominator": len(rows), "success_rate_percent": _rate(successes, len(rows))}


def _write_new(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        fields = ["suite", "task_id", "initial_state_index", "base_success", "pure_lora_success", "pair_outcome", "base_policy_requests", "pure_lora_policy_requests", "base_result_sha256", "pure_lora_result_sha256"]
        with (temporary / "main_results.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (temporary / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        suite_rows: dict[str, list[dict[str, Any]]] = {}
        task_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            suite_rows.setdefault(row["suite"], []).append(row)
            task_rows.setdefault((row["suite"], row["task_id"]), []).append(row)
        suites = {suite: {"base": _summary(values, "base_success"), "pure_lora": _summary(values, "pure_lora_success")} for suite, values in sorted(suite_rows.items())}
        tasks = [{"suite": suite, "task_id": task, "base": _summary(values, "base_success"), "pure_lora": _summary(values, "pure_lora_success")} for (suite, task), values in sorted(task_rows.items())]
        (temporary / "suite_results.json").write_text(json.dumps(suites, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "task_results.json").write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        for child in temporary.iterdir() if temporary.exists() else ():
            child.unlink()
        if temporary.exists():
            temporary.rmdir()
        raise


def audit(base_root: Path, lora_root: Path, e0_manifest: Path, base_summary: Path, selection_lock: Path, output: Path) -> dict[str, Any]:
    expected = _expected_entries(json.loads(e0_manifest.read_text()))
    base = _collect(base_root, expected, "base")
    lora = _collect(lora_root, expected, "pure_lora")
    base_stage = json.loads(base_summary.read_text())
    if base_stage.get("stage") != "E2-base-recovery-200" or base_stage.get("status") != "pass" or base_stage.get("completed") != 200:
        raise ValueError("Base recovery summary is not a completed E2-base-recovery-200")
    lock = json.loads(selection_lock.read_text())
    stable = dict(lock)
    if stable.pop("lock_identity_sha256", None) != _canonical(stable):
        raise ValueError("selection lock identity mismatch")
    rows = []
    pairs = {"both_failure": 0, "base_only": 0, "pure_lora_only": 0, "both_success": 0}
    for entry in expected:
        key = _key(entry)
        base_row, base_path = base[key]
        lora_row, lora_path = lora[key]
        base_success, lora_success = bool(base_row["success"]), bool(lora_row["success"])
        outcome = "both_success" if base_success and lora_success else "base_only" if base_success else "pure_lora_only" if lora_success else "both_failure"
        pairs[outcome] += 1
        rows.append({"suite": key[0], "task_id": key[1], "initial_state_index": key[2], "base_success": base_success, "pure_lora_success": lora_success, "pair_outcome": outcome, "base_policy_requests": int(base_row["policy_requests"]), "pure_lora_policy_requests": int(lora_row["policy_requests"]), "base_result_sha256": _sha256(base_path), "pure_lora_result_sha256": _sha256(lora_path)})
    summary = {"schema_version": 1, "stage": "E2-main-paired-audit", "status": "pass", "denominator": len(rows), "base": _summary(rows, "base_success"), "pure_lora": _summary(rows, "pure_lora_success"), "pure_lora_minus_base_percentage_points": _rate(sum(row["pure_lora_success"] for row in rows), len(rows)) - _rate(sum(row["base_success"] for row in rows), len(rows)), "paired_outcomes": pairs, "identities": {"e0_manifest_sha256": _sha256(e0_manifest), "base_recovery_summary_sha256": _sha256(base_summary), "selection_lock_sha256": _sha256(selection_lock), "selection_lock_identity_sha256": lock["lock_identity_sha256"]}}
    summary["summary_identity_sha256"] = _canonical(summary)
    _write_new(output, rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--pure-lora-root", required=True, type=Path)
    parser.add_argument("--e0-manifest", required=True, type=Path)
    parser.add_argument("--base-summary", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.base_root, args.pure_lora_root, args.e0_manifest, args.base_summary, args.selection_lock, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
