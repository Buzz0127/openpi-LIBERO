"""Fail-closed consistency audit for the pure-LoRA final-report evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "base_completed": 200,
    "main_completed": 400,
    "paired_denominator": 200,
    "base_successes": 0,
    "pure_lora_successes": 25,
    "pure_lora_only": 25,
    "both_failure": 175,
    "e3_completed": 382,
    "e3_successes": 19,
    "e3_exceptions": 0,
}


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def audit(base_path: Path, main_path: Path, paired_path: Path, e3_path: Path) -> dict[str, Any]:
    base, base_hash = _load(base_path)
    main, main_hash = _load(main_path)
    paired, paired_hash = _load(paired_path)
    e3, e3_hash = _load(e3_path)
    _require(base.get("stage") == "E2-base-recovery-200" and base.get("status") == "pass", "Base recovery status mismatch")
    _require(base.get("completed") == EXPECTED["base_completed"], "Base recovery denominator mismatch")
    _require(main.get("stage") == "E2-main-400" and main.get("status") == "pass", "E2 main status mismatch")
    _require(main.get("completed") == EXPECTED["main_completed"], "E2 main denominator mismatch")
    _require(base.get("selection_lock_sha256") == main.get("selection_lock_sha256"), "E2 selection locks differ")
    _require(paired.get("stage") == "E2-main-paired-audit" and paired.get("status") == "pass", "Paired audit status mismatch")
    _require(paired.get("denominator") == EXPECTED["paired_denominator"], "Paired denominator mismatch")
    _require(paired.get("base", {}).get("successes") == EXPECTED["base_successes"], "Base successes mismatch")
    _require(paired.get("pure_lora", {}).get("successes") == EXPECTED["pure_lora_successes"], "LoRA successes mismatch")
    _require(paired.get("paired_outcomes", {}).get("pure_lora_only") == EXPECTED["pure_lora_only"], "LoRA-only count mismatch")
    _require(paired.get("paired_outcomes", {}).get("both_failure") == EXPECTED["both_failure"], "Both-failure count mismatch")
    _require(e3.get("stage") == "G2-e3-user-stopped-partial", "E3 closeout stage mismatch")
    _require(e3.get("completed_valid_unique_episodes") == EXPECTED["e3_completed"], "E3 partial denominator mismatch")
    _require(e3.get("success_episodes") == EXPECTED["e3_successes"], "E3 success count mismatch")
    _require(e3.get("evaluator_exceptions") == EXPECTED["e3_exceptions"], "E3 evaluator exception count mismatch")
    _require(e3.get("next_stage_started") is False and e3.get("port_18001_released") is True, "E3 closeout state mismatch")
    return {
        "schema_version": 1,
        "stage": "F2-final-report-evidence-audit",
        "status": "pass",
        "expected": EXPECTED,
        "artifacts": {
            "base_recovery_summary": {"path": str(base_path), "sha256": base_hash, "summary_identity_sha256": base.get("summary_identity_sha256")},
            "e2_main_summary": {"path": str(main_path), "sha256": main_hash, "summary_identity_sha256": main.get("summary_identity_sha256")},
            "paired_comparison": {"path": str(paired_path), "sha256": paired_hash, "summary_identity_sha256": paired.get("summary_identity_sha256")},
            "e3_partial_closeout": {"path": str(e3_path), "sha256": e3_hash},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", required=True, type=Path)
    parser.add_argument("--main-summary", required=True, type=Path)
    parser.add_argument("--paired-summary", required=True, type=Path)
    parser.add_argument("--e3-closeout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.base_summary, args.main_summary, args.paired_summary, args.e3_closeout)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
