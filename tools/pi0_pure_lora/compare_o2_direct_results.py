"""Fail-closed comparison of two O2 direct-Policy diagnostic result folders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import o2_direct_policy_diagnostic as diagnostic


def _array(root: Path, filename: str, expected: dict[str, Any], *, expected_shape: tuple[int, int]) -> np.ndarray:
    value = np.load(root / filename, allow_pickle=False)
    _, summary = diagnostic.action_summary({"actions": value}, label=filename, expected_shape=expected_shape)
    if summary != expected:
        raise ValueError(f"{root / filename} does not match its recorded identity")
    return value


def load_result(root: Path) -> dict[str, Any]:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if result.get("stage") != "O2-direct-policy-explicit-noise":
        raise ValueError(f"{root} is not an O2 direct diagnostic")
    if not isinstance(result.get("input_bundle_identity_sha256"), str) or not isinstance(result.get("noise_sha256"), str):
        raise ValueError(f"{root} has no fixed input/noise identity")
    return {
        "result": result,
        "normalized": _array(root, "normalized_actions.npy", result["normalized_action"], expected_shape=(50, 32)),
        "physical": _array(root, "actions.npy", result["action"], expected_shape=(50, 7)),
    }


def compare(left_root: Path, right_root: Path) -> dict[str, Any]:
    left, right = load_result(left_root), load_result(right_root)
    if left["result"]["input_bundle_identity_sha256"] != right["result"]["input_bundle_identity_sha256"]:
        raise ValueError("direct diagnostics used different input bundles")
    if left["result"]["noise_sha256"] != right["result"]["noise_sha256"]:
        raise ValueError("direct diagnostics used different explicit noise")
    return {
        "schema_version": 1,
        "stage": "O2-direct-policy-comparison",
        "left": str(left_root),
        "right": str(right_root),
        "input_bundle_identity_sha256": left["result"]["input_bundle_identity_sha256"],
        "noise_sha256": left["result"]["noise_sha256"],
        "normalized_action_difference": diagnostic.compare_actions(left["normalized"], right["normalized"]),
        "physical_action_difference": diagnostic.compare_actions(left["physical"], right["physical"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.left, args.right)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "pass", "normalized_exact": report["normalized_action_difference"]["exact_sha256_equal"], "physical_exact": report["physical_action_difference"]["exact_sha256_equal"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
