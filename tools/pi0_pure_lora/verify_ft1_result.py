#!/usr/bin/env python3
"""Fail-closed terminal acceptance checks for an FT1 0->1000 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(template_path: Path, result_path: Path) -> dict:
    template = json.loads(template_path.read_text())
    unsigned = dict(template)
    claimed = unsigned.pop("template_identity_sha256", None)
    if claimed != _canonical(unsigned):
        raise RuntimeError("FT1 template identity mismatch")
    if template.get("execution_authorized") is not False or template.get("stage") != "FT1-formal-0-1000":
        raise RuntimeError("FT1 template authorization boundary changed")

    result = json.loads(result_path.read_text())
    if (result.get("status"), result.get("stage"), result.get("segment_start"), result.get("segment_end")) != (
        "pass", "formal-pure-lora-segment", 0, 1000,
    ):
        raise RuntimeError("FT1 result stage/status mismatch")
    # The runner reports runtime-manifest keys while the static template carries
    # the six frozen experiment identities.  Bind model/dataset/split through
    # the signed FT0 package identity, and compare the three runtime values
    # that are emitted by the runner directly.
    contract = result.get("ft0_contract", {})
    if contract.get("ft0_package_identity_sha256") != template["ft0_freeze"]["identity"]:
        raise RuntimeError("FT1 frozen package identity mismatch")
    runtime = result.get("identities", {})
    expected_runtime = {
        "config_patch_sha256": template["identities"]["config"],
        "golden_manifest_sha256": template["identities"]["golden"],
        "norm_stats_sha256": template["identities"]["norm"],
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise RuntimeError("FT1 runtime identity mismatch")
    if result.get("metrics_count") != 1000 or result.get("all_metrics_finite") is not True:
        raise RuntimeError("FT1 finite metric evidence missing")
    if result.get("changed_golden_leaf_count") != 20 or result.get("changed_non_golden_leaf_count") != 0:
        raise RuntimeError("FT1 pure-LoRA leaf invariant failed")
    if result.get("checkpoint_steps") != [1000]:
        raise RuntimeError("FT1 fresh trajectory checkpoint history mismatch")
    receipt = result.get("checkpoint_restore_receipt", {})
    required_receipt = (
        "checkpoint_restore_succeeded",
        "parameter_tree_shape_dtype_equal",
        "optimizer_tree_shape_dtype_equal",
        "all_parameter_values_equal_after_restore",
        "all_optimizer_values_equal_after_restore",
        "adapter_values_equal_after_restore",
    )
    if not all(receipt.get(key) is True for key in required_receipt):
        raise RuntimeError("FT1 full-state restore or adapter-composition evidence failed")
    if result.get("next_stage_started") is not False:
        raise RuntimeError("FT1 automatically started another segment")

    report = {
        "schema_version": 1,
        "stage": "FT1-terminal-acceptance",
        "status": "pass",
        "candidate": True,
        "template_identity_sha256": claimed,
        "template_sha256": _sha256(template_path),
        "result_sha256": _sha256(result_path),
        "segment": [0, 1000],
        "next_stage_started": False,
    }
    report["report_identity_sha256"] = _canonical(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.template, args.result)
    _write_new_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
