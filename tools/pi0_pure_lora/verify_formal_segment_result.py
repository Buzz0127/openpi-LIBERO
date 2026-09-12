#!/usr/bin/env python3
"""Fail-closed terminal acceptance for one future FT0-registered segment.

Historical FT1/FT2 terminal reports use a JSON-plus-newline hash.  New
future reports deliberately use ``experiment_identity.canonical_sha256``
(canonical JSON *without* a trailing newline); the prior schema is explicit
on the command line so the two identities cannot be silently interchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import experiment_identity


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _newline_canonical(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_prior(path: Path, result: Path, expected_end: int, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    claimed = unsigned.pop("report_identity_sha256", None)
    if schema == "ft2-newline-v1":
        valid = claimed == _newline_canonical(unsigned) and value.get("stage") == "FT2-terminal-acceptance"
    elif schema == "formal-canonical-v1":
        valid = claimed == experiment_identity.canonical_sha256(unsigned) and value.get("stage") == "formal-segment-terminal-acceptance"
    else:
        raise ValueError(f"unsupported prior identity schema: {schema}")
    if not valid or value.get("status") != "pass" or value.get("candidate") is not True:
        raise RuntimeError("prior terminal acceptance schema/status mismatch")
    if value.get("segment", [None, None])[1] != expected_end:
        raise RuntimeError("prior terminal acceptance boundary mismatch")
    if value.get("result_sha256") != _sha(result) or value.get("next_stage_started") is not False:
        raise RuntimeError("prior terminal acceptance result/auto-next mismatch")
    return value


def verify(
    prior_acceptance: Path, prior_result: Path, result: Path, *, segment_start: int, segment_end: int,
    prior_identity_schema: str,
) -> dict[str, Any]:
    prior = _validate_prior(prior_acceptance, prior_result, segment_start, prior_identity_schema)
    before = json.loads(prior_result.read_text(encoding="utf-8"))
    after = json.loads(result.read_text(encoding="utf-8"))
    if (after.get("status"), after.get("stage"), after.get("segment_start"), after.get("segment_end")) != (
        "pass", "formal-pure-lora-segment", segment_start, segment_end,
    ):
        raise RuntimeError("future result stage/status/boundary mismatch")
    if after.get("metrics_count") != segment_end - segment_start or after.get("all_metrics_finite") is not True:
        raise RuntimeError("future result finite metric count mismatch")
    if after.get("changed_golden_leaf_count") != 20 or after.get("changed_non_golden_leaf_count") != 0:
        raise RuntimeError("future result pure-LoRA invariant failed")
    expected_steps = list(before.get("checkpoint_steps", [])) + [segment_end]
    if after.get("checkpoint_steps") != expected_steps or after.get("next_stage_started") is not False:
        raise RuntimeError("future result checkpoint chain or auto-next mismatch")
    if after.get("identities") != before.get("identities"):
        raise RuntimeError("future experiment identities drifted")
    if after.get("ft0_contract", {}).get("ft0_package_identity_sha256") != before.get("ft0_contract", {}).get("ft0_package_identity_sha256"):
        raise RuntimeError("future FT0 contract identity drifted")
    receipt = after.get("checkpoint_restore_receipt", {})
    required = (
        "checkpoint_restore_succeeded", "parameter_tree_shape_dtype_equal", "optimizer_tree_shape_dtype_equal",
        "all_parameter_values_equal_after_restore", "all_optimizer_values_equal_after_restore",
        "adapter_values_equal_after_restore",
    )
    if not all(receipt.get(key) is True for key in required):
        raise RuntimeError("future restore or adapter proof missing")
    report = {
        "schema_version": 1,
        "stage": "formal-segment-terminal-acceptance",
        "identity_schema": "formal-canonical-v1",
        "status": "pass",
        "candidate": True,
        "prior_identity_schema": prior_identity_schema,
        "prior_acceptance_sha256": _sha(prior_acceptance),
        "prior_result_sha256": _sha(prior_result),
        "result_sha256": _sha(result),
        "segment": [segment_start, segment_end],
        "next_stage_started": False,
    }
    report["report_identity_sha256"] = experiment_identity.canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-acceptance", required=True, type=Path)
    parser.add_argument("--prior-result", required=True, type=Path)
    parser.add_argument("--prior-identity-schema", required=True, choices=("ft2-newline-v1", "formal-canonical-v1"))
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--segment-start", required=True, type=int)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.segment_end <= args.segment_start:
        parser.error("segment end must exceed start")
    report = verify(args.prior_acceptance, args.prior_result, args.result, segment_start=args.segment_start,
                    segment_end=args.segment_end, prior_identity_schema=args.prior_identity_schema)
    _new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
