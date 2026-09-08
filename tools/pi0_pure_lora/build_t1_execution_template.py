#!/usr/bin/env python3
"""Build a static T1 review template: no commands, GPU selection, or execution."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
from pathlib import Path

import experiment_identity as identity
import t1_execution_contract as contract


def build_template(freeze: dict, readiness: dict, spec: dict) -> dict:
    unsigned = dict(freeze)
    claimed = unsigned.pop("package_identity_sha256", None)
    if claimed != hashlib.sha256((json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest():
        raise ValueError("freeze package identity mismatch")
    if freeze.get("execution_authorized") is not False or freeze.get("automatic_next_stage") is not False:
        raise ValueError("freeze package must remain non-authorizing")
    segment = freeze["engineering_segment"]
    if (segment["start"], segment["end"], segment["candidate"], segment["delete_step_100_after_success"], segment["batch_size"], segment["num_workers"]) != (100, 200, False, False, 1, 0):
        raise ValueError("engineering segment changed")
    if freeze["seeds"] != {"training": 42, "evaluation": 7} or set(freeze["identities"]) != contract.IDENTITIES:
        raise ValueError("seed or experiment identity contract changed")
    for key, value in freeze["identities"].items():
        identity.validate_sha256(value, key)
    if readiness.get("status") != "pass" or any(readiness.get(key) is not False for key in ("execution_authorized", "gpu_used", "model_loaded", "checkpoint_loaded", "real_dataset_loaded")):
        raise ValueError("CPU readiness is not a non-executing pass")
    ids = readiness["input_identities"]
    resume = freeze["resume_input"]
    if ids["t1_freeze_package_identity_sha256"] != freeze["package_identity_sha256"] or ids["s1d_acceptance_report_identity_sha256"] != resume["acceptance_report_identity_sha256"] or ids["s1d_checkpoint_tree_sha256"] != resume["checkpoint_tree_sha256"] or ids["s1d_adapter_identity_sha256"] != resume["adapter_identity_sha256"]:
        raise ValueError("readiness/freeze resume identity mismatch")
    source = copy.deepcopy(freeze["source"])
    identity.validate_git_oid(source["head"], "source HEAD")
    if source["status"] != "clean" or readiness["openpi_source"] != source:
        raise ValueError("source snapshot mismatch")
    source["status"] = ""
    contract.absolute(source["root"])
    if set(spec) != {"paths", "tools", "input_files"}:
        raise ValueError("spec requires exactly paths, tools, and input_files")
    paths = copy.deepcopy(spec["paths"])
    attempt = contract.absolute(paths["attempt_dir"])
    for key, filename in {"progress": "progress.json", "result": "runner_result.json", "loader_receipt": "loader_receipt.json", "rng_receipt": "rng_receipt.json", "composition_receipt": "composition_receipt.json", "gpu_events": "gpu_guard.jsonl", "storage_guard_dir": "storage_guard"}.items():
        if key in paths and paths[key] != str(attempt / filename):
            raise ValueError(f"derived output mismatch: {key}")
        paths[key] = str(attempt / filename)
    for key, root, filename in (("checkpoint_200", "checkpoint_root", "200"), ("adapter_200", "adapter_root", "step-00000200"), ("adapter_200_receipt", "adapter_root", "step-00000200.verified.json")):
        expected = str(Path(paths[root]) / filename)
        if key in paths and paths[key] != expected:
            raise ValueError(f"derived target mismatch: {key}")
        paths[key] = expected
    contract.validate_paths(paths)
    if paths["checkpoint_root"] != resume["checkpoint_root"]:
        raise ValueError("checkpoint root differs from frozen S1d input")
    bindings = copy.deepcopy(spec["tools"])
    inputs = copy.deepcopy(spec["input_files"])
    if set(bindings) != contract.TOOL_ROLES or set(inputs) != contract.INPUT_ROLES:
        raise ValueError("tool/input bindings must cover the exact required roles")
    for name, record in {**bindings, **inputs}.items():
        if set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid binding fields: {name}")
        contract.absolute(record["path"])
        identity.validate_sha256(record["sha256"], name)
    if len({record["path"] for record in bindings.values()}) != len(bindings):
        raise ValueError("tool bindings alias")
    for role, readiness_key in {"freeze_package": "t1_freeze_package_sha256", "s1d_acceptance_report": "s1d_acceptance_report_sha256", "model_manifest": "base_model_manifest_file_sha256", "golden_manifest": "golden_manifest_file_sha256"}.items():
        if inputs[role] != {"path": paths[role], "sha256": ids[readiness_key]}:
            raise ValueError(f"input identity does not match readiness: {role}")
    storage = copy.deepcopy(freeze["storage"])
    expected_storage = {"review_line_bytes": 225_000_000_000, "soft_stop_bytes": 240_000_000_000, "hard_limit_bytes": 250_000_000_000, "minimum_uncommitted_bytes": 20_000_000_000, "deletion_authorized": False, "automatic_pruning_allowed": False}
    if any(storage.get(key) != value for key, value in expected_storage.items()):
        raise ValueError("storage preservation policy changed")
    increment = storage["measured_full_state_bytes"] + storage["measured_adapter_bytes"] + storage["atomic_write_margin_bytes"]
    if increment <= 0 or increment > 10 * (1 << 30) or storage["current_billed_bytes"] + increment >= 225_000_000_000:
        raise ValueError("new engineering segment exceeds the frozen review budget")
    result = {"schema_version": 1, "stage": contract.STAGE, "kind": "static_template", "source": source, "identities": copy.deepcopy(freeze["identities"]), "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7, "expected_final_committed_step": 200, "candidate": False, "paths": paths, "tool_bindings": bindings, "input_files": inputs, "resume_input": copy.deepcopy(resume), "freeze_package_identity_sha256": freeze["package_identity_sha256"], "readiness_identity_sha256": identity.canonical_sha256(readiness), "bounds": dict(contract.BOUNDS), "guard_bounds": dict(contract.GUARD_BOUNDS), "storage": storage, "estimated_increment_bytes": increment, "monitor_roots": contract.monitor_roots(source["root"]), "selected_physical_gpu": None, "selected_gpu_uuid": None, "environment": None, "command": None, "execution_authorized": False, "next_stage_auto_start": False, "prune_or_delete_authorized": False, "fresh_preflight_required": True, "execution_ready": False, "committed_tool_snapshot_required": True, "changed_since_previous_runner_readiness": bindings["runner"]["sha256"] != readiness.get("runner_file_sha256", {}).get("run_t1_resume_segment.py"), "historical_readiness_validates_current_tools": False, "source_and_target_runtime_recheck_required": True, "input_hash_heartbeat_note": "1800 seconds covers input rehash, loader positioning and compilation; total A2 wall timeout is 6840 seconds plus at most 70 seconds of cleanup. Real costs remain unmeasured."}
    return contract.signed(result, "template_identity_sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-package", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze_package.read_text())
    readiness = json.loads(args.readiness.read_text())
    if identity.sha256_file(args.freeze_package) != readiness["input_identities"]["t1_freeze_package_sha256"]:
        raise ValueError("freeze file differs from CPU readiness")
    result = build_template(freeze, readiness, json.loads(args.spec.read_text()))
    contract.write_new(args.output, result)
    print(json.dumps({"template_identity_sha256": result["template_identity_sha256"], "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
