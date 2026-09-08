#!/usr/bin/env python3
"""Seal a T1 plan after supplied fresh preflight; never collect samples or launch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import time

import experiment_identity as identity
import t1_execution_contract as contract


def _options(values: dict) -> list[str]:
    return [item for key, value in values.items() for item in ("--" + key.replace("_", "-"), str(value))]


def finalize_plan(template: dict, preflight: dict, *, preflight_sha256: str, now_epoch: float, now_monotonic: float, host_identity: dict) -> dict:
    contract.verify_identity(template, "template_identity_sha256")
    if template.get("kind") != "static_template" or template.get("stage") != contract.STAGE or template.get("command") is not None or template.get("selected_physical_gpu") is not None or template.get("environment") is not None:
        raise ValueError("input is not an unbound static T1 template")
    if template.get("execution_authorized") is not False or template.get("next_stage_auto_start") is not False or template.get("prune_or_delete_authorized") is not False:
        raise ValueError("authorization/pruning flags changed")
    if template["bounds"] != contract.BOUNDS or template["guard_bounds"] != contract.GUARD_BOUNDS or (template["segment_start"], template["segment_end"], template["train_seed"], template["eval_seed"]) != (100, 200, 42, 7):
        raise ValueError("frozen T1 bounds/segment/seeds changed")
    if set(template["tool_bindings"]) != contract.TOOL_ROLES or set(template["input_files"]) != contract.INPUT_ROLES:
        raise ValueError("template file bindings missing")
    if template.get("monitor_roots") != contract.monitor_roots(template["source"]["root"]):
        raise ValueError("storage monitor roots changed")
    identity.validate_sha256(preflight_sha256, "preflight file")
    selected = contract.validate_preflight(preflight, now_epoch=now_epoch, now_monotonic=now_monotonic, host_identity=host_identity)
    paths = template["paths"]
    contract.validate_paths(paths)
    tools = {role: record["path"] for role, record in template["tool_bindings"].items()}
    python = paths["python"]
    resume = template["resume_input"]
    runner = [python, tools["runner"], *_options({"openpi_root": template["source"]["root"], "model_manifest": paths["model_manifest"], "golden_manifest": paths["golden_manifest"], "freeze_package": paths["freeze_package"], "s1d_acceptance_report": paths["s1d_acceptance_report"], "expected_s1d_acceptance_sha256": template["input_files"]["s1d_acceptance_report"]["sha256"], "expected_s1d_report_identity": resume["acceptance_report_identity_sha256"], "expected_checkpoint_tree_sha256": resume["checkpoint_tree_sha256"], "expected_adapter_identity_sha256": resume["adapter_identity_sha256"], "checkpoint_dir": paths["checkpoint_root"], "adapter_root": paths["adapter_root"], "exp_name": Path(paths["checkpoint_root"]).name, "attempt_dir": paths["attempt_dir"], "progress": paths["progress"], "loader_receipt": paths["loader_receipt"], "rng_receipt": paths["rng_receipt"], "composition_receipt": paths["composition_receipt"], "output": paths["result"], "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7})]
    guard = [python, tools["gpu_guard"], *_options({"physical_gpu": selected["index"], "pause_at": 95, "resume_at": 85, "min_free_memory_percent": 15, "resume_free_memory_percent": 20, "terminate_free_memory_percent": 10, "resume_samples": 5, "interval_seconds": 1, "monitor_error_limit": 3, "max_prelaunch_wait_seconds": 300, "max_runtime_seconds": 6000, "terminate_grace_seconds": 15, "kill_grace_seconds": 5, "log": paths["gpu_events"]}), "--", *runner]
    storage = [python, tools["storage_guard"], *_options({"attempt_dir": paths["storage_guard_dir"], "existing_billed_bytes": template["storage"]["current_billed_bytes"], "soft_limit_bytes": 240_000_000_000, "hard_limit_bytes": 250_000_000_000, "timeout_seconds": 6600, "sample_interval_seconds": 1, "near_soft_margin_bytes": 5_000_000_000, "near_sample_interval_seconds": .25, "term_grace_seconds": 45, "kill_grace_seconds": 5, "min_mem_available_bytes": 64_000_000_000, "max_load1_per_cpu": .90, "resource_consecutive_samples": 3, "monitor_failure_consecutive_samples": 2})]
    for root in template["monitor_roots"]:
        storage += ["--monitor-root", root]
    storage += ["--", *guard]
    environment = contract.expected_environment(tools["runner"], paths["attempt_dir"], selected["index"])
    command = ["/usr/bin/env", *[f"{key}={value}" for key, value in environment.items()], *storage]
    required = [paths[key] for key in ("result", "loader_receipt", "rng_receipt", "composition_receipt", "progress", "gpu_events")]
    required += [str(Path(paths["storage_guard_dir"]) / filename) for filename in ("exit_status.json", "run_manifest.json", "samples.jsonl")]
    result = {"schema_version": 1, "stage": contract.STAGE, "kind": "exact_non_authorizing_plan", "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7, "expected_final_committed_step": 200, "candidate": False, "source": copy.deepcopy(template["source"]), "identities": copy.deepcopy(template["identities"]), "paths": copy.deepcopy(paths), "attempt_dir": paths["attempt_dir"], "current_json": paths["current_json"], "progress_json": paths["progress"], "required_outputs": required, "bounds": copy.deepcopy(template["bounds"]), "guard_bounds": copy.deepcopy(template["guard_bounds"]), "command": command, "runner_command": runner, "gpu_guard_command": guard, "storage_guard_command": storage, "environment": environment, "selected_physical_gpu": selected["index"], "selected_gpu_uuid": selected["uuid"], "jax_visible_device_count": 1, "preflight": {"path": paths["preflight"], "sha256": preflight_sha256, "identity_sha256": preflight["preflight_identity_sha256"], "host_identity": host_identity, "collection_finished_epoch_seconds": preflight["collection_finished_epoch_seconds"], "valid_until_epoch_seconds": preflight["collection_finished_epoch_seconds"] + 120}, "template_identity_sha256": template["template_identity_sha256"], "freeze_package_identity_sha256": template["freeze_package_identity_sha256"], "tool_bindings": copy.deepcopy(template["tool_bindings"]), "input_files": copy.deepcopy(template["input_files"]), "resume_input": copy.deepcopy(resume), "storage": copy.deepcopy(template["storage"]), "monitor_roots": copy.deepcopy(template["monitor_roots"]), "tools": {"planner_sha256": template["tool_bindings"]["finalizer"]["sha256"], "orchestrator_sha256": template["tool_bindings"]["orchestrator"]["sha256"], "launcher_sha256": template["tool_bindings"]["launcher"]["sha256"]}, "execution_authorized": False, "next_stage_auto_start": False, "prune_or_delete_authorized": False, "launch_freshness_recheck_required": True, "guard_controls_only_owned_child_pgid": True, "terminal_acceptance_required": True}
    return contract.signed(result, "plan_identity_sha256")


def launch_review(plan: dict, plan_sha256: str) -> dict:
    paths = plan["paths"]
    bindings = plan["tool_bindings"]
    source = plan["source"]
    arguments = _options({"attempt_dir": paths["attempt_dir"], "current_json": paths["current_json"], "stage_plan": paths["stage_plan"], "expected_stage_plan_sha256": plan_sha256, "source_root": source["root"], "expected_branch": source["branch"], "expected_head": source["head"], "expected_upstream": source["upstream"], "stage": plan["stage"], "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7, "progress_json": paths["progress"], "expected_final_committed_step": 200})
    for name, value in sorted(plan["identities"].items()):
        arguments += ["--identity", f"{name}={value}"]
    for path in plan["required_outputs"]:
        arguments += ["--required-output", path]
    arguments += _options({key: value for key, value in plan["bounds"].items() if key != "retry_return_codes"})
    arguments += ["--", *plan["command"]]
    session = "t1-100-200-" + plan["plan_identity_sha256"][:20]
    command = [paths["python"], bindings["launcher"]["path"], *_options({"tmux": paths["tmux"], "session_name": session, "python": paths["python"], "orchestrator": bindings["orchestrator"]["path"], "attempt_dir": paths["attempt_dir"], "startup_timeout_seconds": 30, "required_heartbeat_observations": 2}), "--", *arguments]
    return contract.signed({"schema_version": 1, "plan_identity_sha256": plan["plan_identity_sha256"], "stage_plan_sha256": plan_sha256, "session_name": session, "launch_argv": command, "launch_shell_for_review_only": shlex.join(command), "execution_authorized": False, "next_stage_auto_start": False, "expires_with_preflight": True}, "review_identity_sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if identity.sha256_file(args.template) != args.expected_template_sha256 or identity.sha256_file(args.preflight) != args.expected_preflight_sha256:
        raise ValueError("template or preflight file hash mismatch")
    template = json.loads(args.template.read_text())
    if str(args.output.absolute()) != template["paths"]["stage_plan"] or str(args.preflight.absolute()) != template["paths"]["preflight"]:
        raise ValueError("output/preflight paths differ from static bindings")
    plan = finalize_plan(template, json.loads(args.preflight.read_text()), preflight_sha256=args.expected_preflight_sha256, now_epoch=time.time(), now_monotonic=time.monotonic(), host_identity=contract.current_host())
    contract.validate_launch_bindings(plan)
    review_path = Path(plan["paths"]["launch_review"])
    contract.new_path(args.output)
    contract.new_path(review_path)
    digest = hashlib.sha256(contract.json_bytes(plan)).hexdigest()
    review = launch_review(plan, digest)
    contract.write_new(args.output, plan)
    contract.write_new(review_path, review)
    print(json.dumps({"plan_identity_sha256": plan["plan_identity_sha256"], "stage_plan_sha256": digest, "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
