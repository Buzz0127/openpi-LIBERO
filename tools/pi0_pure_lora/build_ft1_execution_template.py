"""Build a non-executing FT1 0->1000 autonomous execution template."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SHA = re.compile(r"^[0-9a-f]{64}$")
ROLES = {"runner", "contract", "orchestrator", "launcher", "gpu_guard", "storage_guard", "preflight", "terminal_verifier"}


def _canonical(value: dict) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()


def _pairs(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        key, sep, digest = value.partition("=")
        if not sep or key in parsed or not SHA.fullmatch(digest): raise ValueError("tool must be unique ROLE=SHA256")
        parsed[key] = digest
    if set(parsed) != ROLES: raise ValueError(f"tool roles must be exactly {sorted(ROLES)}")
    return parsed


def build(freeze_path: Path, tools: dict[str, str], source: dict[str, str], attempt_parent: Path) -> dict:
    freeze = json.loads(freeze_path.read_text()); unsigned = dict(freeze); claimed = unsigned.pop("package_identity_sha256", None)
    if claimed != _canonical(unsigned) or freeze.get("stage") != "FT0-formal-training-freeze": raise RuntimeError("FT0 freeze identity invalid")
    if freeze.get("execution_authorized") is not False or freeze.get("execution_ready") is not False: raise RuntimeError("FT0 authorization boundary changed")
    if source.get("status") != "" or not source.get("head") or source.get("upstream") != "NONE": raise RuntimeError("source must be clean and upstream-free")
    first = freeze["formal_training"]["segments"][0]
    if first != {"start": 0, "end": 1000, "candidate_adapter_publish": True, "full_state_required_for_resume": True, "user_authorization_required": True, "auto_start_next_segment": False}: raise RuntimeError("FT1 boundary differs from freeze")
    template = {"schema_version": 1, "stage": "FT1-formal-0-1000", "segment_start": 0, "segment_end": 1000, "candidate": True, "source": source, "ft0_freeze": {"path": str(freeze_path.resolve()), "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(), "identity": claimed}, "identities": freeze["identities"], "tools": tools, "attempt_parent": str(attempt_parent.resolve()), "run_root": freeze["formal_training"]["checkpoint_run_root"], "adapter_root": freeze["formal_training"]["adapter_run_root"], "storage": freeze["storage"], "execution_authorized": False, "execution_ready": False, "command": None, "environment": None, "selected_physical_gpu": None, "selected_gpu_uuid": None, "fresh_preflight_required": True, "next_stage_auto_start": False, "terminal_acceptance_required": {"finite_metrics": True, "golden_changed": 20, "non_golden_changed": 0, "full_restore": True, "adapter_composition": True, "old_and_new_full_state_coexist": True, "owned_processes_absent": True}}
    template["template_identity_sha256"] = _canonical(template); return template


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True,type=Path); p.add_argument("--freeze",required=True,type=Path); p.add_argument("--source-root",required=True); p.add_argument("--source-branch",required=True); p.add_argument("--source-head",required=True); p.add_argument("--source-upstream",default="NONE"); p.add_argument("--source-status",default=""); p.add_argument("--attempt-parent",required=True,type=Path); p.add_argument("--tool",action="append",required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    value=build(a.freeze,_pairs(a.tool),{"root":a.source_root,"branch":a.source_branch,"head":a.source_head,"upstream":a.source_upstream,"status":a.source_status},a.attempt_parent)
    a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_name(f".{a.output.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.link(tmp,a.output); tmp.unlink(); print(json.dumps(value,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
