#!/usr/bin/env python3
"""Fail-closed terminal acceptance for the formal FT2 1000->5000 segment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _canon(value: dict) -> str: return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()).hexdigest()
def _new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    try: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)
def verify(ft1_acceptance: Path, ft1_result: Path, result: Path) -> dict:
    prior=json.loads(ft1_acceptance.read_text())
    unsigned=dict(prior); claimed=unsigned.pop("report_identity_sha256",None)
    if claimed != _canon(unsigned) or prior.get("status") != "pass" or prior.get("segment") != [0,1000] or prior.get("result_sha256") != _sha(ft1_result): raise RuntimeError("FT1 acceptance chain mismatch")
    before=json.loads(ft1_result.read_text()); after=json.loads(result.read_text())
    if (after.get("status"),after.get("stage"),after.get("segment_start"),after.get("segment_end")) != ("pass","formal-pure-lora-segment",1000,5000): raise RuntimeError("FT2 result stage/status mismatch")
    if after.get("metrics_count") != 4000 or after.get("all_metrics_finite") is not True or after.get("changed_golden_leaf_count") != 20 or after.get("changed_non_golden_leaf_count") != 0: raise RuntimeError("FT2 metrics or pure-LoRA invariant failed")
    if after.get("checkpoint_steps") != [1000,5000] or after.get("next_stage_started") is not False: raise RuntimeError("FT2 checkpoint history or auto-next violation")
    if after.get("identities") != before.get("identities") or after.get("ft0_contract",{}).get("ft0_package_identity_sha256") != before.get("ft0_contract",{}).get("ft0_package_identity_sha256"): raise RuntimeError("FT2 experiment identity drift")
    receipt=after.get("checkpoint_restore_receipt",{}); required=("checkpoint_restore_succeeded","parameter_tree_shape_dtype_equal","optimizer_tree_shape_dtype_equal","all_parameter_values_equal_after_restore","all_optimizer_values_equal_after_restore","adapter_values_equal_after_restore")
    if not all(receipt.get(k) is True for k in required): raise RuntimeError("FT2 restore or adapter proof missing")
    report={"schema_version":1,"stage":"FT2-terminal-acceptance","status":"pass","candidate":True,"ft1_acceptance_sha256":_sha(ft1_acceptance),"ft1_result_sha256":_sha(ft1_result),"result_sha256":_sha(result),"segment":[1000,5000],"next_stage_started":False}; report["report_identity_sha256"]=_canon(report); return report
def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--ft1-acceptance",required=True,type=Path); p.add_argument("--ft1-result",required=True,type=Path); p.add_argument("--result",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args(); value=verify(a.ft1_acceptance,a.ft1_result,a.result); _new(a.output,value); print(json.dumps(value,sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
