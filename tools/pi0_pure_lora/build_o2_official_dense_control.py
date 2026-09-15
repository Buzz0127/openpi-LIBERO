"""Write an O2 control package only; it never launches a GPU process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import experiment_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-recipe", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recipe = json.loads(args.runtime_recipe.read_text(encoding="utf-8"))
    control = {
        "schema_version": 1,
        "stage": "O2-official-dense-runtime-validation",
        "status": "controls-ready-gpu-preflight-required",
        "execution_authorized": False,
        "pre_registered_protocol": {
            "single_gpu": True,
            "sequential_arms": ["official_pi0_libero_reference", "project_unmerged_bf16", "project_merged_dense_bf16"],
            "warmups_per_arm": 10,
            "timed_requests_per_arm": 50,
            "fixed_observation_and_noise_manifest_required": True,
            "fresh_policy_process_per_arm": True,
            "max_wallclock_seconds": 2700,
            "memory_only_guard_required": True,
        },
        "correctness_requirements": {
            "same_backend_repeatability": "fixed explicit-noise direct Policy calls require identical action SHA-256 for every registered request",
            "direct_official_vs_adapter_dense": "same dense BF16 tree, same input and explicit noise require identical action SHA-256",
            "unmerged_vs_merged": "numeric-version analysis only: require finite equal-shaped outputs and report normalized plus unnormalized translation/rotation/gripper errors; it is not an equivalence gate",
            "thresholds": {
                "same_backend_action_sha256": "exact",
                "direct_official_vs_adapter_dense_action_sha256": "exact",
                "unmerged_vs_merged": "finite-and-shape-required; report-only because BF16 associativity differs"
            },
            "threshold_source": "Identical code/path, fixed input, and explicit noise are required to be bit-identical; BF16 merged versus unmerged is a deliberately new numerical runtime and cannot use an outcome-selected tolerance.",
            "threshold_status": "pre-registered-before-observation",
        },
        "non_goals": ["no E1/E2/E3 launch", "no full-2000 evaluation", "no checkpoint serialization", "no official-checkpoint action equality claim"],
        "runtime_recipe": recipe,
    }
    control["plan_identity_sha256"] = experiment_identity.canonical_sha256(control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(control, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "control-written-gpu-preflight-required", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
