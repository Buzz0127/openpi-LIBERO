"""Create an immutable, identity-gated pure-LoRA training segment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import experiment_identity


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--support-tools-root", required=True, type=Path)
    parser.add_argument("--allowed-run-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--segment-start", required=True, type=int)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = experiment_identity.validate_model_manifest(json.loads(args.model_manifest.read_text()))
    if not args.python.is_file() or not (args.openpi_root / "scripts/train.py").is_file():
        raise FileNotFoundError("fixed Python or OpenPI train.py is missing")
    if not (args.support_tools_root / "pi0_pure_lora" / "adapter_artifact.py").is_file():
        raise FileNotFoundError("pure-LoRA support package is missing")
    if not _inside(args.checkpoint_dir, args.allowed_run_root):
        raise ValueError("checkpoint directory escapes the approved LoRA run root")
    if not _inside(args.adapter_root, args.allowed_run_root):
        raise ValueError("adapter directory escapes the approved LoRA run root")
    if args.segment_start < 0 or args.segment_end <= args.segment_start:
        raise ValueError("segment must satisfy 0 <= start < end")
    if args.train_seed == args.eval_seed:
        raise ValueError("training and evaluation seeds must be distinct")
    experiment_identity.validate_sha256(args.dataset_manifest_sha256, "dataset_manifest_sha256")
    command = [
        str(args.python), str(args.openpi_root / "scripts/train.py"), "pi0_libero_pure_lora",
        f"--exp-name={args.exp_name}", f"--seed={args.train_seed}",
        f"--num-train-steps={args.segment_end}", f"--save-interval={args.segment_end - args.segment_start}",
    ]
    if args.segment_start > 0:
        command.append("--resume")
    plan = {
        "schema_version": 1,
        "stage": "pure_lora_training_segment",
        "model_identity_sha256": model["model_identity_sha256"],
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "dataset_revision": args.dataset_revision,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "adapter_milestone_dir": str((args.adapter_root / args.exp_name / f"step-{args.segment_end:08d}").resolve()),
        "full_state_dir": str(args.checkpoint_dir.resolve()),
        "command": command,
        "required_environment": {
            "PYTHONPATH": ":".join((
                str((args.openpi_root / "src").resolve()),
                str(args.support_tools_root.resolve()),
            )),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "CUDA_VISIBLE_DEVICES": "<selected physical GPU>",
        },
        "rotation_policy": {
            "old_and_new_must_coexist_until_new_restore": True,
            "automatic_pruning_allowed": False,
            "adapter_milestone_required": True,
        },
        "execution_authorized": False,
    }
    plan["plan_identity_sha256"] = experiment_identity.canonical_sha256(plan)
    experiment_identity.atomic_write_new(args.output, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
