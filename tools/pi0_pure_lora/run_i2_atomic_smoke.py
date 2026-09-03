"""Exercise I2 atomic save/restore semantics on a bounded fake train state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from atomic_train_state_store import AtomicTrainStateStore, tree_sha256


def fake_state(step: int) -> dict:
    return {
        "step": np.asarray(step, np.int64),
        "params": {"lora_a": np.arange(4096, dtype=np.float32).reshape(64, 64) + step},
        "opt_state": {
            "count": np.asarray(step, np.int32),
            "mu": np.full((64, 64), step / 1000, np.float32),
            "nu": np.full((64, 64), step / 10000, np.float32),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    store = AtomicTrainStateStore(args.output_dir)
    first = store.save_verified(0, fake_state(0))
    interrupted = False
    try:
        store.save_verified(50, fake_state(50), fail_after_async_save=True)
    except RuntimeError as error:
        if "simulated interruption" not in str(error):
            raise
        interrupted = True
    if store.committed_steps() != [0]:
        raise AssertionError("Interrupted step became committed")
    if tree_sha256(store.restore_verified(0)) != tree_sha256(fake_state(0)):
        raise AssertionError("Last-known-good step failed after interruption")
    second = store.save_verified(100, fake_state(100))
    if store.committed_steps() != [0, 100]:
        raise AssertionError("Old and new verified steps do not coexist")
    restored = store.restore_verified(100)
    if tree_sha256(restored) != tree_sha256(fake_state(100)):
        raise AssertionError("New step restore mismatch")
    report = {
        "schema_version": 1,
        "status": "passed",
        "async_wait_completed": first["async_wait_completed"] and second["async_wait_completed"],
        "interrupted_staging_ignored": interrupted,
        "staging_path_count": len(store.staging_paths()),
        "committed_steps": store.committed_steps(),
        "old_step_retained_after_new_restore": (args.output_dir / "step-00000000").is_dir(),
        "new_step_restore_hash_equal": True,
        "optimizer_and_step_included_in_tree_hash": True,
        "automatic_pruning_enabled": False,
        "deletion_performed": False,
    }
    report_path = args.output_dir.parent / "i2_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
