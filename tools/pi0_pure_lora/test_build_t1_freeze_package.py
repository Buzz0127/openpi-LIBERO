#!/usr/bin/env python3

from __future__ import annotations

import argparse
import unittest

import build_t1_freeze_package as builder


def _args() -> argparse.Namespace:
    keys = sorted(builder.IDENTITY_KEYS)
    tools = ["orchestrator", "launcher", "gpu_guard", "storage_guard", "runner"]
    return argparse.Namespace(
        output=None,
        source_root="/fixed/openpi",
        source_branch="feature/pi0-libero-pure-lora",
        source_head="a" * 40,
        source_status="clean",
        source_upstream="NONE",
        identity=[f"{key}={index:064x}" for index, key in enumerate(keys, 1)],
        required_tool=tools,
        tool_sha256=[f"{key}={index:064x}" for index, key in enumerate(tools, 11)],
        train_seed=42,
        eval_seed=7,
        resume_checkpoint_root="/runs/s1d",
        resume_checkpoint_tree_sha256="1" * 64,
        resume_adapter_identity_sha256="2" * 64,
        resume_receipt_sha256="3" * 64,
        resume_acceptance_identity_sha256="4" * 64,
        engineering_start=100,
        engineering_end=200,
        training_target_step=30000,
        candidate_step=[1000, 5000, 10000, 15000, 20000, 25000, 30000],
        current_billed_bytes=91_765_739_008,
        checkpoint_bytes=5_559_083_375,
        adapter_bytes=199_962_483,
        atomic_margin_bytes=1_000_000_000,
        review_line_bytes=225_000_000_000,
        soft_stop_bytes=240_000_000_000,
        hard_limit_bytes=250_000_000_000,
        minimum_uncommitted_bytes=20_000_000_000,
    )


class T1FreezePackageTest(unittest.TestCase):
    def test_package_is_non_executing_and_recomputes_storage_bound(self) -> None:
        package = builder.build(_args())
        self.assertFalse(package["execution_authorized"])
        self.assertFalse(package["execution_ready"])
        self.assertIsNone(package["execution_command"])
        self.assertFalse(package["automatic_next_stage"])
        self.assertEqual(package["engineering_segment"]["start"], 100)
        self.assertEqual(package["formal_training"]["candidate_steps"][-1], 30000)
        self.assertGreater(package["storage"]["hard_headroom_at_worst_peak_bytes"], 20_000_000_000)

    def test_candidate_drift_fails_closed(self) -> None:
        args = _args()
        args.candidate_step = [1000, 30000]
        with self.assertRaisesRegex(ValueError, "candidate steps must be exactly"):
            builder.build(args)

    def test_storage_review_line_fails_closed(self) -> None:
        args = _args()
        args.current_billed_bytes = 220_000_000_000
        with self.assertRaisesRegex(ValueError, "review line"):
            builder.build(args)

    def test_dirty_source_fails_closed(self) -> None:
        args = _args()
        args.source_status = "modified"
        with self.assertRaisesRegex(ValueError, "must be clean"):
            builder.build(args)


if __name__ == "__main__":
    unittest.main()
