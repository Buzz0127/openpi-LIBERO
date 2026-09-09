from __future__ import annotations

import argparse
import hashlib
import unittest

import build_ft0_formal_training_freeze as ft0


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Ft0FormalTrainingFreezeTests(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            source_root="/remote/openpi", source_branch="feature/pi0-libero-pure-lora",
            source_head="a" * 40, source_status="clean", source_upstream="NONE",
            identity=[f"{key}={digest(key)}" for key in sorted(ft0.IDENTITY_KEYS)],
            config_source="/remote/openpi/src/openpi/training/config.py", config_source_sha256=digest("config"),
            optimizer_source="/remote/openpi/src/openpi/training/optimizer.py", optimizer_source_sha256=digest("optimizer"),
            run_name="pi0-libero-pure-lora-ft0-seed42-v1", checkpoint_root="/runs/checkpoints", adapter_root="/runs/adapters",
            train_seed=42, eval_seed=7, batch_size=1, num_workers=0, target_step=30_000,
            candidate_step=list(ft0.CANDIDATE_STEPS), current_billed_bytes=97_531_600_896,
            full_state_bytes=5_559_145_746, adapter_bytes=199_962_483, atomic_margin_bytes=1_000_000_000,
            review_line_bytes=225_000_000_000, soft_stop_bytes=240_000_000_000,
            hard_limit_bytes=250_000_000_000, minimum_uncommitted_bytes=20_000_000_000,
        )

    def test_builds_nonexecuting_fresh_base_plan(self):
        package = ft0.build(self.args())
        self.assertFalse(package["execution_authorized"])
        self.assertFalse(package["execution_ready"])
        training = package["formal_training"]
        self.assertEqual(training["starts_from"], "pi0_base")
        self.assertFalse(training["starts_from_engineering_step_200"])
        self.assertEqual([(row["start"], row["end"]) for row in training["segments"]], [(0, 1000), (1000, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), (25000, 30000)])
        self.assertTrue(all(not row["auto_start_next_segment"] for row in training["segments"]))
        self.assertEqual(package["storage"]["worst_peak_old_plus_new_full_state_bytes"], 144_404_504_245)
        self.assertGreaterEqual(package["storage"]["hard_headroom_bytes"], 20_000_000_000)
        self.assertEqual(package["package_identity_sha256"], ft0._canonical({key: value for key, value in package.items() if key != "package_identity_sha256"}))

    def test_rejects_noncanonical_candidate_schedule(self):
        args = self.args()
        args.candidate_step[-1] = 29_000
        with self.assertRaisesRegex(ValueError, "candidate steps"):
            ft0.build(args)

    def test_rejects_shared_host_loader_expansion(self):
        args = self.args()
        args.num_workers = 1
        with self.assertRaisesRegex(ValueError, "batch size 1 and zero workers"):
            ft0.build(args)

    def test_rejects_peak_at_review_line(self):
        args = self.args()
        args.review_line_bytes = 144_404_504_245
        with self.assertRaisesRegex(ValueError, "review line"):
            ft0.build(args)


if __name__ == "__main__":
    unittest.main()
