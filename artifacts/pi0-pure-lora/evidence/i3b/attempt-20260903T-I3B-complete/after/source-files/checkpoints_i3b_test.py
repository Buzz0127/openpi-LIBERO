from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from openpi.training import checkpoints
from openpi.training import config


class CheckpointPolicyTest(unittest.TestCase):
    def test_manager_can_disable_automatic_pruning(self):
        with tempfile.TemporaryDirectory() as raw:
            manager, resuming = checkpoints.initialize_checkpoint_dir(
                Path(raw) / "state", keep_period=None, max_to_keep=None, overwrite=False, resume=False
            )
            try:
                self.assertFalse(resuming)
                self.assertIsNone(manager._options.max_to_keep)
            finally:
                manager.close()

    def test_pure_lora_config_requires_verified_rotation_inputs(self):
        candidate = config.get_config("pi0_libero_pure_lora")
        self.assertIsNone(candidate.checkpoint_max_to_keep)
        self.assertTrue(candidate.pure_lora_model_manifest)
        self.assertTrue(candidate.pure_lora_golden_manifest)
        self.assertTrue(candidate.pure_lora_adapter_base_dir)

    def test_existing_debug_config_retains_single_checkpoint_default(self):
        candidate = config.get_config("debug")
        self.assertEqual(candidate.checkpoint_max_to_keep, 1)
        self.assertIsNone(candidate.pure_lora_model_manifest)


if __name__ == "__main__":
    unittest.main()
