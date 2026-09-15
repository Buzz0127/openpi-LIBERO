from __future__ import annotations

import unittest

import numpy as np

import o2_direct_policy_diagnostic as diagnostic


class O2DirectDiagnosticTest(unittest.TestCase):
    def test_action_summary_and_component_difference(self):
        left = np.zeros((50, 7), dtype=np.float32)
        right = left.copy()
        right[0, 0] = .25
        right[1, 4] = .5
        right[2, 6] = .75
        _, summary = diagnostic.action_summary({"actions": left})
        self.assertEqual(len(summary["sha256"]), 64)
        result = diagnostic.compare_actions(left, right)
        self.assertFalse(result["exact_sha256_equal"])
        self.assertEqual(result["translation_max_abs"], .25)
        self.assertEqual(result["rotation_max_abs"], .5)
        self.assertEqual(result["gripper_max_abs"], .75)
        latent = diagnostic.compare_actions(np.zeros((50, 32), dtype=np.float32), np.ones((50, 32), dtype=np.float32))
        self.assertEqual(latent["component_layout"], "model-latent")
        with self.assertRaises(ValueError):
            diagnostic.action_summary({"actions": np.zeros((1, 7), dtype=np.float32)})

    def test_intermediate_helper_requires_jax_policy(self):
        with self.assertRaises(RuntimeError):
            diagnostic.infer_with_intermediates(type("TorchPolicy", (), {"_is_pytorch_model": True})(), {}, np.zeros((50, 32)))


if __name__ == "__main__":
    unittest.main()
