from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import compare_o2_direct_results as comparison
import o2_direct_policy_diagnostic as diagnostic


class CompareO2DirectResultsTest(unittest.TestCase):
    def _write_result(self, root: Path, normalized: np.ndarray, physical: np.ndarray, *, bundle: str = "b" * 64):
        root.mkdir()
        for filename, value in (("normalized_actions.npy", normalized), ("actions.npy", physical)):
            with (root / filename).open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
        _, normal_summary = diagnostic.action_summary({"actions": normalized}, expected_shape=(50, 32))
        _, physical_summary = diagnostic.action_summary({"actions": physical})
        (root / "result.json").write_text(json.dumps({
            "stage": "O2-direct-policy-explicit-noise", "input_bundle_identity_sha256": bundle,
            "noise_sha256": "n" * 64, "normalized_action": normal_summary, "action": physical_summary,
        }), encoding="utf-8")

    def test_reports_both_spaces_and_rejects_identity_mismatch(self):
        zero = np.zeros((50, 7), dtype=np.float32)
        latent = np.zeros((50, 32), dtype=np.float32)
        changed = zero.copy(); changed[0, 0] = .25
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._write_result(root / "left", latent, zero); self._write_result(root / "right", latent, changed)
            report = comparison.compare(root / "left", root / "right")
            self.assertEqual(report["normalized_action_difference"]["component_layout"], "model-latent")
            self.assertEqual(report["physical_action_difference"]["translation_max_abs"], .25)
            self._write_result(root / "wrong", latent, zero, bundle="c" * 64)
            with self.assertRaises(ValueError):
                comparison.compare(root / "left", root / "wrong")


if __name__ == "__main__":
    unittest.main()
