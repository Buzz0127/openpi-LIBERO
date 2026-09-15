from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_o2_official_dense_control as control
import experiment_identity


class O2ControlTest(unittest.TestCase):
    def test_plan_is_non_authorizing_and_content_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); recipe = root / "recipe.json"; output = root / "plan.json"
            recipe.write_text(json.dumps({"runtime_kind": "pi0_pure_lora_official_dense_bf16"}))
            with mock.patch("sys.argv", ["program", "--output", str(output), "--runtime-recipe", str(recipe)]):
                self.assertEqual(control.main(), 0)
            saved = json.loads(output.read_text()); identity = saved.pop("plan_identity_sha256")
            self.assertFalse(saved["execution_authorized"])
            self.assertEqual(identity, experiment_identity.canonical_sha256(saved))


if __name__ == "__main__":
    unittest.main()
