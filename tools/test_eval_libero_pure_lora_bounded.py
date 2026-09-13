from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import eval_libero_pure_lora_bounded as evaluator


class PureLoraEvaluatorTest(unittest.TestCase):
    def test_preregistered_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "states.json"
            path.write_text(json.dumps({"schema_version": 1, "entries": [
                {"split": "development", "suite": "libero_spatial", "task_id": 0, "initial_state_index": 3}
            ]}))
            args = argparse.Namespace(task_state_manifest=path, evaluation_split="development", suite="libero_spatial", task_id=0, initial_states=[3])
            evaluator.validate_task_state_selection(args)
            args.initial_states = [4]
            with self.assertRaisesRegex(ValueError, "not preregistered"):
                evaluator.validate_task_state_selection(args)

    def test_preregistered_full_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "states.json"
            path.write_text(json.dumps({"schema_version": 1, "entries": [
                {"split": "full", "suite": "libero_goal", "task_id": 2, "initial_state_index": 49}
            ]}))
            args = argparse.Namespace(task_state_manifest=path, evaluation_split="full", suite="libero_goal", task_id=2, initial_states=[49])
            evaluator.validate_task_state_selection(args)

    def test_gpu_baseline_limits_must_be_explicit(self) -> None:
        with self.assertRaises(SystemExit):
            evaluator.parse_args(["--task-id", "0", "--initial-states", "0"])

    def test_action_contract_is_50_by_7_compatible(self) -> None:
        self.assertEqual(evaluator.LIBERO_DUMMY_ACTION, [0.0] * 6 + [-1.0])

    def test_base_identity_uses_c0_field_not_manifest_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "base.json"
            expected = "c" * 64
            path.write_text(json.dumps({"identities": {"base_manifest_sha256": expected}}))
            self.assertEqual(evaluator.canonical_base_identity(path), expected)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                evaluator.canonical_base_identity(path)


if __name__ == "__main__":
    unittest.main()
