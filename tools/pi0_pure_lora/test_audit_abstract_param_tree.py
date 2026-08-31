from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import audit_abstract_param_tree as audit


FAKE_LEAVES = [
    {
        "path_parts": ["PaliGemma", "llm", "block_0", "attention", "q_proj", "kernel"],
        "variable_type": "Param",
        "shape": [8, 8],
        "dtype": "float32",
    },
    {
        "path_parts": ["PaliGemma", "llm", "block_0", "attention", "q_proj", "lora_a"],
        "variable_type": "Param",
        "shape": [8, 2],
        "dtype": "float32",
    },
    {
        "path_parts": ["PaliGemma", "llm", "block_0", "attention", "q_proj", "lora_b"],
        "variable_type": "LoRAParam",
        "shape": [2, 8],
        "dtype": "float32",
    },
    {
        "path_parts": ["SigLIP", "encoder", "lora_named_container", "kernel"],
        "variable_type": "Param",
        "shape": [4, 4],
        "dtype": "float32",
    },
    {
        "path_parts": ["Pi0", "state_projection", "bias"],
        "variable_type": "Param",
        "shape": [8],
        "dtype": "float32",
    },
    {
        "path_parts": ["Pi0", "action_projection", "lora_a"],
        "variable_type": "BatchStat",
        "shape": [8, 2],
        "dtype": "float32",
    },
    {
        "path_parts": ["PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_a"],
        "variable_type": "Param",
        "shape": [8, 2],
        "dtype": "float32",
    },
    {
        "path_parts": ["PaliGemma", "llm", "layers", "mlp", "linear_lora_b"],
        "variable_type": "Param",
        "shape": [2, 8],
        "dtype": "float32",
    },
]


def _golden_candidates(records: list[dict[str, object]]) -> set[str]:
    """Independent test oracle: explicit leaf type/name enumeration."""

    accepted_pairs = {
        (variable_type, terminal)
        for variable_type in ("Param", "LoRAParam")
        for terminal in (
            "lora_a",
            "lora_b",
            "gating_einsum_lora_a",
            "gating_einsum_lora_b",
            "linear_lora_a",
            "linear_lora_b",
        )
    }
    result: set[str] = set()
    for record in records:
        parts = record["path_parts"]
        assert isinstance(parts, list)
        pair = (record["variable_type"], parts[-1])
        if pair in accepted_pairs:
            result.add("/".join(parts))
    return result


class AuditAbstractParamTreeTest(unittest.TestCase):
    def test_exact_terminal_and_variable_type_rule_is_not_substring_based(self) -> None:
        report = audit.audit_records(FAKE_LEAVES)
        self.assertEqual(
            set(report["exact_lora_candidate_paths"]), _golden_candidates(FAKE_LEAVES)
        )
        substring_record = next(
            record
            for record in report["records"]
            if "lora_named_container" in record["joined_path"]
        )
        self.assertIs(substring_record["lora_substring_only"], True)
        self.assertIs(substring_record["exact_lora_candidate"], False)
        batch_stat = next(
            record for record in report["records"] if record["variable_type"] == "BatchStat"
        )
        self.assertEqual(batch_stat["leaf_name"], "lora_a")
        self.assertIs(batch_stat["exact_lora_candidate"], False)

    def test_counts_and_protected_base_hints_are_preserved(self) -> None:
        report = audit.audit_records(FAKE_LEAVES)
        self.assertEqual(report["leaf_count"], 8)
        self.assertEqual(report["parameter_count"], 168)
        self.assertEqual(report["exact_lora_candidate_parameter_count"], 64)
        kernel = next(
            record
            for record in report["records"]
            if record["joined_path"].endswith("q_proj/kernel")
        )
        self.assertIs(kernel["base_terminal"], True)
        state_bias = next(
            record
            for record in report["records"]
            if record["joined_path"].endswith("state_projection/bias")
        )
        self.assertIs(state_bias["base_terminal"], True)
        self.assertIn("state_projection", state_bias["subsystem_hints"])

    def test_duplicate_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate serialized leaf paths"):
            audit.audit_records([FAKE_LEAVES[0], FAKE_LEAVES[0]])

    def test_cli_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="g1a-audit-test-") as directory:
            tmp_path = Path(directory)
            input_path = tmp_path / "fake_tree.json"
            output_path = tmp_path / "report.json"
            input_path.write_text(json.dumps(FAKE_LEAVES), encoding="utf-8")
            script = Path(audit.__file__).resolve()
            command = [
                sys.executable,
                str(script),
                "--fake-tree-json",
                str(input_path),
                "--output",
                str(output_path),
            ]
            first = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
