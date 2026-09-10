from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import verify_ft1_result as verifier


class Ft1VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.template = root / "template.json"
        self.result = root / "result.json"
        self.output = root / "acceptance.json"
        template = {
            "schema_version": 1,
            "stage": "FT1-formal-0-1000",
            "execution_authorized": False, "ft0_freeze": {"identity": "freeze"},
            "identities": {"model": "m", "dataset": "d", "norm": "n", "golden": "g", "config": "c", "split": "s"},
        }
        template["template_identity_sha256"] = verifier._canonical(template)
        self.template.write_text(json.dumps(template))
        receipt = {
            "checkpoint_restore_succeeded": True,
            "parameter_tree_shape_dtype_equal": True,
            "optimizer_tree_shape_dtype_equal": True,
            "all_parameter_values_equal_after_restore": True,
            "all_optimizer_values_equal_after_restore": True,
            "adapter_values_equal_after_restore": True,
        }
        self.value = {
            "status": "pass", "stage": "formal-pure-lora-segment", "segment_start": 0, "segment_end": 1000,
            "identities": {"config_patch_sha256": "c", "golden_manifest_sha256": "g", "norm_stats_sha256": "n"},
            "ft0_contract": {"ft0_package_identity_sha256": "freeze"}, "metrics_count": 1000, "all_metrics_finite": True,
            "changed_golden_leaf_count": 20, "changed_non_golden_leaf_count": 0, "checkpoint_steps": [1000],
            "checkpoint_restore_receipt": receipt, "next_stage_started": False,
        }
        self.result.write_text(json.dumps(self.value))

    def test_accepts_complete_ft1_result(self) -> None:
        self.assertEqual(verifier.verify(self.template, self.result)["status"], "pass")

    def test_cli_writes_new_immutable_report(self) -> None:
        report = verifier.verify(self.template, self.result)
        verifier._write_new_json(self.output, report)
        self.assertEqual(json.loads(self.output.read_text())["report_identity_sha256"], report["report_identity_sha256"])
        with self.assertRaises(FileExistsError):
            verifier._write_new_json(self.output, report)

    def test_rejects_missing_adapter_composition_proof(self) -> None:
        self.value["checkpoint_restore_receipt"]["adapter_values_equal_after_restore"] = False
        self.result.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "adapter-composition"):
            verifier.verify(self.template, self.result)

    def test_rejects_frozen_package_identity_mismatch(self) -> None:
        self.value["ft0_contract"]["ft0_package_identity_sha256"] = "other"
        self.result.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "frozen package"):
            verifier.verify(self.template, self.result)

    def test_rejects_frozen_base_change(self) -> None:
        self.value["changed_non_golden_leaf_count"] = 1
        self.result.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "leaf invariant"):
            verifier.verify(self.template, self.result)

    def test_rejects_auto_next_segment(self) -> None:
        self.value["next_stage_started"] = True
        self.result.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "automatically"):
            verifier.verify(self.template, self.result)
