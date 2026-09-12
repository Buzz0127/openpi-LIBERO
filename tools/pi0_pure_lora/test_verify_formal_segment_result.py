from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import experiment_identity
import verify_formal_segment_result as verifier


class FutureVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="future-verifier-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior_result = self.root / "prior-result.json"
        self.result = self.root / "result.json"
        self.prior_acceptance = self.root / "prior-acceptance.json"
        identities = {"model": "m"}
        self.before = {"status": "pass", "stage": "formal-pure-lora-segment", "segment_start": 1000,
                       "segment_end": 5000, "identities": identities, "ft0_contract": {"ft0_package_identity_sha256": "f"},
                       "checkpoint_steps": [1000, 5000]}
        self.prior_result.write_text(json.dumps(self.before), encoding="utf-8")
        unsigned = {"schema_version": 1, "stage": "FT2-terminal-acceptance", "status": "pass", "candidate": True,
                    "result_sha256": self._sha(self.prior_result), "segment": [1000, 5000], "next_stage_started": False}
        acceptance = dict(unsigned); acceptance["report_identity_sha256"] = verifier._newline_canonical(unsigned)
        self.prior_acceptance.write_text(json.dumps(acceptance), encoding="utf-8")
        receipt = {key: True for key in ("checkpoint_restore_succeeded", "parameter_tree_shape_dtype_equal",
                  "optimizer_tree_shape_dtype_equal", "all_parameter_values_equal_after_restore",
                  "all_optimizer_values_equal_after_restore", "adapter_values_equal_after_restore")}
        self.after = {**self.before, "segment_start": 5000, "segment_end": 10000, "metrics_count": 5000,
                      "all_metrics_finite": True, "changed_golden_leaf_count": 20, "changed_non_golden_leaf_count": 0,
                      "checkpoint_steps": [1000, 5000, 10000], "checkpoint_restore_receipt": receipt,
                      "next_stage_started": False}
        self.result.write_text(json.dumps(self.after), encoding="utf-8")

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_accepts_ft2_newline_prior_and_emits_canonical_future_identity(self) -> None:
        report = verifier.verify(self.prior_acceptance, self.prior_result, self.result, segment_start=5000,
                                 segment_end=10000, prior_identity_schema="ft2-newline-v1")
        self.assertEqual(report["identity_schema"], "formal-canonical-v1")
        unsigned = dict(report); claimed = unsigned.pop("report_identity_sha256")
        self.assertEqual(claimed, experiment_identity.canonical_sha256(unsigned))

    def test_wrong_prior_result_identity_fails_closed(self) -> None:
        self.prior_result.write_text(json.dumps({**self.before, "identities": {"model": "different"}}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "result/auto-next"):
            verifier.verify(self.prior_acceptance, self.prior_result, self.result, segment_start=5000,
                            segment_end=10000, prior_identity_schema="ft2-newline-v1")

    def test_non_golden_change_fails_closed(self) -> None:
        self.after["changed_non_golden_leaf_count"] = 1
        self.result.write_text(json.dumps(self.after), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "pure-LoRA invariant"):
            verifier.verify(self.prior_acceptance, self.prior_result, self.result, segment_start=5000,
                            segment_end=10000, prior_identity_schema="ft2-newline-v1")
