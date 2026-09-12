from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import record_c_ft2_decision as recorder


class Cft2DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cft2-decision-")
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.ft1 = self.root / "ft1.json"; self.ft2 = self.root / "ft2.json"; self.a1 = self.root / "a1.json"; self.a2 = self.root / "a2.json"
        one = {"status":"pass","identities":{"x":"y"},"ft0_contract":{"ft0_package_identity_sha256":"f"},"checkpoint_steps":[1000]}; self.ft1.write_text(json.dumps(one))
        ft1_report = {"status":"pass","segment":[0,1000],"result_sha256":self._sha(self.ft1)}; ft1_report["report_identity_sha256"] = recorder.verify_ft2_result._canon(ft1_report); self.a1.write_text(json.dumps(ft1_report))
        receipt = {key: True for key in ("checkpoint_restore_succeeded","parameter_tree_shape_dtype_equal","optimizer_tree_shape_dtype_equal","all_parameter_values_equal_after_restore","all_optimizer_values_equal_after_restore","adapter_values_equal_after_restore")}
        two = {"status":"pass","stage":"formal-pure-lora-segment","segment_start":1000,"segment_end":5000,"metrics_count":4000,"all_metrics_finite":True,"changed_golden_leaf_count":20,"changed_non_golden_leaf_count":0,"checkpoint_steps":[1000,5000],"next_stage_started":False,"identities":{"x":"y"},"ft0_contract":{"ft0_package_identity_sha256":"f"},"checkpoint_restore_receipt":receipt}; self.ft2.write_text(json.dumps(two))
        report = recorder.verify_ft2_result.verify(self.a1,self.ft1,self.ft2); self.a2.write_text(json.dumps(report))
        self.o1=self.root/"o1.json"; self.o2=self.root/"o2.json"; self.o1.write_text(json.dumps({"status":"fail","reason_code":"child_exit_nonzero"})); self.o2.write_text(json.dumps({"status":"fail","reason_code":"committed_step_mismatch"}))
    @staticmethod
    def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
    def test_records_only_narrow_non_authorizing_exception(self) -> None:
        out=self.root/"decision.json"; argv=["record","--output",str(out),"--ft1-result",str(self.ft1),"--ft1-acceptance",str(self.a1),"--ft1-outer-summary",str(self.o1),"--ft2-result",str(self.ft2),"--ft2-acceptance",str(self.a2),"--ft2-outer-summary",str(self.o2)]
        with mock.patch("sys.argv",argv): self.assertEqual(recorder.main(),0)
        value=json.loads(out.read_text()); self.assertFalse(value["execution_authorized"]); self.assertTrue(value["exception_scope"]["historical_outer_summaries_remain_failed"])
