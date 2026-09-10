import json
from pathlib import Path
import tempfile
import unittest
import verify_ft2_result as v

class Ft2Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup); r=Path(self.t.name); self.a=r/'a'; self.one=r/'one'; self.two=r/'two'; self.o=r/'o'
  one={"status":"pass","identities":{"x":"y"},"ft0_contract":{"ft0_package_identity_sha256":"f"}}; self.one.write_text(json.dumps(one)); acc={"status":"pass","segment":[0,1000],"result_sha256":v._sha(self.one)}; acc["report_identity_sha256"]=v._canon(acc); self.a.write_text(json.dumps(acc)); rec={k:True for k in ("checkpoint_restore_succeeded","parameter_tree_shape_dtype_equal","optimizer_tree_shape_dtype_equal","all_parameter_values_equal_after_restore","all_optimizer_values_equal_after_restore","adapter_values_equal_after_restore")}; self.value={"status":"pass","stage":"formal-pure-lora-segment","segment_start":1000,"segment_end":5000,"metrics_count":4000,"all_metrics_finite":True,"changed_golden_leaf_count":20,"changed_non_golden_leaf_count":0,"checkpoint_steps":[1000,5000],"next_stage_started":False,"identities":{"x":"y"},"ft0_contract":{"ft0_package_identity_sha256":"f"},"checkpoint_restore_receipt":rec}; self.two.write_text(json.dumps(self.value))
 def test_accepts(self): self.assertEqual(v.verify(self.a,self.one,self.two)["status"],"pass")
 def test_rejects_history(self):
  self.value["checkpoint_steps"]=[5000]; self.two.write_text(json.dumps(self.value))
  with self.assertRaisesRegex(RuntimeError,"history"): v.verify(self.a,self.one,self.two)
