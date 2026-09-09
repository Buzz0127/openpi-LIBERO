from __future__ import annotations
import hashlib, json
from pathlib import Path
import tempfile, unittest
import build_ft0_formal_training_freeze as ft0
import build_ft1_execution_template as ft1
from test_build_ft0_formal_training_freeze import Ft0FormalTrainingFreezeTests

class Ft1TemplateTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name); a=Ft0FormalTrainingFreezeTests().args(); self.freeze=self.root/'f.json'; self.freeze.write_text(json.dumps(ft0.build(a))); self.tools={x:hashlib.sha256(x.encode()).hexdigest() for x in ft1.ROLES}; self.source={'root':'/remote/openpi','branch':'feature/pi0-libero-pure-lora','head':'a'*40,'upstream':'NONE','status':''}
 def test_nonexecuting_template_binds_ft1(self):
  x=ft1.build(self.freeze,self.tools,self.source,self.root/'attempts'); self.assertFalse(x['execution_authorized']); self.assertEqual((x['segment_start'],x['segment_end']),(0,1000)); self.assertIsNone(x['command']); self.assertEqual(x['template_identity_sha256'],ft1._canonical({k:v for k,v in x.items() if k!='template_identity_sha256'}))
 def test_dirty_source_rejected(self):
  s=dict(self.source); s['status']='M config.py'
  with self.assertRaisesRegex(RuntimeError,'clean'): ft1.build(self.freeze,self.tools,s,self.root)
