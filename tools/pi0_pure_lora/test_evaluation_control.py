import unittest
import sys, tempfile
import json
from pathlib import Path
import evaluation_control as e
import evaluation_lifecycle as lifecycle
import build_e1_plan
H="a"*64
class T(unittest.TestCase):
 def test_lock_and_accounting(self):
  reg=e.register({i*1000:H[:-1]+str(i) for i in range(1,8)},identities={"x":"y"}); ledger={}
  for c in range(1,8):
   for s in range(40): e.record_episode(ledger,{"split":"development","suite":"libero_spatial","task_id":s,"initial_state_index":0,"candidate_step":c*1000,"outcome":"success" if c==1 else "policy_failure","infrastructure_retry":False})
  self.assertEqual(e.select(ledger,reg)["step"],1000)
  with self.assertRaises(ValueError): e.record_episode(ledger,next(iter(ledger.values())))
 def test_policy_failure_never_retries(self):
   with self.assertRaises(ValueError): e.record_episode({}, {"split":"development","suite":"x","task_id":0,"initial_state_index":0,"candidate_step":1,"outcome":"policy_failure","infrastructure_retry":True})
 def test_selection_accepts_frozen_registration_adapter_identity_key(self):
  registration={"identity_sha256":"r"*64,"candidates":[{"step":1000,"adapter_identity":"a"*64}]}
  ledger={}
  for state in range(40): e.record_episode(ledger,{"split":"development","suite":"x","task_id":state,"initial_state_index":0,"candidate_step":1000,"outcome":"success","infrastructure_retry":False})
  self.assertEqual(e.select(ledger,registration)["adapter_identity_sha256"],"a"*64)
 def test_immutable_lock_and_owned_lifecycle(self):
  with tempfile.TemporaryDirectory() as d:
   p=__import__("pathlib").Path(d)/"lock.json"; e.write_lock(p,{"step":1});
   with self.assertRaises(FileExistsError): e.write_lock(p,{"step":2})
  r=e.owned_fake_lifecycle([sys.executable,"-c","pass"]); self.assertTrue(r["reaped"])
 def test_e2e_registration_plan_fake_execution_summary_and_lock(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); adapters={i*1000:(str(i)*64)[:64] for i in range(1,8)}
   reg=e.register(adapters,identities={'base':'b'*64}); index={'registration_identity_sha256':reg['identity_sha256'],'identity_sha256':'c'*64}
   manifest={'_path':str(root/'e0.json'),'entries':[{'split':'development','suite':'libero_spatial','task_id':i,'initial_state_index':i} for i in range(40)]}
   Path(manifest['_path']).write_text(json.dumps({k:v for k,v in manifest.items() if k!='_path'}))
   plan=build_e1_plan.build(reg,manifest,index,video_bytes=0,log_bytes=1,wiring=lifecycle.static_wiring(openpi_root='/openpi',evaluator='/eval.py',libero_python='/libero/python'))
   self.assertFalse(plan['execution_authorized']); self.assertEqual(plan['total_episodes'],280)
   self.assertEqual(plan['static_lifecycle_wiring']['server']['argv_template'][0],lifecycle.FIXED_OPENPI_PYTHON)
   self.assertEqual(plan['static_lifecycle_wiring']['websocket']['host'],'127.0.0.1')
   self.assertEqual(plan['static_lifecycle_wiring']['cleanup']['ownership'],'only child process groups created by the E1 supervisor')
   life=lifecycle.fake_lifecycle([sys.executable,'-c','import time; time.sleep(5)'],[sys.executable,'-c','pass'])
   self.assertTrue(life['server']['term_sent']); self.assertTrue(life['server']['reaped'])
   ledger={}
   for candidate in reg['candidates']:
    for entry in plan['development_entries']:
     e.record_episode(ledger,{'split':'development','suite':entry['suite'],'task_id':entry['task_id'],'initial_state_index':entry['initial_state_index'],'candidate_step':candidate['step'],'outcome':'success' if candidate['step']==1000 else 'policy_failure','infrastructure_retry':False})
   lock=e.select(ledger,reg); e.write_lock(root/'lock.json',lock); self.assertEqual(lock['step'],1000)
 def test_actual_remote_registration_shape_can_be_indexed(self):
  reg={'stage':'E-PREP-registration','identity_sha256':'d'*64,'candidates':[{'step':i*1000,'adapter_identity':(str(i)*64)[:64]} for i in range(1,8)]}
  index=build_e1_plan.build_candidate_index(reg,identities={'base':'b'*64},e0_manifest_sha256='e'*64,exception_decision_sha256='f'*64,tool_snapshot='/snapshot')
  self.assertEqual([x['step'] for x in index['registration_candidates']],[1000,2000,3000,4000,5000,6000,7000])
  self.assertEqual(index['registration_identity_sha256'],reg['identity_sha256'])
