"""Build an immutable non-executing E1 dev plan from frozen inputs."""
import argparse,hashlib,json
from pathlib import Path
import evaluation_control as c
import evaluation_lifecycle as lifecycle

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def normalized_candidates(registration):
 """Accept the immutable v1 remote registry without weakening its identity."""
 raw=registration.get('candidates',[])
 if registration.get('stage') not in {'E-PREP-candidate-registration','E-PREP-registration'} or len(raw)!=7: raise ValueError('requires frozen seven-candidate registration')
 rows=[]
 for row in raw:
  adapter=row.get('adapter_identity_sha256',row.get('adapter_identity'))
  if not isinstance(row.get('step'),int) or not isinstance(adapter,str) or len(adapter)!=64: raise ValueError('invalid registered candidate')
  rows.append({'step':row['step'],'adapter_identity_sha256':adapter,'source':dict(row)})
 if len({x['step'] for x in rows}) != 7 or len({x['adapter_identity_sha256'] for x in rows}) != 7: raise ValueError('candidate set is not unique')
 return sorted(rows,key=lambda x:x['step'])

def build_candidate_index(registration, *, identities, e0_manifest_sha256, exception_decision_sha256, tool_snapshot):
 """Make the full candidate/identity index that the legacy registry lacks."""
 x={'schema_version':1,'stage':'E-PREP-candidate-index','registration_identity_sha256':registration['identity_sha256'],
    'registration_stage':registration['stage'],'registration_candidates':normalized_candidates(registration),
    'identities':dict(identities),'e0_manifest_sha256':e0_manifest_sha256,
    'c_ft2_exception_decision_sha256':exception_decision_sha256,'tool_snapshot':tool_snapshot,
    'execution_authorized':False,'main_may_select':False}
 x['identity_sha256']=c._canon(x); return x

def build(registration, manifest, candidate_index, *, video_bytes, log_bytes, wiring):
 normalized_candidates(registration)
 if candidate_index.get('registration_identity_sha256') != registration.get('identity_sha256'): raise ValueError('candidate index does not bind registration')
 entries=[x for x in manifest.get('entries',[]) if x.get('split')=='development']
 if len(entries)!=40 or len({(x['suite'],x['task_id'],x['initial_state_index']) for x in entries})!=40: raise ValueError('E0 must contain exactly forty unique development states')
 x={'schema_version':2,'stage':'E1-dev-40','execution_authorized':False,'main_may_select':False,
    'registration_identity_sha256':registration['identity_sha256'], 'candidate_index_identity_sha256':candidate_index['identity_sha256'],
    'e0_manifest_identity_sha256':sha256_file(manifest['_path']), 'candidate_count':7,'episodes_per_candidate':40,'total_episodes':280,
    'development_entries':sorted(entries,key=lambda z:(z['suite'],z['task_id'],z['initial_state_index'])),
    'video_budget_bytes':video_bytes,'log_budget_bytes':log_bytes,'requires_fresh_gpu_preflight':True,
    'infrastructure_retry_rule':'record every infrastructure failure; retry only the identical E0 key under the future bounded E1 authorization; policy failures are never retried',
    'selection_rule':['maximum development successes','lowest train step','lexicographically smallest adapter identity'],
    'static_lifecycle_wiring':wiring}
 x['plan_identity_sha256']=c._canon(x); return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--registration',required=True,type=Path);p.add_argument('--candidate-index',required=True,type=Path);p.add_argument('--e0-manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--openpi-root',required=True);p.add_argument('--evaluator',required=True);p.add_argument('--libero-python',required=True);p.add_argument('--video-bytes',type=int,default=0);p.add_argument('--log-bytes',type=int,default=50_000_000);a=p.parse_args()
 if a.output.exists() or a.video_bytes<0 or a.log_bytes<=0: raise ValueError('new bounded output required')
 r=json.loads(a.registration.read_text()); m=json.loads(a.e0_manifest.read_text());m['_path']=str(a.e0_manifest);i=json.loads(a.candidate_index.read_text())
 x=build(r,m,i,video_bytes=a.video_bytes,log_bytes=a.log_bytes,wiring=lifecycle.static_wiring(openpi_root=a.openpi_root,evaluator=a.evaluator,libero_python=a.libero_python)); c._new(a.output,x); print(json.dumps(x,sort_keys=True))
if __name__=='__main__': main()
