"""CPU-only E-PREP control: candidate ledger, episode accounting, selection lock."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import signal, subprocess, time

def _canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _new(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_name("."+path.name+".%d.tmp"%os.getpid())
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.link(tmp,path); tmp.unlink()

def register(candidates, *, identities):
    """Freeze exactly the supplied step->adapter map before first dev episode."""
    if len(candidates)!=7 or len(set(candidates))!=7: raise ValueError("requires seven unique candidates")
    rows=[]
    for step, adapter in sorted(candidates.items()):
        if not isinstance(step,int) or len(adapter)!=64: raise ValueError("invalid candidate")
        rows.append({"step":step,"adapter_identity_sha256":adapter})
    out={"schema_version":1,"stage":"E-PREP-candidate-registration","candidates":rows,"identities":identities,"main_may_select":False}
    out["identity_sha256"]=_canon(out); return out

def episode_key(split,suite,task,state,candidate): return f"{split}:{suite}:{task}:{state}:{candidate}"
def record_episode(ledger, row):
    required={"split","suite","task_id","initial_state_index","candidate_step","outcome","infrastructure_retry"}
    if set(row)!=required or row["outcome"] not in {"success","policy_failure","infrastructure_failure"}: raise ValueError("invalid episode row")
    key=episode_key(row["split"],row["suite"],row["task_id"],row["initial_state_index"],row["candidate_step"])
    if key in ledger: raise ValueError("duplicate episode")
    if row["outcome"]=="policy_failure" and row["infrastructure_retry"]: raise ValueError("policy failures cannot retry")
    ledger[key]=dict(row); return key

def select(ledger, registration):
    rows=registration["candidates"]; scored=[]
    for c in rows:
        adapter=c.get("adapter_identity_sha256",c.get("adapter_identity"))
        if not isinstance(adapter,str) or len(adapter)!=64:
            raise ValueError("invalid registered adapter identity")
        rs=[r for r in ledger.values() if r["split"]=="development" and r["candidate_step"]==c["step"]]
        if len(rs)!=40 or any(r["outcome"]=="infrastructure_failure" for r in rs): raise ValueError("incomplete development denominator")
        scored.append((sum(r["outcome"]=="success" for r in rs),c["step"],adapter))
    best=sorted(scored,key=lambda x:(-x[0],x[1],x[2]))[0]
    return {"schema_version":1,"stage":"E1-selection-lock","dev_successes":best[0],"step":best[1],"adapter_identity_sha256":best[2],"registration_identity_sha256":registration["identity_sha256"]}

def write_lock(path, lock):
    """Publish selection once; a main run must reference this immutable file."""
    lock=dict(lock); lock["lock_identity_sha256"]=_canon(lock); _new(path,lock); return lock

def owned_fake_lifecycle(command, timeout=2):
    """CPU-only lifecycle proof: controls only the process group it created."""
    p=subprocess.Popen(command,start_new_session=True)
    pgid=p.pid; deadline=time.monotonic()+timeout
    while p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
    terminated=False
    if p.poll() is None:
        os.killpg(pgid,signal.SIGTERM); terminated=True; p.wait(timeout=1)
    return {"pgid":pgid,"returncode":p.returncode,"terminated":terminated,"reaped":p.poll() is not None}
