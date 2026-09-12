# E1 dev-40 launch note — prepared, not authorized

This note binds the future development evaluation to seven already trained
pure-LoRA candidates at steps `1000/5000/10000/15000/20000/25000/30000` and
the forty E0 `development` task-state entries.  It does **not** authorize a
server, model, GPU, environment, episode, download, deletion, or Git action.

Before an E1 launch can be considered, its immutable registration/index/plan
must be present and mutually hash-bound.  The runner must then do a fresh
30-second CPU/RAM/two-GPU sample, select one physical GPU with more than 15%
free VRAM, disable JAX preallocation, and run only inside the verified
task-owned process-group guard.  The first real episode is part of the same
candidate's pre-registered dev-40 denominator; it is not a disposable probe.

The static lifecycle is deliberately explicit: fixed OpenPI Python starts the
policy server with a materialized policy directory bound to base, adapter,
canonical-normalization and model manifests; the LIBERO Python evaluator then
connects over `127.0.0.1:{port}`.  TCP readiness is insufficient: the episode
must additionally yield finite actions and a structured result.  On every
exit path the supervisor may TERM/KILL and reap only the server/evaluator
process groups it created.

Each candidate consumes exactly the same 40 development entries.  Policy
failures count and are never retried.  Infrastructure faults retain their
record and may be retried only as the identical pre-registered key under the
future bounded E1 authorization.  Selection is fixed before E1: highest
development successes, then lower train step, then lexicographically smaller
adapter identity.  The immutable selection lock is required before E2; E2 is
not authorized by this note.

No wall-clock performance estimate is asserted here.  It must be measured
from the first authorized development episode, not inferred from training.
