# N1b memory diagnostic summary

## Outcome

The diagnostic passed and isolated the R2 linear RSS growth to the full normalization input path, which decodes and batches image fields even though `RunningStats` consumes only `state` and `actions`.

A state/action-only path reproduced the canonical normalization inputs exactly on all 12 pre-registered representative frames, including episode-boundary clamping and delta-action conversion.

## Controlled comparisons

| Variant | Batches | Frames | Elapsed | First RSS | Last RSS | RSS growth | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| State/action only, release every 10 | 500 | 16,000 | 40.55 s | 1,249,644,544 B | 1,019,600,896 B | -230,043,648 B | pass |
| State/action only, no explicit release | 500 | 16,000 | 38.82 s | 1,251,450,880 B | 1,252,003,840 B | +552,960 B | pass |

The no-release variant grew by only about 1,108 B per batch and remained at approximately 1.25 GB RSS. Therefore `gc.collect`, PyArrow pool release, and `malloc_trim` are not required for stability. Removing image decoding/batching is the decisive change.

For comparison, the original R2 full-transform path rose from about 2.13 GB to 8.70 GB and was stopped at batch 1,862 by the 8-GiB guard.

## Correctness evidence

- 12/12 representative frames: lean `state` exactly equals full transformed `state`.
- 12/12 representative frames: lean delta `actions` equals full transformed `actions` within absolute tolerance `1e-6`.
- Output statistic shapes remained `state=(8,)` and `actions=(7,)` for mean, std, q01, and q99.
- Both runs were offline, CPU-only, guarded, and left the canonical target absent.
- Hugging Face cache allocation remained `80,674,393,600 B`.
- No diagnostic process remained after completion.
- The remote OpenPI worktree retained only its pre-existing G3b changes.

## Evidence hashes

- Release diagnostic report: `5a8a5c298ca620fe71fed99c483319b8af8a90648c106ce97a4f85b029caad97`
- Release guard exit: `444cb86200f3acedd31a73b17c51e1352cc011046a8ab895fe790ce88a714aac`
- No-release diagnostic report: `e0aceeb74bbfa4179185d9b80be91938e3e6e7d44778e9e18d36252b837a573f`
- No-release guard exit: `15feb582d8481656e916eb096cfea1cf12203cca04dae303f8a7e355af8f481b`
- Diagnostic tool: `1d6e2931c47cb5d58f23fae2e1c44fae18130e5617b461a3b73a822a02a367cf`
- Unit test: `e2e235510aa68143c0ddbdd1526c619c915b223ef1653b32d1bd404eb9f4d3c8`

## R3 boundary

R3 may use the validated state/action-only input path without allocator-release calls. It must retain the same data/source identities, exact episode-end clamp semantics, delta conversion, `RunningStats`, batch size 32, source `drop_last=True` remainder disclosure, atomic publish protocol, and existing guard. R3 is not authorized by this diagnostic.
