# N1b R3 completion summary

## Outcome

R3 passed. The validated state/action-only normalization path processed all 8,545 OpenPI full-run batches and atomically published canonical LIBERO target-domain normalization statistics.

## Canonical identity

- Remote asset identity: `/home/wengzr/projects/openpi-lora-assets/libero_pi0_delta_zscore_train1693_v1`
- File: `norm_stats.json`
- SHA-256: `f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`
- Size: 1,951 B
- Native dimensions: `state=8`, `actions=7`
- Protocol: Pi0 z-score input statistics after LIBERO repack and six-dimensional delta-action conversion
- Dataset: 273,465 frames, batch size 32, 8,545 batches, 273,440 processed frames, 25 source `drop_last=True` remainder frames

The file was written to a same-filesystem temporary directory, deserialized and validated, fsynced, and published with one directory `os.replace`.

## Correctness and runtime

- Automation status: pass
- Guard status: completed; child return code 0; child reaped
- R3 child elapsed: 582.89 seconds
- End-to-end automation elapsed: 601.12 seconds
- Probe equivalence: 12/12 state and delta-action checks passed against the full transform path
- Statistic shapes: state `(8,)`, actions `(7,)` for mean, std, q01, and q99
- RSS: 1,247,461,376 B at batch 1 and 1,263,079,424 B at batch 8,545
- Guard peak RSS: 1,271,840,768 B
- Minimum observed MemAvailable: 268,846,317,568 B
- Maximum load1/logical-CPU: 0.023965
- Stage storage growth: 213,660 B
- Hugging Face cache allocation remained 80,674,393,600 B
- No R3 process remained after completion
- Remote OpenPI worktree retained only its pre-existing G3b changes

## Official checkpoint comparison

The new canonical stats and official `pi0_libero` checkpoint stats are not the same normalization artifact:

- Canonical native dimensions: state 8, actions 7
- Official dimensions: state 32, actions 32
- File hashes differ
- Shared native numeric leaves are not exactly equal
- Maximum absolute difference across shared leaves: 0.02160626235008234 (`state.q01[4]`)
- Mean absolute difference across shared leaves: 0.002199438954866469

Therefore Base and pure-LoRA form the controlled primary comparison by explicitly sharing the new canonical stats. Official `pi0_libero` must retain its checkpoint-owned stats and be described as an end-to-end external reference with a different preprocessing/normalization artifact.

## Evidence hashes

- `automation_status.json`: `b0163175ffb512ce60e67c0b03ddcac0d76347c127b9238e98c4ef4bac443bbe`
- `comparison_report.json`: `f9321684d85e6152192ecd50468e0aab0919c995752f5eb66cfa938ca1885674`
- `guard-run/r3_report.json`: `ef508114c87cb518eca663b8e6731d5865187403166961626f61f28d1bd12ea3`
- `guard-run/exit_status.json`: `0614d80ed8de110e00bcd368d5acfc2e3c0250f61935b7965d68810d24405720`
- `canonical/norm_stats.json`: `f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`

R3 does not authorize checkpoint loading, GPU work, training, or evaluation.
