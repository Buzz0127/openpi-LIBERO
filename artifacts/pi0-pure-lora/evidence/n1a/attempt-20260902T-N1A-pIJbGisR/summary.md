# N1a canonical normalization input audit summary

## Result

N1a passed. This attempt audited the canonical LIBERO normalization input protocol and ran a bounded 12-probe, CPU-only, offline smoke. It did not compute or write canonical normalization statistics.

## Bound identities

- Remote OpenPI worktree: `/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`
- Remote Git HEAD: `48d1847417356fb38ecb5db45b569f12b2d148e6`
- Config: `pi0_libero_pure_lora`
- Dataset revision: `a4336d589d589045d1c56423ffdf3b88a0e19b1f`
- Dataset root: `/home/wengzr/projects/openpi-lora-cache/huggingface/lerobot/physical-intelligence/libero`
- Dataset: 273,465 frames, 1,693 episodes, 40 tasks
- D1c report SHA-256: `787b117385dd4f02ccb6440b7052117d74f1ed8566d1397c183ca7c4d9d3df8a`
- Transform identity SHA-256: `9f373bd1b879a13c04cd0adc35cb6d1ff6fb281475318eaaa95cbe6a24781ecb`
- Canonical target: `/home/wengzr/projects/openpi-lora-assets/libero_pi0_delta_zscore_train1693_v1`
- Canonical target before and after: absent

The normalization input chain is exactly:

1. `RepackTransform` maps raw LIBERO image, wrist image, state, actions, and prompt fields.
2. `LiberoInputs(model_type=pi0)` prepares policy inputs.
3. `DeltaActions(mask=(True, True, True, True, True, True, False))` subtracts state from the first six action dimensions and leaves the gripper dimension absolute.
4. `RunningStats` receives only `state` and `actions` and uses z-score statistics for Pi0.

Model tokenization, padding to the model action dimension, checkpoint loading, and model forward execution are outside this normalization-input stage.

## Probe evidence

- 12 deterministic probes passed across dataset start/middle/end and the four task-suite blocks.
- Raw state shape was `(8,)`; raw/transformed action shape was `(50, 7)`.
- Image keys, `(256, 256, 3)` shapes, `uint8` dtype, prompts, finite values, and exact delta conversion all passed.
- Episode-boundary semantics passed. At episode 0 frame 213, for example, 49 future positions were marked padded and every query index was clamped to frame 213. No action was read from the next episode.
- The in-memory probe statistics are explicitly non-canonical and were not saved as `norm_stats.json`.

## Guard and storage evidence

- Guard reason: `completed`; child exit code: 0; child reaped: true.
- Elapsed: 21.747 seconds for the guarded stage; audit child elapsed 18.586 seconds.
- Samples: 8 total, 7 successful resource samples.
- Peak sampled child RSS: 1,227,202,560 B.
- Minimum sampled `MemAvailable`: 268,896,415,744 B.
- Maximum sampled load1/logical-CPU: 0.1876125.
- Final stage growth: 27,648 B, only under `openpi-eval-tools`.
- Hugging Face cache growth: 0 B.
- Canonical asset-root growth: 0 B.
- No TERM or KILL was required; no N1a process remained after the run.
- One final monitor sample raced with normal child exit and reported `VmRSS not found`; the configured two-failure threshold was not reached and the child had already exited successfully.

## Evidence hashes

- `audit_libero_norm_inputs.py`: `e068599c400a2c80daeb369340ad77d2de0e523a646112b848f8a1e712145379`
- `test_audit_libero_norm_inputs.py`: `766c1db0f5c6404646fc553f3f4f4643e231fee52e3304e60bb4469f9074210e`
- `storage_budget_guard.py`: `427e9d32a3405ac1ea35245ca7f8c5dc63f1389cfff44ee2894bf8fb6e683721`
- `guard-run/n1a_report.json`: `3026e60cfb1933a1720c1f8bd6cf8a39ae065e0b76df9788991a1d47157b7897`
- `guard-run/exit_status.json`: `1afdd4c80425bab303952c870fa388f58a62fceb5ee93739b91b28cc686aea7b`
- `guard-run/run_manifest.json`: `7a1b6d3c213952ddfd13fa4baf77ac05fac5bc1d0a3f7cbc89a329fc38cf8ff5`
- `guard-run/samples.jsonl`: `66866ad98f87bff371b450a58fa1f7e2af0c01bdfbd807e0addbafa0c6a985d9`
- `guard-run/child.stdout.log`: `7c07d7adcc0b17791a8204c25f5e6ef359dfc13dddd32c12648561c4f0a22452`
- `guard-run/child.stderr.log`: `c48112475099d26f5a315c8894917060f998ce46ef693b04ab44541eb2b728a1`

Remote and local copies matched for every hash listed above before this summary was added.

## Preserved state and limitations

The remote LoRA worktree was already dirty before N1a and remained unchanged:

```text
 M scripts/compute_norm_stats.py
 M src/openpi/training/config.py
?? scripts/compute_norm_stats_test.py
?? src/openpi/training/config_test.py
```

Those G3b drafts are therefore bound by exact file hashes in `n1a_report.json`, but they are not committed. N1b must either bind and accept those exact hashes explicitly or first place them into a separately approved, reproducible source snapshot.

LeRobot emitted its existing v2.0/global-stats compatibility warning during metadata and dataset construction. No conversion or cache rewrite was performed. The actual offline snapshot and loader-required Arrow cache were reused without growth.

The current OpenPI normalization script includes loader-clamped padded tail actions in `RunningStats`. N1a verified that this means repeated episode-end actions, never actions from the next episode. Excluding padded positions would define a different normalization protocol and requires an explicit route decision before N1b.
