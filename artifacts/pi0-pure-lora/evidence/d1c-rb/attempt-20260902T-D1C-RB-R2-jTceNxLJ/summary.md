# D1c-Rb retained-cache offline reuse validation

## Result

- Status: PASS
- Dataset: `physical-intelligence/libero`
- Revision: `a4336d589d589045d1c56423ffdf3b88a0e19b1f`
- Runtime length: 273,465 frames
- Metadata: 1,693 episodes, 40 tasks, 10 fps
- Guard result: `completed`, child exit 0, wait/reap confirmed
- Runtime: 21.949 seconds

## Retained identities

- Raw snapshot: 1,699 files, 34,938,927,454 bytes
- Raw snapshot identity SHA-256: `64725f83deff33829abf93169602199ec15dbc090756dafe462c77386ef3d85e`
- Arrow cache: 70 shards, 34,941,009,190 bytes
- Arrow path/size manifest SHA-256: `44c5fbe41202cbd29cf209d2856b4e95857f89728697a994d9aa14e0f2c5d700`
- Arrow builder/config/version/fingerprint: `parquet` / `default-f7c44f87cc5984aa` / `0.0.0` / `9c460aabd2aa27d1496e5e38d2060760561f0ac2cd6a110134eefa5b3f153b8d`
- Exactly one raw root and one Arrow tree existed before and after.
- No added or removed HF_HOME files and no HF_HOME byte growth occurred.
- Two pre-existing zero-byte Hugging Face lock files changed only in mtime; no unexpected file changes occurred.

## Resource evidence

- Five non-overlapping roots monitored.
- Stage soft/hard growth limits: 500,000,000 / 1,000,000,000 bytes.
- Maximum observed stage growth: 27,648 bytes, only under evidence output.
- HF_HOME guard allocation before/after: 80,674,393,600 bytes, unchanged.
- Peak observed child RSS: 1,022,668,800 bytes, below 8 GiB.
- Minimum observed MemAvailable: 269,043,109,888 bytes, above 64 GiB.
- Maximum observed load1/logical CPU: 0.5224513, below 0.9.
- One final RSS read raced with normal child exit; the guard then observed child exit 0. The configured two-consecutive-monitor-failure fail-closed threshold was not reached.
- No verifier/guard process remained after exit.

## Evidence hashes

- Verifier: `180bf5f782beb11f6846198c1103a3952b2928e64cdd579b61058344d9885f38`
- Guard: `427e9d32a3405ac1ea35245ca7f8c5dc63f1389cfff44ee2894bf8fb6e683721`
- Reuse report: `787b117385dd4f02ccb6440b7052117d74f1ed8566d1397c183ca7c4d9d3df8a`
- Exit status: `c73c8709d504ee0e08c92a4c9fe1a6261e6073321d91c1fb175f49cfa53392c5`
- Run manifest: `03f61dd78e2b81dc82743bf012054d7b4f1330cb0c78b1b7d6261187191f3002`
- Samples: `cec215d410544401766b0ac18ccb6366509830c33d3d0e32c9486c24791de2b2`

The first attempt at `../attempt-20260902T-D1C-RB-21msdpLg` is retained as a fail-closed diagnostic. It proved successful data construction and zero byte growth but rejected the two normal lock mtime touches. The revised rule and its 10 fake tests are recorded in D1c-Ra revision 2.

LeRobot emitted a compatibility warning that this is a 2.0-format dataset using global rather than per-episode stats. No conversion command was run and no dataset files were changed.
