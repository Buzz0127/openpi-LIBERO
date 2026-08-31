# G2 completion report

G2 implements an exact-path pure-LoRA freeze filter for OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. It does not add a training
configuration or authorize checkpoint, data, GPU, or training work.

## Modified OpenPI files

- `src/openpi/shared/nnx_utils.py`: adds `PathIn`, an immutable exact full-path
  NNX filter with validation for empty, duplicate, or malformed paths.
- `src/openpi/models/pi0_config.py`: defines the 20 audited full adapter paths
  and `get_pure_lora_freeze_filter()`. It rejects every model variant pair
  except `gemma_2b_lora` plus `gemma_300m_lora`.
- `src/openpi/models/pi0_test.py`: adds an independently enumerated 20-path
  oracle, count assertions, partition assertions, and unaudited-variant
  rejection.

The applied source diff is stored as `final_openpi_g2.patch`. The patch input
SHA-256 is `a58256ca93f73a0ad06c727bc4752cef77e437e9b239770c7dcb72ede0da163c`.

## Decisive verification

`g2_filter_verification.json` records:

- `trainable_equals_golden: true`;
- `all_non_golden_params_frozen: true`;
- 20 trainable leaves and 49,987,584 trainable parameters;
- 50 frozen leaves and 3,238,048,528 frozen parameters;
- no missing Golden paths, unexpected trainables, protected trainables,
  base-terminal trainables, overlap, or partition errors.

Both the external frozen-manifest verifier and the two source-level G2 tests
ran on `TFRT_CPU_0` and exited successfully. No checkpoint or dataset was
accessed. Both child process groups were reaped and had no task-attributable
GPU process. One transient resource-read failure occurred during the source
test after the child was finishing; it did not reach the two-consecutive
failure stop condition.

The original G2 run modified the deployment baseline by mistake. R0 later
migrated the exact diff to
`/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`, reran the Golden
and source tests there, and restored the baseline tracked tree with the exact
reverse patch. The baseline retains only its pre-existing untracked `outputs/`.
See `artifacts/pi0-pure-lora/r0/r0_worktree_isolation_report.md` for the current
authoritative topology and evidence.
