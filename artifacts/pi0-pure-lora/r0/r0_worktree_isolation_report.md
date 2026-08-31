# R0 remote worktree isolation report

R0 corrected the earlier G2 execution-location error without deleting evidence
or touching the existing untracked `outputs/` directory.

## Final topology

- Read-only deployment baseline:
  `/home/wengzr/projects/openpi`
- Pure-LoRA worktree:
  `/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`
- Shared Git common directory:
  `/home/wengzr/projects/openpi/.git`
- LoRA branch:
  `feature/pi0-libero-pure-lora`
- Both worktrees start at:
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`

The baseline tracked tree is clean and retains only its pre-existing untracked
`outputs/`. The three G2 modifications exist only in the LoRA worktree. Their
diff SHA-256 is
`1902779007955a3da8b6a7ee9d0660f030e1cf66c09b6751a566a0ead1066b5a`.

## Import isolation

The existing interpreter
`/home/wengzr/projects/openpi/.venv/bin/python` is reused with an explicit
`PYTHONPATH` pointing to the LoRA worktree. `openpi`, `pi0_config`, and
`nnx_utils` all resolve under the LoRA worktree. No second environment was
created and no `.venv` exists inside the LoRA worktree.

The ALOHA and LIBERO submodules are intentionally uninitialized in the new
worktree. G2 parameter-tree verification does not use them. A later stage that
requires either submodule must receive a separate reuse/setup review.

## Verification

The migrated G2 filter again satisfies:

- `trainable paths == frozen Golden paths`;
- 20 trainable leaves and 49,987,584 trainable parameters;
- all 50 non-Golden leaves and 3,238,048,528 parameters frozen;
- no missing, extra, protected, base-terminal, overlap, or partition failures.

The external Golden verifier and both source-level G2 tests completed on CPU.
Both task process groups were reaped, no task GPU process was attributable, and
no source-tree `__pycache__` was created.

The LoRA branch currently has no upstream and its three source changes remain
uncommitted for review. No push or commit was performed.
