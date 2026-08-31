# V0 R0/G1/G2 versioning closeout

V0 closes the reviewed R0/G1/G2 implementation into two independent Git
histories. It does not authorize or begin G3, data or checkpoint access, GPU
work, training, or evaluation.

## Remote OpenPI source identity

- Worktree: `/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora`
- Branch: `feature/pi0-libero-pure-lora` (no upstream configured)
- Base commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Pure-LoRA commit: `48d1847417356fb38ecb5db45b569f12b2d148e6`
- Tree: `4c8004c344e2fcd834a752078012c3fa8fb331a8`
- Commit scope: three source files, 114 insertions, no other paths
- Canonical committed diff SHA-256:
  `1902779007955a3da8b6a7ee9d0660f030e1cf66c09b6751a566a0ead1066b5a`

The deployment baseline remains on `main` at the same base commit with a clean
tracked tree and only its pre-existing untracked `outputs/` directory.

## Exact pure-LoRA proof

The frozen Golden manifest SHA-256 is
`3799cf4d053b013089216be97ab0b57d08dde1dd3c4f04744088ce2e93a32029`.
The existing CPU-only G2 evidence was rechecked before commit and proves:

- trainable paths equal the independently frozen Golden set;
- 20 trainable leaves and 49,987,584 trainable parameters;
- all 50 non-Golden leaves and 3,238,048,528 parameters frozen;
- no missing, unexpected, protected, base-terminal, overlap, or partition
  failures.

An import-origin probe using the existing interpreter and explicit
`PYTHONPATH` resolved `openpi`, `pi0_config`, and `nnx_utils` only from the
isolated LoRA worktree.

## Patch roles

`patches/openpi_pure_lora_g2.patch` is the reviewed hand-authored application
artifact; its SHA-256 is
`a58256ca93f73a0ad06c727bc4752cef77e437e9b239770c7dcb72ede0da163c`.
`artifacts/pi0-pure-lora/g2/final_openpi_g2.patch` is the Git-generated
canonical diff of the verified checkout; its SHA-256 is
`1902779007955a3da8b6a7ee9d0660f030e1cf66c09b6751a566a0ead1066b5a`.
They encode the same applied source additions but are not byte-identical and
do not have the same stable patch-id: file order, blob-index metadata, and
generated hunk context differ.

## Evidence and boundary

The collision-safe remote V0 evidence directory is
`/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/v0/attempt-20260831T-V0-ozn5xi`.
It contains pre/post identities, the staged path set, diff checks, hashes,
commit output, import origins, existing-G2 assertions, and the Mac file audit.

No dependency or submodule setup, network download, checkpoint or dataset
access, model load, GPU operation, training, or evaluation was performed.
The Mac project commit and push identity are verified after this report is
created and therefore are reported by Git itself rather than self-recorded in
this commit-bound document.
