# G1b completion report

G1b is complete for OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and model variants
`gemma_2b_lora` plus `gemma_300m_lora`.

The CPU-only real `nnx.eval_shape` tree contains 70 `nnx.Param` leaves and
3,288,036,112 parameters. The frozen golden adapter manifest contains exactly
20 leaves and 49,987,584 parameters. The remaining 50 leaves and
3,238,048,528 parameters are non-adapters and must remain frozen.

`manifests/pi0_pure_lora/golden_adapter_paths.json` is the only G2 test oracle.
It uses exact full-path membership; substring and regex matching are not
allowed. Its SHA-256 is
`3799cf4d053b013089216be97ab0b57d08dde1dd3c4f04744088ce2e93a32029`.

`golden_manifest_final_verification.json` proves that:

- the manifest equals the independently reconstructed set from six source
  terminals and eight legal parent paths;
- every path, shape, dtype, variable type, and parameter count matches the
  captured real tree;
- no protected non-adapter scope intersects the manifest;
- the pinned `lora.py`, `gemma.py`, `pi0_config.py`, and `pi0.py` hashes match.

The earlier `golden_manifest_verification.json` is retained as historical
evidence for the proposed-manifest stage. It is superseded by
`golden_manifest_final_verification.json` for G2.

This completion does not implement or authorize the G2 freeze filter, model
loading, checkpoint access, data access, GPU work, or training.
