# D1c-Ra local verifier hardening evidence

- Date: 2026-09-02 Asia/Shanghai
- Workspace: `/Users/buzz/MyProjects/openpi-LIBERO-lora`
- Branch: `feature/pi0-libero-pure-lora`
- HEAD: `7b3e371f9a8ef6f3f181e55bbe9e5708671fcd9c`
- Upstream: `origin/feature/pi0-libero-pure-lora` (`+0 -0`)
- Scope: local verifier implementation, static compilation, and fake-tree tests only
- Explicit exclusions: no SSH, network, real LIBERO data, LeRobot import, checkpoint, GPU, training, evaluation, deletion, commit, or push

## Files

- `tools/verify_lerobot_offline_reuse.py`
  - SHA-256: `0e6acdc66c80ad9a69056bc3fa0d972a32f3ee0291b9af89127bcdc4ead6cd96`
  - 451 lines, 20,076 bytes
- `tools/test_verify_lerobot_offline_reuse.py`
  - SHA-256: `029fae11d06b2e206b1743fc25a6a930f9bda2ff1ff0f3405c4ae48eff5d7e1e`
  - 199 lines, 8,219 bytes

## Checks performed

`/usr/bin/python3 -m py_compile` passed with `PYTHONDONTWRITEBYTECODE=1` and the bytecode cache redirected into this attempt.

`/usr/bin/python3 -m unittest -v tools/test_verify_lerobot_offline_reuse.py` passed:

- 8 tests run
- elapsed: 0.034 seconds
- result: OK

The fake tests cover stable reuse, full-HF_HOME duplicate raw detection, duplicate Arrow tree and 100,000,001-byte file detection, offline/proxy preflight fail-closed behavior, explicit Arrow identity, symlink rejection, hardlink rejection, and immutable report output.

## Boundary and result

The verifier now requires pinned raw snapshot evidence, exact `HF_HOME` / `HF_DATASETS_CACHE` / `HF_LEROBOT_HOME`, offline flags, no proxies, exactly one raw root and one Arrow tree, explicit builder/config/version/fingerprint/shard-count/manifest identity, and unchanged full-HF_HOME metadata before and after dataset construction. The real LeRobot import remains deferred to the CLI runtime path and was not imported in this stage.

D1c-Ra result: PASS for local static and fake-tree validation. This is not D1c-Rb evidence and does not prove the retained server cache is reusable.
