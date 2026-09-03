# N1a local static and fake-tree test report

- Scope: syntax validation and fake-object unit tests only; no real OpenPI import, dataset access, checkpoint access, network, or GPU.
- Interpreter: `/usr/bin/python3` locally; the same tool/test files were subsequently checked remotely with `/home/wengzr/projects/openpi/.venv/bin/python`.
- `py_compile`: passed.
- Unit tests: 5 passed in 0.002 seconds locally.
- Remote static/fake tests: 5 passed in 0.036 seconds.
- Pre-existing G3b tests on the remote pinned environment: 8 passed in 0.002 seconds (process wall time approximately 26.9 seconds due to imports).
- Audit tool SHA-256: `e068599c400a2c80daeb369340ad77d2de0e523a646112b848f8a1e712145379`.
- Unit-test SHA-256: `766c1db0f5c6404646fc553f3f4f4643e231fee52e3304e60bb4469f9074210e`.
- Storage guard SHA-256: `427e9d32a3405ac1ea35245ca7f8c5dc63f1389cfff44ee2894bf8fb6e683721`.

The tests exercise deterministic probe selection, exact action-query clamp semantics, fail-closed checks, and report structure without importing the real Pi0 model.
