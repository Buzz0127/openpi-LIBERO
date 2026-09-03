# D1c-Ra revision 2 evidence

- Trigger: the first real D1c-Rb run showed that Hugging Face touches the mtime of two existing zero-byte lock files while reusing the retained Arrow cache.
- Scope: local verifier/test revision only; no server data was accessed by these tests.
- Verifier SHA-256: `180bf5f782beb11f6846198c1103a3952b2928e64cdd579b61058344d9885f38`
- Test SHA-256: `eb7d880ebdf78849696afa4228063e7009554211e20f20379f863a36aa5d52c6`
- `py_compile`: pass
- Unit tests: 10 run in 0.040 seconds, all pass

The new rule permits only an existing path under `datasets/` ending in `.lock` whose file remains zero-byte, single-linked, non-symlinked, and identical in every recorded field except `mtime_ns`. Added lock files, size/allocation/link changes, and all non-lock metadata changes remain failures.
