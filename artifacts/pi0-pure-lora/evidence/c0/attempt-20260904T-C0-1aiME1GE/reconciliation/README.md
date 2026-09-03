# C0 cache reconciliation evidence

`report.json` is the preserved first draft. It correctly found one raw root,
one Arrow root, and unchanged full-HF bytes/files, but did not independently
compare the logical raw-dataset file/byte counts used by D1c-Rb.

`report_v2.json` is authoritative. It additionally separates Hugging Face
download metadata from logical dataset files and verifies the D1c-Rb values:
1,699 logical raw files and 34,938,927,454 bytes.
