from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_future_formal_readiness as builder
import experiment_identity


class FutureReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="future-readiness-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.decision = self.root / "decision.json"
        decision = {"schema_version": 1, "stage": "C-FT2-controlled-exception", "decision": "accept-existing-ft1-ft2-chain",
                    "execution_authorized": False, "next_stage_auto_start": False}
        decision["decision_identity_sha256"] = experiment_identity.canonical_sha256(decision)
        self.decision.write_text(json.dumps(decision), encoding="utf-8")
        self.package = self.root / "ft0.json"
        package = {"schema_version": 1, "stage": "FT0-formal-training-freeze", "execution_authorized": False,
                   "automatic_next_stage": False, "formal_training": {"segments": [{"start": 0, "end": 1000}, {"start": 1000, "end": 5000},
                   *[{"start": start, "end": end} for start, end in builder.formal_future_schedule.FUTURE_SEGMENTS] ]}}
        package["package_identity_sha256"] = builder.formal_segment_contract._canonical(package)
        self.package.write_text(json.dumps(package), encoding="utf-8")
        self.tools = {}
        for name in sorted(builder.REQUIRED_TOOLS):
            path = self.root / f"{name}.py"; path.write_text(name, encoding="utf-8"); self.tools[name] = path

    def test_writes_non_authorizing_registered_future_schedule(self) -> None:
        output = self.root / "readiness.json"
        argv = ["builder", "--output", str(output), "--c-ft2-decision", str(self.decision), "--ft0-package", str(self.package)]
        for name, path in self.tools.items(): argv.extend(["--tool", f"{name}={path}"])
        with mock.patch("sys.argv", argv): self.assertEqual(builder.main(), 0)
        value = json.loads(output.read_text())
        self.assertFalse(value["execution_authorized"])
        self.assertEqual([(row["start"], row["end"]) for row in value["future_segments"]], list(builder.formal_future_schedule.FUTURE_SEGMENTS))

    def test_rejects_noncanonical_c_ft2_decision(self) -> None:
        value = json.loads(self.decision.read_text()); value["decision"] = "forged"; self.decision.write_text(json.dumps(value), encoding="utf-8")
        argv = ["builder", "--output", str(self.root / "no.json"), "--c-ft2-decision", str(self.decision), "--ft0-package", str(self.package)]
        for name, path in self.tools.items(): argv.extend(["--tool", f"{name}={path}"])
        with mock.patch("sys.argv", argv), self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            builder.main()
