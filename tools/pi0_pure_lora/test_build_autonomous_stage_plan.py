#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import autonomous_stage_orchestrator as orchestrator
import build_autonomous_stage_plan as builder


class BuildAutonomousStagePlanTest(unittest.TestCase):
    def test_plan_binds_source_tools_inputs_and_command(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a2-plan-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "a2", str(source)], check=True)
            (source / "fixed").write_text("fixed\n", encoding="utf-8")
            environment = {**os.environ, "GIT_AUTHOR_NAME": "A2", "GIT_AUTHOR_EMAIL": "a2@example.invalid", "GIT_COMMITTER_NAME": "A2", "GIT_COMMITTER_EMAIL": "a2@example.invalid"}
            subprocess.run(["git", "-C", str(source), "add", "fixed"], check=True, env=environment)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "fixed"], check=True, env=environment)
            head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            attempt = root / "attempt-01"
            progress = attempt / "progress.json"
            output = attempt / "probe.json"
            plan = root / "plan.json"
            arguments = [
                "builder", "--output", str(plan), "--attempt-dir", str(attempt),
                "--current-json", str(root / "current.json"), "--source-root", str(source),
                "--expected-branch", "a2", "--expected-head", head, "--expected-upstream", "NONE",
                "--orchestrator", str(Path(orchestrator.__file__)), "--launcher", str(Path(__file__)),
                "--stage", "A2", "--segment-start", "0", "--segment-end", "3",
                "--train-seed", "42", "--eval-seed", "7", "--progress-json", str(progress),
                "--expected-final-committed-step", "3", "--required-output", str(output),
                "--timeout-seconds", "30", "--heartbeat-timeout-seconds", "5",
                "--sample-interval-seconds", "1", "--term-grace-seconds", "2",
                "--kill-grace-seconds", "1", "--max-retries", "0", "--retry-delay-seconds", "1",
                "--max-log-bytes", "4096", "--log-backups", "1",
            ]
            for index, key in enumerate(sorted(builder.orchestrator.REQUIRED_IDENTITIES), 1):
                arguments.extend(["--identity", f"{key}={index:064x}"])
            arguments.extend(["--", sys.executable, "-c", "print('probe')"])
            original = sys.argv
            try:
                sys.argv = arguments
                self.assertEqual(builder.main(), 0)
            finally:
                sys.argv = original
            value = json.loads(plan.read_text())
            claimed = value.pop("plan_identity_sha256")
            self.assertEqual(orchestrator.experiment_identity.canonical_sha256(value), claimed)
            self.assertFalse(value["execution_authorized"])
            self.assertFalse(value["next_stage_auto_start"])
            self.assertEqual(value["source"]["head"], head)


if __name__ == "__main__":
    unittest.main()
