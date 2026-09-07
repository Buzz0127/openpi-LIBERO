#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_a2_autonomy as verifier


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class VerifyA2AutonomyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="a2-verify-")
        self.root = Path(self.temp.name)
        self.attempt = self.root / "attempt-a2-test"
        self.attempt.mkdir()
        self.plan = self.root / "plan.json"
        plan = {
            "plan_identity_sha256": "a" * 64,
            "expected_final_committed_step": 10,
            "required_outputs": [str((self.attempt / "probe-result.json").resolve())],
        }
        _write_json(self.plan, plan)
        self.plan_sha = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        common = {"attempt_id": self.attempt.name}
        _write_json(self.attempt / "status.json", {**common, "status": "pass", "exit_reason": "completed", "current_step": 10, "last_committed_step": 10, "stage_plan_identity_sha256": "a" * 64, "stage_plan_sha256": self.plan_sha})
        _write_json(self.attempt / "summary.json", {**common, "status": "pass", "reason_code": "completed", "retry_count": 0, "next_stage_started": False, "last_progress": {"last_committed_step": 10}, "child_runs": [{"child_pid": 7, "child_pgid": 7, "returncode": 0, "wait_reaped": True, "term_sent": False, "kill_sent": False}]})
        _write_json(self.attempt / "run_manifest.json", {**common, "next_stage_auto_start": False, "stage_plan_identity_sha256": "a" * 64, "stage_plan_sha256": self.plan_sha, "bounds": {"max_log_bytes": 10, "log_backups": 1}})
        _write_json(self.attempt / "progress.json", {"current_step": 10, "last_committed_step": 10})
        _write_json(self.attempt / "probe-result.json", {"status": "pass", "steps": 10, "pid": 7, "pgid": 7, "next_stage_started": False, "offline_environment_verified": True, "proxies_unset": True})
        _write_json(self.attempt / "launcher_receipt.json", {"attempt_dir": str(self.attempt.resolve()), "heartbeat_sequences": [1, 2], "tmux_session_alive": True, "launcher_detaches_after_verification": True})
        (self.attempt / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (self.attempt / "child-00.stdout.log").write_bytes(b"ok")
        files = ["status.json", "summary.json", "run_manifest.json", "progress.json", "probe-result.json", "launcher_receipt.json", "exit_code.txt", "child-00.stdout.log"]
        (self.attempt / "output-files.sha256").write_text("".join(f"{hashlib.sha256((self.attempt / name).read_bytes()).hexdigest()}  {name}\n" for name in files), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(attempt_dir=self.attempt, stage_plan=self.plan, expected_stage_plan_sha256=self.plan_sha, expected_final_committed_step=10, tmux="/usr/bin/false", session_name="unused", max_log_bytes=10, log_backups=1)

    def test_accepts_consistent_terminal_attempt(self) -> None:
        report = verifier.verify(self.args())
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["next_stage_started"])

    def test_rejects_tampered_output(self) -> None:
        probe = json.loads((self.attempt / "probe-result.json").read_text(encoding="utf-8"))
        probe["tampered"] = True
        _write_json(self.attempt / "probe-result.json", probe)
        with self.assertRaisesRegex(RuntimeError, "output SHA-256 mismatch"):
            verifier.verify(self.args())


if __name__ == "__main__":
    unittest.main()
