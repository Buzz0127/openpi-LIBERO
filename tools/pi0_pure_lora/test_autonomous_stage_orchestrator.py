#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import autonomous_stage_orchestrator as orchestrator


SCRIPT = Path(orchestrator.__file__).resolve()


def _write_worker(path: Path) -> None:
    path.write_text(
        "import argparse,json,os,pathlib,signal,sys,time\n"
        "p=argparse.ArgumentParser(); p.add_argument('--progress',type=pathlib.Path); p.add_argument('--counter',type=pathlib.Path); p.add_argument('--steps',type=int); p.add_argument('--fail-first',action='store_true'); p.add_argument('--ignore-term',action='store_true'); a=p.parse_args()\n"
        "if a.ignore_term: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "if a.fail_first and not a.counter.exists(): a.counter.write_text('1'); sys.exit(75)\n"
        "for step in range(1,a.steps+1):\n"
        " t=a.progress.with_name('.progress.tmp'); t.write_text(json.dumps({'current_step':step,'last_committed_step':step if step==a.steps else 0,'recent_metrics':{'loss':1.0/step},'pause_count':0,'resume_count':0,'resource_peaks':{'rss_bytes':1}})); os.replace(t,a.progress); time.sleep(0.08)\n",
        encoding="utf-8",
    )


class AutonomousOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="a2-orchestrator-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "test-branch", str(self.source)], check=True)
        (self.source / "tracked").write_text("fixed\n", encoding="utf-8")
        environment = {**os.environ, "GIT_AUTHOR_NAME": "A2", "GIT_AUTHOR_EMAIL": "a2@example.invalid", "GIT_COMMITTER_NAME": "A2", "GIT_COMMITTER_EMAIL": "a2@example.invalid"}
        subprocess.run(["git", "-C", str(self.source), "add", "tracked"], check=True, env=environment)
        subprocess.run(["git", "-C", str(self.source), "commit", "-q", "-m", "fixed"], check=True, env=environment)
        self.head = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        self.plan = self.root / "plan.json"
        self.worker = self.root / "worker.py"
        _write_worker(self.worker)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def command(
        self, attempt: Path, progress: Path, *worker_args: str,
        extra: list[str] | None = None, child_override: list[str] | None = None,
        bounds_override: dict[str, float] | None = None,
    ) -> list[str]:
        identities = [f"{key}={index:064x}" for index, key in enumerate(sorted(orchestrator.REQUIRED_IDENTITIES), 1)]
        identity_map = dict(value.split("=", 1) for value in identities)
        child = child_override or [sys.executable, str(self.worker), "--progress", str(progress), "--counter", str(self.root / "counter"), "--steps", "3", *worker_args]
        max_retries = 1 if extra and "--max-retries" in extra else 0
        retry_codes = [75] if extra and "--retry-return-code" in extra else []
        bounds = {
            "timeout_seconds": 3.0, "heartbeat_timeout_seconds": 0.45,
            "sample_interval_seconds": 0.04, "term_grace_seconds": 0.12,
            "kill_grace_seconds": 0.5, "max_retries": max_retries,
            "retry_return_codes": retry_codes, "retry_delay_seconds": 0.05,
            "max_log_bytes": 2048, "log_backups": 1,
        }
        bounds.update(bounds_override or {})
        plan = {
            "schema_version": 1, "stage": "A2-test", "segment_start": 0, "segment_end": 3,
            "train_seed": 42, "eval_seed": 7, "expected_final_committed_step": 3,
            "source": orchestrator._source_snapshot(self.source), "identities": identity_map,
            "attempt_dir": str(attempt.resolve()), "current_json": str((self.root / "current.json").resolve()),
            "progress_json": str(progress.resolve()), "required_outputs": [str(progress.resolve())],
            "bounds": bounds, "command": child,
            "tools": {"planner_sha256": "a" * 64, "orchestrator_sha256": orchestrator._sha256(SCRIPT), "launcher_sha256": "b" * 64},
            "execution_authorized": False, "next_stage_auto_start": False,
        }
        plan["plan_identity_sha256"] = orchestrator.experiment_identity.canonical_sha256(plan)
        self.plan.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
        plan_sha = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        command = [
            sys.executable, str(SCRIPT), "--attempt-dir", str(attempt),
            "--current-json", str(self.root / "current.json"), "--stage-plan", str(self.plan),
            "--expected-stage-plan-sha256", plan_sha, "--source-root", str(self.source),
            "--expected-branch", "test-branch", "--expected-head", self.head,
            "--expected-upstream", "NONE", "--stage", "A2-test", "--segment-start", "0",
            "--segment-end", "3", "--train-seed", "42", "--eval-seed", "7",
            "--progress-json", str(progress), "--expected-final-committed-step", "3",
            "--required-output", str(progress), "--timeout-seconds", "3",
            "--heartbeat-timeout-seconds", "0.45", "--sample-interval-seconds", "0.04",
            "--term-grace-seconds", "0.12", "--kill-grace-seconds", "0.5",
            "--retry-delay-seconds", "0.05", "--max-log-bytes", "2048", "--log-backups", "1",
        ]
        for identity in identities:
            command.extend(["--identity", identity])
        for key, value in (bounds_override or {}).items():
            command[command.index("--" + key.replace("_", "-")) + 1] = str(value)
        if extra:
            command.extend(extra)
        return command + ["--", *child]

    def test_success_writes_atomic_status_summary_and_manifest(self) -> None:
        attempt = self.root / "success"
        progress = attempt / "progress.json"
        result = subprocess.run(self.command(attempt, progress), text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((attempt / "summary.json").read_text())
        status = json.loads((attempt / "status.json").read_text())
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(status["last_committed_step"], 3)
        self.assertFalse(summary["next_stage_started"])
        self.assertTrue((attempt / "output-files.sha256").is_file())

    def test_retry_is_bounded_and_then_succeeds(self) -> None:
        attempt = self.root / "retry"
        progress = attempt / "progress.json"
        result = subprocess.run(
            self.command(attempt, progress, "--fail-first", extra=["--max-retries", "1", "--retry-return-code", "75"]),
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((attempt / "summary.json").read_text())
        self.assertEqual(summary["retry_count"], 1)
        self.assertEqual(len(summary["child_runs"]), 2)

    def test_heartbeat_timeout_kills_only_owned_ignoring_group(self) -> None:
        attempt = self.root / "timeout"
        progress = attempt / "progress.json"
        child = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"]
        command = self.command(attempt, progress, child_override=child)
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, orchestrator.EXIT_HEARTBEAT_TIMEOUT, result.stderr)
        summary = json.loads((attempt / "summary.json").read_text())
        record = summary["child_runs"][0]
        self.assertEqual(summary["reason_code"], "heartbeat_timeout")
        self.assertTrue(record["ownership_verified"])
        self.assertTrue(record["term_sent"])
        self.assertTrue(record["kill_sent"])
        self.assertTrue(record["wait_reaped"])

    def test_event_write_exception_after_spawn_cleans_owned_child(self) -> None:
        attempt = self.root / "write-failure"
        pid_file = self.root / "child-pid"
        command = self.command(attempt, attempt / "progress.json", child_override=[
            sys.executable, "-c", "import time; time.sleep(30)",
        ])
        bootstrap = (
            f"import pathlib,sys; sys.path.insert(0,{str(SCRIPT.parent)!r})\n"
            "import autonomous_stage_orchestrator as guard\n"
            "original_popen=guard.subprocess.Popen\n"
            "original_event=guard._append_event\n"
            "def launch(*args,**kwargs):\n"
            " child=original_popen(*args,**kwargs)\n"
            " if kwargs.get('start_new_session'):\n"
            f"  pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            " return child\n"
            "def event(path,value):\n"
            " if value.get('event')=='child_started': raise OSError('fake event disk failure')\n"
            " original_event(path,value)\n"
            "guard.subprocess.Popen=launch\n"
            "guard._append_event=event\n"
            "sys.exit(guard.main())\n"
        )
        result = subprocess.run([sys.executable, "-c", bootstrap, *command[2:]],
                                text=True, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        summary = json.loads((attempt / "summary.json").read_text())
        self.assertEqual(summary["reason_code"], "orchestrator_exception")
        self.assertTrue(summary["child_runs"][0]["group_exit_confirmed"])
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

    def test_t1_launch_bindings_rechecked_before_any_attempt_mkdir(self) -> None:
        stage_parent = self.root / "not-created-stage"
        attempt = stage_parent / "attempt"
        command = self.command(attempt, attempt / "progress.json")
        plan = json.loads(self.plan.read_text())
        plan["stage"] = "T1-engineering-100-200"
        plan["current_json"] = str((stage_parent / "current.json").resolve())
        plan.pop("plan_identity_sha256")
        plan["plan_identity_sha256"] = orchestrator.experiment_identity.canonical_sha256(plan)
        self.plan.write_text(json.dumps(plan) + "\n")
        command[command.index("--stage") + 1] = plan["stage"]
        command[command.index("--current-json") + 1] = plan["current_json"]
        command[command.index("--expected-stage-plan-sha256") + 1] = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        bootstrap = (
            f"import sys,types; sys.path.insert(0,{str(SCRIPT.parent)!r})\n"
            "gate=types.ModuleType('t1_execution_contract')\n"
            "def validate(plan): raise RuntimeError('fake stale launch binding rejected')\n"
            "gate.validate_launch_bindings=validate\n"
            "sys.modules['t1_execution_contract']=gate\n"
            "import autonomous_stage_orchestrator as guard\n"
            "sys.exit(guard.main())\n"
        )
        result = subprocess.run([sys.executable, "-c", bootstrap, *command[2:]],
                                text=True, capture_output=True, timeout=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fake stale launch binding rejected", result.stderr)
        self.assertFalse(stage_parent.exists())

    def test_nested_storage_gpu_guards_forward_term_and_leave_no_owned_process(self) -> None:
        attempt = self.root / "nested"
        leaf_pid = self.root / "leaf-pid"
        metered = self.root / "metered"
        metered.mkdir()
        tools_dir = SCRIPT.parent.parent
        trainer = (
            "import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"pathlib.Path({str(leaf_pid)!r}).write_text(str(os.getpid())); time.sleep(30)"
        )
        gpu_bootstrap = (
            f"import sys; sys.path.insert(0,{str(tools_dir)!r}); import gpu_utilization_guard as g; "
            "g.query_gpu_status=lambda gpu: g.GpuStatus(1,10,100); sys.exit(g.main())"
        )
        gpu_command = [sys.executable, "-c", gpu_bootstrap, "--physical-gpu", "0",
                       "--interval-seconds", "0.01", "--max-prelaunch-wait-seconds", "0.5",
                       "--max-runtime-seconds", "10", "--terminate-grace-seconds", "0.1",
                       "--kill-grace-seconds", "0.3", "--log", str(attempt / "gpu.jsonl"),
                       "--", sys.executable, "-c", trainer]
        storage_bootstrap = (
            f"import sys; sys.path.insert(0,{str(tools_dir)!r}); import storage_budget_guard as g; "
            "g._read_mem_available_bytes=lambda:64*(1<<30); "
            "g._read_child_rss_bytes=lambda pid:1<<20; sys.exit(g.main())"
        )
        storage_command = [sys.executable, "-c", storage_bootstrap,
                           "--attempt-dir", str(attempt / "storage"), "--monitor-root", str(metered),
                           "--existing-billed-bytes", "0", "--soft-limit-bytes", "10000000",
                           "--hard-limit-bytes", "20000000", "--timeout-seconds", "12",
                           "--sample-interval-seconds", "0.02", "--near-sample-interval-seconds", "0.02",
                           "--term-grace-seconds", "0.6", "--kill-grace-seconds", "0.3", "--", *gpu_command]
        command = self.command(attempt, attempt / "progress.json", child_override=storage_command,
                               bounds_override={"timeout_seconds": 15.0, "heartbeat_timeout_seconds": 10.0,
                                                "term_grace_seconds": 1.5})
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not leaf_pid.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.01)
            self.assertTrue(leaf_pid.exists())
            os.kill(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0, stdout + stderr)
            summary = json.loads((attempt / "summary.json").read_text())
            self.assertEqual(summary["reason_code"], "external_signal")
            self.assertTrue(summary["child_runs"][0]["group_exit_confirmed"])
            storage = json.loads((attempt / "storage" / "exit_status.json").read_text())
            self.assertEqual(storage["reason_code"], "external_signal")
            self.assertTrue(storage["group_exit_confirmed"])
            gpu = [json.loads(line) for line in (attempt / "gpu.jsonl").read_text().splitlines()]
            cleanup = next(record for record in gpu if record["event"] == "child_cleanup")
            self.assertTrue(cleanup["kill_sent"])
            self.assertTrue(cleanup["group_exit_confirmed"])
            for pid in (summary["child_runs"][0]["child_pid"], storage["child_pid"], int(leaf_pid.read_text())):
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
