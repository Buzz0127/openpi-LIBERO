"""CPU-only R-FT integration: future driver/verifier under the outer guard."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import autonomous_stage_orchestrator as orchestrator
import experiment_identity


TOOLS = Path(__file__).resolve().parent
ORCHESTRATOR = TOOLS / "autonomous_stage_orchestrator.py"
DRIVER = TOOLS / "run_formal_stage_driver.py"
VERIFIER = TOOLS / "verify_formal_segment_result.py"


class FuturePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rft-fake-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"; self.source.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "RFT", "GIT_AUTHOR_EMAIL": "rft@example.invalid",
               "GIT_COMMITTER_NAME": "RFT", "GIT_COMMITTER_EMAIL": "rft@example.invalid"}
        subprocess.run(["git", "init", "-q", "-b", "rft-test", str(self.source)], check=True)
        (self.source / "tracked").write_text("fixed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.source), "add", "tracked"], check=True, env=env)
        subprocess.run(["git", "-C", str(self.source), "commit", "-q", "-m", "fixed"], check=True, env=env)
        self.head = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        self.prior_result = self.root / "ft2-result.json"
        self.prior_acceptance = self.root / "ft2-acceptance.json"
        self.template = self.root / "future-result.json"
        self.runner = self.root / "fake-runner.py"
        identities = {"model": "m"}
        prior = {"status": "pass", "stage": "formal-pure-lora-segment", "segment_start": 1000, "segment_end": 5000,
                 "identities": identities, "ft0_contract": {"ft0_package_identity_sha256": "f"}, "checkpoint_steps": [1000, 5000]}
        self.prior_result.write_text(json.dumps(prior), encoding="utf-8")
        unsigned = {"schema_version": 1, "stage": "FT2-terminal-acceptance", "status": "pass", "candidate": True,
                    "result_sha256": self._sha(self.prior_result), "segment": [1000, 5000], "next_stage_started": False}
        acceptance = dict(unsigned)
        payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        acceptance["report_identity_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        self.prior_acceptance.write_text(json.dumps(acceptance), encoding="utf-8")
        receipt = {key: True for key in ("checkpoint_restore_succeeded", "parameter_tree_shape_dtype_equal",
                  "optimizer_tree_shape_dtype_equal", "all_parameter_values_equal_after_restore",
                  "all_optimizer_values_equal_after_restore", "adapter_values_equal_after_restore")}
        future = {**prior, "segment_start": 5000, "segment_end": 10000, "metrics_count": 5000,
                  "all_metrics_finite": True, "changed_golden_leaf_count": 20, "changed_non_golden_leaf_count": 0,
                  "checkpoint_steps": [1000, 5000, 10000], "checkpoint_restore_receipt": receipt,
                  "next_stage_started": False}
        self.template.write_text(json.dumps(future), encoding="utf-8")
        self.runner.write_text(
            "import argparse,json,os,pathlib,sys\n"
            "p=argparse.ArgumentParser(); p.add_argument('--template',type=pathlib.Path); p.add_argument('--result',type=pathlib.Path); p.add_argument('--progress',type=pathlib.Path); p.add_argument('--mode'); a=p.parse_args()\n"
            "if a.mode=='result-failure': sys.exit(17)\n"
            "a.result.write_text(a.template.read_text())\n"
            "if a.mode=='progress-failure': sys.exit(0)\n"
            "if a.mode=='verifier-failure':\n value=json.loads(a.result.read_text()); value['changed_non_golden_leaf_count']=1; a.result.write_text(json.dumps(value))\n"
            "tmp=a.progress.with_name('.progress.tmp'); tmp.write_text(json.dumps({'current_step':10000,'last_committed_step':10000})); os.replace(tmp,a.progress)\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _command(self, name: str, mode: str, *, wrong_prior: bool = False) -> tuple[list[str], Path, Path]:
        attempt = self.root / name
        progress, result, acceptance = attempt / "progress.json", attempt / "result.json", attempt / "terminal.json"
        prior = self.prior_acceptance
        if wrong_prior:
            prior = self.root / "wrong-prior.json"
            value = json.loads(self.prior_acceptance.read_text()); value["result_sha256"] = "0" * 64
            prior.write_text(json.dumps(value), encoding="utf-8")
        decision = self.root / f"{name}.c-ft2-decision.json"
        decision_value = {"schema_version": 1, "stage": "C-FT2-controlled-exception", "decision": "accept-existing-ft1-ft2-chain",
                          "execution_authorized": False, "next_stage_auto_start": False,
                          "allowed_next_input": {"segment_start": 5000, "prior_acceptance_sha256": self._sha(prior),
                                                 "prior_result_sha256": self._sha(self.prior_result)}}
        decision_value["decision_identity_sha256"] = experiment_identity.canonical_sha256(decision_value)
        decision.write_text(json.dumps(decision_value), encoding="utf-8")
        preflight = self.root / f"{name}.preflight.json"
        preflight_value = {"sample_count": 30, "samples": [{"gpus": [{"index": 0, "free_memory_percent": 99.0, "utilization_percent": 0.0}]}] * 30,
                           "selected_physical_gpu": 0, "collection_finished_epoch_seconds": __import__("time").time()}
        preflight_value["preflight_identity_sha256"] = experiment_identity.canonical_sha256(preflight_value)
        preflight.write_text(json.dumps(preflight_value), encoding="utf-8")
        identities = {key: f"{index:064x}" for index, key in enumerate(sorted(orchestrator.REQUIRED_IDENTITIES), 1)}
        child = [sys.executable, str(DRIVER), "--runner", str(self.runner), "--verifier", str(VERIFIER),
                 "--prior-acceptance", str(prior), "--prior-result", str(self.prior_result),
                 "--prior-identity-schema", "ft2-newline-v1", "--result", str(result),
                 "--acceptance-output", str(acceptance), "--progress", str(progress),
                 "--preflight-report", str(preflight), "--expected-preflight-sha256", self._sha(preflight), "--selected-physical-gpu", "0",
                 "--c-ft2-decision", str(decision),
                 "--segment-start", "5000", "--segment-end", "10000", "--",
                 "--template", str(self.template), "--result", str(result), "--progress", str(progress), "--mode", mode]
        bounds = {"timeout_seconds": 5.0, "heartbeat_timeout_seconds": 1.0, "sample_interval_seconds": 0.02,
                  "term_grace_seconds": 0.1, "kill_grace_seconds": 0.2, "max_retries": 0, "retry_return_codes": [],
                  "retry_delay_seconds": 0.01, "max_log_bytes": 4096, "log_backups": 1}
        plan = {"schema_version": 1, "stage": "R-FT-fake", "segment_start": 5000, "segment_end": 10000,
                "train_seed": 42, "eval_seed": 7, "expected_final_committed_step": 10000,
                "source": orchestrator._source_snapshot(self.source), "identities": identities,
                "attempt_dir": str(attempt.resolve()), "current_json": str((self.root / "current.json").resolve()),
                "progress_json": str(progress.resolve()), "required_outputs": [str(acceptance.resolve())], "bounds": bounds,
                "command": child, "tools": {"planner_sha256": "a" * 64,
                "orchestrator_sha256": experiment_identity.sha256_file(ORCHESTRATOR), "launcher_sha256": "b" * 64},
                "execution_authorized": False, "next_stage_auto_start": False}
        plan["plan_identity_sha256"] = experiment_identity.canonical_sha256(plan)
        plan_path = self.root / f"{name}.plan.json"; plan_path.write_text(json.dumps(plan), encoding="utf-8")
        command = [sys.executable, str(ORCHESTRATOR), "--attempt-dir", str(attempt), "--current-json", str(self.root / "current.json"),
                   "--stage-plan", str(plan_path), "--expected-stage-plan-sha256", self._sha(plan_path), "--source-root", str(self.source),
                   "--expected-branch", "rft-test", "--expected-head", self.head, "--expected-upstream", "NONE", "--stage", "R-FT-fake",
                   "--segment-start", "5000", "--segment-end", "10000", "--train-seed", "42", "--eval-seed", "7",
                   "--progress-json", str(progress), "--expected-final-committed-step", "10000", "--required-output", str(acceptance),
                   "--timeout-seconds", "5", "--heartbeat-timeout-seconds", "1", "--sample-interval-seconds", "0.02",
                   "--term-grace-seconds", "0.1", "--kill-grace-seconds", "0.2", "--retry-delay-seconds", "0.01",
                   "--max-log-bytes", "4096", "--log-backups", "1"]
        for key, value in identities.items():
            command.extend(["--identity", f"{key}={value}"])
        return [*command, "--", *child], attempt, acceptance

    def test_success_commits_progress_then_terminal_then_outer_pass(self) -> None:
        command, attempt, acceptance = self._command("success", "success")
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(TOOLS.parent), "CUDA_VISIBLE_DEVICES":"0", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(acceptance.is_file())
        self.assertEqual(json.loads((attempt / "progress.json").read_text())["last_committed_step"], 10000)
        self.assertEqual(json.loads((attempt / "summary.json").read_text())["status"], "pass")

    def test_failures_never_publish_future_terminal_acceptance(self) -> None:
        for name, mode, wrong_prior in (("result", "result-failure", False), ("progress", "progress-failure", False),
                                        ("verifier", "verifier-failure", False), ("prior", "success", True)):
            with self.subTest(name=name):
                command, attempt, acceptance = self._command(name, mode, wrong_prior=wrong_prior)
                completed = subprocess.run(command, capture_output=True, text=True, check=False,
                                           env={**os.environ, "PYTHONPATH": str(TOOLS.parent), "CUDA_VISIBLE_DEVICES":"0", "XLA_PYTHON_CLIENT_PREALLOCATE":"false"})
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(acceptance.exists())
                self.assertEqual(json.loads((attempt / "summary.json").read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
