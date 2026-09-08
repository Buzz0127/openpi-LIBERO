"""Fake artifacts only: exercise the independent terminal acceptance boundary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import verify_t1_resume_result as verifier


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def signed(value: dict, key: str, *, newline: bool = False) -> dict:
    value = {name: item for name, item in value.items() if name != key}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + ("\n" if newline else "")
    return {**value, key: digest(raw)}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class FakeAttempt:
    """A few kilobytes of synthetic tensors, checkpoint chunks, and A2 receipts."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.attempt = self.root / "attempt-fake-t1"
        self.attempt.mkdir()
        self.checkpoint = self.root / "checkpoints"
        self.adapter = self.root / "adapters"
        self.golden_path = self.root / "golden.json"
        self.golden = {"review_invariants": {"total_param_leaf_count": 70}, "entries": [
            {"path": f"exact/adapter/{i:02d}", "shape": [1], "parameter_count": 1} for i in range(20)]}
        write(self.golden_path, self.golden)
        self.before = {entry["path"]: digest("before-" + entry["path"]) for entry in self.golden["entries"]}
        self.before.update({f"base/frozen/{i:02d}": digest(f"base-{i}") for i in range(50)})
        self.after = {key: digest("after-" + key) if key.startswith("exact/") else value for key, value in self.before.items()}
        self.norm = b'{"synthetic_norm":true}\n'
        self.ids = {key: digest(key) for key in verifier.IDENTITY_KEYS}
        self.ids["golden"] = verifier._sha256(self.golden_path)
        self.ids["norm"] = hashlib.sha256(self.norm).hexdigest()
        self.model_ids = {"base_manifest_sha256": digest("base"), "golden_manifest_sha256": self.ids["golden"], "norm_stats_sha256": self.ids["norm"], "config_patch_sha256": self.ids["config"]}
        model = signed({"identities": self.model_ids, "openpi_commit": "a" * 40, "model_mode": "base"}, "model_identity_sha256")
        self.ids["model"] = model["model_identity_sha256"]
        self.model_path = self.root / "model.json"
        write(self.model_path, model)
        self.make_checkpoint(100)
        old_adapter_id = self.make_adapter(100, self.before)
        self.acceptance = signed({"status": "pass", "checkpoint_artifact": verifier.artifact_manifest(self.checkpoint), "adapter_artifact": verifier.artifact_manifest(self.adapter), "adapter_identity_sha256": old_adapter_id}, "report_identity_sha256")
        self.acceptance_path = self.root / "s1d.json"
        write(self.acceptance_path, self.acceptance)
        self.freeze = signed({"identities": self.ids, "source": {"head": "a" * 40}, "resume_input": {
            "acceptance_report_identity_sha256": self.acceptance["report_identity_sha256"],
            "checkpoint_tree_sha256": self.acceptance["checkpoint_artifact"]["artifact_tree_sha256"],
            "adapter_identity_sha256": old_adapter_id, "checkpoint_root": str(self.checkpoint),
            "restore_receipt_sha256": verifier._sha256(self.adapter / "step-00000100.verified.json")}}, "package_identity_sha256", newline=True)
        self.freeze_path = self.root / "freeze.json"
        write(self.freeze_path, self.freeze)
        self.make_checkpoint(200)
        self.adapter_id = self.make_adapter(200, self.after)
        self.paths = {name: str(self.attempt / filename) for name, filename in {
            "result": "runner_result.json", "loader_receipt": "loader_receipt.json", "rng_receipt": "rng_receipt.json",
            "composition_receipt": "composition_receipt.json", "progress": "progress.json", "gpu_events": "gpu_guard.jsonl",
            "storage_guard_dir": "storage_guard"}.items()}
        self.loader = {"restored_step": 100, "skipped_batch_count": 100, "first_resumed_batch_index": 100, "position_verified": True,
                       "reference_batch_sha256": digest("batch100"), "resumed_batch_sha256": digest("batch100")}
        self.rng = {"seed": 42, "restored_step": 100, "replayed_split_count": 100,
                    "reference_train_key_sha256": digest("key100"), "resumed_train_key_sha256": digest("key100"),
                    "reference_first_step_key_sha256": digest("key101"), "first_step_key_sha256": digest("key101")}
        self.composition = {"status": "pass", "save_step": 200, "adapter_identity_sha256": self.adapter_id, "parameter_hashes": dict(self.after)}
        self.result = {"status": "pass", "stage": "T1-resume-engineering", "identities": self.ids, "source_head": "a" * 40,
            "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7,
            "batch_size": 1, "num_workers": 0, "shuffle": True, "physical_gpu": 1, "jax_device_count": 1,
            "next_stage_started": False, "old_checkpoint_deleted": False, "metrics_trace": [{"loss": .1, "grad_norm": 1.0} for _ in range(100)],
            "parameter_hashes_before": dict(self.before), "parameter_hashes_after": dict(self.after),
            "checkpoint_restore_receipt": verifier._load(self.adapter / "step-00000200.verified.json")}
        self.progress = {"current_step": 200, "last_committed_step": 200}
        self.gpu_events = [{"event": "guard_started", "physical_gpu": 1}, {"event": "child_started", "child_pid": 900003},
                           {"event": "gpu_sample", "free_memory_percent": 50, "utilization_percent": 40},
                           {"event": "child_exited", "child_pid": 900003, "return_code": 0, "wait_reaped": True, "group_exit_confirmed": True}]
        self.storage_run = {"child_pid": 900002, "expected_child_pgid": 900002, "start_new_session": True,
                            "storage": {"soft_limit_bytes": 240_000_000_000, "hard_limit_bytes": 250_000_000_000}}
        self.storage_exit = {"reason_code": "completed", "child_returncode": 0, "wait_reaped": True, "group_exit_confirmed": True,
            "term_sent": False, "kill_sent": False, "external_signal": None, "final_billed_lora_bytes": 90_000_000_000,
            "final_sample": {"positive_stage_delta_bytes": 5000, "roots": [{"path": key, "billed_delta_bytes": 0} for key in verifier.SHARED_CACHE_ROOTS]}}
        preflight = {"sample_count": 30, "selected_physical_gpu": 1, "selected_gpu_uuid": "GPU-fake1",
            "collection_started_epoch_seconds": 1000, "collection_finished_epoch_seconds": 1029,
            "samples": [{"sample_index": i, "monotonic_seconds": 50 + i,
                "gpus": [{"index": j, "uuid": f"GPU-fake{j}", "free_memory_percent": 50, "utilization_percent": 10} for j in range(2)],
                "host": {"logical_cpu_count": 8, "load1": 1, "load1_per_cpu": .125, "mem_available_bytes": 80_000_000_000}} for i in range(30)]}
        self.preflight_path = self.root / "preflight.json"
        write(self.preflight_path, signed(preflight, "preflight_identity_sha256"))
        self.tool_path = self.root / "fake_tool.py"
        self.tool_path.write_text("# synthetic pinned tool, never executed\n")
        self.required = [str(Path(value)) for key, value in self.paths.items() if key != "storage_guard_dir"]
        self.required += [str(Path(self.paths["storage_guard_dir"]) / name) for name in ("run_manifest.json", "exit_status.json", "samples.jsonl")]
        self.plan = {"stage": "T1-engineering-100-200", "identities": self.ids, "source": {"head": "a" * 40},
            "attempt_dir": str(self.attempt), "execution_authorized": False, "next_stage_auto_start": False,
            "segment_start": 100, "segment_end": 200, "train_seed": 42, "eval_seed": 7, "expected_final_committed_step": 200,
            "freeze_package_identity_sha256": self.freeze["package_identity_sha256"], "paths": self.paths,
            "required_outputs": self.required, "bounds": {"timeout_seconds": 6840, "max_retries": 0, "max_log_bytes": 1024, "log_backups": 2},
            "command": [str(self.tool_path)], "selected_physical_gpu": 1, "selected_gpu_uuid": "GPU-fake1",
            "environment": {"CUDA_VISIBLE_DEVICES": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
            "preflight": {"path": str(self.preflight_path), "sha256": verifier._sha256(self.preflight_path)},
            "input_files": {key: {"path": str(path), "sha256": verifier._sha256(path)} for key, path in {
                "model_manifest": self.model_path, "golden_manifest": self.golden_path, "freeze_package": self.freeze_path, "s1d_acceptance_report": self.acceptance_path}.items()},
            "tool_bindings": {"runner": {"path": str(self.tool_path), "sha256": verifier._sha256(self.tool_path)}}}
        self.plan_path = self.root / "stage_plan.json"
        self.child = {"child_pid": 900001, "child_pgid": 900001, "returncode": 0, "wait_reaped": True, "group_exit_confirmed": True, "term_sent": False, "kill_sent": False}
        self.launcher = {"attempt_dir": str(self.attempt), "heartbeat_sequences": [2, 3], "tmux_session_alive": True, "launcher_detaches_after_verification": True}
        self.args = argparse.Namespace(attempt_dir=self.attempt, stage_plan=self.plan_path,
            freeze_package=self.freeze_path, expected_freeze_identity=self.freeze["package_identity_sha256"],
            s1d_acceptance_report=self.acceptance_path, golden_manifest=self.golden_path,
            checkpoint_dir=self.checkpoint, adapter_root=self.adapter, tmux="/fake/tmux", session_name="fake-t1",
            max_stage_increment_bytes=7_000_000_000, output=self.root / "acceptance.json")
        self.refresh()

    def make_checkpoint(self, step: int):
        base = self.checkpoint / str(step)
        write(base / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 123000 + step})
        for group in ("params", "train_state"):
            write(base / group / "_METADATA", {"synthetic": group})
            write(base / group / "manifest.ocdbt", {"fake_ocdbt": step})
            write(base / group / "d/chunk", {"synthetic_tensor_chunk": step})
        path = base / "assets/canonical/norm_stats.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.norm)

    def make_adapter(self, step: int, parameter_hashes: dict) -> str:
        base = self.adapter / f"step-{step:08d}"
        entries = []
        for i, entry in enumerate(self.golden["entries"]):
            file = f"arrays/{i:04d}.npy"
            write(base / file, {"fake_array": step, "path": entry["path"]})
            entries.append({**entry, "dtype": "float32", "file": file, "file_sha256": verifier._sha256(base / file), "array_sha256": parameter_hashes[entry["path"]]})
        manifest = signed({"schema_version": 1, "artifact_type": "pi0_pure_lora_adapter_only", "identities": self.model_ids,
                           "train_step": step, "train_seed": 42, "entries": entries}, "adapter_identity_sha256")
        write(base / "manifest.json", manifest)
        receipt = {key: True for key in ("checkpoint_restore_succeeded", "parameter_tree_shape_dtype_equal", "optimizer_tree_shape_dtype_equal", "all_parameter_values_equal_after_restore", "all_optimizer_values_equal_after_restore", "adapter_values_equal_after_restore")}
        receipt.update({"parameter_leaf_count": 70, "optimizer_leaf_count": 42, "save_step": step,
                        "automatic_pruning_enabled": False, "old_checkpoint_deletion_performed": False, "adapter_identity_sha256": manifest["adapter_identity_sha256"]})
        write(self.adapter / f"step-{step:08d}.verified.json", receipt)
        return manifest["adapter_identity_sha256"]

    def refresh(self, *, refresh_artifacts=True):
        """Re-sign outer receipts, so negative tests reach the semantic check."""
        for name, value in (("loader_receipt", self.loader), ("rng_receipt", self.rng), ("composition_receipt", self.composition), ("progress", self.progress)):
            write(Path(self.paths[name]), value)
        for name in ("loader_receipt", "rng_receipt", "composition_receipt"):
            self.result[name + "_sha256"] = verifier._sha256(Path(self.paths[name]))
        if refresh_artifacts:
            self.result["checkpoint_artifact_after"] = verifier.artifact_manifest(self.checkpoint)
            self.result["adapter_artifact_after"] = verifier.artifact_manifest(self.adapter)
        write(Path(self.paths["result"]), self.result)
        Path(self.paths["gpu_events"]).write_text("\n".join(json.dumps(row) for row in self.gpu_events) + "\n")
        guard = Path(self.paths["storage_guard_dir"])
        write(guard / "run_manifest.json", self.storage_run)
        write(guard / "exit_status.json", self.storage_exit)
        write(guard / "samples.jsonl", {"fake_storage_sample": True})
        self.plan = signed(self.plan, "plan_identity_sha256")
        write(self.plan_path, self.plan)
        self.args.expected_stage_plan_sha256 = verifier._sha256(self.plan_path)
        common = {"attempt_id": self.attempt.name, "stage_plan_sha256": self.args.expected_stage_plan_sha256,
                  "stage_plan_identity_sha256": self.plan["plan_identity_sha256"], "source": self.plan["source"], "identities": self.ids}
        write(self.attempt / "status.json", {**common, **self.progress, "status": "pass", "exit_reason": "completed"})
        write(self.attempt / "run_manifest.json", {**common, "command": self.plan["command"], "bounds": self.plan["bounds"], "next_stage_auto_start": False,
            "start_time_epoch_seconds": 1030, "child_environment_overrides": {key: "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")},
            "child_environment_unset": ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]})
        write(self.attempt / "summary.json", signed({"attempt_id": self.attempt.name, "status": "pass", "reason_code": "completed",
            "retry_count": 0, "next_stage_started": False, "last_progress": self.progress, "child_runs": [self.child],
            "output_sha256": {path: verifier._sha256(Path(path)) for path in self.required}}, "summary_identity_sha256"))
        write(self.attempt / "launcher_receipt.json", self.launcher)
        (self.attempt / "exit_code.txt").write_text("0\n")
        (self.attempt / "child-00.stdout.log").write_text("fake terminal log\n")
        self.rehash_output_manifest()

    def rehash_output_manifest(self):
        manifest = self.attempt / "output-files.sha256"
        manifest.write_text("".join(f"{verifier._sha256(path)}  {path.relative_to(self.attempt)}\n" for path in sorted(self.attempt.rglob("*")) if path.is_file() and path != manifest))


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fake-t1-verifier-")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = FakeAttempt(Path(self.temporary.name))
        self.audit = mock.patch.object(verifier, "_terminal_process_audit", return_value={"mocked_fake_process_audit": True})
        self.audit.start()
        self.addCleanup(self.audit.stop)

    def verify(self):
        return verifier.verify(self.fixture.args)

    def test_accepts_fake_complete_segment_and_independent_identities(self):
        report = self.verify()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["changed_golden_paths"]), 20)
        self.assertEqual(report["unchanged_non_golden_leaf_count"], 50)
        self.assertEqual(report["checkpoint_artifact"]["file_count"], 16)
        self.assertFalse(report["training_completed"])
        verifier._signed(report, "report_identity_sha256")

    def test_rejects_plan_identity_tampering(self):
        plan = verifier._load(self.fixture.plan_path)
        plan["train_seed"] = 13
        write(self.fixture.plan_path, plan)
        self.fixture.args.expected_stage_plan_sha256 = verifier._sha256(self.fixture.plan_path)
        with self.assertRaisesRegex(RuntimeError, "plan_identity_sha256"):
            self.verify()

    def test_rejects_frozen_identity_tampering(self):
        self.fixture.result["identities"] = {**self.fixture.ids, "model": digest("wrong-model")}
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "runner experiment/source"):
            self.verify()

    def test_rejects_modified_step100_even_when_runner_rehashes_it(self):
        (self.fixture.checkpoint / "100/params/d/chunk").write_text("changed")
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "step-100 file changed"):
            self.verify()

    def test_rejects_deleted_step100_adapter_file(self):
        (self.fixture.adapter / "step-00000100/arrays/0000.npy").unlink()
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "step-100 file changed"):
            self.verify()

    def test_rejects_incomplete_step200_even_when_runner_rehashes_it(self):
        (self.fixture.checkpoint / "200/train_state/manifest.ocdbt").unlink()
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "step-200 incomplete"):
            self.verify()

    def test_rejects_uncommitted_step200_metadata(self):
        write(self.fixture.checkpoint / "200/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": None})
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "atomic commit"):
            self.verify()

    def test_rejects_step200_changed_after_runner_receipt(self):
        (self.fixture.checkpoint / "200/params/d/chunk").write_text("corruption after save")
        with self.assertRaisesRegex(RuntimeError, "artifact tree changed"):
            self.verify()

    def test_rejects_partial_artifacts(self):
        (self.fixture.checkpoint / "200/.unfinished.partial").write_text("pending")
        with self.assertRaisesRegex(RuntimeError, "temporary artifact"):
            self.verify()

    def test_rejects_non_golden_change_despite_claimed_zero_count(self):
        self.fixture.result["parameter_hashes_after"]["base/frozen/00"] = digest("changed-base")
        self.fixture.result["changed_non_golden_leaf_count"] = 0
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "Golden/non-Golden"):
            self.verify()

    def test_rejects_missing_parameter_hash_despite_claimed_count(self):
        del self.fixture.result["parameter_hashes_before"]["base/frozen/00"]
        self.fixture.result["non_golden_leaf_count"] = 50
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "all 70 leaves"):
            self.verify()

    def test_rejects_unchanged_golden_leaf(self):
        key = "exact/adapter/00"
        self.fixture.result["parameter_hashes_after"][key] = self.fixture.before[key]
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "Golden/non-Golden"):
            self.verify()

    def test_rejects_loader_restart(self):
        self.fixture.loader["first_resumed_batch_index"] = 0
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "loader continuity"):
            self.verify()

    def test_rejects_rng_mismatch(self):
        self.fixture.rng["first_step_key_sha256"] = digest("wrong-rng")
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "RNG reference/replay"):
            self.verify()

    def test_rejects_nonfinite_metrics(self):
        self.fixture.result["metrics_trace"][50]["loss"] = float("nan")
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.verify()

    def test_rejects_full_restore_failure(self):
        self.fixture.result["checkpoint_restore_receipt"]["all_optimizer_values_equal_after_restore"] = False
        write(self.fixture.adapter / "step-00000200.verified.json", self.fixture.result["checkpoint_restore_receipt"])
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "restore receipt failed"):
            self.verify()

    def test_rejects_pruning_permission(self):
        self.fixture.result["checkpoint_restore_receipt"]["automatic_pruning_enabled"] = True
        write(self.fixture.adapter / "step-00000200.verified.json", self.fixture.result["checkpoint_restore_receipt"])
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "pruning/deletion"):
            self.verify()

    def test_rejects_composition_wrong_base_leaf(self):
        self.fixture.composition["parameter_hashes"]["base/frozen/00"] = digest("wrong-compose")
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "composition does not equal"):
            self.verify()

    def test_rejects_gpu_guard_error(self):
        self.fixture.gpu_events.insert(2, {"event": "monitor_error"})
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "GPU guard emergency"):
            self.verify()

    def test_rejects_storage_guard_failure(self):
        self.fixture.storage_exit["reason_code"] = "soft_limit"
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "storage guard failed"):
            self.verify()

    def test_rejects_owned_group_without_confirmed_exit(self):
        self.fixture.child["group_exit_confirmed"] = False
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "A2 owned group"):
            self.verify()

    def test_rejects_cache_growth(self):
        self.fixture.storage_exit["final_sample"]["roots"][0]["billed_delta_bytes"] = 1
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "shared caches"):
            self.verify()

    def test_rejects_ineligible_gpu_mapping(self):
        preflight = verifier._load(self.fixture.preflight_path)
        preflight["samples"][-1]["gpus"][1]["free_memory_percent"] = 15
        write(self.fixture.preflight_path, signed(preflight, "preflight_identity_sha256"))
        self.fixture.plan["preflight"]["sha256"] = verifier._sha256(self.fixture.preflight_path)
        self.fixture.refresh()
        with self.assertRaisesRegex(RuntimeError, "strict safety gate"):
            self.verify()

    def test_rejects_missing_a2_output_manifest_entry(self):
        path = self.fixture.attempt / "output-files.sha256"
        path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.endswith("runner_result.json")) + "\n")
        with self.assertRaisesRegex(RuntimeError, "omits required"):
            self.verify()

    def test_rejects_a2_output_content_tampering(self):
        (self.fixture.attempt / "child-00.stdout.log").write_text("changed after manifest")
        with self.assertRaisesRegex(RuntimeError, "A2 output hash mismatch"):
            self.verify()

    def test_rejects_symlink_artifact(self):
        (self.fixture.checkpoint / "200/params/leak").symlink_to(self.fixture.golden_path)
        with self.assertRaisesRegex(RuntimeError, "symlink in artifact"):
            self.verify()


class ProcessAuditTests(unittest.TestCase):
    def test_permission_error_is_not_absence(self):
        with mock.patch.object(verifier.os, "kill", side_effect=PermissionError):
            self.assertFalse(verifier._process_absent(900001))

    def test_process_and_group_checked_without_signalling(self):
        with mock.patch.object(verifier.os, "kill", side_effect=ProcessLookupError) as pid, mock.patch.object(verifier.os, "killpg", side_effect=ProcessLookupError) as group, mock.patch.object(verifier.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            report = verifier._terminal_process_audit([900001], "/fake/tmux", "fake")
        pid.assert_called_once_with(900001, 0)
        group.assert_called_once_with(900001, 0)
        self.assertTrue(report["tmux_absent"])

    def test_live_owned_group_fails(self):
        with mock.patch.object(verifier.os, "kill", side_effect=ProcessLookupError), mock.patch.object(verifier.os, "killpg", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "owned process/group"):
                verifier._terminal_process_audit([900001], "/fake/tmux", "fake")

    def test_tmux_query_error_fails_closed(self):
        with mock.patch.object(verifier, "_process_absent", return_value=True), mock.patch.object(verifier.subprocess, "run", return_value=subprocess.CompletedProcess([], 127)):
            with self.assertRaisesRegex(RuntimeError, "tmux query failed"):
                verifier._terminal_process_audit([900001], "/fake/tmux", "fake")


if __name__ == "__main__":
    unittest.main()
