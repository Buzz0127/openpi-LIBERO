"""Fake 30-sample reports test sealing and runtime freshness, without GPU access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_t1_execution_template as builder
import experiment_identity as identity
import finalize_t1_execution_plan as finalizer
import t1_execution_contract as contract
from test_build_t1_execution_template import fixture

HOST = {"hostname": "fake-host", "boot_id": "fake-boot"}
NOW_EPOCH = 10_000.0
NOW_MONOTONIC = 5_000.0


def preflight_fixture():
    samples = []
    for index in range(30):
        samples.append({"sample_index": index, "monotonic_seconds": 4961.0 + index, "gpus": [{"index": gpu, "uuid": f"GPU-fake-{gpu}", "utilization_percent": 20 if gpu == 0 else 10, "memory_used_mib": 4000, "memory_total_mib": 10000, "free_memory_percent": 60.0} for gpu in range(2)], "host": {"logical_cpu_count": 100, "mem_available_bytes": 100_000_000_000, "load1": 10.0, "load1_per_cpu": .10}})
    return contract.signed({"schema_version": 1, "sample_count": 30, "samples": samples, "selected_physical_gpu": 1, "selected_gpu_uuid": "GPU-fake-1", "launch_gate": dict(contract.PREFLIGHT_GATE), "guard_thresholds": dict(contract.GUARD_THRESHOLDS), "jax_preallocation_required": False, "host_identity": dict(HOST), "collection_started_epoch_seconds": 9961.0, "collection_finished_epoch_seconds": 9990.0}, "preflight_identity_sha256")


class FinalizerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.freeze, self.readiness, self.spec = fixture(self.root)
        self.template = builder.build_template(self.freeze, self.readiness, self.spec)
        self.preflight = preflight_fixture()

    def rehash(self):
        self.preflight.pop("preflight_identity_sha256", None)
        self.preflight = contract.signed(self.preflight, "preflight_identity_sha256")

    def seal(self):
        return finalizer.finalize_plan(self.template, self.preflight, preflight_sha256="a" * 64, now_epoch=NOW_EPOCH, now_monotonic=NOW_MONOTONIC, host_identity=HOST)

    def test_exact_nested_command_and_single_gpu_environment(self):
        plan = self.seal()
        self.assertEqual(plan["selected_physical_gpu"], 1)
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(plan["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(plan["environment"]["PYTHONPATH"], str(self.root / "snapshot" / "tools"))
        self.assertEqual(plan["environment"]["HF_HOME"], "/home/wengzr/projects/openpi-lora-cache/huggingface")
        for key, value in plan["environment"].items():
            self.assertIn(f"{key}={value}", plan["command"])
        self.assertEqual(plan["storage_guard_command"].count("--monitor-root"), 6)
        self.assertIn("--rng-receipt", plan["runner_command"])
        self.assertEqual(plan["gpu_guard_command"][-len(plan["runner_command"]):], plan["runner_command"])
        self.assertEqual(plan["storage_guard_command"][-len(plan["gpu_guard_command"]):], plan["gpu_guard_command"])
        self.assertEqual(plan["command"][-len(plan["storage_guard_command"]):], plan["storage_guard_command"])
        self.assertEqual(plan["bounds"]["max_retries"], 0)
        self.assertEqual(len(plan["required_outputs"]), 9)
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["next_stage_auto_start"])
        review = finalizer.launch_review(plan, "b" * 64)
        self.assertIn(plan["paths"]["stage_plan"], review["launch_argv"])
        self.assertFalse(review["execution_authorized"])

    def test_rehashed_failure_reasons_rejected(self):
        self.preflight["reasons"] = ["unsafe"]
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_runtime_command_environment_tamper_rejected(self):
        plan = self.runtime_plan()
        plan["command"].remove("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        plan.pop("plan_identity_sha256")
        plan = contract.signed(plan, "plan_identity_sha256")
        with self.assertRaises(ValueError): self.runtime_check(plan)

    def test_stale_report_is_rejected(self):
        for sample in self.preflight["samples"]: sample["monotonic_seconds"] -= 200
        self.preflight["collection_started_epoch_seconds"] -= 200
        self.preflight["collection_finished_epoch_seconds"] -= 200
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_future_report_is_rejected(self):
        self.preflight["collection_finished_epoch_seconds"] += 20
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_tampered_identity(self):
        self.preflight["selected_physical_gpu"] = 0
        with self.assertRaises(ValueError): self.seal()

    def test_rehashed_wrong_gpu_choice(self):
        self.preflight["selected_physical_gpu"] = 0
        self.preflight["selected_gpu_uuid"] = "GPU-fake-0"
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_selected_gpu_at_exact_15_percent_rejected(self):
        for gpu in self.preflight["samples"][-1]["gpus"]:
            gpu["memory_used_mib"] = 8500; gpu["free_memory_percent"] = 15.0
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_missing_one_physical_card(self):
        self.preflight["samples"][5]["gpus"].pop()
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_too_short_elapsed_sampling(self):
        for index, sample in enumerate(self.preflight["samples"]): sample["monotonic_seconds"] = 4989 + index / 100
        self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_sample_count_disagrees(self):
        self.preflight["sample_count"] = 31; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_duplicate_card_ids_rejected(self):
        self.preflight["samples"][3]["gpus"][0]["index"] = 1; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_nonfinite_gpu_number_rejected(self):
        self.preflight["samples"][3]["gpus"][0]["utilization_percent"] = float("nan"); self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_fabricated_free_ratio_rejected(self):
        self.preflight["samples"][-1]["gpus"][1]["free_memory_percent"] = 90; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_unsafe_cpu_ram_or_spoofed_load_ratio(self):
        self.preflight["samples"][0]["host"]["load1"] = 95; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_other_host_or_boot_rejected(self):
        self.preflight["host_identity"]["boot_id"] = "other-boot"; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_legacy_report_without_wall_clock_rejected(self):
        del self.preflight["collection_finished_epoch_seconds"]; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_rehashed_weakened_gate_rejected(self):
        self.preflight["launch_gate"]["free_memory_percent_strictly_greater_than"] = 5; self.rehash()
        with self.assertRaises(ValueError): self.seal()

    def test_rehashed_long_timeout_rejected(self):
        self.template["bounds"]["timeout_seconds"] = 8000
        self.template.pop("template_identity_sha256")
        self.template = contract.signed(self.template, "template_identity_sha256")
        with self.assertRaises(ValueError): self.seal()

    def test_attempt_and_checkpoint_target_collisions(self):
        for key in ("attempt_dir", "checkpoint_200", "adapter_200", "adapter_200_receipt"):
            path = Path(self.template["paths"][key]); path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            with self.assertRaises(FileExistsError): contract.validate_collisions(self.template["paths"])
            path.unlink()

    def test_active_current_pointer_rejected(self):
        path = Path(self.template["paths"]["current_json"]); path.parent.mkdir(parents=True)
        path.write_text('{"status":"running"}')
        with self.assertRaises(RuntimeError): contract.validate_collisions(self.template["paths"])

    def runtime_plan(self):
        preflight_path = Path(self.template["paths"]["preflight"])
        preflight_path.write_bytes(contract.json_bytes(self.preflight))
        plan = self.seal(); plan["preflight"]["sha256"] = identity.sha256_file(preflight_path)
        plan.pop("plan_identity_sha256")
        return contract.signed(plan, "plan_identity_sha256")

    def runtime_check(self, plan, elapsed=0):
        with mock.patch.object(contract, "current_host", return_value=HOST), mock.patch.object(contract.time, "time", return_value=NOW_EPOCH + elapsed), mock.patch.object(contract.time, "monotonic", return_value=NOW_MONOTONIC + elapsed):
            contract.validate_launch_bindings(plan)

    def test_runtime_hash_recheck_detects_tool_change(self):
        plan = self.runtime_plan(); self.runtime_check(plan)
        Path(plan["tool_bindings"]["runner"]["path"]).write_text("# changed")
        with self.assertRaises(ValueError): self.runtime_check(plan)

    def test_runtime_hash_recheck_detects_manifest_change(self):
        plan = self.runtime_plan()
        Path(plan["input_files"]["golden_manifest"]["path"]).write_text("changed")
        with self.assertRaises(ValueError): self.runtime_check(plan)

    def test_runtime_expiry_after_sealing(self):
        plan = self.runtime_plan()
        with self.assertRaises(ValueError): self.runtime_check(plan, elapsed=200)

    def test_runtime_env_tamper_rejected(self):
        plan = self.runtime_plan(); plan["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
        plan.pop("plan_identity_sha256"); plan = contract.signed(plan, "plan_identity_sha256")
        with self.assertRaises(ValueError): self.runtime_check(plan)


if __name__ == "__main__": unittest.main()
