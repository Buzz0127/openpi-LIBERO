"""Synthetic CPU-only template and exclusive publication checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_t1_execution_template as builder
import experiment_identity as identity
import t1_execution_contract as contract


def freeze_sign(value):
    unsigned = {key: item for key, item in value.items() if key != "package_identity_sha256"}
    return {**unsigned, "package_identity_sha256": hashlib.sha256((json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()}


def fixture(root: Path):
    """Every file below is a few-byte fake; no repository artifacts are opened."""
    source = {"root": str(root / "source"), "branch": "feature/fake", "head": "a" * 40, "status": "clean", "upstream": "NONE"}
    checkpoint = str(root / "fake-checkpoint-root")
    resume = {"checkpoint_root": checkpoint, "checkpoint_tree_sha256": "1" * 64, "acceptance_report_identity_sha256": "2" * 64, "adapter_identity_sha256": "3" * 64, "checkpoint_step": 100}
    storage = {"current_billed_bytes": 91_765_739_008, "measured_full_state_bytes": 5_559_083_375, "measured_adapter_bytes": 199_962_483, "atomic_write_margin_bytes": 1_000_000_000, "review_line_bytes": 225_000_000_000, "soft_stop_bytes": 240_000_000_000, "hard_limit_bytes": 250_000_000_000, "minimum_uncommitted_bytes": 20_000_000_000, "deletion_authorized": False, "automatic_pruning_allowed": False}
    freeze = freeze_sign({"execution_authorized": False, "automatic_next_stage": False, "engineering_segment": {"start": 100, "end": 200, "candidate": False, "delete_step_100_after_success": False, "batch_size": 1, "num_workers": 0}, "seeds": {"training": 42, "evaluation": 7}, "identities": {key: "a" * 64 for key in contract.IDENTITIES}, "source": source, "resume_input": resume, "storage": storage})
    paths = {"python": str(root / "python"), "tmux": str(root / "tmux"), "attempt_dir": str(root / "stage" / "attempt-fake"), "current_json": str(root / "stage" / "current.json"), "stage_plan": str(root / "plan.json"), "launch_review": str(root / "launch-review.json"), "model_manifest": str(root / "model.json"), "golden_manifest": str(root / "golden.json"), "s1d_acceptance_report": str(root / "s1d.json"), "freeze_package": str(root / "freeze.json"), "checkpoint_root": checkpoint, "adapter_root": str(root / "fake-adapter-root"), "preflight": str(root / "preflight.json")}
    inputs = {}
    for role in contract.INPUT_ROLES:
        path = Path(paths[role]); path.write_text(json.dumps(freeze) if role == "freeze_package" else "{}\n")
        inputs[role] = {"path": str(path), "sha256": identity.sha256_file(path)}
    tool_dir = root / "snapshot" / "tools" / "pi0_pure_lora"; tool_dir.mkdir(parents=True)
    tools = {}
    for role in sorted(contract.TOOL_ROLES):
        path = tool_dir / f"{role}.py"; path.write_text(f"# fake {role}\n")
        tools[role] = {"path": str(path), "sha256": identity.sha256_file(path)}
    readiness = {"status": "pass", "execution_authorized": False, "gpu_used": False, "model_loaded": False, "checkpoint_loaded": False, "real_dataset_loaded": False, "openpi_source": source, "input_identities": {"t1_freeze_package_identity_sha256": freeze["package_identity_sha256"], "s1d_acceptance_report_identity_sha256": resume["acceptance_report_identity_sha256"], "s1d_checkpoint_tree_sha256": resume["checkpoint_tree_sha256"], "s1d_adapter_identity_sha256": resume["adapter_identity_sha256"], **{key: inputs[role]["sha256"] for role, key in {"freeze_package": "t1_freeze_package_sha256", "s1d_acceptance_report": "s1d_acceptance_report_sha256", "model_manifest": "base_model_manifest_file_sha256", "golden_manifest": "golden_manifest_file_sha256"}.items()}}}
    return freeze, readiness, {"paths": paths, "tools": tools, "input_files": inputs}


class TemplateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.freeze, self.readiness, self.spec = fixture(self.root)

    def build(self):
        return builder.build_template(self.freeze, self.readiness, self.spec)

    def test_static_template_has_no_command_gpu_or_authorization(self):
        value = self.build()
        contract.verify_identity(value, "template_identity_sha256")
        for key in ("command", "environment", "selected_physical_gpu", "selected_gpu_uuid"):
            self.assertIsNone(value[key])
        for key in ("execution_authorized", "next_stage_auto_start", "execution_ready", "historical_readiness_validates_current_tools"):
            self.assertIs(value[key], False)
        self.assertLess(value["bounds"]["timeout_seconds"] + value["bounds"]["term_grace_seconds"] + value["bounds"]["kill_grace_seconds"], 7200)

    def test_builder_does_not_read_real_artifacts_or_inspect_gpu(self):
        with mock.patch.object(Path, "open", side_effect=AssertionError("builder opened a file")):
            self.build()

    def test_freeze_identity_tamper(self):
        self.freeze["seeds"]["training"] = 43
        with self.assertRaises(ValueError): self.build()

    def test_rehashed_segment_tamper(self):
        self.freeze["engineering_segment"]["delete_step_100_after_success"] = True
        self.freeze = freeze_sign(self.freeze)
        with self.assertRaises(ValueError): self.build()

    def test_readiness_identity_tamper(self):
        self.readiness["input_identities"]["s1d_checkpoint_tree_sha256"] = "4" * 64
        with self.assertRaises(ValueError): self.build()

    def test_missing_tool_fails_closed(self):
        self.spec["tools"].pop("runner")
        with self.assertRaises(ValueError): self.build()

    def test_manifest_hash_drift(self):
        self.spec["input_files"]["model_manifest"]["sha256"] = "4" * 64
        with self.assertRaises(ValueError): self.build()

    def test_alias_or_escape_outputs_rejected(self):
        self.spec["paths"]["stage_plan"] = self.spec["paths"]["attempt_dir"] + "/plan.json"
        with self.assertRaises(ValueError): self.build()

    def test_checkpoint_output_wrong_root(self):
        self.spec["paths"]["checkpoint_200"] = str(self.root / "elsewhere")
        with self.assertRaises(ValueError): self.build()

    def test_atomic_publication_preserves_existing_file(self):
        path = self.root / "published.json"
        contract.write_new(path, {"first": 1})
        with self.assertRaises(FileExistsError): contract.write_new(path, {"second": 2})
        self.assertEqual(json.loads(path.read_text()), {"first": 1})

    def test_dangling_symlink_collision(self):
        path = self.root / "link.json"; path.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError): contract.write_new(path, {})

    def test_exclusive_link_race_cannot_overwrite(self):
        path = self.root / "race.json"
        original = contract.os.link
        def competing_link(source, target):
            Path(target).write_text("winner")
            return original(source, target)
        with mock.patch.object(contract.os, "link", side_effect=competing_link):
            with self.assertRaises(FileExistsError): contract.write_new(path, {"loser": True})
        self.assertEqual(path.read_text(), "winner")


if __name__ == "__main__": unittest.main()
