#!/usr/bin/env python3

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

import eval_libero_pure_lora_bounded as evaluator
import extract_official_e0_reference as extract
import preregister_e0_splits as e0


ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/libero-benchmark/pi0_libero_official4_seed7/benchmark_summary.json"
EVALUATOR = ROOT / "tools/eval_libero_pure_lora_bounded.py"
C0_MANIFEST = ROOT / "manifests/pi0_pure_lora/base_model_manifest_c0.json"


class E0PreregistrationTest(unittest.TestCase):
    def build(self):
        return e0.build_manifest(SUMMARY, EVALUATOR, C0_MANIFEST, "openpi-commit", "libero-commit")

    def test_deterministic_balanced_disjoint_and_evaluator_compatible(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first, second)
        e0.validate_entries(first["entries"])
        self.assertEqual(first["schema_version"], 1)
        # These are the exact fields consumed by validate_task_state_selection().
        for row in first["entries"]:
            self.assertEqual(set(row), {"split", "suite", "task_id", "initial_state_index"})
        self.assertEqual(sum(row["split"] == "development" for row in first["entries"]), 40)
        self.assertEqual(sum(row["split"] == "main" for row in first["entries"]), 200)

    def test_selection_does_not_depend_on_outcomes(self):
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        altered = copy.deepcopy(summary)
        for task in altered["tasks"]:
            task["failed_initial_states"] = list(range(50))
            task["successes"] = 0
            task["failures"] = 50
        e0.validate_official_protocol(altered)
        self.assertEqual(e0.build_entries(), self.build()["entries"])

    def test_real_evaluator_accepts_every_preregistered_group(self):
        manifest = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            for split, expected_count in (("development", 1), ("main", 5)):
                for suite in e0.SUITES:
                    for task_id in range(e0.TASKS_PER_SUITE):
                        states = [
                            row["initial_state_index"]
                            for row in manifest["entries"]
                            if row["split"] == split and row["suite"] == suite and row["task_id"] == task_id
                        ]
                        self.assertEqual(len(states), expected_count)
                        args = type("Args", (), {
                            "task_state_manifest": path,
                            "evaluation_split": split,
                            "suite": suite,
                            "task_id": task_id,
                            "initial_states": states,
                        })()
                        evaluator.validate_task_state_selection(args)

    def test_manifest_tamper_is_rejected(self):
        manifest = self.build()
        extract.verify_manifest_identity(manifest)
        manifest["entries"][0]["initial_state_index"] = (manifest["entries"][0]["initial_state_index"] + 1) % 50
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            extract.verify_manifest_identity(manifest)

    def test_official_projection_has_40_and_200_episodes(self):
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        report = extract.extract_reference(self.build(), summary)
        self.assertEqual(report["aggregates"]["development"]["episodes"], 40)
        self.assertEqual(report["aggregates"]["main"]["episodes"], 200)
        self.assertEqual(len(report["entries"]), 240)

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            e0.atomic_write_new(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                e0.atomic_write_new(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
