from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import audit_final_report_evidence as audit


class FinalReportEvidenceAuditTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_accepts_locked_results_and_rejects_changed_e3_count(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = self._write(root, "base.json", {"stage": "E2-base-recovery-200", "status": "pass", "completed": 200, "selection_lock_sha256": "x"})
            main = self._write(root, "main.json", {"stage": "E2-main-400", "status": "pass", "completed": 400, "selection_lock_sha256": "x"})
            paired = self._write(root, "paired.json", {"stage": "E2-main-paired-audit", "status": "pass", "denominator": 200, "base": {"successes": 0}, "pure_lora": {"successes": 25}, "paired_outcomes": {"pure_lora_only": 25, "both_failure": 175}})
            e3_value = {"stage": "G2-e3-user-stopped-partial", "completed_valid_unique_episodes": 382, "success_episodes": 19, "evaluator_exceptions": 0, "next_stage_started": False, "port_18001_released": True}
            e3 = self._write(root, "e3.json", e3_value)
            self.assertEqual(audit.audit(base, main, paired, e3)["status"], "pass")
            e3_value["success_episodes"] = 20
            e3.write_text(json.dumps(e3_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "E3 success"):
                audit.audit(base, main, paired, e3)


if __name__ == "__main__":
    unittest.main()
