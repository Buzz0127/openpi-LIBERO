from __future__ import annotations

import unittest

import resume_sequence


class ResumeSequenceProgressTests(unittest.TestCase):
    def test_verify_position_reports_both_replay_phases_without_changing_result(self) -> None:
        events: list[tuple[str, int, int]] = []
        iterator, batch, receipt = resume_sequence.verify_position(
            lambda: range(20), 7, lambda value: str(value), progress=lambda phase, done, total: events.append((phase, done, total)), progress_interval=3,
        )
        self.assertEqual(batch, 7)
        self.assertEqual(next(iterator), 8)
        self.assertTrue(receipt["position_verified"])
        self.assertIn(("reference_loader", 8, 8), events)
        self.assertIn(("positioned_loader", 7, 7), events)

    def test_rejects_nonpositive_progress_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            resume_sequence.verify_position(lambda: range(2), 0, str, progress_interval=0)
