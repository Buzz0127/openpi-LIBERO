#!/usr/bin/env python3

from __future__ import annotations

import itertools
import unittest

import resume_sequence


class ResumeSequenceTest(unittest.TestCase):
    def test_step_100_uses_batch_index_100(self) -> None:
        iterator, batch, receipt = resume_sequence.verify_position(
            lambda: iter(range(1000)), 100, resume_sequence.canonical_fingerprint
        )
        self.assertEqual(batch, 100)
        self.assertEqual(next(iterator), 101)
        self.assertEqual(receipt["skipped_batch_count"], 100)
        self.assertTrue(receipt["position_verified"])

    def test_infinite_epoch_sequence_is_positioned_exactly(self) -> None:
        def factory():
            return itertools.cycle(["a", "b", "c"])
        _, batch, _ = resume_sequence.verify_position(factory, 100, resume_sequence.canonical_fingerprint)
        self.assertEqual(batch, "b")

    def test_negative_step_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resume_sequence.positioned_batch(range(3), -1)

    def test_short_loader_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ended"):
            resume_sequence.verify_position(lambda: iter(range(2)), 3, resume_sequence.canonical_fingerprint)

    def test_mismatched_second_loader_fails_closed(self) -> None:
        count = 0
        def factory():
            nonlocal count
            count += 1
            return iter(range(10)) if count == 1 else iter(range(1, 11))
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            resume_sequence.verify_position(factory, 3, resume_sequence.canonical_fingerprint)


if __name__ == "__main__":
    unittest.main()
