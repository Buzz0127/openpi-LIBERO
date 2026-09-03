from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import audit_libero_norm_inputs as audit


class AuditLiberoNormInputsTest(unittest.TestCase):
    def test_query_indices_clamp_at_episode_end(self) -> None:
        self.assertEqual(audit.expected_query_indices(13, 10, 15, 5), [13, 14, 14, 14, 14])
        self.assertEqual(audit.expected_padding(13, 10, 15, 5), [False, False, True, True, True])

    def test_query_indices_reject_out_of_episode_frame(self) -> None:
        with self.assertRaises(ValueError):
            audit.expected_query_indices(15, 10, 15, 5)

    def test_probe_selection_covers_edges_and_four_blocks(self) -> None:
        tasks = [{"task_index": index, "task": f"task-{index}"} for index in range(40)]
        episodes = [
            {"episode_index": index, "tasks": [f"task-{task_index}"], "length": 3}
            for index, task_index in enumerate([0, 10, 20, 30, 1])
        ]
        ranges = {index: (index * 3, index * 3 + 3) for index in range(len(episodes))}
        selected = audit.select_probe_frames(tasks, episodes, ranges)
        reasons = {reason for record in selected for reason in record["reasons"]}
        for suite in audit.SUITE_BLOCKS:
            self.assertIn(f"{suite}:representative_start", reasons)
            self.assertIn(f"{suite}:representative_end", reasons)
        self.assertIn("first_episode:middle", reasons)
        self.assertIn("last_episode:middle", reasons)
        self.assertEqual(len({record["frame_index"] for record in selected}), len(selected))

    def test_probe_selection_rejects_missing_task_block(self) -> None:
        tasks = [{"task_index": index, "task": f"task-{index}"} for index in range(40)]
        episodes = [{"episode_index": 0, "tasks": ["task-0"], "length": 3}]
        with self.assertRaisesRegex(ValueError, "representative libero_goal"):
            audit.select_probe_frames(tasks, episodes, {0: (0, 3)})

    def test_atomic_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            audit._atomic_json(output, {"status": "first"})
            with self.assertRaises(FileExistsError):
                audit._atomic_json(output, {"status": "second"})
            self.assertEqual(json.loads(output.read_text()), {"status": "first"})


if __name__ == "__main__":
    unittest.main()
