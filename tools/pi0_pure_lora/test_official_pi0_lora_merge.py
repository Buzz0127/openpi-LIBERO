from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

import official_pi0_lora_merge as merge


class OfficialMergeNumpyTest(unittest.TestCase):
    def _golden(self):
        return {
            "entries": [{"path": path} for path in sorted(merge.expected_adapter_paths())],
            "review_invariants": {"adapter_leaf_count": 20},
        }

    def _trees(self):
        base = {}
        adapter = {}
        for index, rule in enumerate(merge.MERGE_RULES):
            leading = (2, 2) if "kv" in rule.name else (2,)
            rows = 3
            rank = 2
            columns = 4
            base[rule.target] = np.full(leading + (rows, columns), index + 1, dtype=np.float32)
            adapter[rule.adapter_a] = np.full(leading + (rows, rank), 0.25, dtype=np.float32)
            adapter[rule.adapter_b] = np.full(leading + (rank, columns), 0.5, dtype=np.float32)
        base["untouched/kernel"] = np.array([9.0], dtype=np.float32)
        return base, adapter

    def test_merge_covers_exact_paths_and_keeps_non_target_identity(self):
        base, adapter = self._trees()
        untouched = base["untouched/kernel"]
        dense, audit = merge.merge_in_place(
            base, adapter, self._golden(),
            attention_scales={
                "gemma_2b.attention": 0.5,
                "gemma_2b.feedforward_declared": 0.5,
                "gemma_300m.attention": 0.25,
                "gemma_300m.feedforward_declared": 0.25,
            },
        )
        self.assertIs(dense["untouched/kernel"], untouched)
        self.assertEqual(audit["adapter_leaf_count"], 20)
        self.assertEqual(audit["target_count"], 10)
        attention = merge.MERGE_RULES[0]
        feedforward = merge.MERGE_RULES[3]
        # A@B is 0.25 for each cell. Attention applies 0.5; FFN does not.
        self.assertTrue(np.allclose(dense[attention.target], 1.125))
        self.assertTrue(np.allclose(dense[feedforward.target], 4.25))

    def test_zero_adapter_restores_base_and_bad_trees_fail_closed(self):
        base, adapter = self._trees()
        original = {key: value.copy() for key, value in base.items()}
        for value in adapter.values():
            value.fill(0)
        dense, _ = merge.merge_in_place(
            base, adapter, self._golden(),
            attention_scales={"gemma_2b.attention": 1, "gemma_300m.attention": 1},
        )
        for path, value in original.items():
            np.testing.assert_array_equal(dense[path], value)
        missing = dict(adapter)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "exactly"):
            merge.merge_in_place(copy.deepcopy(original), missing, self._golden(), attention_scales={"gemma_2b.attention": 1, "gemma_300m.attention": 1})

    def test_validate_dense_reference_rejects_mismatched_shape(self):
        reference = {"a": np.empty((2, 3)), "b": np.empty((4,))}
        dense = {"a": np.empty((2, 3)), "b": np.empty((4,))}
        old_count = merge.EXPECTED_DENSE_LEAF_COUNT
        old_parameters = merge.EXPECTED_DENSE_PARAMETER_COUNT
        try:
            merge.EXPECTED_DENSE_LEAF_COUNT = 2
            merge.EXPECTED_DENSE_PARAMETER_COUNT = 10
            self.assertEqual(merge.validate_dense_reference(dense, reference), {"leaf_count": 2, "parameter_count": 10})
            dense["a"] = np.empty((3, 2))
            with self.assertRaisesRegex(ValueError, "shape"):
                merge.validate_dense_reference(dense, reference)
        finally:
            merge.EXPECTED_DENSE_LEAF_COUNT = old_count
            merge.EXPECTED_DENSE_PARAMETER_COUNT = old_parameters


@unittest.skipUnless(os.environ.get("OPENPI_OFFICIAL_ROOT"), "requires fixed official OpenPI source")
class OfficialOperatorOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["OPENPI_OFFICIAL_ROOT"] + "/src")
        import flax
        import jax
        import jax.numpy as jnp
        from openpi.models import lora
        cls.flax, cls.jax, cls.jnp, cls.lora = flax, jax, jnp, lora

    def test_fixed_official_einsum_and_dense_kernel_agree(self):
        lora = self.lora
        x = self.jnp.arange(12, dtype=self.jnp.float32).reshape(2, 2, 3)
        equation = "BSD,DF->BSF"
        module = lora.Einsum((3, 4), lora_config=lora.LoRAConfig(rank=2, alpha=6.0))
        params = module.init(self.jax.random.key(0), equation, x)
        mutable = self.flax.core.unfreeze(params)
        mutable["params"]["w"] = self.jnp.full((3, 4), 0.5)
        mutable["params"]["lora_a"] = self.jnp.arange(6, dtype=self.jnp.float32).reshape(3, 2) / 10
        mutable["params"]["lora_b"] = self.jnp.arange(8, dtype=self.jnp.float32).reshape(2, 4) / 10
        params = self.flax.core.freeze(mutable)
        expected = module.apply(params, equation, x)
        dense_w = np.asarray(mutable["params"]["w"]) + 3.0 * np.matmul(np.asarray(mutable["params"]["lora_a"]), np.asarray(mutable["params"]["lora_b"]))
        dense = lora.Einsum((3, 4))
        actual = dense.apply({"params": {"w": self.jnp.asarray(dense_w)}}, equation, x)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

    def test_fixed_official_feedforward_has_no_scaling_value_multiplier(self):
        lora = self.lora
        x = self.jnp.arange(6, dtype=self.jnp.float32).reshape(2, 3)
        module = lora.FeedForward(features=3, hidden_dim=4, lora_config=lora.LoRAConfig(rank=2, alpha=6.0))
        params = module.init(self.jax.random.key(1), x)
        mutable = self.flax.core.unfreeze(params)
        for name, value in mutable["params"].items():
            mutable["params"][name] = self.jnp.ones_like(value) / 10
        params = self.flax.core.freeze(mutable)
        expected = module.apply(params, x)
        dense_params = {"params": {
            "gating_einsum": np.asarray(mutable["params"]["gating_einsum"]) + np.matmul(np.asarray(mutable["params"]["gating_einsum_lora_a"]), np.asarray(mutable["params"]["gating_einsum_lora_b"])),
            "linear": np.asarray(mutable["params"]["linear"]) + np.matmul(np.asarray(mutable["params"]["linear_lora_a"]), np.asarray(mutable["params"]["linear_lora_b"])),
        }}
        dense = lora.FeedForward(features=3, hidden_dim=4)
        actual = dense.apply(self.jax.tree.map(self.jnp.asarray, dense_params), x)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
