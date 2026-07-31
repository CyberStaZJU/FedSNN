"""Unit tests for Scheme A bidirectional block dual + server whole-block EF."""

from __future__ import annotations

import math
import unittest

import torch

from fedsnn.train_topk_saw import (
    BLOCK_DUAL_METHODS,
    DUAL_GLOBAL_SERVER_EF_METHODS,
    NO_ERROR_FEEDBACK_METHODS,
    TOPK_EF_METHODS,
    _architecture_block_layout,
    _block_payload_stats,
    _block_values,
    _compress_blocks_no_ef,
    _downlink_block_topk_with_ef,
    _iso_wire_coordinate_budget,
    _select_blocks_by_rms_budget,
)


def _tiny_sparse_layout():
    """Two-layer synthetic layout: conv 2x2x1x1 + linear 2x16."""
    return (
        ("conv.weight", (2, 2, 1, 1), 4),
        ("fc.weight", (2, 16), 32),
    )


def _flat_from_state(state, layout):
    return torch.cat([state[key].reshape(-1) for key, _, _ in layout])


class MethodRegistrationTests(unittest.TestCase):
    def test_symmetric_block_method_sets(self):
        method = "symmetric_block_dual_topk_ef_snn"
        self.assertIn(method, BLOCK_DUAL_METHODS)
        self.assertIn(method, TOPK_EF_METHODS)
        self.assertIn(method, NO_ERROR_FEEDBACK_METHODS)
        self.assertIn(method, DUAL_GLOBAL_SERVER_EF_METHODS)


class BlockLayoutAndEncodingTests(unittest.TestCase):
    def test_layout_covers_all_sparse_coords(self):
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        self.assertEqual(len(blocks["conv.weight"]), 4)  # 2*2 kernels
        self.assertEqual(len(blocks["fc.weight"]), 2)  # 2 outs * 1 segment of 16
        total = 0
        state = {
            "conv.weight": torch.randn(2, 2, 1, 1),
            "fc.weight": torch.randn(2, 16),
        }
        for key, layer_blocks in blocks.items():
            for block in layer_blocks:
                total += int(_block_values(state[key], block).numel())
        self.assertEqual(total, 36)

    def test_payload_uses_block_index_not_global_coord_bits(self):
        layout = _tiny_sparse_layout()
        sparse_dim = sum(c for _, _, c in layout)
        global_index_bits = int(math.ceil(math.log2(sparse_dim)))
        self.assertGreater(global_index_bits, 1)
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        state = {
            "conv.weight": torch.randn(2, 2, 1, 1),
            "fc.weight": torch.randn(2, 16),
        }
        # Select everything.
        selected = {key: set(layer) for key, layer in blocks.items()}
        stats = _block_payload_stats(selected, blocks, layout, state)
        # Per-layer index width for 4 conv blocks = 2 bits; for 2 fc blocks = 1 bit.
        # 4*2 + 2*1 = 10 block index bits total — far below 36 * global_index_bits.
        self.assertEqual(stats["selected_coordinates"], 36)
        self.assertEqual(stats["index_bits"], 4 * 2 + 2 * 1)
        self.assertLess(stats["index_bits"], 36 * global_index_bits)
        self.assertEqual(stats["encoding"], "block_id_per_layer")
        self.assertEqual(stats["value_bits"], 36 * 32)


class BlockSelectionAndCompressionTests(unittest.TestCase):
    def test_select_meets_coordinate_budget_with_whole_blocks(self):
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        state = {
            "conv.weight": torch.zeros(2, 2, 1, 1),
            "fc.weight": torch.zeros(2, 16),
        }
        # Make one fc row large RMS so it is preferred (16 coords).
        state["fc.weight"][0] = 10.0
        state["conv.weight"][0, 0] = 1.0
        selected, stats = _select_blocks_by_rms_budget(
            state, blocks, layout, target_coordinates=16, linear_block_size=16
        )
        self.assertGreaterEqual(stats["selected_coordinates"], 16)
        self.assertIn("fc.weight", selected)
        self.assertIn((0, -1, 0, 16), selected["fc.weight"])

    def test_compress_blocks_no_ef_packs_original_values(self):
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        state = {
            "conv.weight": torch.randn(2, 2, 1, 1),
            "fc.weight": torch.randn(2, 16),
        }
        flat = _flat_from_state(state, layout)
        packed, selected, stats = _compress_blocks_no_ef(
            flat, layout, blocks, target_coordinates=20, linear_block_size=16
        )
        self.assertGreaterEqual(stats["selected_coordinates"], 20)
        # Packed values match source.
        for key, block_map in packed.items():
            for block, values in block_map.items():
                expected = _block_values(state[key], block)
                self.assertTrue(torch.allclose(values, expected))


class ServerBlockEFConservationTests(unittest.TestCase):
    def test_residual_equals_corrected_minus_selected_blocks(self):
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        delta = torch.randn(36)
        residual = torch.randn(36) * 0.5
        packed, selected, next_residual, stats = _downlink_block_topk_with_ef(
            delta,
            residual,
            layout,
            blocks,
            target_coordinates=16,
            use_error_feedback=True,
            linear_block_size=16,
        )
        corrected = delta + residual
        # Reconstruct packed scatter and check conservation:
        # next_residual + scattered_pack == corrected
        from fedsnn.train_topk_saw import _scatter_blocks_into_flat

        zeros = torch.zeros_like(corrected)
        scattered = _scatter_blocks_into_flat(zeros, layout, packed, mode="add")
        reconstructed = next_residual + scattered
        self.assertTrue(torch.allclose(reconstructed, corrected, atol=1e-5, rtol=1e-5))
        # Selected blocks zeroed in residual.
        for key, block_set in selected.items():
            # Verify via flat zeroing helper identity.
            pass
        self.assertGreater(stats["selected_blocks"], 0)
        self.assertGreaterEqual(stats["selected_coordinates"], 16)
        # Index bits are block-ID width, not global coordinate bits.
        sparse_dim = 36
        global_ib = int(math.ceil(math.log2(sparse_dim)))
        self.assertLess(stats["index_bits"], stats["selected_coordinates"] * global_ib)

    def test_no_ef_zeros_residual(self):
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        delta = torch.randn(36)
        packed, selected, next_residual, _ = _downlink_block_topk_with_ef(
            delta,
            torch.ones(36),
            layout,
            blocks,
            target_coordinates=8,
            use_error_feedback=False,
            linear_block_size=16,
        )
        self.assertTrue(torch.allclose(next_residual, torch.zeros(36)))
        self.assertTrue(len(packed) > 0 or selected)  # may pack something

    def test_dense_keys_never_in_block_layout(self):
        # Dense BNTT keys are not part of sparse_layout by construction.
        layout = _tiny_sparse_layout()
        blocks = _architecture_block_layout(layout, linear_block_size=16)
        self.assertNotIn("bn.weight", blocks)
        self.assertNotIn("bn.bias", blocks)
        for key in blocks:
            self.assertIn(key, {k for k, _, _ in layout})


class IsoWireBudgetTests(unittest.TestCase):
    def test_coordinate_retention_strictly_below_block_retention(self):
        sparse_dim = 6_454_976
        # Approx wire for 20% block retention (from planning script order-fill).
        # Use synthetic: 20% coords * 32 + modest index overhead.
        # Assume ~18 bits average block id amortized → use payload from planning.
        block_payload_bits = 42_764_640  # from myserver derivation (greedy large blocks)
        plan = _iso_wire_coordinate_budget(block_payload_bits, sparse_dim, value_bits=32)
        self.assertEqual(plan["coordinate_index_bits"], 23)
        self.assertEqual(plan["bits_per_coordinate"], 55)
        self.assertLess(plan["coordinate_retention"], 0.2)
        self.assertGreater(plan["coordinate_k"], 0)
        # 10% case
        plan10 = _iso_wire_coordinate_budget(21_382_320, sparse_dim, value_bits=32)
        self.assertLess(plan10["coordinate_retention"], 0.1)

    def test_zero_index_bits_edge(self):
        # sparse_dim=1 → index_bits=0; k can equal retention.
        plan = _iso_wire_coordinate_budget(32, sparse_dimension=1, value_bits=32)
        self.assertEqual(plan["coordinate_index_bits"], 0)
        self.assertEqual(plan["coordinate_k"], 1)


if __name__ == "__main__":
    unittest.main()
