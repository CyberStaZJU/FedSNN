"""Unit tests for dual-budget downlink helpers (D1/D2 + dual_global server EF)."""

from __future__ import annotations

import unittest

import torch

from fedsnn.train_topk_saw import (
    DUAL_GLOBAL_SERVER_EF_METHODS,
    NO_ERROR_FEEDBACK_METHODS,
    PURE_TOPK_METHODS,
    TOPK_EF_METHODS,
    _downlink_dual_channel_quota,
    _downlink_topk_with_credit,
    _stable_topk,
)


class DualChannelQuotaTests(unittest.TestCase):
    def test_respects_k_and_prefers_support(self):
        gap = torch.tensor(
            [10.0, 9.0, 8.0, 0.5, 0.4, 0.3, 0.2, 0.1],
            dtype=torch.float32,
        )
        # Support = first 3 coords (largest gaps).
        support = torch.tensor(
            [True, True, True, False, False, False, False, False]
        )
        indices, values = _downlink_dual_channel_quota(
            gap, k=4, support_mask=support, support_share=0.75
        )
        self.assertEqual(int(indices.numel()), 4)
        # 0.75 * 4 = 3 support slots + 1 cold slot.
        selected = set(int(i) for i in indices.tolist())
        self.assertTrue({0, 1, 2}.issubset(selected))
        self.assertEqual(len(selected & {3, 4, 5, 6, 7}), 1)
        # Values must equal gap at selected indices.
        for idx, value in zip(indices.tolist(), values.tolist()):
            self.assertAlmostEqual(value, float(gap[idx]), places=5)

    def test_cold_channel_gets_forced_share(self):
        gap = torch.arange(10, 0, -1, dtype=torch.float32)  # 10..1
        support = torch.zeros(10, dtype=torch.bool)
        support[:8] = True  # support owns the largest 8
        indices, _ = _downlink_dual_channel_quota(
            gap, k=4, support_mask=support, support_share=0.5
        )
        selected = set(int(i) for i in indices.tolist())
        # 2 support (0,1) + 2 cold among {8,9} ranked by |gap| → 8 then 9.
        self.assertEqual(len(selected & {0, 1, 2, 3, 4, 5, 6, 7}), 2)
        self.assertEqual(len(selected & {8, 9}), 2)

    def test_support_share_one_is_pure_support_then_fill(self):
        gap = torch.tensor([1.0, 5.0, 3.0, 4.0], dtype=torch.float32)
        support = torch.tensor([True, False, True, False])
        indices, _ = _downlink_dual_channel_quota(
            gap, k=3, support_mask=support, support_share=1.0
        )
        selected = set(int(i) for i in indices.tolist())
        # Both support coords must appear; remaining fill from cold by |gap|.
        self.assertTrue({0, 2}.issubset(selected))
        self.assertEqual(len(selected), 3)


class GapResidualConservationTests(unittest.TestCase):
    def test_top_k_gap_reduces_residual(self):
        global_flat = torch.randn(64)
        base_flat = torch.randn(64)
        gap = global_flat - base_flat
        k = 16
        indices = _stable_topk(gap.abs(), k)
        new_base = base_flat.clone()
        new_base[indices] = global_flat[indices]
        residual = global_flat - new_base
        # Selected coords are fully caught up.
        self.assertTrue(torch.allclose(residual[indices], torch.zeros(k)))
        # Unselected residual equals original gap there.
        mask = torch.ones(64, dtype=torch.bool)
        mask[indices] = False
        self.assertTrue(torch.allclose(residual[mask], gap[mask]))
        # Residual L2 strictly decreases unless gap was already zero on top-k.
        self.assertLessEqual(float(residual.norm()), float(gap.norm()) + 1e-5)


class DualGlobalServerEFTests(unittest.TestCase):
    def test_method_sets_uplink_noef_downlink_ef(self):
        method = "dual_global_topk_ef_snn"
        self.assertIn(method, TOPK_EF_METHODS)
        self.assertIn(method, NO_ERROR_FEEDBACK_METHODS)
        self.assertNotIn(method, PURE_TOPK_METHODS)
        self.assertIn(method, DUAL_GLOBAL_SERVER_EF_METHODS)

    def test_server_ef_accumulates_unsent_delta(self):
        dim = 32
        k = 4
        credit = torch.ones(dim)
        residual = None
        base = torch.zeros(dim)
        # Two rounds of sparse global increments on disjoint coordinates.
        delta1 = torch.zeros(dim)
        delta1[0] = 5.0
        delta1[1] = 4.0
        delta1[10] = 0.5
        idx1, vals1, residual = _downlink_topk_with_credit(
            delta1, residual, credit, k, use_error_feedback=True
        )
        base[idx1.to(torch.int64)] += vals1
        # Top-k of |delta1| should take 0,1 and the next largest coords.
        self.assertEqual(int(idx1.numel()), k)
        self.assertTrue(torch.allclose(residual[idx1.to(torch.int64)], torch.zeros(k)))
        # Unsent mass on coord 10 remains in residual when not selected.
        if 10 not in set(int(i) for i in idx1.tolist()):
            self.assertAlmostEqual(float(residual[10]), 0.5, places=5)

        delta2 = torch.zeros(dim)
        delta2[10] = 3.0  # stacks with residual if 10 was unsent
        delta2[11] = 2.5
        delta2[12] = 2.0
        delta2[13] = 1.5
        idx2, vals2, residual2 = _downlink_topk_with_credit(
            delta2, residual, credit, k, use_error_feedback=True
        )
        base[idx2.to(torch.int64)] += vals2
        self.assertEqual(int(idx2.numel()), k)
        # EF must make corrected ranking prefer residual+delta mass.
        selected = set(int(i) for i in idx2.tolist())
        self.assertIn(10, selected)
        # Selected residual coords zeroed; overall residual finite.
        self.assertTrue(torch.allclose(residual2[idx2.to(torch.int64)], torch.zeros(k)))
        self.assertTrue(torch.isfinite(residual2).all())

    def test_no_ef_does_not_carry_unsent(self):
        dim = 16
        k = 2
        credit = torch.ones(dim)
        delta = torch.zeros(dim)
        delta[0] = 9.0
        delta[1] = 8.0
        delta[5] = 1.0
        _, _, residual = _downlink_topk_with_credit(
            delta, None, credit, k, use_error_feedback=False
        )
        self.assertTrue(torch.allclose(residual, torch.zeros(dim)))


if __name__ == "__main__":
    unittest.main()
