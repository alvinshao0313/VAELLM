"""CPU tests of the proposed measurement and score-only intervention contracts."""
import unittest

import torch

from experiments.sparse_bit_deoscillation.controller import BankOscillation, matched_masks


class OscillationTest(unittest.TestCase):
    def test_return_ratio_and_intervention_excluded(self):
        score = torch.tensor([-.5, -.5, .5])
        tracker = BankOscillation(score, 1.)
        score.copy_(torch.tensor([.01, -.01, .5]))
        first = tracker.observe(score, step=1)
        self.assertEqual(first["step_flip_events"], 1)
        self.assertFalse(bool(tracker.risk_mask(1).any()))
        score.copy_(torch.tensor([-.01, .01, .5]))
        second = tracker.observe(score, step=2)
        self.assertEqual(second["returns_to_window_start"], 1)
        self.assertEqual(second["flipbacks_within_updates"]["1"], 1)
        risk = tracker.risk_mask(2)
        self.assertEqual(risk.tolist(), [True, False, False])
        hard = score >= 0
        result = tracker.reanchor(score, risk)
        self.assertEqual(result["selected_bits"], 1)
        self.assertTrue(torch.equal(score >= 0, hard))
        third = tracker.observe(score, step=3)
        self.assertEqual(third["natural_u_travel"], 0)
        self.assertEqual(third["step_flip_events"], 0)
        window = tracker.finish_window(score)
        self.assertEqual((window["flip_events"], window["net_bits"], window["return_pairs"]), (3, 1, 1))

    def test_zero_score_hard_one_and_anchor_roundtrip(self):
        score = torch.tensor([0., -.25])
        tracker = BankOscillation(score, 1.)
        tracker.observe(score, step=1)
        state = tracker.state_dict()
        tracker.reanchor(score, torch.tensor([True, True]))
        self.assertEqual(score.tolist(), [.5, -.5])
        tracker.finish_window(score)
        tracker.load_state_dict(state)
        self.assertEqual(tracker.step, 1)
        self.assertEqual(tracker.window_start, 0)
        self.assertTrue(torch.equal(tracker.previous, state["previous"]))
        self.assertEqual(state["previous"].tolist(), [0., -.25])

    def test_controls_match_distance_bank_and_count(self):
        score = torch.tensor([.01, -.02, .03, -.04, .30, -.31, .32, -.33])
        risk = torch.tensor([True, False, False, False, True, False, False, False])
        selected, first, info = matched_masks(score, 1., risk, distance_bins=2, seed=137)
        _, second, _ = matched_masks(score, 1., risk, distance_bins=2, seed=137)
        self.assertTrue(torch.equal(selected, risk))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(bool((first & risk).any()))
        self.assertEqual(int(first[:4].sum()), 1)
        self.assertEqual(int(first[4:].sum()), 1)
        self.assertEqual(info["max_abs_u_bin_width"], .25)

    def test_unmatched_risk_is_reported_and_missing_step_fails(self):
        score = torch.tensor([.01, .3])
        risk, control, info = matched_masks(score, 1., torch.tensor([True, False]), distance_bins=2, seed=0)
        self.assertFalse(bool(risk.any() or control.any()))
        self.assertEqual(info["raw_risk_count"], 1)
        self.assertEqual(info["matched_pair_count"], 0)
        with self.assertRaisesRegex(ValueError, "every optimizer"):
            BankOscillation(score, 1.).observe(score, step=2)


if __name__ == "__main__":
    unittest.main()
