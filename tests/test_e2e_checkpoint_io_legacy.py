import unittest

import torch
from torch import nn

from e2e_common.checkpoint_io import _remap_legacy_decoder_keys_if_needed


class _DummyDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.linear_in = nn.Module()
        self.decoder.linear_in.linear = nn.Linear(4, 4, bias=True)


class LegacyCheckpointRemapTest(unittest.TestCase):
    def test_remap_legacy_decoder_linear_keys(self):
        model = _DummyDecoderModel()
        legacy_state = {
            "decoder.linear_in.weight": torch.randn(4, 4),
            "decoder.linear_in.bias": torch.randn(4),
        }
        remapped = _remap_legacy_decoder_keys_if_needed(model, legacy_state)
        self.assertNotIn("decoder.linear_in.weight", remapped)
        self.assertNotIn("decoder.linear_in.bias", remapped)
        self.assertIn("decoder.linear_in.linear.weight", remapped)
        self.assertIn("decoder.linear_in.linear.bias", remapped)

    def test_keep_new_decoder_linear_keys(self):
        model = _DummyDecoderModel()
        new_style_state = {
            "decoder.linear_in.linear.weight": torch.randn(4, 4),
            "decoder.linear_in.linear.bias": torch.randn(4),
        }
        remapped = _remap_legacy_decoder_keys_if_needed(model, new_style_state)
        self.assertIn("decoder.linear_in.linear.weight", remapped)
        self.assertIn("decoder.linear_in.linear.bias", remapped)
        self.assertEqual(set(remapped.keys()), set(new_style_state.keys()))


if __name__ == "__main__":
    unittest.main()
