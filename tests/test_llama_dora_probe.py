"""Actual dense PEFT0.10 DoRA algebra; no mocked model or GPU."""
from types import SimpleNamespace
import json
import unittest

import torch

from experiments.llama_dora_probe.state import merge_adapter, select_adapter
from train_utils.distill_precision import configure_distill_precision


class DenseDoraTests(unittest.TestCase):
    def test_native_dora_formula_magnitude_and_bf16_rounding(self):
        torch.manual_seed(31)
        base = torch.nn.Sequential(torch.nn.Linear(32, 48, bias=False)).bfloat16()
        selection = select_adapter(base, ['0'], (), use_dora=True)
        wrapped = selection.peft_model
        configure_distill_precision(selection, components=('lora',),
                                   training_args=SimpleNamespace(bf16=True, fp16=False))
        layer = wrapped.get_base_model()[0]
        magnitude = layer.lora_magnitude_vector['default']
        self.assertTrue(any(p is magnitude for p in selection.lora_parameters.values()))
        self.assertEqual(magnitude.dtype, torch.float32)
        with torch.no_grad():
            layer.lora_B['default'].weight.normal_(std=.02)
            magnitude.mul_(1.03)
        weight = layer.base_layer.weight.detach().float()
        delta = layer.lora_B['default'].weight @ layer.lora_A['default'].weight * 2
        combined = weight + delta
        expected = combined * (magnitude / combined.norm(dim=1))[:, None]
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        wrapped.eval()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            before = wrapped(x)
        before.float().square().mean().backward()
        self.assertIsNotNone(magnitude.grad)
        self.assertTrue(torch.isfinite(magnitude.grad).all())
        self.assertGreater(float(magnitude.grad.norm()), 0)
        # Native merge runs outside autocast with FP32 trainable leaves.
        merged = merge_adapter(wrapped, SimpleNamespace(bf16=True, fp16=False))
        self.assertEqual(merged[0].weight.dtype, torch.bfloat16)
        torch.testing.assert_close(merged[0].weight, expected.detach().bfloat16(), rtol=0, atol=0)
        after = merged(x)
        # Matrix storage roundoff is bounded by BF16 unit roundoff; separately
        # computed low-rank outputs can add a few more BF16 rounding operations.
        rounding = merged[0].weight.float() - expected.detach()
        self.assertLessEqual(float(rounding.abs().max()), float(expected.detach().abs().max()) / 256 + 1e-7)
        relative_l2 = float((after.float() - before.float()).norm() / before.float().norm())
        self.assertLess(relative_l2, .02)
        self.assertFalse(any('lora_' in name for name, _ in merged.named_parameters()))
        self.assertFalse(torch.cuda.is_initialized())
        print(json.dumps({'arm': 'dora', 'merged_weight_matches_fp32_formula_cast_exactly': True,
            'unmerged_to_merged_bf16_output_relative_l2': relative_l2,
            'maximum_weight_rounding_error': float(rounding.abs().max()), 'cuda_initialized': False}))

    def test_plain_lora_uses_same_native_merge_and_has_no_magnitude(self):
        torch.manual_seed(31)
        selection = select_adapter(torch.nn.Sequential(torch.nn.Linear(32, 48, bias=False)).bfloat16(),
                                   ['0'], (), use_dora=False)
        configure_distill_precision(selection, components=('lora',),
                                   training_args=SimpleNamespace(bf16=True, fp16=False))
        self.assertFalse(any('magnitude' in name for name in selection.lora_parameters))
        layer = selection.peft_model.get_base_model()[0]
        with torch.no_grad():
            layer.lora_B['default'].weight.normal_(std=.02)
        expected = layer.base_layer.weight.float() + 2 * layer.lora_B['default'].weight @ layer.lora_A['default'].weight
        merged = merge_adapter(selection.peft_model, SimpleNamespace(bf16=True, fp16=False))
        torch.testing.assert_close(merged[0].weight, expected.detach().bfloat16(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
