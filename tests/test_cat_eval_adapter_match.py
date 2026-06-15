import argparse
import os
import tempfile
import unittest

import torch

from tools.cat_eval import _compute_checkpoint_fingerprint, _validate_adapter_checkpoint_match
from train_utils.cat_train_args import process_cat_train_args, resolve_lora_runtime_config
from train_utils.lora_training import (
    build_lora_hidden_layer_weights,
    compute_lora_hidden_alignment_loss,
)


class CatEvalAdapterMatchTest(unittest.TestCase):
    def _build_checkpoint_dir(self, root: str) -> tuple[str, dict]:
        ckpt_dir = os.path.join(root, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        meta = {
            "format": "vaellm_state_dict_with_meta",
            "version": 4,
            "state_dict_file": "pytorch_model.bin",
            "base_model_path": "meta-llama/Llama-3.1-8B",
            "converted_modules": [],
        }
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as handle:
            import json

            json.dump(meta, handle, ensure_ascii=False, indent=2)
        torch.save({"x": torch.tensor([1, 2, 3])}, os.path.join(ckpt_dir, "pytorch_model.bin"))
        return ckpt_dir, meta

    def test_validate_adapter_checkpoint_match_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir, meta = self._build_checkpoint_dir(tmpdir)
            fingerprint = _compute_checkpoint_fingerprint(ckpt_dir, meta)
            adapter_meta = {
                "source_checkpoint_meta_sha256": fingerprint["meta_sha256"],
                "source_checkpoint_state_sha256": fingerprint["state_sha256"],
            }
            resolved = _validate_adapter_checkpoint_match(
                checkpoint_dir=ckpt_dir,
                checkpoint_meta=meta,
                adapter_meta=adapter_meta,
            )
            self.assertEqual(resolved["meta_sha256"], fingerprint["meta_sha256"])
            self.assertEqual(resolved["state_sha256"], fingerprint["state_sha256"])

    def test_validate_adapter_checkpoint_match_fail_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir, meta = self._build_checkpoint_dir(tmpdir)
            adapter_meta = {
                "source_checkpoint_meta_sha256": "0" * 64,
                "source_checkpoint_state_sha256": "1" * 64,
            }
            with self.assertRaises(ValueError):
                _validate_adapter_checkpoint_match(
                    checkpoint_dir=ckpt_dir,
                    checkpoint_meta=meta,
                    adapter_meta=adapter_meta,
                )


class CatLoraHiddenArgsTest(unittest.TestCase):
    def test_lora_hidden_loss_defaults_to_disabled_uniform(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args([])
        cfg = resolve_lora_runtime_config(cat_args, after_category=None)

        self.assertEqual(cfg.hidden_loss_weight, 0.0)
        self.assertEqual(cfg.hidden_layer_weighting, "uniform")

    def test_lora_hidden_loss_resolves_after_category_overrides(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args(
            [
                "--lora_hidden_loss_weight",
                "default=0.01,after:q_proj=0.02",
                "--lora_hidden_layer_weighting",
                "linear_depth",
            ]
        )

        default_cfg = resolve_lora_runtime_config(cat_args, after_category=None)
        q_proj_cfg = resolve_lora_runtime_config(cat_args, after_category="q_proj")

        self.assertEqual(default_cfg.hidden_loss_weight, 0.01)
        self.assertEqual(q_proj_cfg.hidden_loss_weight, 0.02)
        self.assertEqual(default_cfg.hidden_layer_weighting, "linear_depth")
        self.assertEqual(q_proj_cfg.hidden_layer_weighting, "linear_depth")

    def test_lora_hidden_loss_rejects_negative_weight(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--lora_hidden_loss_weight", "default=-0.01"])

    def test_lora_hidden_loss_rejects_unknown_weighting(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--lora_hidden_layer_weighting", "quadratic"])


class CatLoraHiddenLossTest(unittest.TestCase):
    def test_uniform_hidden_loss_skips_embedding_state(self):
        teacher = (
            torch.zeros(1, 2, 1),
            torch.ones(1, 2, 1),
            torch.full((1, 2, 1), 2.0),
        )
        student = (
            torch.full((1, 2, 1), 100.0),
            torch.full((1, 2, 1), 2.0),
            torch.full((1, 2, 1), 4.0),
        )

        loss = compute_lora_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=torch.ones(1, 2),
            layer_weighting="uniform",
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_hidden_loss_attention_mask_excludes_padding_tokens(self):
        teacher = (
            torch.zeros(1, 2, 1),
            torch.tensor([[[1.0], [10.0]]]),
        )
        student = (
            torch.zeros(1, 2, 1),
            torch.tensor([[[2.0], [100.0]]]),
        )

        loss = compute_lora_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=torch.tensor([[1, 0]]),
            layer_weighting="uniform",
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_linear_depth_weights_increase_and_average_to_one(self):
        weights = build_lora_hidden_layer_weights(
            num_layers=4,
            layer_weighting="linear_depth",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertTrue(torch.all(weights[1:] > weights[:-1]))
        self.assertTrue(torch.allclose(weights.mean(), torch.tensor(1.0)))

    def test_linear_depth_hidden_loss_uses_normalized_layer_weights(self):
        teacher = (
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
        )
        student = (
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.full((1, 1, 1), 2.0),
        )

        loss = compute_lora_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=None,
            layer_weighting="linear_depth",
        )

        expected_weights = build_lora_hidden_layer_weights(
            num_layers=2,
            layer_weighting="linear_depth",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(loss, expected_weights[1] / 2.0))


if __name__ == "__main__":
    unittest.main()
