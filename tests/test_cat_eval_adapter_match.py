import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.cat_eval import _compute_checkpoint_fingerprint, _validate_adapter_checkpoint_match
from train_utils import lora_utils
from train_utils.cat_train_args import process_cat_train_args, resolve_distill_runtime_config
from train_utils.lora_training import (
    build_distill_hidden_layer_weights,
    compute_distill_hidden_alignment_loss,
    compute_distill_pre_mlp_hidden_alignment_loss,
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


class CatDistillHiddenArgsTest(unittest.TestCase):
    def test_distill_hidden_loss_defaults_to_disabled_uniform(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args([])
        cfg = resolve_distill_runtime_config(cat_args, after_category=None)

        self.assertEqual(cfg.hidden_loss_weight, 0.0)
        self.assertEqual(cfg.pre_mlp_hidden_loss_weight, 0.0)
        self.assertEqual(cfg.hidden_alignment_layer_weighting, "uniform")

    def test_distill_hidden_loss_resolves_after_category_overrides(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args(
            [
                "--distill_hidden_loss_weight",
                "default=0.01,after:q_proj=0.02",
                "--distill_pre_mlp_hidden_loss_weight",
                "default=0.0,after:o_proj=0.01",
                "--distill_hidden_alignment_layer_weighting",
                "linear_depth",
            ]
        )

        default_cfg = resolve_distill_runtime_config(cat_args, after_category=None)
        q_proj_cfg = resolve_distill_runtime_config(cat_args, after_category="q_proj")
        o_proj_cfg = resolve_distill_runtime_config(cat_args, after_category="o_proj")

        self.assertEqual(default_cfg.hidden_loss_weight, 0.01)
        self.assertEqual(q_proj_cfg.hidden_loss_weight, 0.02)
        self.assertEqual(default_cfg.pre_mlp_hidden_loss_weight, 0.0)
        self.assertEqual(q_proj_cfg.pre_mlp_hidden_loss_weight, 0.0)
        self.assertEqual(o_proj_cfg.pre_mlp_hidden_loss_weight, 0.01)
        self.assertEqual(default_cfg.hidden_alignment_layer_weighting, "linear_depth")
        self.assertEqual(q_proj_cfg.hidden_alignment_layer_weighting, "linear_depth")
        self.assertEqual(o_proj_cfg.hidden_alignment_layer_weighting, "linear_depth")

    def test_distill_hidden_loss_rejects_negative_weight(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--distill_hidden_loss_weight", "default=-0.01"])

    def test_distill_pre_mlp_hidden_loss_rejects_negative_weight(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--distill_pre_mlp_hidden_loss_weight", "default=-0.01"])

    def test_distill_hidden_loss_rejects_unknown_weighting(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--distill_hidden_alignment_layer_weighting", "quadratic"])


class CatDistillHiddenLossTest(unittest.TestCase):
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

        loss = compute_distill_hidden_alignment_loss(
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

        loss = compute_distill_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=torch.tensor([[1, 0]]),
            layer_weighting="uniform",
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_linear_depth_weights_increase_and_average_to_one(self):
        weights = build_distill_hidden_layer_weights(
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

        loss = compute_distill_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=None,
            layer_weighting="linear_depth",
        )

        expected_weights = build_distill_hidden_layer_weights(
            num_layers=2,
            layer_weighting="linear_depth",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(loss, expected_weights[1] / 2.0))

    def test_pre_mlp_hidden_loss_uses_captured_layer_inputs(self):
        teacher = (
            torch.ones(1, 2, 1),
            torch.full((1, 2, 1), 2.0),
        )
        student = (
            torch.full((1, 2, 1), 2.0),
            torch.full((1, 2, 1), 4.0),
        )

        loss = compute_distill_pre_mlp_hidden_alignment_loss(
            teacher_pre_mlp_hiddens=teacher,
            student_pre_mlp_hiddens=student,
            attention_mask=torch.ones(1, 2),
            layer_weighting="uniform",
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_pre_mlp_hidden_loss_attention_mask_excludes_padding_tokens(self):
        teacher = (torch.tensor([[[1.0], [10.0]]]),)
        student = (torch.tensor([[[2.0], [100.0]]]),)

        loss = compute_distill_pre_mlp_hidden_alignment_loss(
            teacher_pre_mlp_hiddens=teacher,
            student_pre_mlp_hiddens=student,
            attention_mask=torch.tensor([[1, 0]]),
            layer_weighting="uniform",
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_pre_mlp_hidden_loss_uses_linear_depth_weights(self):
        teacher = (
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
        )
        student = (
            torch.ones(1, 1, 1),
            torch.full((1, 1, 1), 2.0),
        )

        loss = compute_distill_pre_mlp_hidden_alignment_loss(
            teacher_pre_mlp_hiddens=teacher,
            student_pre_mlp_hiddens=student,
            attention_mask=None,
            layer_weighting="linear_depth",
        )

        expected_weights = build_distill_hidden_layer_weights(
            num_layers=2,
            layer_weighting="linear_depth",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(loss, expected_weights[1] / 2.0))


class CatDistillTrainerSelectionTest(unittest.TestCase):
    def test_pre_mlp_hidden_loss_uses_custom_trainer_for_sft_loss(self):
        captured_kwargs = {}

        class FakeCustomTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        class FakeSFTTrainer:
            def __init__(self, **_kwargs):
                raise AssertionError("SFTTrainer should not be selected when pre-MLP hidden loss is enabled.")

        cfg = SimpleNamespace(
            loss_type="sft",
            temperature=1.0,
            loss_alpha=0.5,
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.01,
            hidden_alignment_layer_weighting="uniform",
        )
        training_args = SimpleNamespace(distill_model_max_length=128)

        with patch.object(lora_utils, "CustomSFTTrainer", FakeCustomTrainer), patch.object(
            lora_utils, "SFTTrainer", FakeSFTTrainer
        ):
            trainer = lora_utils._build_lora_trainer(
                model=object(),
                train_ds=[],
                eval_ds=None,
                sft_args=object(),
                training_args=training_args,
                logger=SimpleNamespace(info=lambda *args, **kwargs: None),
                lora_config=None,
                cfg=cfg,
                hif4_act_controller=None,
                teacher_param_snapshots=[],
            )

        self.assertIsInstance(trainer, FakeCustomTrainer)
        self.assertEqual(captured_kwargs["loss_type"], "sft")
        self.assertEqual(captured_kwargs["hidden_loss_weight"], 0.0)
        self.assertEqual(captured_kwargs["pre_mlp_hidden_loss_weight"], 0.01)
        self.assertEqual(captured_kwargs["hidden_alignment_layer_weighting"], "uniform")


if __name__ == "__main__":
    unittest.main()
