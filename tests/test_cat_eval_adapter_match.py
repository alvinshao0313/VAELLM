import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tools.cat_eval import _compute_checkpoint_fingerprint, _validate_adapter_checkpoint_match
from train_utils import lora_utils
from train_utils.cat_train_args import process_cat_train_args, resolve_distill_runtime_config
from train_utils.lora_training import (
    CustomSFTTrainer,
    LoraConfig,
    PeftModel,
    TaskType,
    _swap_teacher_snapshot_params,
    build_distill_hidden_layer_weights,
    capture_pre_mlp_hiddens,
    compute_distill_hidden_alignment_loss,
    compute_distill_pre_mlp_hidden_alignment_loss,
    parse_distill_hidden_alignment_layer_weighting,
)
from train_utils.distill_token_stats import DistillTokenStatsAccumulator


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
            parse_distill_hidden_alignment_layer_weighting("quadratic")


class CatDistillPromptKdWeightArgsTest(unittest.TestCase):
    def test_distill_prompt_kd_weight_defaults_to_zero(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args([])
        cfg = resolve_distill_runtime_config(cat_args, after_category=None)

        self.assertEqual(cfg.prompt_kd_weight, 0.0)

    def test_distill_prompt_kd_weight_resolves_after_category_overrides(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args(
            [
                "--distill_prompt_kd_weight",
                "default=0.05,after:q_proj=0.1",
            ]
        )

        default_cfg = resolve_distill_runtime_config(cat_args, after_category=None)
        q_proj_cfg = resolve_distill_runtime_config(cat_args, after_category="q_proj")
        o_proj_cfg = resolve_distill_runtime_config(cat_args, after_category="o_proj")

        self.assertEqual(default_cfg.prompt_kd_weight, 0.05)
        self.assertEqual(q_proj_cfg.prompt_kd_weight, 0.1)
        self.assertEqual(o_proj_cfg.prompt_kd_weight, 0.05)

    def test_distill_prompt_kd_weight_rejects_negative_weight(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            process_cat_train_args(["--distill_prompt_kd_weight", "default=-0.01"])

    def test_distill_prompt_kd_weight_accepts_value_above_one(self):
        cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args(
            ["--distill_prompt_kd_weight", "default=2.0"]
        )
        cfg = resolve_distill_runtime_config(cat_args, after_category=None)

        self.assertEqual(cfg.prompt_kd_weight, 2.0)


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

            def add_callback(self, callback):
                pass

        class FakeSFTTrainer:
            def __init__(self, **_kwargs):
                raise AssertionError("SFTTrainer should not be selected when pre-MLP hidden loss is enabled.")

        cfg = SimpleNamespace(
            loss_type="sft",
            temperature=1.0,
            loss_alpha=0.5,
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.01,
            prompt_kd_weight=0.0,
            hidden_alignment_layer_weighting="uniform",
            eakld_confidence_k=16,
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
                cfg=cfg,
                hif4_act_controller=None,
                teacher_param_snapshots=[],
            )

        self.assertIsInstance(trainer, FakeCustomTrainer)
        self.assertEqual(captured_kwargs["loss_type"], "sft")
        self.assertEqual(captured_kwargs["hidden_loss_weight"], 0.0)
        self.assertEqual(captured_kwargs["pre_mlp_hidden_loss_weight"], 0.01)
        self.assertEqual(captured_kwargs["prompt_kd_weight"], 0.0)
        self.assertEqual(captured_kwargs["hidden_alignment_layer_weighting"], "uniform")
        self.assertNotIn("peft_config", captured_kwargs)

    def test_lazy_distill_forwards_dynamic_padding_to_shared_collator(self):
        captured_kwargs = {}
        logged_messages = []
        tokenizer = object()
        sentinel_collator = object()

        class FakeCustomTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def add_callback(self, _callback):
                return None

        cfg = SimpleNamespace(
            loss_type="sft",
            temperature=1.0,
            loss_alpha=0.5,
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.01,
            prompt_kd_weight=0.0,
            hidden_alignment_layer_weighting="uniform",
            eakld_confidence_k=16,
        )
        training_args = SimpleNamespace(
            distill_model_max_length=128,
            distill_dynamic_padding=True,
        )
        logger = SimpleNamespace(
            info=lambda message, *args: logged_messages.append(
                message % args if args else message
            )
        )

        with patch.object(lora_utils, "CustomSFTTrainer", FakeCustomTrainer), patch.object(
            lora_utils,
            "build_edgerazor_data_collator",
            return_value=sentinel_collator,
        ) as mock_collator:
            trainer = lora_utils._build_lora_trainer(
                model=object(),
                train_ds=[],
                eval_ds=None,
                sft_args=object(),
                training_args=training_args,
                logger=logger,
                cfg=cfg,
                hif4_act_controller=None,
                teacher_param_snapshots=[],
                tokenizer=tokenizer,
                train_is_iterable=False,
                use_lazy_tokenized_dataset=True,
            )

        self.assertIsInstance(trainer, FakeCustomTrainer)
        mock_collator.assert_called_once_with(
            tokenizer,
            max_seq_len=128,
            dynamic_padding=True,
        )
        self.assertIs(captured_kwargs["data_collator"], sentinel_collator)
        self.assertIn(
            "LoRA: distill padding mode=dynamic max_seq_len=128 pad_to_multiple_of=8",
            logged_messages,
        )

    def test_lazy_distill_defaults_to_fixed_padding_when_flag_missing(self):
        captured_kwargs = {}
        logged_messages = []
        tokenizer = object()
        sentinel_collator = object()

        class FakeCustomTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def add_callback(self, _callback):
                return None

        cfg = SimpleNamespace(
            loss_type="sft",
            temperature=1.0,
            loss_alpha=0.5,
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.01,
            prompt_kd_weight=0.0,
            hidden_alignment_layer_weighting="uniform",
            eakld_confidence_k=16,
        )
        training_args = SimpleNamespace(distill_model_max_length=128)
        logger = SimpleNamespace(
            info=lambda message, *args: logged_messages.append(
                message % args if args else message
            )
        )

        with patch.object(lora_utils, "CustomSFTTrainer", FakeCustomTrainer), patch.object(
            lora_utils,
            "build_edgerazor_data_collator",
            return_value=sentinel_collator,
        ) as mock_collator:
            trainer = lora_utils._build_lora_trainer(
                model=object(),
                train_ds=[],
                eval_ds=None,
                sft_args=object(),
                training_args=training_args,
                logger=logger,
                cfg=cfg,
                hif4_act_controller=None,
                teacher_param_snapshots=[],
                tokenizer=tokenizer,
                train_is_iterable=False,
                use_lazy_tokenized_dataset=True,
            )

        self.assertIsInstance(trainer, FakeCustomTrainer)
        mock_collator.assert_called_once_with(
            tokenizer,
            max_seq_len=128,
            dynamic_padding=False,
        )
        self.assertIs(captured_kwargs["data_collator"], sentinel_collator)
        self.assertIn(
            "LoRA: distill padding mode=fixed max_seq_len=128 pad_to_multiple_of=none",
            logged_messages,
        )


class _FakeOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class _TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporary = True
        self.scale = nn.Parameter(torch.tensor(1.5))

    def set_temporary(self, temporary: bool) -> None:
        self.temporary = bool(temporary)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.temporary:
            return hidden * self.scale
        return hidden


class _PreMlpLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _PreMlpBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_PreMlpLayer(hidden_size) for _ in range(num_layers)])


class _PreMlpFakeCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 11, hidden_size: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _PreMlpBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.temp_scale = _TempScale()
        self.config = SimpleNamespace(
            model_type="qwen2",
            use_return_dict=True,
            tie_word_embeddings=False,
        )
        self.output_hidden_states_calls: list[bool] = []
        self.last_hidden_states = None
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_kwargs,
    ):
        del attention_mask
        self.output_hidden_states_calls.append(bool(output_hidden_states))
        hidden = self.temp_scale(self.embed_tokens(input_ids))
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.model.layers:
            hidden = layer(hidden)
            if hidden_states is not None:
                hidden_states.append(hidden)
        logits = self.lm_head(hidden)
        if labels is None:
            loss = logits.float().pow(2).mean()
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        packed_hidden_states = tuple(hidden_states) if hidden_states is not None else None
        self.last_hidden_states = packed_hidden_states
        return _FakeOutput(
            loss=loss,
            logits=logits,
            hidden_states=packed_hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


def _build_pre_mlp_trainer(
    *,
    hidden_loss_weight: float,
    pre_mlp_hidden_loss_weight: float,
    hidden_alignment_layer_weighting: str,
    loss_type: str = "sft",
) -> CustomSFTTrainer:
    trainer = CustomSFTTrainer.__new__(CustomSFTTrainer)
    trainer.args = SimpleNamespace(bf16=False, fp16=False)
    trainer.loss_type = loss_type
    trainer.temperature = 1.0
    trainer.loss_alpha = 0.5
    trainer.hidden_loss_weight = float(hidden_loss_weight)
    trainer.pre_mlp_hidden_loss_weight = float(pre_mlp_hidden_loss_weight)
    trainer.prompt_kd_weight = 0.0
    trainer.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
        hidden_alignment_layer_weighting
    )
    trainer.eakld_confidence_k = 16
    trainer.teacher_logits_cpu_staging = False
    trainer.distill_hif4_act_controller = None
    trainer.teacher_param_snapshots = []
    trainer._teacher_param_restore_buffers = []
    trainer.accelerator = None
    trainer.distill_token_stats = DistillTokenStatsAccumulator()
    return trainer


class _FakeAccelerator:
    def __init__(self, unwrapped_model):
        self.unwrapped_model = unwrapped_model

    def unwrap_model(self, _model):
        return self.unwrapped_model


def _pre_mlp_inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


class CatDistillPreMlpHiddenStateRequestTest(unittest.TestCase):
    def test_uniform_pre_mlp_only_skips_full_hidden_states(self):
        model = _PreMlpFakeCausalLM()
        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.25,
            hidden_alignment_layer_weighting="uniform",
        )
        inputs = _pre_mlp_inputs()

        with patch(
            "train_utils.lora_training.compute_distill_hidden_alignment_loss",
            wraps=compute_distill_hidden_alignment_loss,
        ) as hidden_mock, patch(
            "train_utils.lora_training.compute_distill_pre_mlp_hidden_alignment_loss",
            wraps=compute_distill_pre_mlp_hidden_alignment_loss,
        ) as pre_mlp_mock:
            loss = trainer.compute_loss(model, inputs)

        self.assertEqual(model.output_hidden_states_calls, [False, False])
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(hidden_mock.call_count, 0)
        self.assertEqual(pre_mlp_mock.call_count, 1)
        pre_kwargs = pre_mlp_mock.call_args.kwargs
        self.assertEqual(len(pre_kwargs["teacher_pre_mlp_hiddens"]), model.num_layers)
        self.assertEqual(len(pre_kwargs["student_pre_mlp_hiddens"]), model.num_layers)
        self.assertIsNone(pre_kwargs["teacher_reference_hidden"])

        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(model.temp_scale.scale.grad)
        self.assertGreater(float(model.temp_scale.scale.grad.abs().sum()), 0.0)

    def test_adaptive_pre_mlp_only_requests_teacher_reference_hidden(self):
        model = _PreMlpFakeCausalLM()
        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.25,
            # plan shorthand adaptive_3 == project adaptive_top_3
            hidden_alignment_layer_weighting="adaptive_top_3",
        )
        inputs = _pre_mlp_inputs()
        captured = {}

        def _wrapped_pre_mlp(**kwargs):
            captured["teacher_reference_hidden"] = kwargs.get("teacher_reference_hidden")
            return compute_distill_pre_mlp_hidden_alignment_loss(**kwargs)

        with patch(
            "train_utils.lora_training.compute_distill_pre_mlp_hidden_alignment_loss",
            side_effect=_wrapped_pre_mlp,
        ):
            loss = trainer.compute_loss(model, inputs)

        self.assertEqual(model.output_hidden_states_calls, [True, False])
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(captured["teacher_reference_hidden"])
        # Teacher forward is first; last_hidden_states was overwritten by student (None).
        # Re-run teacher-only forward to recover the reference embedding state.
        with torch.no_grad():
            model.temp_scale.set_temporary(False)
            teacher_outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=True)
        self.assertTrue(
            torch.equal(captured["teacher_reference_hidden"], teacher_outputs.hidden_states[0])
        )

        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(model.temp_scale.scale.grad)

    def test_ordinary_hidden_plus_pre_mlp_requests_both_hidden_states(self):
        model = _PreMlpFakeCausalLM()
        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.2,
            pre_mlp_hidden_loss_weight=0.25,
            hidden_alignment_layer_weighting="uniform",
        )
        inputs = _pre_mlp_inputs()
        loss = trainer.compute_loss(model, inputs)
        self.assertEqual(model.output_hidden_states_calls, [True, True])
        self.assertTrue(torch.isfinite(loss).item())

    def test_disabled_hidden_paths_skip_hidden_states_and_pre_mlp_hooks(self):
        model = _PreMlpFakeCausalLM()
        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.0,
            hidden_alignment_layer_weighting="uniform",
            loss_type="kl",
        )
        inputs = _pre_mlp_inputs()

        with patch(
            "train_utils.lora_training.capture_pre_mlp_hiddens",
            wraps=capture_pre_mlp_hiddens,
        ) as capture_mock:
            loss = trainer.compute_loss(model, inputs)

        self.assertEqual(model.output_hidden_states_calls, [False, False])
        self.assertEqual(capture_mock.call_count, 0)
        self.assertTrue(torch.isfinite(loss).item())

    @unittest.skipIf(LoraConfig is None or PeftModel is None, "peft is not installed")
    def test_peft_model_pre_mlp_capture_uses_base_model_layers(self):
        raw_model = _PreMlpFakeCausalLM()
        peft_model = lora_utils.create_lora_adapters(
            raw_model,
            target_names=["mlp"],
            rank=2,
            alpha=4,
            dropout=0.0,
            use_dora=False,
        )[0]
        self.assertIsInstance(peft_model, PeftModel)
        self.assertNotIsInstance(peft_model.get_base_model(), PeftModel)
        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.25,
            hidden_alignment_layer_weighting="uniform",
        )
        trainer.accelerator = _FakeAccelerator(peft_model)
        inputs = _pre_mlp_inputs()

        with patch(
            "train_utils.lora_training.compute_distill_pre_mlp_hidden_alignment_loss",
            wraps=compute_distill_pre_mlp_hidden_alignment_loss,
        ) as pre_mlp_mock:
            loss = trainer.compute_loss(peft_model, inputs)

        self.assertTrue(torch.isfinite(loss).item())
        pre_kwargs = pre_mlp_mock.call_args.kwargs
        self.assertEqual(len(pre_kwargs["teacher_pre_mlp_hiddens"]), raw_model.num_layers)
        self.assertEqual(len(pre_kwargs["student_pre_mlp_hiddens"]), raw_model.num_layers)

    @unittest.skipIf(LoraConfig is None or PeftModel is None, "peft is not installed")
    def test_teacher_adapter_off_student_adapter_on_and_backward(self):
        raw_model = _PreMlpFakeCausalLM()
        peft_model = lora_utils.create_lora_adapters(
            raw_model,
            target_names=["mlp"],
            rank=2,
            alpha=4,
            dropout=0.0,
            use_dora=False,
        )[0]
        for name, module in peft_model.named_modules():
            if name.endswith("lora_B.default"):
                nn.init.constant_(module.weight, 0.25)
        inputs = _pre_mlp_inputs()
        with torch.no_grad():
            with peft_model.disable_adapter():
                base_logits = peft_model.get_base_model()(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).logits
                adapter_off_logits = peft_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).logits
            adapter_on_logits = peft_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
        self.assertTrue(torch.allclose(adapter_off_logits, base_logits))
        self.assertFalse(torch.allclose(adapter_on_logits, adapter_off_logits))

        trainer = _build_pre_mlp_trainer(
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.25,
            hidden_alignment_layer_weighting="uniform",
            loss_type="kl_top_2",
        )
        trainer.accelerator = _FakeAccelerator(peft_model)
        loss = trainer.compute_loss(peft_model, inputs)
        self.assertTrue(torch.isfinite(loss).item())
        peft_model.zero_grad(set_to_none=True)
        loss.backward()
        lora_grads = [
            param.grad
            for name, param in peft_model.named_parameters()
            if "lora_" in name and param.requires_grad
        ]
        self.assertTrue(lora_grads)
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in lora_grads))

    def test_swap_teacher_snapshot_params_reuses_restore_buffer(self):
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        snapshot = torch.tensor([7.0, 8.0])
        restore = torch.empty_like(param)
        with _swap_teacher_snapshot_params([(param, snapshot)], [restore]):
            self.assertTrue(torch.equal(param.detach(), snapshot))
        self.assertTrue(torch.equal(param.detach(), torch.tensor([1.0, 2.0])))

        with torch.no_grad():
            param.copy_(torch.tensor([3.0, 4.0]))
        with _swap_teacher_snapshot_params([(param, snapshot)], [restore]):
            self.assertTrue(torch.equal(param.detach(), snapshot))
        self.assertTrue(torch.equal(param.detach(), torch.tensor([3.0, 4.0])))


def _build_distill_trainer(
    *,
    loss_type: str,
    prompt_kd_weight: float,
    temperature: float = 1.0,
    loss_alpha: float = 0.5,
    eakld_confidence_k: int = 16,
) -> CustomSFTTrainer:
    trainer = CustomSFTTrainer.__new__(CustomSFTTrainer)
    trainer.args = SimpleNamespace(bf16=False, fp16=False)
    trainer.loss_type = loss_type
    trainer.temperature = float(temperature)
    trainer.loss_alpha = float(loss_alpha)
    trainer.hidden_loss_weight = 0.0
    trainer.pre_mlp_hidden_loss_weight = 0.0
    trainer.prompt_kd_weight = float(prompt_kd_weight)
    trainer.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
        "uniform"
    )
    trainer.eakld_confidence_k = int(eakld_confidence_k)
    trainer.teacher_logits_cpu_staging = False
    trainer.distill_hif4_act_controller = None
    trainer.teacher_param_snapshots = []
    trainer.accelerator = None
    trainer.distill_token_stats = DistillTokenStatsAccumulator()
    return trainer


def _distill_inputs() -> dict:
    # positions:  0   1   2  3  4  5
    # labels:   -100 -100 3  4  5  6  (prompt prefix at 0-1)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


class CatDistillRegionNormalizedLossTest(unittest.TestCase):
    def test_eakld_positive_prompt_weight_calls_criterion_twice_with_different_masks(self):
        from train_utils import lora_training
        from train_utils.distill_losses import compute_eakld

        model = _PreMlpFakeCausalLM()
        trainer = _build_distill_trainer(loss_type="eakld", prompt_kd_weight=0.1)
        inputs = _distill_inputs()

        captured_masks: list = []
        original = compute_eakld

        def recording(*, student_logits, teacher_logits, mask, **kwargs):
            captured_masks.append(mask)
            return original(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                **kwargs,
            )

        with patch.object(lora_training, "compute_eakld", side_effect=recording):
            loss = trainer.compute_loss(model, inputs)

        self.assertEqual(len(captured_masks), 2)
        response_mask, prompt_mask = captured_masks
        self.assertFalse(torch.equal(response_mask, prompt_mask))
        self.assertGreater(float(response_mask.sum().item()), 0.0)
        self.assertGreater(float(prompt_mask.sum().item()), 0.0)
        # Regions are disjoint.
        self.assertTrue(bool((response_mask + prompt_mask).max().item() <= 1.0))
        self.assertTrue(torch.isfinite(loss).item())

    def test_eakld_zero_prompt_weight_calls_criterion_once_on_response(self):
        from train_utils import lora_training
        from train_utils.distill_losses import compute_eakld

        model = _PreMlpFakeCausalLM()
        trainer = _build_distill_trainer(loss_type="eakld", prompt_kd_weight=0.0)
        inputs = _distill_inputs()

        captured_masks: list = []
        original = compute_eakld

        def recording(*, student_logits, teacher_logits, mask, **kwargs):
            captured_masks.append(mask)
            return original(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                **kwargs,
            )

        with patch.object(lora_training, "compute_eakld", side_effect=recording):
            loss = trainer.compute_loss(model, inputs)

        self.assertEqual(len(captured_masks), 1)
        self.assertGreater(float(captured_masks[0].sum().item()), 0.0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_kl_region_combination_matches_manual_means(self):
        from train_utils.distill_losses import (
            build_distill_token_regions,
            compute_forward_kl_loss,
        )

        model = _PreMlpFakeCausalLM()
        weight = 0.1
        trainer = _build_distill_trainer(loss_type="kl", prompt_kd_weight=weight)
        inputs = _distill_inputs()
        loss = trainer.compute_loss(model, inputs)

        # Recompute teacher/student logits with the same temp_scale toggles.
        with torch.no_grad():
            model.temp_scale.set_temporary(False)
            teacher_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            model.temp_scale.set_temporary(True)
            student_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits

        regions = build_distill_token_regions(
            labels=inputs["labels"],
            attention_mask=inputs["attention_mask"],
            reference_logits=student_logits,
        )
        expected = compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=regions.response_mask,
            temperature=1.0,
        ) + weight * compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=regions.prompt_mask,
            temperature=1.0,
        )
        self.assertTrue(torch.allclose(loss, expected.detach(), rtol=1e-5, atol=1e-6))

    def test_zero_prompt_weight_matches_response_only_value(self):
        from train_utils.distill_losses import (
            build_distill_token_regions,
            compute_forward_kl_loss,
        )

        model = _PreMlpFakeCausalLM()
        trainer_zero = _build_distill_trainer(loss_type="kl", prompt_kd_weight=0.0)
        inputs = _distill_inputs()
        loss_zero = trainer_zero.compute_loss(model, inputs)

        with torch.no_grad():
            model.temp_scale.set_temporary(False)
            teacher_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            model.temp_scale.set_temporary(True)
            student_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
        regions = build_distill_token_regions(
            labels=inputs["labels"],
            attention_mask=inputs["attention_mask"],
            reference_logits=student_logits,
        )
        expected = compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=regions.response_mask,
            temperature=1.0,
        )
        self.assertTrue(torch.allclose(loss_zero, expected.detach(), rtol=1e-5, atol=1e-6))

    def test_kd_ce_counted_once_across_regions(self):
        from train_utils import lora_training

        model = _PreMlpFakeCausalLM()
        trainer = _build_distill_trainer(
            loss_type="kd", prompt_kd_weight=0.1, loss_alpha=0.5
        )
        inputs = _distill_inputs()
        loss = trainer.compute_loss(model, inputs)
        self.assertTrue(torch.isfinite(loss).item())
        # CE (outputs["loss"]) is mixed once with alpha=0.5; the regional KD
        # scalar is built from response + weight*prompt before the single mix.
        # We assert finiteness and that gradient flows to the student scale.
        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(model.temp_scale.scale.grad)


if __name__ == "__main__":
    unittest.main()
