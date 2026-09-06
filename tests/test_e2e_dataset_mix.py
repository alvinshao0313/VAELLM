import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import IterableDataset
from transformers.trainer_utils import IntervalStrategy

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.trainer import (
    build_vae_hidden_layer_weights,
    compute_vae_hidden_alignment_loss,
)
from e2e_common.data import DatasetMixSourcePreset, _record_to_text, build_datasets
from e2e_common import lazy_datasets as lazy_module
from e2e_common.lazy_datasets import (
    _LazyPresetIterableDataset,
    _iter_raw_records_for_worker,
    build_mixed_lazy_dataset,
    encode_text_lm_record,
    is_iterable_training_dataset,
)
from train_utils.lora_data import build_calibration_input_ids, prepare_distill_datasets


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, list):
            input_ids = [self._encode(text) for text in texts]
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(ids) for ids in input_ids],
            }
        ids = self._encode(texts)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    @staticmethod
    def _encode(text):
        tokens = str(text).split()
        token_count = max(1, len(tokens))
        return list(range(token_count))


class ContentTokenizer(DummyTokenizer):
    # Non-empty sentinel so render_messages allows apply_chat_template fallback path.
    chat_template = "content-join"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **_kwargs,
    ):
        parts = []
        ids = []
        mask = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            role_text = f"<|{role}|>"
            parts.append(f"{role_text} {content}")
            role_ids = self._encode(role_text)
            content_ids = self._encode(content)
            ids.extend(role_ids)
            ids.extend(content_ids)
            mask.extend([False] * len(role_ids))
            mask.extend([role == "assistant"] * len(content_ids))
        if add_generation_prompt:
            parts.append("<|assistant|>")
            prompt_ids = self._encode("<|assistant|>")
            ids.extend(prompt_ids)
            mask.extend([False] * len(prompt_ids))
        text = "\n".join(parts)
        if not tokenize:
            return text
        if return_assistant_tokens_mask:
            # Unsupported in this stub: force distill_data prefix-boundary fallback.
            raise TypeError("assistant mask unsupported")
        if return_dict:
            return {"input_ids": ids}
        return ids

    @staticmethod
    def _encode(text):
        tokens = str(text).split()
        if not tokens:
            return [0]
        return [sum(ord(ch) for ch in token) % 1000 for token in tokens]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._encode(text)


def _make_text_dataset(prefix: str, count: int, *, words: int = 8):
    text = " ".join(f"{prefix}_{idx}" for idx in range(words))
    rows = [text for _ in range(count)]
    return Dataset.from_dict({"text": rows})


def _make_openorca_dataset(count: int, *, variable_lengths: bool = False):
    questions = []
    responses = []
    for idx in range(count):
        question_words = (idx % 7) + 1 if variable_lengths else 3
        response_words = ((idx * 3) % 11) + 1 if variable_lengths else 3
        questions.append(" ".join(f"q{idx}_{word_idx}" for word_idx in range(question_words)))
        responses.append(" ".join(f"a{idx}_{word_idx}" for word_idx in range(response_words)))
    return Dataset.from_dict(
        {
            "question": questions,
            "response": responses,
            "system_prompt": ["sys"] * count,
        }
    )


def _make_alpaca_dataset(count: int, *, variable_lengths: bool = False):
    instructions = []
    inputs = []
    outputs = []
    for idx in range(count):
        input_words = (idx % 5) + 1 if variable_lengths else 3
        output_words = ((idx * 2) % 7) + 1 if variable_lengths else 3
        instructions.append(f"inst{idx}")
        inputs.append(" ".join(f"in{idx}_{word_idx}" for word_idx in range(input_words)))
        outputs.append(" ".join(f"out{idx}_{word_idx}" for word_idx in range(output_words)))
    return Dataset.from_dict(
        {
            "instruction": instructions,
            "input": inputs,
            "output": outputs,
        }
    )


def _tensorish_to_tuple(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


def _dataset_signature(dataset, *, limit: int = 8):
    if is_iterable_training_dataset(dataset):
        return [
            _tensorish_to_tuple(row["input_ids"])
            for _idx, row in zip(range(limit), dataset)
        ]
    return [_tensorish_to_tuple(dataset[idx]["input_ids"]) for idx in range(min(len(dataset), limit))]


class DatasetMixArgsTest(unittest.TestCase):
    def _checkpoint_dir(self):
        tmp = tempfile.TemporaryDirectory()
        with open(f"{tmp.name}/checkpoint_meta.json", "w", encoding="utf-8") as handle:
            handle.write("{}")
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_parse_args_normalizes_dataset_mix(self):
        e2e_args, _hf_args, training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=3,fineweb_edu=1",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.data.dataset_mix_sources, ("openorca", "fineweb_edu"))
        self.assertEqual(len(e2e_args.data.dataset_mix_weights), 2)
        self.assertAlmostEqual(sum(e2e_args.data.dataset_mix_weights), 1.0)
        self.assertEqual(e2e_args.data.dataset_mix, "openorca=0.75,fineweb_edu=0.25")
        self.assertEqual(training_args.max_steps, 10)

    def test_parse_args_decoder_lr_defaults_to_none_and_accepts_override(self):
        default_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--steps",
                "10",
            ]
        )
        self.assertIsNone(default_args.opt.decoder_lr)

        explicit_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--decoder_lr",
                "2e-4",
                "--steps",
                "10",
            ]
        )
        self.assertAlmostEqual(explicit_args.opt.decoder_lr, 2e-4)

    def test_parse_args_rejects_duplicate_alias(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_checkpoint_dir",
                    self._checkpoint_dir(),
                    "--dataset_mix",
                    "openorca=1,openorca=1",
                    "--steps",
                    "10",
                ]
            )

    def test_parse_args_accepts_shared_text_field_with_dataset_mix(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--text_field",
                "text",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.data.text_field, "text")

    def test_parse_args_accepts_long_dataset_aliases(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "longalpaca=2,longalign=1",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.data.dataset_mix_sources, ("longalpaca", "longalign"))
        self.assertEqual(e2e_args.data.dataset_mix, "longalpaca=0.666666666667,longalign=0.333333333333")

    def test_parse_args_accepts_parallel_mode_dp(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--parallel_mode",
                "dp",
                "--offload_mode",
                "none",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.runtime.parallel_mode, "dp")

    def test_parse_args_accepts_pre_mlp_and_teacher_model_offload(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--pre_mlp_hidden_loss_weight",
                "0.01",
                "--teacher_output_offload",
                "cpu",
                "--teacher_model_offload",
                "cpu",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.loss.pre_mlp_hidden_loss_weight, 0.01)
        self.assertEqual(e2e_args.runtime.teacher_output_offload, "cpu")
        self.assertEqual(e2e_args.runtime.teacher_model_offload, "cpu")

    def test_parse_args_rejects_negative_pre_mlp_hidden_loss_weight(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_checkpoint_dir",
                    self._checkpoint_dir(),
                    "--dataset_mix",
                    "openorca=1",
                    "--pre_mlp_hidden_loss_weight",
                    "-0.01",
                    "--steps",
                    "10",
                ]
            )

    def test_parse_args_rejects_teacher_model_offload_without_cpu_targets(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_checkpoint_dir",
                    self._checkpoint_dir(),
                    "--dataset_mix",
                    "openorca=1",
                    "--teacher_model_offload",
                    "cpu",
                    "--steps",
                    "10",
                ]
            )

    def test_parse_args_rejects_layer_mp_under_torchrun(self):
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "4"}, clear=False):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--student_checkpoint_dir",
                        self._checkpoint_dir(),
                        "--dataset_mix",
                        "openorca=1",
                        "--parallel_mode",
                        "layer_mp",
                        "--steps",
                        "10",
                    ]
                )

    def test_parse_args_rejects_dp_with_streaming_offload(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_checkpoint_dir",
                    self._checkpoint_dir(),
                    "--dataset_mix",
                    "openorca=1",
                    "--parallel_mode",
                    "dp",
                    "--offload_mode",
                    "streaming",
                    "--steps",
                    "10",
                ]
            )


    def test_parse_args_eval_before_save_requires_tasks_and_save_steps(self):
        with self.assertRaises((SystemExit, ValueError)):
            parse_args(
                [
                    "--student_checkpoint_dir",
                    self._checkpoint_dir(),
                    "--dataset_mix",
                    "openorca=1",
                    "--eval_before_save",
                    "true",
                    "--save_strategy",
                    "steps",
                    "--save_steps",
                    "100",
                    "--steps",
                    "10",
                ]
            )
        e2e_args, _hf_args, training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--eval_tasks",
                "boolq,rte",
                "--save_strategy",
                "steps",
                "--save_steps",
                "100",
                "--steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.runtime.evaluation.eval_tasks, "boolq,rte")
        self.assertEqual(training_args.save_steps, 100)

    def test_parse_args_rejects_removed_dataset_cli_flags(self):
        for flag, value in (
            ("--dataset_num_proc", "2"),
            ("--eval_file", "dummy_eval.txt"),
            ("--max_eval_samples", "8"),
        ):
            with self.subTest(flag=flag):
                with self.assertRaises((SystemExit, ValueError)):
                    parse_args(
                        [
                            "--student_checkpoint_dir",
                            self._checkpoint_dir(),
                            "--dataset_mix",
                            "openorca=1",
                            flag,
                            value,
                            "--steps",
                            "10",
                        ]
                    )

    def test_parse_args_dynamic_padding_defaults_true(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--steps",
                "1",
            ]
        )
        self.assertTrue(e2e_args.data.dynamic_padding)

    def test_parse_args_accepts_dynamic_padding_true(self):
        e2e_args, _hf_args, _training_args = parse_args(
            [
                "--student_checkpoint_dir",
                self._checkpoint_dir(),
                "--dataset_mix",
                "openorca=1",
                "--dynamic_padding",
                "true",
                "--steps",
                "1",
            ]
        )
        self.assertTrue(e2e_args.data.dynamic_padding)


class VAEE2EPromptLossWeightArgsTest(unittest.TestCase):
    def _parse_with_checkpoint(self, extra_args, *, dataset_mix="openorca=1.0"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/checkpoint_meta.json", "w", encoding="utf-8") as handle:
                handle.write("{}")
            args, _hf_args, _training_args = parse_args(
                [
                    "--student_checkpoint_dir",
                    tmpdir,
                    "--dataset_mix",
                    dataset_mix,
                    "--steps",
                    "1",
                    *extra_args,
                ]
            )
        return args

    def test_prompt_loss_weight_defaults_to_zero(self):
        args = self._parse_with_checkpoint([])
        self.assertEqual(args.loss.prompt_loss_weight, 0.0)

    def test_prompt_loss_weight_accepts_fractional_value(self):
        args = self._parse_with_checkpoint(["--prompt_loss_weight", "0.05"])
        self.assertEqual(args.loss.prompt_loss_weight, 0.05)

    def test_prompt_loss_weight_accepts_value_above_one(self):
        args = self._parse_with_checkpoint(["--prompt_loss_weight", "2.0"])
        self.assertEqual(args.loss.prompt_loss_weight, 2.0)

    def test_prompt_loss_weight_rejects_negative_weight(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--prompt_loss_weight", "-0.01"])

    def test_legacy_prompt_kd_weight_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--prompt_kd_weight", "0.05"])

    def test_mcqa_dataset_task_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(
                ["--dataset_task", "mcqa"],
                dataset_mix="race=1.0",
            )

    def test_choice_kd_loss_types_are_rejected(self):
        for loss_type in ("choice_kd", "choice_kd_ce"):
            with self.subTest(loss_type=loss_type):
                with self.assertRaises(SystemExit):
                    self._parse_with_checkpoint(
                        ["--dataset_task", "sft", "--loss_type", loss_type],
                        dataset_mix="race=1.0",
                    )


class VAEE2EModelLevelLoraArgsTest(unittest.TestCase):
    def _parse_with_checkpoint(self, extra_args):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/checkpoint_meta.json", "w", encoding="utf-8") as handle:
                handle.write("{}")
            args, _hf_args, _training_args = parse_args(
                [
                    "--student_checkpoint_dir",
                    tmpdir,
                    "--dataset_mix",
                    "openorca=1.0",
                    "--steps",
                    "1",
                    *extra_args,
                ]
            )
        return args

    def test_lora_mode_defaults_plain_full_space_config(self):
        args = self._parse_with_checkpoint(["--train_mode", "lora"])
        self.assertEqual(args.train_mode, "lora")
        self.assertEqual(args.lora.rank, 12)
        self.assertEqual(args.lora.alpha, 24.0)
        self.assertEqual(args.lora.dropout, 0.03)

    def test_lora_mode_accepts_explicit_structural_config(self):
        args = self._parse_with_checkpoint(
            [
                "--train_mode",
                "lora",
                "--lora_rank",
                "7",
                "--lora_alpha",
                "14",
                "--lora_dropout",
                "0.1",
            ]
        )
        self.assertEqual(args.lora.rank, 7)
        self.assertEqual(args.lora.alpha, 14.0)
        self.assertEqual(args.lora.dropout, 0.1)
        self.assertTrue(args.lora.rank_explicit)
        self.assertTrue(args.lora.alpha_explicit)
        self.assertTrue(args.lora.dropout_explicit)

    def test_decoder_mode_retains_valid_inactive_lora_structure_without_changing_mode(self):
        args = self._parse_with_checkpoint(
            [
                "--train_mode",
                "decoder",
                "--lora_rank",
                "12",
                "--lora_alpha",
                "24",
                "--lora_dropout",
                "0.03",
            ]
        )
        self.assertEqual(args.train_mode, "decoder")
        self.assertEqual(args.lora.rank, 12)

    def test_lora_config_rejects_invalid_structural_values(self):
        for flag, value in (
            ("--lora_rank", "0"),
            ("--lora_alpha", "0"),
            ("--lora_dropout", "1.0"),
        ):
            with self.subTest(flag=flag, value=value):
                with self.assertRaises(SystemExit):
                    self._parse_with_checkpoint(["--train_mode", "lora", flag, value])

    def test_removed_lora_and_decoder_aux_flags_are_rejected(self):
        removed = (
            ("--finetune_mode", "compressed_lora"),
            ("--compressed_lora_scope", "full"),
            ("--tune_final_norm", "true"),
            ("--use_post_norm_head_linear", "true"),
            ("--vae_tune_bias", "true"),
        )
        for flag, value in removed:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    self._parse_with_checkpoint([flag, value])


class VAEE2ETrainerPromptKdMaskHelperTest(unittest.TestCase):
    def test_private_helper_forwards_regions_without_prompt_kd_weight(self):
        from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
        from train_utils.distill_losses import DistillTokenRegions

        trainer = VAEDecoderE2ETrainer.__new__(VAEDecoderE2ETrainer)
        # prompt_kd_weight must NOT be forwarded into mask/region construction.
        trainer.prompt_kd_weight = 0.1
        labels = torch.tensor([[-100, -100, 3, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(labels)
        reference_logits = torch.zeros(1, 4, 8, dtype=torch.float32)
        inputs = {"labels": labels, "attention_mask": attention_mask}

        sentinel_response = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
        sentinel_prompt = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        sentinel_regions = DistillTokenRegions(
            response_mask=sentinel_response,
            prompt_mask=sentinel_prompt,
        )

        with mock.patch(
            "compressed_e2e_fintuning.trainer.build_distill_token_regions",
            return_value=sentinel_regions,
        ) as mocked:
            actual = trainer._build_distill_token_regions(inputs, reference_logits)

        self.assertIs(actual, sentinel_regions)
        mocked.assert_called_once_with(
            labels=labels,
            attention_mask=attention_mask,
            reference_logits=reference_logits,
        )
        # The helper must not thread prompt_kd_weight into region construction.
        forwarded_kwargs = mocked.call_args.kwargs
        self.assertNotIn("prompt_kd_weight", forwarded_kwargs)
        self.assertIs(actual.response_mask, sentinel_response)
        self.assertIs(actual.prompt_mask, sentinel_prompt)


class VAEE2EHiddenLossArgsTest(unittest.TestCase):
    def _parse_with_checkpoint(self, extra_args):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/checkpoint_meta.json", "w", encoding="utf-8") as handle:
                handle.write("{}")
            args, _hf_args, _training_args = parse_args(
                [
                    "--student_checkpoint_dir",
                    tmpdir,
                    "--dataset_mix",
                    "openorca=1.0",
                    "--steps",
                    "1",
                    *extra_args,
                ]
            )
        return args

    def test_hidden_loss_defaults_to_disabled_uniform(self):
        args = self._parse_with_checkpoint([])

        self.assertEqual(args.loss.hidden_loss_weight, 0.0)
        self.assertEqual(args.loss.hidden_layer_weighting, "uniform")

    def test_hidden_loss_accepts_linear_depth(self):
        args = self._parse_with_checkpoint(
            [
                "--hidden_loss_weight",
                "0.003",
                "--hidden_layer_weighting",
                "linear_depth",
            ]
        )

        self.assertEqual(args.loss.hidden_loss_weight, 0.003)
        self.assertEqual(args.loss.hidden_layer_weighting, "linear_depth")

    def test_hidden_loss_accepts_adaptive_top_3(self):
        args = self._parse_with_checkpoint(
            [
                "--hidden_loss_weight",
                "0.1",
                "--hidden_layer_weighting",
                "adaptive_top_3",
            ]
        )

        self.assertEqual(args.loss.hidden_loss_weight, 0.1)
        self.assertEqual(args.loss.hidden_layer_weighting, "adaptive_top_3")

    def test_hidden_loss_rejects_negative_weight(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--hidden_loss_weight", "-0.1"])

    def test_hidden_loss_rejects_unknown_weighting(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--hidden_layer_weighting", "quadratic"])

    def test_teacher_output_offload_accepts_cpu_mode(self):
        args = self._parse_with_checkpoint(
            [
                "--teacher_output_offload",
                "cpu",
                "--teacher_output_pin_memory",
                "true",
                "--teacher_output_chunk_tokens",
                "8",
            ]
        )
        self.assertEqual(args.runtime.teacher_output_offload, "cpu")
        self.assertTrue(args.runtime.teacher_output_pin_memory)
        self.assertEqual(args.runtime.teacher_output_chunk_tokens, 8)

    def test_teacher_output_offload_rejects_invalid_mode(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(
                ["--teacher_output_offload", "disk"]
            )

    def test_teacher_output_chunk_tokens_must_be_positive(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(
                ["--teacher_output_chunk_tokens", "0"]
            )


class VAEE2EHiddenLossTest(unittest.TestCase):
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

        loss = compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=torch.ones(1, 2),
            layer_weighting="uniform",
            loss_device=torch.device("cpu"),
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

        loss = compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=torch.tensor([[1, 0]]),
            layer_weighting="uniform",
            loss_device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_linear_depth_weights_increase_and_average_to_one(self):
        weights = build_vae_hidden_layer_weights(
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

        loss = compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=None,
            layer_weighting="linear_depth",
            loss_device=torch.device("cpu"),
        )

        expected_weights = build_vae_hidden_layer_weights(
            num_layers=2,
            layer_weighting="linear_depth",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(loss, expected_weights[1] / 2.0))

    def test_adaptive_top_1_selects_largest_teacher_layer_change(self):
        # Block0 ~= embedding (high cosine), block1 jumps a lot (low cosine) -> select block1.
        teacher = (
            torch.ones(1, 2, 2),
            torch.ones(1, 2, 2),
            torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        )
        student = (
            torch.ones(1, 2, 2),
            torch.full((1, 2, 2), 2.0),  # block0 relative MSE = 1
            torch.full((1, 2, 2), 3.0),  # block1 relative MSE = ((3-0)^2+(3-1)^2)/((0)^2+(1)^2) = 13
        )

        loss = compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher,
            student_hidden_states=student,
            attention_mask=None,
            layer_weighting="adaptive_top_1",
            loss_device=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(13.0)))


class DatasetMixBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ContentTokenizer()
        self.training_args = SimpleNamespace(
            model_max_length=4,
            max_steps=3,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
            world_size=1,
            seed=7,
            eval_strategy=IntervalStrategy.STEPS,
        )

    def test_record_to_text_supports_race_and_sciq(self):
        race_text = _record_to_text(
            {
                "article": "passage words here",
                "question": "which option",
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer": "B",
            },
            text_field="article",
            text_format="race_mcqa",
        )
        sciq_text = _record_to_text(
            {
                "support": "support text",
                "question": "what is the answer",
                "correct_answer": "the answer",
            },
            text_field="support",
            text_format="sciq_qa",
        )
        self.assertIn("### Passage:", race_text)
        self.assertIn("### Response:\nbeta", race_text)
        self.assertIn("### Support:", sciq_text)
        self.assertIn("### Response:\nthe answer", sciq_text)

    def test_record_to_text_supports_longalign_chat(self):
        longalign_text = _record_to_text(
            {
                "messages": [
                    {"role": "system", "content": "follow instructions carefully"},
                    {"role": "user", "content": "summarize this passage"},
                    {"role": "assistant", "content": "here is the summary"},
                ]
            },
            text_field="messages",
            text_format="longalign_chat",
        )
        invalid_text = _record_to_text(
            {
                "messages": [
                    {"role": "user", "content": "question only"},
                ]
            },
            text_field="messages",
            text_format="longalign_chat",
        )
        self.assertIn("### System:", longalign_text)
        self.assertIn("### User:", longalign_text)
        self.assertIn("### Assistant:", longalign_text)
        self.assertIsNone(invalid_text)

    def test_build_datasets_mix_interleaves_and_resizes_sources(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "question": ["q1", "q2", "q3"],
                                "response": ["a1", "a2", "a3"],
                                "system_prompt": ["sys", "sys", "sys"],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "question": ["vq1", "vq2"],
                                "response": ["va1", "va2"],
                                "system_prompt": ["sys", "sys"],
                            }
                        ),
                    }
                )
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "instruction": ["inst1", "inst2"],
                                "input": ["input1 words words", "input2 words words"],
                                "output": ["out1 words words", "out2 words words"],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "instruction": ["vinst1", "vinst2"],
                                "input": ["vin1 words words", "vin2 words words"],
                                "output": ["vout1 words words", "vout2 words words"],
                            }
                        ),
                    }
                )
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, self.training_args, self.tokenizer)

        self.assertEqual(data_info["dataset_mode"], "mix")
        self.assertEqual(data_info["dataset_mix_sources"], ["openorca", "alpaca"])
        self.assertTrue(data_info["lazy_iterable"])
        self.assertEqual(len(data_info["source_stats"]), 2)
        self.assertGreater(len(list(train_dataset)), 0)
        self.assertIsNone(eval_dataset)
        for source_stat in data_info["source_stats"]:
            self.assertEqual(source_stat["sampling_policy"], "lazy_streaming")
            self.assertGreaterEqual(source_stat["processed_raw_rows"], 1)
            self.assertNotIn("packed_rows", source_stat)

    def test_build_datasets_mix_limits_train_preprocessing(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )
        training_args = SimpleNamespace(
            model_max_length=4,
            max_steps=1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            world_size=1,
            seed=7,
            eval_strategy=IntervalStrategy.NO,
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(5000)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(5000)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _train_dataset, eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)

        self.assertIsNone(eval_dataset)
        for source_stat in data_info["source_stats"]:
            self.assertEqual(source_stat["raw_rows"], 5000)
            self.assertEqual(source_stat["processed_raw_rows"], 5000)
            self.assertFalse(source_stat["limited_preprocessing"])

    def test_build_datasets_mix_is_deterministic_for_same_seed(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        first_args = SimpleNamespace(
            model_max_length=4,
            max_steps=8,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            world_size=1,
            seed=17,
            eval_strategy=IntervalStrategy.NO,
        )
        second_args = SimpleNamespace(**first_args.__dict__)

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            first_train, _first_eval, first_info = build_datasets(args, first_args, ContentTokenizer())
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            second_train, _second_eval, second_info = build_datasets(args, second_args, ContentTokenizer())

        self.assertEqual(_dataset_signature(first_train), _dataset_signature(second_train))
        self.assertEqual(first_info["source_stats"], second_info["source_stats"])

    def test_build_datasets_mix_changes_with_different_seed(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        seed_17_args = SimpleNamespace(
            model_max_length=4,
            max_steps=8,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            world_size=1,
            seed=17,
            eval_strategy=IntervalStrategy.NO,
        )
        seed_23_args = SimpleNamespace(**{**seed_17_args.__dict__, "seed": 23})

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            seed_17_train, _seed_17_eval, _seed_17_info = build_datasets(args, seed_17_args, ContentTokenizer())
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            seed_23_train, _seed_23_eval, _seed_23_info = build_datasets(args, seed_23_args, ContentTokenizer())

        self.assertNotEqual(_dataset_signature(seed_17_train), _dataset_signature(seed_23_train))

    def test_build_datasets_mix_repeats_short_source_to_target(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )
        training_args = SimpleNamespace(
            model_max_length=4,
            max_steps=10,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            world_size=1,
            seed=7,
            eval_strategy=IntervalStrategy.NO,
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(1)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(1)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _train_dataset, _eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)

        for source_stat in data_info["source_stats"]:
            self.assertEqual(source_stat["processed_raw_rows"], source_stat["raw_rows"])
            self.assertFalse(source_stat["limited_preprocessing"])
            self.assertNotIn("packed_rows", source_stat)

    def test_build_datasets_mix_rejects_empty_packed_source(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )
        training_args = SimpleNamespace(
            model_max_length=64,
            max_steps=3,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
            world_size=1,
            seed=7,
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "question": ["short"],
                                "response": ["tiny"],
                                "system_prompt": [""],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "question": ["short"],
                                "response": ["tiny"],
                                "system_prompt": [""],
                            }
                        ),
                    }
                )
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "instruction": ["inst1", "inst2"],
                                "input": ["input1 words words", "input2 words words"],
                                "output": ["out1 words words", "out2 words words"],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "instruction": ["vinst1"],
                                "input": ["vin1 words words"],
                                "output": ["vout1 words words"],
                            }
                        ),
                    }
                )
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)
        self.assertGreater(len(list(train_dataset)), 0)
        self.assertIsNone(eval_dataset)
        self.assertEqual(len(data_info["source_stats"]), 2)

    def test_build_datasets_mix_supports_long_sources_without_eval(self):
        args = SimpleNamespace(
            dataset_mix_spec="longalpaca=0.5,longalign=0.5",
            dataset_mix_sources=["longalpaca", "longalign"],
            dataset_mix_weights=[0.5, 0.5],
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Yukang/LongAlpaca-12k":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "instruction": ["inst1", "inst2"],
                                "input": ["long input words words", "another long input words"],
                                "output": ["out1 words words", "out2 words words"],
                            }
                        )
                    }
                )
            if path == "zai-org/LongAlign-10k":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "messages": [
                                    [
                                        {"role": "user", "content": "question words words"},
                                        {"role": "assistant", "content": "answer words words"},
                                    ],
                                    [
                                        {"role": "system", "content": "be concise"},
                                        {"role": "user", "content": "another question words"},
                                        {"role": "assistant", "content": "another answer words"},
                                    ],
                                ],
                                "length": [100, 200],
                            }
                        )
                    }
                )
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, self.training_args, self.tokenizer)

        self.assertEqual(data_info["dataset_mode"], "mix")
        self.assertEqual(data_info["dataset_mix_sources"], ["longalpaca", "longalign"])
        self.assertIsNone(eval_dataset)
        self.assertGreater(len(list(train_dataset)), 0)
        self.assertEqual(len(data_info["source_stats"]), 2)
        for source_stat in data_info["source_stats"]:
            self.assertIn(source_stat["alias"], {"longalpaca", "longalign"})
            self.assertNotIn("packed_rows", source_stat)

    def test_build_datasets_single_skips_eval_when_eval_strategy_is_no(self):
        args = SimpleNamespace(
            dataset_name="dummy",
            dataset_config_name=None,
            train_split="train",
            eval_split="validation",
            train_file="dummy.txt",
            text_field="text",
            max_train_samples=None,
            dataset_mix_spec=None,
            dataset_task="lm",
        )
        training_args = SimpleNamespace(
            model_max_length=4,
            eval_strategy=IntervalStrategy.NO,
            seed=0,
            data_seed=0,
            group_by_length=True,
        )

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["train words words words", "more train words words"]}),
                "validation": Dataset.from_dict({"text": ["eval words words words", "more eval words words"]}),
            }
        )
        with mock.patch("datasets.load_dataset", return_value=dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)

        self.assertGreater(len(train_dataset), 0)
        self.assertIsNone(eval_dataset)
        self.assertEqual(data_info["dataset_mode"], "file")

    def test_build_datasets_mix_skips_eval_when_eval_strategy_is_no(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )
        training_args = SimpleNamespace(
            model_max_length=4,
            max_steps=3,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
            world_size=1,
            seed=7,
            eval_strategy=IntervalStrategy.NO,
        )

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "question": ["q1", "q2", "q3"],
                                "response": ["a1", "a2", "a3"],
                                "system_prompt": ["sys", "sys", "sys"],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "question": ["vq1", "vq2"],
                                "response": ["va1", "va2"],
                                "system_prompt": ["sys", "sys"],
                            }
                        ),
                    }
                )
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "instruction": ["inst1", "inst2"],
                                "input": ["input1 words words", "input2 words words"],
                                "output": ["out1 words words", "out2 words words"],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {
                                "instruction": ["vinst1", "vinst2"],
                                "input": ["vin1 words words", "vin2 words words"],
                                "output": ["vout1 words words", "vout2 words words"],
                            }
                        ),
                    }
                )
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)

        self.assertGreater(len(list(train_dataset)), 0)
        self.assertIsNone(eval_dataset)
        self.assertTrue(data_info["lazy_iterable"])
        self.assertEqual(len(data_info["source_stats"]), 2)


class DistillDataTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_prepare_distill_datasets_lazy_mix_returns_iterable(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(5000)})
            if path == "Yukang/LongAlpaca-12k":
                return DatasetDict({"train": _make_alpaca_dataset(5000)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_distill_datasets(
                "alpaca=0.5,longalpaca=0.5",
                seed=7,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )

        self.assertEqual(dataset_mix_spec, "alpaca=0.5,longalpaca=0.5")
        self.assertTrue(is_iterable_training_dataset(train_ds))
        self.assertIsNone(eval_ds)
        for source_info in source_stats:
            self.assertEqual(source_info["raw_rows"], 5000)
            self.assertIsNone(source_info["actual_rows"])
            self.assertFalse(source_info["limited_preprocessing"])
            self.assertEqual(source_info["sampling_policy"], "lazy_streaming")
            self.assertTrue(source_info["is_iterable"])
            self.assertNotIn("packed_rows", source_info)

    def test_prepare_distill_datasets_single_source_is_lazy_iterable(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec, source_stats, train_ds, eval_ds, _eval_split = prepare_distill_datasets(
                "openorca=1.0",
                seed=17,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )

        self.assertTrue(is_iterable_training_dataset(train_ds))
        self.assertIsNone(eval_ds)
        self.assertIsNone(source_stats[0]["actual_rows"])
        self.assertTrue(source_stats[0]["is_iterable"])
        first = next(iter(train_ds))
        self.assertIn("input_ids", first)

    def test_prepare_distill_datasets_is_deterministic_for_same_seed(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_a, stats_a, train_a, _eval_a, _split_a = prepare_distill_datasets(
                "openorca=1.0",
                seed=17,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_b, stats_b, train_b, _eval_b, _split_b = prepare_distill_datasets(
                "openorca=1.0",
                seed=17,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )

        first_a = next(iter(train_a))
        first_b = next(iter(train_b))
        self.assertTrue(torch.equal(first_a["input_ids"], first_b["input_ids"]))
        self.assertEqual(stats_a, stats_b)

    def test_prepare_distill_datasets_changes_with_different_seed(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_a, _stats_a, train_a, _eval_a, _split_a = prepare_distill_datasets(
                "openorca=1.0",
                seed=17,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_b, _stats_b, train_b, _eval_b, _split_b = prepare_distill_datasets(
                "openorca=1.0",
                seed=23,
                tokenizer=self.tokenizer,
                max_seq_len=32,
            )

        pairs_a = [row["input_ids"].tolist() for row, _ in zip(train_a, range(8))]
        pairs_b = [row["input_ids"].tolist() for row, _ in zip(train_b, range(8))]
        self.assertNotEqual(pairs_a, pairs_b)

    def test_prepare_distill_datasets_rejects_empty_source(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": Dataset.from_dict({"question": [], "response": [], "system_prompt": []})})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            with self.assertRaises(ValueError):
                prepare_distill_datasets(
                    "openorca=1.0",
                    seed=7,
                    tokenizer=self.tokenizer,
                    max_seq_len=32,
                )

    def test_prepare_distill_datasets_requires_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "requires tokenizer"):
            prepare_distill_datasets("openorca=1.0", seed=7)

    def test_build_calibration_input_ids_lazy_stream(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(5000)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            blocks = build_calibration_input_ids(
                "openorca=1.0",
                tokenizer=self.tokenizer,
                nsamples=2,
                seqlen=4,
                seed=7,
            )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(tuple(blocks[0].shape), (1, 4))
        self.assertEqual(tuple(blocks[1].shape), (1, 4))

    def test_build_calibration_input_ids_rejects_empty_text_source(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "question": [""],
                                "response": [""],
                                "system_prompt": [""],
                            }
                        )
                    }
                )
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            with self.assertRaises(ValueError):
                build_calibration_input_ids(
                    "openorca=1.0",
                    tokenizer=self.tokenizer,
                    nsamples=1,
                    seqlen=4,
                    seed=7,
                )


class LazyIterableWorkerAndRawCacheTest(unittest.TestCase):
    def _preset(self):
        return DatasetMixSourcePreset(
            alias="tinymsg",
            path="tiny/path",
            config=None,
            train_split="train",
            eval_split=None,
            text_field="messages",
            text_format="edgerazor_messages",
            supports_lm=True,
            supports_sft=True,
        )

    def test_iterable_worker_shard_indexable_dataset_has_no_duplicates(self):
        raw_dataset = [{"id": idx} for idx in range(32)]

        class IdDataset(_LazyPresetIterableDataset):
            def _encode_record(self, record):
                return {"id": torch.tensor(int(record["id"]))}

        dataset = IdDataset(
            raw_dataset,
            tokenizer=object(),
            max_seq_len=8,
            task="messages",
            preset=self._preset(),
            seed=1,
        )
        baseline = [int(row["id"].item()) for row in dataset]
        self.assertEqual(baseline, list(range(32)))

        worker0 = SimpleNamespace(id=0, num_workers=2)
        worker1 = SimpleNamespace(id=1, num_workers=2)
        with mock.patch("e2e_common.lazy_datasets.get_worker_info", return_value=worker0):
            ids0 = [int(row["id"].item()) for row in dataset]
        with mock.patch("e2e_common.lazy_datasets.get_worker_info", return_value=worker1):
            ids1 = [int(row["id"].item()) for row in dataset]
        merged = ids0 + ids1
        self.assertEqual(len(merged), 32)
        self.assertEqual(set(merged), set(range(32)))
        self.assertEqual(len(set(merged)), 32)

    def test_iterable_worker_shard_non_indexable_stream_has_no_duplicates(self):
        class RawStream:
            def __iter__(self):
                for idx in range(32):
                    yield {"id": idx}

        worker0 = SimpleNamespace(id=0, num_workers=2)
        worker1 = SimpleNamespace(id=1, num_workers=2)
        with mock.patch("e2e_common.lazy_datasets.get_worker_info", return_value=worker0):
            ids0 = [record["id"] for record in _iter_raw_records_for_worker(RawStream())]
        with mock.patch("e2e_common.lazy_datasets.get_worker_info", return_value=worker1):
            ids1 = [record["id"] for record in _iter_raw_records_for_worker(RawStream())]
        self.assertEqual(set(ids0).intersection(ids1), set())
        self.assertEqual(set(ids0 + ids1), set(range(32)))

    def test_raw_dataset_cache_reuses_unshuffled_source_and_preserves_seed_order(self):
        raw_dataset = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": f"question_{idx}"},
                        {"role": "assistant", "content": f"answer_{idx}"},
                    ]
                    for idx in range(24)
                ]
            }
        )
        preset = self._preset()
        tokenizer = ContentTokenizer()
        cache = {}
        load_calls = []

        def fake_load_preset_raw_datasets(received_preset):
            load_calls.append(received_preset.alias)
            return raw_dataset, None

        def signature(dataset):
            return _dataset_signature(dataset, limit=8)

        with mock.patch.dict(
            "e2e_common.lazy_datasets.DATASET_MIX_SOURCE_PRESETS",
            {"tinymsg": preset},
            clear=False,
        ), mock.patch(
            "e2e_common.lazy_datasets._load_preset_raw_datasets",
            side_effect=fake_load_preset_raw_datasets,
        ):
            _spec31, _stats31, cached31, _eval31, _split31 = prepare_distill_datasets(
                "tinymsg=1.0",
                seed=31,
                tokenizer=tokenizer,
                max_seq_len=32,
                raw_dataset_cache=cache,
            )
            _spec32, _stats32, cached32, _eval32, _split32 = prepare_distill_datasets(
                "tinymsg=1.0",
                seed=32,
                tokenizer=tokenizer,
                max_seq_len=32,
                raw_dataset_cache=cache,
            )

        self.assertEqual(load_calls, ["tinymsg"])
        self.assertEqual(list(cache.values()), [raw_dataset])
        cached31_sig = signature(cached31)
        cached32_sig = signature(cached32)
        self.assertNotEqual(cached31_sig, cached32_sig)

        with mock.patch.dict(
            "e2e_common.lazy_datasets.DATASET_MIX_SOURCE_PRESETS",
            {"tinymsg": preset},
            clear=False,
        ), mock.patch(
            "e2e_common.lazy_datasets._load_preset_raw_datasets",
            side_effect=fake_load_preset_raw_datasets,
        ):
            _spec31b, _stats31b, uncached31, _eval31b, _split31b = prepare_distill_datasets(
                "tinymsg=1.0",
                seed=31,
                tokenizer=tokenizer,
                max_seq_len=32,
            )
            _spec32b, _stats32b, uncached32, _eval32b, _split32b = prepare_distill_datasets(
                "tinymsg=1.0",
                seed=32,
                tokenizer=tokenizer,
                max_seq_len=32,
            )

        self.assertEqual(cached31_sig, signature(uncached31))
        self.assertEqual(cached32_sig, signature(uncached32))


def _make_fineweb_dataset(count: int):
    return Dataset.from_dict(
        {"text": [f"fineweb unique marker {idx} document text" for idx in range(count)]}
    )


def _make_race_dataset(count: int):
    return Dataset.from_dict(
        {
            "article": [f"race passage {idx}" for idx in range(count)],
            "question": [f"race question {idx}" for idx in range(count)],
            "options": [["alpha", "beta", "gamma", "delta"] for _ in range(count)],
            "answer": ["B" for _ in range(count)],
        }
    )


def _make_sciq_dataset(count: int):
    return Dataset.from_dict(
        {
            "support": [f"sciq support {idx}" for idx in range(count)],
            "question": [f"sciq question {idx}" for idx in range(count)],
            "correct_answer": [f"sciq answer {idx}" for idx in range(count)],
        }
    )


def _make_longalign_dataset(count: int):
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": f"longalign question {idx}"},
                    {"role": "assistant", "content": f"longalign answer {idx}"},
                ]
                for idx in range(count)
            ]
        }
    )


class LazyHeterogeneousLmMixTest(unittest.TestCase):
    def _patch_raw_loaders(self, raw_by_alias):
        def load_raw(preset):
            alias = str(preset.alias)
            if alias not in raw_by_alias:
                raise AssertionError(f"unexpected preset {alias}")
            return raw_by_alias[alias], None

        return mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw)

    def test_lm_heterogeneous_mix_builds_iterable_dataset(self):
        raw_by_alias = {
            "openorca": _make_openorca_dataset(8),
            "alpaca": _make_alpaca_dataset(8),
        }
        with self._patch_raw_loaders(raw_by_alias):
            _spec, _stats, dataset, is_iterable = build_mixed_lazy_dataset(
                "openorca=0.5,alpaca=0.5",
                task="lm",
                tokenizer=ContentTokenizer(),
                max_seq_len=32,
                seed=31,
            )

        self.assertTrue(is_iterable)
        self.assertIsInstance(dataset, IterableDataset)
        first = next(iter(dataset))
        self.assertIn("input_ids", first)
        self.assertIn("labels", first)
        self.assertEqual(first["input_ids"].tolist(), first["labels"].tolist())

    def test_lm_heterogeneous_mix_uses_each_source_preset(self):
        from train_utils.distill_data import encode_canonical_record

        tokenizer = ContentTokenizer()
        openorca_raw = _make_openorca_dataset(16)
        fineweb_raw = _make_fineweb_dataset(16)
        raw_by_alias = {
            "openorca": openorca_raw,
            "fineweb_edu": fineweb_raw,
        }

        expected_openorca = {
            tuple(
                encode_canonical_record(
                    dict(row),
                    tokenizer,
                    text_format="openorca",
                    text_field="text",
                    task="lm",
                    model_max_length=64,
                )["input_ids"].tolist()
            )
            for row in openorca_raw
        }
        expected_fineweb = {
            tuple(
                encode_canonical_record(
                    dict(row),
                    tokenizer,
                    text_format="text",
                    text_field="text",
                    task="lm",
                    model_max_length=64,
                )["input_ids"].tolist()
            )
            for row in fineweb_raw
        }
        self.assertTrue(expected_openorca.isdisjoint(expected_fineweb))

        with self._patch_raw_loaders(raw_by_alias):
            _spec, _stats, dataset, _is_iterable = build_mixed_lazy_dataset(
                "openorca=0.5,fineweb_edu=0.5",
                task="lm",
                tokenizer=tokenizer,
                max_seq_len=64,
                seed=31,
            )

        seen_openorca = 0
        seen_fineweb = 0
        for row in dataset:
            signature = tuple(int(token) for token in row["input_ids"].tolist())
            if signature in expected_openorca:
                seen_openorca += 1
            elif signature in expected_fineweb:
                seen_fineweb += 1
            else:
                self.fail(f"sample was not encoded by either source preset: {signature}")
            self.assertEqual(row["input_ids"].tolist(), row["labels"].tolist())

        self.assertGreater(seen_openorca, 0)
        self.assertGreater(seen_fineweb, 0)

    def test_sft_heterogeneous_mix_is_source_aware(self):
        raw_by_alias = {
            "openorca": _make_openorca_dataset(4),
            "alpaca": _make_alpaca_dataset(4),
        }
        tokenizer = ContentTokenizer()
        with self._patch_raw_loaders(raw_by_alias):
            _spec, _stats, dataset, is_iterable = build_mixed_lazy_dataset(
                "openorca=0.5,alpaca=0.5",
                task="sft",
                tokenizer=tokenizer,
                max_seq_len=64,
                seed=0,
            )
        self.assertTrue(is_iterable)
        rows = []
        for idx, row in enumerate(dataset):
            rows.append(row)
            if idx >= 5:
                break
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(set(row.keys()), {"input_ids", "attention_mask", "labels"})
            self.assertTrue(any(int(v) != -100 for v in row["labels"].tolist()))

    def test_requested_seven_source_lm_mix_can_enter_training(self):
        raw_by_alias = {
            "openorca": _make_openorca_dataset(4),
            "fineweb_edu": _make_fineweb_dataset(4),
            "race": _make_race_dataset(4),
            "sciq": _make_sciq_dataset(4),
            "alpaca": _make_alpaca_dataset(4),
            "longalpaca": _make_alpaca_dataset(4),
            "longalign": _make_longalign_dataset(4),
        }
        mix_spec = "openorca=0.20,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.10,longalign=0.10"
        tokenizer = ContentTokenizer()

        with self._patch_raw_loaders(raw_by_alias):
            _spec, source_stats, dataset, is_iterable = build_mixed_lazy_dataset(
                mix_spec,
                task="lm",
                tokenizer=tokenizer,
                max_seq_len=64,
                seed=31,
            )

        self.assertTrue(is_iterable)
        self.assertEqual(
            [item["alias"] for item in source_stats],
            ["openorca", "fineweb_edu", "race", "sciq", "alpaca", "longalpaca", "longalign"],
        )
        for item in source_stats:
            self.assertNotIn("packed_rows", item)

        rows = list(dataset)
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertGreater(int(row["input_ids"].numel()), 0)
            self.assertEqual(row["input_ids"].tolist(), row["labels"].tolist())


if __name__ == "__main__":
    unittest.main()
