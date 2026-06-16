import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, DatasetDict
import torch
from transformers.trainer_utils import IntervalStrategy

from e2e_common.data import _record_to_text, build_datasets
from raw_e2e_fintuning.args import parse_args
from train_utils.lora_data import build_calibration_input_ids, prepare_distill_datasets
from vae_e2e_fintuning.args import parse_args as parse_vae_e2e_args
from vae_e2e_fintuning.trainer import (
    build_vae_hidden_layer_weights,
    compute_vae_hidden_alignment_loss,
)


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

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
    return [_tensorish_to_tuple(dataset[idx]["input_ids"]) for idx in range(min(len(dataset), limit))]


def _build_datasets_with_recorded_num_proc(args, training_args, tokenizer, dataset):
    requested_num_proc = []
    original_map = Dataset.map

    def _wrapped_map(self, *map_args, **map_kwargs):
        requested_num_proc.append(map_kwargs.get("num_proc"))
        call_kwargs = dict(map_kwargs)
        if call_kwargs.get("num_proc") == 2:
            call_kwargs["num_proc"] = None
        return original_map(self, *map_args, **call_kwargs)

    with mock.patch("e2e_common.data.load_dataset", return_value=dataset), mock.patch(
        "datasets.arrow_dataset.Dataset.map",
        new=_wrapped_map,
    ):
        built = build_datasets(args, training_args, tokenizer)
    return built, requested_num_proc


class DatasetMixArgsTest(unittest.TestCase):
    def test_parse_args_normalizes_dataset_mix(self):
        e2e_args, _hf_args, training_args = parse_args(
                [
                    "--student_model_path",
                    "dummy-model",
                    "--dataset_mix",
                    "openorca=3,fineweb_edu=1",
                    "--max_steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.dataset_mix_sources, ["openorca", "fineweb_edu"])
        self.assertEqual(len(e2e_args.dataset_mix_weights), 2)
        self.assertAlmostEqual(sum(e2e_args.dataset_mix_weights), 1.0)
        self.assertEqual(e2e_args.dataset_mix_spec, "openorca=0.75,fineweb_edu=0.25")
        self.assertEqual(training_args.max_steps, 10)

    def test_parse_args_rejects_duplicate_alias(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_model_path",
                    "dummy-model",
                    "--dataset_mix",
                    "openorca=1,openorca=1",
                    "--max_steps",
                    "10",
                ]
            )

    def test_parse_args_rejects_explicit_single_source_args(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--student_model_path",
                    "dummy-model",
                    "--dataset_mix",
                    "openorca=1",
                    "--text_field",
                    "text",
                    "--max_steps",
                    "10",
                ]
            )

    def test_parse_args_accepts_long_dataset_aliases(self):
        e2e_args, _hf_args, _training_args = parse_args(
                [
                    "--student_model_path",
                    "dummy-model",
                    "--dataset_mix",
                    "longalpaca=2,longalign=1",
                    "--max_steps",
                "10",
            ]
        )
        self.assertEqual(e2e_args.dataset_mix_sources, ["longalpaca", "longalign"])
        self.assertEqual(e2e_args.dataset_mix_spec, "longalpaca=0.666666666667,longalign=0.333333333333")


class VAEE2EHiddenLossArgsTest(unittest.TestCase):
    def _parse_with_checkpoint(self, extra_args):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/checkpoint_meta.json", "w", encoding="utf-8") as handle:
                handle.write("{}")
            args, _hf_args, _training_args = parse_vae_e2e_args(
                [
                    "--student_checkpoint_dir",
                    tmpdir,
                    "--dataset_mix",
                    "openorca=1.0",
                    "--max_steps",
                    "1",
                    *extra_args,
                ]
            )
        return args

    def test_hidden_loss_defaults_to_disabled_uniform(self):
        args = self._parse_with_checkpoint([])

        self.assertEqual(args.hidden_loss_weight, 0.0)
        self.assertEqual(args.hidden_layer_weighting, "uniform")

    def test_hidden_loss_accepts_linear_depth(self):
        args = self._parse_with_checkpoint(
            [
                "--hidden_loss_weight",
                "0.003",
                "--hidden_layer_weighting",
                "linear_depth",
            ]
        )

        self.assertEqual(args.hidden_loss_weight, 0.003)
        self.assertEqual(args.hidden_layer_weighting, "linear_depth")

    def test_hidden_loss_rejects_negative_weight(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--hidden_loss_weight", "-0.1"])

    def test_hidden_loss_rejects_unknown_weighting(self):
        with self.assertRaises(SystemExit):
            self._parse_with_checkpoint(["--hidden_layer_weighting", "quadratic"])


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


class DatasetMixBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
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
        self.assertEqual(data_info["dataset_mix_target_examples"], 14)
        self.assertEqual(len(data_info["source_stats"]), 2)
        self.assertGreaterEqual(len(train_dataset), data_info["required_train_examples"])
        self.assertLessEqual(len(train_dataset), data_info["dataset_mix_target_examples"])
        self.assertIsNotNone(eval_dataset)
        self.assertGreater(len(eval_dataset), 0)
        for source_stat in data_info["source_stats"]:
            self.assertEqual(source_stat["target_rows"], 7)
            self.assertGreaterEqual(source_stat["repeat_factor"], 1.0)
            self.assertEqual(source_stat["sampling_policy"], "shuffled_raw_streaming_pack")
            self.assertEqual(source_stat["collected_packed_rows"], source_stat["packed_rows"])
            self.assertGreaterEqual(source_stat["processed_raw_rows"], 1)

    def test_build_datasets_mix_limits_train_preprocessing(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
            dataset_num_proc=1,
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
            self.assertEqual(source_stat["processed_raw_rows"], 4096)
            self.assertLess(source_stat["processed_raw_rows"], source_stat["raw_rows"])
            self.assertTrue(source_stat["limited_preprocessing"])
            self.assertEqual(source_stat["target_rows"], 1)

    def test_build_datasets_mix_is_deterministic_for_same_seed(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
            dataset_num_proc=1,
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
            first_train, _first_eval, first_info = build_datasets(args, first_args, self.tokenizer)
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            second_train, _second_eval, second_info = build_datasets(args, second_args, self.tokenizer)

        self.assertEqual(_dataset_signature(first_train), _dataset_signature(second_train))
        self.assertEqual(first_info["source_stats"], second_info["source_stats"])

    def test_build_datasets_mix_changes_with_different_seed(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
            dataset_num_proc=1,
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
            seed_17_train, _seed_17_eval, _seed_17_info = build_datasets(args, seed_17_args, self.tokenizer)
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            seed_23_train, _seed_23_eval, _seed_23_info = build_datasets(args, seed_23_args, self.tokenizer)

        self.assertNotEqual(_dataset_signature(seed_17_train), _dataset_signature(seed_23_train))

    def test_build_datasets_mix_repeats_short_source_to_target(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
            dataset_num_proc=1,
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
            self.assertLess(source_stat["packed_rows"], source_stat["target_rows"])
            self.assertGreater(source_stat["repeat_factor"], 1.0)

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
            with self.assertRaises(ValueError):
                build_datasets(args, training_args, self.tokenizer)

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
        self.assertGreater(len(train_dataset), 0)
        self.assertEqual(len(data_info["source_stats"]), 2)
        for source_stat in data_info["source_stats"]:
            self.assertIn(source_stat["alias"], {"longalpaca", "longalign"})
            self.assertGreaterEqual(source_stat["packed_rows"], 1)

    def test_build_datasets_single_skips_eval_when_eval_strategy_is_no(self):
        args = SimpleNamespace(
            dataset_name="dummy",
            dataset_config_name=None,
            train_split="train",
            eval_split="validation",
            train_file="dummy.txt",
            eval_file="dummy_eval.txt",
            text_field="text",
            max_train_samples=None,
            max_eval_samples=None,
            dataset_num_proc=1,
            dataset_mix_spec=None,
        )
        training_args = SimpleNamespace(
            model_max_length=4,
            eval_strategy=IntervalStrategy.NO,
        )

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["train words words words", "more train words words"]}),
                "validation": Dataset.from_dict({"text": ["eval words words words", "more eval words words"]}),
            }
        )
        with mock.patch("e2e_common.data.load_dataset", return_value=dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, training_args, self.tokenizer)

        self.assertGreater(len(train_dataset), 0)
        self.assertIsNone(eval_dataset)
        self.assertEqual(data_info["dataset_mode"], "single")

    def test_build_datasets_mix_skips_eval_when_eval_strategy_is_no(self):
        args = SimpleNamespace(
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
            dataset_num_proc=1,
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

        self.assertGreater(len(train_dataset), 0)
        self.assertIsNone(eval_dataset)
        self.assertEqual([stat["eval_packed_rows"] for stat in data_info["source_stats"]], [0, 0])

    def test_build_datasets_num_proc_preserves_single_source_outputs(self):
        args_base = {
            "dataset_name": "dummy",
            "dataset_config_name": None,
            "train_split": "train",
            "eval_split": "validation",
            "train_file": "dummy.txt",
            "eval_file": "dummy_eval.txt",
            "text_field": "text",
            "max_train_samples": None,
            "max_eval_samples": None,
            "dataset_mix_spec": None,
        }
        training_args = SimpleNamespace(
            model_max_length=4,
            eval_strategy=IntervalStrategy.STEPS,
        )
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": [
                            "alpha beta gamma delta",
                            "epsilon zeta eta theta",
                            "iota kappa lambda mu",
                            "nu xi omicron pi",
                        ]
                    }
                ),
                "validation": Dataset.from_dict({"text": ["rho sigma tau upsilon", "phi chi psi omega"]}),
            }
        )

        (train_single, eval_single, _), single_requested = _build_datasets_with_recorded_num_proc(
            SimpleNamespace(dataset_num_proc=1, **args_base),
            training_args,
            self.tokenizer,
            dataset,
        )
        (train_multi, eval_multi, _), multi_requested = _build_datasets_with_recorded_num_proc(
            SimpleNamespace(dataset_num_proc=2, **args_base),
            training_args,
            self.tokenizer,
            dataset,
        )

        self.assertEqual(len(train_single), len(train_multi))
        self.assertEqual(len(eval_single), len(eval_multi))
        self.assertEqual(set(train_single.column_names), set(train_multi.column_names))
        self.assertTrue(torch.equal(train_single[0]["input_ids"], train_multi[0]["input_ids"]))
        self.assertTrue(torch.equal(train_single[0]["labels"], train_multi[0]["labels"]))
        self.assertTrue(torch.equal(eval_single[0]["input_ids"], eval_multi[0]["input_ids"]))
        self.assertNotIn(2, single_requested)
        self.assertIn(2, multi_requested)

    def test_build_datasets_num_proc_preserves_structured_outputs(self):
        args_base = {
            "dataset_name": "dummy",
            "dataset_config_name": None,
            "train_split": "train",
            "eval_split": "validation",
            "train_file": "dummy.txt",
            "eval_file": "dummy_eval.txt",
            "text_field": "text",
            "max_train_samples": None,
            "max_eval_samples": None,
            "dataset_mix_spec": None,
        }
        training_args = SimpleNamespace(
            model_max_length=4,
            eval_strategy=IntervalStrategy.STEPS,
        )
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "question": ["q1 words words", "q2 words words"],
                        "response": ["a1 words words", "a2 words words"],
                        "system_prompt": ["sys words", "sys words"],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "question": ["vq1 words words", "vq2 words words"],
                        "response": ["va1 words words", "va2 words words"],
                        "system_prompt": ["vsys words", "vsys words"],
                    }
                ),
            }
        )

        (train_single, eval_single, _), single_requested = _build_datasets_with_recorded_num_proc(
            SimpleNamespace(dataset_num_proc=1, **args_base),
            training_args,
            self.tokenizer,
            dataset,
        )
        (train_multi, eval_multi, _), multi_requested = _build_datasets_with_recorded_num_proc(
            SimpleNamespace(dataset_num_proc=2, **args_base),
            training_args,
            self.tokenizer,
            dataset,
        )

        self.assertEqual(len(train_single), len(train_multi))
        self.assertEqual(len(eval_single), len(eval_multi))
        self.assertTrue(torch.equal(train_single[0]["input_ids"], train_multi[0]["input_ids"]))
        self.assertTrue(torch.equal(train_single[0]["labels"], train_multi[0]["labels"]))
        self.assertTrue(torch.equal(eval_single[0]["input_ids"], eval_multi[0]["input_ids"]))
        self.assertNotIn(2, single_requested)
        self.assertIn(2, multi_requested)


class DistillDataTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.num_proc_patch = mock.patch.dict("os.environ", {"CAT_DISTILL_DATASET_NUM_PROC": "1"})
        self.num_proc_patch.start()
        self.addCleanup(self.num_proc_patch.stop)

    def test_prepare_distill_datasets_limits_source_preprocessing(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(5000)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(5000)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_distill_datasets(
                "openorca=0.5,alpaca=0.5",
                nsamples=2,
                seed=7,
            )

        self.assertEqual(dataset_mix_spec, "openorca=0.5,alpaca=0.5")
        self.assertEqual(len(train_ds), 2)
        self.assertIsNone(eval_ds)
        for source_info in source_stats:
            self.assertEqual(source_info["target_rows"], 1)
            self.assertEqual(source_info["actual_rows"], 1)
            self.assertEqual(source_info["raw_rows"], 5000)
            self.assertEqual(source_info["processed_raw_rows"], 4096)
            self.assertLess(source_info["processed_raw_rows"], source_info["raw_rows"])
            self.assertTrue(source_info["limited_preprocessing"])
            self.assertEqual(source_info["sampling_policy"], "shuffled_raw_streaming_text")

    def test_prepare_distill_datasets_is_deterministic_for_same_seed(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_a, stats_a, train_a, _eval_a, _split_a = prepare_distill_datasets(
                "openorca=0.5,alpaca=0.5",
                nsamples=10,
                seed=17,
            )
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_b, stats_b, train_b, _eval_b, _split_b = prepare_distill_datasets(
                "openorca=0.5,alpaca=0.5",
                nsamples=10,
                seed=17,
            )

        self.assertEqual([row["text"] for row in train_a], [row["text"] for row in train_b])
        self.assertEqual(stats_a, stats_b)

    def test_prepare_distill_datasets_changes_with_different_seed(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(128, variable_lengths=True)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(128, variable_lengths=True)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_a, _stats_a, train_a, _eval_a, _split_a = prepare_distill_datasets(
                "openorca=0.5,alpaca=0.5",
                nsamples=10,
                seed=17,
            )
        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            _spec_b, _stats_b, train_b, _eval_b, _split_b = prepare_distill_datasets(
                "openorca=0.5,alpaca=0.5",
                nsamples=10,
                seed=23,
            )

        self.assertNotEqual([row["text"] for row in train_a], [row["text"] for row in train_b])

    def test_prepare_distill_datasets_rejects_short_source(self):
        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(1)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(1)})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset):
            with self.assertRaises(ValueError):
                prepare_distill_datasets(
                    "openorca=0.5,alpaca=0.5",
                    nsamples=4,
                    seed=7,
                )

    def test_build_calibration_input_ids_limits_source_preprocessing(self):
        prepared_chunk_rows = []

        def fake_load_dataset(*, path, name=None, **_kwargs):
            if path == "Open-Orca/OpenOrca":
                return DatasetDict({"train": _make_openorca_dataset(5000)})
            if path == "vicgalle/alpaca-gpt4":
                return DatasetDict({"train": _make_alpaca_dataset(5000)})
            raise AssertionError(f"unexpected dataset path: {path}")

        from e2e_common.data import _prepare_text_dataset as original_prepare_text_dataset

        def wrapped_prepare_text_dataset(dataset, **kwargs):
            prepared_chunk_rows.append(len(dataset))
            return original_prepare_text_dataset(dataset, **kwargs)

        with mock.patch("e2e_common.data.load_dataset", side_effect=fake_load_dataset), mock.patch(
            "e2e_common.data._prepare_text_dataset",
            side_effect=wrapped_prepare_text_dataset,
        ):
            blocks = build_calibration_input_ids(
                "openorca=0.5,alpaca=0.5",
                tokenizer=self.tokenizer,
                nsamples=2,
                seqlen=4,
                seed=7,
            )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(prepared_chunk_rows, [4096, 4096])

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

if __name__ == "__main__":
    unittest.main()
