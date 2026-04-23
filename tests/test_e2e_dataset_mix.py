import unittest
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, DatasetDict
import torch
from transformers.trainer_utils import IntervalStrategy

from e2e_common.data import _record_to_text, build_datasets
from raw_e2e_fintuning.args import parse_args


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, texts):
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
            train_file=None,
            eval_file=None,
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
            "train_file": None,
            "eval_file": None,
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
            "train_file": None,
            "eval_file": None,
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

if __name__ == "__main__":
    unittest.main()
