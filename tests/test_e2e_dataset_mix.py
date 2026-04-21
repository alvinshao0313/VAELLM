import unittest
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, DatasetDict

from e2e_fintuning.args import parse_args
from e2e_fintuning.data import _record_to_text, build_datasets
from e2e_fintuning.runtime import _validate_resume_checkpoint_config


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


class DatasetMixArgsTest(unittest.TestCase):
    def test_parse_args_normalizes_dataset_mix(self):
        e2e_args, _hf_args, training_args = parse_args(
            [
                "--student_checkpoint_dir",
                "dummy-checkpoint",
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
                    "--student_checkpoint_dir",
                    "dummy-checkpoint",
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
                    "--student_checkpoint_dir",
                    "dummy-checkpoint",
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
                "--student_checkpoint_dir",
                "dummy-checkpoint",
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

        with mock.patch("e2e_fintuning.data.load_dataset", side_effect=fake_load_dataset):
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

        with mock.patch("e2e_fintuning.data.load_dataset", side_effect=fake_load_dataset):
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

        with mock.patch("e2e_fintuning.data.load_dataset", side_effect=fake_load_dataset):
            train_dataset, eval_dataset, data_info = build_datasets(args, self.training_args, self.tokenizer)

        self.assertEqual(data_info["dataset_mode"], "mix")
        self.assertEqual(data_info["dataset_mix_sources"], ["longalpaca", "longalign"])
        self.assertIsNone(eval_dataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertEqual(len(data_info["source_stats"]), 2)
        for source_stat in data_info["source_stats"]:
            self.assertIn(source_stat["alias"], {"longalpaca", "longalign"})
            self.assertGreaterEqual(source_stat["packed_rows"], 1)


class DatasetMixResumeValidationTest(unittest.TestCase):
    def test_resume_config_rejects_dataset_mix_mismatch(self):
        args = SimpleNamespace(
            target_module_names=None,
            vae_lora_rank=8,
            vae_lora_alpha=16.0,
            vae_lora_dropout=0.0,
            vae_lora_tune_bias=False,
            vae_lora_variant="plain",
            vae_lora_init_mode="zero",
            vae_lora_use_rslora=False,
            vae_lora_use_dora=False,
            tune_final_norm=False,
            use_post_norm_head_linear=False,
            vae_adalora_target_r=8,
            vae_adalora_init_r=12,
            vae_adalora_tinit=0,
            vae_adalora_tfinal=0,
            vae_adalora_delta_t=1,
            vae_adalora_beta1=0.85,
            vae_adalora_beta2=0.85,
            vae_adalora_orth_reg_weight=0.5,
            loss_type="kd_top_1000",
            post_attn=False,
            lora_hif4_act=False,
            prewarm_frozen_vae=True,
            dataset_mix_spec="openorca=0.5,alpaca=0.5",
            dataset_mix_sources=["openorca", "alpaca"],
            dataset_mix_weights=[0.5, 0.5],
        )
        training_args = SimpleNamespace(max_steps=10, model_max_length=4096)
        meta = {
            "extra_meta": {
                "stage": "e2e_fintuning",
                "target_decoder_layers": [0, 1],
                "target_module_names": None,
                "vae_lora_rank": 8,
                "vae_lora_alpha": 16.0,
                "vae_lora_dropout": 0.0,
                "vae_lora_tune_bias": False,
                "vae_lora_bias_mode": "none",
                "tune_final_norm": False,
                "use_post_norm_head_linear": False,
                "vae_lora_variant": "plain",
                "vae_lora_init_mode": "zero",
                "vae_lora_use_rslora": False,
                "vae_lora_use_dora": False,
                "dataset_mode": "mix",
                "dataset_mix_spec": "openorca=0.6,alpaca=0.4",
                "dataset_mix_sources": ["openorca", "alpaca"],
                "dataset_mix_weights": [0.6, 0.4],
                "dataset_mix_block_size": 4096,
            }
        }
        with self.assertRaises(ValueError):
            _validate_resume_checkpoint_config(
                args=args,
                meta=meta,
                decoder_layer_ids=[0, 1],
                training_args=training_args,
            )


if __name__ == "__main__":
    unittest.main()
