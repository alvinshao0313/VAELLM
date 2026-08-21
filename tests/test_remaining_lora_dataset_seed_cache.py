import unittest
from types import SimpleNamespace
from unittest import mock

from train_utils import lora_utils


class RemainingLoraDatasetSeedCacheTests(unittest.TestCase):
    def test_remaining_lora_cache_uses_base_seed_and_prepares_once(self):
        vae_args = SimpleNamespace()
        tokenizer = object()
        training_args = SimpleNamespace(distill_model_max_length=8192)
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        cfg0 = SimpleNamespace(
            dataset="tiny_a=0.7,tiny_b=0.3",
            base_seed=31,
            round_idx=0,
            seed=31,
        )
        cfg1 = SimpleNamespace(
            dataset="tiny_a=0.7,tiny_b=0.3",
            base_seed=31,
            round_idx=1,
            seed=32,
        )
        sentinel = ("mix", [], object(), None, None)

        with mock.patch.object(
            lora_utils,
            "prepare_distill_datasets",
            return_value=sentinel,
        ) as prepare_mock:
            first = lora_utils._prepare_or_reuse_remaining_lora_distill_dataset(
                vae_args=vae_args,
                cfg=cfg0,
                tokenizer=tokenizer,
                training_args=training_args,
                logger=logger,
            )
            second = lora_utils._prepare_or_reuse_remaining_lora_distill_dataset(
                vae_args=vae_args,
                cfg=cfg1,
                tokenizer=tokenizer,
                training_args=training_args,
                logger=logger,
            )

        self.assertIs(first, second)
        self.assertIs(first, sentinel)
        prepare_mock.assert_called_once()
        _args, kwargs = prepare_mock.call_args
        self.assertEqual(kwargs["seed"], 31)
        cache = vae_args._cached_remaining_lora_distill_datasets
        self.assertIn(("tiny_a=0.7,tiny_b=0.3", 8192, 31, id(tokenizer)), cache)
        self.assertNotIn(("tiny_a=0.7,tiny_b=0.3", 8192, 32, id(tokenizer)), cache)

    def test_remaining_lora_cache_base_seed_change_prepares_new_dataset(self):
        vae_args = SimpleNamespace()
        tokenizer = object()
        training_args = SimpleNamespace(distill_model_max_length=8192)
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        cfg31 = SimpleNamespace(
            dataset="tiny_a=0.7,tiny_b=0.3",
            base_seed=31,
            round_idx=0,
            seed=31,
        )
        cfg33 = SimpleNamespace(
            dataset="tiny_a=0.7,tiny_b=0.3",
            base_seed=33,
            round_idx=0,
            seed=33,
        )
        first_sentinel = ("mix31", [], object(), None, None)
        second_sentinel = ("mix33", [], object(), None, None)

        with mock.patch.object(
            lora_utils,
            "prepare_distill_datasets",
            side_effect=(first_sentinel, second_sentinel),
        ) as prepare_mock:
            first = lora_utils._prepare_or_reuse_remaining_lora_distill_dataset(
                vae_args=vae_args,
                cfg=cfg31,
                tokenizer=tokenizer,
                training_args=training_args,
                logger=logger,
            )
            second = lora_utils._prepare_or_reuse_remaining_lora_distill_dataset(
                vae_args=vae_args,
                cfg=cfg33,
                tokenizer=tokenizer,
                training_args=training_args,
                logger=logger,
            )

        self.assertIs(first, first_sentinel)
        self.assertIs(second, second_sentinel)
        self.assertEqual(prepare_mock.call_count, 2)
        self.assertEqual(prepare_mock.call_args_list[0].kwargs["seed"], 31)
        self.assertEqual(prepare_mock.call_args_list[1].kwargs["seed"], 33)

    def test_build_sft_args_separates_trainer_seed_and_data_seed(self):
        cfg = lora_utils._ResolvedDistillStageConfig(
            device="cpu",
            base_seed=31,
            round_idx=4,
            seed=35,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            steps=1,
            batch_size=1,
            lr=1e-4,
            weight_decay=0.0,
            log_every=1,
            temperature=1.0,
            loss_alpha=0.5,
            loss_type="sft",
            hidden_loss_weight=0.0,
            pre_mlp_hidden_loss_weight=0.0,
            prompt_kd_weight=0.0,
            hidden_alignment_layer_weighting="uniform",
            eakld_confidence_k=16,
            dataset="tiny_a=1.0",
            use_dora=False,
            use_distill_hif4_act=False,
            distill_tune_final_norm=False,
            distill_use_post_norm_head_linear=False,
        )
        cat_args = SimpleNamespace(
            output_dir=".result/test_remaining_lora_dataset_seed_cache",
            deterministic=False,
        )
        training_args = SimpleNamespace(
            distill_gradient_checkpointing_kwargs=None,
            distill_group_by_length=False,
            distill_gradient_accumulation_steps=1,
            distill_gradient_checkpointing=False,
            distill_optim="adamw_torch",
            fp16=False,
            bf16=False,
            distill_max_grad_norm=1.0,
            distill_warmup_ratio=0.0,
            distill_lr_scheduler_type="linear",
            distill_dataloader_num_workers=0,
        )

        args = lora_utils._build_sft_args(
            cat_args=cat_args,
            training_args=training_args,
            cfg=cfg,
            train_is_iterable=True,
            logger=None,
        )

        self.assertEqual(args.seed, 35)
        self.assertEqual(args.data_seed, 31)


if __name__ == "__main__":
    unittest.main()
