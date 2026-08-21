import unittest
from types import SimpleNamespace
from unittest import mock

from torch import nn

from train_utils.cat_after_category_distill import run_after_category_distill
from train_utils.lora_utils import RemainingLoraFinetuneResult
from train_utils.utils import collect_linears


_CATEGORIES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        for category in _CATEGORIES:
            setattr(self, category, nn.Linear(2, 2))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])


class CatInlineRemainingLoraTests(unittest.TestCase):
    def test_seven_category_remaining_target_state_machine(self):
        model = _TinyModel()
        expected = {
            "q_proj": _CATEGORIES[1:],
            "k_proj": _CATEGORIES[2:],
            "v_proj": _CATEGORIES[3:],
            "o_proj": _CATEGORIES[4:],
            "gate_proj": _CATEGORIES[5:],
            "up_proj": _CATEGORIES[6:],
            "down_proj": [],
        }
        for category in _CATEGORIES:
            setattr(model.model.layers[0], category, nn.Identity())
            remaining = collect_linears(
                model,
                transpose_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
                only_decoder_projections=True,
                target_categories=_CATEGORIES,
            )
            self.assertEqual(list(dict.fromkeys(ref.category for ref in remaining)), expected[category])

    def _args(self, *, final_norm: bool, post_norm: bool, steps="default=1"):
        cat_args = SimpleNamespace(
            distill_after_category="remaining_lora",
            distill_tune_final_norm=bool(final_norm),
            distill_use_post_norm_head_linear=bool(post_norm),
            distill_steps=steps,
        )
        training_args = SimpleNamespace()
        vae_args = SimpleNamespace()
        return cat_args, vae_args, training_args

    def test_down_extra_trainables_do_not_skip_when_remaining_linears_empty(self):
        model = _TinyModel()
        for category in _CATEGORIES:
            setattr(model.model.layers[0], category, nn.Identity())
        cat_args, vae_args, training_args = self._args(final_norm=True, post_norm=True)
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        with mock.patch(
            "train_utils.cat_after_category_distill.lora_finetune_remaining_categories",
            return_value=RemainingLoraFinetuneResult(model=model, did_train=True),
        ) as mocked:
            result = run_after_category_distill(
                model=model,
                category="down_proj",
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=logger,
                lora_round_idx=6,
                transpose_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
                only_decoder_projections=True,
                target_categories=_CATEGORIES,
            )

        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["target_names"], [])
        self.assertEqual(mocked.call_args.kwargs["remaining_categories"], [])
        self.assertTrue(result.did_train)
        self.assertEqual(result.trained_target_count, 0)
        self.assertEqual(result.next_lora_round_idx, 7)

    def test_down_without_extra_trainables_still_skips_when_remaining_linears_empty(self):
        model = _TinyModel()
        for category in _CATEGORIES:
            setattr(model.model.layers[0], category, nn.Identity())
        cat_args, vae_args, training_args = self._args(final_norm=False, post_norm=False)
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        with mock.patch(
            "train_utils.cat_after_category_distill.lora_finetune_remaining_categories"
        ) as mocked:
            result = run_after_category_distill(
                model=model,
                category="down_proj",
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=logger,
                lora_round_idx=6,
                transpose_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
                only_decoder_projections=True,
                target_categories=_CATEGORIES,
            )

        mocked.assert_not_called()
        self.assertFalse(result.did_train)
        self.assertEqual(result.trained_target_count, 0)
        self.assertEqual(result.next_lora_round_idx, 6)

    def test_steps_zero_result_does_not_increment_round(self):
        model = _TinyModel()
        setattr(model.model.layers[0], "q_proj", nn.Identity())
        cat_args, vae_args, training_args = self._args(final_norm=True, post_norm=True, steps="default=0")
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        with mock.patch(
            "train_utils.cat_after_category_distill.lora_finetune_remaining_categories",
            return_value=RemainingLoraFinetuneResult(model=model, did_train=False),
        ):
            result = run_after_category_distill(
                model=model,
                category="q_proj",
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=logger,
                lora_round_idx=0,
                transpose_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
                only_decoder_projections=True,
                target_categories=_CATEGORIES,
            )

        self.assertFalse(result.did_train)
        self.assertEqual(result.next_lora_round_idx, 0)
