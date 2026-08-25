import copy
import logging
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import List
from unittest import mock

import torch
from torch import nn

from e2e_common.compressed_subspace_lora import (
    CompressedSubspacePeftProxy,
    PeftZeroLinearCarrier,
    export_compressed_subspace_peft_lora_to_vae_low_rank,
    inject_compressed_subspace_peft_lora,
    iter_named_compressed_subspace_peft_proxies,
    wrap_vae_linears_with_compressed_subspace_peft_proxy,
)
from e2e_common.peft_proxy import is_peft_lora_linear
from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    LOW_RANK_SCOPE_FULL,
    VALID_LOW_RANK_SCOPES,
)
from litebsq.vae_linear import VAELinear
from train_utils import cat_after_category_distill as cad
from train_utils.cat_arg_overrides import OverrideTable
from train_utils.cat_after_category_distill import (
    _run_compressed_category_distill,
    _validate_existing_model_low_rank_scope,
)
from train_utils.cat_train_args import (
    _normalize_cat_train_script_args,
    build_cat_train_parser,
)
from train_utils.lora_utils import _ResolvedDistillStageConfig


def _make_decoder(latent_dim: int = 9, codebook_dim: int = 4) -> Decoder:
    decoder = Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    return decoder.to(dtype=torch.float32)


def _make_vq_bits(*, compressed_out: int, compressed_in: int, codebook_dim: int = 4, latent_dim: int = 9):
    expected = int(compressed_out) * int(compressed_in)
    n_blocks = expected // int(codebook_dim)
    rows = []
    for block_idx in range(n_blocks):
        pattern = [((block_idx + bit_idx) % 2) == 0 for bit_idx in range(latent_dim)]
        rows.append([pattern])
    return torch.tensor(rows, dtype=torch.bool)


def _build_vae_linear(
    *,
    in_features: int = 6,
    out_features: int = 4,
    compressed_in_features: int | None = None,
    low_rank_a: torch.Tensor | None = None,
    low_rank_b: torch.Tensor | None = None,
    low_rank_scope: str = LOW_RANK_SCOPE_FULL,
) -> VAELinear:
    compressed_in = int(in_features if compressed_in_features is None else compressed_in_features)
    kwargs = {}
    if compressed_in < in_features:
        protected = torch.arange(in_features - compressed_in, dtype=torch.long)
        kwargs["protected_input_indices"] = protected
        kwargs["protected_input_weight"] = torch.randn(int(protected.numel()), out_features)
    codebook_dim = 4
    bits = _make_vq_bits(
        compressed_out=out_features,
        compressed_in=compressed_in,
        codebook_dim=codebook_dim,
    )
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=copy.deepcopy(_make_decoder(codebook_dim=codebook_dim)),
        codebook_dim=codebook_dim,
        transpose=False,
        compressed_in_features=compressed_in,
        compressed_out_features=out_features,
        low_rank_a=low_rank_a,
        low_rank_b=low_rank_b,
        low_rank_scope=low_rank_scope,
        **kwargs,
    )


class _TinyCategoryModel(nn.Module):
    def __init__(self, modules: dict[str, VAELinear]):
        super().__init__()
        self.layers = nn.ModuleDict()
        for name, module in modules.items():
            self.layers[name] = module


def _override(name: str, value):
    return OverrideTable(
        arg_name=name,
        allowed_selectors=("default", "after"),
        has_default=True,
        default=value,
    )


def _cat_args_for_compressed(*, scope=LOW_RANK_SCOPE_FULL):
    return SimpleNamespace(
        compressed_lora_scope=scope,
        distill_reset_completed=False,
        seed=0,
        distill_dataset="dummy",
        train_device="cpu",
        lora_rank=_override("lora_rank", 2),
        lora_alpha=_override("lora_alpha", 2.0),
        lora_dropout=_override("lora_dropout", 0.0),
        lora_use_dora=_override("lora_use_dora", False),
        distill_steps=_override("distill_steps", 1),
        distill_batch_size=_override("distill_batch_size", 1),
        distill_lr=_override("distill_lr", 1e-4),
        distill_decoder_lr=_override("distill_decoder_lr", None),
        distill_weight_decay=_override("distill_weight_decay", 0.0),
        distill_log_every=_override("distill_log_every", 1),
        distill_temperature=_override("distill_temperature", 1.0),
        distill_loss_alpha=_override("distill_loss_alpha", 1.0),
        distill_loss_type=_override("distill_loss_type", "sft"),
        distill_hidden_loss_weight=_override("distill_hidden_loss_weight", 0.0),
        distill_pre_mlp_hidden_loss_weight=_override("distill_pre_mlp_hidden_loss_weight", 0.0),
        distill_prompt_kd_weight=_override("distill_prompt_kd_weight", 0.0),
        distill_hidden_alignment_layer_weighting="uniform",
        distill_eakld_confidence_k=16,
    )


def _fake_cfg(
    *,
    rank: int = 2,
    steps: int = 1,
    lr: float = 1e-4,
    decoder_lr=None,
) -> _ResolvedDistillStageConfig:
    return _ResolvedDistillStageConfig(
        device="cpu",
        base_seed=0,
        round_idx=0,
        seed=0,
        rank=int(rank),
        alpha=float(rank),
        dropout=0.0,
        steps=int(steps),
        batch_size=1,
        lr=float(lr),
        decoder_lr=decoder_lr,
        weight_decay=0.0,
        log_every=1,
        temperature=1.0,
        loss_alpha=1.0,
        loss_type="sft",
        hidden_loss_weight=0.0,
        pre_mlp_hidden_loss_weight=0.0,
        prompt_kd_weight=0.0,
        hidden_alignment_layer_weighting="uniform",
        eakld_confidence_k=1,
        dataset="dummy",
        use_dora=False,
        use_distill_hif4_act=False,
        distill_tune_final_norm=False,
        distill_use_post_norm_head_linear=False,
    )


def _patch_distill_pre_train(monkeypatch, *, calls: List[str]):
    monkeypatch.setattr(cad, "_resolve_distill_stage_config", lambda **_kwargs: _fake_cfg())
    monkeypatch.setattr(cad, "_ensure_lora_stack_available", lambda: None)
    monkeypatch.setattr(cad, "_ensure_lora_tokenizer_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        cad,
        "prepare_distill_datasets",
        lambda *_args, **_kwargs: ("mix", [], [{"input_ids": [1]}], None, None),
    )
    monkeypatch.setattr(cad, "is_iterable_training_dataset", lambda _ds: False)
    monkeypatch.setattr(cad, "dataset_length_or_none", lambda _ds: 1)
    monkeypatch.setattr(cad, "_freeze_model_for_lora", lambda *args, **kwargs: None)
    monkeypatch.setattr(cad, "_restore_model_use_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cad,
        "prime_model_vae_linear_cache",
        lambda *args, **kwargs: {"total": 0, "warmed": 0, "skipped": 0, "failed": 0},
    )
    monkeypatch.setattr(
        cad,
        "prime_named_vae_linear_cache",
        lambda *args, **kwargs: {"total": 0, "warmed": 0, "skipped": 0, "failed": 0},
    )
    monkeypatch.setattr(cad, "_set_proxy_decoder_adapter_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(cad, "_enable_compressed_lora_params_only", lambda *args, **kwargs: [])
    monkeypatch.setattr(cad, "_enable_subspace_compressed_lora_params_only", lambda *args, **kwargs: [])
    monkeypatch.setattr(cad, "_unwrap_peft_proxies_without_export", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        cad,
        "unwrap_compressed_subspace_peft_proxies",
        lambda *args, **kwargs: 0,
    )

    real_wrap_sub = cad.wrap_vae_linears_with_compressed_subspace_peft_proxy
    real_inject_sub = cad.inject_compressed_subspace_peft_lora

    def materialize(*_args, **_kwargs):
        calls.append("materialize")

    def ensure(*_args, **_kwargs):
        calls.append("ensure")
        return 1

    def wrap_full(_model, targets):
        calls.append("wrap_full")
        return [name for name, _ in targets]

    def wrap_sub(model, targets):
        calls.append("wrap_subspace")
        return real_wrap_sub(model, targets)

    def inject_sub(model, **kwargs):
        calls.append("inject_subspace")
        injected = real_inject_sub(model, **kwargs)
        for _name, proxy in iter_named_compressed_subspace_peft_proxies(model):
            carrier = proxy.compressed_subspace_adapter_linear
            assert is_peft_lora_linear(carrier)
            assert int(carrier.base_layer.weight.numel()) == 1
            assert isinstance(carrier.base_layer, PeftZeroLinearCarrier)
        return injected

    monkeypatch.setattr(cad, "materialize_peft_proxy_decoded_linears", materialize)
    monkeypatch.setattr(cad, "ensure_peft_vae_proxy_adapter", ensure)
    monkeypatch.setattr(cad, "_wrap_targets_as_peft_proxies", wrap_full)
    monkeypatch.setattr(cad, "wrap_vae_linears_with_compressed_subspace_peft_proxy", wrap_sub)
    monkeypatch.setattr(cad, "inject_compressed_subspace_peft_lora", inject_sub)


class CatCompressedLoraScopeParserTests(unittest.TestCase):
    def _parse_normalize(self, argv):
        parser = build_cat_train_parser()
        raw_args, _ = parser.parse_known_args(list(argv))
        return _normalize_cat_train_script_args(raw_args)

    def test_default_compressed_lora_scope_is_full(self):
        cat_args = self._parse_normalize([])
        self.assertEqual(cat_args.compressed_lora_scope, LOW_RANK_SCOPE_FULL)

    def test_explicit_full_compressed_lora_scope(self):
        cat_args = self._parse_normalize(["--compressed_lora_scope", "full"])
        self.assertEqual(cat_args.compressed_lora_scope, LOW_RANK_SCOPE_FULL)

    def test_explicit_subspace_compressed_lora_scope(self):
        cat_args = self._parse_normalize(["--compressed_lora_scope", "compressed_subspace"])
        self.assertEqual(cat_args.compressed_lora_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)

    def test_illegal_compressed_lora_scope_fails(self):
        parser = build_cat_train_parser()
        with self.assertRaises(SystemExit):
            parser.parse_known_args(["--compressed_lora_scope", "block"])

    def test_parser_choices_match_valid_scopes(self):
        parser = build_cat_train_parser()
        action = next(
            item for item in parser._actions if "--compressed_lora_scope" in getattr(item, "option_strings", [])
        )
        self.assertEqual(tuple(action.choices), tuple(sorted(VALID_LOW_RANK_SCOPES)))
        self.assertEqual(action.default, LOW_RANK_SCOPE_FULL)


class CatCompressedLoraScopeGuardTests(unittest.TestCase):
    def test_current_full_requested_full_allowed(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "down_proj": _build_vae_linear(
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 6),
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                    compressed_in_features=6,
                )
            }
        )
        stored = _validate_existing_model_low_rank_scope(
            model, requested_scope=LOW_RANK_SCOPE_FULL
        )
        self.assertEqual(stored, LOW_RANK_SCOPE_FULL)

    def test_current_subspace_requested_subspace_allowed(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 4),
                    low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
                )
            }
        )
        stored = _validate_existing_model_low_rank_scope(
            model, requested_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
        )
        self.assertEqual(stored, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)

    def test_current_full_requested_subspace_errors(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "down_proj": _build_vae_linear(
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 6),
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                    compressed_in_features=6,
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_existing_model_low_rank_scope(
                model, requested_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
            )

    def test_current_subspace_requested_full_errors(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 4),
                    low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_existing_model_low_rank_scope(
                model, requested_scope=LOW_RANK_SCOPE_FULL
            )

    def test_no_low_rank_fresh_subspace_allowed(self):
        model = _TinyCategoryModel(
            {
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                )
            }
        )
        stored = _validate_existing_model_low_rank_scope(
            model, requested_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
        )
        self.assertIsNone(stored)

    def test_previous_full_current_fresh_requested_subspace_errors_before_wrap(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "q_proj": _build_vae_linear(
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 6),
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                    compressed_in_features=6,
                ),
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                ),
            }
        )
        cat_args = _cat_args_for_compressed(scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        with self.assertRaisesRegex(ValueError, "Existing model low-rank scope"):
            _run_compressed_category_distill(
                model=model,
                category="down_proj",
                mode="compressed_lora",
                cat_args=cat_args,
                vae_args=SimpleNamespace(),
                training_args=SimpleNamespace(bf16=False),
                logger=logging.getLogger("test"),
                lora_round_idx=0,
            )

    def test_previous_subspace_current_fresh_requested_full_errors_before_wrap(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "q_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 4),
                    low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
                ),
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                ),
            }
        )
        cat_args = _cat_args_for_compressed(scope=LOW_RANK_SCOPE_FULL)
        with self.assertRaisesRegex(ValueError, "Existing model low-rank scope"):
            _run_compressed_category_distill(
                model=model,
                category="down_proj",
                mode="compressed_lora",
                cat_args=cat_args,
                vae_args=SimpleNamespace(),
                training_args=SimpleNamespace(bf16=False),
                logger=logging.getLogger("test"),
                lora_round_idx=0,
            )

    def test_mixed_scopes_error_regardless_of_requested(self):
        rank = 2
        model = _TinyCategoryModel(
            {
                "q_proj": _build_vae_linear(
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 6),
                    low_rank_scope=LOW_RANK_SCOPE_FULL,
                    compressed_in_features=6,
                ),
                "down_proj": _build_vae_linear(
                    compressed_in_features=4,
                    low_rank_a=torch.randn(4, rank),
                    low_rank_b=torch.randn(rank, 4),
                    low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "mixed low-rank scopes"):
            _validate_existing_model_low_rank_scope(
                model, requested_scope=LOW_RANK_SCOPE_FULL
            )
        with self.assertRaisesRegex(ValueError, "mixed low-rank scopes"):
            _validate_existing_model_low_rank_scope(
                model, requested_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
            )


def test_full_route_calls_materialize_not_subspace(monkeypatch):
    calls: List[str] = []
    _patch_distill_pre_train(monkeypatch, calls=calls)
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cat_args = _cat_args_for_compressed(scope=LOW_RANK_SCOPE_FULL)
    result = _run_compressed_category_distill(
        model=model,
        category="down_proj",
        mode="compressed_lora",
        cat_args=cat_args,
        vae_args=SimpleNamespace(_cached_lora_tokenizer=object()),
        training_args=SimpleNamespace(bf16=False, distill_model_max_length=32),
        logger=logging.getLogger("test"),
        lora_round_idx=0,
    )
    assert "materialize" in calls
    assert "ensure" in calls
    assert "wrap_full" in calls
    assert "wrap_subspace" not in calls
    assert "inject_subspace" not in calls
    assert int(result.trained_target_count) == 0


def test_subspace_route_calls_wrap_inject_not_materialize(monkeypatch):
    calls: List[str] = []
    _patch_distill_pre_train(monkeypatch, calls=calls)
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cat_args = _cat_args_for_compressed(scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
    result = _run_compressed_category_distill(
        model=model,
        category="down_proj",
        mode="compressed_lora",
        cat_args=cat_args,
        vae_args=SimpleNamespace(_cached_lora_tokenizer=object()),
        training_args=SimpleNamespace(bf16=False, distill_model_max_length=32),
        logger=logging.getLogger("test"),
        lora_round_idx=0,
    )
    assert "wrap_subspace" in calls
    assert "inject_subspace" in calls
    assert "materialize" not in calls
    assert "ensure" not in calls
    assert "wrap_full" not in calls
    assert int(result.trained_target_count) == 0


def test_compressed_no_target_skip_does_not_advance_round():
    result = _run_compressed_category_distill(
        model=_TinyCategoryModel({}),
        category="down_proj",
        mode="compressed_lora",
        cat_args=_cat_args_for_compressed(),
        vae_args=SimpleNamespace(),
        training_args=SimpleNamespace(bf16=False),
        logger=logging.getLogger("test"),
        lora_round_idx=4,
    )

    assert result.did_train is False
    assert int(result.trained_target_count) == 0
    assert int(result.next_lora_round_idx) == 4
    assert result.distill_meta["newly_compressed_target_count"] == 0


def test_compressed_steps_zero_skip_does_not_advance_round():
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cfg = replace(_fake_cfg(), steps=0)
    cat_args = SimpleNamespace(
        compressed_lora_scope=LOW_RANK_SCOPE_FULL,
        distill_reset_completed=False,
    )

    with mock.patch.object(cad, "_resolve_distill_stage_config", return_value=cfg):
        result = _run_compressed_category_distill(
            model=model,
            category="down_proj",
            mode="compressed_lora",
            cat_args=cat_args,
            vae_args=SimpleNamespace(),
            training_args=SimpleNamespace(bf16=False),
            logger=logging.getLogger("test"),
            lora_round_idx=7,
        )

    assert result.did_train is False
    assert int(result.trained_target_count) == 0
    assert int(result.next_lora_round_idx) == 7


def test_inline_compressed_mode_preserves_newly_compressed_count_when_steps_zero():
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cfg = _fake_cfg(steps=0)
    with mock.patch.object(cad, "_resolve_distill_stage_config", return_value=cfg):
        result = _run_compressed_category_distill(
            model=model,
            category="down_proj",
            mode="compressed_lora",
            cat_args=_cat_args_for_compressed(),
            vae_args=SimpleNamespace(),
            training_args=SimpleNamespace(bf16=False),
            logger=logging.getLogger("test"),
            lora_round_idx=7,
            newly_compressed_target_count=7,
        )

    assert result.did_train is False
    assert int(result.trained_target_count) == 7
    assert int(result.next_lora_round_idx) == 7
    assert result.distill_meta["newly_compressed_target_count"] == 7


def test_checkpoint_compressed_mode_count_remains_zero():
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cfg = _fake_cfg(steps=0)
    with mock.patch.object(cad, "_resolve_distill_stage_config", return_value=cfg):
        result = _run_compressed_category_distill(
            model=model,
            category="down_proj",
            mode="compressed_lora",
            cat_args=_cat_args_for_compressed(),
            vae_args=SimpleNamespace(),
            training_args=SimpleNamespace(bf16=False),
            logger=logging.getLogger("test"),
            lora_round_idx=7,
            newly_compressed_target_count=0,
        )

    assert int(result.trained_target_count) == 0
    assert result.distill_meta["newly_compressed_target_count"] == 0


def _patch_decoder_train(monkeypatch, *, cfg, captured):
    monkeypatch.setattr(cad, "_resolve_distill_stage_config", lambda **_kwargs: cfg)
    monkeypatch.setattr(cad, "_ensure_lora_stack_available", lambda: None)
    monkeypatch.setattr(cad, "_ensure_lora_tokenizer_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        cad,
        "prepare_distill_datasets",
        lambda *_args, **_kwargs: ("mix", [], [{"input_ids": [1]}], None, None),
    )
    monkeypatch.setattr(cad, "is_iterable_training_dataset", lambda _ds: False)
    monkeypatch.setattr(cad, "dataset_length_or_none", lambda _ds: 1)
    monkeypatch.setattr(cad, "_freeze_model_for_lora", lambda *args, **kwargs: None)
    monkeypatch.setattr(cad, "_restore_model_use_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cad,
        "prime_named_vae_linear_cache",
        lambda *args, **kwargs: {"total": 0, "warmed": 0, "skipped": 0, "failed": 0},
    )
    monkeypatch.setattr(cad, "_build_sft_args", lambda **_kwargs: object())

    class FakeTrainer:
        def __init__(self, model, decoder_params, decoder_lr):
            self.model = model
            self.optimizer = SimpleNamespace(
                param_groups=[
                    {
                        "lr": decoder_lr,
                        "weight_decay": 0.0,
                        "params": list(decoder_params),
                    }
                ]
            )

    def fake_build_lora_trainer(**kwargs):
        captured["decoder_lr"] = kwargs["decoder_lr"]
        captured["decoder_param_count"] = len(list(kwargs["decoder_params"]))
        trainer = FakeTrainer(kwargs["model"], kwargs["decoder_params"], kwargs["decoder_lr"])
        captured["optimizer_group_lr"] = trainer.optimizer.param_groups[0]["lr"]
        captured["optimizer_group_weight_decay"] = trainer.optimizer.param_groups[0]["weight_decay"]
        return trainer

    monkeypatch.setattr(cad, "_build_lora_trainer", fake_build_lora_trainer)
    monkeypatch.setattr(cad, "_train_without_merging_peft_adapters", lambda trainer, **_kwargs: trainer.model)


def test_decoder_metadata_uses_explicit_decoder_lr(monkeypatch):
    captured = {}
    cfg = _fake_cfg(lr=2e-5, decoder_lr=5e-5)
    _patch_decoder_train(monkeypatch, cfg=cfg, captured=captured)
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )

    result = _run_compressed_category_distill(
        model=model,
        category="down_proj",
        mode="decoder",
        cat_args=_cat_args_for_compressed(),
        vae_args=SimpleNamespace(_cached_lora_tokenizer=object()),
        training_args=SimpleNamespace(bf16=False, distill_model_max_length=32),
        logger=logging.getLogger("test"),
        lora_round_idx=0,
    )

    assert captured["decoder_param_count"] > 0
    assert captured["decoder_lr"] == 5e-5
    assert captured["optimizer_group_lr"] == 5e-5
    assert captured["optimizer_group_weight_decay"] == 0.0
    assert result.distill_meta["resolved_distill_lr"] == 2e-5
    assert result.distill_meta["resolved_decoder_lr"] == 5e-5
    assert result.distill_meta["decoder_weight_decay"] == 0.0


def test_decoder_metadata_inherits_distill_lr(monkeypatch):
    captured = {}
    cfg = _fake_cfg(lr=2e-5, decoder_lr=None)
    _patch_decoder_train(monkeypatch, cfg=cfg, captured=captured)
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )

    result = _run_compressed_category_distill(
        model=model,
        category="down_proj",
        mode="decoder",
        cat_args=_cat_args_for_compressed(),
        vae_args=SimpleNamespace(_cached_lora_tokenizer=object()),
        training_args=SimpleNamespace(bf16=False, distill_model_max_length=32),
        logger=logging.getLogger("test"),
        lora_round_idx=0,
    )

    assert captured["decoder_lr"] == 2e-5
    assert result.distill_meta["resolved_decoder_lr"] == 2e-5


def test_no_decoder_group_metadata_is_null():
    model = _TinyCategoryModel(
        {
            "down_proj": _build_vae_linear(
                compressed_in_features=4,
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
        }
    )
    cfg = _fake_cfg(steps=0, lr=2e-5, decoder_lr=5e-5)
    with mock.patch.object(cad, "_resolve_distill_stage_config", return_value=cfg):
        result = _run_compressed_category_distill(
            model=model,
            category="down_proj",
            mode="compressed_lora",
            cat_args=_cat_args_for_compressed(),
            vae_args=SimpleNamespace(),
            training_args=SimpleNamespace(bf16=False),
            logger=logging.getLogger("test"),
            lora_round_idx=0,
        )

    assert result.distill_meta["resolved_decoder_lr"] is None
    assert result.distill_meta["decoder_weight_decay"] is None


def test_subspace_export_restores_bare_vae_linear_with_compressed_scope():
    layer = _build_vae_linear(compressed_in_features=4, low_rank_scope=LOW_RANK_SCOPE_FULL)
    model = _TinyCategoryModel({"down_proj": layer})
    name = "layers.down_proj"
    wrap_vae_linears_with_compressed_subspace_peft_proxy(model, [(name, layer)])
    inject_compressed_subspace_peft_lora(model, rank=2, alpha=2.0, dropout=0.0)
    proxy = model.layers["down_proj"]
    assert isinstance(proxy, CompressedSubspacePeftProxy)
    carrier = proxy.compressed_subspace_adapter_linear
    with torch.no_grad():
        carrier.lora_A["default"].weight.copy_(torch.randn_like(carrier.lora_A["default"].weight))
        carrier.lora_B["default"].weight.copy_(torch.randn_like(carrier.lora_B["default"].weight))
    exported = export_compressed_subspace_peft_lora_to_vae_low_rank(
        model,
        module_names=[name],
        allow_overwrite=False,
    )
    assert int(exported) == 1
    restored = model.layers["down_proj"]
    assert isinstance(restored, VAELinear)
    assert not isinstance(restored, CompressedSubspacePeftProxy)
    assert restored.low_rank_scope == LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
    assert tuple(restored.low_rank_a.shape) == (4, 2)
    assert tuple(restored.low_rank_b.shape) == (2, 4)
    assert list(iter_named_compressed_subspace_peft_proxies(model)) == []


if __name__ == "__main__":
    unittest.main()
