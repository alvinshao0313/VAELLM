import copy
import logging
import tempfile
import unittest
from typing import List, Tuple
from unittest import mock

import peft
import torch
from peft import PeftModel
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from compressed_e2e_fintuning import runtime as e2e_runtime
from compressed_e2e_fintuning.runtime import (
    _build_subspace_low_rank_peft_model,
    _peft_base_model,
    _prepare_compressed_lora_train_model,
)
from compressed_e2e_fintuning.trainables import (
    select_vae_decoder_trainables,
    validate_selected_low_rank_scope,
)
from e2e_common.compressed_subspace_lora import (
    CompressedSubspacePeftProxy,
    PeftZeroLinearCarrier,
    extract_subspace_peft_low_rank_payloads,
    iter_named_compressed_subspace_peft_proxies,
)
from e2e_common.low_rank_lora import write_low_rank_payloads_to_compressed_model
from e2e_common.peft_proxy import is_peft_lora_linear
from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    LOW_RANK_SCOPE_FULL,
)
from litebsq.vae_linear import VAELinear


def _require_peft_010() -> None:
    assert peft.__version__ == "0.10.0", (
        f"Expected peft==0.10.0 from environment.yml, got {peft.__version__!r}."
    )


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


MODULE_NAME = "model.layers.0.mlp.down_proj"


class TinyE2EConfig(PretrainedConfig):
    model_type = "tiny_e2e_subspace"

    def __init__(self, hidden_size: int = 6, vocab_size: int = 13, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)


class TinyE2ECausalLM(PreTrainedModel):
    config_class = TinyE2EConfig

    def __init__(self, config: TinyE2EConfig, layer: VAELinear):
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.down_proj = layer
        self.embed = nn.Embedding(config.vocab_size, int(layer.in_features))
        self.lm_head = nn.Linear(int(layer.out_features), config.vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        hidden = self.model.layers[0].mlp.down_proj(x)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


def _make_full_selected(rank: int = 2) -> List[Tuple[str, VAELinear]]:
    layer = _build_vae_linear(
        in_features=4,
        out_features=4,
        low_rank_a=torch.randn(4, rank),
        low_rank_b=torch.randn(rank, 4),
        low_rank_scope=LOW_RANK_SCOPE_FULL,
    )
    return [(MODULE_NAME, layer)]


def _make_subspace_selected(
    *,
    rank: int = 2,
    in_features: int = 6,
    out_features: int = 4,
    compressed_in: int = 4,
) -> Tuple[List[Tuple[str, VAELinear]], torch.Tensor, torch.Tensor]:
    low_rank_a = torch.randn(out_features, rank)
    low_rank_b = torch.randn(rank, compressed_in)
    layer = _build_vae_linear(
        in_features=in_features,
        out_features=out_features,
        compressed_in_features=compressed_in,
        low_rank_a=low_rank_a.clone(),
        low_rank_b=low_rank_b.clone(),
        low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    )
    return [(MODULE_NAME, layer)], low_rank_a, low_rank_b


def _copy_payloads(selected_modules):
    return {
        name: (
            module.low_rank_a.detach().cpu().contiguous().clone(),
            module.low_rank_b.detach().cpu().contiguous().clone(),
        )
        for name, module in selected_modules
    }


class ValidateSelectedLowRankScopeTests(unittest.TestCase):
    def test_all_full_returns_full(self):
        selected = _make_full_selected()
        self.assertEqual(validate_selected_low_rank_scope(selected), LOW_RANK_SCOPE_FULL)

    def test_all_subspace_returns_compressed_subspace(self):
        selected, _, _ = _make_subspace_selected()
        self.assertEqual(
            validate_selected_low_rank_scope(selected),
            LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )

    def test_mixed_scope_raises(self):
        full = _make_full_selected()[0]
        subspace, _, _ = _make_subspace_selected()
        mixed = [full, ("model.layers.1.mlp.down_proj", subspace[0][1])]
        with self.assertRaisesRegex(ValueError, "mixed low-rank scopes"):
            validate_selected_low_rank_scope(mixed)


class CompressedLoraRouteTests(unittest.TestCase):
    def test_full_route_uses_materialize_and_full_builder(self):
        _require_peft_010()
        selected = _make_full_selected()
        model = TinyE2ECausalLM(TinyE2EConfig(hidden_size=4), selected[0][1])
        payloads = _copy_payloads(selected)
        calls = {"materialize": 0, "full_build": 0, "subspace_build": 0}
        log = logging.getLogger("test_e2e_full_route")

        def fake_materialize(*_args, **_kwargs):
            calls["materialize"] += 1
            return 1

        def fake_full_build(model_arg, **kwargs):
            calls["full_build"] += 1
            selection = mock.Mock()
            selection.target_modules = sorted(kwargs["low_rank_payloads"].keys())
            selection.train_mode = "compressed_lora"
            return model_arg, selection

        def fake_subspace_build(*_args, **_kwargs):
            calls["subspace_build"] += 1
            raise AssertionError("subspace builder must not be called for full scope")

        with mock.patch.object(
            e2e_runtime, "_materialize_selected_vae_linears_without_low_rank", side_effect=fake_materialize
        ), mock.patch.object(
            e2e_runtime, "_build_low_rank_peft_model", side_effect=fake_full_build
        ), mock.patch.object(
            e2e_runtime, "_build_subspace_low_rank_peft_model", side_effect=fake_subspace_build
        ):
            peft_model, selection = _prepare_compressed_lora_train_model(
                model,
                selected_modules=selected,
                target_module_suffixes=["down_proj"],
                low_rank_scope=LOW_RANK_SCOPE_FULL,
                low_rank_rank=2,
                low_rank_payloads=payloads,
                decoder_layer_ids=[0],
                parallel_stage_decode=False,
                decode_group_size=1,
                decode_device="cpu",
                log=log,
            )
        self.assertEqual(calls["materialize"], 1)
        self.assertEqual(calls["full_build"], 1)
        self.assertEqual(calls["subspace_build"], 0)
        self.assertIs(peft_model, model)
        self.assertEqual(selection.train_mode, "compressed_lora")

    def test_subspace_route_builds_root_peft_proxy_carrier(self):
        _require_peft_010()
        selected, low_rank_a, low_rank_b = _make_subspace_selected()
        model = TinyE2ECausalLM(TinyE2EConfig(), selected[0][1])
        payloads = _copy_payloads(selected)
        rank = int(low_rank_a.shape[1])
        log = logging.getLogger("test_e2e_subspace_route")
        calls = {"materialize": 0, "full_build": 0}

        def boom_materialize(*_args, **_kwargs):
            calls["materialize"] += 1
            raise AssertionError("full materialize must not be called for subspace")

        def boom_full_build(*_args, **_kwargs):
            calls["full_build"] += 1
            raise AssertionError("full PEFT builder must not be called for subspace")

        with mock.patch.object(
            e2e_runtime, "_materialize_selected_vae_linears_without_low_rank", side_effect=boom_materialize
        ), mock.patch.object(
            e2e_runtime, "_build_low_rank_peft_model", side_effect=boom_full_build
        ):
            peft_model, selection = _prepare_compressed_lora_train_model(
                model,
                selected_modules=selected,
                target_module_suffixes=["down_proj"],
                low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
                low_rank_rank=rank,
                low_rank_payloads=payloads,
                decoder_layer_ids=[0],
                parallel_stage_decode=False,
                decode_group_size=1,
                decode_device="cpu",
                log=log,
            )

        self.assertEqual(calls["materialize"], 0)
        self.assertEqual(calls["full_build"], 0)
        self.assertIsInstance(peft_model, PeftModel)
        self.assertIs(_peft_base_model(peft_model), peft_model.get_base_model())

        proxies = list(iter_named_compressed_subspace_peft_proxies(peft_model))
        self.assertEqual([name for name, _ in proxies], [MODULE_NAME])
        proxy = proxies[0][1]
        self.assertIsInstance(proxy, CompressedSubspacePeftProxy)
        carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME)
        self.assertTrue(is_peft_lora_linear(carrier))
        self.assertIsInstance(carrier.base_layer, PeftZeroLinearCarrier)
        self.assertEqual(int(carrier.base_layer.weight.numel()), 1)

        expected_trainable = int(low_rank_a.numel()) + int(low_rank_b.numel())
        self.assertEqual(int(selection.trainable_parameter_count), expected_trainable)
        self.assertEqual(selection.low_rank_modules, [MODULE_NAME])
        self.assertEqual(selection.train_mode, "compressed_lora")
        self.assertTrue(all("lora_" in name for name in selection.trainable_parameter_names))


class FinalExportRoundtripTests(unittest.TestCase):
    def test_subspace_export_roundtrip_updates_payload_keeps_protection(self):
        _require_peft_010()
        selected, source_a, source_b = _make_subspace_selected()
        source_layer = selected[0][1]
        protected_weight = source_layer.protected_input_weight.detach().clone()
        source_model = TinyE2ECausalLM(TinyE2EConfig(), source_layer)
        export_layer = copy.deepcopy(source_layer)
        export_model = TinyE2ECausalLM(TinyE2EConfig(), export_layer)

        payloads = _copy_payloads(selected)
        peft_model, selection = _build_subspace_low_rank_peft_model(
            source_model,
            selected_modules=selected,
            low_rank_payloads=payloads,
            rank=int(source_a.shape[1]),
            decoder_layer_ids=[0],
            target_module_suffixes=["down_proj"],
            parallel_stage_decode=False,
            log=logging.getLogger("test_e2e_export"),
        )

        proxy = dict(iter_named_compressed_subspace_peft_proxies(peft_model))[MODULE_NAME]
        carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME)
        with torch.no_grad():
            carrier.lora_A["default"].weight.add_(0.25)
            carrier.lora_B["default"].weight.add_(0.125)

        exported = extract_subspace_peft_low_rank_payloads(
            peft_model,
            module_names=selection.target_modules,
        )
        written = write_low_rank_payloads_to_compressed_model(
            export_model,
            exported,
            expected_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        self.assertEqual(int(written), 1)

        restored = export_model.model.layers[0].mlp.down_proj
        self.assertIsInstance(restored, VAELinear)
        self.assertEqual(restored.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        self.assertFalse(torch.allclose(restored.low_rank_a.cpu(), source_a, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(restored.low_rank_b.cpu(), source_b, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                restored.protected_input_weight.cpu(),
                protected_weight.cpu(),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(list(iter_named_compressed_subspace_peft_proxies(export_model)), [])
        self.assertFalse(any(is_peft_lora_linear(m) for _, m in export_model.named_modules()))

        x = torch.randn(2, 3, int(restored.in_features))
        y = restored(x)
        self.assertTrue(torch.isfinite(y).all())


class BothModeSubspaceTests(unittest.TestCase):
    def test_both_selects_decoder_and_subspace_low_rank_without_proxy(self):
        selected, _, _ = _make_subspace_selected(rank=2)
        host = TinyE2ECausalLM(TinyE2EConfig(), selected[0][1])
        scope = validate_selected_low_rank_scope(selected)
        self.assertEqual(scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)

        selection = select_vae_decoder_trainables(
            host,
            decoder_layer_ids=[0],
            target_module_names=["down_proj"],
            parallel_stage_decode=False,
            train_mode="both",
        )
        layer = host.model.layers[0].mlp.down_proj
        self.assertIsInstance(layer, VAELinear)
        self.assertNotIsInstance(layer, CompressedSubspacePeftProxy)
        self.assertTrue(bool(layer.low_rank_a.requires_grad))
        self.assertTrue(bool(layer.low_rank_b.requires_grad))
        decoder_trainable = any(
            bool(p.requires_grad)
            for p in layer.get_stage_part_decoder(stage_idx=0, part_idx=0).parameters()
        )
        self.assertTrue(decoder_trainable)
        self.assertEqual(selection.train_mode, "both")
        self.assertIn(MODULE_NAME, selection.low_rank_modules)
        self.assertEqual(list(iter_named_compressed_subspace_peft_proxies(host)), [])


class SubspaceResumeTests(unittest.TestCase):
    def test_trainer_resume_continues_from_checkpoint_not_source_init(self):
        _require_peft_010()
        selected, source_a, source_b = _make_subspace_selected()
        source_layer = selected[0][1]
        payloads = _copy_payloads(selected)
        model = TinyE2ECausalLM(TinyE2EConfig(), source_layer)
        peft_model, selection = _build_subspace_low_rank_peft_model(
            model,
            selected_modules=selected,
            low_rank_payloads=payloads,
            rank=int(source_a.shape[1]),
            decoder_layer_ids=[0],
            target_module_suffixes=["down_proj"],
            parallel_stage_decode=False,
            log=logging.getLogger("test_e2e_resume_build"),
        )

        class _ToyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                ids = torch.randint(0, 13, (4,), dtype=torch.long)
                return {"input_ids": ids, "labels": ids.clone()}

        def _rebuild_from_source_payloads() -> PeftModel:
            fresh_selected, _, _ = _make_subspace_selected()
            fresh_selected[0][1].low_rank_a.data.copy_(source_a)
            fresh_selected[0][1].low_rank_b.data.copy_(source_b)
            fresh_model = TinyE2ECausalLM(TinyE2EConfig(), fresh_selected[0][1])
            fresh_payloads = _copy_payloads(fresh_selected)
            rebuilt, _selection = _build_subspace_low_rank_peft_model(
                fresh_model,
                selected_modules=fresh_selected,
                low_rank_payloads=fresh_payloads,
                rank=int(source_a.shape[1]),
                decoder_layer_ids=[0],
                target_module_suffixes=["down_proj"],
                parallel_stage_decode=False,
                log=logging.getLogger("test_e2e_resume_rebuild"),
            )
            return rebuilt

        def _carrier_ab(peft_obj: PeftModel):
            proxy = dict(iter_named_compressed_subspace_peft_proxies(peft_obj))[MODULE_NAME]
            carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME)
            return (
                carrier.lora_A["default"].weight.detach().cpu().clone(),
                carrier.lora_B["default"].weight.detach().cpu().clone(),
            )

        def _training_args(tmp: str, *, max_steps: int, save_steps: int | None) -> TrainingArguments:
            kwargs = dict(
                output_dir=tmp,
                max_steps=int(max_steps),
                logging_strategy="no",
                report_to=[],
                per_device_train_batch_size=1,
                learning_rate=1e-2,
                lr_scheduler_type="constant",
                remove_unused_columns=False,
                save_safetensors=False,
                use_cpu=True,
                dataloader_num_workers=0,
                label_names=["labels"],
            )
            if save_steps is None:
                kwargs["save_strategy"] = "no"
            else:
                kwargs["save_strategy"] = "steps"
                kwargs["save_steps"] = int(save_steps)
            return TrainingArguments(**kwargs)

        def _unwrap_trainer_model(trainer: Trainer) -> PeftModel:
            model_obj = trainer.model
            if getattr(trainer, "accelerator", None) is not None:
                model_obj = trainer.accelerator.unwrap_model(model_obj)
            elif hasattr(model_obj, "module"):
                model_obj = model_obj.module
            return model_obj

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=peft_model,
                args=_training_args(tmp, max_steps=1, save_steps=1),
                train_dataset=_ToyDataset(),
            )
            trainer.train()
            self.assertEqual(int(trainer.state.global_step), 1)
            ckpt_dir = f"{tmp}/checkpoint-1"
            step1_a, step1_b = _carrier_ab(_unwrap_trainer_model(trainer))

            # Fresh isomorphic reconstruction from the same source subspace payloads.
            load_model = _rebuild_from_source_payloads()
            source_init_a, _source_init_b = _carrier_ab(load_model)
            self.assertTrue(torch.allclose(source_init_a, source_b, atol=1e-5, rtol=1e-5))
            load_trainer = Trainer(
                model=load_model,
                args=_training_args(tmp, max_steps=2, save_steps=None),
                train_dataset=_ToyDataset(),
            )
            # Load checkpoint weights before continuing; A/B must match step1, not source init.
            load_trainer._load_from_checkpoint(ckpt_dir)
            loaded_a, loaded_b = _carrier_ab(_unwrap_trainer_model(load_trainer))
            self.assertTrue(torch.allclose(loaded_a, step1_a, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(loaded_b, step1_b, atol=1e-5, rtol=1e-5))
            self.assertFalse(torch.allclose(loaded_a, source_init_a, atol=1e-5, rtol=1e-5))

            # Resume training from the same checkpoint through step 2.
            resumed_model = _rebuild_from_source_payloads()
            resume_trainer = Trainer(
                model=resumed_model,
                args=_training_args(tmp, max_steps=2, save_steps=2),
                train_dataset=_ToyDataset(),
            )
            resume_trainer.train(resume_from_checkpoint=ckpt_dir)
            self.assertEqual(int(resume_trainer.state.global_step), 2)
            step2_model = _unwrap_trainer_model(resume_trainer)
            step2_a, step2_b = _carrier_ab(step2_model)
            changed_from_step1 = (not torch.allclose(step2_a, step1_a, atol=1e-8, rtol=1e-8)) or (
                not torch.allclose(step2_b, step1_b, atol=1e-8, rtol=1e-8)
            )
            self.assertTrue(changed_from_step1)

            exported = extract_subspace_peft_low_rank_payloads(
                step2_model,
                module_names=selection.target_modules,
            )
            export_layer = _build_vae_linear(
                in_features=6,
                out_features=4,
                compressed_in_features=4,
                low_rank_a=source_a.clone(),
                low_rank_b=source_b.clone(),
                low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
            )
            export_host = TinyE2ECausalLM(TinyE2EConfig(), export_layer)
            written = write_low_rank_payloads_to_compressed_model(
                export_host,
                exported,
                expected_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
            )
            self.assertEqual(int(written), 1)
            restored = export_host.model.layers[0].mlp.down_proj
            y = restored(torch.randn(1, 2, 6))
            self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
