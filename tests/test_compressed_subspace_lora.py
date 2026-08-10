import copy
import json
import os
import tempfile
import unittest

import peft
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.mapping import inject_adapter_in_model
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from e2e_common.compressed_subspace_lora import (
    CompressedSubspacePeftProxy,
    PeftZeroLinearCarrier,
    export_compressed_subspace_peft_lora_to_vae_low_rank,
    extract_subspace_peft_low_rank_payloads,
    initialize_subspace_peft_lora_from_low_rank,
    inject_compressed_subspace_peft_lora,
    iter_named_compressed_subspace_peft_proxies,
    wrap_vae_linears_with_compressed_subspace_peft_proxy,
)
from e2e_common.peft_proxy import (
    detach_and_clear_vae_low_rank_payloads,
    ensure_peft_vae_linear_proxy,
    ensure_peft_vae_proxy_adapter,
    export_peft_proxy_lora_to_low_rank,
    is_peft_lora_linear,
)
from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    LOW_RANK_SCOPE_FULL,
)
from litebsq.vae_linear import VAELinear
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    _collect_vae_linear_specs,
    load_checkpoint_into_model,
    save_model_checkpoint,
)


def _require_peft_010() -> None:
    assert peft.__version__ == "0.10.0", (
        f"Expected peft==0.10.0 from environment.yml, got {peft.__version__!r}. "
        "Restore the pinned bitvae environment before continuing."
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
    decoder = decoder.to(dtype=torch.float32)
    with torch.no_grad():
        for idx, param in enumerate(decoder.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values + float(idx + 1))
    return decoder


def _make_vq_bits(*, compressed_out: int, compressed_in: int, codebook_dim: int = 4, latent_dim: int = 9) -> torch.Tensor:
    expected = int(compressed_out) * int(compressed_in)
    if expected % int(codebook_dim) != 0:
        raise ValueError(
            f"compressed_out*compressed_in={expected} not divisible by codebook_dim={codebook_dim}"
        )
    n_blocks = expected // int(codebook_dim)
    rows = []
    for block_idx in range(n_blocks):
        pattern = [((block_idx + bit_idx) % 2) == 0 for bit_idx in range(latent_dim)]
        rows.append([pattern])
    return torch.tensor(rows, dtype=torch.bool)


def _build_vae_linear(
    *,
    in_features: int,
    out_features: int,
    compressed_in_features: int | None = None,
    compressed_out_features: int | None = None,
    protected_input_indices: torch.Tensor | None = None,
    protected_output_indices: torch.Tensor | None = None,
    low_rank_a: torch.Tensor | None = None,
    low_rank_b: torch.Tensor | None = None,
    low_rank_scope: str = LOW_RANK_SCOPE_FULL,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> VAELinear:
    device = torch.device("cpu") if device is None else device
    compressed_in = int(in_features if compressed_in_features is None else compressed_in_features)
    compressed_out = int(out_features if compressed_out_features is None else compressed_out_features)
    codebook_dim = 4
    bits = _make_vq_bits(
        compressed_out=compressed_out,
        compressed_in=compressed_in,
        codebook_dim=codebook_dim,
    )
    decoder = _make_decoder(latent_dim=9, codebook_dim=codebook_dim)

    kwargs: dict = {}
    if protected_input_indices is not None:
        protected_count = int(protected_input_indices.numel())
        kwargs["protected_input_indices"] = protected_input_indices.to(dtype=torch.long)
        kwargs["protected_input_weight"] = torch.randn(
            protected_count,
            out_features,
            dtype=dtype,
            device=device,
        )
    if protected_output_indices is not None:
        protected_out_count = int(protected_output_indices.numel())
        kwargs["protected_output_indices"] = protected_output_indices.to(dtype=torch.long)
        kwargs["protected_output_weight"] = torch.randn(
            protected_out_count,
            in_features,
            dtype=dtype,
            device=device,
        )

    layer = VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=copy.deepcopy(decoder),
        codebook_dim=codebook_dim,
        transpose=False,
        compressed_in_features=compressed_in,
        compressed_out_features=compressed_out,
        low_rank_a=low_rank_a,
        low_rank_b=low_rank_b,
        low_rank_scope=low_rank_scope,
        **kwargs,
    )
    return layer.to(device=device, dtype=dtype)


class TinyCarrierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.compressed_subspace_adapter_linear = PeftZeroLinearCarrier(
            7,
            5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


class TinyCarrierConfig(PretrainedConfig):
    model_type = "tiny_carrier"

    def __init__(self, hidden_size: int = 7, vocab_size: int = 11, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)


class TinyCarrierCausalLM(PreTrainedModel):
    config_class = TinyCarrierConfig

    def __init__(self, config: TinyCarrierConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.compressed_subspace_adapter_linear = PeftZeroLinearCarrier(
            7,
            5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.lm_head = nn.Linear(5, config.vocab_size, bias=False)

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        hidden = self.compressed_subspace_adapter_linear(x)
        return CausalLMOutput(logits=self.lm_head(hidden))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class TinyProxyHost(nn.Module):
    def __init__(self, layer: VAELinear):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.down_proj = layer


class CompressedSubspaceLoRATests(unittest.TestCase):
    def test_full_low_rank_scope_is_default(self):
        layer = _build_vae_linear(in_features=4, out_features=4)
        self.assertEqual(layer.low_rank_scope, LOW_RANK_SCOPE_FULL)

    def test_full_low_rank_scope_keeps_existing_full_shape_contract(self):
        rank = 2
        low_rank_a = torch.randn(4, rank, dtype=torch.float32)
        low_rank_b = torch.randn(rank, 4, dtype=torch.float32)
        layer = _build_vae_linear(
            in_features=4,
            out_features=4,
            low_rank_a=low_rank_a,
            low_rank_b=low_rank_b,
            low_rank_scope=LOW_RANK_SCOPE_FULL,
        )
        self.assertEqual(tuple(layer.low_rank_a.shape), (4, rank))
        self.assertEqual(tuple(layer.low_rank_b.shape), (rank, 4))
        with self.assertRaises(ValueError):
            _build_vae_linear(
                in_features=4,
                out_features=4,
                low_rank_a=torch.randn(3, rank),
                low_rank_b=torch.randn(rank, 4),
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )

    def test_subspace_input_protection_keeps_protected_columns_unchanged(self):
        protected = torch.tensor([1, 4], dtype=torch.long)
        rank = 2
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=protected,
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        compressed = torch.randn(4, 4, dtype=torch.float32)
        w_base = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=False,
        )
        layer.low_rank_a = nn.Parameter(torch.randn(4, rank), requires_grad=False)
        layer.low_rank_b = nn.Parameter(torch.randn(rank, 4), requires_grad=False)
        layer._validate_low_rank_payload_tensors(
            layer.low_rank_a,
            layer.low_rank_b,
            scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        w_final = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=True,
        )
        self.assertTrue(torch.allclose(w_final[:, protected], w_base[:, protected], atol=1e-6, rtol=1e-6))
        keep = [i for i in range(6) if i not in set(protected.tolist())]
        changed = False
        for col in keep:
            if not torch.allclose(w_final[:, col], w_base[:, col], atol=1e-6, rtol=1e-6):
                changed = True
                break
        self.assertTrue(changed)

    def test_subspace_output_protection_keeps_protected_rows_unchanged(self):
        protected = torch.tensor([0, 3], dtype=torch.long)
        rank = 2
        layer = _build_vae_linear(
            in_features=4,
            out_features=6,
            compressed_out_features=4,
            protected_output_indices=protected,
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        compressed = torch.randn(4, 4, dtype=torch.float32)
        w_base = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=False,
        )
        layer.low_rank_a = nn.Parameter(torch.randn(4, rank), requires_grad=False)
        layer.low_rank_b = nn.Parameter(torch.randn(rank, 4), requires_grad=False)
        w_final = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=True,
        )
        self.assertTrue(torch.allclose(w_final[protected, :], w_base[protected, :], atol=1e-6, rtol=1e-6))
        keep = [i for i in range(6) if i not in set(protected.tolist())]
        changed = False
        for row in keep:
            if not torch.allclose(w_final[row, :], w_base[row, :], atol=1e-6, rtol=1e-6):
                changed = True
                break
        self.assertTrue(changed)

    def test_subspace_no_protection_matches_full_numerically(self):
        rank = 2
        low_rank_a = torch.randn(4, rank, dtype=torch.float32)
        low_rank_b = torch.randn(rank, 4, dtype=torch.float32)
        compressed = torch.randn(4, 4, dtype=torch.float32)
        full_layer = _build_vae_linear(
            in_features=4,
            out_features=4,
            low_rank_a=low_rank_a.clone(),
            low_rank_b=low_rank_b.clone(),
            low_rank_scope=LOW_RANK_SCOPE_FULL,
        )
        sub_layer = _build_vae_linear(
            in_features=4,
            out_features=4,
            low_rank_a=low_rank_a.clone(),
            low_rank_b=low_rank_b.clone(),
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        w_full = full_layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=True,
        )
        w_sub = sub_layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=True,
        )
        self.assertTrue(torch.allclose(w_full, w_sub, atol=1e-6, rtol=1e-6))

    def test_full_scope_still_allows_delta_on_protected_coordinates(self):
        protected = torch.tensor([1, 4], dtype=torch.long)
        rank = 2
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=protected,
            low_rank_scope=LOW_RANK_SCOPE_FULL,
        )
        compressed = torch.randn(4, 4, dtype=torch.float32)
        w_base = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=False,
        )
        low_rank_a = torch.zeros(4, rank, dtype=torch.float32)
        low_rank_b = torch.zeros(rank, 6, dtype=torch.float32)
        low_rank_a[:, 0] = 1.0
        low_rank_b[0, protected] = 2.0
        layer.low_rank_a = nn.Parameter(low_rank_a, requires_grad=False)
        layer.low_rank_b = nn.Parameter(low_rank_b, requires_grad=False)
        w_final = layer._finalize_decoded_weight_from_compressed(
            compressed.clone(),
            torch.float32,
            include_low_rank=True,
        )
        self.assertFalse(
            torch.allclose(w_final[:, protected], w_base[:, protected], atol=1e-6, rtol=1e-6)
        )

    def test_invalid_subspace_low_rank_shape_fails(self):
        with self.assertRaises(ValueError):
            _build_vae_linear(
                in_features=6,
                out_features=4,
                compressed_in_features=4,
                protected_input_indices=torch.tensor([1, 4], dtype=torch.long),
                low_rank_a=torch.randn(4, 2),
                low_rank_b=torch.randn(2, 6),
                low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
            )

    def test_peft_zero_carrier_has_constant_base_storage(self):
        carrier = PeftZeroLinearCarrier(11, 13, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(int(carrier.weight.numel()), 1)
        self.assertEqual(tuple(carrier.weight.shape), (1, 1))
        self.assertEqual(int(carrier.in_features), 11)
        self.assertEqual(int(carrier.out_features), 13)
        self.assertFalse(bool(carrier.weight.requires_grad))
        x = torch.randn(2, 11)
        y = carrier(x)
        self.assertEqual(tuple(y.shape), (2, 13))
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))

    def test_peft_zero_carrier_inject_adapter_gate(self):
        _require_peft_010()
        model = TinyCarrierModel()
        inject_adapter_in_model(
            LoraConfig(
                task_type=None,
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                target_modules=["compressed_subspace_adapter_linear"],
                bias="none",
                inference_mode=False,
                init_lora_weights=True,
            ),
            model,
        )
        lin = model.compressed_subspace_adapter_linear
        self.assertTrue(is_peft_lora_linear(lin))
        base = lin.base_layer
        self.assertIsInstance(base, PeftZeroLinearCarrier)
        self.assertEqual(tuple(base.weight.shape), (1, 1))
        self.assertEqual(int(base.weight.numel()), 1)
        self.assertEqual(int(base.in_features), 7)
        self.assertEqual(int(base.out_features), 5)
        self.assertEqual(tuple(lin.lora_A["default"].weight.shape), (2, 7))
        self.assertEqual(tuple(lin.lora_B["default"].weight.shape), (5, 2))
        x = torch.randn(3, 7)
        y0 = lin(x)
        self.assertTrue(torch.allclose(y0, torch.zeros_like(y0)))
        with torch.no_grad():
            lin.lora_A["default"].weight.copy_(torch.randn_like(lin.lora_A["default"].weight))
            lin.lora_B["default"].weight.copy_(torch.randn_like(lin.lora_B["default"].weight))
        x = torch.randn(3, 7, requires_grad=True)
        y = lin(x)
        scaling = float(lin.scaling["default"])
        expected = (x @ lin.lora_A["default"].weight.T @ lin.lora_B["default"].weight.T) * scaling
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-6))
        y.sum().backward()
        self.assertTrue(torch.isfinite(lin.lora_A["default"].weight.grad).all())
        self.assertTrue(torch.isfinite(lin.lora_B["default"].weight.grad).all())
        self.assertIsNone(base.weight.grad)
        self.assertTrue(torch.equal(base.weight.detach(), torch.zeros_like(base.weight)))

    def test_peft_zero_carrier_get_peft_model_gate(self):
        _require_peft_010()
        model = TinyCarrierCausalLM(TinyCarrierConfig())
        peft_model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=2,
                target_modules=["compressed_subspace_adapter_linear"],
                lora_alpha=2,
                lora_dropout=0.0,
                bias="none",
                init_lora_weights=True,
            ),
        )
        base_model = peft_model.get_base_model()
        lin = base_model.compressed_subspace_adapter_linear
        self.assertTrue(is_peft_lora_linear(lin))
        self.assertIsInstance(lin.base_layer, PeftZeroLinearCarrier)
        self.assertEqual(int(lin.base_layer.weight.numel()), 1)
        self.assertEqual(tuple(lin.lora_A["default"].weight.shape), (2, 7))
        self.assertEqual(tuple(lin.lora_B["default"].weight.shape), (5, 2))
        ids = torch.randint(0, 11, (2, 3))
        out = peft_model(input_ids=ids)
        out.logits.sum().backward()
        self.assertIsNotNone(peft_model.get_base_model())

    def test_subspace_peft_initial_delta_is_zero(self):
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=torch.tensor([1, 4], dtype=torch.long),
        )
        host = TinyProxyHost(layer)
        wrap_vae_linears_with_compressed_subspace_peft_proxy(
            host,
            [("model.layers.0.mlp.down_proj", layer)],
        )
        inject_compressed_subspace_peft_lora(host, rank=2, alpha=2.0, dropout=0.0)
        proxy = host.model.layers[0].mlp.down_proj
        x = torch.randn(3, 6, dtype=torch.float32)
        base_out = proxy.base_layer(x)
        proxy_out = proxy(x)
        self.assertTrue(torch.allclose(proxy_out, base_out, atol=1e-6, rtol=1e-6))

    def test_subspace_peft_forward_matches_exported_vae_linear(self):
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=torch.tensor([1, 4], dtype=torch.long),
        )
        host = TinyProxyHost(layer)
        name = "model.layers.0.mlp.down_proj"
        wrap_vae_linears_with_compressed_subspace_peft_proxy(host, [(name, layer)])
        inject_compressed_subspace_peft_lora(host, rank=2, alpha=2.0, dropout=0.0)
        proxy = host.model.layers[0].mlp.down_proj
        carrier = proxy.compressed_subspace_adapter_linear
        with torch.no_grad():
            carrier.lora_A["default"].weight.copy_(torch.randn_like(carrier.lora_A["default"].weight))
            carrier.lora_B["default"].weight.copy_(torch.randn_like(carrier.lora_B["default"].weight))
        x = torch.randn(5, 6, dtype=torch.float32)
        proxy.eval()
        y_proxy = proxy(x)
        export_compressed_subspace_peft_lora_to_vae_low_rank(
            host,
            module_names=[name],
            allow_overwrite=False,
        )
        exported = host.model.layers[0].mlp.down_proj
        self.assertIsInstance(exported, VAELinear)
        self.assertEqual(exported.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        exported.eval()
        y_exported = exported(x)
        self.assertTrue(torch.allclose(y_proxy, y_exported, atol=1e-6, rtol=1e-6))

    def test_subspace_peft_payload_restore_export_roundtrip(self):
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=torch.tensor([1, 4], dtype=torch.long),
        )
        host = TinyProxyHost(layer)
        name = "model.layers.0.mlp.down_proj"
        wrap_vae_linears_with_compressed_subspace_peft_proxy(host, [(name, layer)])
        inject_compressed_subspace_peft_lora(host, rank=2, alpha=4.0, dropout=0.0)
        low_rank_a = torch.randn(4, 2, dtype=torch.float32)
        low_rank_b = torch.randn(2, 4, dtype=torch.float32)
        initialize_subspace_peft_lora_from_low_rank(
            host,
            {name: (low_rank_a, low_rank_b)},
            module_names=[name],
        )
        payloads = extract_subspace_peft_low_rank_payloads(host, module_names=[name])
        got_a, got_b = payloads[name]
        self.assertTrue(torch.allclose(got_a, low_rank_a, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(got_b, low_rank_b, atol=1e-6, rtol=1e-6))
        export_compressed_subspace_peft_lora_to_vae_low_rank(
            host,
            module_names=[name],
            allow_overwrite=False,
        )
        exported = host.model.layers[0].mlp.down_proj
        self.assertTrue(torch.allclose(exported.low_rank_a.cpu(), low_rank_a, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(exported.low_rank_b.cpu(), low_rank_b, atol=1e-6, rtol=1e-6))

    def test_subspace_proxy_root_resolves_original_names_under_root_peft_model(self):
        _require_peft_010()
        layer = _build_vae_linear(
            in_features=4,
            out_features=4,
        )
        host = TinyProxyHost(layer)
        name = "model.layers.0.mlp.down_proj"
        wrap_vae_linears_with_compressed_subspace_peft_proxy(host, [(name, layer)])

        class TinyHostConfig(PretrainedConfig):
            model_type = "tiny_host"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.vocab_size = 11
                self.hidden_size = 4

        class TinyHostCausalLM(PreTrainedModel):
            config_class = TinyHostConfig

            def __init__(self, config, inner: nn.Module):
                super().__init__(config)
                self.model = inner.model
                self.lm_head = nn.Linear(4, config.vocab_size, bias=False)

            def forward(self, input_ids=None, **kwargs):
                x = torch.randn(input_ids.shape[0], input_ids.shape[1], 4)
                hidden = self.model.layers[0].mlp.down_proj(x)
                return CausalLMOutput(logits=self.lm_head(hidden))

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids}

        causal = TinyHostCausalLM(TinyHostConfig(), host)
        peft_model = get_peft_model(
            causal,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=2,
                target_modules=[CompressedSubspacePeftProxy.CARRIER_NAME],
                lora_alpha=2,
                lora_dropout=0.0,
                bias="none",
                init_lora_weights=True,
            ),
        )
        names = [n for n, _ in iter_named_compressed_subspace_peft_proxies(peft_model)]
        self.assertEqual(names, [name])
        carrier = peft_model.get_base_model().model.layers[0].mlp.down_proj.compressed_subspace_adapter_linear
        self.assertTrue(is_peft_lora_linear(carrier))

    def test_subspace_proxy_uses_base_device_and_dtype(self):
        layer = _build_vae_linear(in_features=4, out_features=4, dtype=torch.float32)
        # Move a floating parameter/buffer reference via decoder weights.
        layer = layer.to(dtype=torch.bfloat16)
        proxy = CompressedSubspacePeftProxy(layer)
        carrier = proxy.compressed_subspace_adapter_linear
        self.assertEqual(carrier.weight.dtype, torch.bfloat16)
        self.assertEqual(carrier.weight.device.type, "cpu")
        self.assertEqual(int(carrier.weight.numel()), 1)

    def test_full_export_sets_scope_full_and_detach_preserves_scope(self):
        rank = 2
        subspace_layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=torch.tensor([1, 4], dtype=torch.long),
            low_rank_a=torch.randn(4, rank, dtype=torch.float32),
            low_rank_b=torch.randn(rank, 4, dtype=torch.float32),
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        payloads = detach_and_clear_vae_low_rank_payloads([("proj", subspace_layer)])
        self.assertIn("proj", payloads)
        self.assertEqual(subspace_layer.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        self.assertIsNone(getattr(subspace_layer, "low_rank_a", None))
        self.assertIsNone(getattr(subspace_layer, "low_rank_b", None))

        full_layer = _build_vae_linear(in_features=4, out_features=4)
        full_layer.low_rank_scope = LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
        host = TinyProxyHost(full_layer)
        name = "model.layers.0.mlp.down_proj"
        ensure_peft_vae_linear_proxy(host, name, full_layer)
        ensure_peft_vae_proxy_adapter(
            host,
            variant="plain",
            rank=rank,
            alpha=float(rank),
            dropout=0.0,
            init_mode="gaussian",
        )
        exported = export_peft_proxy_lora_to_low_rank(host, module_names=[name])
        self.assertEqual(int(exported), 1)
        restored = host.model.layers[0].mlp.down_proj
        self.assertIsInstance(restored, VAELinear)
        self.assertEqual(restored.low_rank_scope, LOW_RANK_SCOPE_FULL)
        self.assertEqual(tuple(restored.low_rank_a.shape), (4, rank))
        self.assertEqual(tuple(restored.low_rank_b.shape), (rank, 4))

    def test_tiny_one_step_subspace_peft_smoke(self):
        """Plan §23: channel protect -> proxy/carrier -> PEFT -> train step -> export -> ckpt."""
        _require_peft_010()
        torch.manual_seed(0)

        protected = torch.tensor([1, 4], dtype=torch.long)
        rank = 2
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=protected,
        )
        host = TinyProxyHost(layer)
        name = "model.layers.0.mlp.down_proj"
        wrap_vae_linears_with_compressed_subspace_peft_proxy(host, [(name, layer)])
        injected = inject_compressed_subspace_peft_lora(
            host, rank=rank, alpha=float(rank), dropout=0.0
        )
        self.assertEqual(int(injected), 1)

        proxy = host.model.layers[0].mlp.down_proj
        self.assertIsInstance(proxy, CompressedSubspacePeftProxy)
        carrier = proxy.compressed_subspace_adapter_linear
        self.assertTrue(is_peft_lora_linear(carrier))
        base_carrier = carrier.base_layer
        self.assertIsInstance(base_carrier, PeftZeroLinearCarrier)
        self.assertEqual(int(base_carrier.weight.numel()), 1)
        self.assertFalse(bool(base_carrier.weight.requires_grad))
        self.assertTrue(torch.equal(base_carrier.weight.detach(), torch.zeros_like(base_carrier.weight)))

        adapter_name = "default"
        lora_a = carrier.lora_A[adapter_name].weight
        lora_b = carrier.lora_B[adapter_name].weight
        # Ensure non-zero LoRA so grads flow on the first step (PEFT default keeps B=0).
        with torch.no_grad():
            lora_a.copy_(torch.randn_like(lora_a))
            lora_b.copy_(torch.randn_like(lora_b) * 0.1)

        for param in host.parameters():
            param.requires_grad_(False)
        lora_a.requires_grad_(True)
        lora_b.requires_grad_(True)

        before_a = lora_a.detach().clone()
        before_b = lora_b.detach().clone()
        optimizer = torch.optim.SGD([lora_a, lora_b], lr=0.05)

        x = torch.randn(5, 6, dtype=torch.float32)
        target = torch.randn(5, 4, dtype=torch.float32)
        host.train()
        optimizer.zero_grad(set_to_none=True)
        pred = proxy(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        self.assertIsNotNone(lora_a.grad)
        self.assertIsNotNone(lora_b.grad)
        self.assertTrue(torch.isfinite(lora_a.grad).all())
        self.assertTrue(torch.isfinite(lora_b.grad).all())
        self.assertIsNone(base_carrier.weight.grad)
        self.assertEqual(int(base_carrier.weight.numel()), 1)
        self.assertFalse(bool(base_carrier.weight.requires_grad))
        self.assertTrue(torch.equal(base_carrier.weight.detach(), torch.zeros_like(base_carrier.weight)))

        optimizer.step()
        changed = (not torch.equal(lora_a.detach(), before_a)) or (
            not torch.equal(lora_b.detach(), before_b)
        )
        self.assertTrue(changed)
        self.assertEqual(int(base_carrier.weight.numel()), 1)
        self.assertFalse(bool(base_carrier.weight.requires_grad))
        self.assertTrue(torch.equal(base_carrier.weight.detach(), torch.zeros_like(base_carrier.weight)))

        host.eval()
        with torch.no_grad():
            y_proxy = proxy(x).detach().clone()
            base_out = proxy.base_layer(x).detach().clone()
            # Activation-space check: protected input coords must not contribute LoRA delta.
            # Flip protected inputs; subspace LoRA sees only non-protected columns, so output
            # delta vs base must stay identical for those flips.
            x_flip = x.clone()
            x_flip[:, protected] = x_flip[:, protected] + 3.0
            delta_orig = (proxy(x) - proxy.base_layer(x)).detach()
            delta_flip = (proxy(x_flip) - proxy.base_layer(x_flip)).detach()
            self.assertTrue(torch.allclose(delta_orig, delta_flip, atol=1e-5, rtol=1e-5))
            self.assertFalse(torch.allclose(y_proxy, base_out, atol=1e-6, rtol=1e-6))

        payloads = extract_subspace_peft_low_rank_payloads(host, module_names=[name])
        self.assertIn(name, payloads)
        low_rank_a, low_rank_b = payloads[name]
        self.assertEqual(tuple(low_rank_a.shape), (4, rank))
        self.assertEqual(tuple(low_rank_b.shape), (rank, 4))

        export_compressed_subspace_peft_lora_to_vae_low_rank(
            host,
            module_names=[name],
            allow_overwrite=False,
        )
        exported = host.model.layers[0].mlp.down_proj
        self.assertIsInstance(exported, VAELinear)
        self.assertEqual(list(iter_named_compressed_subspace_peft_proxies(host)), [])
        for module_name, module in host.named_modules():
            self.assertNotIsInstance(
                module,
                CompressedSubspacePeftProxy,
                msg=f"proxy still present at {module_name}",
            )
            self.assertNotIsInstance(
                module,
                PeftZeroLinearCarrier,
                msg=f"carrier still present at {module_name}",
            )
            self.assertFalse(
                is_peft_lora_linear(module),
                msg=f"PEFT LoRA linear still present at {module_name}",
            )

        self.assertEqual(exported.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        w_base = exported._decode_weight(dtype=torch.float32, include_low_rank=False)
        w_final = exported._decode_weight(dtype=torch.float32, include_low_rank=True)
        self.assertTrue(
            torch.allclose(w_final[:, protected], w_base[:, protected], atol=1e-6, rtol=1e-6)
        )
        keep = [i for i in range(6) if i not in set(protected.tolist())]
        changed_coords = False
        for col in keep:
            if not torch.allclose(w_final[:, col], w_base[:, col], atol=1e-6, rtol=1e-6):
                changed_coords = True
                break
        self.assertTrue(changed_coords)

        exported.eval()
        with torch.no_grad():
            y_exported = exported(x).detach().clone()
        self.assertTrue(torch.allclose(y_proxy, y_exported, atol=1e-5, rtol=1e-5))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(host, tmpdir, save_config=False)
            restored_host = TinyProxyHost(nn.Linear(6, 4, bias=False))
            restored_host, _, _ = load_checkpoint_into_model(restored_host, tmpdir)
            loaded = restored_host.model.layers[0].mlp.down_proj
            self.assertIsInstance(loaded, VAELinear)
            self.assertEqual(loaded.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
            self.assertEqual(list(iter_named_compressed_subspace_peft_proxies(restored_host)), [])
            loaded.eval()
            with torch.no_grad():
                y_loaded = loaded(x).detach().clone()
            self.assertTrue(torch.allclose(y_proxy, y_loaded, atol=1e-5, rtol=1e-5))


class LowRankScopeCheckpointIOTests(unittest.TestCase):
    def test_old_full_spec_missing_scope_loads_as_full(self):
        rank = 2
        layer = _build_vae_linear(
            in_features=4,
            out_features=4,
            low_rank_a=torch.randn(4, rank),
            low_rank_b=torch.randn(rank, 4),
            low_rank_scope=LOW_RANK_SCOPE_FULL,
        )
        model = nn.Module()
        model.proj = layer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            meta_path = os.path.join(tmpdir, META_FILENAME)
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertNotIn("low_rank_scope", meta["converted_modules"][0])
            restored = nn.Module()
            restored.proj = nn.Linear(4, 4, bias=False)
            restored, _, _ = load_checkpoint_into_model(restored, tmpdir)
            self.assertEqual(restored.proj.low_rank_scope, LOW_RANK_SCOPE_FULL)

    def test_new_full_save_omits_low_rank_scope_key(self):
        rank = 2
        layer = _build_vae_linear(
            in_features=4,
            out_features=4,
            low_rank_a=torch.randn(4, rank),
            low_rank_b=torch.randn(rank, 4),
            low_rank_scope=LOW_RANK_SCOPE_FULL,
        )
        model = nn.Module()
        model.proj = layer
        specs = _collect_vae_linear_specs(model)
        self.assertEqual(len(specs), 1)
        self.assertNotIn("low_rank_scope", specs[0])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            with open(os.path.join(tmpdir, META_FILENAME), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertNotIn("low_rank_scope", meta["converted_modules"][0])

    def test_subspace_checkpoint_roundtrip_with_input_protection(self):
        rank = 2
        protected = torch.tensor([1, 4], dtype=torch.long)
        low_rank_a = torch.randn(4, rank, dtype=torch.float32)
        low_rank_b = torch.randn(rank, 4, dtype=torch.float32)
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=protected,
            low_rank_a=low_rank_a.clone(),
            low_rank_b=low_rank_b.clone(),
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        model = nn.Module()
        model.proj = layer
        x = torch.randn(2, 6, dtype=torch.float32)
        source_weight = layer._decode_weight(dtype=torch.float32).detach().clone()
        source_out = layer(x).detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            with open(os.path.join(tmpdir, META_FILENAME), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(
                meta["converted_modules"][0]["low_rank_scope"],
                LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
            )
            restored = nn.Module()
            restored.proj = nn.Linear(6, 4, bias=False)
            restored, _, _ = load_checkpoint_into_model(restored, tmpdir)
            loaded = restored.proj
            self.assertEqual(loaded.low_rank_scope, LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
            self.assertEqual(int(loaded.low_rank_a.shape[0]), int(loaded.compressed_out_features))
            self.assertEqual(int(loaded.low_rank_b.shape[1]), int(loaded.compressed_in_features))
            self.assertTrue(
                torch.allclose(loaded._decode_weight(dtype=torch.float32), source_weight, atol=1e-6, rtol=1e-6)
            )
            self.assertTrue(torch.allclose(loaded(x), source_out, atol=1e-5, rtol=1e-5))

    def test_corrupted_incompatible_shape_scope_fails(self):
        rank = 2
        protected = torch.tensor([1, 4], dtype=torch.long)
        layer = _build_vae_linear(
            in_features=6,
            out_features=4,
            compressed_in_features=4,
            protected_input_indices=protected,
            low_rank_a=torch.randn(4, rank),
            low_rank_b=torch.randn(rank, 4),
            low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        model = nn.Module()
        model.proj = layer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_checkpoint(model, tmpdir, save_config=False)
            meta_path = os.path.join(tmpdir, META_FILENAME)
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)

            subspace_as_full = copy.deepcopy(meta)
            subspace_as_full["converted_modules"][0]["low_rank_scope"] = LOW_RANK_SCOPE_FULL
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(subspace_as_full, handle)
            bad_full = nn.Module()
            bad_full.proj = nn.Linear(6, 4, bias=False)
            with self.assertRaises(ValueError):
                load_checkpoint_into_model(bad_full, tmpdir)

            full_layer = _build_vae_linear(
                in_features=6,
                out_features=4,
                compressed_in_features=4,
                protected_input_indices=protected,
                low_rank_a=torch.randn(4, rank),
                low_rank_b=torch.randn(rank, 6),
                low_rank_scope=LOW_RANK_SCOPE_FULL,
            )
            full_model = nn.Module()
            full_model.proj = full_layer
            save_model_checkpoint(full_model, tmpdir, save_config=False)
            with open(meta_path, "r", encoding="utf-8") as handle:
                full_meta = json.load(handle)
            full_as_subspace = copy.deepcopy(full_meta)
            full_as_subspace["converted_modules"][0]["low_rank_scope"] = LOW_RANK_SCOPE_COMPRESSED_SUBSPACE
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(full_as_subspace, handle)
            bad_subspace = nn.Module()
            bad_subspace.proj = nn.Linear(6, 4, bias=False)
            with self.assertRaises(ValueError):
                load_checkpoint_into_model(bad_subspace, tmpdir)


def _wrap_inject_subspace_host(
    *,
    in_features: int = 6,
    out_features: int = 4,
    compressed_in_features: int = 4,
    protected_input_indices: torch.Tensor | None = None,
    rank: int = 2,
) -> tuple[TinyProxyHost, str, CompressedSubspacePeftProxy]:
    if protected_input_indices is None:
        protected_input_indices = torch.tensor([1, 4], dtype=torch.long)
    layer = _build_vae_linear(
        in_features=in_features,
        out_features=out_features,
        compressed_in_features=compressed_in_features,
        protected_input_indices=protected_input_indices,
    )
    host = TinyProxyHost(layer)
    name = "model.layers.0.mlp.down_proj"
    wrap_vae_linears_with_compressed_subspace_peft_proxy(host, [(name, layer)])
    inject_compressed_subspace_peft_lora(host, rank=rank, alpha=float(rank), dropout=0.0)
    proxy = host.model.layers[0].mlp.down_proj
    assert isinstance(proxy, CompressedSubspacePeftProxy)
    return host, name, proxy


def test_hif4_collect_treats_subspace_proxy_as_one_logical_linear():
    from train_utils.hif4_act import collect_hif4_act_modules

    host, name, proxy = _wrap_inject_subspace_host()
    collected = collect_hif4_act_modules(host)
    names = [module_name for module_name, _module in collected]
    assert name in names
    assert names.count(name) == 1
    matched = [module for module_name, module in collected if module_name == name]
    assert len(matched) == 1
    assert matched[0] is proxy


def test_hif4_collect_does_not_hook_subspace_proxy_descendants():
    from train_utils.hif4_act import collect_hif4_act_modules

    host, name, _proxy = _wrap_inject_subspace_host()
    names = [module_name for module_name, _module in collect_hif4_act_modules(host)]
    assert not any(module_name.startswith(f"{name}.") for module_name in names)


def test_temporary_mode_visits_subspace_proxy_once():
    from e2e_common.temporary_mode import set_model_temporary

    host, _name, proxy = _wrap_inject_subspace_host()
    calls: list[tuple[str, bool]] = []
    original_proxy_set = proxy.set_temporary
    original_base_set = proxy.base_layer.set_temporary

    def tracking_proxy_set(temporary: bool = True):
        calls.append(("proxy", bool(temporary)))
        return original_proxy_set(temporary)

    def tracking_base_set(temporary: bool = True):
        calls.append(("base", bool(temporary)))
        return original_base_set(temporary)

    proxy.set_temporary = tracking_proxy_set
    proxy.base_layer.set_temporary = tracking_base_set

    set_model_temporary(host, False)
    assert calls.count(("proxy", False)) == 1
    assert calls.count(("base", False)) == 1


def test_vae_module_refs_yield_subspace_proxy_not_nested_base_layer():
    from e2e_common.proxy_trainables import iter_named_vae_module_refs

    host, name, proxy = _wrap_inject_subspace_host()
    refs = list(iter_named_vae_module_refs(host))
    matching = [ref for ref in refs if ref.name == name]
    assert len(matching) == 1
    assert matching[0].module is proxy
    assert matching[0].base_layer is proxy.base_layer
    assert isinstance(matching[0].module, CompressedSubspacePeftProxy)
    assert not any(ref.name.startswith(f"{name}.") for ref in refs)


if __name__ == "__main__":
    unittest.main()
