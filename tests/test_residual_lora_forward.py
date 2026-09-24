from __future__ import annotations

from typing import Callable

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn

from e2e_common.residual_lora import (
    collect_residual_lora_parameters,
    get_residual_lora_topology,
    install_residual_lora,
)
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM


def _qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    ))


def _llama() -> LlamaForCausalLM:
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    ))


def _layer_call(model, hidden_states, *, output_attentions=False, use_cache=False):
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    return model.model.layers[0](
        hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        output_attentions=output_attentions,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
    )


def _install(model, mode: str):
    install_residual_lora(model, rank=2, alpha=4.0, dropout=0.0, mode=mode)


def _randomize_b(model):
    with torch.no_grad():
        for layer in model.model.layers:
            layer._residual_lora_attention.lora_B.weight.normal_(mean=0.0, std=0.05)
            layer._residual_lora_mlp.lora_B.weight.normal_(mean=0.0, std=0.05)


@pytest.mark.parametrize("factory", [_qwen3, _llama])
@pytest.mark.parametrize("mode", ["replace", "additive"])
def test_stateless_forward_matches_manual_two_site_formula(factory: Callable[[], nn.Module], mode: str):
    torch.manual_seed(101)
    model = factory().eval()
    install_residual_lora(model, rank=2, alpha=4.0, dropout=0.0, layer_indices=[0], mode=mode)
    layer = model.model.layers[0]
    if mode == "additive":
        with torch.no_grad():
            layer._residual_lora_attention.lora_B.weight.normal_(mean=0.0, std=0.05)
            layer._residual_lora_mlp.lora_B.weight.normal_(mean=0.0, std=0.05)
    x = torch.randn(1, 4, 16)

    position_ids = torch.arange(x.shape[1]).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(x, position_ids)
    attention_out, _ = layer.self_attn(
        hidden_states=layer.input_layernorm(x), attention_mask=None,
        position_ids=position_ids, output_attentions=False, use_cache=False,
        position_embeddings=position_embeddings,
    )
    after_attention = layer._residual_lora_attention(x) + attention_out
    mlp_out = layer.mlp(layer.post_attention_layernorm(after_attention))
    expected = layer._residual_lora_mlp(after_attention) + mlp_out
    actual = layer(
        x, attention_mask=None, position_ids=position_ids,
        output_attentions=False, use_cache=False,
        position_embeddings=position_embeddings,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("mode", ["replace", "additive"])
def test_reentrant_and_nonreentrant_checkpoint_match_normal_backward_repeatedly(mode: str):
    torch.manual_seed(103)
    normal = _qwen3().train()
    _install(normal, mode)
    checkpointed = _qwen3().train()
    _install(checkpointed, mode)
    checkpointed.load_state_dict(normal.state_dict(), strict=True)
    if mode == "additive":
        _randomize_b(normal)
        checkpointed.load_state_dict(normal.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4]])

    normal_loss = normal(input_ids=ids).logits.float().square().mean()
    normal_loss.backward()
    normal_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in collect_residual_lora_parameters(normal).items()
    }

    for use_reentrant in (False, True):
        checkpointed.zero_grad(set_to_none=True)
        checkpointed.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
        first_loss = checkpointed(input_ids=ids).logits.float().square().mean()
        first_loss.backward()
        first_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in collect_residual_lora_parameters(checkpointed).items()
        }
        torch.testing.assert_close(first_loss.detach(), normal_loss.detach(), rtol=1e-5, atol=1e-6)
        for name in normal_grads:
            torch.testing.assert_close(first_grads[name], normal_grads[name], rtol=1e-5, atol=1e-6)

        checkpointed.zero_grad(set_to_none=True)
        second_loss = checkpointed(input_ids=ids).logits.float().square().mean()
        second_loss.backward()
        for name, parameter in collect_residual_lora_parameters(checkpointed).items():
            torch.testing.assert_close(parameter.grad, first_grads[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("factory", [_qwen3, _llama])
@pytest.mark.parametrize("mode", ["replace", "additive"])
def test_kv_cache_segmented_decode_matches_full(factory: Callable[[], nn.Module], mode: str):
    torch.manual_seed(107)
    model = factory().eval()
    model.config.use_cache = True
    _install(model, mode)
    if mode == "additive":
        _randomize_b(model)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=True)
        first = model(input_ids=ids[:, :3], use_cache=True)
        second = model(input_ids=ids[:, 3:], past_key_values=first.past_key_values, use_cache=True)
    torch.testing.assert_close(second.logits, full.logits[:, 3:], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["replace", "additive"])
def test_output_attentions_and_bfloat16_manual_path_have_no_subtraction(mode: str):
    torch.manual_seed(109)
    model = _qwen3().eval().to(dtype=torch.bfloat16)
    _install(model, mode)
    if mode == "additive":
        _randomize_b(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        outputs = model(input_ids=ids, output_attentions=True, use_cache=False)
    assert outputs.attentions is not None
    assert len(outputs.attentions) == 2
    assert outputs.logits.dtype == torch.bfloat16

    layer = model.model.layers[0]
    x = model.model.embed_tokens(ids)
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(x, position_ids)
    attn, _ = layer.self_attn(
        hidden_states=layer.input_layernorm(x), attention_mask=None,
        position_ids=position_ids, output_attentions=False, use_cache=False,
        position_embeddings=position_embeddings,
    )
    h = layer._residual_lora_attention(x) + attn
    expected = layer._residual_lora_mlp(h) + layer.mlp(layer.post_attention_layernorm(h))
    actual = layer(
        x, position_ids=position_ids, output_attentions=False, use_cache=False,
        position_embeddings=position_embeddings,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_matching_install_is_idempotent_and_plain_model_topology_is_none():
    assert get_residual_lora_topology(nn.Linear(4, 4)) is None
    model = _qwen3().eval()
    install_residual_lora(model, rank=2, alpha=4.0, dropout=0.0, layer_indices=[0], mode="replace")
    before = {name: parameter.detach().clone() for name, parameter in collect_residual_lora_parameters(model).items()}
    install_residual_lora(model, rank=2, alpha=4.0, dropout=0.0, layer_indices=[0], mode="replace")
    after = collect_residual_lora_parameters(model)
    for name in before:
        torch.testing.assert_close(before[name], after[name])
    with pytest.raises(ValueError, match="conflicts"):
        install_residual_lora(model, rank=2, alpha=8.0, dropout=0.0, layer_indices=[0], mode="replace")


@pytest.mark.parametrize("factory", [_qwen3, _llama])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_additive_b0_is_exact_identity_for_qwen3_llama_and_bfloat16(factory, dtype):
    torch.manual_seed(127)
    base = factory().eval().to(dtype=dtype)
    adapted = factory().eval().to(dtype=dtype)
    adapted.load_state_dict(base.state_dict(), strict=True)
    install_residual_lora(adapted, rank=2, alpha=4.0, dropout=0.0, mode="additive")
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = base(input_ids=ids).logits
        actual = adapted(input_ids=ids).logits
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
