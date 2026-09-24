from __future__ import annotations

import copy

import pytest
import torch

from e2e_common.residual_lora import (
    ResidualLoraSite,
    collect_residual_lora_parameters,
    enable_residual_lora,
    get_residual_lora_topology,
    install_residual_lora,
    remove_residual_lora,
)
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM


def _qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False))


def _llama() -> LlamaForCausalLM:
    return LlamaForCausalLM(LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False))


def test_residual_lora_replaces_skip_and_has_two_sites():
    torch.manual_seed(7)
    model = _qwen3().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(input_ids=ids).logits
    install_residual_lora(model, rank=2, alpha=4.0, mode="replace")
    assert get_residual_lora_topology(model) == {
        "version": 1, "mode": "replace", "family": "qwen3", "layer_indices": [0, 1],
        "sites": {"attention": {"rank": 2, "alpha": 4.0, "dropout": 0.0},
                   "mlp": {"rank": 2, "alpha": 4.0, "dropout": 0.0}},
    }
    params = collect_residual_lora_parameters(model)
    assert len(params) == 8 and all(p.requires_grad for p in params.values())
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert not torch.allclose(before, after)
    for layer in model.model.layers:
        assert torch.all(layer._residual_lora_attention.lora_B.weight != 0)
        assert torch.all(layer._residual_lora_mlp.lora_B.weight != 0)


@pytest.mark.parametrize("mode", ["additive", "replace"])
def test_site_specific_configs_and_gradients(mode):
    model = _qwen3().train()
    install_residual_lora(model, rank=2, alpha=4.0, layer_indices=[1], mode=mode,
        site_configs={"attention": {"rank": 1, "alpha": 2.0}, "mlp": {"rank": 3, "alpha": 6.0}})
    spec = get_residual_lora_topology(model)
    assert spec["layer_indices"] == [1] and spec["sites"]["attention"]["rank"] == 1
    params = collect_residual_lora_parameters(model)
    model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    a_has_grad = any(bool(p.grad.abs().sum()) for n, p in params.items() if ".lora_A." in n)
    assert a_has_grad is (mode == "replace")  # Additive B starts at zero.
    assert any(bool(p.grad.abs().sum()) for n, p in params.items() if ".lora_B." in n)
    optimizer = torch.optim.SGD(params.values(), lr=0.1)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float().square().mean().backward()
    assert any(bool(p.grad.abs().sum()) for n, p in params.items() if ".lora_A." in n)
    enable_residual_lora(model, enabled=False)
    assert all(not p.requires_grad for p in params.values())


def test_state_dict_restore_and_forward_reinstall():
    torch.manual_seed(13)
    model = _qwen3().eval()
    install_residual_lora(model, rank=2, alpha=4.0)
    with torch.no_grad():
        for n, p in collect_residual_lora_parameters(model).items():
            if ".lora_B." in n:
                p.normal_(mean=0.0, std=0.03)
    ids = torch.tensor([[4, 3, 2, 1]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    state = copy.deepcopy(model.state_dict())
    restored = _qwen3().eval()
    install_residual_lora(restored, rank=2, alpha=4.0)
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = restored(input_ids=ids).logits
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-6)
    assert remove_residual_lora(restored) == 2
    with torch.no_grad():
        removed = restored(input_ids=ids).logits
    assert not torch.allclose(actual, removed)


def test_llama_supported_and_opt_rejected():
    llama = _llama().eval()
    install_residual_lora(llama, rank=1, alpha=1.0)
    assert get_residual_lora_topology(llama)["family"] == "llama"
    from transformers import OPTConfig, OPTForCausalLM
    opt = OPTForCausalLM(OPTConfig(vocab_size=31, hidden_size=16, ffn_dim=32,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=32))
    with pytest.raises(ValueError, match="supports only Llama and Qwen3"):
        install_residual_lora(opt, rank=1, alpha=1.0)


@pytest.mark.parametrize("mode", ["additive", "replace"])
def test_residual_site_is_scaled_b_a_projection(mode):
    site = ResidualLoraSite(3, rank=2, alpha=4.0, dropout=0.0, mode=mode).eval()
    with torch.no_grad():
        site.lora_A.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        site.lora_B.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    x = torch.tensor([[2.0, 3.0, 5.0]])
    torch.testing.assert_close(site(x), torch.tensor([[4.0, 6.0, 0.0]]) + (x if mode == "additive" else 0))
