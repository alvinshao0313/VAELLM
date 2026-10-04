from copy import deepcopy

import pytest
import torch

from litebsq.bitpack import build_bitpack_u8_spec
from train_utils.checkpoint_bpw import audit_checkpoint, audit_state_dict, estimate_qwen3_cat_bpw


def _fixture(bits=9):
    name = "model.layers.0.self_attn.q_proj"
    vq = build_bitpack_u8_spec(logical_shape=(4, 1, bits))
    spec = {"name": name, "in_features": 8, "out_features": 4,
            "compressed_in_features": 8, "compressed_out_features": 4,
            "codebook_dim": 8, "parallel_parts": 1, "residual_stages": 1,
            "vq_weights": [vq], "has_bias": False}
    meta = {"format": "vaellm_model_checkpoint_v6", "schema_version": 6,
            "checkpoint_kind": "final_model", "lora_config": None,
            "compressed_targets": [name], "converted_modules": [spec]}
    state = {f"{name}.vq_weight": torch.zeros(vq["shape"], dtype=torch.uint8),
             f"{name}.decoder.linear.weight": torch.zeros((8, bits), dtype=torch.bfloat16)}
    return meta, state, name


def test_padding_and_dense_inventory_and_tied_embeddings():
    meta, state, name = _fixture()
    dense = "model.layers.0.self_attn.k_proj"
    meta["pending_dense_targets"] = [dense]
    state[f"{dense}.weight"] = torch.zeros((4, 8), dtype=torch.float16)
    embedding = torch.ones((7, 8), dtype=torch.bfloat16)
    state["model.embed_tokens.weight"] = embedding
    state["lm_head.weight"] = embedding
    result = audit_state_dict(meta, state, max_bpw=100)
    assert result["original_linear_weights"] == 64
    assert result["user_linear_bits"] == 36 + 72 * 16 + 32 * 16
    assert result["packed_linear_bits"] == 64 + 72 * 16 + 32 * 16
    assert result["full_model_tensor_bytes"] == (64 + 72 * 16 + 32 * 16) // 8 + 112
    assert result["code_padding_bits"] == 28
    assert len(result["storage_aliases"]) == 1


def test_shared_decoder_not_counted_twice():
    meta, state, name = _fixture(8)
    second = "model.layers.1.self_attn.q_proj"
    spec = deepcopy(meta["converted_modules"][0])
    spec["name"] = second
    meta["compressed_targets"].append(second)
    meta["converted_modules"].append(spec)
    state[f"{second}.vq_weight"] = state[f"{name}.vq_weight"].clone()
    state[f"{second}.decoder.linear.weight"] = state[f"{name}.decoder.linear.weight"]
    result = audit_state_dict(meta, state, max_bpw=100)
    assert result["components"]["decoder"]["storage_bits"] == 64 * 16
    assert result["per_target"][name]["budget_bits"] == result["per_target"][second]["budget_bits"]


def test_unknown_payload_is_charged_and_metadata_missing_is_rejected():
    meta, state, name = _fixture(8)
    state[f"{name}.future_payload"] = torch.zeros(3, dtype=torch.float32)
    result = audit_state_dict(meta, state, max_bpw=100)
    assert result["components"]["other_target"]["budget_bits"] == 96
    assert result["unclassified_target_keys"] == [f"{name}.future_payload"]
    meta["converted_modules"][0]["protected_input_indices"] = {"shape": [1], "dtype": "int64"}
    with pytest.raises(ValueError, match="metadata tensor"):
        audit_state_dict(meta, state)


def test_rank_and_protection_contracts():
    meta, state, name = _fixture(8)
    state[f"{name}.low_rank_a"] = torch.zeros((4, 17), dtype=torch.bfloat16)
    state[f"{name}.low_rank_b"] = torch.zeros((17, 8), dtype=torch.bfloat16)
    state[f"{name}.protected_input_indices"] = torch.zeros(1, dtype=torch.int64)
    result = audit_state_dict(meta, state, max_bpw=1000)
    assert not result["budget_passed"]
    assert len(result["contract_violations"]) == 2


def test_cross_scope_alias_is_rejected():
    meta, state, name = _fixture(8)
    state["lm_head.weight"] = state[f"{name}.decoder.linear.weight"]
    with pytest.raises(ValueError, match="alias crosses"):
        audit_state_dict(meta, state)


@pytest.mark.parametrize("residual", [False, True])
def test_zero_rank_budget_rejects_any_lora(residual):
    meta, state, name = _fixture(8)
    assert audit_state_dict(meta, state, max_bpw=100, max_lora_rank=0)["budget_passed"]
    if residual:
        state["model.layers.0.mlp._residual_lora_adapter.lora_A.weight"] = torch.zeros((1, 8))
        state["model.layers.0.mlp._residual_lora_adapter.lora_B.weight"] = torch.zeros((4, 1))
    else:
        state[f"{name}.low_rank_a"] = torch.zeros((4, 1))
        state[f"{name}.low_rank_b"] = torch.zeros((1, 8))
    result = audit_state_dict(meta, state, max_bpw=100, max_lora_rank=0)
    assert not result["budget_passed"]
    assert any("LoRA rank" in item for item in result["contract_violations"])


def test_omitted_projection_and_training_step_rejected():
    meta, state, name = _fixture(8)
    state["model.layers.0.mlp.down_proj.weight"] = torch.zeros((4, 8))
    with pytest.raises(ValueError, match="omitted"):
        audit_state_dict(meta, state)
    meta["checkpoint_kind"] = "training_step"
    with pytest.raises(ValueError, match="stable"):
        audit_state_dict(meta, state)


def test_actual_checkpoint_file_loading(tmp_path):
    import json
    meta, state, name = _fixture(8)
    (tmp_path / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    torch.save(state, tmp_path / "pytorch_model.bin")
    result = audit_checkpoint(tmp_path, max_bpw=100)
    assert result["state_dict_file_bytes"] > result["full_model_tensor_bytes"]


def test_preflight_matches_effective_symmetric_architecture():
    config = {"model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288,
              "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
    category = {"codebook_bits": 32, "codebook_dim": 32, "residual_stages": 2,
                "base_ch": 128, "num_res_blocks": 0, "decoder_num_res_blocks": 1,
                "decoder_type": "symmetric", "norm_type": "layer", "weight_rotation": "two_sided"}
    categories = {key: category for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
    result = estimate_qwen3_cat_bpw(config, categories, lora_rank=16)
    assert result["original_linear_weights"] == 6945767424
    assert result["resolved_categories"]["q_proj"]["decoder_num_res_blocks"] == 0
    assert result["user_linear_bpw"] == pytest.approx(2.1136793053668477)
    assert result["measurement"] == "architecture_estimate_not_checkpoint"


@pytest.mark.parametrize("packed", [False, True])
def test_real_vae_linear_v6_save_accounting(tmp_path, packed):
    from litebsq.autoencoder import Decoder
    from litebsq.vae_linear import VAELinear
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.q_proj = VAELinear(
        in_features=8, out_features=4, bias=None, codebook_dim=8, transpose=False,
        stage_vq_weights=[torch.zeros((4, 1, 9), dtype=torch.bool) for _ in range(2)],
        stage_decoders=[Decoder(in_dim=9, out_dim=8, hidden_dim=4,
                               num_res_blocks=0, norm_type="layer",
                               decoder_type="symmetric").to(torch.bfloat16) for _ in range(2)],
    )
    if packed:
        root.model.q_proj.pack_parallel_stage_decoder_(trainable=False)
    destination = tmp_path / "final_model"
    save_v6_full_checkpoint(root, str(destination), checkpoint_kind="final_model",
                            compressed_targets=["model.q_proj"], save_config=False)
    result = audit_checkpoint(destination, max_bpw=100)
    expected_bits = sum(t.numel() * t.element_size() * 8 for t in root.state_dict().values())
    assert result["packed_linear_bits"] == expected_bits
    assert result["user_linear_bits"] == expected_bits - 2 * (64 - 36)
    assert not result["unclassified_target_keys"]
