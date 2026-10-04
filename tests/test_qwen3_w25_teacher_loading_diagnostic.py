"""CPU contracts for teacher loader observations; no full HF model is loaded."""
from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from experiments.qwen3_w25.kl_gradient_diagnostic import batch_manifest
from experiments.qwen3_w25.teacher_loading_diagnostic import compare_ce, model_inventory, response_predictions


def test_real_qwen3_rope_inventory_observes_whole_model_cast_and_original_alias():
    rope = Qwen3RotaryEmbedding(Qwen3Config(hidden_size=128, num_attention_heads=1, num_key_value_heads=1, head_dim=128))
    before = model_inventory(rope)["rope_attributes"][""]
    assert before["inv_freq"]["dtype"] == "torch.float32"
    assert before["same_python_tensor"] and before["same_storage"]
    rope.to(dtype=torch.bfloat16)
    after = model_inventory(rope)["rope_attributes"][""]
    assert after["inv_freq"]["dtype"] == "torch.bfloat16"
    assert after["original_inv_freq"]["dtype"] == "torch.float32"
    assert not after["same_python_tensor"] and not after["same_storage"]
    assert after["original_inv_freq"]["values_fp64"] == before["inv_freq"]["values_fp64"]
    assert after["inv_freq"]["values_fp64"] != before["inv_freq"]["values_fp64"]


def test_causal_response_ce_topk_probabilities_and_membership_match_full_softmax():
    batch = {"input_ids": torch.tensor([[1, 2, 3, 0]]), "labels": torch.tensor([[-100, -100, 3, -100]]),
             "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    logits = torch.arange(128).float().repeat(1, 4, 1) / 10
    manifest = batch_manifest(batch, 256, sample_offset=1024)
    rows = response_predictions(logits, batch, manifest)
    token = rows[0]["response_tokens"][0]
    probability = torch.softmax(logits[0, 1], dim=0)
    assert token["prediction_position"] == 1 and token["target_position"] == 2 and token["label_id"] == 3
    assert token["label_ce"] == pytest.approx(float(-probability[3].log()), rel=2e-6)
    assert token["top5_ids"] == [127, 126, 125, 124, 123]
    assert token["top5_probabilities"] == pytest.approx(probability[token["top5_ids"]].tolist(), rel=2e-6)
    assert not token["label_in_teacher_top100"]
    altered = deepcopy(rows)
    altered[0]["response_tokens"][0]["label_ce"] += .02
    result = compare_ce(rows, altered, reference_field="label_ce", current_field="label_ce")
    assert result["max_abs_difference"] == pytest.approx(.02) and result["numerical_acceptance_threshold"] is None
    altered[0]["response_tokens"][0]["label_id"] = 4
    with pytest.raises(ValueError, match="token identities"):
        compare_ce(rows, altered, reference_field="label_ce", current_field="label_ce")
