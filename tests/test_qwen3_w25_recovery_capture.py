from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.linear_output.calibration import capture_inputs
from experiments.qwen3_w25.local_recovery import TARGETS, capture_shard_inputs, shard_targets


def _teacher_and_batches():
    torch.manual_seed(912)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    train = {"input_ids": torch.tensor([[4, 5, 6, 7, 0, 0], [8, 9, 10, 11, 12, 13]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])}
    heldout = {"input_ids": torch.tensor([[14, 15, 16, 0, 0], [17, 18, 19, 20, 0]]),
               "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])}
    targets = tuple(reversed(TARGETS[:14]))
    return teacher, train, heldout, targets


def test_real_qwen_multi_capture_matches_every_single_target_and_preserves_state():
    teacher, train, heldout, targets = _teacher_and_batches()
    initial = deepcopy(teacher.state_dict())
    actual = capture_shard_inputs(teacher, targets, train, heldout)
    for name in targets:
        layer = teacher.get_submodule(name)
        assert torch.equal(actual[name]["reference_weight"], layer.weight)
        for split, batch in (("train", train), ("heldout", heldout)):
            expected = capture_inputs(teacher, name, batch, microbatch_size=1)
            assert actual[name][split + "_inputs"].device.type == "cpu"
            assert torch.equal(actual[name][split + "_inputs"], expected), (name, split)
        assert not layer._forward_pre_hooks
    assert all(torch.equal(value, initial[key]) for key, value in teacher.state_dict().items())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    with torch.no_grad():
        assert torch.isfinite(teacher(**train, use_cache=False).logits).all()


def test_teacher_forward_failure_removes_all_capture_hooks():
    teacher, train, heldout, targets = _teacher_and_batches()
    train["input_ids"][0, 0] = 128
    with pytest.raises(IndexError):
        capture_shard_inputs(teacher, targets, train, heldout)
    assert all(not teacher.get_submodule(name)._forward_pre_hooks for name in targets)


@pytest.mark.parametrize("num_shards", [3, 4, 36])
def test_shards_cover_every_projection_once_and_keep_layers_together(num_shards):
    groups = [shard_targets(index, num_shards) for index in range(num_shards)]
    flat = [name for group in groups for name in group]
    assert len(flat) == len(set(flat)) == 252
    assert set(flat) == set(TARGETS)
    for index, group in enumerate(groups):
        assert all(int(name.split(".")[2]) % num_shards == index for name in group)
        assert len(group) % 7 == 0
