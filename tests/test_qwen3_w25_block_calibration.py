from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.block_calibration import capture_block_batches


def _fixture():
    torch.manual_seed(913)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, max_position_embeddings=32, attention_dropout=0.0,
                        use_cache=True, output_attentions=True)
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 0, 0, 0], [0, 0, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
    }
    return teacher, batch


def _assert_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.device.type == "cpu" and not actual.requires_grad and actual.grad_fn is None
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def test_real_bf16_qwen_capture_matches_full_teacher_and_replays_each_document():
    teacher, batch = _fixture()
    state = deepcopy(teacher.state_dict())
    input_snapshot = deepcopy(batch)
    target = teacher.model.layers[1]
    expected = []

    def pre(_module, args, kwargs):
        expected.append({"hidden_states": args[0].detach().clone(), "kwargs": deepcopy(kwargs)})

    def post(_module, _args, output):
        expected[-1]["output"] = output[0].detach().clone()

    handles = [target.register_forward_pre_hook(pre, with_kwargs=True), target.register_forward_hook(post)]
    try:
        with torch.no_grad():
            for index in range(len(batch["input_ids"])):
                teacher(input_ids=batch["input_ids"][index:index + 1],
                        attention_mask=batch["attention_mask"][index:index + 1],
                        use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    downstream_calls = []
    later_hook = teacher.model.layers[2].register_forward_pre_hook(
        lambda _module, _args: downstream_calls.append(True))
    try:
        actual = capture_block_batches(teacher, 1, batch)
    finally:
        later_hook.remove()
    assert not downstream_calls
    assert len(actual) == len(expected) == 3
    for index, (item, reference) in enumerate(zip(actual, expected)):
        _assert_tree_equal(item.hidden_states, reference["hidden_states"])
        _assert_tree_equal(item.target_hidden_states, reference["output"])
        _assert_tree_equal(item.forward_kwargs, reference["kwargs"])
        assert item.hidden_states.shape == (1, 6, 16)
        assert item.hidden_states.dtype == item.target_hidden_states.dtype == torch.bfloat16
        assert torch.equal(item.valid_mask, batch["attention_mask"][index:index + 1].bool())
        assert item.forward_kwargs["use_cache"] is False
        assert item.forward_kwargs["output_attentions"] is False
        assert item.forward_kwargs["past_key_value"] is None
        assert torch.equal(item.forward_kwargs["cache_position"], torch.arange(6))
        assert item.forward_kwargs["attention_mask"].shape == (1, 1, 6, 6)
        with torch.no_grad():
            replayed = target(item.hidden_states, **item.forward_kwargs)[0]
        assert torch.equal(replayed, item.target_hidden_states)
    assert not torch.equal(actual[0].forward_kwargs["attention_mask"], actual[1].forward_kwargs["attention_mask"])
    for field in ("attention_mask", "cache_position"):
        assert actual[0].forward_kwargs[field].data_ptr() != actual[1].forward_kwargs[field].data_ptr()
    assert not target._forward_hooks and not target._forward_pre_hooks
    _assert_tree_equal(batch, input_snapshot)
    _assert_tree_equal(teacher.state_dict(), state)
    assert all(not module.training for module in teacher.modules())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())


def test_real_teacher_failure_cleans_capture_hooks_without_removing_existing_hooks():
    teacher, batch = _fixture()
    target = teacher.model.layers[1]
    called = []
    existing = target.register_forward_hook(lambda _module, _args, _output: called.append(True))
    initial_pre, initial_post = dict(target._forward_pre_hooks), dict(target._forward_hooks)
    batch["input_ids"][1, 0] = teacher.config.vocab_size
    try:
        with pytest.raises(IndexError):
            capture_block_batches(teacher, 1, batch)
        assert dict(target._forward_pre_hooks) == initial_pre
        assert dict(target._forward_hooks) == initial_post
        assert called == [True]
        batch["input_ids"][1, 0] = 0
        assert len(capture_block_batches(teacher, 1, batch)) == 3
    finally:
        existing.remove()


@pytest.mark.parametrize("bad_index", [-1, 3, True])
def test_invalid_layer_is_rejected(bad_index):
    teacher, batch = _fixture()
    with pytest.raises(ValueError, match="block_index"):
        capture_block_batches(teacher, bad_index, batch)


def test_capture_rejects_trainable_teacher_and_invalid_mask():
    teacher, batch = _fixture()
    teacher.model.layers[1].train()
    with pytest.raises(ValueError, match="eval-mode"):
        capture_block_batches(teacher, 1, batch)
    teacher.eval()
    next(teacher.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        capture_block_batches(teacher, 1, batch)
    teacher.requires_grad_(False)
    for invalid in (torch.zeros_like(batch["attention_mask"]), batch["attention_mask"].float().fill_(0.5)):
        with pytest.raises(ValueError, match="binary attention_mask"):
            capture_block_batches(teacher, 1, {**batch, "attention_mask": invalid})
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in teacher.model.layers)
