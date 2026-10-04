from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.block_calibration import _cpu_copy
from experiments.qwen3_w25.local_block import PROJECTIONS
from experiments.qwen3_w25.sequential_block_calibration import SequentialBlockCalibration
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


def _fixture(*, compressed=True):
    torch.manual_seed(724)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, max_position_embeddings=32, attention_dropout=0.0)
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    student = deepcopy(teacher)
    if compressed:
        for index, block in enumerate(student.model.layers):
            for name in PROJECTIONS:
                linear = block.get_submodule(name)
                decoders = [Decoder(in_dim=8, out_dim=4, hidden_dim=8, num_res_blocks=0,
                                    norm_type="layer", decoder_type="symmetric").bfloat16()
                            for _ in range(2)]
                with torch.no_grad():
                    for decoder in decoders:
                        decoder.linear_out.linear.weight.mul_(0.05)
                        decoder.linear_out.linear.bias.mul_(0.05)
                projection = VAELinear(
                    in_features=linear.in_features, out_features=linear.out_features, bias=None,
                    codebook_dim=4, transpose=name == "mlp.down_proj",
                    stage_vq_weights=[torch.randint(0, 2, (linear.weight.numel() // 4, 1, 8), dtype=torch.bool)
                                      for _ in range(2)],
                    stage_decoders=decoders,
                    weight_rotation_spec=rotation_spec("two_sided", 0, 31, f"{index}.{name}"),
                )
                projection.pack_parallel_stage_decoder_(trainable=False)
                projection.cache_decoded_weight = index == 0
                block.set_submodule(name, projection)
        student.eval().requires_grad_(False)
    else:
        with torch.no_grad():
            student.model.layers[0].mlp.down_proj.weight.mul_(4)
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 0, 0, 0], [0, 0, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
    }
    return teacher, student, batch


def _reference(model, batch):
    layers = [[] for _ in model.model.layers]
    normalized = []
    handles = []
    for index, block in enumerate(model.model.layers):
        def pre(_module, args, kwargs, index=index):
            layers[index].append({"input": _cpu_copy(args[0]), "kwargs": _cpu_copy(kwargs)})

        def post(_module, _args, output, index=index):
            layers[index][-1]["output"] = _cpu_copy(output[0])

        handles.extend((block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post)))
    try:
        with torch.no_grad():
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
                output = model.model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0),
                                     use_cache=False, output_attentions=False, output_hidden_states=False)
                normalized.append(_cpu_copy(output.last_hidden_state))
    finally:
        for handle in handles:
            handle.remove()
    return layers, normalized


def _assert_tree(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.device.type == "cpu" and actual.grad_fn is None and not actual.requires_grad
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_tree(a, b)
    else:
        assert actual == expected


def test_two_real_compressed_blocks_follow_separate_streams_without_prefix_replay_or_reset():
    teacher, student, batch = _fixture()
    teacher_reference, teacher_norm = _reference(teacher, batch)
    student_reference, _ = _reference(student, batch)
    snapshots = [deepcopy(model.state_dict()) for model in (teacher, student)]
    parameter_ids = [{name: id(p) for name, p in model.named_parameters()} for model in (teacher, student)]
    flags = {name: module.cache_decoded_weight for name, module in student.named_modules() if isinstance(module, VAELinear)}
    calls = {"teacher": [0, 0], "student": [0, 0]}
    handles = []
    for label, model in (("teacher", teacher), ("student", student)):
        for index, block in enumerate(model.model.layers):
            def post(_module, _args, _output, label=label, index=index):
                calls[label][index] += 1
            handles.append(block.register_forward_hook(post))
    try:
        stream = SequentialBlockCalibration(teacher, student, batch)
        assert calls == {"teacher": [0, 0], "student": [0, 0]}
        assert stream.initialization_passes == {"teacher": 3, "student": 3}
        for index in range(2):
            assert stream.layer_index == index and not stream.complete
            prepared = stream.prepare(teacher.model.layers[index])
            for doc, item in enumerate(prepared):
                _assert_tree(item.hidden_states, student_reference[index][doc]["input"])
                _assert_tree(item.target_hidden_states, teacher_reference[index][doc]["output"])
                _assert_tree(item.forward_kwargs, student_reference[index][doc]["kwargs"])
                _assert_tree(item.valid_mask, batch["attention_mask"][doc:doc + 1].bool())
                assert item.forward_kwargs["attention_mask"].shape == (1, 1, 6, 6)
                assert torch.equal(item.forward_kwargs["cache_position"], torch.arange(6))
                assert torch.equal(item.forward_kwargs["position_ids"], torch.arange(6).unsqueeze(0))
            if index == 1:
                assert any(not torch.equal(item.hidden_states, teacher_reference[index][doc]["input"])
                           for doc, item in enumerate(prepared))
                assert all(not torch.equal(item.target_hidden_states, teacher_norm[doc])
                           for doc, item in enumerate(prepared))
            assert not torch.equal(prepared[0].forward_kwargs["attention_mask"], prepared[1].forward_kwargs["attention_mask"])
            stream.advance(student.model.layers[index])
            for doc, value in enumerate(stream._student_hidden):
                _assert_tree(value, student_reference[index][doc]["output"])
        assert stream.complete and stream.layer_index == 2
        assert calls == {"teacher": [3, 3], "student": [3, 3]}
    finally:
        for handle in handles:
            handle.remove()
    for model, before, ids in zip((teacher, student), snapshots, parameter_ids):
        _assert_tree(model.state_dict(), before)
        assert ids == {name: id(p) for name, p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert flags == {name: module.cache_decoded_weight for name, module in student.named_modules() if isinstance(module, VAELinear)}
    assert all(module._cached_weight is None for module in student.modules() if isinstance(module, VAELinear))


def test_advance_uses_updated_native_decoder_clears_stale_weights_and_protects_cached_inputs():
    teacher, student, batch = _fixture()
    old_reference, _ = _reference(student, batch)  # First-layer native weights are now cached.
    stream = SequentialBlockCalibration(teacher, student, batch)
    prepared = stream.prepare(teacher.model.layers[0])
    prepared[0].hidden_states.zero_()
    prepared[0].target_hidden_states.zero_()
    prepared[0].forward_kwargs["position_embeddings"][0].zero_()
    prepared[0].valid_mask.zero_()
    projection = student.model.layers[0].mlp.down_proj
    with torch.no_grad():
        projection._parallel_stage_decoder.linear_out.conv.weight.add_(0.03)
    stream.advance(student.model.layers[0])
    expected, _ = _reference(student, batch)
    for doc, actual in enumerate(stream._student_hidden):
        _assert_tree(actual, expected[0][doc]["output"])
    assert any(not torch.equal(actual, old_reference[0][doc]["output"])
               for doc, actual in enumerate(stream._student_hidden))
    second = stream.prepare(teacher.model.layers[1])
    teacher_expected, _ = _reference(teacher, batch)
    for doc, actual in enumerate(second):
        _assert_tree(actual.target_hidden_states, teacher_expected[1][doc]["output"])
        _assert_tree(actual.hidden_states, expected[1][doc]["input"])


def test_current_layer_backward_has_no_teacher_or_prefix_graph():
    teacher, student, batch = _fixture(compressed=False)
    stream = SequentialBlockCalibration(teacher, student, batch)
    stream.prepare(teacher.model.layers[0])
    stream.advance(student.model.layers[0])
    prepared = stream.prepare(teacher.model.layers[1])
    block = student.model.layers[1]
    block.requires_grad_(True)
    item = prepared[0]
    value = block(item.hidden_states, **item.forward_kwargs)[0]
    (value.float() - item.target_hidden_states.float()).square()[item.valid_mask].mean().backward()
    assert any(p.grad is not None and bool(p.grad.ne(0).any()) for p in block.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.model.layers[0].parameters())
    assert student.model.embed_tokens.weight.grad is None
    for hidden in stream._teacher_hidden + stream._student_hidden:
        assert hidden.grad_fn is None and not hidden.requires_grad
    assert item.hidden_states.grad is None and item.target_hidden_states.grad is None
    block.zero_grad(set_to_none=True)
    block.requires_grad_(False)
    stream.advance(block)
    assert stream.complete


def test_order_and_native_freezing_failures_do_not_advance_or_reset_the_stream():
    teacher, student, batch = _fixture()
    stream = SequentialBlockCalibration(teacher, student, batch)
    with pytest.raises(RuntimeError, match="Prepare"):
        stream.advance(student.model.layers[0])
    with pytest.raises(ValueError, match="block 0"):
        stream.prepare(teacher.model.layers[1])
    stream.prepare(teacher.model.layers[0])
    target = deepcopy(stream._teacher_hidden)
    with pytest.raises(RuntimeError, match="Advance"):
        stream.prepare(teacher.model.layers[0])
    projection = student.model.layers[0].mlp.down_proj
    projection._decoder_compute_dtype = torch.float32
    with pytest.raises(ValueError, match="native BF16"):
        stream.advance(student.model.layers[0])
    del projection._decoder_compute_dtype
    student.model.layers[0].input_layernorm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        stream.advance(student.model.layers[0])
    student.requires_grad_(False)
    assert stream.layer_index == 0
    _assert_tree(stream._teacher_hidden, target)
    stream.advance(student.model.layers[0])
    stream.prepare(teacher.model.layers[1])
    stream.advance(student.model.layers[1])
    with pytest.raises(RuntimeError, match="already completed"):
        stream.prepare(teacher.model.layers[0])


def test_genuine_input_failure_preserves_existing_hooks_and_removes_capture_hook():
    teacher, student, batch = _fixture(compressed=False)
    block = teacher.model.layers[0]
    calls = []
    handle = block.register_forward_pre_hook(lambda _module, _args: calls.append(True))
    before = dict(block._forward_pre_hooks)
    batch["input_ids"][1, 0] = teacher.config.vocab_size
    try:
        with pytest.raises(IndexError):
            SequentialBlockCalibration(teacher, student, batch)
        assert dict(block._forward_pre_hooks) == before
        assert calls == [True]
        assert not student.model.layers[0]._forward_pre_hooks
    finally:
        handle.remove()


def test_separate_real_rope_kwargs_must_match_and_models_must_be_frozen():
    teacher, student, batch = _fixture(compressed=False)
    with torch.no_grad():
        student.model.rotary_emb.inv_freq.mul_(0.5)
    with pytest.raises(ValueError, match="position_embeddings"):
        SequentialBlockCalibration(teacher, student, batch)
    assert not teacher.model.layers[0]._forward_pre_hooks
    assert not student.model.layers[0]._forward_pre_hooks
    student.train()
    with pytest.raises(ValueError, match="eval-mode"):
        SequentialBlockCalibration(teacher, student, batch)
