from copy import deepcopy
from contextlib import contextmanager
import json
from types import MethodType

import pytest
import torch
from torch.nn import functional as F

from experiments.qwen3_w25.block_weight_graph import block_weight_graph
from experiments.qwen3_w25.local_block import PROJECTIONS, _squared_error
from test_qwen3_w25_local_block import _fixture


@contextmanager
def _projection_forwards(targets):
    """Pre-optimization reference: a fresh decoder graph for every batch."""
    def forward(module, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            weight = module._decode_weight(dtype=torch.float32, include_low_rank=False)
            bias = None if module.bias is None else module.bias.to(torch.bfloat16)
            return F.linear(inputs, weight.to(torch.bfloat16), bias)
    try:
        for _, module in targets:
            module.forward = MethodType(forward, module)
        yield
    finally:
        for _, module in targets:
            del module.forward


def _prepare(block, checkpoint):
    block.eval().requires_grad_(False)
    targets, parameters = [], {}
    for name in PROJECTIONS:
        module = block.get_submodule(name)
        module._parallel_stage_decoder.float()
        module._parallel_stage_decoder.use_checkpoint = checkpoint
        module.enable_trainable_decode(parallel_stage_decode=True)
        module._decoder_compute_dtype = torch.float32
        targets.append((name, module))
        parameters.update({name + "._parallel_stage_decoder." + key: p
                           for key, p in module._parallel_stage_decoder.named_parameters()})
    return targets, parameters


def _relative(actual, expected):
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_block_two_phase_gradients_and_adam_step_match_per_batch_path(checkpoint):
    original, train, heldout = _fixture()
    # Include different lengths, batch sizes and padding; teacher graphs cannot
    # become part of either gradient path.
    batches = train + heldout
    for batch in batches:
        batch.hidden_states.requires_grad_(True)
        batch.target_hidden_states.requires_grad_(True)
    reference, candidate = deepcopy(original), deepcopy(original)
    old_targets, old_params = _prepare(reference, checkpoint)
    targets, params = _prepare(candidate, checkpoint)
    initial = deepcopy(candidate.state_dict())
    ids = {name: id(p) for name, p in candidate.named_parameters()}
    flags = {name: p.requires_grad for name, p in candidate.named_parameters()}
    norm = sum(int(batch.valid_mask.sum()) for batch in batches) * original.hidden_size
    old_optimizer = torch.optim.AdamW(old_params.values(), lr=0.0002, weight_decay=0.0)
    optimizer = torch.optim.AdamW(params.values(), lr=0.0002, weight_decay=0.0)
    old_losses = []
    with _projection_forwards(old_targets):
        for batch in batches:
            loss = _squared_error(reference, batch, torch.device("cpu")) / norm
            loss.backward()
            old_losses.append(float(loss.detach()))
    losses = []
    with torch.autocast("cpu", dtype=torch.bfloat16), block_weight_graph(targets, requires_grad=True) as graph:
        for name in PROJECTIONS:
            weight, leaf = graph.decoded[name], graph.leaves[name]
            assert leaf.is_leaf and leaf.dtype == torch.float32 and leaf.grad is None
            assert leaf.data_ptr() == weight.data_ptr() and leaf.untyped_storage().nbytes() == weight.untyped_storage().nbytes()
        for batch in batches:
            loss = _squared_error(candidate, batch, torch.device("cpu")) / norm
            loss.backward()
            losses.append(float(loss.detach()))
            assert all(p.grad is None for p in params.values())
        assert all(leaf.grad is not None and leaf.grad.dtype == torch.float32 for leaf in graph.leaves.values())
        graph.backward_to_decoders()
        with pytest.raises(RuntimeError, match="exactly one"):
            graph.backward_to_decoders()
    assert not graph.decoded and not graph.leaves
    assert losses == old_losses
    per_decoder = {}
    for prefix in PROJECTIONS:
        names = [name for name in params if name.startswith(prefix + ".")]
        actual = torch.cat([params[name].grad.flatten() for name in names])
        expected = torch.cat([old_params[name].grad.flatten() for name in names])
        per_decoder[prefix] = _relative(actual, expected)
        assert per_decoder[prefix] <= 2e-5
        for name in names:
            atol = max(1e-9, float(old_params[name].grad.abs().max()) * 2e-6)
            torch.testing.assert_close(params[name].grad, old_params[name].grad, rtol=3e-5, atol=atol)
    assert all(p.grad is None for name, p in candidate.named_parameters() if name not in params)
    assert all(batch.hidden_states.grad is None and batch.target_hidden_states.grad is None for batch in batches)
    old_optimizer.step()
    optimizer.step()
    max_parameter_error = 0.0
    for name, value in candidate.state_dict().items():
        if name not in params:
            assert torch.equal(value, initial[name]), name
        else:
            expected = reference.state_dict()[name]
            max_parameter_error = max(max_parameter_error, float((value - expected).abs().max()))
            torch.testing.assert_close(value, expected, rtol=0, atol=1e-7)
    for prefix in PROJECTIONS:
        names = [name for name in params if name.startswith(prefix + ".")]
        actual = torch.cat([(params[name].detach() - initial[name]).flatten() for name in names])
        expected = torch.cat([(old_params[name].detach() - initial[name]).flatten() for name in names])
        assert _relative(actual, expected) <= 2e-4
    assert ids == {name: id(p) for name, p in candidate.named_parameters()}
    assert flags == {name: p.requires_grad for name, p in candidate.named_parameters()}
    assert all("forward" not in module.__dict__ for _, module in targets)
    assert all(state[key].dtype == torch.float32 for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq"))
    print(json.dumps({"checkpoint": checkpoint, "gradient_relative_l2_by_decoder": per_decoder,
                      "adam_parameter_max_abs_difference": max_parameter_error}))


def test_fixed_endpoint_decodes_once_and_restores_forwards_on_failure():
    block, train, heldout = _fixture()
    targets, parameters = _prepare(block, True)
    before = deepcopy(block.state_dict())
    expected = []
    with torch.no_grad(), _projection_forwards(targets):
        for batch in train + heldout:
            expected.append(float(_squared_error(block, batch, torch.device("cpu"))))
    decode_calls = {name: 0 for name in PROJECTIONS}
    handles = []

    def count(name):
        def hook(_module, _args):
            decode_calls[name] += 1
        return hook

    for name, module in targets:
        handles.append(module._parallel_stage_decoder.register_forward_pre_hook(count(name)))
    try:
        with pytest.raises(RuntimeError, match="caller cancelled"), torch.no_grad(), block_weight_graph(targets, requires_grad=False) as graph:
            actual = [float(_squared_error(block, batch, torch.device("cpu"))) for batch in train + heldout]
            assert actual == expected
            assert all(not weight.requires_grad for weight in graph.decoded.values())
            assert all(not leaf.requires_grad and leaf.grad is None for leaf in graph.leaves.values())
            assert set(decode_calls.values()) == {1}
            raise RuntimeError("caller cancelled")
    finally:
        for handle in handles:
            handle.remove()
    assert all("forward" not in module.__dict__ for _, module in targets)
    assert all(p.grad is None for p in parameters.values())
    assert all(torch.equal(value, before[name]) for name, value in block.state_dict().items())
    assert not graph.decoded and not graph.leaves
