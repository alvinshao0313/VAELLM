from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.local_block import (
    BlockBatch, PROJECTIONS, _squared_error, optimize_local_block,
)
from experiments.qwen3_w25.block_weight_graph import block_weight_graph
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from litebsq.protected_channel_quant import encode_protected_channel_weight
from rotation.weight_preprocess import rotation_spec


def _fixture(*, protected_int8=False):
    torch.manual_seed(1984)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, max_position_embeddings=32, attention_dropout=0.0,
                        attention_bias=True)
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    block = deepcopy(teacher.model.layers[1])
    for name in PROJECTIONS:
        linear = block.get_submodule(name)
        decoders = [Decoder(in_dim=8, out_dim=4, hidden_dim=8, num_res_blocks=0,
                            norm_type="layer", decoder_type="symmetric", use_checkpoint=True).bfloat16()
                    for _ in range(2)]
        with torch.no_grad():
            for decoder in decoders:
                decoder.linear_out.linear.weight.mul_(0.08)
                decoder.linear_out.linear.bias.mul_(0.08)
        extras = {}
        compressed_in_features = linear.in_features
        if protected_int8:
            indices = torch.tensor([0, 3, 7, 11])
            qvalues, scales = encode_protected_channel_weight(
                linear.weight[:, indices].t(), quant_format="int8")
            compressed_in_features -= len(indices)
            extras = dict(compressed_in_features=compressed_in_features,
                          protected_input_indices=indices, protected_input_qvalues=qvalues,
                          protected_input_scales=scales, protected_channel_quant_format="int8")
        module = VAELinear(
            in_features=linear.in_features, out_features=linear.out_features,
            bias=None if linear.bias is None else nn.Parameter(linear.bias.detach().clone()),
            codebook_dim=4, transpose=name == "mlp.down_proj",
            stage_vq_weights=[torch.randint(0, 2, (linear.out_features * compressed_in_features // 4, 1, 8), dtype=torch.bool)
                              for _ in range(2)],
            stage_decoders=decoders,
            weight_rotation_spec=None if protected_int8 else rotation_spec("two_sided", 0, 31, name),
            **extras,
        )
        module.pack_parallel_stage_decoder_(trainable=False)
        block.set_submodule(name, module)
        # A real dense teacher close to the compressed initial point makes this
        # tiny mathematical fixture learnable without fitting a synthetic loss.
        altered = deepcopy(module)
        with torch.no_grad():
            altered._parallel_stage_decoder.linear_out.conv.weight.add_(
                0.003 * torch.randn_like(altered._parallel_stage_decoder.linear_out.conv.weight))
            weight = altered._decode_weight(dtype=torch.bfloat16)
            teacher.get_submodule("model.layers.1." + name).weight.copy_(weight)
    block.eval()

    def capture(ids, mask):
        captured = {}

        def pre(_module, args, kwargs):
            captured["inputs"] = args[0].detach().clone()
            captured["kwargs"] = deepcopy(kwargs)

        def post(_module, _args, output):
            captured["target"] = output[0].detach().clone()

        a = teacher.model.layers[1].register_forward_pre_hook(pre, with_kwargs=True)
        b = teacher.model.layers[1].register_forward_hook(post)
        try:
            with torch.no_grad():
                teacher.model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask), use_cache=False)
        finally:
            a.remove()
            b.remove()
        return BlockBatch(captured["inputs"], captured["target"], captured["kwargs"], torch.tensor(mask))

    train = [capture([[3, 4, 5, 0], [6, 7, 8, 9]], [[1, 1, 1, 0], [1, 1, 1, 1]]),
             capture([[10, 11, 12]], [[1, 1, 1]])]
    heldout = [capture([[13, 14, 15, 16, 17]], [[1, 1, 1, 1, 1]])]
    return block, train, heldout


@pytest.mark.parametrize("protected_int8", [False, True])
def test_fp32_decode_bf16_projection_matches_real_dense_block_and_backward(protected_int8):
    block, train, _ = _fixture(protected_int8=protected_int8)
    dense = deepcopy(block)
    targets = [(name, block.get_submodule(name)) for name in PROJECTIONS]
    observed = []
    hooks = []
    for name, module in targets:
        module.enable_trainable_decode(parallel_stage_decode=True)
        module._parallel_stage_decoder.float()
        module._decoder_compute_dtype = torch.float32
        layer = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None).bfloat16()
        with torch.no_grad():
            layer.weight.copy_(module._decode_weight(dtype=torch.float32).bfloat16())
            if layer.bias is not None:
                layer.bias.copy_(module.bias)
        dense.set_submodule(name, layer)
        hooks.append(module._parallel_stage_decoder.linear_out.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].dtype)))
    batch = train[0]
    try:
        with block_weight_graph(targets, requires_grad=True) as graph, torch.autocast("cpu", dtype=torch.bfloat16):
            actual = block(batch.hidden_states, **batch.forward_kwargs)[0]
            expected = dense(batch.hidden_states, **batch.forward_kwargs)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.float().square().mean().backward()
            graph.backward_to_decoders()
    finally:
        for hook in hooks:
            hook.remove()
    assert observed and set(observed) == {torch.float32}
    for _, module in targets:
        assert "forward" not in module.__dict__
        gradients = [p.grad for p in module._parallel_stage_decoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.float().square().sum()) for g in gradients) > 0


@pytest.mark.parametrize("protected_int8", [False, True])
def test_joint_training_freezes_codes_norms_bias_and_teacher_graph_and_restores_storage(protected_int8):
    block, train, heldout = _fixture(protected_int8=protected_int8)
    before = deepcopy(block.state_dict())
    ids = {name: id(p) for name, p in block.named_parameters()}
    flags = {name: p.requires_grad for name, p in block.named_parameters()}
    for batch in train + heldout:
        batch.hidden_states.requires_grad_(True)
        batch.target_hidden_states.requires_grad_(True)
    report = optimize_local_block(block, train, heldout, steps=24, lr=0.0002)
    assert report["settings"]["train_tokens"] == 10
    assert report["settings"]["heldout_tokens"] == 5
    assert report["settings"]["projection_dtype"] == "torch.bfloat16"
    assert report["settings"]["gradient_path"] == "two_phase_fp32_weight_gradient_then_decoder_vjp"
    assert report["after_fp32_decode"]["train_mse"] < report["before_fp32_decode"]["train_mse"]
    assert report["after_cast"]["train_mse"] < report["before_native"]["train_mse"]
    evidence = report["evidence"]
    assert evidence["optimizer_moment_dtypes"] == ["torch.float32"]
    for key in ("stage_gradient_l2_max", "stage_parameter_delta_l2_fp32", "stage_parameter_delta_l2_saved"):
        assert set(evidence[key]) == set(PROJECTIONS)
        assert all(value > 0 for pair in evidence[key].values() for value in pair)
    assert evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"]
    assert evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"]
    assert ids == {name: id(p) for name, p in block.named_parameters()}
    assert flags == {name: p.requires_grad for name, p in block.named_parameters()}
    for name, value in block.state_dict().items():
        assert value.dtype == before[name].dtype
        if "._parallel_stage_decoder." not in name:
            assert torch.equal(value, before[name]), name
    for name in PROJECTIONS:
        module = block.get_submodule(name)
        assert "forward" not in module.__dict__ and not hasattr(module, "_decoder_compute_dtype")
        assert not module.trainable_decode and module.cache_decoded_weight and module._cached_weight is None
    assert all(p.grad is None for p in block.parameters())
    assert all(batch.hidden_states.grad is None and batch.target_hidden_states.grad is None for batch in train + heldout)
    with torch.no_grad():
        actual = sum(float(_squared_error(block, batch, torch.device("cpu"))) for batch in train) / (10 * 16)
    assert actual == report["after_cast"]["train_mse"]


def test_padding_is_excluded_and_callback_failure_rolls_back_exactly():
    block, train, heldout = _fixture()
    before = deepcopy(block.state_dict())
    with torch.no_grad():
        expected = _squared_error(block, train[0], torch.device("cpu"))
        train[0].target_hidden_states[0, 3] = 1000
        assert expected == _squared_error(block, train[0], torch.device("cpu"))
    block.train()
    block.self_attn.o_proj._decoder_compute_dtype = torch.bfloat16

    def stop(_report):
        raise RuntimeError("caller cancelled")

    with pytest.raises(RuntimeError, match="caller cancelled"):
        optimize_local_block(block, train, heldout, steps=2, lr=0.002, progress_callback=stop)
    assert block.training
    assert block.self_attn.o_proj._decoder_compute_dtype == torch.bfloat16
    assert all(torch.equal(value, before[name]) and value.dtype == before[name].dtype
               for name, value in block.state_dict().items())
    assert all(p.grad is None for p in block.parameters())
    assert all("forward" not in block.get_submodule(name).__dict__ for name in PROJECTIONS)


def test_missing_rope_and_lora_reject_before_mutation():
    block, train, heldout = _fixture()
    before = deepcopy(block.state_dict())
    rope = train[0].forward_kwargs.pop("position_embeddings")
    with pytest.raises(ValueError, match="position_embeddings"):
        optimize_local_block(block, train, heldout, steps=1, lr=0.001)
    train[0].forward_kwargs["position_embeddings"] = rope
    assert all(torch.equal(value, before[name]) for name, value in block.state_dict().items())
    block.self_attn.q_proj.low_rank_a = nn.Parameter(torch.zeros(16, 2, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="LoRA"):
        optimize_local_block(block, train, heldout, steps=1, lr=0.001)
