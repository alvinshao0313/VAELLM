from copy import deepcopy
from io import BytesIO

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.joint_block import _bit_configurations, optimize_joint_block
from experiments.qwen3_w25.local_bits import _changed_bits, calibrate_local_bits_cpu
from experiments.qwen3_w25.local_block import BlockBatch, PROJECTIONS, _squared_error
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from test_qwen3_w25_local_block import _fixture


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA/Triton SparseBit required")


def _arguments():
    return dict(steps=16, decoder_lr=0.0002, seed=31,
                bit_lrs={name: 0.01 for name in PROJECTIONS},
                round_steps={name: 8 for name in PROJECTIONS})


def _identity(block):
    return ({name: id(p) for name, p in block.named_parameters()},
            {name: p.requires_grad for name, p in block.named_parameters()},
            {name: child.training for name, child in block.named_modules()})


def _assert_clean(block, identity):
    assert _identity(block) == identity
    assert all(p.grad is None for p in block.parameters())
    assert not hasattr(block, "sparse_bit_tuning")
    for name in PROJECTIONS:
        module = block.get_submodule(name)
        assert not hasattr(module, "_sparse_bit_binding") and "forward" not in module.__dict__
        assert not hasattr(module, "_decoder_compute_dtype")
        assert not module.trainable_decode and module.cache_decoded_weight and module.parallel_stage_decode


def test_invalid_explicit_configuration_and_cpu_fail_before_mutation():
    block, train, heldout = _fixture(protected_int8=True)
    block.train()
    block.get_submodule(PROJECTIONS[0])._parallel_stage_decoder.eval()
    original, identity = deepcopy(block.state_dict()), _identity(block)
    cases = [
        ({"bit_lrs": {}}, "exactly the seven"),
        ({"round_steps": {"unknown": 3}}, "exactly the seven"),
        ({"bit_lrs": dict.fromkeys(PROJECTIONS, "auto")}, "finite positive"),
        ({"bit_lrs": dict.fromkeys(PROJECTIONS, float("nan"))}, "finite positive"),
        ({"bit_lrs": dict.fromkeys(PROJECTIONS, True)}, "finite positive"),
        ({"round_steps": dict.fromkeys(PROJECTIONS, 0)}, "positive integers"),
        ({"round_steps": dict.fromkeys(PROJECTIONS, True)}, "positive integers"),
        ({"seed": True}, "seed must"),
        ({"steps": 0}, "steps must"),
        ({"decoder_lr": 0}, "lr finite"),
        ({}, "real CUDA/Triton"),
    ]
    for changes, message in cases:
        with pytest.raises((ValueError, RuntimeError), match=message):
            optimize_joint_block(block, train, heldout, **(_arguments() | changes))
        _assert_clean(block, identity)
        for name, tensor in block.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_configuration_preserves_independent_explicit_projection_budgets():
    rates = {name: 0.001 * (index + 1) for index, name in enumerate(PROJECTIONS)}
    rounds = {name: 7 + index for index, name in enumerate(PROJECTIONS)}
    configurations = _bit_configurations(rates, rounds, 31)
    for name, config in configurations.items():
        assert config.resolved_lr() == rates[name] and config.round_steps == rounds[name]
        assert config.active_ratio == 1 and config.proxy_coordinates == "decoder_sensitivity"
        assert config.optimizer == "adamw" and config.weight_decay == 0


def test_other_source_profile_is_rejected_without_mutation():
    block, train, heldout = _fixture()
    original, identity = deepcopy(block.state_dict()), _identity(block)
    with pytest.raises(ValueError, match="no-rotation INT8"):
        optimize_joint_block(block, train, heldout, **_arguments())
    _assert_clean(block, identity)
    for name, tensor in block.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def _cuda_fixture():
    # Keep the production decoder dimensions. The older CPU-only block fixture's
    # K8/D4/H8 is outside the native fused CUDA matmul kernel's supported tiles.
    torch.manual_seed(1984)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, max_position_embeddings=32, attention_dropout=0.0,
                        attention_bias=True)
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    block = deepcopy(teacher.model.layers[1])
    for name in PROJECTIONS:
        linear = block.get_submodule(name)
        latent = 48 if name in ("self_attn.o_proj", "mlp.down_proj") else 32
        decoders = [Decoder(in_dim=latent, out_dim=32, hidden_dim=128, num_res_blocks=0,
                            norm_type="layer", decoder_type="symmetric", use_checkpoint=True).bfloat16()
                    for _ in range(2)]
        with torch.no_grad():
            for decoder in decoders:
                decoder.linear_out.linear.weight.mul_(0.08)
                decoder.linear_out.linear.bias.mul_(0.08)
        indices = torch.arange(0, linear.in_features, linear.in_features // 32)
        qvalues, scales = encode_protected_channel_weight(linear.weight[:, indices].T, quant_format="int8")
        compressed = linear.in_features - len(indices)
        module = VAELinear(
            in_features=linear.in_features, out_features=linear.out_features, bias=deepcopy(linear.bias),
            codebook_dim=32, compressed_in_features=compressed,
            transpose=name in ("self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"),
            stage_vq_weights=[torch.randint(0, 2, (linear.out_features * compressed // 32, 1, latent),
                                           dtype=torch.bool) for _ in range(2)],
            stage_decoders=decoders, protected_input_indices=indices, protected_input_qvalues=qvalues,
            protected_input_scales=scales, protected_channel_quant_format="int8")
        module.pack_parallel_stage_decoder_(trainable=False)
        block.set_submodule(name, module)
        altered = deepcopy(module)
        with torch.no_grad():
            altered._parallel_stage_decoder.linear_out.conv.weight.add_(
                0.003 * torch.randn_like(altered._parallel_stage_decoder.linear_out.conv.weight))
            teacher.get_submodule("model.layers.1." + name).weight.copy_(altered._decode_weight(dtype=torch.bfloat16))
    block.eval()

    def capture(ids, mask):
        captured = {}

        def before(_module, args, kwargs):
            captured.update(inputs=args[0].detach().clone(), kwargs=deepcopy(kwargs))

        def after(_module, _args, output):
            captured["target"] = output[0].detach().clone()

        first = teacher.model.layers[1].register_forward_pre_hook(before, with_kwargs=True)
        last = teacher.model.layers[1].register_forward_hook(after)
        try:
            with torch.no_grad():
                teacher.model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask), use_cache=False)
        finally:
            first.remove()
            last.remove()
        return BlockBatch(captured["inputs"], captured["target"], captured["kwargs"], torch.tensor(mask))

    train = [capture([[3, 4, 5, 0], [6, 7, 8, 9]], [[1, 1, 1, 0], [1, 1, 1, 1]]),
             capture([[10, 11, 12]], [[1, 1, 1]])]
    heldout = [capture([[13, 14, 15, 16, 17]], [[1, 1, 1, 1, 1]])]
    rates = {name: max(stage["scale"] for stage in calibrate_local_bits_cpu(
        block.get_submodule(name))["stages"]) / 2 for name in PROJECTIONS}
    block.cuda()
    for name in PROJECTIONS:
        block.get_submodule(name)._build_parallel_stage_decode_plan()
    return block, train, heldout, rates


@CUDA_REQUIRED
def test_real_joint_update_frozen_payload_and_native_state_roundtrip():
    block, train, heldout, rates = _cuda_fixture()
    original = {name: tensor.cpu().clone() for name, tensor in block.state_dict().items()}
    identity = _identity(block)
    for batch in train + heldout:
        batch.hidden_states.requires_grad_(True)
        batch.target_hidden_states.requires_grad_(True)
    report = optimize_joint_block(block, train, heldout, **(_arguments() | {"bit_lrs": rates}))
    evidence = report["evidence"]
    assert evidence["has_committed_code_change"] and evidence["final_changed_bits"] > 0
    assert len(evidence["code_state_keys"]) == 14
    assert evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"]
    assert evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"]
    for key in ("proxy_parameter_dtypes", "proxy_gradient_dtypes", "bit_optimizer_moment_dtypes",
                "decoder_optimizer_moment_dtypes"):
        assert evidence[key] == ["torch.float32"]
    for name, module_report in report["modules"].items():
        assert len(module_report["step_telemetry"]) == 16
        assert module_report["rounds_ended"] >= 2
        for key in ("stage_proxy_gradient_l2_max", "stage_decoder_gradient_l2_max",
                    "stage_parameter_delta_l2_fp32", "stage_parameter_delta_l2_saved"):
            assert all(value > 0 for value in module_report[key])
        module = block.get_submodule(name)
        for stage, key in enumerate(module_report["code_state_keys"]):
            shape = module.get_stage_part_vq_spec(stage, 0)["logical_shape"]
            changed = _changed_bits(original[name + "." + key], module.state_dict()[key], shape)
            assert changed == module_report["stage_changed_bits"][stage]
    for name, tensor in block.state_dict().items():
        assert tensor.dtype == original[name].dtype
        if name not in evidence["allowed_state_keys"]:
            torch.testing.assert_close(tensor.cpu(), original[name], rtol=0, atol=0)
    _assert_clean(block, identity)
    assert all(batch.hidden_states.grad is None and batch.target_hidden_states.grad is None for batch in train + heldout)
    with torch.no_grad():
        measured = sum(float(_squared_error(block, batch, torch.device("cuda:0"))) for batch in heldout)
    assert measured / (5 * block.hidden_size) == report["after_native"]["heldout_mse"]
    saved = BytesIO()
    torch.save(block.state_dict(), saved)
    saved.seek(0)
    restored, _, _, _ = _cuda_fixture()
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    for name in PROJECTIONS:
        restored.get_submodule(name)._build_parallel_stage_decode_plan()
        restored.get_submodule(name).clear_decoded_weight_cache()
    for name, tensor in block.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        assert float(_squared_error(restored, heldout[0], torch.device("cuda:0"))) == measured


@CUDA_REQUIRED
def test_real_joint_callback_failure_restores_the_entire_block():
    block, train, heldout, rates = _cuda_fixture()
    block.train()
    block.get_submodule(PROJECTIONS[0])._parallel_stage_decoder.eval()
    original = {name: tensor.cpu().clone() for name, tensor in block.state_dict().items()}
    identity = _identity(block)

    def cancel(row):
        if sum(value["step_flip_count"] for value in row["modules"].values()) > 0:
            raise RuntimeError("cancel after real joint update")

    with pytest.raises(RuntimeError, match="cancel after real joint update"):
        optimize_joint_block(block, train, heldout, progress_callback=cancel,
                             **(_arguments() | {"bit_lrs": rates}))
    _assert_clean(block, identity)
    for name, tensor in block.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), original[name], rtol=0, atol=0)


@CUDA_REQUIRED
def test_zero_final_code_change_is_reported_without_rejecting_decoder_updates():
    block, train, heldout, _rates = _cuda_fixture()
    report = optimize_joint_block(block, train, heldout, **(
        _arguments() | {"bit_lrs": dict.fromkeys(PROJECTIONS, 1e-12)}))
    assert not report["evidence"]["has_committed_code_change"]
    assert report["evidence"]["final_changed_bits"] == 0
    assert all(value > 0 for item in report["modules"].values() for value in item["stage_parameter_delta_l2_saved"])
