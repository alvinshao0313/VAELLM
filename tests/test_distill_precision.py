from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from e2e_common.full_lora import finalize_model_level_lora
from litebsq.autoencoder import Decoder
from litebsq.parallel_layers import ParallelLinear
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint
from train_utils.config.cli import parse_cat_cli, parse_e2e_cli
from train_utils.config.configs import AuxTrainableConfig, parse_distill_fp32_components
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state, restore_model_level_mutable_state
from train_utils.model_level_trainables import build_model_level_trainable_selection, finalize_lm_head_linear_if_needed
from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state


TARGET = "model.layers.0.self_attn.q_proj"
ALL = ("lora", "decoder", "norm", "lm_head")


def _args(dtype=torch.bfloat16):
    return SimpleNamespace(bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16)


def _model(dtype=torch.bfloat16):
    torch.manual_seed(41)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config).to(dtype=dtype)
    model.config.use_cache = False
    decoder = Decoder(in_dim=8, out_dim=8, hidden_dim=16, num_res_blocks=0,
                      norm_type="layer", decoder_type="symmetric").to(dtype=dtype)
    layer = VAELinear(in_features=16, out_features=16, bias=None, original_weight=None,
                      vq_weight=torch.randint(0, 2, (32, 1, 8), dtype=torch.bool),
                      decoder=decoder, codebook_dim=8, transpose=False)
    model.model.layers[0].self_attn.q_proj = layer
    return model, layer


def _selection(model, layer, components=ALL, dtype=torch.bfloat16, payload=None):
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(norm_train_mode="final", lm_head_train_mode="linear"),
        compressed_modules=[(TARGET, layer)], rank=2, alpha=2, dropout=0,
        train_decoder=True, train_lora=True, fp32_components=components,
        initial_low_rank_payloads=payload,
    )
    configure_distill_precision(selection, components=components, training_args=_args(dtype))
    return selection


def _step(selection, optimizer, dtype=torch.bfloat16):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None):
        output = selection.peft_model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
        loss = output.float().square().mean()
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("parser,base", [
    (parse_e2e_cli, ["--student_checkpoint_dir", "unused", "--dataset_mix", "alpaca=1"]),
    (parse_cat_cli, ["--model_path", "unused", "--compression_categories", "q_proj"]),
])
def test_cli_component_selection(parser, base):
    cfg = parser(base + ["--distill_fp32_components", "norm,decoder,norm,lora,lm_head"])
    opt = cfg.opt if hasattr(cfg, "opt") else cfg.resolve_after_category_config("q_proj").opt
    assert opt.distill_fp32_components == ALL
    cfg = parser(base)
    opt = cfg.opt if hasattr(cfg, "opt") else cfg.resolve_after_category_config("q_proj").opt
    assert opt.distill_fp32_components == ()


@pytest.mark.parametrize("value", ["none,norm", "all", "", "norm,", "encoder"])
def test_invalid_components(value):
    with pytest.raises(ValueError):
        parse_distill_fp32_components(value)


def test_selected_leaf_parameters_and_adam_states():
    model, layer = _model()
    frozen = model.model.embed_tokens.weight
    selection = _selection(model, layer, components=("norm", "lm_head"))
    assert model.model.embed_tokens.weight is frozen
    assert frozen.dtype == torch.bfloat16 and not frozen.requires_grad
    assert all(p.dtype == torch.bfloat16 for p in selection.lora_parameters.values())
    norm = next(iter(selection.norm_parameters.values()))
    before = norm.detach().clone()
    optimizer = torch.optim.AdamW([p for p in selection.peft_model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(3):
        _step(selection, optimizer)
    assert not torch.equal(norm, before)
    assert norm.grad.dtype == torch.float32
    assert optimizer.state[norm]["exp_avg"].dtype == torch.float32
    assert optimizer.state[norm]["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(norm.detach().to(torch.bfloat16), before.to(torch.bfloat16))


def test_fp32_payload_initialized_without_bf16_rounding():
    model, layer = _model()
    a = torch.full((16, 2), 0.10001, dtype=torch.float32)
    b = torch.full((2, 16), 0.20001, dtype=torch.float32)
    selection = _selection(model, layer, payload={TARGET: (a, b)})
    parameters = selection.lora_parameters
    assert any(torch.equal(p, a) for p in parameters.values())
    assert any(torch.equal(p, b) for p in parameters.values())


@pytest.mark.parametrize("collect,restore", [
    (collect_model_level_mutable_state, restore_model_level_mutable_state),
    (collect_e2e_mutable_state, restore_e2e_mutable_state),
])
def test_fp32_training_state_exact_resume(tmp_path, collect, restore):
    model, layer = _model()
    selection = _selection(model, layer)
    optimizer = torch.optim.AdamW([p for p in selection.peft_model.parameters() if p.requires_grad], lr=1e-4)
    _step(selection, optimizer)
    state, _, manifest = collect(selection.peft_model, selection=selection, selected_vae_modules=[(TARGET, layer)])
    path = tmp_path / "step.pt"
    torch.save({"model": state, "optimizer": optimizer.state_dict()}, path)
    assert all(t.dtype == torch.float32 for t in state.values())
    _step(selection, optimizer)
    expected = {k: v.detach().clone() for k, v in selection.peft_model.state_dict().items()}
    resumed, resumed_layer = _model()
    resumed_selection = _selection(resumed, resumed_layer)
    payload = torch.load(path, weights_only=True)
    restore(resumed_selection.peft_model, selection=resumed_selection,
            selected_vae_modules=[(TARGET, resumed_layer)], checkpoint_state=payload["model"],
            checkpoint_manifest=manifest)
    resumed_optimizer = torch.optim.AdamW([p for p in resumed_selection.peft_model.parameters() if p.requires_grad], lr=1e-4)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    _step(resumed_selection, resumed_optimizer)
    for name, tensor in resumed_selection.peft_model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, None])
def test_export_reload_and_category_continuation(tmp_path, dtype):
    model, layer = _model()
    selection = _selection(model, layer, dtype=dtype)
    optimizer = torch.optim.AdamW([p for p in selection.peft_model.parameters() if p.requires_grad], lr=1e-4)
    _step(selection, optimizer, dtype=dtype)
    model = finalize_model_level_lora(selection.peft_model, compressed_proxy_names=[TARGET])
    finalize_lm_head_linear_if_needed(model, lm_head_train_mode="linear")
    layer.disable_trainable_decode()
    packed_before = {k: v.clone() for k, v in model.state_dict().items() if v.dtype == torch.uint8}
    prepare_model_export(model, _args(dtype))
    model.eval()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(input_ids=inputs).logits
    saved = save_v6_full_checkpoint(model, str(tmp_path / "model"), checkpoint_kind="final_model", compressed_targets=[TARGET])
    initial, _ = _model()
    loaded, meta, _ = load_v6_full_checkpoint_into_model(initial, saved["output_dir"])
    for name, parameter in loaded.named_parameters():
        assert parameter.dtype == dict(model.named_parameters())[name].dtype
        if dtype is not None:
            assert parameter.dtype == dtype
    for name, tensor in packed_before.items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    with torch.no_grad():
        actual = loaded(input_ids=inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if dtype is None:
        assert loaded.model.norm.weight.dtype == torch.float32
        assert loaded.lm_head.weight.dtype == torch.float32
    # A subsequent category must start identically in memory and after reload.
    results = []
    for stage_model in (model, loaded):
        stage_layer = stage_model.model.layers[0].self_attn.q_proj
        payload = {TARGET: (stage_layer.low_rank_a.detach().clone(), stage_layer.low_rank_b.detach().clone())}
        stage_selection = _selection(stage_model, stage_layer, dtype=dtype, payload=payload)
        stage_optimizer = torch.optim.AdamW([p for p in stage_selection.peft_model.parameters() if p.requires_grad], lr=1e-4)
        _step(stage_selection, stage_optimizer, dtype=dtype)
        results.append({k: v.detach().clone() for k, v in stage_selection.peft_model.state_dict().items()})
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name


def test_fp16_scaler_rejects_unselected_half_parameters():
    model, layer = _model(torch.float16)
    with pytest.raises(ValueError, match="GradScaler"):
        _selection(model, layer, components=("norm",), dtype=torch.float16)


@pytest.mark.parametrize("mode", ["none", "linear", "full", "lora"])
def test_head_inventory_and_inactive_components(mode):
    model, layer = _model()
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(lm_head_train_mode=mode), compressed_modules=[(TARGET, layer)],
        rank=2, alpha=2, dropout=0, train_decoder=False, train_lora=False, fp32_components=ALL,
    )
    counts = configure_distill_precision(selection, components=ALL, training_args=_args())
    assert counts["lora"] == counts["decoder"] == counts["norm"] == 0
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in layer.parameters())
    if mode != "none":
        assert counts["lm_head"] > 0
        optimizer = torch.optim.AdamW(list(selection.lm_head_parameters.values()), lr=1e-4)
        _step(selection, optimizer)
        assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in selection.lm_head_parameters.values())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, None])
def test_dense_lora_fuses_with_one_rounding(dtype):
    from peft.tuners.lora.layer import Linear
    from e2e_common.full_lora import _merge_dense_peft_lora_into_base

    base = nn.Linear(4, 8, bias=False).bfloat16()
    layer = Linear(base, "default", r=2, lora_alpha=2)
    layer.lora_A["default"].float()
    layer.lora_B["default"].float()
    with torch.no_grad():
        layer.lora_A["default"].weight.fill_(0.10001)
        layer.lora_B["default"].weight.fill_(0.20001)
    expected = base.weight.float() + layer.lora_B["default"].weight @ layer.lora_A["default"].weight
    fused = _merge_dense_peft_lora_into_base(layer, dtype)
    assert fused.weight.dtype == (dtype or torch.float32)
    torch.testing.assert_close(fused.weight, expected.to(dtype or torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["cat", "e2e"])
def test_component_choice_is_exact_resume_constraint(tmp_path, kind):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, TrainingArguments
    from train_utils.cat_after_category_common import resolve_cat_after_category_stage
    from train_utils.cat_step_resume_v6 import build_cat_step_immutable_resume_contract, validate_cat_step_immutable_resume_contract
    from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, validate_e2e_immutable_resume_contract

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    contracts = []
    for components in ("none", "norm"):
        if kind == "e2e":
            cfg = parse_e2e_cli(["--student_checkpoint_dir", "unused", "--dataset_mix", "alpaca=1", "--distill_fp32_components", components])
            contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=args, tokenizer=tokenizer,
                input_checkpoint_id="fixed", resolved_target_layers=[0], resolved_target_modules=[TARGET], teacher_identity=None)
            validate = validate_e2e_immutable_resume_contract
        else:
            cfg = parse_cat_cli(["--model_path", "unused", "--compression_categories", "q_proj", "--dataset_mix", "alpaca=1", "--after_category_mode", "current_decoder", "--distill_fp32_components", components])
            stage = resolve_cat_after_category_stage(cat_args=cfg, training_args=args, category="q_proj", round_idx=0)
            contract = build_cat_step_immutable_resume_contract(stage=stage, trainer_args=args, tokenizer=tokenizer,
                round_base_checkpoint_id="fixed", active_category="q_proj", round_base_meta={},
                lora_target_names=[], decoder_target_names=[TARGET], teacher_identity=None, dataset_identity={}, lora_config=None)
            validate = validate_cat_step_immutable_resume_contract
        contracts.append(contract)
    assert "distill_fp32_components" not in contracts[0]["optimization"]
    assert contracts[1]["optimization"]["distill_fp32_components"] == ["norm"]
    validate(contracts[0], contracts[0])
    with pytest.raises(ValueError, match="immutable contract mismatch"):
        validate(contracts[0], contracts[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_parallel_bias_does_not_promote_activation(dtype):
    layer = ParallelLinear(8, 16, num_models=2).float()
    x = torch.randn(12, 2, 8, dtype=dtype, requires_grad=True)
    out = layer(x)
    assert out.dtype == dtype
    out.float().square().mean().backward()
    assert layer.conv.weight.grad.dtype == torch.float32
    assert layer.conv.bias.grad.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for packed kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("stages,decoder_type,checkpoint,norm", [
    (1, "linear", False, "layer"), (2, "symmetric", True, "layer"), (2, "asymmetric", True, "layer"),
    (1, "symmetric", True, "layer"), (1, "symmetric", True, "rms"), (2, "symmetric", True, "rms"),
    (1, "symmetric", True, "group"), (2, "symmetric", True, "group"),
    (1, "symmetric", False, "batch"), (2, "symmetric", False, "batch"),
])
def test_decoder_cuda_parameter_activation_precision(dtype, stages, decoder_type, checkpoint, norm):
    from train_utils.distill_precision import install_precision_runtime
    bits = [torch.randint(0, 2, (32, 1, 8), dtype=torch.bool) for _ in range(stages)]
    decoders = [Decoder(in_dim=8, out_dim=8, hidden_dim=16, num_res_blocks=1,
                        norm_type=norm, decoder_type=decoder_type, use_checkpoint=checkpoint).float()
                for _ in range(stages)]
    layer = VAELinear(in_features=16, out_features=16, bias=None, original_weight=None,
                      vq_weight=None, decoder=None, stage_vq_weights=bits, stage_decoders=decoders,
                      codebook_dim=8, stage_codebook_dims=[8] * stages, transpose=False).cuda()
    layer.pack_parallel_stage_decoder_(trainable=True)
    layer.enable_trainable_decode(parallel_stage_decode=True)
    install_precision_runtime(layer, dtype)
    x = torch.randn(3, 16, device="cuda", dtype=dtype, requires_grad=True)
    layer.reset_fuse_stats()
    output = layer(x)
    assert output.dtype == dtype
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=16)
    scaler.scale(output.float().square().mean()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in layer.parameters() if p.requires_grad)
    assert all(state["exp_avg"].dtype == torch.float32 and state["exp_avg_sq"].dtype == torch.float32 for state in optimizer.state.values())
    assert layer.get_fuse_stats()["packed_u8_linear_hit"] > 0
    with torch.no_grad():
        assert layer(x).dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for packed kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_decoder_sparse_bit_joint_gradients(dtype):
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager

    model, layer = _model(dtype)
    model.cuda()
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(), compressed_modules=[(TARGET, layer)],
        rank=2, alpha=2, dropout=0, train_decoder=True, train_lora=False,
        decoder_execution_mode="decoder_sparse_bit",
    )
    configure_distill_precision(selection, components=("decoder",), training_args=_args(dtype))
    manager = SparseBitTuningManager(
        root_model=model, targets=[(TARGET, layer)], target_devices={TARGET: torch.device("cuda:0")},
        training_seed=42, config=SparseBitTuningConfig(enabled=True, active_ratio=0.5, round_steps=5),
        streaming=False,
    )
    manager.initialize_scores()
    score_dtypes = [p.dtype for p in manager.score_module.parameters()]
    out = layer(torch.randn(3, 16, device="cuda", dtype=dtype, requires_grad=True))
    out.float().square().mean().backward()
    assert out.dtype == dtype
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in selection.decoder_parameters.values())
    assert all(p.grad is not None and p.grad.dtype == p.dtype for p in manager.score_module.parameters())
    assert score_dtypes == [p.dtype for p in manager.score_module.parameters()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused prewarm")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_decoder_grouped_prewarm(dtype):
    from litebsq.vae_linear_prewarm import prime_named_vae_linear_cache
    from train_utils.distill_precision import install_precision_runtime

    layers = []
    for _ in range(2):
        bits = [torch.randint(0, 2, (16, 1, 16), dtype=torch.bool) for _ in range(2)]
        decoders = [Decoder(in_dim=16, out_dim=16, hidden_dim=32, num_res_blocks=0,
                            norm_type="layer", decoder_type="symmetric").float() for _ in range(2)]
        layer = VAELinear(in_features=16, out_features=16, bias=None, original_weight=None,
                          vq_weight=None, decoder=None, stage_vq_weights=bits, stage_decoders=decoders,
                          stage_codebook_dims=[16, 16], codebook_dim=16, transpose=False).cuda().eval()
        install_precision_runtime(layer, dtype)
        layers.append(layer)
    result = prime_named_vae_linear_cache([(f"model.layers.{i}.q_proj", m) for i, m in enumerate(layers)], group_size=2)
    assert result["warmed"] == 2
    for layer in layers:
        assert layer._cached_weight.dtype == dtype
        assert all(p.dtype == torch.float32 for p in layer.parameters())
