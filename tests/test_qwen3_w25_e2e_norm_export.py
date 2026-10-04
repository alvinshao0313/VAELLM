"""Real CPU final-norm/decoder state and BF16 projection contracts; not an 8B/GPU substitute."""

from argparse import Namespace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload, _finalize_decoders
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, collect_e2e_mutable_state
from experiments.qwen3_w25.assemble_categories import CATEGORIES, _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _validate_export_storage
from experiments.qwen3_w25.norm_step_export_callback import stage_commands
from experiments.qwen3_w25.e2e_norm_audit import audit_norm_state
from experiments.qwen3_w25.e2e_norm_export import NORM_KEY, restore_norm_state, validate_configuration
from experiments.qwen3_w25.local_bits import _code_banks
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.config.configs import (
    AuxTrainableConfig, DistillDataConfig, DistillLossConfig, DistillOptimizationConfig,
    DistillRuntimeConfig, LoRAConfig,
)
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.train_args import HFArguments, TrainingArguments


TARGET = "model.layers.0.self_attn.q_proj"
TARGETS = tuple("model.layers.0." + ("mlp." if name in ("gate_proj", "up_proj", "down_proj") else "self_attn.")
                + name for name in CATEGORIES)


def _configuration():
    cfg = Namespace(train_mode="decoder", data=DistillDataConfig(seed=0), loss=DistillLossConfig(),
        opt=DistillOptimizationConfig(steps=2000, distill_fp32_components=("decoder", "norm")),
        aux=AuxTrainableConfig(norm_train_mode="final", norm_lr=1e-4),
        runtime=DistillRuntimeConfig(parallel_mode="dp"), lora=LoRAConfig(),
        target_layers=list(range(36)), target_modules="all", bit_active_ratio=0.05,
        bit_optimizer="adam", bit_lr="4e-5", bit_round_steps="128", bit_weight_decay=0.0,
        bit_proxy_coordinates="decoder_sensitivity")
    training = TrainingArguments(output_dir="unused-cpu-norm-fixture", use_cpu=True, report_to=[],
        max_steps=2000, seed=0, bf16=True, fp16=False, tf32=False, eval_strategy="no",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=400, save_total_limit=1)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=training, tokenizer=tokenizer,
        input_checkpoint_id="cpu-norm-base", resolved_target_layers=tuple(range(36)),
        resolved_target_modules=CATEGORIES, teacher_identity=None)
    contract["lora"] = None
    snapshot = json.loads(json.dumps(_build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=HFArguments(), training_args=training, run_output_dir=training.output_dir)))
    meta = {"train_mode": "decoder", "lora_config": None, "norm_train_mode": "final", "lm_head_train_mode": "none",
            "round_base_checkpoint_id": "cpu-norm-base", "target_layers": list(range(36)),
            "target_modules": list(CATEGORIES), "immutable_resume_contract": contract}
    return meta, {"global_step": 400, "max_steps": 2000}, snapshot


def test_contract_matches_actual_enums_and_preserves_schedule():
    meta, state, snapshot = _configuration()
    before = deepcopy((meta, state, snapshot))
    assert validate_configuration(meta, state, snapshot) == meta["immutable_resume_contract"]
    assert (meta, state, snapshot) == before
    for key, value in (("norm_train_mode", "all"), ("lm_head_train_mode", "full"),
                       ("norm_lr", 3e-5), ("residual_lora_mode", "additive")):
        changed = deepcopy(snapshot)
        changed["canonical_config"]["aux"][key] = value
        with pytest.raises(ValueError):
            validate_configuration(meta, state, changed)
    for key, value in (("train_mode", "decoder_sparse_bit"), ("lora_config", {"rank": 8})):
        with pytest.raises(ValueError):
            validate_configuration({**meta, key: value}, state, snapshot)
    with pytest.raises(ValueError):
        validate_configuration(meta, {**state, "max_steps": 400}, snapshot)
    changed = deepcopy(snapshot)
    changed["canonical_config"]["opt"]["distill_fp32_components"] = ["decoder"]
    with pytest.raises(ValueError):
        validate_configuration(meta, state, changed)


def _model():
    torch.manual_seed(97)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1, head_dim=64,
        tie_word_embeddings=False)).bfloat16()
    model.config.use_cache = False
    for name in TARGETS:
        dense = model.get_submodule(name)
        inputs, outputs = dense.in_features, dense.out_features
        protected = torch.arange(32, dtype=torch.int64) * 7
        qvalues, scales = encode_protected_channel_weight(torch.randn(32, outputs), quant_format="int8")
        module = VAELinear(in_features=inputs, out_features=outputs, bias=None, codebook_dim=32, transpose=False,
            compressed_in_features=inputs - 32, protected_input_indices=protected, protected_input_qvalues=qvalues,
            protected_input_scales=scales, protected_channel_quant_format="int8",
            stage_vq_weights=[torch.randint(0, 2, (outputs * (inputs - 32) // 32, 1, 8), dtype=torch.bool) for _ in range(2)],
            stage_decoders=[Decoder(in_dim=8, out_dim=32, hidden_dim=8, num_res_blocks=0,
                norm_type="layer", decoder_type="symmetric").bfloat16() for _ in range(2)])
        module.pack_parallel_stage_decoder_(trainable=False)
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, module)
    return model


def _real_export(tmp_path):
    model = _model()
    original = deepcopy(model.state_dict())
    base = tmp_path / "base"
    saved_base = v6.save_v6_full_checkpoint(model, str(base), checkpoint_kind="final_model",
        compressed_targets=TARGETS, base_model_path="cpu-real-norm-fixture")
    selected = tuple((name, model.get_submodule(name)) for name in TARGETS)
    inventory = {name: sorted(name + "._parallel_stage_decoder." + key
                             for key in module._parallel_stage_decoder.state_dict()) for name, module in selected}
    banks = {name: [(name + "." + key, shape) for key, shape in _code_banks(module)] for name, module in selected}
    selection = build_model_level_trainable_selection(model,
        aux=AuxTrainableConfig(norm_train_mode="final", norm_lr=1e-4), compressed_modules=selected,
        rank=8, alpha=16, dropout=0, train_decoder=True, train_lora=False,
        fp32_components=("decoder", "norm"), freeze=True)
    args = SimpleNamespace(bf16=True, fp16=False)
    configure_distill_precision(selection, components=("decoder", "norm"), training_args=args)
    assert len(selection.norm_parameters) == 1 and not selection.lm_head_parameters and not selection.lora_parameters
    assert not model.lm_head.weight.requires_grad and model.model.norm.weight.requires_grad
    optimizer = torch.optim.AdamW([
        {"params": list(selection.decoder_parameters.values()), "lr": 3e-5},
        {"params": list(selection.norm_parameters.values()), "lr": 1e-4},
    ], weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float().square().mean()
        loss.backward()
        optimizer.step()
    mutable, classes, manifest = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    assert set(classes.values()) == {"decoder", "norm"}
    mutable = {key: value.detach().clone() for key, value in mutable.items()}
    step = tmp_path / "step"
    v6.save_v6_training_step_payload(str(step), round_base_ref=str(base),
        round_base_checkpoint_id=saved_base["checkpoint_id"], mutable_state=mutable,
        mutable_state_manifest=manifest, compressed_targets=TARGETS, train_mode="decoder", norm_train_mode="final",
        base_model_path="cpu-real-norm-fixture", immutable_resume_contract={"cpu_serialization_test": True})
    restored, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(base), strict=True)
    selected = tuple((name, restored.get_submodule(name)) for name in TARGETS)
    _, _, snapshot = _configuration()
    report = restore_norm_state(restored, selected, checkpoint=step, snapshot=snapshot)
    assert report["tensors_compared"] == len(mutable)
    assert report["component_counts"]["norm"] == 1
    _assert_state_equal({key: restored.state_dict()[key] for key in mutable}, mutable, "Actual FP32 restore")
    assert all(restored.state_dict()[key].dtype == torch.float32 for key in mutable)
    _finalize_decoders(selected)
    selected_norm = restored.model.norm.weight.detach().clone()
    prepare_model_export(restored, args)
    final = tmp_path / "final"
    restored.eval().requires_grad_(False)
    saved = v6.save_v6_full_checkpoint(restored, str(final), checkpoint_kind="final_model",
        compressed_targets=TARGETS, base_model_path="cpu-real-norm-fixture", train_mode="decoder",
        norm_train_mode="final")
    fresh, meta, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(final), strict=True)
    assert not meta["post_norm_head_linear"] and meta["checkpoint_id"] == saved["checkpoint_id"]
    actual = torch.load(final / "pytorch_model.bin", weights_only=True, map_location="cpu")
    _assert_state_equal(actual, fresh.state_dict(), "Fresh BF16 strict reload")
    state_report = audit_norm_state(original, actual, inventory, banks, selected_norm)
    before, after = (audit_checkpoint(path, max_bpw=2.5, max_lora_rank=0) for path in (base, final))
    _validate_export_storage(before, after, state_report)
    return original, actual, inventory, banks, selected_norm, state_report


def test_real_update_restore_freeze_and_native_bf16_projection(tmp_path):
    original, actual, inventory, banks, selected_norm, report = _real_export(tmp_path)
    assert report["changed_norm_fp32_elements"] > 0 and report["changed_decoder_elements"] > 0
    assert report["changed_norm_bf16_elements"] == 0 and not report["norm_update_survived_bf16"]
    assert report["norm_matches_selected_fp32_cast_exactly"]
    assert torch.equal(actual[NORM_KEY], selected_norm.bfloat16())
    assert report["changed_code_bits"] == 0 and report["packed_codes_unchanged"]
    assert report["frozen_tensors_compared"] == len(original) - sum(len(keys) for keys in inventory.values()) - 1
    assert report["actual_tensor_storage_unchanged"] and report["non_decoder_non_final_norm_bytes_unchanged"]
    assert report["original_head_bytes_unchanged"] and torch.equal(original["lm_head.weight"], actual["lm_head.weight"])


def test_state_audit_rejects_frozen_change_wrong_projection_or_missing_update(tmp_path):
    original, actual, inventory, banks, selected_norm, _ = _real_export(tmp_path)
    for key, value in (
        (NORM_KEY, actual[NORM_KEY].float()), (NORM_KEY, actual[NORM_KEY] + 1),
        (NORM_KEY, torch.full_like(actual[NORM_KEY], float("nan"))),
        ("model.embed_tokens.weight", actual["model.embed_tokens.weight"] + 1),
        ("lm_head.weight", actual["lm_head.weight"] + 1),
        ("model.layers.0.input_layernorm.weight", actual["model.layers.0.input_layernorm.weight"] + 1),
        (banks[TARGET][0][0], actual[banks[TARGET][0][0]].bitwise_xor(1)),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            audit_norm_state(original, {**actual, key: value}, inventory, banks, selected_norm)
    with pytest.raises(ValueError):
        audit_norm_state(original, actual, inventory, banks, original[NORM_KEY].float())
    with pytest.raises(ValueError):
        audit_norm_state(original, {**actual, "lm_head.post_norm_linear.weight": torch.eye(256)},
                         inventory, banks, selected_norm)


def test_callback_uses_fixed_full_shards_and_separate_diagnostic_mode():
    kwargs = dict(checkpoint="step", source_dir="base", output_dir="out")
    commands = stage_commands(**kwargs)
    assert [name for name, _, _ in commands] == ["export", "audit", "mmlu", "seven", "summary"]
    for name, command, _ in commands:
        assert "--lm_limit" not in command
        if name in ("mmlu", "seven"):
            assert command[command.index("--lm_batch_size") + 1] == ("64" if name == "mmlu" else "32")
            assert command[command.index("--num_fewshot") + 1] == "0"
            assert "--strict" in command
    assert commands[0][1][2] == "experiments.qwen3_w25.e2e_norm_export"
    assert commands[1][1][2] == "experiments.qwen3_w25.e2e_norm_audit"
    assert commands[1][1][-2:] == ["--training_checkpoint_dir", "step"]
    diagnostic = stage_commands(**kwargs, diagnostic_eval_limit=64)
    assert len(diagnostic) == 4
    assert all("--lm_limit" in command for name, command, _ in diagnostic if name in ("mmlu", "seven"))
