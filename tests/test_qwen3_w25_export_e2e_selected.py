"""CPU contract and real decoder serialization checks; no SparseBit/GPU emulation."""

from argparse import Namespace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload, _finalize_decoders
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, collect_e2e_mutable_state
from experiments.qwen3_w25.assemble_categories import CATEGORIES, _assert_state_equal
from experiments.qwen3_w25.export_e2e_selected import restore_decoder_state, validate_configuration
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.config.configs import (
    AuxTrainableConfig, DistillDataConfig, DistillLossConfig, DistillOptimizationConfig,
    DistillRuntimeConfig, LoRAConfig,
)
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.train_args import HFArguments, TrainingArguments


def _configuration(mode="decoder", *, total_steps=2000, selected_step=400, coordinates="unit", eval_after_save=False, parallel_mode="dp"):
    cfg = Namespace(train_mode=mode, data=DistillDataConfig(seed=31), loss=DistillLossConfig(),
        opt=DistillOptimizationConfig(steps=total_steps, distill_fp32_components=("decoder",)),
        aux=AuxTrainableConfig(), runtime=DistillRuntimeConfig(parallel_mode=parallel_mode), lora=LoRAConfig(),
        target_layers=list(range(36)), target_modules="all", bit_active_ratio=0.03,
        bit_optimizer="rms_sgd", bit_lr="0.05", bit_round_steps="58", bit_weight_decay=0.0,
        bit_proxy_coordinates=coordinates)
    cfg.runtime.evaluation.eval_after_save = eval_after_save
    training = TrainingArguments(output_dir="unused-cpu-export-fixture", use_cpu=True, report_to=[],
        max_steps=total_steps, seed=31, bf16=True, fp16=False, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", eval_on_start=False, dataloader_num_workers=8,
        dataloader_drop_last=False, save_strategy="steps", save_steps=400)
    assert training.eval_strategy is IntervalStrategy.NO
    assert training.save_strategy is SaveStrategy.STEPS
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=training, tokenizer=tokenizer,
        input_checkpoint_id="cpu-contract-base", resolved_target_layers=tuple(range(36)),
        resolved_target_modules=CATEGORIES, teacher_identity=None)
    contract["lora"] = None
    snapshot = json.loads(json.dumps(_build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=HFArguments(), training_args=training, run_output_dir=training.output_dir)))
    meta = {"train_mode": mode, "lora_config": None, "norm_train_mode": "none", "lm_head_train_mode": "none",
            "round_base_checkpoint_id": "cpu-contract-base", "target_layers": list(range(36)),
            "target_modules": list(CATEGORIES), "immutable_resume_contract": contract}
    return meta, {"global_step": selected_step, "max_steps": total_steps}, snapshot


@pytest.mark.parametrize("mode", ["decoder", "decoder_sparse_bit"])
@pytest.mark.parametrize("parallel_mode", ["dp", "layer_mp"])
def test_contract_keeps_original_schedule_and_rejects_topology_or_math_changes(mode, parallel_mode):
    meta, state, snapshot = _configuration(mode, parallel_mode=parallel_mode)
    before = deepcopy((meta, state, snapshot))
    assert validate_configuration(meta, state, snapshot) == meta["immutable_resume_contract"]
    assert (meta, state, snapshot) == before
    for changed_state, changed_snapshot in (
        ({**state, "max_steps": 400}, snapshot),
        (state, {**snapshot, "canonical_config": {**snapshot["canonical_config"], "opt": {
            **snapshot["canonical_config"]["opt"], "steps": 400}}}),
        (state, {**snapshot, "canonical_config": {**snapshot["canonical_config"], "aux": {
            **snapshot["canonical_config"]["aux"], "norm_train_mode": "all"}}}),
    ):
        with pytest.raises(ValueError):
            validate_configuration(meta, changed_state, changed_snapshot)
    for changes in ({"train_mode": "lora"}, {"lora_config": {"rank": 8}}, {"norm_train_mode": "all"}):
        with pytest.raises(ValueError):
            validate_configuration({**meta, **changes}, state, snapshot)


def test_selected_short_check_does_not_require_mid_eval_or_formal_step_budget():
    meta, state, snapshot = _configuration(total_steps=8, selected_step=4)
    assert snapshot["canonical_config"]["runtime"]["evaluation"]["eval_tasks"] is None
    assert not snapshot["canonical_config"]["runtime"]["evaluation"]["eval_after_save"]
    assert validate_configuration(meta, state, snapshot)["optimization"]["steps"] == 8


@pytest.mark.parametrize("eval_after_save", [False, True])
def test_real_training_argument_enums_match_snapshot_and_reject_evaluation_changes(eval_after_save):
    meta, state, snapshot = _configuration(eval_after_save=eval_after_save)
    execution = meta["immutable_resume_contract"]["evaluation_execution"]
    assert execution["eval_strategy"] == str(IntervalStrategy.NO).lower()
    assert snapshot["training_args"]["eval_strategy"] == "no"
    assert snapshot["training_args"]["save_strategy"] == "steps"
    assert validate_configuration(meta, state, snapshot) == meta["immutable_resume_contract"]
    changed_training = {"eval_strategy": "steps", "eval_on_start": True}
    if eval_after_save:
        assert execution["save_strategy"] == str(SaveStrategy.STEPS).lower()
        assert execution["save_steps"] == 400
        changed_training.update(save_strategy="epoch", save_steps=200)
    else:
        assert "save_strategy" not in execution and "save_steps" not in execution
    for key, value in changed_training.items():
        changed = deepcopy(snapshot)
        changed["training_args"][key] = value
        with pytest.raises(ValueError, match="immutable contract mismatch"):
            validate_configuration(meta, state, changed)
    changed = deepcopy(snapshot)
    changed["canonical_config"]["runtime"]["evaluation"]["eval_after_save"] = not eval_after_save
    with pytest.raises(ValueError, match="immutable contract mismatch"):
        validate_configuration(meta, state, changed)


def test_sensitivity_contract_keeps_saved_coordinates_without_recalibration():
    meta, state, snapshot = _configuration("decoder_sparse_bit", coordinates="decoder_sensitivity")
    assert validate_configuration(meta, state, snapshot)["sparse_bit"]["proxy_coordinates"] == "decoder_sensitivity"
    snapshot["canonical_config"]["bit_proxy_coordinates"] = "unit"
    with pytest.raises(ValueError, match="immutable contract mismatch"):
        validate_configuration(meta, state, snapshot)
    snapshot["canonical_config"]["bit_proxy_coordinates"] = "other_coordinates"
    with pytest.raises(ValueError, match="Unsupported snapshot"):
        validate_configuration(meta, state, snapshot)


def _model():
    torch.manual_seed(13)
    model = nn.Module()
    model.backbone = nn.Linear(8, 8, bias=False).bfloat16()
    protected = torch.tensor([0, 3])
    qvalues, scales = encode_protected_channel_weight(torch.randn(2, 8), quant_format="int8")
    model.proj = VAELinear(in_features=8, out_features=8, bias=None, codebook_dim=2, transpose=False,
        compressed_in_features=6, protected_input_indices=protected, protected_input_qvalues=qvalues,
        protected_input_scales=scales, protected_channel_quant_format="int8",
        stage_vq_weights=[torch.randint(0, 2, (24, 1, 4), dtype=torch.bool) for _ in range(2)],
        stage_decoders=[Decoder(in_dim=4, out_dim=2, hidden_dim=8, num_res_blocks=0,
            norm_type="layer", decoder_type="symmetric").bfloat16() for _ in range(2)])
    model.proj.pack_parallel_stage_decoder_(trainable=False)
    return model.eval()


def test_real_decoder_mutable_restore_bf16_export_and_strict_reload(tmp_path):
    _meta, _state, snapshot = _configuration()
    model = _model()
    original = deepcopy(model.state_dict())
    base = tmp_path / "base"
    saved_base = v6.save_v6_full_checkpoint(model, str(base), checkpoint_kind="round_base",
        compressed_targets=("proj",), base_model_path="cpu-real-decoder-fixture", save_config=False)
    selection = build_model_level_trainable_selection(model, aux=AuxTrainableConfig(),
        compressed_modules=(("proj", model.proj),), rank=8, alpha=16, dropout=0,
        train_decoder=True, train_lora=False, fp32_components=("decoder",), freeze=True)
    configure_distill_precision(selection, components=("decoder",), training_args=SimpleNamespace(bf16=True, fp16=False))
    with torch.no_grad():
        next(iter(selection.decoder_parameters.values())).add_(0.125)
    mutable, _classes, manifest = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=(("proj", model.proj),))
    mutable = {key: tensor.detach().clone() for key, tensor in mutable.items()}
    step = tmp_path / "step"
    v6.save_v6_training_step_payload(str(step), round_base_ref=str(base),
        round_base_checkpoint_id=saved_base["checkpoint_id"], mutable_state=mutable,
        mutable_state_manifest=manifest, compressed_targets=("proj",), train_mode="decoder",
        base_model_path="cpu-real-decoder-fixture", immutable_resume_contract={"cpu_serialization_test": True})

    restored, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(base), strict=True)
    selected = (("proj", restored.proj),)
    assert restore_decoder_state(restored, selected, checkpoint=step, snapshot=snapshot) == len(mutable)
    _assert_state_equal({key: restored.state_dict()[key] for key in mutable}, mutable, "Real restored FP32 mutable state")
    assert all(restored.state_dict()[key].dtype == torch.float32 for key in mutable)
    frozen = {key: tensor for key, tensor in original.items() if key not in mutable}
    _assert_state_equal({key: restored.state_dict()[key] for key in frozen}, frozen, "Frozen packed/INT8/backbone")
    _finalize_decoders(selected)
    prepare_model_export(restored, SimpleNamespace(bf16=True, fp16=False))
    restored.requires_grad_(False)
    expected = deepcopy(restored.state_dict())
    assert any(not torch.equal(expected[key], original[key]) for key in mutable)
    final = tmp_path / "final"
    saved_final = v6.save_v6_full_checkpoint(restored, str(final), checkpoint_kind="final_model",
        compressed_targets=("proj",), base_model_path="cpu-real-decoder-fixture", save_config=False)
    fresh, meta, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(final), strict=True)
    assert meta["checkpoint_id"] == saved_final["checkpoint_id"] != saved_base["checkpoint_id"]
    _assert_state_equal(fresh.state_dict(), expected, "Fresh native BF16 strict reload")
    _assert_state_equal({key: fresh.state_dict()[key] for key in frozen}, frozen, "Frozen final payload")
