"""Native subset restore contracts on CPU; actual GPU training is a separate short run.

The exact sidecar uses real manager layout/samplers and the independent packed
reference, since production score projection is a CUDA kernel. It does not claim
that a CPU fixture executes the GPU optimizer or an actual round transition.
"""
from argparse import Namespace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload, _finalize_decoders
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract, collect_e2e_mutable_state
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
from experiments.qwen3_w25.e2e_subset_export import restore_decoder_state, selected_names, validate_configuration
from experiments.qwen3_w25.subset_step_export_callback import stage_commands
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.exact_checkpoint import EXACT_FORMAT, EXACT_VERSION, exact_state_path
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.reference import apply_bank_scores_reference, read_bank_hard_bits, scores_from_hard_bits
from train_utils import checkpoint_v6 as v6
from train_utils.config.configs import AuxTrainableConfig, DistillDataConfig, DistillLossConfig, DistillOptimizationConfig, DistillRuntimeConfig, LoRAConfig
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.train_args import HFArguments, TrainingArguments
from test_qwen3_w25_e2e_aux_export import _model, TARGETS


TARGET = "model.layers.0.mlp.down_proj"


def _configuration(layers=(0,)):
    cfg = Namespace(train_mode="decoder_sparse_bit", data=DistillDataConfig(seed=0),
        loss=DistillLossConfig(loss_type="kd_top_mass", alpha=0.5, prompt_loss_weight=0),
        opt=DistillOptimizationConfig(steps=2000, distill_fp32_components=("decoder",)),
        aux=AuxTrainableConfig(), runtime=DistillRuntimeConfig(parallel_mode="dp"), lora=LoRAConfig(),
        target_layers=list(layers), target_modules=["down_proj"], bit_active_ratio=1.0,
        bit_optimizer="adam", bit_lr="4e-5", bit_round_steps="512", bit_weight_decay=0.0,
        bit_proxy_coordinates="decoder_sensitivity")
    training = TrainingArguments(output_dir="unused-cpu-subset-fixture", use_cpu=True, report_to=[],
        max_steps=2000, seed=0, bf16=True, fp16=False, tf32=False, eval_strategy="no",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=2, save_total_limit=1, warmup_ratio=0.01)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=training, tokenizer=tokenizer,
        input_checkpoint_id="cpu-subset-base", resolved_target_layers=layers,
        resolved_target_modules=("down_proj",), teacher_identity=None)
    contract["lora"] = None
    snapshot = json.loads(json.dumps(_build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=HFArguments(), training_args=training, run_output_dir=training.output_dir)))
    meta = {"train_mode": cfg.train_mode, "lora_config": None, "norm_train_mode": "none", "lm_head_train_mode": "none",
        "round_base_checkpoint_id": "cpu-subset-base", "target_layers": list(layers),
        "target_modules": ["down_proj"], "immutable_resume_contract": contract}
    return meta, {"global_step": 2, "max_steps": 2000}, snapshot


@pytest.mark.parametrize("layers", [(0,), (0, 1, 2, 3)])
def test_subset_contract_preserves_original_schedule_and_exact_selection(layers):
    meta, state, snapshot = _configuration(layers)
    before = deepcopy((meta, state, snapshot))
    assert len(selected_names(layers, ["down_proj"])) == len(layers)
    assert validate_configuration(meta, state, snapshot) == meta["immutable_resume_contract"]
    assert (meta, state, snapshot) == before
    for bad_layers, bad_modules in (([1], ["down_proj"]), ([0, 2], ["down_proj"]), ([0], ["gate_proj"]),
                                    ([0], "all"), (list(range(36)), ["down_proj"])):
        with pytest.raises(ValueError):
            selected_names(bad_layers, bad_modules)
    for field, value in (("bit_active_ratio", 0.05), ("bit_round_steps", "128")):
        bad = deepcopy(snapshot)
        bad["canonical_config"][field] = value
        with pytest.raises(ValueError):
            validate_configuration(meta, state, bad)
    with pytest.raises(ValueError):
        validate_configuration(meta, {**state, "max_steps": 4}, snapshot)


def _save_reference_sidecar(model, selected, checkpoint, contract):
    config = SparseBitTuningConfig(enabled=True, **contract["sparse_bit"]).normalized()
    manager = SparseBitTuningManager(root_model=model, targets=selected,
        target_devices={name: torch.device("cpu") for name, _ in selected}, training_seed=0,
        config=config, streaming=False)
    manager.configure_schedule(total_optimizer_steps=2000)
    scales = {spec.canonical_key: 0.0125 + i * 0.001 for i, spec in enumerate(manager.bank_specs)}
    manager._set_coordinate_scales(scales)
    packed = {}
    for spec in manager.bank_specs:
        assert spec.n_active == spec.n_bits
        key = spec.canonical_key
        sampler = manager.sampler_states[key].advance()
        manager.sampler_states[key] = sampler
        storage = model.get_submodule(spec.module_path).get_stage_part_vq_storage(spec.stage_idx, spec.part_idx)
        bits = read_bank_hard_bits(storage, state=sampler, model_idx=0, logical_in_dim=spec.latent_dim)
        score = scores_from_hard_bits(bits, proxy_scale=scales[key])
        score[::7] *= -0.5
        score[1] = 0
        with torch.no_grad():
            manager.score_module.score_view(spec).copy_(score)
        packed[key], _ = apply_bank_scores_reference(storage, state=sampler, scores=score,
            model_idx=0, logical_in_dim=spec.latent_dim)
    state = {"format": "sparse_bit_tuning_exact_state", "version": manager.coordinate_state_version,
        **manager._coordinate_metadata(), "training_seed": 0, "streaming": False,
        "bit_active_ratio": 1.0, "target_modules": sorted(name for name, _ in selected),
        "banks": [{field: getattr(spec, field) for field in ("canonical_key", "module_path", "stage_idx", "part_idx",
            "n_bits", "n_active", "chunk_id", "score_start", "score_end")} for spec in manager.bank_specs],
        "packed_banks": packed, "score_chunks": [x.detach().clone() for x in manager.score_module.score_chunks],
        "sampler_states": {key: value.to_metadata() for key, value in manager.sampler_states.items()},
        "pending_next_states": {}, "bit_round_steps": 512, "stable_steps": 10,
        "global_bit_round": 1, "bit_round_step": 5, "stable_counter": 0,
        "cumulative_flip_count": 7, "had_flip": True, "bit_optimizer": manager.bit_optimizer.state_dict()}
    manager.detach_runtime()
    path = __import__("pathlib").Path(exact_state_path(str(checkpoint)))
    path.parent.mkdir(parents=True)
    torch.save({"format": EXACT_FORMAT, "version": EXACT_VERSION, "state": state}, path)
    return state, path


def test_actual_subset_restore_fp32_cast_full_bit_snapshot_and_frozen_modules(tmp_path):
    meta, _, snapshot = _configuration()
    model = _model()
    original = deepcopy(model.state_dict())
    base = tmp_path / "base"
    base_meta = v6.save_v6_full_checkpoint(model, str(base), checkpoint_kind="final_model",
        compressed_targets=TARGETS, base_model_path="cpu-subset-fixture")
    selected = ((TARGET, model.get_submodule(TARGET)),)
    selection = build_model_level_trainable_selection(model, aux=AuxTrainableConfig(), compressed_modules=selected,
        rank=8, alpha=16, dropout=0, train_decoder=True, train_lora=False, fp32_components=("decoder",),
        decoder_execution_mode="decoder_sparse_bit", freeze=True)
    args = SimpleNamespace(bf16=True, fp16=False)
    configure_distill_precision(selection, components=("decoder",), training_args=args)
    optimizer = torch.optim.AdamW(list(selection.decoder_parameters.values()), lr=1e-4, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float().square().mean()
        loss.backward()
        assert all(name.startswith(TARGET + ".") for name, p in model.named_parameters() if p.grad is not None)
        optimizer.step()
    mutable, classes, manifest = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    mutable = {key: value.detach().clone() for key, value in mutable.items()}
    assert len(mutable) == 6 and set(classes.values()) == {"decoder"}
    step = tmp_path / "step"
    v6.save_v6_training_step_payload(str(step), round_base_ref=str(base), round_base_checkpoint_id=base_meta["checkpoint_id"],
        mutable_state=mutable, mutable_state_manifest=manifest, compressed_targets=TARGETS,
        train_mode="decoder_sparse_bit", base_model_path="cpu-subset-fixture", immutable_resume_contract={"cpu_test": True})
    sidecar, sidecar_path = _save_reference_sidecar(model, selected, step, meta["immutable_resume_contract"])
    restored, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(base), strict=True)
    selected = ((TARGET, restored.get_submodule(TARGET)),)
    assert restore_decoder_state(restored, selected, checkpoint=step, snapshot=snapshot) == 6
    bit_report = restore_packed_bits_cpu(selected, checkpoint=step, contract=meta["immutable_resume_contract"])
    assert bit_report["packed_banks_compared"] == 2 and bit_report["global_bit_round"] == 1
    assert bit_report["bit_round_step"] == 5 and bit_report["active_score_signs_verified"] == sum(x["n_bits"] for x in sidecar["banks"])
    assert not bit_report["exact_training_state_restored"] and not bit_report["internal_bit_optimizer_restored"]
    for row in sidecar["banks"]:
        assert torch.equal(restored.get_submodule(TARGET).get_stage_part_vq_storage(row["stage_idx"], 0),
                           sidecar["packed_banks"][row["canonical_key"]])
    _assert_state_equal({key: restored.state_dict()[key] for key in mutable}, mutable, "Subset exact FP32 restore")
    assert _finalize_decoders(selected) == 1
    prepare_model_export(restored, args)
    allowed = set(mutable) | {TARGET + ".vq_weight", TARGET + ".vq_weight_s1"}
    frozen = {key: value for key, value in original.items() if key not in allowed}
    _assert_state_equal({key: restored.state_dict()[key] for key in frozen}, frozen, "All unselected projections/backbone frozen")
    assert all(torch.equal(restored.state_dict()[key], value.bfloat16()) for key, value in mutable.items())
    final = tmp_path / "final"
    restored.requires_grad_(False)
    v6.save_v6_full_checkpoint(restored, str(final), checkpoint_kind="final_model", compressed_targets=TARGETS,
        base_model_path="cpu-subset-fixture", train_mode="decoder_sparse_bit")
    fresh, _, _ = v6.load_v6_full_checkpoint_into_model(_model(), str(final), strict=True)
    _assert_state_equal(fresh.state_dict(), restored.state_dict(), "Fresh full model with subset endpoint")
    bad = deepcopy(sidecar)
    bad["global_bit_round"] = 2
    torch.save({"format": EXACT_FORMAT, "version": EXACT_VERSION, "state": bad}, sidecar_path)
    unchanged = deepcopy(fresh.state_dict())
    with pytest.raises(ValueError, match="sampler identity"):
        restore_packed_bits_cpu(((TARGET, fresh.get_submodule(TARGET)),), checkpoint=step, contract=meta["immutable_resume_contract"])
    _assert_state_equal(fresh.state_dict(), unchanged, "Rejected round mismatch is nonmutating")


def test_full_callback_protocol_and_private_limited_diagnostic():
    kwargs = dict(checkpoint="saved-step", source_dir="source", output_dir="output")
    commands = stage_commands(**kwargs)
    assert [x[0] for x in commands] == ["export", "audit", "mmlu", "seven", "summary"]
    assert commands[0][1][2] == "experiments.qwen3_w25.e2e_subset_export"
    assert commands[1][1][2] == "experiments.qwen3_w25.e2e_subset_audit"
    assert commands[1][1][-2:] == ["--training_checkpoint_dir", "saved-step"]
    for name, command, _ in commands:
        assert "--lm_limit" not in command
        if name in ("mmlu", "seven"):
            assert command[command.index("--lm_batch_size") + 1] == ("64" if name == "mmlu" else "32")
    limited = stage_commands(**kwargs, diagnostic_eval_limit=64)
    assert len(limited) == 4
