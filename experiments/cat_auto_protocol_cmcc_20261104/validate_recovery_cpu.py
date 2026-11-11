"""CPU state/choice mechanisms only; no model-scale KD, quality or GPU claim."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch
from torch import nn
from transformers import BertTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerControl, TrainerState, TrainingArguments

from e2e_common.full_lora import collect_exact_peft_lora_config
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
from train_utils.cat_step_resume_v6 import build_cat_step_immutable_resume_contract
from train_utils.cat_train_runtime import load_cat_resume_distill_progress
from train_utils.config.configs import (AfterCategoryResolvedConfig, AuxTrainableConfig, DistillDataConfig,
    DistillLossConfig, DistillOptimizationConfig, DistillRuntimeConfig, LoRAConfig)
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract

from .contracts import digest
from .live import export_selected_node, source_pins
from .mutable import capture_mutable, restore_mutable, select_native_scope
from .nodes import FullEightScore, NodeScore, RecoveryQuota, RecoverySettings, TrajectoryResult, best_trajectory, load_complete_eight_score
from .recovery import NativeNodeTrainer, recovery_seeds, retained_node_directories
from .replay import SelectedCategory, StopAtSelectedNode, StrictMainChain


CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError, FileNotFoundError, NotImplementedError):
        return
    raise AssertionError("A violated state/choice contract was accepted.")


def score(value, source):
    return FullEightScore({f"task{index}": value for index in range(8)}, "explicit_CPU_rule_fixture", dict(
        checkpoint_id="fixture_deployment_" + source, source_checkpoint_id=source,
        protocol_id="CPU_rule_fixture_only", evaluation_digest=digest(dict(source=source, value=value)),
        leaf_count=8, sample_count=8, expected_leaf_count=8, expected_sample_count=8, phase="cpu_fixture"))


def trajectory(category="a", *, raw="raw_a", anchor=True, values=(.6, .6, .6, .6),
               raw_value=.5, settings=None, candidate="planned_a", identity=None):
    settings = settings or RecoverySettings()
    identity = identity or f"{raw}_{settings.scope}_{settings.decoder_lr}_{settings.schedule_max}"
    return TrajectoryResult(category, "live_parent", raw, "recipe_" + raw, candidate, identity,
        anchor, 100, 120, settings, score(raw_value, raw), tuple(NodeScore(step, score(value, f"{identity}_{step}"),
        f"{identity}_{step}", .3 if step == 375 else .2 if step == 500 else None)
        for step, value in zip(settings.node_steps, values, strict=True)), "same_fixed_train_readout")


def choice_checks():
    anchor = trajectory()
    contender = trajectory(raw="raw_b", anchor=False, candidate="planned_b")
    assert best_trajectory([contender, anchor]) == (anchor, anchor.nodes[-1])
    rejects(FullEightScore, {"only_one_task": .5}, "fixture", {})
    rejects(score, float("nan"), "node")
    invalid = replace(contender.raw_score, evidence={**contender.raw_score.evidence, "protocol_id": "different"})
    rejects(best_trajectory, [anchor, replace(contender, raw_score=invalid,
        nodes=tuple(replace(node, score=replace(node.score, evidence={**node.score.evidence, "protocol_id": "different"})) for node in contender.nodes))])
    peak = trajectory(values=(.7, .68, .66, .65))
    quota = RecoveryQuota(["a"])
    for row in (peak, contender):
        quota.register_start(category_id="a", raw_id=row.raw_id, settings=row.settings, kind="baseline")
    arm = quota.decide([peak, contender], prefix_targets=["old", "current"], current_targets=["current"], future_dense_targets=["future"])
    assert arm.settings.decoder_lr == .5e-5 and arm.interaction_eligible and quota.interaction_raw(arm, [peak, contender]) == contender.raw_id
    rejects(quota.select_completed_category, [peak, contender])
    extra = [trajectory(raw=row.raw_id, anchor=row.is_anchor, candidate=row.candidate_id,
        settings=arm.settings, values=(.61, .62, .63, .64)) for row in (peak, contender)]
    for row, kind in zip(extra, ("third", "interaction"), strict=True):
        quota.register_start(category_id="a", raw_id=row.raw_id, settings=row.settings, kind=kind)
    chosen, node, certificate = quota.select_completed_category([peak, contender, *extra])
    assert chosen == peak and node.step == 125 and certificate["all_announced_cells_complete"]
    assert quota.counts()["actual_started"] == 4 and quota.counts()["actual_node_evaluations"] == 16
    rejects(quota.register_start, category_id="a", raw_id="third_raw", settings=arm.settings, kind="interaction")
    long = RecoveryQuota(["a", "b", "c"])
    lengths = []
    for category in long.categories:
        row = trajectory(category=category, raw=category, values=(.51, .52, .53, .54))
        result = long.decide([row], prefix_targets=["current"], current_targets=["current"], future_dense_targets=[])
        lengths.append(result.settings.schedule_max)
        extra = trajectory(category=category, raw=category, settings=result.settings,
            values=(.5,) * len(result.settings.node_steps))
        long.select_completed_category([row, extra])
    assert lengths == [1000, 1000, 500] and long.long_categories == ["a", "b"]
    scope = RecoveryQuota(["a"]).decide([trajectory(raw_value=.8)],
        prefix_targets=["old", "current"], current_targets=["current"], future_dense_targets=["future"])
    assert scope.settings.scope == "remaining_lora_current_decoder"
    assert RecoveryQuota(["a"]).decide([trajectory(raw_value=.8)], prefix_targets=["current"], current_targets=["current"], future_dense_targets=[]) is None
    callback = StopAtSelectedNode(schedule_max=500, selected_step=250)
    args, state, control = SimpleNamespace(max_steps=500), TrainerState(), TrainerControl()
    callback.on_train_begin(args, state, control)
    state.global_step = 250
    callback.on_step_end(args, state, control)
    assert control.should_training_stop and control.should_save and args.max_steps == 500
    rejects(callback.on_train_begin, SimpleNamespace(max_steps=250), TrainerState(), TrainerControl())
    return dict(rule_fixture_only=True, tie_later_then_anchor=True, incomplete_or_foreign_protocol_rejected=True,
        two_long_schedules_max=True, interaction_preregistered_and_required=True, quota=quota.counts(), replay_original_T_preserved=True)


def tiny_native_model():
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=32, attention_dropout=.1, tie_word_embeddings=False,
        bos_token_id=2, eos_token_id=3, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    prefix, current, future = [], [], []
    for index in range(2):
        for category in CATEGORIES:
            name = f"model.layers.{index}.{'self_attn' if category in CATEGORIES[:4] else 'mlp'}.{category}"
            old = model.get_submodule(name)
            if category in ("q_proj", "k_proj"):
                bits = torch.randint(0, 2, (old.out_features, 1, 9), dtype=torch.bool)
                decoder = Decoder(in_dim=9, out_dim=16, hidden_dim=8, num_res_blocks=0,
                    norm_type="layer", decoder_type="linear", use_checkpoint=False, num_models=1).to(dtype=torch.bfloat16).extract_single(0)
                layer = VAELinear(in_features=16, out_features=old.out_features, bias=None,
                    original_weight=None, vq_weight=bits, decoder=decoder, codebook_dim=16, transpose=False)
                layer.pack_parallel_stage_decoder_(trainable=False)
                parent_name, field = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), field, layer)
                prefix.append(name)
                if category == "k_proj":
                    current.append(name)
            else:
                future.append(name)
    return model.eval().requires_grad_(False), tuple(prefix), tuple(current), tuple(future)


def real_native_state_checks(root: Path):
    torch.set_num_threads(2)
    torch.manual_seed(731)
    model, prefix, current, future = tiny_native_model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None,
        runtime_identity=dict(phase="cpu_fixture", cat_shared=dict(channel_refresh_after_category=False)))
    raw_dir = root / "round_base"
    raw = v6.save_v6_full_checkpoint(model, str(raw_dir), checkpoint_kind="round_base",
        compressed_targets=prefix, pending_dense_targets=future, completed_categories=["q_proj"],
        compression_categories=CATEGORIES, target_layers=[0, 1], target_modules=CATEGORIES,
        cat_runtime_state=runtime, extra_meta=dict(cat_auto_phase="cpu_fixture",
            distill_stage_history=[dict(category="q_proj", did_train=False)]))
    raw_meta = v6.load_v6_meta(str(raw_dir))
    settings, seeds = RecoverySettings(), recovery_seeds(dict(fixture_model="tiny_Qwen3"), "k_proj", 31)
    precision = SimpleNamespace(bf16=True, fp16=False)
    selected = select_native_scope(model, settings=settings, prefix_targets=prefix,
        current_targets=current, future_dense_targets=future, precision=precision, lora_init_seed=seeds["lora_init"])
    assert len(selected.decoder_targets) == 4 and len(selected.dense_targets) == 10
    initial, components, manifest = capture_mutable(selected)
    assert set(components.values()) == {"decoder", "lora"}
    with torch.no_grad():
        for parameter in selected.selection.decoder_parameters.values():
            parameter.add_(.01)
        for parameter in selected.selection.lora_parameters.values():
            parameter.add_(.02)
    restore_mutable(selected, initial, manifest)
    missing = dict(initial)
    missing.pop(next(iter(missing)))
    rejects(restore_mutable, selected, missing, manifest)
    vocab = root / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntiny\ntrain\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    rows = [dict(input_ids=torch.tensor([2, 5, 6, 7, 8, 3]), attention_mask=torch.ones(6, dtype=torch.long),
        labels=torch.tensor([2, 5, 6, 7, 8, 3])) for _ in range(4)]
    args = TrainingArguments(output_dir=str(root / "trainer"), use_cpu=True, bf16=True,
        max_steps=500, per_device_train_batch_size=2, learning_rate=1e-4,
        weight_decay=0., lr_scheduler_type="cosine", warmup_ratio=.01,
        save_safetensors=False, report_to=[], disable_tqdm=True, logging_strategy="no",
        dataloader_num_workers=0, remove_unused_columns=False)
    class TensorRows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            return rows[index]
    trainer = NativeNodeTrainer(model=selected.model, args=args, train_dataset=TensorRows(),
        processing_class=tokenizer, data_collator=lambda batch: {key: torch.stack([row[key] for row in batch]) for key in batch[0]},
        loss_type="sft", teacher_output_offload="none", teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=8, recovery_settings=settings, semantic_seeds=seeds, technical_stop_after=2)
    attach_model_level_optimizer_contract(trainer, selection=selected.selection,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-4, decoder_lr=1e-5, weight_decay=0.))
    lora = collect_exact_peft_lora_config(selected.model, default_rank=12, alpha=24, dropout=.1)
    cfg = AfterCategoryResolvedConfig(data=DistillDataConfig(), loss=DistillLossConfig(loss_type="sft"),
        opt=DistillOptimizationConfig(steps=500, learning_rate=1e-4, decoder_lr=1e-5,
            distill_fp32_components=("decoder", "lora")), lora=LoRAConfig(rank=12, alpha=24, dropout=.1),
        aux=AuxTrainableConfig(), runtime=DistillRuntimeConfig())
    contract = build_cat_step_immutable_resume_contract(stage=SimpleNamespace(config=cfg, mode=settings.scope),
        trainer_args=args, tokenizer=tokenizer, round_base_checkpoint_id=raw["checkpoint_id"], active_category="k_proj",
        round_base_meta=raw_meta, lora_target_names=future, decoder_target_names=prefix,
        teacher_identity=None, dataset_identity=dict(phase="cpu_fixture", purpose="real_native_CPU_save_return_mechanism_only"), lora_config=lora)
    context = dict(round_base_dir=str(raw_dir), round_base_checkpoint_id=raw["checkpoint_id"],
        active_category="k_proj", after_category_mode=settings.scope, compressed_targets=prefix,
        pending_dense_targets=future, skip_targets=(), completed_categories=["q_proj"],
        compression_categories=CATEGORIES, target_layers=[0, 1], target_modules=CATEGORIES,
        lora_config=lora, resolved_learning_rates=dict(learning_rate=1e-4, decoder_lr=1e-5),
        immutable_resume_contract=contract, base_model_path="explicit_cpu_fixture_only")
    trainer.configure_v6_step_checkpoint(context=context, selected_vae_modules=selected.decoder_targets)
    trainer.train()
    assert trainer.state.global_step == 2 and trainer.args.max_steps == 500 and len(trainer.node_save_audits) == 2
    first, second = root / "trainer/checkpoint-1", root / "trainer/checkpoint-2"
    first_values, _ = v6.load_v6_training_model_state(str(first), map_location="cpu")
    second_values, _ = v6.load_v6_training_model_state(str(second), map_location="cpu")
    assert any(not torch.equal(first_values[name], second_values[name]) for name in first_values)
    assert all(any(not torch.equal(first_values[name], second_values[name]) for name in first_values
        if name.startswith("base_model.model." + owner + ".")) for owner in prefix)
    assert any(not torch.equal(first_values[name], second_values[name]) for name in first_values if components[name] == "lora")
    node = v6.load_v6_training_step_meta(str(second))
    recipe = SelectedCategory("k_proj", "fixture_recipe", "fixture_planned_candidate", settings, 2,
        "fixture_trajectory", node["checkpoint_id"], raw["checkpoint_id"], seeds, phase="cpu_fixture")
    report = export_selected_node(node_directory=second, destination=root / "boundary", selected=recipe,
        current_targets=current, phase="cpu_fixture")
    assert not report["formal_eligible"] and report["fresh_evidence"]["state_exact"]
    progress = load_cat_resume_distill_progress(str(root / "boundary"))
    assert list(progress.completed_categories) == ["q_proj", "k_proj"] and progress.lora_round_idx == 1
    rejects(retained_node_directories, root / "trainer", settings)
    rejects(StrictMainChain(bf16_root_id="fixture", categories=["k_proj"]).commit, recipe,
        live_parent_id="fixture", boundary_reference={}, selection_certificate={},
        actual_round_base_id=raw["checkpoint_id"], restored_node_id=node["checkpoint_id"], runtime_identity=runtime["runtime_identity"])
    current_model, _, _ = v6.load_v6_full_checkpoint_into_model(tiny_native_model()[0], str(raw_dir), strict=True)
    narrower = select_native_scope(current_model, settings=replace(settings, scope="remaining_lora_current_decoder"),
        prefix_targets=prefix, current_targets=current, future_dense_targets=future, precision=precision, lora_init_seed=seeds["lora_init"])
    assert len(narrower.decoder_targets) == 2 and len(narrower.dense_targets) == 10
    frozen_q = {name: value for name, value in narrower.model.state_dict().items() if ".q_proj." in name}
    assert frozen_q and all(not parameter.requires_grad for name, parameter in narrower.model.named_parameters() if ".q_proj." in name)
    narrower_state, _, _ = capture_mutable(narrower)
    assert not any(".q_proj." in name for name in narrower_state)
    rejects(select_native_scope, tiny_native_model()[0], settings=replace(settings, scope="current_decoder"),
        prefix_targets=prefix, current_targets=current, future_dense_targets=future, precision=precision, lora_init_seed=seeds["lora_init"])
    return dict(phase="cpu_fixture", actual_architecture="Qwen3 2 layers hidden16; CPU SFT mechanism fixture, not protocol KD",
        all_prefix_decoders=4, future_dense_lora_owners=10, actual_mutable_tensors=len(initial),
        components=sorted(set(components.values())), native_save1_return_to_step2=True,
        original_schedule_max=500, actual_steps=2, save_audits=trainer.node_save_audits,
        complete_round_base_overlay=True, independent_fresh_state_exact=True,
        frozen_export_fields=report["export_evidence"]["frozen_fields_exact"],
        narrowed_decoder_owners=2, narrowed_future_lora_owners=10, missing_mutable_and_wrong_scope_rejected=True,
        original_raw_tensor_count=len(original), boundary_progress_exact=True, formal_eligible=False,
        forward_scope="CPU training mechanism only; no real-model GPU/native-KD/eight-task/own-live-successor proof")


def existing_loader_checks(root: Path):
    """Historical A250 is evidence only; its deleted weights are never opened."""
    plan_path = root / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    endpoint = plan["endpoints"]["250"]
    summary = root / "evaluation/A250/full8/full8_summary.json"
    summary_value = json.loads(summary.read_text(encoding="utf-8"))
    expected_metrics = dict(boolq="acc,none", rte="acc,none", winogrande="acc,none", arc_easy="acc_norm,none",
        arc_challenge="acc_norm,none", openbookqa="acc_norm,none", piqa="acc_norm,none", mmlu="acc,none")
    with plan_path.open("rb") as stream:
        plan_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(protocol_id="historical_A250_A500_original_full8_" + plan_sha,
        leaf_count=plan["leaf_tasks"], sample_count=plan["sample_count"], task_metric_keys=expected_metrics,
        plan_path=str(plan_path), plan_sha256=plan_sha, checkpoint_meta_sha256=endpoint["checkpoint_meta_sha256"],
        expected_stages=endpoint["stages"])
    paths = {row["path"]: str(summary.parent / Path(row["path"]).name) for row in summary_value["input_results"]}
    kwargs = dict(summary_path=summary, evaluation_path=summary.parent / "evaluation.json",
        checkpoint_dir=root / "A250/model", source_checkpoint_id=endpoint["checkpoint_id"], protocol_manifest=manifest)
    result = load_complete_eight_score(**kwargs, artifact_path_map=paths)
    assert abs(float(result.mean * 100) - summary_value["mean_percent"]) <= 1e-12
    wrong_paths = dict(paths)
    source = next(iter(wrong_paths))
    wrong_paths[source] = str(summary.parent / "seven.json") if Path(source).name == "mmlu.json" else str(summary.parent / "mmlu.json")
    rejects(load_complete_eight_score, **kwargs, artifact_path_map=wrong_paths)
    if not all(Path(path).is_file() for path in paths):
        rejects(load_complete_eight_score, **kwargs)
    rejects(load_complete_eight_score, **{**kwargs, "protocol_manifest": {**manifest, "sample_count": 1}}, artifact_path_map=paths)
    return dict(status="actual_existing_complete_full8_loader_verified", source="historical A250 deployment endpoint",
        checkpoint_id=endpoint["checkpoint_id"], mean_percent=float(result.mean * 100),
        leaf_count=result.evidence["leaf_count"], sample_count=result.evidence["sample_count"],
        evaluation_digest=result.evidence["evaluation_digest"], explicit_artifact_resolutions=result.evidence["explicit_artifact_resolutions"],
        wrong_shard_sha_missing_map_partial_count_rejected=True, model_weights_read=False,
        boundary="Historical complete-score compatibility only; A250 weights were deleted and this is no fresh/new-boundary proof.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("all", "rules-and-loader"), default="all")
    parser.add_argument("--existing-full8-root")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Run this CPU-only validator with CUDA_VISIBLE_DEVICES='' before any CUDA initialization.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    choice = choice_checks()
    if args.scope == "all":
        with tempfile.TemporaryDirectory(prefix="recovery_state_fixture_", dir=output.parent) as directory:
            native = real_native_state_checks(Path(directory))
    else:
        native = dict(status="not_repeated", reason="No save/restore computation changed; reuse the separately recorded native CPU state evidence.")
    loader = existing_loader_checks(Path(args.existing_full8_root)) if args.existing_full8_root else dict(status="not_requested")
    if args.scope == "rules-and-loader" and not args.existing_full8_root:
        raise ValueError("The read-only rules-and-loader scope requires the explicit existing full8 artifact root.")
    report = dict(status="cpu_mechanisms_verified", choice=choice, native=native, existing_loader=loader,
        source_pins=source_pins(), CUDA_initialized=torch.cuda.is_initialized(),
        elapsed_seconds=time.monotonic() - started, real_model_gpu_pilot="not_authorized_not_run")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], output=str(output), elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
